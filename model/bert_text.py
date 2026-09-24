from __future__ import annotations

import contextlib
import os
from pathlib import Path

import torch
from torch import nn

from utils.data import SEQ_LEN, TEXT_DIM


def align_bert_outputs(hidden: torch.Tensor, attention_mask: torch.Tensor,
                       aligned_steps: int = SEQ_LEN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """保留已对齐前缀，并将超出词位汇成一个文本尾部向量。"""
    rows, tails, tail_flags = [], [], []
    for index in range(hidden.shape[0]):
        active_positions = torch.where(attention_mask[index].bool())[0]
        if active_positions.numel() < 2:
            raise ValueError("BERT sequence must include CLS and SEP")
        end = int(active_positions[-1].item()) + 1
        if int(active_positions[0].item()) != 0:
            raise ValueError("BERT CLS position must remain visible")
        length = end
        if length <= aligned_steps:
            row = hidden[index, :length]
            pad = hidden.new_zeros((aligned_steps - length, hidden.shape[-1]))
            rows.append(torch.cat((row, pad), dim=0))
            tails.append(hidden[index, 0] * 0.0)
            tail_flags.append(False)
        else:
            prefix = hidden[index, :aligned_steps - 1]
            separator = hidden[index, length - 1:length]
            rows.append(torch.cat((prefix, separator), dim=0))
            tail = hidden[index, aligned_steps - 1:length - 1]
            tail_mask = attention_mask[index, aligned_steps - 1:length - 1].bool()
            if tail_mask.any():
                tails.append(tail[tail_mask].mean(dim=0))
                tail_flags.append(True)
            else:
                tails.append(hidden[index, 0] * 0.0)
                tail_flags.append(False)
    return (torch.stack(rows), torch.stack(tails),
            torch.tensor(tail_flags, device=hidden.device, dtype=torch.bool))


class FineTunedBertTextEncoder(nn.Module):
    """冻结底层、微调 BERT 顶层，并为长转写单独汇总尾部。"""

    def __init__(self, model_name: str, revision: str, freeze_bottom_layers: int = 8,
                 gradient_checkpointing: bool = True):
        super().__init__()
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "cache" / "huggingface"))
        os.environ.setdefault("USE_TF", "0")
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("BERT 微调需要安装 transformers") from error
        try:
            self.backbone = AutoModel.from_pretrained(model_name, revision=revision,
                                                      local_files_only=True)
        except OSError as error:
            raise RuntimeError(f"本地找不到预训练 BERT 权重：{model_name}") from error
        if self.backbone.config.hidden_size != TEXT_DIM:
            raise ValueError(f"Expected a {TEXT_DIM}-dimensional BERT, got {self.backbone.config.hidden_size}")

        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise ValueError("当前模型不支持 BERT encoder.layer 冻结策略")
        if not 0 <= freeze_bottom_layers < len(layers):
            raise ValueError(f"freeze_bottom_layers must be between 0 and {len(layers) - 1}")
        self.adaptation_enabled = True
        self.freeze_bottom_layers = int(freeze_bottom_layers)
        self.frozen_layers = list(layers[:self.freeze_bottom_layers])
        for parameter in self.backbone.embeddings.parameters():
            parameter.requires_grad_(False)
        for layer in self.frozen_layers:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        pooler = getattr(self.backbone, "pooler", None)
        if pooler is not None:
            for parameter in pooler.parameters():
                parameter.requires_grad_(False)

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def set_adaptation_enabled(self, enabled: bool):
        """预热期间关闭BERT梯度与dropout，保留可训练参数标识。"""
        self.adaptation_enabled = enabled
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and not self.adaptation_enabled:
            self.backbone.eval()
        if mode:
            self.backbone.embeddings.eval()
            for layer in self.frozen_layers:
                layer.eval()
            pooler = getattr(self.backbone, "pooler", None)
            if pooler is not None:
                pooler.eval()
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        with (contextlib.nullcontext() if self.adaptation_enabled else torch.no_grad()):
            hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        aligned, tail, tail_available = align_bert_outputs(hidden, attention_mask)
        return aligned, tail, tail_available
