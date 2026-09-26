"""冻结问题二预测器，再学习问题三的时序交互残差。"""

from __future__ import annotations

import torch
from torch import nn

from .fuse_net import FactorizedAffectiveModel
from .transformer_encoder import TemporalTransformerEncoder


class Q3ResidualModel(nn.Module):
    """以已验证的单模型预测为起点，仅训练轻量上下文交互分支。"""

    def __init__(self, base_config: dict, dropout: float = 0.2, heads: int = 4,
                 embedding_dim: int = 96, hidden_dim: int = 64):
        super().__init__()
        if hidden_dim != 64:
            raise ValueError("Q3 residual width must match the source FUSE encoder")
        if base_config.get("architecture") != "fuse":
            raise ValueError("Q3 residual backbone requires a FUSE checkpoint")
        config = {key: value for key, value in base_config.items() if key != "architecture"}
        self.base = FactorizedAffectiveModel(**config)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.text_mode = self.base.text_mode
        self.regression_mode = self.base.regression_mode
        self.bert_finetune = self.base.bert_finetune
        width = 128
        self.audio_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.vision_attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.temporal = TemporalTransformerEncoder(width, width // 2, heads, 1, dropout)
        self.pool_score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)
        self.class_delta = nn.Linear(width, 3)
        self.magnitude_delta = nn.Linear(width, 1)
        nn.init.zeros_(self.class_delta.weight)
        nn.init.zeros_(self.class_delta.bias)
        nn.init.zeros_(self.magnitude_delta.weight)
        nn.init.zeros_(self.magnitude_delta.bias)

    def fit_input_stats(self, train: dict) -> None:
        self.base.fit_input_stats(train)

    @staticmethod
    def _attend(query: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                attention: nn.MultiheadAttention) -> torch.Tensor:
        safe = mask.bool().clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        result = attention(query, values, values, key_padding_mask=~safe,
                           need_weights=False)[0]
        return result * (~empty)[:, None, None].to(result.dtype)

    def forward(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                text_features=None, bert_input_ids=None, bert_attention_mask=None,
                bert_token_type_ids=None):
        self.base.eval()
        with torch.no_grad():
            original = self.base(tokens, lengths, audio, vision, text_mask, audio_mask,
                                 vision_mask, text_features, bert_input_ids,
                                 bert_attention_mask, bert_token_type_ids)
        encoded = original["encoded"]
        text = encoded["text"]
        audio_context = self._attend(text, encoded["audio"], audio_mask, self.audio_attention)
        vision_context = self._attend(text, encoded["vision"], vision_mask, self.vision_attention)
        fused = self.temporal(text + audio_context + vision_context, text_mask)
        weights = torch.softmax(self.pool_score(fused).squeeze(-1).masked_fill(~text_mask.bool(), -1e4),
                                dim=1) * text_mask.to(fused.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = self.dropout((fused * weights.unsqueeze(-1)).sum(dim=1))
        logits = original["logits"] + self.class_delta(pooled)
        magnitude = (original["magnitude"] + self.magnitude_delta(pooled).squeeze(-1)).clamp(0, 3)
        probabilities = logits.softmax(dim=-1)
        sentiment = magnitude * (probabilities[:, 2] - probabilities[:, 0])
        return {"logits": logits, "magnitude": magnitude, "sentiment": sentiment,
                "time_weights": weights, "modality_weights": original["modality_weights"],
                "pooled": pooled}
