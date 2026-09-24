'''
Author: Orange
Date: 2026-09-23 15:48
LastEditors: Orange
LastEditTime: 2026-09-23 22:02
FilePath: text.py
Description: 

'''
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


DEFAULT_BERT = "google-bert/bert-base-uncased"
DEFAULT_BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
ALIGNED_TEXT_STEPS = 50


def _set_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "cache" / "huggingface"))
    os.environ.setdefault("USE_TF", "0")


def load_text_encoder(device: torch.device, model_name: str = DEFAULT_BERT,
                      revision: str = DEFAULT_BERT_REVISION):
    _set_hf_cache()
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError("Install transformers to encode text for bert-mode checkpoints") from error
    return AutoModel.from_pretrained(model_name, revision=revision, local_files_only=True).to(device).eval().requires_grad_(False)


def load_text_tokenizer(model_name: str = DEFAULT_BERT,
                        revision: str = DEFAULT_BERT_REVISION):
    """从本地缓存载入 BERT tokenizer。"""
    _set_hf_cache()
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("BERT 微调需要安装 transformers") from error
    return AutoTokenizer.from_pretrained(model_name, revision=revision,
                                          local_files_only=True, use_fast=True)


def _head_tail(ids: list[int], token_types: list[int], tokenizer, max_length: int):
    if len(ids) <= max_length:
        return ids, token_types, False
    body_budget = max_length - 2
    head = (body_budget + 1) // 2
    tail = body_budget - head
    body, body_types = ids[1:-1], token_types[1:-1]
    kept_ids = [ids[0], *body[:head], *body[-tail:], ids[-1]] if tail else [ids[0], *body[:head], ids[-1]]
    kept_types = [token_types[0], *body_types[:head], *body_types[-tail:], token_types[-1]] if tail else [token_types[0], *body_types[:head], token_types[-1]]
    return kept_ids, kept_types, True


def prepare_full_text_inputs(data: dict, tokenizer, max_length: int = 512) -> dict[str, np.ndarray | dict[str, int]]:
    """用完整转写生成 BERT 输入，并检查与已对齐前缀相同。"""
    if max_length < ALIGNED_TEXT_STEPS:
        raise ValueError(f"BERT max_length must be at least {ALIGNED_TEXT_STEPS}")
    texts = data.get("raw_text")
    tokens = data.get("tokens")
    if texts is None or tokens is None or len(texts) != len(tokens):
        raise ValueError("Full-text BERT mode requires raw_text and aligned text_bert tokens")

    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer must define CLS and SEP token ids")

    encoded = tokenizer(list(texts), add_special_tokens=True, truncation=False,
                        padding=False, return_token_type_ids=True)
    rows, type_rows = [], []
    long_count, truncated_count, max_observed = 0, 0, 0
    aligned = np.asarray(tokens)
    for index, text in enumerate(texts):
        ids = list(encoded["input_ids"][index])
        token_types = list(encoded.get("token_type_ids", [[0] * len(row) for row in encoded["input_ids"]])[index])
        max_observed = max(max_observed, len(ids))
        active = aligned[index, 1].astype(bool)
        stored = aligned[index, 0, active].tolist()
        if len(ids) <= ALIGNED_TEXT_STEPS:
            matches = stored == ids
        else:
            matches = (len(stored) == ALIGNED_TEXT_STEPS
                       and stored[:ALIGNED_TEXT_STEPS - 1] == ids[:ALIGNED_TEXT_STEPS - 1]
                       and stored[-1] == sep_id)
        if not matches:
            sample_id = data.get("ids", [index] * len(texts))[index]
            raise ValueError(f"raw_text BERT prefix does not match text_bert for sample {sample_id}")
        long_count += int(len(ids) > ALIGNED_TEXT_STEPS)
        ids, token_types, was_truncated = _head_tail(ids, token_types, tokenizer, max_length)
        truncated_count += int(was_truncated)
        rows.append(ids)
        type_rows.append(token_types)

    width = max(map(len, rows), default=1)
    input_ids = np.full((len(rows), width), int(tokenizer.pad_token_id or 0), dtype=np.int64)
    attention = np.zeros((len(rows), width), dtype=np.int64)
    token_types = np.zeros((len(rows), width), dtype=np.int64)
    for index, (ids, types) in enumerate(zip(rows, type_rows)):
        input_ids[index, :len(ids)] = ids
        attention[index, :len(ids)] = 1
        token_types[index, :len(types)] = types
    return {
        "bert_input_ids": input_ids,
        "bert_attention_mask": attention,
        "bert_token_type_ids": token_types,
        "bert_text_stats": {
            "n_samples": len(rows),
            "n_over_50_tokens": long_count,
            "max_original_tokens": max_observed,
            "n_head_tail_truncated": truncated_count,
            "max_length": max_length,
        },
    }


def prepare_bert_inputs(data: dict, tokenizer=None, max_length: int = 512,
                        source: str = "raw_text") -> dict:
    """选择原始50位输入或完整转写，训练和推理使用同一来源。"""
    if source == "raw_text":
        return prepare_full_text_inputs(data, tokenizer, max_length)
    if source != "text_bert":
        raise ValueError(f"Unknown BERT input source: {source}")
    tokens = np.asarray(data["tokens"])
    return {
        "bert_input_ids": tokens[:, 0].astype(np.int64, copy=True),
        "bert_attention_mask": tokens[:, 1].astype(np.int64, copy=True),
        "bert_token_type_ids": tokens[:, 2].astype(np.int64, copy=True),
        "bert_text_stats": {"source": source, "n_samples": len(tokens), "max_length": 50},
    }


def mask_full_text_attention(attention: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
    """把 50 词位文本掩码映射到完整 BERT 序列。"""
    masked = attention.clone().bool()
    for row in range(attention.shape[0]):
        active = torch.where(attention[row].bool())[0]
        if active.numel() <= 2:
            continue
        end = int(active[-1].item()) + 1
        # 前48个词位逐一对应；局部缺失时一并遮住无对齐索引的长尾。
        prefix_end = min(ALIGNED_TEXT_STEPS - 1, end - 1)
        visible = text_mask[row, 1:prefix_end].bool()
        masked[row, 1:prefix_end] &= visible
        if not visible.all():
            masked[row, prefix_end:end - 1] = False
    return masked.to(attention.dtype)


@torch.no_grad()
def encode_text(batch: dict[str, torch.Tensor], encoder, text_mask: torch.Tensor | None = None) -> torch.Tensor:
    tokens = batch["tokens"]
    observed = batch["text_mask"] if text_mask is None else text_mask
    special = (tokens[:, 0] == 101) | (tokens[:, 0] == 102)
    attention = tokens[:, 1].bool() & (observed | special)
    return encoder(input_ids=tokens[:, 0], attention_mask=attention.long(),
                   token_type_ids=tokens[:, 2]).last_hidden_state
