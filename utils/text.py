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

import torch


def load_text_encoder(device: torch.device):
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "cache" / "huggingface"))
    os.environ.setdefault("USE_TF", "0")
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError("Install transformers to encode text for bert-mode checkpoints") from error
    return AutoModel.from_pretrained("google-bert/bert-base-uncased", local_files_only=True).to(device).eval().requires_grad_(False)


@torch.no_grad()
def encode_text(batch: dict[str, torch.Tensor], encoder, text_mask: torch.Tensor | None = None) -> torch.Tensor:
    tokens = batch["tokens"]
    observed = batch["text_mask"] if text_mask is None else text_mask
    special = (tokens[:, 0] == 101) | (tokens[:, 0] == 102)
    attention = tokens[:, 1].bool() & (observed | special)
    return encoder(input_ids=tokens[:, 0], attention_mask=attention.long(),
                   token_type_ids=tokens[:, 2]).last_hidden_state
