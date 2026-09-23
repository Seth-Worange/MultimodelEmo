"""Contextual BERT features with explicit subword-to-original-word mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_loader import OriginalWord
from .io_utils import write_json


@dataclass
class TokenMapping:
    window_index: int
    token_position: int
    token_id: int
    token: str
    word_index: int
    subword_index: int


@dataclass
class TextFeatureResult:
    features: np.ndarray
    mask: np.ndarray
    token_mappings: list[TokenMapping]
    feature_dim: int
    valid_length: int
    model_name: str
    model_revision: str | None
    pooling: str
    windows: list[dict[str, Any]]

    def metadata(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "valid_length": self.valid_length,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "pooling": self.pooling,
            "windows": self.windows,
            "token_mappings": [asdict(item) for item in self.token_mappings],
        }


def _make_windows(
    token_ids_by_word: list[list[int]], capacity: int, overlap_words: int,
) -> list[list[tuple[int, list[int], int]]]:
    """Return windows of (word index, token ids, subword offset), never truncating."""
    windows: list[list[tuple[int, list[int], int]]] = []
    start = 0
    while start < len(token_ids_by_word):
        ids = token_ids_by_word[start]
        if len(ids) > capacity:
            for offset in range(0, len(ids), capacity):
                windows.append([(start, ids[offset:offset + capacity], offset)])
            start += 1
            continue
        current: list[tuple[int, list[int], int]] = []
        used = 0
        end = start
        while end < len(token_ids_by_word):
            word_ids = token_ids_by_word[end]
            if len(word_ids) > capacity or (current and used + len(word_ids) > capacity):
                break
            current.append((end, word_ids, 0))
            used += len(word_ids)
            end += 1
            if used == capacity:
                break
        if not current:  # Defensive: the oversized case above should catch this.
            start += 1
            continue
        windows.append(current)
        if end >= len(token_ids_by_word):
            break
        start = max(start + 1, end - max(0, overlap_words))
    return windows


def extract_text_features(
    original_words: list[OriginalWord],
    *,
    model_name: str = "google-bert/bert-base-uncased",
    pooling: str = "mean",
    device: str = "cpu",
    overlap_words: int = 32,
    output_dir: Path | None = None,
) -> TextFeatureResult:
    """Extract a representation for every original whitespace-delimited word."""
    if pooling != "mean":
        raise ValueError("Only auditable mean subword pooling is supported")
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Could not import torch/transformers; use the clean environment from q1_v2/README.md"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for auditable token mapping")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    words = [word.text for word in original_words]
    token_ids_by_word: list[list[int]] = []
    for word in words:
        encoded = tokenizer(word, add_special_tokens=False)
        token_ids_by_word.append([int(value) for value in encoded["input_ids"]])

    model_max = int(getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
    special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
    capacity = model_max - special_count
    if capacity <= 0:
        raise RuntimeError(f"Invalid model capacity: max={model_max}, special={special_count}")
    windows = _make_windows(token_ids_by_word, capacity, overlap_words)
    hidden_size = int(model.config.hidden_size)
    collected: list[list[np.ndarray]] = [[] for _ in original_words]
    token_mappings: list[TokenMapping] = []
    window_metadata: list[dict[str, Any]] = []

    with torch.inference_mode():
        for window_index, window in enumerate(windows):
            flat_ids: list[int] = []
            flat_map: list[tuple[int, int]] = []
            for word_index, ids, offset in window:
                for local_index, token_id in enumerate(ids):
                    flat_ids.append(token_id)
                    flat_map.append((word_index, offset + local_index))
            full_ids = tokenizer.build_inputs_with_special_tokens(flat_ids)
            special_mask = tokenizer.get_special_tokens_mask(flat_ids, already_has_special_tokens=False)
            if len(full_ids) != len(special_mask):
                raise RuntimeError("Tokenizer returned inconsistent special-token mapping")
            inputs: dict[str, Any] = {
                "input_ids": torch.tensor([full_ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=device),
            }
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(flat_ids)
            if "token_type_ids" in getattr(tokenizer, "model_input_names", []):
                inputs["token_type_ids"] = torch.tensor([token_type_ids], dtype=torch.long, device=device)
            hidden = model(**inputs).last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
            raw_position = 0
            for token_position, is_special in enumerate(special_mask):
                if is_special:
                    continue
                word_index, subword_index = flat_map[raw_position]
                token_id = flat_ids[raw_position]
                collected[word_index].append(hidden[token_position])
                token_mappings.append(TokenMapping(
                    window_index=window_index,
                    token_position=token_position,
                    token_id=token_id,
                    token=tokenizer.convert_ids_to_tokens(token_id),
                    word_index=word_index,
                    subword_index=subword_index,
                ))
                raw_position += 1
            window_metadata.append({
                "window_index": window_index,
                "word_indices": sorted({item[0] for item in window}),
                "model_input_length": len(full_ids),
                "content_token_count": len(flat_ids),
            })

    features = np.full((len(original_words), hidden_size), np.nan, dtype=np.float32)
    mask = np.zeros(len(original_words), dtype=np.uint8)
    for index, representations in enumerate(collected):
        if representations:
            features[index] = np.mean(np.stack(representations), axis=0)
            mask[index] = 1
    result = TextFeatureResult(
        features=features,
        mask=mask,
        token_mappings=token_mappings,
        feature_dim=hidden_size,
        valid_length=int(mask.sum()),
        model_name=model_name,
        model_revision=getattr(model.config, "_commit_hash", None),
        pooling=pooling,
        windows=window_metadata,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir / "text_features.npz", features=features, mask=mask)
        write_json(output_dir / "text_features_metadata.json", result.metadata())
    return result
