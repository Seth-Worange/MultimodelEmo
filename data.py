"""Loading and validation for the supplied aligned MOSEI features."""

from __future__ import annotations

import pickle
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

SEQ_LEN = 50
VOCAB_SIZE = 30522
TEXT_DIM = 768
AUDIO_DIM = 74
VISION_DIM = 35


def resolve_data_root(path: str | Path | None = None) -> Path:
    root = Path(path) if path else Path(__file__).resolve().parent / "data"
    if not (root / "附件2-数据集特征文件").is_dir():
        raise FileNotFoundError(f"Expected 附件2-数据集特征文件 under {root}")
    return root


def read_pickle(path: str | Path) -> Any:
    """Read trusted contest pickles, including files written with NumPy 2 paths."""
    # The supplied files are trusted local inputs; never unpickle arbitrary downloads.
    if "numpy._core" not in sys.modules:
        try:
            numpy_core = import_module("numpy._core")
        except ModuleNotFoundError:
            numpy_core = import_module("numpy.core")
        sys.modules["numpy._core"] = numpy_core
        for module_name in ("numeric", "multiarray"):
            module = import_module(f"{numpy_core.__name__}.{module_name}")
            sys.modules[f"numpy._core.{module_name}"] = module
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _check_tokens(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (3, SEQ_LEN):
        raise ValueError(f"text_bert must have shape (N, 3, {SEQ_LEN}), got {values.shape}")
    rounded = np.rint(values)
    if not np.isfinite(values).all() or not np.allclose(values, rounded, atol=1e-4):
        raise ValueError("text_bert contains non-integer or non-finite token values")
    tokens = rounded.astype(np.int64)
    if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= VOCAB_SIZE:
        raise ValueError(f"Token ids must be in [0, {VOCAB_SIZE})")
    return tokens


def _feature_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).all(axis=-1) & (np.abs(x).max(axis=-1) > 1e-8)


def prepare_split(raw: dict[str, Any], *, need_teacher: bool = True) -> dict[str, np.ndarray]:
    tokens = _check_tokens(np.asarray(raw["text_bert"]))
    audio = np.asarray(raw["audio"], dtype=np.float32)
    vision = np.asarray(raw["vision"], dtype=np.float32)
    if audio.shape[1:] != (SEQ_LEN, AUDIO_DIM) or vision.shape[1:] != (SEQ_LEN, VISION_DIM):
        raise ValueError(f"Expected audio/vision (*, 50, 74/35), got {audio.shape}, {vision.shape}")
    attention = tokens[:, 1, :] > 0
    content = attention & (tokens[:, 0, :] != 101) & (tokens[:, 0, :] != 102)
    out = {
        "tokens": tokens,
        "lengths": attention.sum(axis=1).clip(1).astype(np.int64),
        "text_mask": content,
        "audio": np.nan_to_num(audio),
        "vision": np.nan_to_num(vision),
        "audio_mask": _feature_mask(audio),
        "vision_mask": _feature_mask(vision),
        "classes": np.asarray(raw["classification_labels"], dtype=np.int64),
        "sentiment": np.asarray(raw["regression_labels"], dtype=np.float32),
    }
    if not np.isin(out["classes"], [0, 1, 2]).all():
        raise ValueError("classification_labels must use 0=Negative, 1=Neutral, 2=Positive")
    for name in ("tokens", "audio", "vision", "classes", "sentiment"):
        if not np.isfinite(out[name]).all():
            raise ValueError(f"Non-finite values in {name}")
    if need_teacher:
        teacher = np.asarray(raw["text"], dtype=np.float32)
        if teacher.shape != (len(tokens), SEQ_LEN, TEXT_DIM):
            raise ValueError(f"text teacher must have shape (N, 50, 768), got {teacher.shape}")
        out["teacher"] = np.nan_to_num(teacher)
    return out


def load_main(data_root: str | Path | None, split: str, *, need_teacher: bool = True) -> dict[str, np.ndarray]:
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown split: {split}")
    root = resolve_data_root(data_root)
    path = root / "附件2-数据集特征文件" / "aligned_50.pkl"
    raw = read_pickle(path)
    if split not in raw:
        raise KeyError(f"{path} has no {split!r} split")
    return prepare_split(raw[split], need_teacher=need_teacher)


def prepare_sample(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize one attached test item to the same 50-position model interface."""
    if "test" in raw and isinstance(raw["test"], dict):
        raw = raw["test"]
    tokens = np.asarray(raw["text_bert"])
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.shape != (3, SEQ_LEN):
        raise ValueError(f"Expected one text_bert item with shape (3, 50), got {tokens.shape}")
    tokens = _check_tokens(tokens[None, ...])[0]

    def feature(name: str, dim: int) -> np.ndarray:
        value = np.asarray(raw[name], dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.shape != (SEQ_LEN, dim):
            raise ValueError(f"{name} must have shape (50, {dim}), got {value.shape}")
        return np.nan_to_num(value)

    attention = tokens[1] > 0
    content = attention & (tokens[0] != 101) & (tokens[0] != 102)
    audio, vision = feature("audio", AUDIO_DIM), feature("vision", VISION_DIM)
    item = {
        "tokens": tokens,
        "lengths": np.asarray(max(1, int(attention.sum())), dtype=np.int64),
        "text_mask": content,
        "audio": audio,
        "vision": vision,
        "audio_mask": _feature_mask(audio),
        "vision_mask": _feature_mask(vision),
        "id": str(raw.get("id", "")),
        "raw_text": str(raw.get("raw_text", "")),
    }
    if "text" in raw:
        teacher = np.asarray(raw["text"], dtype=np.float32)
        if teacher.ndim == 3 and teacher.shape[0] == 1:
            teacher = teacher[0]
        if teacher.shape != (SEQ_LEN, TEXT_DIM):
            raise ValueError(f"text teacher must have shape ({SEQ_LEN}, {TEXT_DIM}), got {teacher.shape}")
        item["teacher"] = np.nan_to_num(teacher)
    return item


def demo() -> None:
    tokens = np.zeros((1, 3, SEQ_LEN), dtype=np.float32)
    tokens[0, 0, :3] = [101, 200, 102]
    tokens[0, 1, :3] = 1
    assert _check_tokens(tokens).dtype == np.int64
    try:
        _check_tokens(np.full((1, 3, SEQ_LEN), VOCAB_SIZE, dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("Out-of-range token ids must fail validation")


if __name__ == "__main__":
    demo()
    print("data.py checks passed")
