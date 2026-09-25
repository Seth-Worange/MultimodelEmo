"""Train-only feature normalization that preserves missing-value semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InputNormalization:
    """Per-feature statistics fitted on valid training positions only."""

    audio_mean: np.ndarray
    audio_std: np.ndarray
    vision_mean: np.ndarray
    vision_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "audio_mean": self.audio_mean.astype(np.float32).tolist(),
            "audio_std": self.audio_std.astype(np.float32).tolist(),
            "vision_mean": self.vision_mean.astype(np.float32).tolist(),
            "vision_std": self.vision_std.astype(np.float32).tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "InputNormalization | None":
        if not value:
            return None
        required = ("audio_mean", "audio_std", "vision_mean", "vision_std")
        if any(key not in value for key in required):
            raise ValueError("incomplete input normalization statistics")
        return cls(*(np.asarray(value[key], dtype=np.float32) for key in required))


def _stats(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(mask, dtype=bool)
    flat = np.asarray(values, dtype=np.float64)[valid]
    if flat.size == 0:
        raise ValueError("cannot fit normalization without valid feature positions")
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # Constant features are already standardized by definition; keep them finite.
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def fit_input_normalization(data: dict[str, Any]) -> InputNormalization:
    """Fit audio/vision statistics from the training split only."""
    return InputNormalization(
        *_stats(data["audio"], data["audio_mask"]),
        *_stats(data["vision"], data["vision_mask"]),
    )


def apply_input_normalization(data: dict[str, Any], stats: InputNormalization) -> dict[str, Any]:
    """Normalize valid positions and force missing/padding positions back to zero.

    The function returns a shallow copy and never mutates the caller's arrays or tensors.
    It accepts NumPy arrays (loader/training) and torch tensors (tests/inference helpers).
    """
    out = dict(data)
    for name, mean, std in (
        ("audio", stats.audio_mean, stats.audio_std),
        ("vision", stats.vision_mean, stats.vision_std),
    ):
        value = data[name]
        mask = data[f"{name}_mask"]
        if hasattr(value, "clone"):
            normalized = value.clone()
            mean_t = value.new_tensor(mean)
            std_t = value.new_tensor(std)
            normalized = (normalized - mean_t) / std_t
            normalized = normalized * mask.unsqueeze(-1).to(normalized.dtype)
        else:
            normalized = (np.asarray(value, dtype=np.float32).copy() - mean) / std
            normalized *= np.asarray(mask, dtype=normalized.dtype)[..., None]
        out[name] = normalized
    return out


def normalization_summary(stats: InputNormalization | None) -> dict[str, Any] | None:
    return stats.to_dict() if stats is not None else None
