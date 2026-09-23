"""Contiguous local-drop augmentation used by training and robustness analysis."""

from __future__ import annotations

import random

import torch


def _drop_window(mask: torch.Tensor, rate: float, location: str | None, rng: random.Random) -> torch.Tensor:
    positions = torch.where(mask)[0]
    if positions.numel() == 0:
        return positions
    span = max(1, round(rate * positions.numel()))
    lo, hi = int(positions[0]), int(positions[-1]) + 1
    start_hi = max(lo, hi - span)
    if location == "start":
        start = lo
    elif location == "middle":
        start = max(lo, (lo + hi - span) // 2)
    elif location == "end":
        start = start_hi
    else:
        start = rng.randint(lo, start_hi)
    index = torch.arange(mask.numel(), device=mask.device)
    return torch.where(mask & (index >= start) & (index < start + span))[0]


def mask_batch(
    batch: dict[str, torch.Tensor],
    *,
    seed: int,
    rate_range: tuple[float, float] = (0.1, 0.4),
    probability: float = 0.75,
    modality: str | None = None,
    rate: float | None = None,
    location: str | None = None,
) -> dict[str, torch.Tensor]:
    """Return a cloned batch with one or two continuous modality spans hidden."""
    rng = random.Random(seed)
    out = batch.copy()
    for key in ("tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask"):
        out[key] = batch[key].clone()
    names = ("text", "audio", "vision")
    masks = {name: out[f"{name}_mask"] for name in names}
    for row in range(len(out["tokens"])):
        if modality is None:
            if rng.random() >= probability:
                continue
            count = rng.choice((1, 1, 1, 2))
            selected = rng.sample(names, count)
        else:
            selected = [modality]
        for name in selected:
            base_mask = masks[name][row]
            if not base_mask.any():
                continue
            missing_rate = rate if rate is not None else rng.uniform(*rate_range)
            drops = _drop_window(base_mask, missing_rate, location, rng)
            if drops.numel() == 0:
                continue
            masks[name][row, drops] = False
            if name == "text":
                out["tokens"][row, 0, drops] = 0
            else:
                out[name][row, drops] = 0
    return out


def demo() -> None:
    batch = {
        "tokens": torch.ones(2, 3, 50, dtype=torch.long),
        "audio": torch.ones(2, 50, 74),
        "vision": torch.ones(2, 50, 35),
        "text_mask": torch.ones(2, 50, dtype=torch.bool),
        "audio_mask": torch.ones(2, 50, dtype=torch.bool),
        "vision_mask": torch.ones(2, 50, dtype=torch.bool),
    }
    masked = mask_batch(batch, seed=7, probability=1.0, modality="audio", rate=0.2, location="middle")
    assert torch.equal(batch["audio_mask"].sum(dim=1) - masked["audio_mask"].sum(dim=1), torch.full((2,), 10))
    assert masked["audio"].sum() < batch["audio"].sum()
    assert batch["audio"].sum() == 100 * 74


if __name__ == "__main__":
    demo()
    print("augmentation.py checks passed")
