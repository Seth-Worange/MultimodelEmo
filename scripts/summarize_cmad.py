"""Summarize validation metrics and Gate modality weights for CMAD runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from model import AffectiveModel
from scripts.train import as_tensors, make_batch, metrics, model_inputs, move_inputs
from utils.augmentation import mask_batch
from utils.data import load_main


def load_model(path: Path, device: torch.device) -> AffectiveModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = AffectiveModel(**{key: value for key, value in config.items()
                              if key in {"embedding_dim", "hidden_dim", "fusion", "text_mode",
                                         "audio_dynamics", "regression_mode", "dropout"}})
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def score(model: AffectiveModel, data: dict[str, torch.Tensor], device: torch.device,
          batch_size: int, masked: bool) -> dict[str, float]:
    logits, sentiment, weights = [], [], []
    for start in range(0, len(data["tokens"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        batch = make_batch(data, indices)
        if masked:
            batch = mask_batch(batch, seed=2026 + start, probability=1.0,
                               pattern="interval", ratios=(0.4,),
                               interval_modalities=("audio", "vision"),
                               location="middle")
        batch = move_inputs(batch, device)
        output = model(*model_inputs(batch))
        logits.append(output["logits"].cpu().numpy())
        sentiment.append(output["sentiment"].cpu().numpy())
        valid = output["valid_steps"]
        weights.append((output["modality_weights"], valid))
    result = metrics(data["classes"].numpy(), data["sentiment"].numpy(),
                     np.concatenate(logits), np.concatenate(sentiment))
    weight_values = torch.cat([value for value, _ in weights], dim=0)
    valid_values = torch.cat([value for _, value in weights], dim=0)
    numerator = (weight_values * valid_values.unsqueeze(-1)).sum(dim=(0, 1))
    denominator = valid_values.sum().clamp_min(1)
    mean_weights_tensor = numerator / denominator
    mean_weights = mean_weights_tensor.cpu().numpy()
    if bool(valid_values.any()) and not torch.isclose(
            mean_weights_tensor.sum(), torch.ones((), dtype=mean_weights_tensor.dtype), atol=1e-5):
        raise RuntimeError(f"valid-only modality weights must sum to 1, got {mean_weights.sum()}")
    result.update({"alpha_text": float(mean_weights[0]),
                   "alpha_audio": float(mean_weights[1]),
                   "alpha_vision": float(mean_weights[2])})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("..") / "data")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "cmad_summary.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    args = parser.parse_args()
    device = torch.device("cpu")
    data = as_tensors(load_main(args.data_root, "valid", need_teacher=True))
    rows = []
    for checkpoint in args.checkpoint:
        model = load_model(checkpoint, device)
        for view, masked in (("clean", False), ("interval_audio_vision_40_middle", True)):
            rows.append({"checkpoint": str(checkpoint), "view": view,
                         **score(model, data, device, args.batch_size, masked)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
