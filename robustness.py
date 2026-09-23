"""Measure validation performance by missing modality, interval position, and rate."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from augmentation import mask_batch
from data import load_main
from infer import load_model
from train import as_tensors, make_batch, metrics, model_inputs, move_inputs


@torch.inference_mode()
def score_case(model, data, device, batch_size, *, modality=None, rate=None, location=None, seed=2026):
    model.eval()
    rng = random.Random(seed)
    logits_all, sentiment_all = [], []
    for start in range(0, len(data["tokens"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        batch = make_batch(data, ix)
        if modality is not None:
            batch = mask_batch(batch, seed=rng.randrange(2**31), probability=1.0,
                               modality=modality, rate=rate, location=location)
        batch = move_inputs(batch, device)
        output = model(*model_inputs(batch))
        logits_all.append(output["logits"].cpu().numpy())
        sentiment_all.append(output["sentiment"].cpu().numpy())
    return metrics(data["classes"].numpy(), data["sentiment"].numpy(),
                   np.concatenate(logits_all), np.concatenate(sentiment_all))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="repeat to evaluate an ensemble of checkpoints")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "runs" / "main" / "robustness.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device)
    data = as_tensors(load_main(args.data_root, "valid", need_teacher=model.text_mode == "bert"))
    rows = []
    cases = [(None, 0.0, "none")] + [(m, rate, location) for m in ("text", "audio", "vision")
                                        for rate in args.rates for location in ("start", "middle", "end")]
    for index, (modality, rate, location) in enumerate(cases):
        result = score_case(model, data, device, args.batch_size, modality=modality,
                            rate=rate if modality else None, location=location if modality else None,
                            seed=2026 + index)
        row = {"modality": modality or "none", "missing_rate": rate, "location": location,
               "n": len(data["tokens"]), **result}
        rows.append(row)
        print(f"{row['modality']:>6} rate={rate:.1f} {location:>6} "
              f"F1={result['macro_f1']:.4f} MAE={result['mae']:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} validation cases to {args.output}")


if __name__ == "__main__":
    main()
