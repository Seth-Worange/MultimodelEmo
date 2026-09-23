"""Evaluate the selected checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import load_main
from infer import load_model
from train import as_tensors, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="repeat to evaluate an ensemble of checkpoints")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "runs" / "main" / "test_metrics.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device)
    data = as_tensors(load_main(args.data_root, "test", need_teacher=model.text_mode == "bert"))
    clean, masked = evaluate(model, data, args.batch_size, device, seed=2026)
    result = {"n": len(data["tokens"]), "clean": clean, "masked": masked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
