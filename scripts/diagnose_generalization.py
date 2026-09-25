"""比较同一检查点在附件2训练集与验证集的完整输入错误。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.infer import load_model
from scripts.train import as_tensors, metrics, predict_split
from utils.config import parse_config_args
from utils.data import load_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parse_config_args(parser, "diagnose")
    if not args.checkpoint or not args.output:
        parser.error("checkpoint and output are required")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    report = {"checkpoint": str(args.checkpoint), "splits": {}}
    for split in ("train", "valid"):
        data = as_tensors(load_main(args.data_root, split))
        predicted = predict_split(model, data, args.batch_size, device, views=("clean",))["clean"]
        true = data["classes"].numpy()
        guess = predicted["logits"].argmax(1)
        target = data["sentiment"].numpy()
        value = predicted["sentiment"]
        confusion = np.zeros((3, 3), dtype=np.int64)
        np.add.at(confusion, (true, guess), 1)
        report["splits"][split] = {
            "n": len(true), "metrics": metrics(true, target, predicted["logits"], value),
            "confusion_rows_true": confusion.tolist(),
            "neutral_recall": float(confusion[1, 1] / max(1, confusion[1].sum())),
            "strong_mae": float(np.abs(target[np.abs(target) > 1] - value[np.abs(target) > 1]).mean()),
        }
        del data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], indent=2))


if __name__ == "__main__":
    main()
