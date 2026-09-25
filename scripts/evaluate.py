"""在验证集或独立测试集上评估模型，可输出逐样本预测用于错误归因。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from utils.data import load_main
from utils.config import parse_config_args
from scripts.infer import load_model
from utils.normalization import apply_input_normalization
from scripts.train import VIEWS, as_tensors, evaluate, predict_split

CSV_FIELDS = ("id", "label_class", "label_strength") + tuple(
    f"{view}_{name}" for view in VIEWS
    for name in ("class", "correct", "strength", "p_negative", "p_neutral", "p_positive")
)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def confusion(classes: np.ndarray, predictions: np.ndarray, name: str) -> str:
    lines = [f"混淆矩阵（行=真实 0负/1中/2正，列=预测）[{name}]"]
    for label in range(3):
        counts = [int(((classes == label) & (predictions == other)).sum()) for other in range(3)]
        lines.append("  " + " ".join(f"{value:5d}" for value in counts))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append",
                        help="可重复指定多个检查点并集成")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "runs" / "main" / "test_metrics.json")
    parser.add_argument("--per-sample", type=Path, default=None,
                        help="输出逐样本预测 CSV 的路径，用于错误归因")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--drop-modalities", nargs="+", default=None,
                        choices=["text", "audio", "vision"],
                        help="评估时永久遮蔽指定模态（对应训练时的单模态基线）")
    args = parse_config_args(parser, "evaluate")
    if not args.checkpoint:
        parser.error("provide --checkpoint or configure checkpoints")
    drop = tuple(args.drop_modalities or ())

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device)
    raw_data = load_main(args.data_root, args.split, need_teacher=model.text_mode == "bert")
    if getattr(model, "input_normalization", None) is not None:
        raw_data = apply_input_normalization(raw_data, model.input_normalization)
    data = as_tensors(raw_data)
    text_encoder = None
    view_metrics = evaluate(model, data, args.batch_size, device, seed=2026,
                            text_encoder=text_encoder, drop=drop)
    result = {"split": args.split, "n": len(data["tokens"]), "views": list(VIEWS), **view_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved {args.output}")

    if args.per_sample:
        predicted = predict_split(model, data, args.batch_size, device,
                                  text_encoder=text_encoder, seed=2026, drop=drop)
        classes = data["classes"].numpy()
        sentiment = data["sentiment"].numpy()
        rows = []
        for index, sample_id in enumerate(data["ids"]):
            row = {"id": sample_id, "label_class": int(classes[index]),
                   "label_strength": round(float(sentiment[index]), 6)}
            for view in VIEWS:
                probabilities = softmax(predicted[view]["logits"][index])
                guess = int(probabilities.argmax())
                row[f"{view}_class"] = guess
                row[f"{view}_correct"] = int(guess == classes[index])
                row[f"{view}_strength"] = round(float(predicted[view]["sentiment"][index]), 6)
                row[f"{view}_p_negative"] = round(float(probabilities[0]), 6)
                row[f"{view}_p_neutral"] = round(float(probabilities[1]), 6)
                row[f"{view}_p_positive"] = round(float(probabilities[2]), 6)
            rows.append(row)
        args.per_sample.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} per-sample rows to {args.per_sample}")
        for view in VIEWS:
            guesses = np.asarray([row[f"{view}_class"] for row in rows])
            print(confusion(classes, guesses, view))


if __name__ == "__main__":
    main()
