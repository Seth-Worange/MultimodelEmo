"""按标签类别与强度比较两组验证集逐样本预测。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_predictions(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def summarize(rows: list[dict], subset: np.ndarray, view: str) -> dict:
    chosen = [row for row, keep in zip(rows, subset) if keep]
    if not chosen:
        return {"n": 0}
    truth = np.array([float(row["label_strength"]) for row in chosen])
    prediction = np.array([float(row[f"{view}_strength"]) for row in chosen])
    correct = np.array([int(row[f"{view}_correct"]) for row in chosen])
    return {"n": len(chosen), "accuracy": float(correct.mean()),
            "mae": float(np.abs(truth - prediction).mean()),
            "mean_predicted_abs": float(np.abs(prediction).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = read_predictions(args.first), read_predictions(args.second)
    if first.keys() != second.keys():
        raise ValueError("Two prediction files must contain identical sample IDs")
    ids = list(first)
    first_rows, second_rows = [first[key] for key in ids], [second[key] for key in ids]
    labels = np.array([int(row["label_class"]) for row in first_rows])
    strength = np.array([abs(float(row["label_strength"])) for row in first_rows])
    groups = {
        "all": np.ones(len(ids), dtype=bool),
        "negative": labels == 0,
        "neutral": labels == 1,
        "positive": labels == 2,
        "weak_nonzero": (strength > 0) & (strength <= 0.5),
        "medium": (strength > 0.5) & (strength <= 1),
        "strong": strength > 1,
    }
    report = {"first": str(args.first), "second": str(args.second), "groups": {}}
    for group, selection in groups.items():
        report["groups"][group] = {
            name: summarize(rows, selection, "clean")
            for name, rows in (("first", first_rows), ("second", second_rows))
        }
    a = np.array([int(row["clean_correct"]) for row in first_rows], dtype=bool)
    b = np.array([int(row["clean_correct"]) for row in second_rows], dtype=bool)
    report["paired"] = {"first_only_correct": int((a & ~b).sum()),
                        "second_only_correct": int((~a & b).sum()),
                        "both_wrong": int((~a & ~b).sum())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
