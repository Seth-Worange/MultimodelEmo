"""按源视频分组自助重采样，估计最终验证指标的有限样本不确定性。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "diagnostics" / "goal67" / "mixed_ensemble_valid.csv"
TARGET = ROOT / "outputs" / "diagnostics" / "goal67" / "q2_group_bootstrap.json"
VIEWS = ("clean", "local", "whole", "interval")


def metrics(data: dict[str, np.ndarray], indices: np.ndarray, view: str) -> tuple[float, float, float]:
    y = data["label_class"][indices]
    prediction = data[f"{view}_class"][indices]
    strength = data[f"{view}_strength"][indices].copy()
    strength[prediction == 1] = 0.0
    truth = data["label_strength"][indices]
    return (float(np.mean(prediction == y)),
            float(f1_score(y, prediction, average="macro", labels=[0, 1, 2], zero_division=0)),
            float(np.mean(np.abs(strength - truth))))


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {key: np.array([float(row[key]) for row in rows]) for key in
            ["label_class", "label_strength"] +
            [f"{view}_{field}" for view in VIEWS for field in ("class", "strength")]}
    group_names = np.array([row["id"].split("$_$")[0] for row in rows])
    groups = np.unique(group_names)
    by_group = [np.flatnonzero(group_names == group) for group in groups]
    rng = np.random.default_rng(2026)
    draws = {view: [] for view in VIEWS}
    for _ in range(2000):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([by_group[i] for i in sampled])
        for view in VIEWS:
            draws[view].append(metrics(data, indices, view))
    draw_arrays = {view: np.asarray(values) for view, values in draws.items()}
    complete = np.arange(len(rows))
    report = {"n_samples": len(rows), "n_video_groups": len(groups), "draws": 2000,
              "seed": 2026, "resampling_unit": "source_video", "views": {}}
    for view in VIEWS:
        array = draw_arrays[view]
        diff = array - draw_arrays["clean"]
        report["views"][view] = {
            "point": metrics(data, complete, view),
            "ci95": np.quantile(array, [.025, .975], axis=0).T.tolist(),
            "delta_vs_clean_point": (np.array(metrics(data, complete, view)) -
                                     np.array(metrics(data, complete, "clean"))).tolist(),
            "delta_vs_clean_ci95": np.quantile(diff, [.025, .975], axis=0).T.tolist(),
        }
    TARGET.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"n={len(rows)} video_groups={len(groups)} bootstrap_draws=2000")
    for view in VIEWS:
        print(view, report["views"][view]["point"], report["views"][view]["ci95"])


if __name__ == "__main__":
    main()
