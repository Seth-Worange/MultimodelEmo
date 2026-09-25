"""Summarize B0/B1/B2 validation selection and the frozen R1-R5 suite."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent / "outputs" / "q2_fuse_missing"
FIXED = {
    "R1": ("audio+vision", "0.4", "middle"),
    "R2": ("text", "0.4", "middle"),
    "R3": ("vision", "0.4", "middle"),
    "R4": ("text+vision", "0.4", "middle"),
    "R5": ("audio+vision", "0.6", "middle"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = []
    for name in ("B0", "B1", "B2"):
        folder = ROOT / name / "s2026"
        metrics = read_csv(folder / "metrics.csv")
        best = min(metrics, key=lambda row: float(row["val_score"]))
        robustness = read_csv(folder / "robustness.csv")
        fixed = {}
        for label, (modalities, rate, location) in FIXED.items():
            match = next(row for row in robustness
                         if row["pattern"] == "interval"
                         and row["missing_modalities"] == modalities
                         and row["missing_rate"] == rate
                         and row["location"] == location)
            fixed[label] = match
        robust_f1 = sum(float(fixed[label]["macro_f1"]) for label in FIXED) / len(FIXED)
        robust_mae = sum(float(fixed[label]["mae"]) for label in FIXED) / len(FIXED)
        row = {
            "candidate": name,
            "best_epoch": best["epoch"],
            "selection_score": best["val_score"],
            "clean_accuracy": best["clean_accuracy"],
            "clean_macro_f1": best["clean_macro_f1"],
            "clean_mae": best["clean_mae"],
            "clean_pearson": best["clean_pearson"],
            "robust_f1": f"{robust_f1:.9f}",
            "robust_mae": f"{robust_mae:.9f}",
        }
        for label, value in fixed.items():
            row[f"{label}_macro_f1"] = value["macro_f1"]
            row[f"{label}_mae"] = value["mae"]
        rows.append(row)
    output = ROOT / "candidate_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["candidate"], "score=", row["selection_score"],
              "clean_f1=", row["clean_macro_f1"], "robust_f1=", row["robust_f1"],
              "robust_mae=", row["robust_mae"])
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
