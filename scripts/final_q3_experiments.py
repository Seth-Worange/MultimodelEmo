"""按完整输入验证分数选择问题三骨干并生成专项解释。"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    root = Path("outputs/final_experiments")
    config = yaml.safe_load(Path("config/final_experiments.yaml").read_text(encoding="utf-8"))
    with (root / "model_comparison.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    candidates = [row for row in rows if row["task"] == "q3" or
                  row["name"] in config["q2_candidates"]]
    selected = min(candidates, key=lambda row:
        .5 * (1 - float(row["clean_macro_f1"])) + float(row["clean_mae"]) / 6)
    name = selected["name"]
    checkpoint = config["q3_candidates"].get(name, f"outputs/runs/{name}/best.pt")
    path = root / "q3/predictions"
    path.mkdir(parents=True, exist_ok=True)
    selection = {"name": name, "checkpoint": checkpoint,
                 "criterion": "clean Macro-F1 and MAE equal weighted; neutral_zero=false",
                 "candidates": candidates, "commands": []}

    def run(module, args, label):
        command = [sys.executable, "-u", "-m", module, *map(str, args)]
        selection["commands"].append(command)
        (root / "q3/selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print(label, flush=True)
        with (root / f"{label}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    run("scripts.train", ["--config", "config/q2_fuse_lowaux_ab_noavailability.yaml"], "ab_noavailability_train")
    run("scripts.evaluate", ["--config", "config/q2_fuse_lowaux_ab_noavailability.yaml",
        "--per-sample", root / "q2/ab_noavailability_s2026/predictions.csv"], "ab_noavailability_evaluate")
    run("scripts.infer", ["--part", "q2", "--checkpoint", config["q2_mainline"], "--data-root", "data",
        "--device", "cuda", "--output-dir", root / "q2/predictions"], "q2_special_predictions")
    run("scripts.validate_q3_explanations", ["--checkpoint", checkpoint,
        "--output-dir", root / "q3/validation_explanation"], "q3_validation_explanations")
    alignment = Path("outputs/runs/main/q3_alignment.json")
    run("scripts.infer", ["--part", "q3", "--checkpoint", checkpoint, "--data-root", "data",
        "--device", "cuda", "--alignment-file", alignment, "--output-dir", path], "q3_special_predictions")
    run("scripts.refine_q3_evidence", ["--checkpoint", checkpoint, "--data-root", "data", "--device", "cuda",
        "--alignment-file", alignment, "--evidence-csv", path / "q3_evidence_windows.csv",
        "--predictions-csv", path / "q3_predictions.csv", "--output-dir", path], "q3_special_refinement")
    run("scripts.build_q3_cards", ["--output-dir", path, "--alignment-file", alignment], "q3_cards")
    print(f"Q3 selected {name}; all outputs saved", flush=True)


if __name__ == "__main__":
    main()
