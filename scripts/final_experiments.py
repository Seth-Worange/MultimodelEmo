"""统一运行问题二、三的验证与消融，保留命令和输出。"""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from utils.selection import selection_score, view_score


def main():
    config = yaml.safe_load(Path("config/final_experiments.yaml").read_text(encoding="utf-8"))
    root = Path("outputs/final_experiments")
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "branch": subprocess.check_output(
        ["git", "branch", "--show-current"], text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "commands": [], "checkpoints": {}}

    def run(module, args, name):
        command = [sys.executable, "-u", "-m", module, *map(str, args)]
        manifest["commands"].append(command)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(name, flush=True)
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    rows = []
    candidates = [("q2", name, f"outputs/runs/{name}/best.pt")
                  for name in config["q2_candidates"] + config["ablations"]]
    candidates += [("q3", name, path) for name, path in config["q3_candidates"].items()]
    for task, name, checkpoint in candidates:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest["checkpoints"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        out = root / task / name
        out.mkdir(parents=True, exist_ok=True)
        run("scripts.evaluate", ["--checkpoint", path, "--split", "valid", "--data-root", "data",
            "--device", "cuda", "--evaluation-protocol", "sample_v2", "--seed", 2026,
            "--output", out / "validation.json", "--per-sample", out / "predictions.csv"], f"{task}_{name}")
        metrics = json.loads((out / "validation.json").read_text(encoding="utf-8"))
        rows.append({"task": task, "name": name,
                     "selection_score": selection_score(metrics) if task == "q2" else view_score(metrics["clean"]),
                     **{f"{view}_{key}": value for view in ("clean", "local", "interval", "whole")
                        for key, value in metrics[view].items()}})
    with (root / "model_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run("scripts.robustness", ["--checkpoint", config["q2_mainline"], "--data-root", "data",
        "--device", "cuda", "--rates", *config["rates"], "--locations", *config["locations"],
        "--overlap-probability", 0, "--output", root / "q2/robustness.csv"], "q2_robustness")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prediction comparison and robustness complete", flush=True)


if __name__ == "__main__":
    main()
