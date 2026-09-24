"""计算"额外测量量"并落盘，供绘图与建表引用（避免图中出现手抄数字）。

当前产出：
- ``analysis/results/modality_attribution.csv``：门控基线与 FUSE-Net 的模态归因份额。
  归因定义为 MDF/门控在**每个位置**给出的跨模态聚合权重，按有效位取平均。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis import loader

RESULTS = Path(__file__).resolve().parent / "results"

CHECKPOINTS = {
    "门控基线": "outputs/runs/availability_s2026/best.pt",
    "FUSE-Net": "outputs/runs/fuse_s2026/best.pt",
}


def attribution(checkpoint: Path, device: torch.device, limit: int = 512) -> dict:
    from scripts.infer import load_model, run_model
    from scripts.train import as_tensors, make_batch, move_inputs
    from utils.data import load_main

    model = load_model(checkpoint, device)
    data = as_tensors(load_main(None, "valid", need_teacher=model.text_mode == "bert"))
    shares = []
    with torch.no_grad():
        for start in range(0, min(limit, len(data["tokens"])), 64):
            indices = torch.arange(start, min(start + 64, len(data["tokens"])))
            batch = move_inputs(make_batch(data, indices), device)
            output = run_model(model, batch)
            weights = output["modality_weights"]
            valid = output["time_weights"] > 1e-6
            for row in range(weights.shape[0]):
                if valid[row].any():
                    shares.append(weights[row][valid[row]].mean(0).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    array = np.asarray(shares)
    return {"text": float(array[:, 0].mean()), "audio": float(array[:, 1].mean()),
            "vision": float(array[:, 2].mean()), "n": int(array.shape[0])}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, path in CHECKPOINTS.items():
        stats = attribution(loader.PROJECT / path, device)
        rows.append({"model": name, **stats})
        print(f"{name}: text={stats['text']:.4f} audio={stats['audio']:.4f} "
              f"vision={stats['vision']:.4f} (n={stats['n']})")
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "modality_attribution.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {RESULTS / 'modality_attribution.csv'}")


if __name__ == "__main__":
    main()
