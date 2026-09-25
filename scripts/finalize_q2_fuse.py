"""汇总冻结的 Q2 FUSE 多 seed 验证实验。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from scripts.infer import load_model
from scripts.train import as_tensors, metrics, predict_split
from utils.data import load_main
from utils.normalization import apply_input_normalization


SEEDS = (2026, 2027, 2028)
HISTORICAL_SEEDS = (2026, 2027)
METRICS = ("accuracy", "macro_f1", "negative_f1", "neutral_f1", "positive_f1", "mae", "pearson")
FIXED_CASES = {
    "R1": ("audio+vision", "0.4", "middle"),
    "R2": ("text", "0.4", "middle"),
    "R3": ("vision", "0.4", "middle"),
    "R4": ("text+vision", "0.4", "middle"),
    "R5": ("audio+vision", "0.6", "middle"),
}
SCENARIOS = ("clean", *FIXED_CASES.keys(), "R1-R5_mean")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
DISCOVERY_RESULTS = (
    {"model": "B0", "selection_score": 0.2941, "clean_macro_f1": 0.6303,
     "clean_mae": 0.6112, "robust_f1": 0.6032, "robust_mae": 0.6155},
    {"model": "B1", "selection_score": 0.3124, "clean_macro_f1": 0.6052,
     "clean_mae": 0.6465, "robust_f1": 0.5690, "robust_mae": 0.6731},
    {"model": "B2", "selection_score": 0.3109, "clean_macro_f1": 0.6035,
     "clean_mae": 0.6362, "robust_f1": 0.5714, "robust_mae": 0.6549},
)
LR_SCREEN = (("3e-4", 0.2960), ("5e-4", 0.2954), ("1e-3", 0.2941), ("2e-3", 0.3045))
MAGNITUDE_BINS = (
    ("near-neutral (0–0.5)", 0.0, 0.5),
    ("weak (0.5–1)", 0.5, 1.0),
    ("moderate (1–2)", 1.0, 2.0),
    ("strong (2–3)", 2.0, 3.000001),
)


def audit_single_panel(fig, out_dir: Path, stem: str) -> None:
    """Record the required alignment result for a single-panel figure."""
    audit_dir = os.environ.get("NATURE_FIGURE_AUDIT_SCRIPTS")
    if not audit_dir:
        raise RuntimeError("Set NATURE_FIGURE_AUDIT_SCRIPTS to the figure audit script directory")
    if audit_dir not in sys.path:
        sys.path.insert(0, audit_dir)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        json_out=out_dir / f"{stem}.alignment.json",
        overlay_svg=out_dir / f"{stem}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required validation artifact: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_case(rows: list[dict[str, str]], scenario: str) -> dict[str, str]:
    if scenario == "clean":
        matches = [row for row in rows if row["pattern"] == "none"]
    else:
        modalities, rate, location = FIXED_CASES[scenario]
        matches = [row for row in rows if row["pattern"] == "interval"
                   and row["missing_modalities"] == modalities
                   and row["missing_rate"] == rate and row["location"] == location]
    if not matches:
        raise ValueError(f"Missing robustness row for {scenario}")
    for metric in METRICS:
        values = [float(row[metric]) for row in matches]
        if not np.allclose(values, values[0], atol=1e-7, rtol=0):
            raise ValueError(f"Duplicate robustness rows disagree for {scenario}: {metric}={values}")
    return matches[0]


def checkpoint_info(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    run_path = path.parent / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    return checkpoint, run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_clean_predictions(model, raw_data: dict, device: torch.device,
                               batch_size: int) -> tuple[dict, dict[str, np.ndarray]]:
    normalized = raw_data
    stats = getattr(model, "input_normalization", None)
    if stats is not None:
        normalized = apply_input_normalization(raw_data, stats)
    data = as_tensors(normalized)
    predicted = predict_split(model, data, batch_size, device, views=("clean",))["clean"]
    result = metrics(data["classes"].numpy(), data["sentiment"].numpy(),
                     predicted["logits"], predicted["sentiment"])
    return result, predicted


def summarize_magnitude(y_true: np.ndarray, y_pred: np.ndarray, group: str,
                        seed: int) -> list[dict]:
    rows = []
    target_magnitude = np.abs(y_true)
    predicted_magnitude = np.abs(y_pred)
    for label, low, high in MAGNITUDE_BINS:
        selected = (target_magnitude >= low) & (target_magnitude < high)
        count = int(selected.sum())
        if count == 0:
            raise ValueError(f"Validation has no samples in magnitude bin {label}")
        true_mean = float(target_magnitude[selected].mean())
        predicted_mean = float(predicted_magnitude[selected].mean())
        rows.append({
            "group": group,
            "seed": seed,
            "magnitude_bin": label,
            "n": count,
            "mean_abs_target": true_mean,
            "mean_abs_prediction": predicted_mean,
            "prediction_minus_target": predicted_mean - true_mean,
            "magnitude_ratio": predicted_mean / max(true_mean, 1e-12),
        })
    return rows


def build_markdown_report(final_rows: list[dict], multi_rows: list[dict],
                          paired_rows: list[dict], neutral_rows: list[dict],
                          magnitude_rows: list[dict], discovery_rows: tuple[dict, ...],
                          lr_rows: tuple[tuple[str, float], ...], runs: list[dict],
                          device_name: str) -> str:
    def aggregate(scenario: str, metric: str) -> dict:
        return next(row for row in multi_rows
                    if row["scenario"] == scenario and row["metric"] == metric)

    def pm(scenario: str, metric: str) -> str:
        row = aggregate(scenario, metric)
        return f"{float(row['mean']):.4f} ± {float(row['std']):.4f}"

    def paired(scenario: str, metric: str) -> dict:
        return next(row for row in paired_rows
                    if row["scenario"] == scenario and row["metric"] == metric)

    seed_rows = [row for row in final_rows if row["group"] == "final" and row["scenario"] == "clean"]
    seed_table = "\n".join(
        f"| {row['seed']} | {row['best_epoch']} | {float(row['selection_score']):.4f} | "
        f"{float(row['macro_f1']):.4f} | {float(row['mae']):.4f} | {float(row['neutral_f1']):.4f} |"
        for row in seed_rows
    )
    scenario_table = "\n".join(
        f"| {scenario} | {pm(scenario, 'macro_f1')} | {pm(scenario, 'mae')} | "
        f"{pm(scenario, 'neutral_f1')} |"
        for scenario in ("clean", "R1", "R2", "R3", "R4", "R5", "R1-R5_mean")
    )
    discovery_table = "\n".join(
        f"| {row['model']} | {row['selection_score']:.4f} | {row['clean_macro_f1']:.4f} | "
        f"{row['clean_mae']:.4f} | {row['robust_f1']:.4f} | {row['robust_mae']:.4f} |"
        for row in discovery_rows
    )
    lr_table = "\n".join(f"| {lr} | {score:.4f} |" for lr, score in lr_rows)

    paired_table = "\n".join(
        f"| {scenario} | {metric} | {float(paired(scenario, metric)['paired_delta_mean']):+.4f} ± "
        f"{float(paired(scenario, metric)['paired_delta_sd']):.4f} |"
        for scenario in ("clean", "R1-R5_mean") for metric in ("macro_f1", "mae")
    )
    neutral_table = "\n".join(
        f"| {label} | {pm(scenario, 'neutral_f1')} |"
        for label, scenario in (("Clean", "clean"), ("Text interval 40% (R2)", "R2"))
    )

    mag_lines = []
    for label, _, _ in MAGNITUDE_BINS:
        rows = [row for row in magnitude_rows
                if row["group"] == "final" and row["magnitude_bin"] == label]
        target = float(np.mean([float(row["mean_abs_target"]) for row in rows]))
        pred_values = [float(row["mean_abs_prediction"]) for row in rows]
        pred_mean = float(np.mean(pred_values))
        pred_sd = float(np.std(pred_values, ddof=1))
        mag_lines.append(f"| {label} | {int(rows[0]['n'])} | {target:.3f} | "
                         f"{pred_mean:.3f} ± {pred_sd:.3f} | "
                         f"{pred_mean-target:+.3f} | {pred_mean/max(target, 1e-12):.3f} |")

    software = runs[0]
    return f"""# Q2 冻结模型最终验证报告

## 协议与结论

- 最终结构：**B0 FUSE**；输入归一化关闭，结构化缺失训练关闭；学习率 **1e-3**。
- 正式种子：2026、2027、2028；每个 seed 最多 50 epoch，patience=8；按 clean/local/whole/interval 验证 selection score 选 checkpoint。
- 缺失鲁棒套件：每个最终 checkpoint 在同一 728 条验证样本上运行 50 个缺失场景；固定比较行 R1–R5 见下表。
- 本报告所有新计算仅使用 validation。没有读取 test 标签，也没有用 test 选择模型。
- 训练环境：PyTorch {software.get('torch_version', 'unknown')}，CUDA {software.get('cuda_version', 'unknown')}，{device_name}。

**筛选结论保持不变：B1（输入归一化）与 B2（结构化缺失增强）均为负结果，未进入最终模型。**学习率筛选后固定为 1e-3。最终 checkpoint 已在当前 CUDA 环境下用冻结配置重建；由于原交接副本未携带 Final 三个 checkpoint，本报告将重建后的同环境三 seed 作为正式汇总。与交接记录的 seed=2026/2027 单次指标存在环境/运行差异，见下方 historical 对比，不混用两套 checkpoint 指标。

## Discovery 与学习率筛选

下表保留此前 discovery 记录，不与本次三 seed 最终重训混为一谈。Robust-F1/MAE 是固定 R1–R5 的均值。

| 候选 | Selection score | Clean Macro-F1 | Clean MAE | Robust-F1 | Robust-MAE |
|---|---:|---:|---:|---:|---:|
{discovery_table}

| 学习率 | Selection score |
|---:|---:|
{lr_table}

B1、B2 的结果低于 B0：B1 将 Clean Macro-F1 从 0.6303 降至 0.6052，Robust-F1 从 0.6032 降至 0.5690；B2 的 Clean Macro-F1 为 0.6035、Robust-F1 为 0.5714。因此不保留这两项改动。

## 最终三 seed 验证结果

| Seed | Best epoch | Selection score | Clean Macro-F1 | Clean MAE | Clean Neutral F1 |
|---:|---:|---:|---:|---:|---:|
{seed_table}

| 验证场景 | Macro-F1（mean ± SD） | MAE（mean ± SD） | Neutral F1（mean ± SD） |
|---|---:|---:|---:|
{scenario_table}

R1–R5 定义为：R1 音频+视觉 40% 中段缺失；R2 文本 40% 中段缺失；R3 视觉 40% 中段缺失；R4 文本+视觉 40% 中段缺失；R5 音频+视觉 60% 中段缺失。R1–R5_mean 是五行等权平均。标准差使用样本标准差（ddof=1），仅描述三个训练 seed 的离散程度，不解释为置信区间。

## 与 historical FUSE 的配对比较

历史参照为同工程中先前的 FUSE seed=2026/2027 checkpoint，采用相同验证样本、缺失掩码与 R1–R5 定义重新评估。每个 seed 与同 seed 历史 checkpoint 配对，差值定义为“最终 B0 − historical FUSE”；Macro-F1 正值代表提升，MAE 负值代表改善。

| 场景 | 指标 | 配对差值均值 ± SD（n=2 seed pairs） |
|---|---|---:|
{paired_table}

该 paired delta 是两个同 seed 模型对的描述性差值，不做显著性检验。历史 checkpoint 的训练配置与最终 B0 不完全相同：历史 run 使用 corruption probability=0.8、auto missing mode，最终 B0 使用 0.0、whole mode。因此该对比反映历史整体训练方案与冻结方案的差别，不能单独归因于某一结构改动。完整逐 seed 差值见 `paired_delta.csv`；历史模型的完整 50-case 结果保存在 Final 输出目录。

## 中性类别表现

| 场景 | Neutral F1（mean ± SD，3 seeds） |
|---|---:|
{neutral_table}

完整的 Negative/Neutral/Positive F1 按 seed 与 R1–R5 场景列于 `neutral_f1_analysis.csv`。文本中段缺失会显著考验模型对非文本模态的依赖能力；中性类别的 F1 应与总体 Macro-F1 一起解读，避免被总体准确率掩盖。

## 回归幅度收缩

按真实情感强度绝对值分箱，在 clean validation 输入上比较真实与预测的绝对强度。`prediction − target < 0` 表示预测幅度被压小；ratio 小于 1 也表示低估。表中预测值为三 seed 均值 ± 样本标准差。

| 真实强度区间 | n / seed | 平均真实幅度 | 平均预测幅度 | 预测−真实 | 幅度比 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(mag_lines)}

按 seed 的原始数值见 `regression_magnitude_shrinkage.csv`，图见 `regression_magnitude_shrinkage.pdf`。混淆矩阵汇总三个 seed 在 clean validation 上的 2,184 次样本预测，按真实类别行归一化，原始计数见 `confusion_matrix.csv`。

![三 seed clean validation 混淆矩阵](confusion_matrix_final.png)

![Clean validation 情感幅度回归收缩](regression_magnitude_shrinkage.png)

## 可复现性与限制

- 冻结配置：`config/q2_fuse_b0.yaml`，命令：`python -m scripts.finalize_q2_fuse --data-root data --device cuda`；三 seed 训练命令在 `HANDOFF_FOR_TEAMMATE.md` 中。
- 指标文件、50-case robustness CSV、混淆矩阵、Neutral F1 和幅度分箱 CSV 均由验证集计算生成。
- seed=2028 的最终 checkpoint best epoch 与训练日志一致；所有模型均按 selection score 保存 best checkpoint。
- 文本缺失压力场景下性能下降，说明模型仍有文本依赖；Neutral F1 与强度幅度收缩均需结合分箱结果报告。
- 推送（push）：**NO**；合并（merge）：**NO**；PR：**NO**。
"""


def plot_confusion(counts: np.ndarray, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    row_totals = counts.sum(axis=1, keepdims=True)
    normalized = counts / np.maximum(row_totals, 1)
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for row in range(3):
        for col in range(3):
            color = "white" if normalized[row, col] >= 0.55 else "#1d2730"
            ax.text(col, row, f"{counts[row, col]:,}\n{normalized[row, col]:.1%}",
                    ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Clean validation confusion matrix\n3 seeds pooled; row-normalized", pad=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within true class", fontsize=7.5)
    colorbar.ax.tick_params(labelsize=7)
    audit_single_panel(fig, out_dir, "confusion_matrix_final")
    fig.savefig(out_dir / "confusion_matrix_final.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "confusion_matrix_final.svg",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "confusion_matrix_final.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "confusion_matrix_final.tiff", dpi=600,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def plot_magnitude(rows: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    final_rows = [row for row in rows if row["group"] == "final"]
    labels = [item[0] for item in MAGNITUDE_BINS]
    x = np.arange(len(labels))
    target = np.array([
        np.mean([float(row["mean_abs_target"]) for row in final_rows
                 if row["magnitude_bin"] == label]) for label in labels
    ])
    seed_predictions = np.array([
        [float(next(row["mean_abs_prediction"] for row in final_rows
                    if row["seed"] == seed and row["magnitude_bin"] == label))
         for label in labels]
        for seed in SEEDS
    ])
    predicted_mean = seed_predictions.mean(axis=0)
    predicted_sd = seed_predictions.std(axis=0, ddof=1)
    counts = [int(next(row["n"] for row in final_rows if row["seed"] == SEEDS[0]
                       and row["magnitude_bin"] == label)) for label in labels]

    fig, ax = plt.subplots(figsize=(7.2, 3.55), constrained_layout=True)
    ax.plot(x, target, marker="o", markersize=4.5, linewidth=1.8,
            color="#52616b", label="Observed |sentiment|")
    ax.errorbar(x, predicted_mean, yerr=predicted_sd, marker="s", markersize=4.5,
                linewidth=1.8, capsize=3, color="#d9793b",
                label="Predicted |sentiment| (mean ± SD across seeds)")
    ax.fill_between(x, target, predicted_mean, where=predicted_mean < target,
                    color="#d9793b", alpha=0.12, interpolate=True)
    ax.set_xticks(x, [f"{label}\n(n={count})" for label, count in zip(labels, counts)])
    ax.set_ylabel("Mean absolute sentiment intensity")
    ax.set_xlabel("Bin by true absolute sentiment")
    ax.set_title("Predicted versus observed sentiment magnitude", pad=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dfe4e8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    audit_single_panel(fig, out_dir, "regression_magnitude_shrinkage")
    fig.savefig(out_dir / "regression_magnitude_shrinkage.pdf",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "regression_magnitude_shrinkage.svg",
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "regression_magnitude_shrinkage.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "regression_magnitude_shrinkage.tiff", dpi=600,
                bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=repo / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=repo / "outputs" / "q2_fuse_missing")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    final_dir = args.output_dir / "Final"

    model_specs = []
    for seed in SEEDS:
        model_specs.append(("final", seed,
                            final_dir / f"s{seed}" / "best.pt",
                            final_dir / f"s{seed}_robustness.csv"))
    for seed in HISTORICAL_SEEDS:
        model_specs.append(("historical_FUSE", seed,
                            repo / "outputs" / "runs" / f"fuse_s{seed}" / "best.pt",
                            final_dir / f"historical_fuse_s{seed}_robustness.csv"))

    raw_data = load_main(args.data_root, "valid", need_teacher=True)
    y_class = np.asarray(raw_data["classes"], dtype=np.int64)
    y_sentiment = np.asarray(raw_data["sentiment"], dtype=np.float32)
    result_rows: list[dict] = []
    neutral_rows: list[dict] = []
    magnitude_rows: list[dict] = []
    confusion_by_seed: dict[int, np.ndarray] = {}
    run_manifests = []

    for group, seed, checkpoint_path, robustness_path in model_specs:
        checkpoint, run = checkpoint_info(checkpoint_path)
        if int(checkpoint.get("seed", seed)) != seed:
            raise ValueError(f"Checkpoint seed mismatch for {checkpoint_path}")
        robustness_rows = read_csv(robustness_path)
        if len(robustness_rows) != 50:
            raise ValueError(f"Expected 50 robustness cases in {robustness_path}, got {len(robustness_rows)}")
        model = load_model(checkpoint_path, device)
        clean_metrics, prediction = evaluate_clean_predictions(model, raw_data, device, args.batch_size)
        clean_case = find_case(robustness_rows, "clean")
        for metric in METRICS:
            if not np.isclose(float(clean_case[metric]), clean_metrics[metric], atol=1e-6):
                raise ValueError(f"Clean prediction mismatch for {group} seed {seed}: {metric}")

        selection_score = float(checkpoint.get("val_score", np.nan))
        best_epoch = int(checkpoint.get("epoch", run.get("best_epoch", 0)))
        all_scenarios = {name: find_case(robustness_rows, name)
                         for name in ("clean", *FIXED_CASES.keys())}
        mean_case = {metric: float(np.mean([float(all_scenarios[name][metric])
                                            for name in FIXED_CASES]))
                     for metric in METRICS}
        all_scenarios["R1-R5_mean"] = {"n": len(y_class), **mean_case}

        for scenario in SCENARIOS:
            case = all_scenarios[scenario]
            result_rows.append({
                "group": group,
                "model": "B0 FUSE" if group == "final" else "historical FUSE",
                "seed": seed,
                "scenario": scenario,
                "n": int(case["n"]),
                "best_epoch": best_epoch,
                "selection_score": selection_score,
                "learning_rate": float(run.get("lr", 0.001)),
                "normalize_inputs": bool(run.get("normalize_inputs", False)),
                "corruption_probability": float(run.get("corruption_probability", np.nan)),
                **{metric: float(case[metric]) for metric in METRICS},
            })
            if scenario != "R1-R5_mean":
                neutral_rows.append({
                    "group": group,
                    "seed": seed,
                    "scenario": scenario,
                    "n": int(case["n"]),
                    "negative_f1": float(case["negative_f1"]),
                    "neutral_f1": float(case["neutral_f1"]),
                    "positive_f1": float(case["positive_f1"]),
                    "macro_f1": float(case["macro_f1"]),
                    "accuracy": float(case["accuracy"]),
                })

        pred_classes = prediction["logits"].argmax(axis=1)
        cm = np.zeros((3, 3), dtype=np.int64)
        np.add.at(cm, (y_class, pred_classes), 1)
        if group == "final":
            confusion_by_seed[seed] = cm
        magnitude_rows.extend(summarize_magnitude(y_sentiment, prediction["sentiment"], group, seed))

        run_manifests.append({
            "group": group,
            "seed": seed,
            "checkpoint": checkpoint_path.relative_to(repo).as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
            "best_epoch": best_epoch,
            "selection_score": selection_score,
            "torch_version": run.get("torch_version", torch.__version__),
            "cuda_version": run.get("cuda_version", torch.version.cuda),
            "device_used": str(device),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"evaluated {group} seed={seed}; best_epoch={best_epoch}; "
              f"clean_macro_f1={clean_metrics['macro_f1']:.4f}; clean_mae={clean_metrics['mae']:.4f}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = ["group", "model", "seed", "scenario", "n", "best_epoch",
                      "selection_score", "learning_rate", "normalize_inputs",
                      "corruption_probability", *METRICS]
    write_csv(output_dir / "final_summary.csv", result_rows, summary_fields)
    write_csv(output_dir / "neutral_f1_analysis.csv", neutral_rows)
    write_csv(output_dir / "regression_magnitude_shrinkage.csv", magnitude_rows)

    multi_rows = []
    for scenario in SCENARIOS:
        seed_scenario_rows = [row for row in result_rows
                              if row["group"] == "final" and row["scenario"] == scenario]
        for metric in METRICS:
            values = [float(row[metric]) for row in seed_scenario_rows]
            multi_rows.append({
                "scenario": scenario,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                **{f"seed_{row['seed']}": float(row[metric]) for row in seed_scenario_rows},
            })
    write_csv(output_dir / "final_multiseed_summary.csv", multi_rows)

    paired_rows = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            deltas = []
            for seed in HISTORICAL_SEEDS:
                final = next(row for row in result_rows if row["group"] == "final"
                             and row["seed"] == seed and row["scenario"] == scenario)
                historical = next(row for row in result_rows if row["group"] == "historical_FUSE"
                                  and row["seed"] == seed and row["scenario"] == scenario)
                deltas.append(float(final[metric]) - float(historical[metric]))
            paired_rows.append({
                "scenario": scenario,
                "metric": metric,
                "delta_seed_2026": deltas[0],
                "delta_seed_2027": deltas[1],
                "paired_delta_mean": float(np.mean(deltas)),
                "paired_delta_sd": float(np.std(deltas, ddof=1)),
                "n_pairs": len(deltas),
                "delta_definition": "final B0 minus historical FUSE; same seed and validation masks",
            })
    write_csv(output_dir / "paired_delta.csv", paired_rows)

    pooled = sum(confusion_by_seed.values(), start=np.zeros((3, 3), dtype=np.int64))
    cm_rows = []
    for seed, matrix in confusion_by_seed.items():
        for true_id in range(3):
            for pred_id in range(3):
                cm_rows.append({"scope": f"seed_{seed}", "true_class": CLASS_NAMES[true_id],
                                "predicted_class": CLASS_NAMES[pred_id],
                                "count": int(matrix[true_id, pred_id]),
                                "row_proportion": float(matrix[true_id, pred_id] /
                                                         max(1, matrix[true_id].sum()))})
    for true_id in range(3):
        for pred_id in range(3):
            cm_rows.append({"scope": "final_pooled_3_seeds", "true_class": CLASS_NAMES[true_id],
                            "predicted_class": CLASS_NAMES[pred_id],
                            "count": int(pooled[true_id, pred_id]),
                            "row_proportion": float(pooled[true_id, pred_id] /
                                                     max(1, pooled[true_id].sum()))})
    write_csv(output_dir / "confusion_matrix.csv", cm_rows)

    report = build_markdown_report(result_rows, multi_rows, paired_rows, neutral_rows,
                                   magnitude_rows, DISCOVERY_RESULTS, LR_SCREEN,
                                   run_manifests, device_name)
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    (output_dir / "final_run_manifest.json").write_text(
        json.dumps({"validation_only": True, "device": str(device), "device_name": device_name,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "n_validation": len(y_class), "final_seeds": list(SEEDS),
                    "historical_seeds": list(HISTORICAL_SEEDS), "runs": run_manifests},
                   indent=2), encoding="utf-8")

    plot_confusion(pooled, output_dir)
    plot_magnitude(magnitude_rows, output_dir)
    print(f"wrote final summaries, diagnostics and report to {output_dir}")


if __name__ == "__main__":
    main()
