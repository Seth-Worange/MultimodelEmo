"""根据最终验证与专项推理文件生成论文主结果图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from analysis.style import MODALITY_COLORS, setup_style

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT.parent / "thesis" / "figures"


def save_figure(fig: plt.Figure, name: str) -> None:
    """保留可编辑矢量图和审阅用位图。"""
    path = FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".tiff"), dpi=600)
    plt.close(fig)


def q2_views() -> None:
    report = json.loads((ROOT / "outputs/diagnostics/goal67/q2_best_valid.json").read_text(encoding="utf-8"))
    views = [("clean", "完整输入"), ("local", "局部短游程"),
             ("whole", "整段缺失"), ("interval", "连续区间")]
    scores = [report[key]["macro_f1"] for key, _ in views]
    fig, ax = plt.subplots(figsize=(6.42, 2.45), layout="constrained")
    ys = list(range(len(views) - 1, -1, -1))
    ax.hlines(ys, xmin=0.58, xmax=scores, color="#B7C8D1", linewidth=1.8)
    ax.scatter(scores, ys, s=52, color=["#0072B2", "#009E73", "#D55E00", "#CC79A7"], zorder=3)
    ax.axvline(scores[0], color="#697982", linewidth=.9, linestyle="--")
    for y, score in zip(ys, scores):
        ax.text(score + .0025, y, f"{score:.4f}", va="center", fontsize=8)
    ax.set_yticks(ys, [label for _, label in views])
    ax.set_xlim(.58, .67)
    ax.set_ylim(-.55, 3.55)
    ax.set_xlabel("验证集 macro-F1（728条，越高越好）")
    ax.grid(axis="x", color="#E6ECEF", linewidth=.7)
    ax.set_axisbelow(True)
    save_figure(fig, "fig_q2_best")


def q3_shares() -> None:
    rows = pd.read_csv(ROOT / "outputs/predictions_q3_mixed_neutral/q3_predictions.csv")
    assert len(rows) == 20
    fig, ax = plt.subplots(figsize=(6.42, 4.7), layout="constrained")
    bottom = pd.Series(0.0, index=rows.index)
    for key, label in (("text", "文本"), ("audio", "语音"), ("vision", "视觉")):
        values = rows[f"class_share_{key}"].astype(float)
        ax.barh(rows["id"].astype(str).str.zfill(2), values, left=bottom,
                height=.73, color=MODALITY_COLORS[key], label=label)
        bottom += values
    ax.invert_yaxis()
    ax.set_xlim(0, 1.07)
    ax.set_xlabel("正Shapley贡献份额")
    ax.set_ylabel("附件4样本编号")
    ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".50", ".75", "1"])
    ax.grid(axis="x", color="#E6ECEF", linewidth=.7)
    ax.set_axisbelow(True)
    for y, main in enumerate(rows["main_modality"]):
        if main == "vision":
            ax.scatter([1.035], [y], marker="s", color="#555555", s=13, clip_on=False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False)
    save_figure(fig, "fig_q3_share")


def q2_error_structure() -> None:
    rows = pd.read_csv(ROOT / "outputs/diagnostics/goal67/mixed_ensemble_valid.csv")
    labels = ["负向", "中性", "正向"]
    confusion = np.zeros((3, 3), dtype=int)
    for truth, pred in zip(rows["label_class"], rows["clean_class"]):
        confusion[int(truth), int(pred)] += 1
    recall = confusion / confusion.sum(axis=1, keepdims=True)
    bands = [rows["label_strength"].eq(0),
             rows["label_strength"].abs().between(0, .5, inclusive="right"),
             rows["label_strength"].abs().between(.5, 1, inclusive="right"),
             rows["label_strength"].abs().gt(1)]
    names = ["中性=0", "弱(0,0.5]", "中(0.5,1]", "强>1"]
    strengths = np.where(rows["clean_class"].eq(1), 0, rows["clean_strength"])
    maes = [np.abs(rows.loc[mask, "label_strength"] - strengths[mask]).mean() for mask in bands]
    fig, axes = plt.subplots(1, 2, figsize=(6.42, 2.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1.05, 1]})
    axes[0].imshow(recall, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{recall[i, j]:.2f}\n({confusion[i, j]})", ha="center", va="center",
                         fontsize=8, color="white" if recall[i, j] > .55 else "#263746")
    axes[0].set_xticks(range(3), labels)
    axes[0].set_yticks(range(3), labels)
    axes[0].set_xlabel("预测类别")
    axes[0].set_ylabel("真实类别")
    axes[0].set_title("a 逐类归一化混淆矩阵")
    bars = axes[1].barh(names[::-1], maes[::-1], color=["#CC79A7", "#009E73", "#E69F00", "#0072B2"])
    axes[1].set_xlim(0, max(maes) * 1.2)
    axes[1].set_xlabel("强度 MAE")
    axes[1].set_title("b 按真实强度分层")
    for bar, value in zip(bars, maes[::-1]):
        axes[1].text(value + .015, bar.get_y() + bar.get_height() / 2,
                     f"{value:.3f}", va="center", fontsize=8)
    save_figure(fig, "fig_q2_error_structure")


def q3_timeline() -> None:
    predictions = pd.read_csv(ROOT / "outputs/predictions_q3_mixed_neutral/q3_predictions.csv")
    windows = pd.read_csv(ROOT / "outputs/predictions_q3_mixed_neutral/q3_selected_evidence.csv")
    fig, ax = plt.subplots(figsize=(6.42, 4.65), layout="constrained")
    for i, (pred, win) in enumerate(zip(predictions.itertuples(), windows.itertuples())):
        total = float(pred.aligned_last_word_end)
        covered = float(pred.mapped_time_end)
        ax.hlines(i, 0, total, color="#C9D5DA", linewidth=5, zorder=1)
        ax.hlines(i, 0, covered, color="#8DADB9", linewidth=5, zorder=2)
        color = MODALITY_COLORS[win.modality]
        ax.hlines(i, win.start_seconds, win.end_seconds, color=color, linewidth=5, zorder=3)
    ax.invert_yaxis()
    ax.set_yticks(range(20), predictions["id"].astype(str).str.zfill(2))
    ax.set_xlim(0, max(predictions["aligned_last_word_end"]) * 1.03)
    ax.set_xlabel("视频音轨时间（秒）")
    ax.set_ylabel("附件4样本编号")
    ax.grid(axis="x", color="#E6ECEF", linewidth=.7)
    ax.set_axisbelow(True)
    for name, label in (("text", "文本证据"), ("audio", "语音证据"), ("vision", "视觉证据")):
        ax.plot([], [], color=MODALITY_COLORS[name], linewidth=5, label=label)
    ax.plot([], [], color="#C9D5DA", linewidth=5, label="未覆盖尾段")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False)
    save_figure(fig, "fig_q3_timeline")


def q2_group_uncertainty() -> None:
    report = json.loads((ROOT / "outputs/diagnostics/goal67/q2_group_bootstrap.json").read_text(encoding="utf-8"))
    views = [("clean", "完整"), ("local", "局部"), ("whole", "整段声画"), ("interval", "连续区间")]
    fig, axes = plt.subplots(1, 2, figsize=(6.42, 2.6), layout="constrained")
    y = np.arange(4)[::-1]
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for i, (key, label) in enumerate(views):
        value = report["views"][key]
        point = value["point"][1]
        low, high = value["ci95"][1]
        axes[0].errorbar(point, y[i], xerr=[[point-low], [high-point]], fmt="o", color=colors[i],
                         capsize=2.5, markersize=4, linewidth=1.1)
        if i:
            delta = value["delta_vs_clean_point"][1]
            low, high = value["delta_vs_clean_ci95"][1]
            axes[1].errorbar(delta, y[i], xerr=[[delta-low], [high-delta]], fmt="o", color=colors[i],
                             capsize=2.5, markersize=4, linewidth=1.1)
    axes[0].set_yticks(y, [name for _, name in views]); axes[0].set_xlim(.54, .71)
    axes[0].set_xlabel("macro-F1 与95%区间")
    axes[0].set_title("a  按源视频重采样")
    axes[1].set_yticks(y, [name for _, name in views]); axes[1].set_xlim(-.075,.025)
    axes[1].set_xlabel("相对完整输入的F1变化")
    axes[1].set_title("b  配对差值区间")
    axes[1].axvline(0,color="#748892",linestyle="--",linewidth=.9)
    for ax in axes:
        ax.set_ylim(-.6,3.6); ax.grid(axis="x",color="#E7ECEE",linewidth=.7); ax.set_axisbelow(True)
    save_figure(fig,"fig_q2_group_uncertainty")


def main() -> None:
    setup_style(base_fontsize=8.5)
    plt.rcParams.update({"font.family": "sans-serif", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.dpi": 600})
    q2_views()
    q3_shares()
    q2_error_structure()
    q3_timeline()
    q2_group_uncertainty()


if __name__ == "__main__":
    main()
