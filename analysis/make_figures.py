"""生成论文全部数据插图（PDF 矢量 + PNG 预览 + 灰度预览）。

用法：
    python -m analysis.make_figures            # 全部图
    python -m analysis.make_figures --only data_profile training

每张图对应论文中的一个论证目标，图注（caption）写在 thesis/paper.tex 中；
本模块只负责把数据画成图，不在图内重复图注信息。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from analysis import loader
from analysis.style import (CLASS_COLORS, CLASS_LABELS, HALF_WIDTH_IN, MODALITY_COLORS,
                            MODALITY_LABELS, OKABE_ITO, TEXT_WIDTH_IN, VIEW_COLORS,
                            VIEW_LABELS, annotate, panel_labels, save, setup_style)

FIGDIR = loader.FIGURES


# ====================================================================== 图 2
def fig_data_profile() -> None:
    """附件2 数据画像：序列长度结构、情感强度分布、类别不均衡、文本截断比例。"""
    meta = loader.all_splits_meta()
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.6))

    ax = axes[0, 0]
    bins = np.arange(0, 56, 2)
    series = [("token 数（含 CLS/SEP）", meta["tokens"], "-"),
              ("内容 token 数", meta["content"], "--"),
              ("转写词数", meta["words"], ":")]
    for label, values, style in series:
        ax.hist(values, bins=bins, histtype="step", linewidth=1.4, linestyle=style, label=label)
    ax.set_xlabel("序列长度")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 660)                     # 预留顶部空白，避免标注压在曲线上
    ax.legend(loc="upper right", fontsize=6.8)
    annotate(ax, f"中位 token 数 {int(meta['tokens'].median())}，内容位 {int(meta['content'].median())}\n"
                 f"50 位中仅 {int(meta['content'].median())} 位承载内容",
             xy=(0.02, 0.97), fontsize=6.6)
    ax.set_title("50 位时序网格的真实信息量", fontsize=8.5)

    ax = axes[0, 1]
    edges = np.arange(-3.05, 3.06, 0.1)
    for index, (color, label) in enumerate(zip(CLASS_COLORS, CLASS_LABELS)):
        subset = meta.loc[meta["class"] == index, "strength"]
        ax.hist(subset, bins=edges, color=color, alpha=0.85, label=label)
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("情感强度 $y\\in[-3,3]$")
    ax.set_ylabel("样本数")
    ax.legend(loc="upper left", title="极性", title_fontsize=7)
    ax.set_title("情感强度与极性标签分布", fontsize=8.5)

    ax = axes[1, 0]
    splits = ["train", "valid", "test"]
    width = 0.26
    positions = np.arange(len(splits))
    for index, (color, label) in enumerate(zip(CLASS_COLORS, CLASS_LABELS)):
        counts = [int(((meta["split"] == s) & (meta["class"] == index)).sum()) for s in splits]
        total = [int((meta["split"] == s).sum()) for s in splits]
        ratio = [c / t for c, t in zip(counts, total)]
        bars = ax.bar(positions + (index - 1) * width, ratio, width, color=color, label=label)
        for bar, value in zip(bars, ratio):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(positions, [f"{s}\n(n={int((meta['split'] == s).sum())})" for s in splits])
    ax.set_ylabel("类别占比")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title(f"三划分的类别不均衡（中性占比 {meta[meta['class']==1].shape[0]/meta.shape[0]:.2f}）", fontsize=8.5)

    ax = axes[1, 1]
    trunc = meta.groupby("split")["truncated"].mean().reindex(splits)
    bars = ax.bar(splits, trunc.values * 100, color=OKABE_ITO["vermillion"], width=0.5)
    for bar, value in zip(bars, trunc.values * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=7)
    worst = 1 - (meta.loc[meta["truncated"], "tokens"] / meta.loc[meta["truncated"], "words"]).min()
    ax.set_ylabel("被截断样本比例 (%)")
    ax.set_ylim(0, max(trunc.values * 100) * 1.6)
    ax.set_title("超过 50 token 的文本被截断", fontsize=8.5)
    annotate(ax, f"共 {int(meta['truncated'].sum())} 条被截断\n最严重者丢失 {worst:.0%} 的词",
             xy=(0.03, 0.92))

    panel_labels(axes.ravel(), "abcd")
    save(fig, FIGDIR / "fig_data_profile")


# ====================================================================== 图 3
def fig_grid_layout() -> None:
    """50 位网格布局：CLS/SEP 特殊槽、内容位、填充位与子词复制。"""
    cases = [("train", 0), ("train", 3)]
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 3.4), sharex=True)
    for panel, (ax, (split, index)) in enumerate(zip(axes, cases)):
        item = loader.grid_layout_example(split, index)
        m = item["m"]
        # 子词复制位置：同一词被切成多 token 时，声画向量在相邻位上完全相同
        audio = np.asarray(loader.aligned()[split]["audio"][index])
        repeated = np.zeros(50, dtype=bool)
        for position in range(2, m - 1):
            if audio[position].shape == audio[position - 1].shape and \
                    np.array_equal(audio[position], audio[position - 1]) and \
                    np.abs(audio[position]).max() > 1e-8:
                repeated[position] = True
        rows = [("文本", item["text_content"], MODALITY_COLORS["text"]),
                ("语音", item["audio_valid"], MODALITY_COLORS["audio"]),
                ("视觉", item["vision_valid"], MODALITY_COLORS["vision"]),
                ("子词复制", repeated, OKABE_ITO["purple"])]
        ax.broken_barh([(m, 50 - m)], (-0.42, 4.0), facecolors="#F2F2F2")
        for row_index, (name, valid, color) in enumerate(rows):
            y = len(rows) - 1 - row_index
            ax.broken_barh([(i, 0.9) for i in np.where(~valid)[0]], (y - 0.32, 0.64),
                           facecolors="#E4E4E4", edgecolors="none")
            ax.broken_barh([(i, 0.9) for i in np.where(valid)[0]], (y - 0.32, 0.64),
                           facecolors=color, edgecolors="none")
        # 特殊槽覆盖着色
        ax.broken_barh([(0, 0.9), (m - 1, 0.9)], (3 - 0.34, 0.68), facecolors="#9A9A9A")
        ax.set_yticks([3, 2, 1, 0], [name for name, _, _ in rows])
        ax.set_ylim(-0.65, 3.6)
        ax.set_xlim(-0.6, 50)
        ax.set_title(f"({chr(97 + panel)}) 样本 “{item['raw_text'][:30]}…”："
                     f"{item['words']} 词 → {m} token（内容位 {int(item['text_content'].sum())}，"
                     f"子词复制 {int(repeated.sum())} 处）", fontsize=7.2, loc="left")
    axes[-1].set_xlabel("50 位时序网格的位置编号")
    handles = [Patch(facecolor="#9A9A9A", label="CLS/SEP 特殊槽（声画置零）"),
               Patch(facecolor=MODALITY_COLORS["text"], label="文本内容位"),
               Patch(facecolor=MODALITY_COLORS["audio"], label="语音有值"),
               Patch(facecolor=MODALITY_COLORS["vision"], label="视觉有值"),
               Patch(facecolor="#E4E4E4", label="零值/填充"),
               Patch(facecolor=OKABE_ITO["purple"], label="子词复制位")]
    axes[0].legend(handles=handles, loc="lower center", ncol=6, fontsize=6.2,
                   bbox_to_anchor=(0.5, 1.20))
    save(fig, FIGDIR / "fig_grid_layout")


# ====================================================================== 图 4
def fig_q1_alignment() -> None:
    """问题1 典型样本的词级三模态对齐结果。"""
    payload = loader.q1_feature("-3g5yACwYnA", "13")
    words = [str(w) for w in payload["words"]]
    times = payload["times"]
    audio = payload["audio"]
    prosody = payload["prosody"]
    audio_mask = payload["audio_mask"].astype(bool)
    face = payload["face_mask"].astype(bool)
    pose = payload["pose_mask"].astype(bool)
    nearest = payload["vision_nearest_mask"].astype(bool)
    centers = (times[:, 0] + times[:, 1]) / 2

    fig, axes = plt.subplots(4, 1, figsize=(TEXT_WIDTH_IN, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.1, 1.1, 0.8]})

    ax = axes[0]
    ax.barh(0, times[:, 1] - times[:, 0], left=times[:, 0], height=0.55,
            color=MODALITY_COLORS["text"], edgecolor="white", linewidth=0.4)
    for word, start in zip(words, times[:, 0]):
        ax.text(start, 0.34, word, rotation=42, ha="left", va="bottom", fontsize=6.0)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([])
    ax.set_ylabel("文本词区间")

    ax = axes[1]
    ax.bar(centers, audio[:, 40], width=(times[:, 1] - times[:, 0]) * 0.85,
           color=[MODALITY_COLORS["audio"] if flag else "#CCCCCC" for flag in audio_mask])
    ax.set_ylabel("语音能量\n(log-RMS)")
    annotate(ax, f"{len(words)} 个词，语速 "
                 f"{float(payload['utterance_speech_rate_wps']):.2f} 词/秒", xy=(0.02, 0.18))

    ax = axes[2]
    pitch = prosody[:, 4]
    ax.plot(centers, pitch, marker="o", color=MODALITY_COLORS["audio"], linewidth=1.2,
            label="相对音高均值 (半音)")
    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--")
    ax.set_ylabel("韵律\n(半音)")
    ax.legend(loc="upper left")

    ax = axes[3]
    for index in range(len(words)):
        ax.plot(centers[index], 1.0, "o" if face[index] else "x",
                color=MODALITY_COLORS["vision"] if face[index] else "#BBBBBB", markersize=4)
        if pose[index]:
            ax.plot(centers[index], 0.65, "s", color=OKABE_ITO["vermillion"], markersize=3.2)
        if nearest[index]:
            ax.plot(centers[index], 1.32, "^", color=OKABE_ITO["purple"], markersize=3.2)
    ax.set_ylim(0.4, 1.6)
    ax.set_yticks([0.65, 1.0, 1.32], ["姿态", "人脸", "最近帧补位"])
    ax.set_xlabel("时间 (秒)")
    ax.set_xlim(0, float(times[:, 1].max()) * 1.02)

    panel_labels(axes, "abcd", x=-0.045, y=1.02)
    save(fig, FIGDIR / "fig_q1_alignment")


# ====================================================================== 图 5
def fig_q1_coverage() -> None:
    """问题一的质量核验：视觉覆盖率分档、最近帧补位比例、强制对齐置信度。

    覆盖率用"分档计数"而非"逐样本排序曲线"表达：后者把 100 条样本画成 100 条
    互相遮蔽的折线，既看不清也无法读数；分档后每条样本都被唯一地计入某一档，
    且"完全无观测"的样本数可以直接读出。
    """
    frame = loader.manifest()
    profile = loader.defect_attachment1().set_index("group")
    near = frame["nearest_frame_fraction"].astype(float).to_numpy()
    score = frame["mean_alignment_score"].astype(float).to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0),
                             gridspec_kw={"width_ratios": [1.30, 1.0, 1.0]})

    # ---- (a) 覆盖率分档：三个特征 × 四个覆盖档（分组柱，颜色编码特征）
    ax = axes[0]
    bands = ["zero", "low", "mid", "high"]
    band_labels = ["无观测", "≤0.5", "0.5–0.9", ">0.9"]
    groups = ["人脸", "姿态", "任一视觉"]
    group_colors = [OKABE_ITO["vermillion"], MODALITY_COLORS["text"],
                    MODALITY_COLORS["vision"]]
    width = 0.26
    positions = np.arange(len(bands))
    for offset, (group, color) in enumerate(zip(groups, group_colors)):
        values = [int(profile.loc[group, band]) for band in bands]
        bars = ax.bar(positions + (offset - 1) * width, values, width, color=color,
                      label=group, edgecolor="white", linewidth=0.4)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value),
                    ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(positions, band_labels)
    ax.set_xlabel("词位覆盖率区间")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 132)
    ax.legend(loc="upper left", fontsize=6.6, ncol=3, columnspacing=0.9, handletextpad=0.4)
    annotate(ax, f"{int(profile.loc['任一视觉', 'zero'])} 条连姿态也检不出",
             xy=(0.03, 0.83), fontsize=6.4)
    ax.set_title("视觉特征覆盖率分档", fontsize=8.4)

    # ---- (b) 最近帧补位比例
    ax = axes[1]
    ax.hist(near, bins=np.arange(0, 0.50, 0.035), color=OKABE_ITO["purple"],
            edgecolor="white", linewidth=0.5)
    # 只用半高参考线，避免竖线穿过左上角的结论标注
    ax.vlines(near.mean(), 0, 13.5, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("最近帧补位的词位比例")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 27)
    annotate(ax, f"均值 {near.mean():.3f}\n{int((near > 0).sum())}/{len(near)} 条存在补位\n"
                 f"最大 {near.max():.3f}", xy=(0.03, 0.97), fontsize=6.4)
    ax.set_title("视觉观测的补位占比", fontsize=8.4)

    # ---- (c) 强制对齐置信度
    ax = axes[2]
    ax.hist(score, bins=np.arange(0, 0.92, 0.05), color=OKABE_ITO["blue"],
            edgecolor="white", linewidth=0.5)
    ax.vlines(0.3, 0, 13.5, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.0)
    ax.set_xlabel("平均强制对齐置信度")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 27)
    annotate(ax, f"低于 0.3 的 {int((score < 0.3).sum())} 条\n已列入人工复核清单\n"
                 f"中位 {np.median(score):.3f}", xy=(0.03, 0.97), fontsize=6.4)
    ax.set_title("词级强制对齐质量", fontsize=8.4)

    panel_labels(axes, "abc", y=1.04)
    save(fig, FIGDIR / "fig_q1_coverage")


# ====================================================================== 图 6
def fig_training() -> None:
    """两种融合结构的训练曲线与最佳轮次。"""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9))
    runs = [("门控基线 s2026", "availability_s2026", "-", OKABE_ITO["blue"]),
            ("门控基线 s2027", "availability_s2027", "--", OKABE_ITO["sky"]),
            ("FUSE-Net s2026", "fuse_s2026", "-", OKABE_ITO["vermillion"]),
            ("FUSE-Net s2027", "fuse_s2027", "--", OKABE_ITO["orange"])]
    ax = axes[0]
    for label, run, style, color in runs:
        path = loader.RUNS / run / "metrics.csv"
        if not path.is_file():
            continue
        curve = pd.read_csv(path)
        ax.plot(curve["epoch"], curve["val_score"], style, color=color, marker="o",
                markersize=2.6, label=label)
        best = curve.loc[curve["val_score"].idxmin()]
        ax.plot(best["epoch"], best["val_score"], marker="*", markersize=8,
                color=color, markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证集选型分数（越低越好）")
    ax.set_ylim(0.275, 0.470)
    ax.legend(loc="upper right", fontsize=6.6)
    ax.set_title("验证分数曲线与最佳轮（星号）", fontsize=8.5)
    annotate(ax, "最佳轮均落在第 2–6 轮", xy=(0.03, 0.97), fontsize=6.8)

    ax = axes[1]
    curve = pd.read_csv(loader.RUNS / "fuse_s2026" / "metrics.csv")
    for view in ("clean", "local", "whole", "interval"):
        ax.plot(curve["epoch"], curve[f"{view}_macro_f1"], marker="o", markersize=2.6,
                color=VIEW_COLORS[view], label=VIEW_LABELS[view])
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证集 macro-F1")
    ax.set_ylim(0.500, 0.705)          # 顶部留白给图例，避免图例被坐标框裁切
    ax.legend(loc="upper center", ncol=4, fontsize=6.4, columnspacing=1.2,
              handletextpad=0.4)
    ax.set_title("FUSE-Net 四个缺失视图的 macro-F1", fontsize=8.5)

    panel_labels(axes, "ab", y=1.05)
    save(fig, FIGDIR / "fig_training")


# ====================================================================== 图 7
def fig_robustness() -> None:
    """缺失类型 / 缺失率 / 缺失位置的性能影响规律。"""
    robust = loader.robustness("fuse_availability")
    base = float(robust.loc[robust["pattern"] == "none", "macro_f1"].iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 3.0),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.25]})

    # (a) 整段模态缺失（横向条形，避免中文标签重叠）
    ax = axes[0]
    whole = robust[robust["pattern"] == "whole"].copy()
    order = [("audio", "仅语音"), ("vision", "仅视觉"), ("audio+vision", "语音+视觉"),
             ("text", "文本")]
    labels, values, colors = ["完整输入"], [base], ["#666666"]
    for key, label in order:
        row = whole[whole["missing_modalities"] == key]
        if len(row):
            labels.append(label)
            values.append(float(row["macro_f1"].iloc[0]))
            colors.append(MODALITY_COLORS.get(key, OKABE_ITO["vermillion"]))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.5)
    for position, value in zip(y, values):
        ax.text(value + 0.012, position, f"{value:.3f}", va="center", fontsize=6.4)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, 0.88)
    ax.set_xlabel("macro-F1")
    annotate(ax, f"文本缺失即崩塌\n（{values[-1] - base:+.3f}）",
             xy=(0.42, 0.19), fontsize=6.6, va="center", ha="left")
    ax.set_title("整段模态缺失", fontsize=8.5)

    # (b) 连续区间：缺失率 × 位置热力图（顺序色标，深色=性能好；数值同时写在格内）
    # 色标已由格内数值完整给出，故不再挂 colorbar——它会把坐标区挤窄，
    # 导致标题比坐标区还宽而与面板标记相撞。
    ax = axes[1]
    interval = robust[(robust["pattern"] == "interval")
                      & (robust["missing_modalities"] == "audio+vision")]
    pivot = interval.pivot_table(index="missing_rate", columns="location", values="macro_f1")
    pivot = pivot[["start", "middle", "end", "random"]]
    low, high = float(np.nanmin(pivot.values)), float(np.nanmax(pivot.values))
    pad = max(0.002, (high - low) * 0.15)
    image = ax.imshow(pivot.values, cmap="Blues", aspect="auto",
                      vmin=low - pad, vmax=high + pad)
    ax.set_xticks(range(pivot.shape[1]), ["起始", "中段", "末尾", "随机"])
    ax.set_yticks(range(pivot.shape[0]), [f"{r:.0%}" for r in pivot.index])
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.values[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=6.3,
                    color="white" if value > (low + high) / 2 else "#222222")
    ax.set_xlabel("缺失位置")
    ax.set_ylabel("缺失长度占有效序列比例")
    ax.set_title("语音+视觉连续区间缺失", fontsize=8.5)

    # (c) 局部短游程：两个模型的率扫描对比
    ax = axes[2]
    series = []
    new = robust[(robust["pattern"] == "local") & (robust["location"] == "random")]
    series.append(("FUSE-Net", new, OKABE_ITO["vermillion"], "-", "o"))
    old_path = loader.RUNS / "availability" / "robustness.csv"
    if old_path.is_file():
        old = pd.read_csv(old_path)
        old = old[(old["missing_modalities"] == "audio+vision(local)")
                  & (old["location"] == "random")]
        series.append(("门控基线", old, OKABE_ITO["blue"], "--", "s"))
    for label, frame, color, style, marker in series:
        ax.plot(frame["missing_rate"], frame["macro_f1"], style, color=color, marker=marker,
                markersize=3.4, label=label)
    ax.set_xlabel("局部缺失率")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.603, 0.640)          # 底部留白给图例，避免图例压住曲线
    ax.legend(loc="lower left", fontsize=6.6)
    annotate(ax, "率从 10% 升至 60%，\nmacro-F1 变化 ≤ 0.015", xy=(0.02, 0.97), fontsize=6.6)
    ax.set_title("语音+视觉零散短游程", fontsize=8.5)

    panel_labels(axes, "abc", y=1.04, x=-0.03)
    save(fig, FIGDIR / "fig_robustness")


# ====================================================================== 图 8
def fig_modality_ablation() -> None:
    """单模态基线与三模态模型的对比：声画的增益落在强度回归而非极性分类。

    刻意**不使用从非零基线截断的柱形图**（那会夸大差异）。此处统一改成"相对纯文本
    的性能变化"，零基线是真实语义（0 = 与纯文本持平），分类与回归两个面板都把
    "正向=更好"作为统一方向，从而可以直接比较。
    """
    entries = [
        ("纯文本", loader.deep_experiment("text_only"), OKABE_ITO["grey"]),
        ("纯语音", loader.deep_experiment("audio_only"), MODALITY_COLORS["audio"]),
        ("纯视觉", loader.deep_experiment("vision_only"), MODALITY_COLORS["vision"]),
        ("三模态\n门控基线", loader.evaluate_result("A 门控基线（附件3同分布增强）").get("test", {}),
         OKABE_ITO["blue"]),
        ("三模态\nFUSE-Net", loader.evaluate_result("E FUSE-Net（多尺度缺失）").get("test", {}),
         OKABE_ITO["vermillion"]),
    ]
    names, f1, mae, colors = [], [], [], []
    for label, result, color in entries:
        if not result or "clean" not in result:
            continue
        names.append(label)
        f1.append(result["clean"]["macro_f1"])
        mae.append(result["clean"]["mae"])
        colors.append(color)
    base_f1, base_mae = f1[0], mae[0]
    delta_f1 = [value - base_f1 for value in f1]
    delta_mae = [base_mae - value for value in mae]        # 正值 = MAE 下降 = 更好

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9))
    for ax, values, absolute, ylabel, title in (
            (axes[0], delta_f1, f1, "相对纯文本的 macro-F1 变化\n（正值 = 更好）",
             "极性分类：声画未带来增益"),
            (axes[1], delta_mae, mae, "相对纯文本的 MAE 改善量\n（正值 = 更好）",
             "强度回归：声画显著更优")):
        positions = np.arange(len(names))
        ax.axhline(0, color="#555555", linewidth=0.8)
        bars = ax.bar(positions, values, color=colors, width=0.62)
        for bar, value, raw in zip(bars, values, absolute):
            offset = 0.012 if value >= 0 else -0.012
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                    f"{value:+.3f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=6.6)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value + (0.055 if value >= 0 else -0.055),
                    f"({raw:.3f})", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=5.8, color="#666666")
        ax.set_xticks(positions, names)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.set_ylabel(ylabel, fontsize=7.4)
        ax.set_title(title, fontsize=8.5)
        span = max(abs(min(values)), abs(max(values)))
        ax.set_ylim(-span * 1.45, span * 1.45)
        annotate(ax, f"纯文本基准 {absolute[0]:.3f}", xy=(0.03, 0.05), fontsize=6.4)
    panel_labels(axes, "ab", y=1.05)
    save(fig, FIGDIR / "fig_modality_ablation")


# ====================================================================== 图 9
def fig_attribution() -> None:
    """模态归因：门控塌陷与 FUSE 的修复，以及问题3 的 Shapley 贡献。"""
    attribution = loader.read_csv(Path(__file__).resolve().parent / "results"
                                  / "modality_attribution.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    ax = axes[0]
    models = list(attribution["model"])
    left = np.zeros(len(models))
    for modality in ("text", "audio", "vision"):
        values = attribution[modality].to_numpy()
        ax.barh(models, values, left=left, color=MODALITY_COLORS[modality], height=0.42,
                label=MODALITY_LABELS[modality], hatch=None)
        for index, (value, base) in enumerate(zip(values, left)):
            if value > 0.05:
                ax.text(base + value / 2, index, f"{value:.2f}", ha="center", va="center",
                        fontsize=6.8, color="white")
        left += values
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("融合权重份额")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.24), fontsize=6.8)
    audio = attribution["audio"].to_numpy()
    annotate(ax, f"语音份额 {audio[0]:.3f} → {audio[1]:.3f}\n"
                 f"视觉份额 {attribution['vision'].to_numpy()[0]:.3f} → "
                 f"{attribution['vision'].to_numpy()[1]:.3f}",
             xy=(0.03, 0.50), fontsize=6.8, va="center")   # 两条柱之间的空白
    ax.set_title("融合层的模态归因（验证集）", fontsize=8.5)

    ax = axes[1]
    predictions = loader.q3_predictions()
    data = [predictions[f"shapley_class_{m}"].to_numpy() for m in ("text", "audio", "vision")]
    parts = ax.violinplot(data, showextrema=False, widths=0.75)
    for body, modality in zip(parts["bodies"], ("text", "audio", "vision")):
        body.set_facecolor(MODALITY_COLORS[modality])
        body.set_edgecolor(MODALITY_COLORS[modality])
        body.set_alpha(0.35)
    rng = np.random.default_rng(7)
    for index, (values, modality) in enumerate(zip(data, ("text", "audio", "vision")), start=1):
        ax.scatter(index + rng.uniform(-0.1, 0.1, len(values)), values, s=11,
                   color=MODALITY_COLORS[modality], alpha=0.85, linewidths=0)
        ax.hlines(np.mean(values), index - 0.26, index + 0.26, color="#333333", linewidth=1.4)
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks([1, 2, 3], ["文本", "语音", "视觉"])
    ax.set_ylabel("Shapley 贡献（类别 margin）")
    low = min(float(np.min(v)) for v in data)
    high = max(float(np.max(v)) for v in data)
    ax.set_ylim(low - 0.6, high + 5.2)     # 顶部留白放结论标注，避免压住小提琴
    annotate(ax, "语音/视觉均值为负\n边际贡献接近噪声", xy=(0.97, 0.97), fontsize=6.6,
             ha="right")
    ax.set_title(f"附件4 样本的模态 Shapley 贡献（n={len(predictions)}）", fontsize=8.5)

    panel_labels(axes, "ab", y=1.05)
    save(fig, FIGDIR / "fig_attribution")


# ====================================================================== 图 10
def fig_q3_evidence() -> None:
    """问题3 的证据定位：窗口遮蔽的 margin 下降与逐样本模态份额。"""
    evidence = loader.q3_evidence()
    predictions = loader.q3_predictions()
    # 附件4 的样本编号在特征文件里是 1–20（不带前导零），id 必须按实际取值匹配。
    sample = str(int(predictions["id"].iloc[0]))
    subset = evidence[evidence["id"].astype(int) == int(sample)].sort_values(
        "class_margin_drop", ascending=False).head(12)
    if subset.empty:
        raise ValueError(f"附件4 样本 {sample} 没有证据窗口，请检查 q3_evidence_windows.csv")
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    y = np.arange(len(subset))[::-1]
    colors = [MODALITY_COLORS.get(m, "#888888") for m in subset["modality"]]
    ax.barh(y, subset["class_margin_drop"], color=colors, height=0.62)
    labels = []
    for _, row in subset.iterrows():
        word = str(row["evidence_words"]) if isinstance(row["evidence_words"], str) else ""
        word = (word[:22] + "…") if len(word) > 22 else word
        labels.append(word or f"位置 {int(row['position_start'])}–{int(row['position_end_exclusive'])}")
    ax.set_yticks(y, labels, fontsize=6.4)
    ax.set_xlim(0, float(subset["class_margin_drop"].max()) * 1.22)
    ax.set_xlabel("遮蔽该窗口后类别 margin 的下降量")
    ax.legend(handles=[Patch(facecolor=MODALITY_COLORS[m], label=MODALITY_LABELS[m])
                       for m in ("text", "audio", "vision")], loc="lower right", fontsize=6.6)
    ax.set_title(f"典型样本 {sample} 的证据窗口", fontsize=8.2)

    ax = axes[1]
    shares = predictions[["class_share_text", "class_share_audio", "class_share_vision"]].to_numpy()
    left = np.zeros(len(shares))
    order = np.argsort(-shares[:, 0])
    for index, modality in enumerate(("text", "audio", "vision")):
        values = shares[order, index]
        ax.barh(np.arange(len(shares)), values, left=left, height=0.7,
                color=MODALITY_COLORS[modality], label=MODALITY_LABELS[modality])
        left += values
    ax.set_yticks(np.arange(len(shares)), [str(i) for i in predictions["id"].to_numpy()[order]],
                  fontsize=6.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, len(shares) - 0.4)
    ax.set_xlabel("模态作用份额")
    ax.set_ylabel("附件4 样本编号")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16), fontsize=6.6)
    ax.set_title("逐样本模态作用份额", fontsize=8.2)

    panel_labels(axes, "ab", y=1.04)
    save(fig, FIGDIR / "fig_q3_evidence")


# ====================================================================== 图 11
def fig_confusion() -> None:
    """测试集混淆矩阵：门控基线与 FUSE-Net。"""
    sources = [("（a）门控基线", loader.RUNS / "availability" / "validation_per_sample.csv"),
               ("（b）FUSE-Net", loader.OUTPUTS / "experiments" / "fuseE_test_per_sample.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.0))
    for ax, (name, path) in zip(axes, sources):
        frame = loader.read_csv(path)
        truth = frame["label_class"].to_numpy()
        guess = frame["clean_class"].to_numpy()
        matrix = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = int(((truth == i) & (guess == j)).sum())
        row_sum = matrix.sum(1, keepdims=True)
        normalized = matrix / np.maximum(row_sum, 1)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.0%}", ha="center", va="center",
                        fontsize=6.6, color="white" if normalized[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(3), CLASS_LABELS)
        ax.set_yticks(range(3), CLASS_LABELS)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        recalls = np.diag(normalized)
        f1 = []
        for c in range(3):
            tp = matrix[c, c]
            precision = tp / max(1, matrix[:, c].sum())
            recall = tp / max(1, matrix[c].sum())
            f1.append(2 * precision * recall / max(1e-9, precision + recall))
        ax.set_title(f"{name} 宏平均 F1 = {np.mean(f1):.3f}\n"
                     f"中性召回 {recalls[1]:.2f}", fontsize=8.0)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="按真实类归一化")
    save(fig, FIGDIR / "fig_confusion")


# ====================================================================== 图 12
def fig_fuse_regularization() -> None:
    """FUSE-Net 六个正则分量的收敛过程，佐证分解与重建目标确实被优化。"""
    curve = loader.reg_curve("fuse_s2026")
    naming = {
        "reg_contrast": "对比分离 $\\mathcal{L}_{con}$",
        "reg_info": "信息增益 $\\mathcal{L}_{info}$",
        "reg_dual": "对偶一致 $\\mathcal{L}_{dual}$",
        "reg_mrc": "变分重建 $\\mathcal{L}_{mrc}$",
        "reg_kl": "重建 KL $\\mathcal{L}_{KL}$",
        "reg_cross": "交叉重建 $\\mathcal{L}_{cross}$",
    }
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.82, 3.0))
    styles = ["-", "--", "-.", ":", "-", "--"]
    markers = ["o", "s", "^", "v", "D", "P"]
    columns = [c for c in curve.columns if c != "epoch"]
    for column, style, marker in zip(columns, styles, markers):
        ax.plot(curve["epoch"], curve[column], style, marker=marker, markersize=2.6,
                label=naming.get(column, column.replace("reg_", "")))
    ax.set_yscale("log")
    ax.set_ylim(0.002, 6.0)               # 顶部留白，图例不再压住 contrast 曲线
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("正则分量（对数刻度）")
    ax.legend(ncol=3, loc="upper center", fontsize=6.4, columnspacing=1.4, handletextpad=0.5)
    ax.set_title("FUSE-Net 正则项收敛曲线（种子 2026）", fontsize=8.5)
    save(fig, FIGDIR / "fig_fuse_regularization")


# ====================================================================== 图 13
def fig_grid_evidence() -> None:
    """50 位对齐网格"结构反演"的三条定量证据（支撑问题二的核心创新点）。

    (a) 音频非零位是否恰好等于文本内容位——若成立则"零值即缺失"的掩码语义可用；
    (b) 词数 W 与 token 数 m 的关系 m = W + 2 + 子词复制数——若成立则索引层复制规则可验证；
    (c) 逐位置的"有值样本比例"——位置 0 与各样本的 m−1 处应恒为空，尾部应整段为零。
    """
    evidence = loader.grid_evidence()
    positions = loader.grid_position_profile()
    total = int(positions["n_samples"].iloc[0])

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.30, wspace=0.26)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, :])]

    # ---- (a) 零值语义核验
    ax = axes[0]
    rng = np.random.default_rng(20260924)
    jitter = rng.uniform(-0.28, 0.28, len(evidence))
    match = evidence["audio_zero_eq_content"] == 1
    ax.scatter(evidence.loc[match, "content"] + jitter[match], evidence.loc[match, "audio_valid"],
               s=4, color=MODALITY_COLORS["audio"], alpha=0.45, linewidths=0,
               label=f"一致（{int(match.sum())}）")
    ax.scatter(evidence.loc[~match, "content"] + jitter[~match], evidence.loc[~match, "audio_valid"],
               s=16, color=OKABE_ITO["vermillion"], marker="x", linewidths=0.8,
               label=f"不一致（{int((~match).sum())}）")
    limit = max(evidence["content"].max(), evidence["audio_valid"].max()) + 2
    ax.plot([0, limit], [0, limit], color="#666666", linestyle="--", linewidth=0.8,
            label="$y=x$")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("文本内容位数")
    ax.set_ylabel("语音非零位数")
    ax.legend(loc="upper left", fontsize=6.2)
    annotate(ax, f"对角线命中率 {match.mean():.2%}", xy=(0.97, 0.05), fontsize=6.4,
             ha="right", va="bottom")
    ax.set_title("零值语义核验", fontsize=8.2)

    # ---- (b) 索引层子词复制规则（截断样本单独说明，避免坐标被拉到 300 词而压扁点云）
    ax = axes[1]
    shown = evidence["words"] <= 52
    repeats = evidence.loc[shown, "expected_repeat"].to_numpy()
    buckets = [(0, 0, "无复制"), (1, 2, "1–2 处"), (3, 99, "≥3 处")]
    colors = [OKABE_ITO["grey"], OKABE_ITO["sky"], OKABE_ITO["purple"]]
    for (low, high, label), color in zip(buckets, colors):
        mask = (repeats >= low) & (repeats <= high)
        ax.scatter(evidence.loc[shown].loc[mask, "words"], evidence.loc[shown].loc[mask, "tokens"],
                   s=5, color=color, alpha=0.6, linewidths=0, label=f"{label}（{int(mask.sum())}）")
    line = np.arange(0, 54)
    ax.plot(line, line + 2, color="#333333", linestyle="-", linewidth=1.0,
            label="内容位下界 $m=W+2$")
    ax.set_xlim(0, 53)
    ax.set_ylim(0, 53)
    exact = evidence["repeated_audio"] == evidence["expected_repeat"]
    ax.legend(loc="lower right", fontsize=6.2)
    annotate(ax, f"复制数 $=m-2-W$：{exact.mean():.1%} 命中", xy=(0.10, 0.06),
             fontsize=6.2, va="bottom")
    ax.set_xlabel("转写词数 $W$")
    ax.set_ylabel("token 数 $m$")
    ax.set_title("索引层复制规则核验", fontsize=8.2)

    # ---- (c) 逐位置有值比例
    ax = axes[2]
    for name, color, style in (("text", MODALITY_COLORS["text"], "-"),
                               ("audio", MODALITY_COLORS["audio"], "--"),
                               ("vision", MODALITY_COLORS["vision"], ":")):
        label = {"text": "文本内容位", "audio": "语音有值", "vision": "视觉有值"}[name]
        ax.plot(positions["position"], positions[f"{name}_rate"], style, color=color,
                linewidth=1.3, label=label)
    ax.axvspan(-0.5, 0.5, color="#9A9A9A", alpha=0.25)
    ax.text(2.5, 0.58, "位置 0 恒为 CLS 槽（声画置零）", fontsize=6.4, ha="left",
            va="center", color="#444444")
    reach = int((positions["audio_rate"] > 0).sum())
    ax.axvline(reach - 0.5, color="#666666", linestyle="-.", linewidth=0.8)
    ax.text(reach - 1.2, 0.40, f"位置 ≥{reach}\n全部样本\n均为零填充", fontsize=6.4,
            ha="right", va="center", color="#444444")
    ax.set_xlabel("50 位时序网格的位置编号")
    ax.set_ylabel("有值样本比例")
    ax.set_xlim(-0.8, 49.8)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=6.6)
    ax.set_title(f"逐位置有效频率（$n={total}$）：位置 0 与末端为特殊槽，尾部为整段零填充",
                 fontsize=8.2)

    panel_labels(axes, "abc", y=1.05)
    save(fig, FIGDIR / "fig_grid_evidence")


# ====================================================================== 图 14
def fig_data_defects() -> None:
    """四个附件的缺陷画像：附件1 容器碎片化、附件2 稀疏与截断、附件3 额外缺失、附件4 通道失效。"""
    facts = loader.diagnostics_facts()
    meta = loader.video_meta()
    evidence = loader.grid_evidence()
    defect3 = loader.defect_attachment3()
    defect4 = loader.defect_attachment4()

    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 5.2))

    # ---- (a) 附件1 容器参数碎片化
    ax = axes[0, 0]
    combo = (meta.assign(tag=meta["width"].astype(str) + "×" + meta["height"].astype(str))
             .groupby("tag").size().sort_values(ascending=False))
    y = np.arange(len(combo))[::-1]
    ax.barh(y, combo.values, height=0.6, color=OKABE_ITO["sky"])
    for position, value in zip(y, combo.values):
        ax.text(value + 0.8, position, str(value), va="center", fontsize=6.2)
    ax.set_yticks(y, [f"{tag} / {rate:g}fps" for tag, rate in zip(
        combo.index, [meta.loc[meta["width"].astype(str) + "×" + meta["height"].astype(str) == tag,
                           "fps"].mode().iloc[0] for tag in combo.index])], fontsize=6.2)
    ax.set_xlim(0, combo.max() * 1.25)
    ax.set_xlabel("视频条数")
    ax.set_title("附件1 容器碎片化", fontsize=8.2)
    annotate(ax, f"{facts['a1_resolutions']} 种分辨率 × {facts['a1_frame_rates']} 种帧率\n"
                 f"时长 {facts['a1_seconds_min']:.1f}–{facts['a1_seconds_max']:.1f} s"
                 f"（相差 {facts['a1_seconds_ratio']:.0f} 倍）",
             xy=(0.42, 0.34), fontsize=6.4, va="center")

    # ---- (b) 附件2 稀疏度与截断
    ax = axes[0, 1]
    ax.hist(evidence["content"], bins=np.arange(0, 52, 2), color=MODALITY_COLORS["text"],
            edgecolor="white", linewidth=0.5, label="有效内容位")
    ax.hist(evidence.loc[evidence["truncated"] == 1, "content"], bins=np.arange(0, 52, 2),
            color=OKABE_ITO["vermillion"], edgecolor="white", linewidth=0.5, label="其中被截断样本")
    ax.set_xlabel("50 位网格中的有效位数")
    ax.set_ylabel("样本数")
    ax.set_ylim(0, 520)                    # 顶部留白，标注不与图例/柱顶相撞
    ax.legend(loc="upper left", fontsize=6.4)
    annotate(ax, f"中位 {facts['a2_content_median']:.0f}/50；截断 {facts['a2_truncated']} 条\n"
                 f"最严重丢失 {facts['a2_truncated_worst_loss']:.0%} 的词",
             xy=(0.97, 0.95), fontsize=6.4, ha="right")
    ax.set_title("附件2 稀疏网格与超长截断", fontsize=8.2)

    # ---- (c) 附件3 额外缺失分布
    ax = axes[1, 0]
    order = np.argsort(-defect3["extra_missing_rate"].to_numpy())
    rate = defect3["extra_missing_rate"].to_numpy()[order] * 100
    ax.bar(np.arange(len(rate)), rate, color=OKABE_ITO["orange"], width=0.7)
    ax.axhline(facts["a3_extra_missing_share_pooled"] * 100, color="#333333",
               linestyle="--", linewidth=0.9)
    ax.set_xlabel("附件3 样本（按额外缺失率降序）")
    ax.set_ylabel("额外缺失位占比 (%)")
    ax.set_ylim(0, 82)                     # 顶部留白，标注不压住最高的几根柱
    annotate(ax, f"汇总 {facts['a3_extra_missing_share_pooled']:.1%}；"
                 f"逐样本 0–{facts['a3_extra_missing_rate_max']:.0%}\n"
                 f"语音与视觉缺失位置完全相同 "
                 f"{facts['a3_audio_vision_same_count']}/{facts['a3_samples']}",
             xy=(0.03, 0.97), fontsize=6.4)
    ax.set_title("附件3 额外缺失的形态", fontsize=8.2)

    # ---- (d) 附件4 逐样本视觉缺失（语音全程不缺）
    ax = axes[1, 1]
    missing = (defect4["vision_missing"] / defect4["content"]).to_numpy() * 100
    dead = [str(v) for v in facts["a4_vision_allzero"]]
    colors = [OKABE_ITO["vermillion"] if str(name) in dead else OKABE_ITO["grey"]
              for name in defect4["name"]]
    bars = ax.bar(np.arange(len(missing)), missing, color=colors, width=0.72)
    for bar, value in zip(bars, missing):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                    ha="center", va="bottom", fontsize=6.0)
    ax.set_xticks(np.arange(len(missing)), [str(v) for v in defect4["name"]], fontsize=5.6)
    ax.set_xlabel("附件4 样本编号")
    ax.set_ylabel("视觉缺失位占内容位比例 (%)")
    ax.set_ylim(0, 122)                    # 顶部留白放标注
    annotate(ax, f"仅样本 {'、'.join(dead)} 的视觉整段失效\n"
                 f"（{int(defect4.loc[defect4['name'].astype(str).isin(dead), 'vision_missing'].sum())}"
                 f" 位全零）；语音全程无缺失", xy=(0.03, 0.97), fontsize=6.4)
    ax.set_title(f"附件4 通道有效性（n={facts['a4_samples']}）", fontsize=8.2)

    panel_labels(axes.ravel(), "abcd", y=1.06)
    save(fig, FIGDIR / "fig_data_defects")


FIGURES_MAP = {
    "data_profile": fig_data_profile,
    "grid_layout": fig_grid_layout,
    "grid_evidence": fig_grid_evidence,
    "data_defects": fig_data_defects,
    "q1_alignment": fig_q1_alignment,
    "q1_coverage": fig_q1_coverage,
    "training": fig_training,
    "robustness": fig_robustness,
    "modality_ablation": fig_modality_ablation,
    "attribution": fig_attribution,
    "q3_evidence": fig_q3_evidence,
    "confusion": fig_confusion,
    "fuse_regularization": fig_fuse_regularization,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES_MAP), default=None)
    parser.add_argument("--lang", default="zh")
    args = parser.parse_args()
    font = setup_style(lang=args.lang)
    print(f"中文字体：{font or '未找到（将回退为拉丁字体）'}")
    targets = args.only or sorted(FIGURES_MAP)
    for name in targets:
        FIGURES_MAP[name]()
        print(f"  [{name}] 完成")
    print(f"共 {len(targets)} 张图输出到 {FIGDIR}")


if __name__ == "__main__":
    main()
