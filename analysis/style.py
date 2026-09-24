"""出版级绘图样式：中文支持 + 色盲安全配色 + 按最终尺寸出图。

设计要点（遵循科研配图规范）：
1. figsize 直接设为论文中的实际尺寸（研赛 A4 正文宽 165 mm ≈ 6.50 in），
   导出后不再在 LaTeX 里缩放，保证字号是真实字号。
2. 字号在最终尺寸下 7–9 pt，最小不低于 6 pt。
3. 配色使用 Okabe-Ito 色盲安全方案，并对同一图中的类别做冗余编码
   （颜色 + 线型/marker/填充纹理），灰度打印仍可区分。
4. 中文优先 Microsoft YaHei / SimHei；负号使用 ASCII 连字符避免方框。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------- 画布尺寸
TEXT_WIDTH_IN = 6.42      # 正文宽 165 mm，留 2 mm 余量
HALF_WIDTH_IN = 3.10      # 双栏并排时的单栏宽
CM = 1 / 2.54

# ---------------------------------------------------------------- 配色方案
# Okabe-Ito：对红绿色盲友好，且灰度下明度可分。
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}
# 模态统一配色：全文同一模态同色（文本=蓝、语音=橙、视觉=绿）
MODALITY_COLORS = {"text": OKABE_ITO["blue"], "audio": OKABE_ITO["orange"],
                   "vision": OKABE_ITO["green"]}
MODALITY_LABELS = {"text": "文本", "audio": "语音", "vision": "视觉"}
# 缺失视图配色
VIEW_COLORS = {"clean": OKABE_ITO["blue"], "local": OKABE_ITO["green"],
               "whole": OKABE_ITO["orange"], "interval": OKABE_ITO["vermillion"]}
VIEW_LABELS = {"clean": "完整输入", "local": "局部短游程", "whole": "整段缺失",
               "interval": "连续区间"}
# 类别配色（负/中/正）
CLASS_COLORS = ["#0072B2", "#999999", "#D55E00"]
CLASS_LABELS = ["负向", "中性", "正向"]


def _pick_cjk_font() -> str | None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun"):
        if name in available:
            return name
    return None


def setup_style(lang: str = "zh", base_fontsize: float = 8.0, dpi: int = 300) -> str | None:
    """配置全局绘图样式，返回选中的中文字体名（None 表示未找到）。"""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": dpi,
        "figure.constrained_layout.use": True,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 0.5,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 0.5,
        "ytick.labelsize": base_fontsize - 0.5,
        "legend.fontsize": base_fontsize - 0.5,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=[OKABE_ITO["blue"], OKABE_ITO["orange"],
                                             OKABE_ITO["green"], OKABE_ITO["vermillion"],
                                             OKABE_ITO["purple"], OKABE_ITO["sky"]]),
        "axes.unicode_minus": False,   # 负号用 ASCII，避免中文字体缺字变方框
        "pdf.fonttype": 42,            # TrueType 嵌入，PDF 文字可复制
        "ps.fonttype": 42,
    })
    cjk = _pick_cjk_font() if lang == "zh" else None
    if cjk:
        plt.rcParams["font.sans-serif"] = [cjk, "Arial", "DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
    return cjk


def save(fig, path: str | Path, *, png_preview: bool = True,
         grayscale: bool = True) -> list[Path]:
    """按最终尺寸导出 PDF（矢量，用于投稿）+ PNG 预览 + 灰度预览。

    灰度预览用于自检"仅靠颜色区分"的问题：把预览 PNG 转灰后若类别仍可分辨，
    说明配色通过了色觉障碍/黑白打印检查。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf)
    written = [pdf]
    if png_preview:
        preview = path.with_suffix(".png")
        fig.savefig(preview, dpi=300)
        written.append(preview)
        if grayscale:
            try:
                from PIL import Image

                gray = path.with_name(path.stem + "_gray.png")
                with Image.open(preview) as image:
                    image.convert("L").save(gray)
                written.append(gray)
            except Exception:  # Pillow 缺失时不阻断出图
                pass
    plt.close(fig)
    return written


def panel_labels(axes, labels: str = "abcdef", *, style: str = "paren",
                 x: float = -0.02, y: float = 1.06) -> None:
    """用 figure 坐标统一打 (a)(b)(c) 面板标记，自动横竖对齐。"""
    fmt = ("({})" if style == "paren" else "{}")
    for ax, label in zip(axes, labels):
        ax.text(x, y, fmt.format(label), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom", ha="left")


def annotate(ax, text: str, *, xy=(0.02, 0.96), color="#333333", fontsize=7.0, **kw):
    """在坐标轴内写结论性标注（关键结论标在图上）。

    默认左上角对齐；调用方可用 `ha` / `va` 覆盖（原先的固定关键字会造成重复传参错误）。
    """
    options = {"ha": "left", "va": "top"}
    options.update(kw)
    ax.text(*xy, text, transform=ax.transAxes, fontsize=fontsize, color=color, **options)
