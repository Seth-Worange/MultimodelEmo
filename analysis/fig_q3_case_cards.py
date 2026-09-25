"""从交互卡同源数据生成论文可打印的两条典型样本卡。"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from analysis.style import MODALITY_COLORS, setup_style


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "predictions_q3_mixed_neutral"
VIDEO = (ROOT / "data" / "附件4-可解释专项视频样本与特征文件"
         / "附件4-可解释专项视频样本与特征文件" / "未对齐版本" / "videos")
FIGURES = ROOT.parent / "thesis" / "figures"


def rows(name: str) -> dict[str, dict[str, str]]:
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def frame_at(sample: str, second: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO / f"{sample}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode video {sample} at {second:.2f}s")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    setup_style(base_fontsize=8.5)
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})
    predictions, evidence = rows("q3_predictions.csv"), rows("q3_selected_evidence.csv")
    fig = plt.figure(figsize=(6.42, 5.45), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.6], hspace=.2)
    cases = [("09", "负向词义明确"), ("13", "视觉整段缺失")]
    for row_index, (sample_id, note) in enumerate(cases):
        pred, ev = predictions[sample_id], evidence[sample_id]
        start, end = float(ev["start_seconds"]), float(ev["end_seconds"])
        image = frame_at(sample_id, (start + end) / 2)
        ax_img = fig.add_subplot(grid[row_index, 0]); ax_img.imshow(image); ax_img.set_axis_off()
        ax_img.set_title(f"{chr(97+row_index)}  样本{sample_id} · {note}", loc="left", fontsize=8.5)
        ax = fig.add_subplot(grid[row_index, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        polarity = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}[pred["polarity"]]
        ax.text(.02, .94, f"预测 {polarity}  |  强度 {float(pred['sentiment_strength']):+.3f}",
                fontweight="bold", fontsize=9, color="#193C48")
        ax.text(.02, .80, "三类概率", fontsize=7.5, color="#637E88")
        keys = [("p_negative", "负", "#BD5447"), ("p_neutral", "中", "#788D99"),
                ("p_positive", "正", "#278C72")]
        for i, (key, label, color) in enumerate(keys):
            y=.72-i*.10; ax.text(.02,y,label,va="center",fontsize=7.5)
            ax.barh(y, float(pred[key])*.47, left=.10, height=.05, color=color)
            ax.text(.61,y,f"{float(pred[key]):.3f}",va="center",fontsize=7.5)
        ax.text(.02,.38,"正Shapley份额",fontsize=7.5,color="#637E88")
        left=.02
        for key in ("text","audio","vision"):
            value=float(pred[f"class_share_{key}"])
            if value:
                ax.barh(.31,value*.68,left=left,height=.065,color=MODALITY_COLORS[key])
            left+=value*.68
        shares=" / ".join(f"{m} {float(pred[f'class_share_{k}']):.2f}" for k,m in
                          (("text","文"),("audio","声"),("vision","视")))
        ax.text(.02,.22,shares,fontsize=7.2,color="#385A65")
        phrase=ev["evidence_words"].replace("\n"," ")
        ax.text(.02,.12,f"证据：{phrase}",fontsize=8,color="#975B26",fontweight="bold")
        ax.text(.02,.04,f"{start:.2f}–{end:.2f} 秒  ·  删除 {float(ev['class_margin_drop']):.2f}  ·  插入 {float(ev['insertion_margin_gain']):.2f}",
                fontsize=7.1,color="#617D88")
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGURES / f"fig_q3_case_cards.{extension}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
