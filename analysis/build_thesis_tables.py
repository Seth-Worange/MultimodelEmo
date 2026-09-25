"""从已保存的专项预测生成论文中的逐样本表。"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "predictions_q3_mixed_neutral"
TARGET = ROOT.parent / "thesis" / "tables"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tex_escape(value: str) -> str:
    table = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
    return "".join(table.get(char, char) for char in value)


def write_predictions(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{longtable}{rllrrrrl}",
        r"\caption{附件4全部20条无标签样本的模型输出。份额是当前预测类别的正Shapley贡献；局部表示转写只有部分进入50位网格。}\label{tab:q3all}\\",
        r"\toprule",
        r"编号 & 极性 & 强度 & 文本份额 & 语音份额 & 视觉份额 & 映射覆盖率 & 主要模态\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"编号 & 极性 & 强度 & 文本份额 & 语音份额 & 视觉份额 & 映射覆盖率 & 主要模态\\",
        r"\midrule\endhead",
    ]
    labels = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}
    modality = {"text": "文本", "audio": "语音", "vision": "视觉"}
    for row in rows:
        values = [
            row["id"], labels[row["polarity"]], f'{float(row["sentiment_strength"]):+.3f}',
            f'{float(row["class_share_text"]):.3f}', f'{float(row["class_share_audio"]):.3f}',
            f'{float(row["class_share_vision"]):.3f}',
            f'{100 * float(row["aligned_word_coverage"]):.1f}\\%', modality[row["main_modality"]],
        ]
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (TARGET / "tab_q3_all_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_evidence(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{longtable}{rlp{5.2cm}rrr}",
        r"\caption{附件4全部20条样本所选局部证据。删除、插入和联合值均针对固定预测类别的logit margin，单位为margin；时间来自词级对齐。}\label{tab:q3evidenceall}\\",
        r"\toprule",
        r"编号 & 模态 & 证据词组及时间（秒） & 删除下降 & 插入增益 & 联合分数\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"编号 & 模态 & 证据词组及时间（秒） & 删除下降 & 插入增益 & 联合分数\\",
        r"\midrule\endhead",
    ]
    modality = {"text": "文本", "audio": "语音", "vision": "视觉"}
    for row in rows:
        phrase = tex_escape(row["evidence_words"])
        time = f'({float(row["start_seconds"]):.2f}--{float(row["end_seconds"]):.2f})'
        values = [
            row["id"], modality[row["modality"]], f"{phrase} {time}",
            f'{float(row["class_margin_drop"]):.3f}',
            f'{float(row["insertion_margin_gain"]):.3f}',
            f'{float(row["joint_evidence_score"]):.3f}',
        ]
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (TARGET / "tab_q3_evidence_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_q2_special(rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{longtable}{rllrrrr}",
        r"\caption{附件3全部30条无标签样本的最终预测。三列概率依次为负向、中性、正向；不据此计算专项准确率。}\label{tab:q2specialall}\\",
        r"\toprule",
        r"编号 & 极性 & 强度 & $p_-$ & $p_0$ & $p_+$ & 视觉整段可用\\",
        r"\midrule\endfirsthead",
        r"\toprule",
        r"编号 & 极性 & 强度 & $p_-$ & $p_0$ & $p_+$ & 视觉整段可用\\",
        r"\midrule\endhead",
    ]
    labels = {"Negative": "负向", "Neutral": "中性", "Positive": "正向"}
    for row in rows:
        fields = [row["id"].split("_")[-1], labels[row["polarity"]],
                  f'{float(row["sentiment_strength"]):+.3f}',
                  f'{float(row["p_negative"]):.3f}', f'{float(row["p_neutral"]):.3f}',
                  f'{float(row["p_positive"]):.3f}',
                  "否" if "vision" in row["missing_modalities"].split(",") else "是"]
        lines.append(" & ".join(fields) + r"\\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (TARGET / "tab_q2_special_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    predictions = read_csv("q3_predictions.csv")
    evidence = read_csv("q3_selected_evidence.csv")
    if len(predictions) != 20 or len(evidence) != 20:
        raise ValueError("需要完整的20条附件4结果")
    if [row["id"] for row in predictions] != [row["id"] for row in evidence]:
        raise ValueError("预测与证据的样本顺序不一致")
    write_predictions(predictions)
    write_evidence(evidence)
    q2_special = ROOT / "outputs" / "predictions_q2_best" / "q2_predictions.csv"
    with q2_special.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 30:
        raise ValueError("需要完整的30条附件3结果")
    write_q2_special(rows)


if __name__ == "__main__":
    main()
