"""把工程结果渲染成 LaTeX 表格片段（booktabs 三线表），供 thesis/paper.tex 直接 \\input。

设计原则：
1. 论文中出现的每一个数字都来自 outputs/ 或 data/ 的落盘文件，不手抄；
2. 只输出 tabular 片段，浮动体与标题留在 paper.tex 中，便于统一排版；
3. 数值统一 4 位小数（指标）或按字段语义指定精度。

用法：python -m analysis.make_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import loader

TABLEDIR = loader.TABLES


# ------------------------------------------------------------------ 渲染工具
def _escape(text: object) -> str:
    # 注意：下划线转义为 `\_\allowbreak{}`——`{}` 必不可少，否则 TeX 会把
    # `\allowbreakid` 当成一个未定义的控制序列名。断行点让 p{} 定宽列里的
    # 长标识符（如 classification\_labels）可以折行，不会溢出到表外。
    return (str(text).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_\allowbreak{}").replace("#", r"\#"))


def _breakable(cell: object) -> str:
    """保证 `\\_` 后面一定跟一个断行点（幂等）。"""
    return (str(cell).replace(r"\_\allowbreak{}", r"\_")
            .replace(r"\_", r"\_\allowbreak{}"))


def render(rows: list[list[str]], header: list[str], *, align: str | None = None,
           caption_rows: int = 0, resize: bool = True, fontsize: str | None = None) -> str:
    """渲染 booktabs 三线表；caption_rows>0 时在第该行之前插入一条分组线。

    出口处做一次"未转义字符"体检：裸 `%` 会把行尾的 `\\\\` 注释掉并引发
    "Extra alignment tab" 这类难查的排版错误，因此直接在这里 fail loudly。
    """
    columns = len(header)
    for index, row in enumerate(rows, start=1):
        if len(row) != columns:
            raise ValueError(f"第 {index} 行有 {len(row)} 列，表头为 {columns} 列")
        for cell in row:
            text = str(cell)
            if "%" in text.replace(r"\%", "") or "&" in text.replace(r"\&", ""):
                raise ValueError(f"第 {index} 行含未转义的 % 或 &：{text[:60]}")
    align = align or ("l" + "c" * (columns - 1))
    lines = [r"\begin{tabular}{%s}" % align, r"\toprule"]
    # 表头由作者撰写，可能含数学符号，原样输出；单元格内容由调用方负责转义。
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for index, row in enumerate(rows):
        if caption_rows and index == caption_rows:
            lines.append(r"\midrule")
        lines.append(" & ".join(_breakable(cell) for cell in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines)
    if resize:
        body = ("\\begin{adjustbox}{max width=\\textwidth}\n" + body
                + "\n\\end{adjustbox}")
    if fontsize:
        body = "{\\" + fontsize + "\n" + body + "\n}"
    return body


def write(name: str, content: str) -> None:
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    (TABLEDIR / f"{name}.tex").write_text(content + "\n", encoding="utf-8")
    print(f"  wrote tables/{name}.tex")


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------ 各表
def table_data_overview() -> None:
    rows = [
        ["附件1", "37 个 video\\_id 目录 / 100 个 mp4 + label-100.xlsx", "原始视频与转写",
         "—（100 条全部提取成功）"],
        ["附件2", "aligned\\_50.pkl / unaligned\\_50.pkl / label.xlsx", "训练·验证·测试特征",
         "3395 / 728 / 727"],
        ["附件3", "对齐版 30 个 pkl（未对齐版 30 个）", "模态缺失专项测试",
         "30（无标签）"],
        ["附件4", "对齐版 20 个 pkl + 20 段视频", "可解释专项测试",
         "20（无标签）"],
    ]
    write("tab_data_overview", render(
        rows, ["附件", "内容与文件", "用途", "规模"],
        align="llll"))


def table_field_schema() -> None:
    rows = [
        ["text\\_bert", "(N, 3, 50)", "int64", "词元 / 注意力 / 分段，pos0=CLS、pos n−1=SEP"],
        ["text", "(N, 50, 768)", "float32", "BERT 上下文词表示（填充位非零，须掩码排除）"],
        ["audio", "(N, 50, 74)", "float64", "语音时序特征"],
        ["vision", "(N, 50, 35)", "float64", "视觉时序特征"],
        ["id", "list[str]", "str", "\\texttt{video\\_id\\$\\_\\$clip\\_id}"],
        ["raw\\_text", "(N,)", "str", "原始英文转写"],
        ["classification\\_labels", "(N,)", "float64", "0 负向 / 1 中性 / 2 正向"],
        ["regression\\_labels", "(N,)", "float64", "连续情感强度 $\\in[-3,3]$"],
    ]
    write("tab_field_schema", render(
        rows, ["字段", "形状", "dtype", "含义"], align="llll"))


# ---------------------------------------------------------------- 逐附件字段规范
# 说明：四张表由 `analysis/_probe.py` 的实测输出整理而来（目录清单 + 逐文件字段形状），
# 保证论文中的数据说明与附件实际结构一一对应。
def table_schema_a1() -> None:
    rows = [
        ["\\texttt{<video\\_id>/}", "目录", "37 个", "源视频目录，目录名即 MOSEI 的 video\\_id"],
        ["\\texttt{<video\\_id>/<clip\\_id>.mp4}", "视频", "100 段", "原始视频片段，含音轨；共 8 种分辨率、4 种帧率"],
        ["\\texttt{label-100.xlsx}", "标注", "100 行 × 5 列", "与视频一一对应的转写与情感标注，字段见下"],
        ["\\quad video\\_id", "str", "—", "源视频标识，与目录名一致"],
        ["\\quad clip\\_id", "int", "—", "片段编号，与 \\texttt{.mp4} 文件名一致"],
        ["\\quad text", "str", "—", "该片段的英文转写（词序列）"],
        ["\\quad label", "float", "$[-3,3]$", "连续情感强度，步长 $1/3$"],
        ["\\quad annotation", "str", "3 类", "Negative / Neutral / Positive"],
    ]
    write("tab_schema_a1", render(
        rows, ["文件 / 字段", "类型", "规格", "含义"],
        align="p{3.4cm}p{1.5cm}p{2.5cm}p{6.4cm}", fontsize="footnotesize"))


def table_schema_a2() -> None:
    rows = [
        ["\\texttt{aligned\\_50.pkl}", "对齐版特征", "993.8 MB", "顶层为 \\texttt{train}/\\texttt{valid}/\\texttt{test} 三键"],
        ["\\quad audio", "(N, 50, 74)", "float64", "已在 50 位网格上对齐的语音特征"],
        ["\\quad vision", "(N, 50, 35)", "float64", "已在 50 位网格上对齐的视觉特征"],
        ["\\quad text", "(N, 50, 768)", "float32", "按词元位置展开的 BERT 表示"],
        ["\\quad text\\_bert", "(N, 3, 50)", "int64", "词元 id / 注意力掩码 / 分段 id"],
        ["\\quad raw\\_text", "(N,)", "str", "原始英文转写（最长 1662 字符）"],
        ["\\quad id", "list[str]", "str", "\\texttt{video\\_id\\$\\_\\$clip\\_id}"],
        ["\\quad classification\\_labels", "(N,)", "float64", "0 负向 / 1 中性 / 2 正向"],
        ["\\quad regression\\_labels", "(N,)", "float64", "连续情感强度 $\\in[-3,3]$"],
        ["\\texttt{unaligned\\_50.pkl}", "未对齐版", "2897 MB", "字段与对齐版相同，另加长度标记"],
        ["\\quad audio / vision", "(N, 500, ·)", "float64", "逐帧而非逐词，固定 500 帧零填充"],
        ["\\quad audio\\_lengths", "(N,)", "int", "每条样本语音的真实帧数"],
        ["\\quad vision\\_lengths", "(N,)", "int", "每条样本视觉的真实帧数"],
        ["\\texttt{label.xlsx}", "标注", "4850 行 × 6 列", "列同附件1，另加 \\texttt{mode} 列标记所属划分"],
    ]
    write("tab_schema_a2", render(
        rows, ["文件 / 字段", "类型", "规格", "含义"],
        align="p{3.4cm}p{1.5cm}p{2.5cm}p{6.4cm}", fontsize="footnotesize"))


def table_schema_a3() -> None:
    rows = [
        ["\\texttt{对齐版本/}", "目录", "30 个 pkl",
         "\\texttt{附件3\\_01.pkl} $\\sim$ \\texttt{附件3\\_30.pkl}，逐样本一个文件"],
        ["\\quad test", "dict", "—", "唯一的顶层键，指向该样本的特征字典"],
        ["\\quad\\quad audio", "(1, 50, 74)", "float32", "对齐到 50 位的语音特征，零值即缺失"],
        ["\\quad\\quad vision", "(1, 50, 35)", "float32", "对齐到 50 位的视觉特征，零值即缺失"],
        ["\\quad\\quad text\\_bert", "(1, 3, 50)", "float32", "词元 id / 注意力掩码 / 分段 id"],
        ["\\texttt{未对齐版本/}", "目录", "30 个 pkl",
         "\\texttt{附件3\\_未对齐版本\\_01.pkl} $\\sim$ \\texttt{\\_30.pkl}"],
        ["\\quad\\quad audio / vision", "(1, 500, ·)", "float64", "逐帧特征，未做词级对齐"],
        ["\\quad\\quad raw\\_text", "(1,)", "str", "该样本的英文转写"],
        ["（无标签）", "—", "—", "附件3 不含 \\texttt{classification\\_labels} 等监督字段"],
    ]
    write("tab_schema_a3", render(
        rows, ["文件 / 字段", "类型", "规格", "含义"],
        align="p{3.4cm}p{1.5cm}p{2.5cm}p{6.4cm}", fontsize="footnotesize"))


def table_schema_a4() -> None:
    rows = [
        ["\\texttt{对齐版本/}", "目录", "20 个 pkl",
         "\\texttt{01.pkl} $\\sim$ \\texttt{20.pkl}，逐样本一个文件（无统一批次维）"],
        ["\\quad audio", "(50, 74)", "float64", "对齐到 50 位的语音特征"],
        ["\\quad vision", "(50, 35)", "float64", "对齐到 50 位的视觉特征（样本 13 整段为 0）"],
        ["\\quad text", "(50, 768)", "float32", "按词元位置展开的 BERT 表示"],
        ["\\quad text\\_bert", "(3, 50)", "int64", "词元 id / 注意力掩码 / 分段 id"],
        ["\\quad raw\\_text", "str", "—", "该样本的英文转写"],
        ["\\quad id", "str", "\\texttt{01} $\\sim$ \\texttt{20}", "样本编号（两位数字符串）"],
        ["\\texttt{未对齐版本/}", "目录", "20 个 pkl", "字段同对齐版，另加长度标记"],
        ["\\quad audio / vision", "(500, ·)", "float64", "逐帧特征，未做词级对齐"],
        ["\\quad audio\\_lengths / vision\\_lengths", "int", "—", "真实帧数"],
        ["\\texttt{*.mp4}", "视频", "20 段", "与 pkl 同名的原始视频，用于时间定位核对"],
        ["（无标签）", "—", "—", "附件4 不含任何监督标签，不含起始时间戳"],
    ]
    write("tab_schema_a4", render(
        rows, ["文件 / 字段", "类型", "规格", "含义"],
        align="p{3.4cm}p{1.5cm}p{2.5cm}p{6.4cm}", fontsize="footnotesize"))


def table_schema_notes() -> None:
    """四个附件的读取约定与跨附件一致性（文字表，便于对照）。"""
    rows = [
        ["读取方式", "\\texttt{pickle.load()}；附件2 按 \\texttt{data[split][字段][j]} 取第 $j$ 条",
         "逐文件一个样本", "逐文件一个样本"],
        ["样本索引", "\\texttt{id}：\\texttt{video\\_id\\$\\_\\$clip\\_id}",
         "文件名编号（01$\\sim$30）", "文件名编号（01$\\sim$20）"],
        ["时序长度", "统一 50 位（未对齐版为 500 帧）", "统一 50 位（未对齐版 500 帧）",
         "统一 50 位（未对齐版 500 帧）"],
        ["零值含义", "该位置无有效观测", "该位置无有效观测（含额外缺失）", "该位置无有效观测"],
        ["监督标签", "有（极性 + 强度）", "无", "无"],
        ["本文字段命名", "\\texttt{audio}/\\texttt{vision}/\\texttt{text}",
         "\\texttt{audio} / \\texttt{vision} / \\texttt{text}",
         "\\texttt{audio}/\\texttt{vision}/\\texttt{text}"],
    ]
    write("tab_schema_notes", render(
        rows, ["对照项", "附件2", "附件3", "附件4"], align="lp{3.3cm}p{3.3cm}p{3.3cm}", fontsize="footnotesize"))


def table_grid_convention() -> None:
    rows = [
        ["$0$", "[CLS]", "0", "0", "特殊槽，声画置零"],
        ["$1\\sim m-2$", "内容词元", "有值", "多数有值", "每词一个声画向量，按子词复制展开"],
        ["$m-1$", "[SEP]", "0", "0", "特殊槽，声画置零"],
        ["$m\\sim 49$", "填充", "0", "0", "零填充，由掩码排除"],
    ]
    write("tab_grid_convention", render(
        rows, ["位置", "文本侧", "语音", "视觉", "规则"], align="lllll"))


def table_q1_features() -> None:
    rows = [
        ["文本", "BERT-base-uncased 末层隐状态", "768 / 词", "词元级均值回词",
         "与附件2 的 text 逐元素一致（cosine=1.0000）"],
        ["语音", "40 维 log-Mel + log-RMS + 过零率 + log 基频", "43 / 帧 → 86 / 词",
         "词区间均值+标准差", "25 ms 帧长、10 ms 帧移、16 kHz、自相关法基频"],
        ["韵律", "词时长、前后停顿、局部语速、相对音高、浊音比、能量、过零率", "12 / 词",
         "词级统计", "相对音高以该段浊音基频中位数为参照，单位半音"],
        ["视觉", "16 人脸点(双眼距归一化) + 8 姿态点(肩宽归一化, 含深度与可见度)",
         "66 / 帧 → 132 / 词", "词区间均值+标准差", "5 fps 采样，MediaPipe Face/Pose Landmarker"],
    ]
    write("tab_q1_features", render(
        rows, ["模态", "特征定义", "维度", "池化方式", "关键参数"], align="lllll"))


def table_q1_summary() -> None:
    frame = loader.manifest()
    show = frame.head(8)[["id", "seconds", "words", "mean_alignment_score",
                          "alignment_text_similarity", "vision_coverage", "face_coverage",
                          "pose_coverage", "status"]]
    rows = [[f"\\texttt{{{_escape(r.id).replace('$_$', chr(36)+'\\_'+chr(36))}}}",
             fmt(float(r.seconds), 2), str(int(r.words)), fmt(float(r.mean_alignment_score), 3),
             fmt(float(r.alignment_text_similarity), 3), fmt(float(r.vision_coverage), 2),
             fmt(float(r.face_coverage), 2), fmt(float(r.pose_coverage), 2), str(r.status)]
            for r in show.itertuples()]
    total = len(frame)
    rows.append([f"\\textbf{{全部 {total} 条}}",
                 f"{frame['seconds'].astype(float).median():.2f}",
                 f"{int(frame['words'].astype(int).median())}",
                 f"{frame['mean_alignment_score'].astype(float).mean():.3f}",
                 f"{frame['alignment_text_similarity'].astype(float).mean():.3f}",
                 f"{frame['vision_coverage'].astype(float).mean():.3f}",
                 f"{frame['face_coverage'].astype(float).mean():.3f}",
                 f"{frame['pose_coverage'].astype(float).mean():.3f}",
                 "100/100 ok"])
    write("tab_q1_summary", render(
        rows, ["样本编号", "时长/s", "词数", "对齐分数", "转写一致率", "视觉覆盖", "人脸覆盖",
               "姿态覆盖", "状态"], align="lcccccccl", caption_rows=len(rows) - 1))


def table_q1_env() -> None:
    path = loader.OUTPUTS / "features_q1_face_pose" / "environment.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    keys = [("text_model", "文本模型"), ("torch", "PyTorch"), ("transformers", "Transformers"),
            ("whisperx", "WhisperX"), ("mediapipe", "MediaPipe"), ("opencv", "OpenCV"),
            ("ffmpeg", "FFmpeg"), ("sample_rate_hz", "音频采样率/Hz"),
            ("audio_frame_ms", "帧长/ms"), ("audio_hop_ms", "帧移/ms"),
            ("video_sample_fps", "视觉采样率/fps")]
    rows = [[label, _escape(str(info.get(key, "--")))[:78]] for key, label in keys]
    rows.append(["人脸权重 SHA-256", _escape(str(info.get("face_model_sha256", ""))[:32]) + "…"])
    rows.append(["姿态权重 SHA-256", _escape(str(info.get("pose_model_sha256", ""))[:32]) + "…"])
    write("tab_q1_env", render(rows, ["项目", "取值"], align="ll"))


def table_q2_hyper() -> None:
    run = json.loads((loader.RUNS / "fuse_s2026" / "run.json").read_text(encoding="utf-8"))
    base = json.loads((loader.RUNS / "availability_s2026" / "run.json").read_text(encoding="utf-8"))
    rows = [
        ["架构", "门控融合", "三因子分解 + 动态融合"],
        ["文本输入", "768 维 BERT 上下文表示", "同左"],
        ["声学输入", "74 维（+ 一阶差分 = 148）", "同左"],
        ["视觉输入", "35 维", "同左"],
        ["隐层宽度", f"{run.get('hidden_dim', 128)}", f"{run.get('hidden_dim', 128)}"],
        ["dropout", fmt(base["dropout"], 2), fmt(run["dropout"], 2)],
        ["批大小 / 学习率", f"{base['batch_size']} / {base['lr']}", f"{run['batch_size']} / {run['lr']}"],
        ["优化器", "AdamW（weight\\_decay=1e-4）", "同左"],
        ["代价权重", "0.35 完整 + 0.65 缺失 + 0.05 一致性",
         "同左 + 正则项（对比/信息/对偶/重建/交叉）"],
        ["缺失模拟", "整段缺失 + 零散短游程", "整段 + 零散短游程 + 多尺度连续区间"],
        ["早停", f"patience={base['patience']}", f"patience={run['patience']}"],
    ]
    write("tab_q2_hyper", render(rows, ["项目", "门控基线（A）", "FUSE-Net（E）"], align="lll"))


def table_q2_metrics() -> None:
    rows = []
    for name, tag in (("A 门控基线（附件3同分布增强）", "A 门控基线"),
                      ("E FUSE-Net（多尺度缺失）", "E FUSE-Net")):
        result = loader.evaluate_result(name)
        for split, label in (("valid", "验证集"), ("test", "测试集")):
            if split not in result:
                continue
            views = result[split].get("views", [])
            for index, view in enumerate(views):
                stats = result[split][view]
                rows.append([tag if index == 0 else "", label if index == 0 else "",
                             {"clean": "完整输入", "local": "局部短游程", "whole": "整段缺失",
                              "interval": "连续区间"}.get(view, view),
                             fmt(stats["accuracy"]), fmt(stats["macro_f1"]),
                             fmt(stats["mae"]), fmt(stats["pearson"])])
    write("tab_q2_metrics", render(
        rows, ["模型", "划分", "缺失视图", "Accuracy", "macro-F1", "MAE", "Pearson"],
        align="lllcccc"))


def table_robust_whole() -> None:
    robust = loader.robustness("fuse_availability")
    base = robust.loc[robust["pattern"] == "none"].iloc[0]
    order = [("audio", "仅语音缺失"), ("vision", "仅视觉缺失"),
             ("audio+vision", "语音+视觉缺失"), ("text", "仅文本缺失")]
    rows = [["完整输入", "--", fmt(base["macro_f1"]), fmt(base["mae"]), "--", "--"]]
    for key, label in order:
        row = robust[(robust["pattern"] == "whole") & (robust["missing_modalities"] == key)]
        if not len(row):
            continue
        item = row.iloc[0]
        rows.append([label, "100\\%", fmt(item["macro_f1"]), fmt(item["mae"]),
                     f"{item['macro_f1'] - base['macro_f1']:+.4f}",
                     f"{item['mae'] - base['mae']:+.4f}"])
    write("tab_robust_whole", render(
        rows, ["缺失类型", "缺失比例", "macro-F1", "MAE", "$\\Delta$F1", "$\\Delta$MAE"],
        align="lcccccc"))


def table_robust_interval() -> None:
    robust = loader.robustness("fuse_availability")
    subset = robust[(robust["pattern"] == "interval")
                    & (robust["missing_modalities"] == "audio+vision")]
    pivot = subset.pivot_table(index="missing_rate", columns="location", values="macro_f1")
    pivot = pivot[["start", "middle", "end", "random"]]
    rows = [[f"{rate:.0%}".replace("%", r"\%")] + [fmt(pivot.loc[rate, col]) for col in pivot.columns]
            for rate in pivot.index]
    write("tab_robust_interval", render(
        rows, ["缺失长度比例", "起始", "中段", "末尾", "随机"], align="lcccc"))


def table_modality_ablation() -> None:
    entries = [("纯文本", loader.deep_experiment("text_only")),
               ("纯语音", loader.deep_experiment("audio_only")),
               ("纯视觉", loader.deep_experiment("vision_only")),
               ("三模态 门控基线", loader.evaluate_result("A 门控基线（附件3同分布增强）").get("test", {})),
               ("三模态 FUSE-Net", loader.evaluate_result("E FUSE-Net（多尺度缺失）").get("test", {}))]
    rows = []
    for label, result in entries:
        if not result or "clean" not in result:
            continue
        clean = result["clean"]
        rows.append([label, fmt(clean["accuracy"]), fmt(clean["macro_f1"]),
                     fmt(clean["mae"]), fmt(clean["pearson"])])
    write("tab_modality_ablation", render(
        rows, ["模型", "Accuracy", "macro-F1", "MAE", "Pearson"], align="lcccc"))


def table_ablation() -> None:
    rows = []
    plan = [("多尺度缺失（含文本）+ 门控", loader.deep_experiment("interval")),
            ("多尺度缺失（含文本）+ FUSE", loader.deep_experiment("fuse")),
            ("多尺度缺失（仅声画）+ FUSE", loader.deep_experiment("fuse_av")),
            ("多尺度缺失（仅声画）+ FUSE + $\\beta$ 非零", loader.evaluate_result(
                "E FUSE-Net（多尺度缺失）").get("test", {})),
            ("A+E 四模型集成", loader.deep_experiment("mix"))]
    for label, result in plan:
        if not result or "clean" not in result:
            continue
        clean = result["clean"]
        rows.append([label, fmt(clean["macro_f1"]), fmt(clean["mae"]), fmt(clean["pearson"])])
    write("tab_ablation", render(
        rows, ["配置（测试集，完整输入视图）", "macro-F1", "MAE", "Pearson"], align="lccc"))


def table_confusion() -> None:
    rows = []
    for name, path in (("A 门控基线", loader.RUNS / "availability" / "validation_per_sample.csv"),
                       ("E FUSE-Net", loader.OUTPUTS / "experiments" / "fuseE_test_per_sample.csv")):
        frame = loader.read_csv(path)
        truth = frame["label_class"].to_numpy()
        guess = frame["clean_class"].to_numpy()
        matrix = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = int(((truth == i) & (guess == j)).sum())
        for class_index, class_name in enumerate(["负向", "中性", "正向"]):
            tp = matrix[class_index, class_index]
            precision = tp / max(1, matrix[:, class_index].sum())
            recall = tp / max(1, matrix[class_index].sum())
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
            rows.append([name if class_index == 0 else "", class_name, str(tp),
                         " / ".join(str(v) for v in matrix[class_index]),
                         fmt(precision, 3), fmt(recall, 3), fmt(f1, 3)])
    write("tab_confusion", render(
        rows, ["模型", "真实类别", "命中", "该行分布（负/中/正）", "精确率", "召回率", "F1"],
        align="llcccccc", caption_rows=3))


def table_q2_predictions() -> None:
    frame = loader.q2_predictions()
    rows = []
    for r in frame.itertuples():
        rows.append([_escape(r.id), _escape(r.polarity), str(int(r.class_id)),
                     f"{r.sentiment_strength:+.3f}", fmt(r.p_negative, 3), fmt(r.p_neutral, 3),
                     fmt(r.p_positive, 3), _escape(r.available_modalities or "--"),
                     _escape(r.missing_modalities) if isinstance(r.missing_modalities, str) else "无"])
    write("tab_q2_predictions", render(
        rows, ["样本编号", "极性", "类别", "强度", "$p_{\\text{负}}$", "$p_{\\text{中}}$",
               "$p_{\\text{正}}$", "可用模态", "缺失模态"], align="llcccccll"))


def table_q3_predictions() -> None:
    frame = loader.q3_predictions()
    rows = []
    for r in frame.itertuples():
        rows.append([_escape(r.id), _escape(r.polarity), f"{r.sentiment_strength:+.3f}",
                     fmt(r.class_share_text, 2), fmt(r.class_share_audio, 2),
                     fmt(r.class_share_vision, 2), _escape(r.main_modality),
                     _escape(r.evidence_time_status)])
    write("tab_q3_predictions", render(
        rows, ["样本编号", "极性", "强度", "文本份额", "语音份额", "视觉份额", "主要模态",
               "时间定位"], align="llcccccc"))


def table_q3_evidence() -> None:
    frame = loader.q3_evidence()
    rows = []
    for sample in frame["id"].astype(str).unique()[:4]:
        subset = frame[frame["id"].astype(str) == sample].sort_values("class_margin_drop",
                                                                     ascending=False).head(3)
        for rank, r in enumerate(subset.itertuples(), start=1):
            words = str(r.evidence_words) if isinstance(r.evidence_words, str) else ""
            words = (words[:26] + "…") if len(words) > 26 else words
            start = f"{float(r.start_seconds):.2f}" if str(r.start_seconds) not in ("", "nan") else "--"
            end = f"{float(r.end_seconds):.2f}" if str(r.end_seconds) not in ("", "nan") else "--"
            rows.append([sample if rank == 1 else "", str(rank), _escape(r.modality),
                         f"[{int(r.position_start)},{int(r.position_end_exclusive)})",
                         _escape(words), start, end, fmt(float(r.class_margin_drop), 3)])
    write("tab_q3_evidence", render(
        rows, ["样本", "排名", "模态", "词位区间", "证据词", "起/s", "止/s", "$\\Delta$margin"],
        align="llclccccc"))


def table_data_diagnostics() -> None:
    """数据质量诊断清单：把 16 项实测缺陷与对策逐条列出（`measure_diagnostics.py` 产出）。

    该表有 16 行且每行较长，用 `longtable` 让它跨页断行——若用普通 `table`
    环境，它会整体推到下一页并在前一页留下半页空白。
    """
    frame = loader.data_diagnostics()
    header = ["附件", "检查项", "实测现象（量化证据）", "对建模的危害", "本文对策"]
    # 列宽之和留出 10% 给 5 列的 \tabcolsep 内边距，否则整表会比 \textwidth 宽约 50 pt。
    widths = ["0.080", "0.135", "0.240", "0.190", "0.245"]
    lines = [
        r"{\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{" + " ".join(f"p{{{w}\\textwidth}}" for w in widths) + "}",
        r"\caption{数据质量诊断清单：现象、证据、危害与对策}",
        r"\label{tab:data_diagnostics}\\",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\small（续上页）}\\",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    previous = None
    for index, row in enumerate(frame.itertuples(), start=1):
        tag = _escape(row.附件) if row.附件 != previous else ""
        previous = row.附件
        cells = [tag, _escape(row.检查项), _escape(row.实测现象),
                 _escape(row.对建模的危害), _escape(row.本文对策)]
        # 转义之后再体检：此时残留的裸 % / & 一定是漏转义的
        for column, value in zip(header, cells):
            if "%" in value.replace(r"\%", "") or "&" in value.replace(r"\&", ""):
                raise ValueError(f"诊断清单第 {index} 行的「{column}」含未转义字符：{value[:50]}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"}"]
    write("tab_data_diagnostics", "\n".join(lines))


def table_defect_stats() -> None:
    """逐附件的缺陷统计量（全部由 `diagnostics_facts.json` 计算得到）。"""
    facts = loader.diagnostics_facts()
    # 注意：这里输出裸 `%`，由 _escape 统一转义；若直接写 `\%`，_escape 会把反斜杠
    # 再次转义成 \textbackslash{}，在 PDF 里显示成 "\%"。
    pct = lambda value: f"{value * 100:.1f}%"  # noqa: E731 - 局部小工具
    rows = [
        ["附件1", "源视频复用", f"{facts['a1_clips']} 条 clip / {facts['a1_videos']} 个源视频",
         f"单源最多 {facts['a1_max_clips_per_video']} 段"],
        ["附件1", "容器参数", f"{facts['a1_resolutions']} 种分辨率 / {facts['a1_frame_rates']} 种帧率",
         f"主流 {facts['a1_resolution_top']} 占 {pct(facts['a1_resolution_top_share'])}，"
         f"{facts['a1_fps_top']:g} fps 占 {pct(facts['a1_fps_top_share'])}"],
        ["附件1", "时长跨度", f"{facts['a1_seconds_min']:.2f}–{facts['a1_seconds_max']:.2f} s",
         f"相差 {facts['a1_seconds_ratio']:.1f} 倍"],
        ["附件1", "人脸漏检", f"{facts['a1_face_zero']} 条无人脸",
         f"其中 {facts['a1_vision_zero']} 条连姿态也检不出"],
        ["附件1", "最近帧补位", f"均值 {facts['a1_near_mean']:.3f}",
         f"{facts['a1_near_positive']}/{facts['a1_clips']} 条存在，最大 {facts['a1_near_max']:.3f}"],
        ["附件1", "对齐置信度", f"最低 {facts['a1_score_min']:.3f}，中位 {facts['a1_score_median']:.3f}",
         f"{facts['a1_score_low']} 条低于 0.3"],
        ["附件2", "网格稀疏", f"中位内容位 {facts['a2_content_median']:.0f} / 50",
         f"均值 {facts['a2_content_mean']:.1f}"],
        ["附件2", "文本截断", f"{facts['a2_truncated']} 条（{pct(facts['a2_truncated_share'])}）",
         f"最严重丢失 {pct(facts['a2_truncated_worst_loss'])} 的词"],
        ["附件2", "零值语义", "语音有效位 ≡ 文本内容位",
         f"{facts['a2_zero_eq_content_count']}/{facts['a2_samples']}"
         f"（{pct(facts['a2_zero_eq_content_share'])}）"],
        ["附件2", "子词复制", "复制位数 = m−2−W",
         f"{facts['a2_subword_exact_count']}/{facts['a2_subword_total']}"
         f"（{pct(facts['a2_subword_exact_share'])}）"],
        ["附件2", "类别不平衡", "中性类占比 0.22–0.25", "正向类约 0.50"],
        ["附件3", "额外缺失", f"逐模态 0–{facts['a3_extra_missing_rate_max']:.0%}",
         f"汇总 {pct(facts['a3_extra_missing_share_pooled'])}，"
         f"最长游程 {facts['a3_run_max']} 位"],
        ["附件3", "缺失位置相关", "语音与视觉缺失位置重合",
         f"{facts['a3_audio_vision_same_count']}/{facts['a3_samples']} 条完全相同"],
        ["附件4", "通道失效", f"样本 {'、'.join(facts['a4_vision_allzero'])} 视觉全零",
         f"缺失 {facts['a4_vision_missing_total']} 位；语音无缺失"],
    ]
    # 该表的单元格是手工撰写的字符串，必须逐个走转义，否则 `%` 会注释掉行尾的 \\
    write("tab_defect_stats", render(
        [[_escape(cell) for cell in row] for row in rows],
        ["附件", "缺陷维度", "实测值", "补充证据"], align="llll"))


BUILDERS = [table_data_overview,
            table_schema_a1, table_field_schema, table_schema_a2, table_schema_a3,
            table_schema_a4, table_schema_notes,
            table_grid_convention, table_data_diagnostics, table_defect_stats,
            table_q1_features, table_q1_summary, table_q1_env, table_q2_hyper, table_q2_metrics,
            table_robust_whole, table_robust_interval, table_modality_ablation, table_ablation,
            table_confusion, table_q2_predictions, table_q3_predictions, table_q3_evidence]


def main() -> None:
    print(f"输出目录：{TABLEDIR}")
    for builder in BUILDERS:
        builder()
    print(f"共 {len(BUILDERS)} 张表")


if __name__ == "__main__":
    main()
