"""统一读取工程中已落盘的全部结果产物，供绘图与建表使用。

所有论文中的数字都经由这里读取，避免手抄；任何一张图、一张表都能由
`python -m analysis.make_figures` / `make_tables` 重新生成。
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTPUTS = PROJECT / "outputs"
RUNS = OUTPUTS / "runs"
EXPERIMENTS = OUTPUTS / "experiments"
THESIS = PROJECT.parent / "thesis"
FIGURES = THESIS / "figures"
TABLES = THESIS / "tables"


# ------------------------------------------------------------------ 基础工具
def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path, **kw) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kw)


def word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", str(text).strip()) if w])


# ------------------------------------------------------------------ 附件2 原始特征
@lru_cache(maxsize=1)
def aligned() -> dict:
    """加载附件2 对齐特征（约 1 GB 常驻内存）。"""
    path = DATA / "附件2-数据集特征文件" / "aligned_50.pkl"
    sys.modules.setdefault("numpy._core", np.core)
    with path.open("rb") as handle:
        return pickle.load(handle)


def split_meta(split: str) -> pd.DataFrame:
    """派生附件2 某一划分的关键结构量：词数、token 数、内容位、有效位等。"""
    raw = aligned()[split]
    tokens = np.asarray(raw["text_bert"])
    attention = tokens[:, 1, :] > 0
    content = attention & (tokens[:, 0, :] != 0) & (tokens[:, 0, :] != 101) & (tokens[:, 0, :] != 102)
    audio = np.asarray(raw["audio"])
    vision = np.asarray(raw["vision"])
    texts = np.asarray(raw["raw_text"])
    frame = pd.DataFrame({
        "id": [str(x) for x in np.asarray(raw["id"])],
        "split": split,
        "tokens": attention.sum(1),
        "content": content.sum(1),
        "words": [word_count(x) for x in texts],
        "audio_valid": (np.abs(audio).max(-1) > 1e-8).sum(1),
        "vision_valid": (np.abs(vision).max(-1) > 1e-8).sum(1),
        "class": np.asarray(raw["classification_labels"]).astype(int),
        "strength": np.asarray(raw["regression_labels"]).astype(float),
    })
    frame["truncated"] = frame["words"] > frame["tokens"]
    return frame


def all_splits_meta() -> pd.DataFrame:
    return pd.concat([split_meta(s) for s in ("train", "valid", "test")], ignore_index=True)


def grid_layout_example(split: str = "train", index: int = 0) -> dict:
    """取一条样本，返回 50 位网格上各模态的"是否有值 / 是否特殊槽"掩码，用于结构示意图。"""
    raw = aligned()[split]
    tokens = np.asarray(raw["text_bert"])[index]
    audio = np.asarray(raw["audio"])[index]
    vision = np.asarray(raw["vision"])[index]
    attention = tokens[1] > 0
    m = int(attention.sum())
    return {
        "m": m,
        "words": word_count(np.asarray(raw["raw_text"])[index]),
        "text_content": attention & (tokens[0] != 0) & (tokens[0] != 101) & (tokens[0] != 102),
        "cls": tokens[0] == 101,
        "sep": tokens[0] == 102,
        "audio_valid": np.abs(audio).max(-1) > 1e-8,
        "vision_valid": np.abs(vision).max(-1) > 1e-8,
        "raw_text": str(np.asarray(raw["raw_text"])[index]),
    }


# ------------------------------------------------------------------ 训练/评估结果
RUN_TABLE = {
    "A 门控基线（附件3同分布增强）": RUNS / "availability",
    "E FUSE-Net（多尺度缺失）": RUNS / "fuse_availability",
}
CKPT_RUNS = {
    "门控基线 s2026": RUNS / "availability_s2026",
    "门控基线 s2027": RUNS / "availability_s2027",
    "FUSE-Net s2026": RUNS / "fuse_s2026",
    "FUSE-Net s2027": RUNS / "fuse_s2027",
    "纯文本单模态": RUNS / "text_only",
    "纯语音单模态": EXPERIMENTS / "audio_only",
    "纯视觉单模态": EXPERIMENTS / "vision_only",
}


def metrics_curve(run: str | Path) -> pd.DataFrame:
    """接受完整目录或 outputs/runs 下的运行名。"""
    path = Path(run)
    if not (path / "metrics.csv").is_file():
        path = RUNS / run
    return read_csv(path / "metrics.csv")


def best_row(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["val_score"].idxmin()]


def evaluate_result(name: str) -> dict:
    """读取某个方案在验证集/测试集上的分视图指标。"""
    out = {}
    for split, fname in (("valid", "validation_metrics.json"), ("test", "test_metrics.json")):
        path = RUN_TABLE[name] / fname
        if path.is_file():
            out[split] = read_json(path)
    return out


def deep_experiment(name: str, split: str = "test") -> dict:
    """读取 outputs/experiments 下的一次性对照实验指标。"""
    path = EXPERIMENTS / f"{name}_{split}.json"
    return read_json(path) if path.is_file() else {}


def robustness(run: str = "fuse_availability") -> pd.DataFrame:
    return read_csv(RUNS / run / "robustness.csv")


def per_sample(run: str = "availability") -> pd.DataFrame:
    """逐样本预测（注意：availability 目录下该文件名含 validation，实际为 test 划分）。"""
    return read_csv(RUNS / run / "validation_per_sample.csv")


def reg_curve(run: str) -> pd.DataFrame:
    curve = metrics_curve(run)
    cols = ["epoch"] + [c for c in curve.columns if c.startswith("reg_")]
    return curve[cols]


# ------------------------------------------------------------------ 问题1 / 问题3 产物
def manifest() -> pd.DataFrame:
    return read_csv(OUTPUTS / "features_q1_face_pose" / "manifest.csv")


def q1_summary() -> pd.DataFrame:
    return read_csv(OUTPUTS / "figures_q1" / "q1_summary.csv")


def q1_feature(video_id: str, clip_id: str) -> dict:
    path = OUTPUTS / "features_q1_face_pose" / f"{video_id}_{clip_id}.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def q2_predictions() -> pd.DataFrame:
    return read_csv(OUTPUTS / "predictions_availability" / "q2_predictions.csv")


def q3_predictions() -> pd.DataFrame:
    return read_csv(OUTPUTS / "predictions_availability_q3" / "q3_predictions.csv")


def q3_evidence() -> pd.DataFrame:
    return read_csv(OUTPUTS / "predictions_availability_q3" / "q3_evidence_windows.csv")


def q3_alignment() -> dict:
    return read_json(RUNS / "main" / "q3_alignment.json")


# ------------------------------------------------------------------ 数据质量诊断
DIAGNOSTICS = Path(__file__).resolve().parent / "results"


def diagnostics_facts() -> dict:
    """`measure_diagnostics.py` 落盘的关键数字（数据缺陷实测值）。"""
    return read_json(DIAGNOSTICS / "diagnostics_facts.json")


def data_diagnostics() -> pd.DataFrame:
    """逐项诊断清单：附件 / 检查项 / 实测现象 / 危害 / 对策。"""
    return read_csv(DIAGNOSTICS / "data_diagnostics.csv")


def grid_evidence() -> pd.DataFrame:
    """50 位网格结构反演的逐样本证据（有效位、子词复制、零值语义）。"""
    return read_csv(DIAGNOSTICS / "grid_evidence.csv")


def grid_position_profile() -> pd.DataFrame:
    """逐位置的"有值样本比例"，用于展示 CLS/SEP 特殊槽与尾部零填充。"""
    return read_csv(DIAGNOSTICS / "grid_position_profile.csv")


def defect_attachment1() -> pd.DataFrame:
    return read_csv(DIAGNOSTICS / "defect_attachment1.csv")


def defect_attachment3() -> pd.DataFrame:
    return read_csv(DIAGNOSTICS / "defect_attachment3.csv")


def defect_attachment4() -> pd.DataFrame:
    return read_csv(DIAGNOSTICS / "defect_attachment4.csv")


def video_meta() -> pd.DataFrame:
    return read_csv(DIAGNOSTICS / "a1_video_meta.csv")


def video_features(video_id: str, clip_id: str) -> dict:
    """逐帧音视频特征不落盘，脚本只保存词级结果；此函数保留接口以便扩展。"""
    raise NotImplementedError("逐帧特征未落盘，如需请扩展 features_q1.py")


# ------------------------------------------------------------------ 便捷摘要
def headline_numbers() -> dict:
    """论文摘要与结论中用到的核心数字，集中一处便于核对。"""
    numbers = {}
    for name in RUN_TABLE:
        result = evaluate_result(name)
        if "test" in result:
            numbers[name] = {view: result["test"][view] for view in result["test"].get("views", [])}
    return numbers
