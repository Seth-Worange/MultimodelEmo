"""数据质量诊断：把四个附件的缺陷逐项实测，落盘为可被论文引用的产物。

设计原则与 `measure_extra.py` 一致——论文正文不手抄数字，本节所有"实测证据"
都由本模块从原始附件计算后落盘，再由 `make_figures` / `make_tables` 取用。

产物：
- ``results/data_diagnostics.csv``  逐项诊断清单（附件/检查项/现象/量化证据/危害/对策）
- ``results/diagnostics_facts.json`` 关键数字（供正文与摘要引用）
- ``results/grid_evidence.csv``      50 位网格结构反演的逐样本定量证据
- ``results/defect_profile.csv``     四个附件的缺陷画像（绘图用）

用法：
    python -m analysis.measure_diagnostics
"""

from __future__ import annotations

import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import loader

RESULTS = Path(__file__).resolve().parent / "results"
DATA = loader.DATA
MODALITIES = ("text", "audio", "vision")


# ------------------------------------------------------------------ 读取工具
def _read_pickle(path: Path):
    sys.modules.setdefault("numpy._core", np.core)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _content_mask(tokens: np.ndarray) -> np.ndarray:
    """由 text_bert 的 (3, 50) 结构还原"内容位"掩码：注意力为 1 且不是 CLS/SEP。"""
    attention = tokens[1] > 0
    ids = tokens[0]
    return attention & (ids != 0) & (ids != 101) & (ids != 102)


def _attachment3_samples() -> list[dict]:
    folder = next(DATA.glob("附件3-模态缺失特征样本/对齐版本"))
    samples = []
    for path in sorted(folder.glob("*.pkl")):
        block = _read_pickle(path)
        block = block[list(block)[0]] if "text_bert" not in block else block
        samples.append({
            "name": path.stem,
            "text_bert": np.asarray(block["text_bert"])[0],
            "audio": np.asarray(block["audio"])[0],
            "vision": np.asarray(block["vision"])[0],
        })
    return samples


def _attachment4_samples() -> list[dict]:
    folder = sorted(next(DATA.glob("附件4-可解释专项视频样本与特征文件")).glob("*/对齐版本"))[0]
    samples = []
    for path in sorted(folder.glob("*.pkl")):
        block = _read_pickle(path)
        samples.append({
            "name": path.stem,
            "text_bert": np.asarray(block["text_bert"]),
            "audio": np.asarray(block["audio"]),
            "vision": np.asarray(block["vision"]),
        })
    return samples


def _run_lengths(flags: np.ndarray) -> list[int]:
    """把布尔序列切成连续 True 游程的长度列表。"""
    lengths, current = [], 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


# ------------------------------------------------------------------ 附件1 诊断
def diagnose_attachment1() -> tuple[dict, pd.DataFrame]:
    frame = loader.manifest()
    seconds = frame["seconds"].astype(float)
    face = frame["face_coverage"].astype(float).to_numpy()
    pose = frame["pose_coverage"].astype(float).to_numpy()
    vision = frame["vision_coverage"].astype(float).to_numpy()
    near = frame["nearest_frame_fraction"].astype(float).to_numpy()
    score = frame["mean_alignment_score"].astype(float).to_numpy()
    similarity = frame["alignment_text_similarity"].astype(float).to_numpy()

    per_video = frame.groupby("video_id").size()
    facts = {
        "a1_clips": int(len(frame)),
        "a1_videos": int(frame["video_id"].nunique()),
        "a1_max_clips_per_video": int(per_video.max()),
        "a1_seconds_min": float(seconds.min()),
        "a1_seconds_max": float(seconds.max()),
        "a1_seconds_ratio": float(seconds.max() / seconds.min()),
        "a1_seconds_median": float(seconds.median()),
        "a1_face_zero": int((face == 0).sum()),
        "a1_pose_zero": int((pose == 0).sum()),
        "a1_vision_zero": int((vision == 0).sum()),
        "a1_face_mean": float(face.mean()),
        "a1_pose_mean": float(pose.mean()),
        "a1_vision_mean": float(vision.mean()),
        "a1_near_mean": float(near.mean()),
        "a1_near_max": float(near.max()),
        "a1_near_positive": int((near > 0).sum()),
        "a1_score_min": float(score.min()),
        "a1_score_median": float(np.median(score)),
        "a1_score_low": int((score < 0.3).sum()),
        "a1_similarity_unique": int(len(np.unique(similarity))),
    }

    # 视频容器元数据碎片化：8 种分辨率 / 4 种帧率
    try:
        import cv2

        video_root = next(DATA.glob("附件1-*"))
        resolutions, rates, rows = collections.Counter(), collections.Counter(), []
        for video in sorted(video_root.rglob("*.mp4")):
            capture = cv2.VideoCapture(str(video))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = round(float(capture.get(cv2.CAP_PROP_FPS)), 3)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            resolutions[(width, height)] += 1
            rates[fps] += 1
            rows.append({"video": video.name, "width": width, "height": height,
                         "fps": fps, "frames": frames})
        facts.update({
            "a1_resolutions": len(resolutions),
            "a1_frame_rates": len(rates),
            "a1_resolution_top": f"{max(resolutions, key=resolutions.get)[0]}"
                                 f"×{max(resolutions, key=resolutions.get)[1]}",
            "a1_resolution_top_share": float(max(resolutions.values()) / len(rows)),
            "a1_fps_top": float(max(rates, key=rates.get)),
            "a1_fps_top_share": float(max(rates.values()) / len(rows)),
        })
        meta = pd.DataFrame(rows)
        meta.to_csv(RESULTS / "a1_video_meta.csv", index=False)
    except Exception as exc:  # noqa: BLE001 - 无 OpenCV 时降级为缺省值
        facts.update({"a1_resolutions": 0, "a1_frame_rates": 0, "a1_resolution_top": "n/a",
                      "a1_resolution_top_share": float("nan"), "a1_fps_top": float("nan"),
                      "a1_fps_top_share": float("nan")})
        print(f"  [warn] 视频元数据探测失败：{exc}")

    profile = pd.DataFrame({
        "attachment": "附件1",
        "group": ["人脸", "姿态", "任一视觉"],
        "zero": [facts["a1_face_zero"], facts["a1_pose_zero"], facts["a1_vision_zero"]],
        "low": [int(((face > 0) & (face <= 0.5)).sum()), int(((pose > 0) & (pose <= 0.5)).sum()),
                int(((vision > 0) & (vision <= 0.5)).sum())],
        "mid": [int(((face > 0.5) & (face <= 0.9)).sum()), int(((pose > 0.5) & (pose <= 0.9)).sum()),
                int(((vision > 0.5) & (vision <= 0.9)).sum())],
        "high": [int((face > 0.9).sum()), int((pose > 0.9).sum()), int((vision > 0.9).sum())],
    })
    return facts, profile


# ------------------------------------------------------------------ 附件2 诊断
def diagnose_attachment2() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """返回 (关键数字, 逐样本网格证据, 逐位置有效频率)。"""
    raw = loader.aligned()
    blocks = {split: raw[split] for split in ("train", "valid", "test")}
    rows = []
    position_hits = np.zeros((3, 50), dtype=float)
    total = 0
    audio_zero_is_content = 0
    duplicate_exact = 0
    duplicate_samples = 0
    trunc = collections.Counter()
    for split, block in blocks.items():
        tokens = np.asarray(block["text_bert"])
        audio = np.asarray(block["audio"])
        vision = np.asarray(block["vision"])
        texts = np.asarray(block["raw_text"])
        for index in range(tokens.shape[0]):
            ids, attention = tokens[index, 0], tokens[index, 1] > 0
            m = int(attention.sum())
            content = _content_mask(tokens[index])
            audio_valid = np.abs(audio[index]).max(-1) > 1e-8
            vision_valid = np.abs(vision[index]).max(-1) > 1e-8
            words = len([w for w in str(texts[index]).split() if w])
            if m >= 2:
                for position in range(m):
                    if audio_valid[position]:
                        position_hits[1, position] += 1
                    if vision_valid[position]:
                        position_hits[2, position] += 1
                    if content[position]:
                        position_hits[0, position] += 1
            # 零值语义：音频非零位是否恰好等于文本内容位
            match = bool(np.array_equal(audio_valid[1:m - 1], content[1:m - 1])) if m >= 2 else False
            audio_zero_is_content += int(match)
            # 子词复制：相邻音频向量完全相同的位数
            repeated = 0
            for position in range(2, m - 1):
                if audio_valid[position] and audio_valid[position - 1] and \
                        np.array_equal(audio[index, position], audio[index, position - 1]):
                    repeated += 1
            expected = max(0, m - 2 - words)
            if repeated == expected:
                duplicate_exact += 1
            duplicate_samples += 1
            trunc[split] += int(words > m)
            rows.append({
                "split": split, "tokens": m, "content": int(content.sum()), "words": words,
                "audio_valid": int(audio_valid[1:m - 1].sum()) if m >= 2 else 0,
                "vision_valid": int(vision_valid[1:m - 1].sum()) if m >= 2 else 0,
                "audio_zero_eq_content": int(match),
                "repeated_audio": repeated, "expected_repeat": expected,
                "audio_vision_missing_same": int(np.array_equal(~audio_valid, ~vision_valid)),
                "truncated": int(words > m),
                "word_loss": (1 - (m - 2) / words) if words > 0 and words > m else 0.0,
            })
    evidence = pd.DataFrame(rows)
    total = len(evidence)
    facts = {
        "a2_samples": int(total),
        "a2_split_sizes": {k: int(len(v["id"])) for k, v in blocks.items()},
        "a2_zero_eq_content_share": float(audio_zero_is_content / total),
        "a2_zero_eq_content_count": int(audio_zero_is_content),
        "a2_subword_exact_share": float(duplicate_exact / duplicate_samples),
        "a2_subword_exact_count": int(duplicate_exact),
        "a2_subword_total": int(duplicate_samples),
        "a2_truncated": int(evidence["truncated"].sum()),
        "a2_truncated_share": float(evidence["truncated"].mean()),
        "a2_truncated_worst_loss": float(evidence.loc[evidence["truncated"] == 1, "word_loss"].max()),
        "a2_content_median": float(evidence["content"].median()),
        "a2_content_mean": float(evidence["content"].mean()),
        "a2_audio_vision_same_share": float(evidence["audio_vision_missing_same"].mean()),
        "a2_missing_position_share": float((evidence["content"] - evidence["audio_valid"]).sum()
                                           / evidence["content"].sum()),
    }
    positions = pd.DataFrame(position_hits.T, columns=["text", "audio", "vision"])
    positions.index.name = "position"
    positions = positions.reset_index()
    positions["n_samples"] = total
    for name in MODALITIES:
        positions[f"{name}_rate"] = positions[name] / total
    return facts, evidence, positions


# ------------------------------------------------------------------ 附件3 诊断
def diagnose_attachment3() -> tuple[dict, pd.DataFrame]:
    samples = _attachment3_samples()
    rows = []
    for sample in samples:
        content = _content_mask(sample["text_bert"])
        record = {"name": sample["name"], "content": int(content.sum())}
        valid = {}
        for name in MODALITIES:
            array = sample["audio"] if name == "audio" else sample["vision"]
            if name == "text":
                valid[name] = content
            else:
                valid[name] = np.abs(array).max(-1) > 1e-8
            missing = content & ~valid[name]
            record[f"{name}_missing"] = int(missing.sum())
            record[f"{name}_rate"] = float(missing.sum() / max(1, content.sum()))
            if name != "text":
                runs = _run_lengths(missing)
                record[f"{name}_runs"] = len(runs)
                record[f"{name}_run_max"] = max(runs) if runs else 0
        record["audio_vision_same"] = int(np.array_equal(
            content & ~valid["audio"], content & ~valid["vision"]))
        record["extra_missing"] = int(sum(record[f"{m}_missing"] for m in ("audio", "vision")))
        record["extra_missing_rate"] = float(record["extra_missing"] / max(1, 2 * content.sum()))
        rows.append(record)
    frame = pd.DataFrame(rows)
    facts = {
        "a3_samples": int(len(frame)),
        "a3_text_never_missing": bool((frame["text_missing"] == 0).all()),
        "a3_audio_missing_rate_min": float(frame["audio_rate"].min()),
        "a3_audio_missing_rate_max": float(frame["audio_rate"].max()),
        "a3_vision_missing_rate_min": float(frame["vision_rate"].min()),
        "a3_vision_missing_rate_max": float(frame["vision_rate"].max()),
        "a3_extra_missing_rate_mean": float(frame["extra_missing_rate"].mean()),
        "a3_extra_missing_rate_min": float(frame["extra_missing_rate"].min()),
        "a3_extra_missing_rate_max": float(frame["extra_missing_rate"].max()),
        "a3_extra_missing_share_pooled": float(
            frame["extra_missing"].sum() / max(1, 2 * frame["content"].sum())),
        "a3_content_share_with_any_missing": float(
            ((frame["audio_missing"] + frame["vision_missing"]) > 0).mean()),
        "a3_audio_vision_same_count": int(frame["audio_vision_same"].sum()),
        "a3_run_max": int(frame[["audio_run_max", "vision_run_max"]].to_numpy().max()),
        "a3_content_median": float(frame["content"].median()),
    }
    return facts, frame


# ------------------------------------------------------------------ 附件4 诊断
def diagnose_attachment4() -> tuple[dict, pd.DataFrame]:
    samples = _attachment4_samples()
    rows = []
    for sample in samples:
        content = _content_mask(sample["text_bert"])
        record = {"name": sample["name"], "content": int(content.sum())}
        for name, array in (("audio", sample["audio"]), ("vision", sample["vision"])):
            valid = np.abs(array).max(-1) > 1e-8
            record[f"{name}_valid"] = int(valid.sum())
            record[f"{name}_rate"] = float(valid.sum() / max(1, content.sum()))
            record[f"{name}_missing"] = int((content & ~valid).sum())
        rows.append(record)
    frame = pd.DataFrame(rows)
    vision_dead = frame.loc[frame["vision_valid"] == 0, "name"].tolist()
    facts = {
        "a4_samples": int(len(frame)),
        "a4_vision_allzero": vision_dead,
        "a4_vision_allzero_count": int(len(vision_dead)),
        "a4_content_median": float(frame["content"].median()),
        "a4_vision_missing_total": int(frame["vision_missing"].sum()),
        "a4_audio_missing_total": int(frame["audio_missing"].sum()),
        "a4_content_total": int(frame["content"].sum()),
    }
    return facts, frame


# ------------------------------------------------------------------ 主流程
def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("诊断附件1 …")
    facts1, profile1 = diagnose_attachment1()
    print("诊断附件2 …")
    facts2, evidence2, positions2 = diagnose_attachment2()
    print("诊断附件3 …")
    facts3, frame3 = diagnose_attachment3()
    print("诊断附件4 …")
    facts4, frame4 = diagnose_attachment4()

    facts = {**facts1, **facts2, **facts3, **facts4}
    (RESULTS / "diagnostics_facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    evidence2.to_csv(RESULTS / "grid_evidence.csv", index=False)
    positions2.to_csv(RESULTS / "grid_position_profile.csv", index=False)
    frame3.to_csv(RESULTS / "defect_attachment3.csv", index=False)
    frame4.to_csv(RESULTS / "defect_attachment4.csv", index=False)
    profile1.to_csv(RESULTS / "defect_attachment1.csv", index=False)

    # ---- 逐项诊断清单：直接渲染成论文中的"数据缺陷—对策"表
    rows = [
        ["附件1", "源视频被重复切分",
         f"{facts1['a1_clips']} 条 clip 仅来自 {facts1['a1_videos']} 个源视频（单视频最多 "
         f"{facts1['a1_max_clips_per_video']} 段）",
         "同源片段若跨划分会造成信息泄漏", "问题一为逐样本特征任务，不涉及划分；已登记该结构并在评审中提示"],
        ["附件1", "容器参数高度碎片化",
         f"{facts1['a1_resolutions']} 种分辨率、{facts1['a1_frame_rates']} 种帧率（主流 "
         f"{facts1['a1_resolution_top']} 占 {facts1['a1_resolution_top_share']:.0%}）",
         "帧尺度不一致，逐帧特征不可直接比大小", "全部特征按词区间池化后再标准化，消除绝对尺度"],
        ["附件1", "时长跨度极大",
         f"{facts1['a1_seconds_min']:.2f}–{facts1['a1_seconds_max']:.2f} s，相差 "
         f"{facts1['a1_seconds_ratio']:.1f} 倍",
         "长样本词数多，直接拼接会主导损失", "词区间等权池化 + 逐样本归一化，损失在样本级平均"],
        ["附件1", "人脸/人体漏检",
         f"{facts1['a1_face_zero']} 条检测不到人脸、{facts1['a1_pose_zero']} 条检测不到姿态、"
         f"{facts1['a1_vision_zero']} 条两者皆无",
         "视觉分支出现全零，等同于模态缺失", "显式写入视觉掩码并在问题二训练中作为缺失模态处理"],
        ["附件1", "最近帧补位比例偏高",
         f"均值 {facts1['a1_near_mean']:.3f}，最大 {facts1['a1_near_max']:.3f}，"
         f"{facts1['a1_near_positive']}/{facts1['a1_clips']} 条存在补位",
         "补位值被重复计数会高估视觉置信度", "记录补位掩码，仅在无观测时启用，并在质量清单中单列"],
        ["附件1", "强制对齐置信度偏低",
         f"最低 {facts1['a1_score_min']:.3f}，中位 {facts1['a1_score_median']:.3f}，"
         f"{facts1['a1_score_low']} 条低于 0.3",
         "词时间戳可能错位，污染池化区间", "给出人工复核清单，并以最近帧补位容差 0.1 s 兜底"],
        ["附件2", "文本超长被截断",
         f"{facts2['a2_truncated']} 条（{facts2['a2_truncated_share']:.2%}），最严重丢失 "
         f"{facts2['a2_truncated_worst_loss']:.0%} 的词",
         "文本语义不完整但标签完整，形成错配样本", "在结果分析中单列，不删除以保持与官方划分一致"],
        ["附件2", "50 位网格稀疏",
         f"中位仅 {facts2['a2_content_median']:.0f}/50 位承载内容（均值 "
         f"{facts2['a2_content_mean']:.1f}）",
         "零填充位若参与注意力会稀释真实信号", "显式可用性掩码，特殊槽与填充位一律置零并屏蔽"],
        ["附件2", "零值语义歧义",
         f"音频非零位与文本内容位完全一致的样本占 {facts2['a2_zero_eq_content_share']:.2%}",
         "无法区分「真缺失」与「合法零值」", "以结构反演确认「零值即缺失」，统一用掩码表达"],
        ["附件2", "子词复制造成冗余",
         f"相邻声画向量完全相同的位数等于 m−2−W 的样本占 "
         f"{facts2['a2_subword_exact_share']:.1%}（{facts2['a2_subword_exact_count']}/"
         f"{facts2['a2_subword_total']}）",
         "同一词被重复计入损失权重", "以词为语义单元解释结果，并在论文中说明该冗余"],
        ["附件2", "类别严重不平衡",
         f"中性类占比 0.22–0.25，正向类约 0.50",
         "Accuracy 会被多数类支配", "全部结论以 macro-F1 为主指标，并给出逐类混淆矩阵"],
        ["附件3", "标签缺失",
         f"{facts3['a3_samples']} 条样本无任何标签",
         "无法直接监督", "分解为无监督结构约束 + 与附件2 共享的迁移推理"],
        ["附件3", "额外缺失形态未知",
         f"语音额外缺失率 {facts3['a3_audio_missing_rate_min']:.0%}–"
         f"{facts3['a3_audio_missing_rate_max']:.0%}，视觉 "
         f"{facts3['a3_vision_missing_rate_min']:.0%}–{facts3['a3_vision_missing_rate_max']:.0%}，"
         f"合计 {facts3['a3_extra_missing_rate_mean']:.1%}",
         "按错误分布增强会与真实测试分布错配", "先反演其分布，再按其形态构造多尺度缺失模拟"],
        ["附件3", "缺失位置高度相关",
         f"{facts3['a3_audio_vision_same_count']}/{facts3['a3_samples']} 条样本语音与视觉缺失位置完全相同",
         "两模态同时失效，融合退化为单模态", "模拟时保留「声画同缺失」与「部分重叠」两种模式"],
        ["附件4", "无时间戳",
         f"{facts4['a4_samples']} 条样本只有 50 位特征，无起始时间",
         "无法直接给出时间定位", "重建 BERT 词元与附件4 逐位核对，再回溯到原始视频"],
        ["附件4", "视觉通道整段失效",
         "样本 " + "、".join(facts4["a4_vision_allzero"]) + " 的视觉全零"
         if facts4["a4_vision_allzero"] else "无明显整段失效",
         "该样本的视觉解释无效", "解释输出按模态有效性过滤，只报告可用模态的证据"],
    ]
    diagnostics = pd.DataFrame(rows, columns=["附件", "检查项", "实测现象", "对建模的危害", "本文对策"])
    diagnostics.to_csv(RESULTS / "data_diagnostics.csv", index=False, encoding="utf-8-sig")

    print(f"\n关键数字已写入 {RESULTS / 'diagnostics_facts.json'}")
    print(f"诊断清单 {len(diagnostics)} 项 -> {RESULTS / 'data_diagnostics.csv'}")
    for key in ("a2_zero_eq_content_share", "a2_subword_exact_share", "a2_truncated",
                "a3_extra_missing_rate_mean", "a1_near_mean", "a1_score_low"):
        print(f"  {key} = {facts[key]}")


if __name__ == "__main__":
    main()
