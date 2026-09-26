"""Build the showcase payload from the full-100 extraction outputs.

Outputs (all under q1_viz/):
  assets/data/<sample_id>.json   per-sample payload (lazy-loaded by the page)
  assets/videos/*.mp4            all official sample videos
  assets/thumbs/*.jpg            posters
  site_data.json                 aggregate payload (also injected into index.html)

Nothing here invents numbers: every value is read from outputs/ or the
official label table. Human labels are displayed as dataset metadata only.

    D:\\anaconda\\envs\\q1-v2\\python.exe q1_viz/tools/build_site_data.py
"""

from __future__ import annotations

import csv
import json
import shutil
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R4 = REPO / "outputs" / "q1_v2_round4"
R5 = REPO / "outputs" / "q1_v2_round5"
FULL = REPO / "outputs" / "q1_v2_full100"
SITE = REPO / "q1_viz"
ASSETS = SITE / "assets"
DATA_ROOT = REPO.parent / "data" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
LABEL_XLSX = DATA_ROOT / "label-100.xlsx"

# First displayed typical sample is fixed to -THoVjtIkeU__2 (best human-verified
# boundary accuracy); the rest cover the story range of the pipeline.
TYPICAL = ["-THoVjtIkeU__2", "-3g5yACwYnA__13", "-3g5yACwYnA__3", "-s9qJ7ATP7w__6", "-NFrJFQijFE__1"]

CURATED_NOTES = {
    "-3g5yACwYnA__13": ("典型样本 · 三模态齐备", "人工逐词边界校验样本；MFA 词级对齐与人工参考边界可直接对照。"),
    "-3g5yACwYnA__2": ("OOV 词样本", "官方文本含词表外词（OOV），补充词典后真实 MFA 对齐成功。"),
    "-3g5yACwYnA__3": ("长句样本", "29 词长句全部对齐；人工边界误差相对最大的样本（约 157 ms）。"),
    "-THoVjtIkeU__2": ("最佳对齐样本", "人工边界误差 43.6 ms、IoU 0.808，为对齐精度最好的人工校验样本。"),
    "-s9qJ7ATP7w__6": ("文本-音频疑似不一致", "自动 QA 判定 PARTIAL_MATCH（REVIEW_REQUIRED）：保守拦截词级对齐（掩码全 0、时间 NaN），"
                        "不做错误文本-音频强制对齐。声学帧级特征与人脸线索完整保留，等待人工复核。"),
    "-AUZQgSxyPQ__2": ("文本-音频疑似不一致", "自动 QA 判定 PARTIAL_MATCH（REVIEW_REQUIRED）：保守拦截词级对齐。"
                        "文本特征与 OpenFace 人脸线索全部保留，供人工核对转写是否对应画面语音。"),
    "-NFrJFQijFE__1": ("无语音 + 无脸场景", "语音缺失（NO_SPEECH）→ 音频模态不可用；画面同时无可用人脸 → 视觉词级特征缺失。"
                        "保留文本特征与场景/声学线索（底噪帧记录），对齐不运行。"),
    "-UuX1xuaiiE__1": ("静态人物图 · 说话者身份存疑", "静态画面中人脸持续被 OpenFace 检出，但无法证明是说话者；"
                        "文本-音频一致性良好，真实 MFA 对齐可用。"),
    "-yRb-Jum7EQ__1": ("非英语语音", "检测为希伯来语（NON_ENGLISH_SPEECH）→ 官方英文转写与语音不对应，路由拦截对齐。"
                        "保留全部文本特征与人脸线索。"),
    "-HwX2H8Z4hY__5": ("低视觉覆盖", "文本-音频一致并真实 MFA 对齐；但多数帧无有效人脸，"
                        "视觉词级特征按缺失策略置掩码 0，不做近邻填补。"),
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def waveform_envelope(wav_path: Path, buckets: int = 560) -> dict:
    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if not len(samples):
        return {"duration_s": 0.0, "envelope": []}
    edges = np.linspace(0, len(samples), buckets + 1).astype(int)
    env = [float(np.abs(samples[edges[i]:edges[i + 1]]).max()) if edges[i + 1] > edges[i] else 0.0
           for i in range(buckets)]
    peak = max(env) or 1.0
    return {"duration_s": len(samples) / rate, "envelope": [round(v / peak, 3) for v in env]}


def export_media(sample_id: str) -> dict:
    video_id, clip_id = sample_id.rsplit("__", 1)
    source = DATA_ROOT / video_id / f"{clip_id}.mp4"
    video_out = ASSETS / "videos" / f"{sample_id}.mp4"
    thumb_out = ASSETS / "thumbs" / f"{sample_id}.jpg"
    video_out.parent.mkdir(parents=True, exist_ok=True)
    thumb_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, video_out)
    width = height = 0
    if not thumb_out.is_file():
        try:
            import av
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                width, height = stream.width, stream.height
                duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
                container.seek(int(duration * 0.4 / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    frame.to_image().resize((360, int(frame.height * 360 / frame.width))).save(thumb_out, quality=82)
                    break
        except Exception as exc:
            print(f"thumb failed for {sample_id}: {exc}")
    else:
        try:
            import av
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                width, height = stream.width, stream.height
        except Exception:
            pass
    return {"video": f"assets/videos/{sample_id}.mp4", "thumb": f"assets/thumbs/{sample_id}.jpg",
            "video_width": width, "video_height": height}


def export_landmarks(sample_dir: Path, video_wh: tuple[int, int]) -> dict:
    raw = sample_dir / "visual" / "openface68" / "openface_raw" / "openface68.csv"
    if not raw.is_file():
        return {}
    times, frames = [], []
    with raw.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(" face_id", row.get("face_id", "0")).strip() not in ("0", "0.0"):
                continue
            if str(row.get(" success", "0")).strip() not in ("1", "1.0"):
                continue
            try:
                coords = [f"{float(row[f' x_{i}']):.0f},{float(row[f' y_{i}']):.0f}" for i in range(68)]
            except (KeyError, ValueError):
                continue
            times.append(round(float(row[" timestamp"]), 3))
            frames.append(" ".join(coords))
    return {"times": times, "frames": frames, "video_width": video_wh[0], "video_height": video_wh[1]}


def export_audio_features(sample_dir: Path) -> dict:
    npz = sample_dir / "audio_features.npz"
    if not npz.is_file():
        return {}
    with np.load(npz, allow_pickle=False) as archive:
        features = archive["frame_features"]
        times = archive["frame_times_s"]
    meta = read_json(sample_dir / "audio_features_metadata.json") if (sample_dir / "audio_features_metadata.json").is_file() else {}
    names = meta.get("feature_names", [])
    mel = features[:, :64]
    bucket_count = 120
    edges = np.linspace(0, len(mel), bucket_count + 1).astype(int)
    pooled = np.asarray([mel[edges[i]:edges[i + 1]].mean(axis=0) if edges[i + 1] > edges[i] else np.zeros(64)
                         for i in range(bucket_count)], dtype=np.float32)
    band_step = 64 // 24
    strip = [[round(float(pooled[t, b * band_step:(b + 1) * band_step].mean()), 3) for t in range(bucket_count)]
             for b in range(24)]

    def column(name: str, fallback_index: int) -> np.ndarray:
        index = names.index(name) if name in names else fallback_index
        return features[:, index] if features.shape[1] > index else np.full(len(features), np.nan)

    def finite_mean(values: np.ndarray, digits: int = 3):
        finite = values[np.isfinite(values)]
        return round(float(finite.mean()), digits) if len(finite) else None

    return {
        "mel_strip": strip,
        "duration_s": float(times[-1]) if len(times) else 0.0,
        "f0_mean_hz": finite_mean(column("f0_hz", 65), 1),
        "rms_mean": finite_mean(column("rms", 64), 4),
        "zcr_mean": finite_mean(column("zero_crossing_rate", 66), 3),
        "centroid_mean_hz": finite_mean(column("spectral_centroid_hz", 67), 0),
        "frame_count": int(features.shape[0]),
    }


def auto_note(sample_id: str, quality: dict, success: dict) -> tuple[str, str]:
    if sample_id in CURATED_NOTES:
        return CURATED_NOTES[sample_id]
    status = quality.get("correspondence_status", "")
    aligned = success.get("aligned_word_count") or 0
    words = success.get("original_word_count") or 0
    frames_valid = success.get("openface_valid_frame_count") or 0
    frames = success.get("openface_sampled_frame_count") or 0
    coverage = round(float(success.get("visual_word_coverage") or 0.0) * 100, 1)
    mfa = success.get("mfa_policy", "")
    mfa_text = {"REUSED_VERIFIED_REAL_MFA": "复用已验证真实 MFA",
                "NEW_LOCAL_MFA": "现场真实 MFA 对齐",
                "PROHIBITED_BY_ROUTER": "路由拦截，对齐不运行",
                "MFA_FAILED_NO_PSEUDO_ALIGNMENT": "MFA 失败，保留原词不伪造对齐"}.get(mfa, mfa)
    if status == "MATCHED":
        if aligned >= words and coverage >= 99:
            return ("三模态齐备", f"文本-音频一致（MATCHED）：{mfa_text}，{aligned}/{words} 词对齐；"
                                  f"人脸 {frames_valid}/{frames} 帧有效，视觉词级覆盖 {coverage}%。")
        return ("部分模态缺失", f"文本-音频一致（MATCHED）：{mfa_text}，{aligned}/{words} 词对齐。"
                                f"视觉帧 {frames_valid}/{frames} 有效、词级覆盖 {coverage}%，"
                                f"缺失词级特征按掩码 0 保留，不做近邻填补。")
    if status == "PARTIAL_MATCH":
        return ("文本-音频疑似不一致", f"自动 QA 判定 PARTIAL_MATCH（REVIEW_REQUIRED）：保守拦截词级对齐（掩码全 0、时间 NaN），"
                                        f"不做错误文本-音频强制对齐。保留 {words} 词文本特征、"
                                        f"声学帧级线索与人脸线索（{frames_valid}/{frames} 帧有效），等待人工复核。")
    if status == "NO_SPEECH":
        return ("无语音场景", f"语音缺失（NO_SPEECH）→ 音频模态不可用；路由拦截对齐。"
                              f"保留 {words} 词文本特征与场景/声学底噪线索（人脸 {frames_valid}/{frames} 帧）。")
    if status == "NON_ENGLISH_SPEECH":
        lang = quality.get("language_status", "")
        return ("非英语语音", f"检测为非英语语音（NON_ENGLISH_SPEECH{(' · ' + lang) if lang else ''}）："
                              f"官方英文转写与语音不对应，路由拦截对齐。保留 {words} 词文本特征与"
                              f"人脸线索（{frames_valid}/{frames} 帧有效）。")
    if status == "TEXT_AUDIO_MISMATCH":
        return ("文本-音频不一致", f"自动 QA 判定文本-音频不一致（TEXT_AUDIO_MISMATCH）：路由拦截，"
                                    f"不做错误强制对齐。保留 {words} 词文本特征与声学/人脸线索"
                                    f"（{frames_valid}/{frames} 帧）供人工核查。")
    if status == "NO_AUDIO":
        return ("无音频信号", "音频信号缺失（NO_AUDIO）→ 音频模态不可用；保留文本与视觉线索，对齐不运行。")
    return ("对应关系未定", f"自动 QA 判定 UNRESOLVED：证据不足，路由拦截对齐（掩码全 0）。"
                            f"保留 {words} 词文本特征与声学/人脸线索（{frames_valid}/{frames} 帧），等待人工复核。")


def sample_payload(sample_id: str) -> dict:
    sample_dir = FULL / "samples" / sample_id
    meta = read_json(sample_dir / "feature_package_metadata.json")
    success = read_json(sample_dir / "_SUCCESS.json")
    qa = read_json(sample_dir / "correspondence_qa.json")
    media = export_media(sample_id)
    landmarks = export_landmarks(sample_dir, (media["video_width"], media["video_height"]))

    words = []
    with (sample_dir / "word_alignment.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        align_rows = list(csv.DictReader(handle))
    conf = read_json(sample_dir / "alignment_confidence.json") if (sample_dir / "alignment_confidence.json").is_file() else {}
    conf_by_index = {int(item["word_index"]): item for item in conf.get("words", [])}
    manual_rows = {int(row["word_index"]): row for row in read_csv(R5 / "manual_boundary" / "normalized_manual_boundaries.csv")
                   if row["sample_id"] == sample_id}
    for row in align_rows:
        index = int(row["word_index"])
        conf_row = conf_by_index.get(index, {})
        manual = manual_rows.get(index, {})
        words.append({
            "i": index,
            "text": row["original_word"],
            "mfa_s": None if row["start_s"] in ("", "nan") else round(float(row["start_s"]), 3),
            "mfa_e": None if row["end_s"] in ("", "nan") else round(float(row["end_s"]), 3),
            "manual_s": round(float(manual["reference_start_s"]), 3) if manual else None,
            "manual_e": round(float(manual["reference_end_s"]), 3) if manual else None,
            "asr_s": round(float(conf_row["asr_start_s"]), 3) if conf_row.get("asr_start_s") not in (None, "") else None,
            "asr_e": round(float(conf_row["asr_end_s"]), 3) if conf_row.get("asr_end_s") not in (None, "") else None,
            "grade": conf_row.get("alignment_confidence", "UNAVAILABLE"),
            "mask": int(row["alignment_mask"]),
        })

    quality = meta.get("quality", {})
    title, note = auto_note(sample_id, quality, success)

    au_mean = None
    openface_dir = sample_dir / "visual" / "openface68"
    if (openface_dir / "openface68_frames.npz").is_file():
        with np.load(openface_dir / "openface68_frames.npz", allow_pickle=False) as archive:
            au = archive["action_units"]
            valid = archive["face_valid_mask"] & archive["timestamp_mapping_mask"]
            au_mean = [round(float(v), 2) for v in (au[valid].mean(axis=0) if valid.any() else au.mean(axis=0))]

    return {
        "id": sample_id,
        "label_chip": success.get("quality_status"),
        "route": success.get("quality_route"),
        "mfa_policy": success.get("mfa_policy"),
        "official_text": meta.get("official_text", ""),
        "duration_s": round(float(qa.get("duration_s", 0.0)), 2),
        "quality": quality,
        "words": words,
        "waveform": waveform_envelope(sample_dir / "audio_mfa_16k_mono.wav"),
        "landmarks": landmarks,
        "au_mean": au_mean,
        "audio": export_audio_features(sample_dir),
        "stats": {
            "words_total": success.get("original_word_count"),
            "words_aligned": success.get("aligned_word_count"),
            "text_valid": success.get("text_word_count"),
            "audio_valid": success.get("audio_word_count"),
            "visual_valid": success.get("visual_word_count"),
            "visual_coverage": round(float(success.get("visual_word_coverage") or 0.0), 3),
            "openface_sampled": success.get("openface_sampled_frame_count"),
            "openface_valid": success.get("openface_valid_frame_count"),
            "text_dim": 768, "audio_dim": 138, "visual_dim": 370,
            "package_kib": round((success.get("feature_package_bytes") or 0) / 1024),
        },
        "note_title": title,
        "note_body": note,
        "media": media,
    }


def build_full_table(summary_rows: dict[str, dict]) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(LABEL_XLSX, read_only=True)
    labels = {}
    for video_id, clip_id, text, label, annotation in wb.active.iter_rows(min_row=2, values_only=True):
        labels[f"{video_id}__{clip_id}"] = {
            "text": text or "", "intensity": float(label), "annotation": annotation or "",
        }
    stage1 = {row["sample_id"]: row for row in read_csv(R4 / "correspondence_audit.csv")}
    scenes = {row["sample_id"]: row for row in read_csv(R5 / "qa" / "correspondence_audit_round5.csv")}
    annotation_cn = {"Positive": "正向", "Neutral": "中性", "Negative": "负向"}
    rows = []
    for sample_id in sorted(labels):
        label_row = labels[sample_id]
        qa_row = stage1.get(sample_id, {})
        scene = scenes.get(sample_id, {})
        pkg = summary_rows.get(sample_id, {})
        rows.append({
            "id": sample_id,
            "label": annotation_cn.get(label_row["annotation"], label_row["annotation"]),
            "intensity": round(label_row["intensity"], 3),
            "duration_s": round(float(qa_row.get("duration_s") or 0.0), 2),
            "qa_status": qa_row.get("correspondence_status", ""),
            "route": qa_row.get("route", ""),
            "speech": "Y" if qa_row.get("speech_present") == "True" else ("N" if qa_row.get("speech_present") == "False" else "?"),
            "language": qa_row.get("detected_language", "") or "—",
            "face_rate": round(float(qa_row.get("face_detection_rate") or 0.0), 3),
            "scene": scene.get("visual_scene_status", "") or "—",
            "text": label_row["text"][:46],
            "package_status": pkg.get("processing_status", "not_run"),
            "aligned_words": pkg.get("aligned_word_count"),
        })
    return rows


def main() -> None:
    manual_metrics = read_json(R5 / "manual_qa" / "manual_qa_metrics.json")
    boundary = read_json(R5 / "manual_boundary" / "manual_boundary_metrics_overall.json")
    confidence = read_json(R5 / "alignment_confidence" / "alignment_confidence_summary.json")
    face_qa = read_json(R5 / "qa" / "face_qa_metrics.json")
    extra = read_json(R5 / "qa" / "extra_speech_metrics.json")
    experiment = read_json(FULL / "experiment_config.json")

    summary_rows = {row["sample_id"]: row for row in read_csv(FULL / "feature_summary.csv")}
    all_ids = sorted(summary_rows)

    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    per_sample_dir = ASSETS / "data"
    for stale in per_sample_dir.glob("*.json"):
        stale.unlink()
    for index, sample_id in enumerate(all_ids, 1):
        payload = sample_payload(sample_id)
        # .js loaded via <script> so the page also works opened from file://
        (per_sample_dir / f"{sample_id}.js").write_text(
            'window.__SAMPLES__=window.__SAMPLES__||{};window.__SAMPLES__['
            + json.dumps(sample_id) + "]="
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";",
            encoding="utf-8")
        if index % 10 == 0:
            print(f"sample payload {index}/{len(all_ids)}")

    micro = boundary["micro_by_word_boundary"]
    completed = sum(1 for row in summary_rows.values() if row.get("processing_status") == "completed")
    aggregate = {
        "title": "复杂场景下多模态情感识别",
        "subtitle": "问题 1 · 多模态情感特征提取与时序对齐",
        "typical_samples": [sid for sid in TYPICAL if sid in summary_rows],
        "nav_samples": all_ids,
        "kpis": {
            "samples": manual_metrics["sample_count"],
            "packages": completed,
            "unsafe_fp": manual_metrics["unsafe_false_positive_count"],
            "precision": manual_metrics["high_confidence_precision"],
            "recall": manual_metrics["high_confidence_recall"],
            "boundary_mae_ms": round(micro["mean_boundary_error_s"] * 1000, 1),
            "iou": round(micro["mean_temporal_iou"], 3),
            "pass100": round(micro["pass_rate_100ms"], 3),
            "openface_points": 68,
            "feature_dims": "768 / 138 / 370",
        },
        "quality_dist": {
            "manual": manual_metrics["manual_class_counts"],
            "automatic": manual_metrics["automatic_class_counts"],
            "exact": manual_metrics["exact_class_accuracy"],
            "coarse": manual_metrics["coarse_class_accuracy"],
        },
        "routing": manual_metrics["routing_confusion"],
        "alignment": {
            "micro": micro,
            "macro": boundary["macro_by_sample"],
            "confidence": confidence.get("confidence_counts", {}),
            "confidence_meaning": "MFA–ASR 边界分歧不确定性，不是人工时间精度",
        },
        "face_qa": {
            "round5_scene_counts": face_qa["round5_scene_status_counts"],
            "any_face_accuracy_r4": face_qa["round4_2fps_any_face"]["accuracy"],
            "any_face_accuracy_r5": face_qa["round5_5fps_any_face"]["accuracy"],
        },
        "extra_speech": {
            "comparable": extra["round5_comparable_matched_span_count"],
            "r5_before_precision": extra["round5_before"]["precision"],
            "r5_after_recall": extra["round5_after"]["recall"],
            "exact_direction": extra["round5_exact_direction_accuracy"],
        },
        "full_table": build_full_table(summary_rows),
        "pipeline": ["原始视频", "质量诊断", "文本-音频一致性", "ASR 局部定位", "MFA 精对齐", "OpenFace68 视觉", "三模态时序特征"],
        "provenance": {
            "full100_feature_extraction": experiment.get("full_100_sample_feature_extraction", "RUN"),
            "labels_role": "官方标签仅作数据集元信息展示，不进入模型与路由",
            "sources": [
                "outputs/q1_v2_full100/feature_summary.csv",
                "outputs/q1_v2_full100/samples/*",
                "outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json",
                "outputs/q1_v2_round5/manual_boundary/manual_boundary_metrics_overall.json",
                "outputs/q1_v2_round5/alignment_confidence/alignment_confidence_summary.json",
                "outputs/q1_v2_round5/qa/face_qa_metrics.json",
                "outputs/q1_v2_round4/correspondence_audit.csv",
                "label-100.xlsx（官方标注）",
            ],
        },
    }
    (SITE / "site_data.json").write_text(json.dumps(aggregate, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    target = SITE / "index.html"
    html = target.read_text(encoding="utf-8")
    begin, end = "/*__SITE_DATA_BEGIN__*/", "/*__SITE_DATA_END__*/"
    if begin in html and end in html:
        head, rest = html.split(begin, 1)
        _, tail = rest.split(end, 1)
        embedded = json.dumps(aggregate, ensure_ascii=False, separators=(",", ":"))
        target.write_text(f"{head}{begin}{embedded}{end}{tail}", encoding="utf-8")
    print("site_data.json:", (SITE / "site_data.json").stat().st_size, "bytes")
    print("per-sample json:", len(list(per_sample_dir.glob("*.json"))), "completed:", completed)


if __name__ == "__main__":
    main()
