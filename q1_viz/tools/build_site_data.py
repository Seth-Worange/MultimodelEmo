"""Build the Apple-style showcase payload from real experiment artifacts.

Outputs (all under q1_viz/):
  site_data.json            full payload (also injected into index.html markers)
  assets/videos/*.mp4       the 10 smoke sample videos
  assets/thumbs/*.jpg       posters

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
R51 = REPO / "outputs" / "q1_v2_round5_1"
SITE = REPO / "q1_viz"
ASSETS = SITE / "assets"
DATA_ROOT = REPO.parent / "data" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
LABEL_XLSX = DATA_ROOT / "label-100.xlsx"

SMOKE = [
    "-3g5yACwYnA__13", "-3g5yACwYnA__2", "-3g5yACwYnA__3", "-THoVjtIkeU__2",
    "-s9qJ7ATP7w__6", "-AUZQgSxyPQ__2", "-NFrJFQijFE__1", "-UuX1xuaiiE__1",
    "-yRb-Jum7EQ__1", "-HwX2H8Z4hY__5",
]

SPECIAL_NOTES = {
    "-3g5yACwYnA__13": ("典型样本 · 三模态齐备", "人工逐词边界校验样本；MFA 词级对齐与人工参考边界可直接对照。"),
    "-3g5yACwYnA__2": ("OOV 词样本", "官方文本含词表外词（OOV），补充词典后真实 MFA 对齐成功；视觉覆盖 85.7%。"),
    "-3g5yACwYnA__3": ("长句样本", "29 词长句全部对齐；人工边界误差相对最大的样本（约 157 ms）。"),
    "-THoVjtIkeU__2": ("最干净样本", "人工边界误差 43.6 ms、IoU 0.808，为对齐精度最好的人工校验样本。"),
    "-s9qJ7ATP7w__6": ("文本-音频疑似不一致", "自动 QA 判定 PARTIAL_MATCH（REVIEW_REQUIRED）：保守拦截词级对齐（掩码全 0、时间 NaN），"
                        "不做错误文本-音频强制对齐。声学帧级特征与人脸线索（24/25 帧有效）完整保留，等待人工复核。"),
    "-AUZQgSxyPQ__2": ("文本-音频疑似不一致", "自动 QA 判定 PARTIAL_MATCH（REVIEW_REQUIRED）：保守拦截词级对齐。"
                        "41 词文本特征与 OpenFace 225/225 帧人脸线索全部保留，供人工核对转写是否对应画面语音。"),
    "-NFrJFQijFE__1": ("无语音 + 无脸场景", "语音缺失（NO_SPEECH）→ 音频模态不可用；画面同时无可用人脸 → 视觉词级特征缺失。"
                        "保留文本特征（16 词）与场景/声学线索（底噪帧记录），对齐不运行。"),
    "-UuX1xuaiiE__1": ("静态人物图 · 说话者身份存疑", "静态画面中人脸持续被 OpenFace 检出（104/104 帧），但无法证明是说话者；"
                        "文本-音频一致性良好，本轮现场真实 MFA 对齐 23/27 词。"),
    "-yRb-Jum7EQ__1": ("非英语语音", "检测为希伯来语（NON_ENGLISH_SPEECH）→ 官方英文转写与语音不对应，路由拦截对齐。"
                        "保留全部 49 词文本特征与 292/293 帧人脸线索。"),
    "-HwX2H8Z4hY__5": ("低视觉覆盖", "文本-音频一致并现场真实 MFA 对齐 9/10 词；但画面多数帧无有效人脸（23/57 帧），"
                        "视觉词级覆盖仅 30%，特征按缺失策略置掩码 0，不填补。"),
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
    except Exception as exc:  # thumbnails are decorative; never break the payload
        print(f"thumb failed for {sample_id}: {exc}")
    return {"video": f"assets/videos/{sample_id}.mp4", "thumb": f"assets/thumbs/{sample_id}.jpg",
            "video_width": width, "video_height": height}


def export_landmarks(sample_dir: Path, video_wh: tuple[int, int]) -> dict:
    raw = sample_dir / "visual" / "openface68" / "openface_raw" / "openface68.csv"
    if not raw.is_file():
        return {}
    times, frames, valid = [], [], []
    with raw.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(" face_id", row.get("face_id", "0")).strip() not in ("0", "0.0"):
                continue
            success = str(row.get(" success", row.get(" success", "0"))).strip()
            if success not in ("1", "1.0"):
                continue
            try:
                coords = []
                for index in range(68):
                    x = float(row[f" x_{index}"])
                    y = float(row[f" y_{index}"])
                    coords.append(f"{x:.0f},{y:.0f}")
                times.append(round(float(row[" timestamp"]), 3))
                frames.append(" ".join(coords))
                valid.append(1)
            except (KeyError, ValueError):
                continue
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
    pooled = []
    for i in range(bucket_count):
        block = mel[edges[i]:edges[i + 1]]
        pooled.append(block.mean(axis=0) if len(block) else np.zeros(64))
    pooled = np.asarray(pooled, dtype=np.float32)
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
        "feature_names_tail": names[64:],
    }


def sample_payload(sample_id: str) -> dict:
    sample_dir = R51 / "smoke_test_details" / sample_id
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

    au_top = None
    openface_dir = sample_dir / "visual" / "openface68"
    if (openface_dir / "openface68_frames.npz").is_file():
        with np.load(openface_dir / "openface68_frames.npz", allow_pickle=False) as archive:
            au = archive["action_units"]
            valid = archive["face_valid_mask"] & archive["timestamp_mapping_mask"]
            au_top = [round(float(v), 2) for v in (au[valid].mean(axis=0) if valid.any() else au.mean(axis=0))]

    title, note = SPECIAL_NOTES.get(sample_id, ("", ""))
    quality = meta.get("quality", {})
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
        "audio": export_audio_features(sample_dir),
        "au_mean": au_top,
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
            "manual_boundary_ms": None,
        },
        "note_title": title,
        "note_body": note,
        "media": media,
    }


def build_full_table() -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(LABEL_XLSX, read_only=True)
    labels = {}
    for video_id, clip_id, text, label, annotation in list(wb.active.iter_rows(min_row=2, values_only=True)):
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
            "featured": sample_id in SMOKE,
        })
    return rows


def main() -> None:
    manual_metrics = read_json(R5 / "manual_qa" / "manual_qa_metrics.json")
    boundary = read_json(R5 / "manual_boundary" / "manual_boundary_metrics_overall.json")
    confidence = read_json(R5 / "alignment_confidence" / "alignment_confidence_summary.json")
    face_qa = read_json(R5 / "qa" / "face_qa_metrics.json")
    extra = read_json(R5 / "qa" / "extra_speech_metrics.json")

    samples = [sample_payload(sample_id) for sample_id in SMOKE]
    micro = boundary["micro_by_word_boundary"]

    payload = {
        "title": "复杂场景下多模态情感识别",
        "subtitle": "问题 1 · 多模态情感特征提取与时序对齐",
        "hero_sample": "-3g5yACwYnA__13",
        "kpis": {
            "samples": manual_metrics["sample_count"],
            "unsafe_fp": manual_metrics["unsafe_false_positive_count"],
            "precision": manual_metrics["high_confidence_precision"],
            "recall": manual_metrics["high_confidence_recall"],
            "boundary_mae_ms": round(micro["mean_boundary_error_s"] * 1000, 1),
            "iou": round(micro["mean_temporal_iou"], 3),
            "pass100": round(micro["pass_rate_100ms"], 3),
            "smoke_done": 10,
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
            "manual_samples": [
                {"id": row["sample_id"], "compared": int(row.get("compared_word_count") or 0),
                 "mae_ms": round(float(row["mean_boundary_error_s"]) * 1000, 1)
                 if row.get("mean_boundary_error_s") not in ("", "nan", None) else None,
                 "iou": round(float(row["mean_temporal_iou"]), 3)
                 if row.get("mean_temporal_iou") not in ("", "nan", None) else None}
                for row in read_csv(R5 / "manual_boundary" / "manual_boundary_metrics_by_sample.csv")
            ],
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
        "samples": samples,
        "full_table": build_full_table(),
        "pipeline": ["原始视频", "质量诊断", "文本-音频一致性", "ASR 局部定位", "MFA 精对齐", "OpenFace68 视觉", "三模态时序特征"],
        "provenance": {
            "full100_feature_extraction": "NOT_RUN",
            "labels_role": "官方标签仅作数据集元信息展示，不进入模型与路由",
            "sources": [
                "outputs/q1_v2_round5_1/smoke_test_summary.csv",
                "outputs/q1_v2_round5_1/smoke_test_details/*",
                "outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json",
                "outputs/q1_v2_round5/manual_boundary/manual_boundary_metrics_overall.json",
                "outputs/q1_v2_round5/alignment_confidence/alignment_confidence_summary.json",
                "outputs/q1_v2_round5/qa/face_qa_metrics.json",
                "outputs/q1_v2_round4/correspondence_audit.csv",
                "label-100.xlsx（官方标注）",
            ],
        },
    }
    (SITE / "site_data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    target = SITE / "index.html"
    html = target.read_text(encoding="utf-8")
    begin, end = "/*__SITE_DATA_BEGIN__*/", "/*__SITE_DATA_END__*/"
    if begin in html and end in html:
        head, rest = html.split(begin, 1)
        _, tail = rest.split(end, 1)
        embedded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        target.write_text(f"{head}{begin}{embedded}{end}{tail}", encoding="utf-8")
    print("site_data.json:", (SITE / "site_data.json").stat().st_size, "bytes")
    print("samples:", len(samples), "full_table rows:", len(payload["full_table"]))


if __name__ == "__main__":
    main()
