"""Aggregate validated Q1 outputs into q1_viz/public/frontend_data.json.

Reads only experiment artifacts under outputs/ (never human labels as model
inputs; human QA files are evaluation evidence shown on the page as-is).
Regenerate after each pipeline run:

    D:\\anaconda\\envs\\q1-v2\\python.exe q1_viz/tools/build_frontend_data.py
"""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R5 = REPO / "outputs" / "q1_v2_round5"
R51 = REPO / "outputs" / "q1_v2_round5_1"
PUBLIC = REPO / "q1_viz" / "public"
DATA_ROOT = REPO.parent / "data" / "附件1-数据集原始多模态样本" / "MOSEI数据集部分原始视频-100条"
TIMELINE_SAMPLE = "-3g5yACwYnA__13"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def waveform_envelope(wav_path: Path, buckets: int = 480) -> dict:
    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    duration_s = len(samples) / rate if rate else 0.0
    if not len(samples):
        return {"duration_s": 0.0, "sample_rate": rate, "envelope": []}
    edges = np.linspace(0, len(samples), buckets + 1).astype(int)
    envelope = [float(np.abs(samples[edges[i]:edges[i + 1]]).max()) if edges[i + 1] > edges[i] else 0.0
                for i in range(buckets)]
    peak = max(envelope) or 1.0
    return {"duration_s": duration_s, "sample_rate": rate,
            "envelope": [round(value / peak, 4) for value in envelope]}


def extract_thumbnails(video_path: Path, out_dir: Path, times_s: list[float],
                       width: int = 160) -> list[str]:
    try:
        import av
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        targets = [t for t in times_s if duration == 0.0 or t <= duration]
        for index, target in enumerate(targets):
            container.seek(int(target / stream.time_base), stream=stream)
            frame = None
            for candidate in container.decode(stream):
                frame = candidate
                break
            if frame is None:
                continue
            image = frame.to_image().resize((width, int(frame.height * width / frame.width)))
            name = f"{TIMELINE_SAMPLE}_{index:02d}.jpg"
            image.save(out_dir / name, quality=82)
            names.append(name)
    return names


def timeline_payload() -> dict:
    sample_dir = R51 / "smoke_test_details" / TIMELINE_SAMPLE
    words = read_csv(sample_dir / "word_alignment.csv")
    manual_rows = [row for row in read_csv(R5 / "manual_boundary" / "normalized_manual_boundaries.csv")
                   if row["sample_id"] == TIMELINE_SAMPLE]
    manual_by_index = {int(row["word_index"]): row for row in manual_rows}
    confidence = read_json(sample_dir / "alignment_confidence.json")
    conf_by_index = {int(item["word_index"]): item for item in confidence.get("words", [])}
    word_payload = []
    for row in words:
        index = int(row["word_index"])
        manual = manual_by_index.get(index, {})
        conf = conf_by_index.get(index, {})
        word_payload.append({
            "word_index": index,
            "word": row["original_word"],
            "mfa_start_s": None if row["start_s"] in ("", "nan") else round(float(row["start_s"]), 4),
            "mfa_end_s": None if row["end_s"] in ("", "nan") else round(float(row["end_s"]), 4),
            "manual_start_s": round(float(manual["reference_start_s"]), 4) if manual else None,
            "manual_end_s": round(float(manual["reference_end_s"]), 4) if manual else None,
            "alignment_confidence": conf.get("alignment_confidence", "UNAVAILABLE"),
            "human_boundary_error_s": round(float(conf["human_max_boundary_error_s"]), 4)
            if conf.get("human_max_boundary_error_s") not in (None, "") else None,
        })
    wav = sample_dir / "audio_mfa_16k_mono.wav"
    thumbs = extract_thumbnails(
        DATA_ROOT / "-3g5yACwYnA" / "13.mp4", PUBLIC / "thumbs",
        [round(float(item["mfa_start_s"]), 2) for item in word_payload[::3] if item["mfa_start_s"]] or [0.5])
    return {
        "sample_id": TIMELINE_SAMPLE,
        "official_text": read_json(sample_dir / "correspondence_qa.json").get("official_text", ""),
        "duration_s": read_json(sample_dir / "media_metadata.json").get("duration_s"),
        "words": word_payload,
        "waveform": waveform_envelope(wav) if wav.is_file() else None,
        "thumbnails": thumbs,
    }


def main() -> None:
    manual_metrics = read_json(R5 / "manual_qa" / "manual_qa_metrics.json")
    extra_speech = read_json(R5 / "qa" / "extra_speech_metrics.json")
    face_qa = read_json(R5 / "qa" / "face_qa_metrics.json")
    boundary_overall = read_json(R5 / "manual_boundary" / "manual_boundary_metrics_overall.json")
    confidence_summary = read_json(R5 / "alignment_confidence" / "alignment_confidence_summary.json")
    smoke_rows = read_csv(R51 / "smoke_test_summary.csv")

    confusion_rows = read_csv(R5 / "manual_qa" / "manual_qa_confusion_matrix.csv")
    row_labels = sorted({row["manual_class"] for row in confusion_rows})
    col_labels = sorted({row["automatic_class"] for row in confusion_rows})
    counts = {(row["manual_class"], row["automatic_class"]): int(row["count"]) for row in confusion_rows}
    confusion = {
        "row_labels": row_labels,
        "columns": col_labels,
        "matrix": [[counts.get((r, c), 0) for c in col_labels] for r in row_labels],
    }

    boundary_by_sample = []
    for row in read_csv(R5 / "manual_boundary" / "manual_boundary_metrics_by_sample.csv"):
        boundary_by_sample.append({
            "sample_id": row["sample_id"],
            "manual_words": int(row.get("manual_word_count") or row.get("reference_word_count") or 0),
            "compared_words": int(row.get("compared_word_count") or 0),
            "mean_boundary_error_ms": round(float(row["mean_boundary_error_s"]) * 1000, 1)
            if row.get("mean_boundary_error_s") not in ("", "nan", None) else None,
            "mean_temporal_iou": round(float(row["mean_temporal_iou"]), 3)
            if row.get("mean_temporal_iou") not in ("", "nan", None) else None,
            "pass_rate_100ms": round(float(row["pass_rate_100ms"]), 3)
            if row.get("pass_rate_100ms") not in ("", "nan", None) else None,
        })

    backend_pivot: dict[str, dict] = {}
    for row in read_csv(R5 / "visual_backend" / "visual_backend_comparison.csv"):
        entry = backend_pivot.setdefault(row["sample_id"], {"sample_id": row["sample_id"]})
        key = row["backend"]
        entry[f"{key}_valid"] = int(row.get("face_success_frame_count") or 0)
        entry[f"{key}_sampled"] = int(row.get("sampled_frame_count") or 0)
        entry[f"{key}_runtime_s"] = round(float(row.get("runtime_s") or 0.0), 2)
    backend_rows = sorted(backend_pivot.values(), key=lambda item: item["sample_id"])

    samples = {}
    for row in smoke_rows:
        sample_id = row["sample_id"]
        detail = R51 / "smoke_test_details" / sample_id
        entry = {"summary": row}
        if (detail / "feature_package_metadata.json").is_file():
            meta = read_json(detail / "feature_package_metadata.json")
            entry["quality"] = meta.get("quality", {})
            entry["mfa_policy"] = meta.get("mfa_policy")
            entry["official_text"] = meta.get("official_text", "")
            entry["word_count"] = meta.get("word_count")
            entry["visual_backend"] = meta.get("visual_backend")
        if (detail / "_SUCCESS.json").is_file():
            entry["success"] = read_json(detail / "_SUCCESS.json")
        if (detail / "correspondence_qa.json").is_file():
            qa = read_json(detail / "correspondence_qa.json")
            entry["qa"] = {key: qa.get(key) for key in (
                "duration_s", "speech_duration_s", "detected_language", "language_probability",
                "official_word_count", "asr_word_count", "face_present", "face_detection_rate",
                "correspondence_status", "route")}
        if (detail / "word_alignment.csv").is_file():
            entry["alignment_words"] = [{
                "word_index": int(item["word_index"]), "word": item["original_word"],
                "start_s": None if item["start_s"] in ("", "nan") else round(float(item["start_s"]), 4),
                "end_s": None if item["end_s"] in ("", "nan") else round(float(item["end_s"]), 4),
                "alignment_mask": int(item["alignment_mask"]),
            } for item in read_csv(detail / "word_alignment.csv")]
        samples[sample_id] = entry

    payload = {
        "title": "复杂场景下多模态情感识别 · 问题1 数据处理与时序特征构建",
        "subtitle": "CMU-MOSEI 100条原始视频 · 质量诊断 · 文本-音频一致性 · ASR局部定位 + MFA · OpenFace68 视觉特征",
        "pipeline": ["原始视频", "质量诊断", "文本-音频一致性检测", "ASR局部定位", "MFA时间对齐",
                     "OpenFace68视觉特征", "三模态时序特征"],
        "quality_overview": {
            "sample_count": manual_metrics["sample_count"],
            "manual_class_counts": manual_metrics["manual_class_counts"],
            "automatic_class_counts": manual_metrics["automatic_class_counts"],
            "exact_class_accuracy": manual_metrics["exact_class_accuracy"],
            "coarse_class_accuracy": manual_metrics["coarse_class_accuracy"],
        },
        "correspondence_qa": {
            "high_confidence_precision": manual_metrics["high_confidence_precision"],
            "high_confidence_recall": manual_metrics["high_confidence_recall"],
            "high_confidence_f1": manual_metrics["high_confidence_f1"],
            "unsafe_false_positive_count": manual_metrics["unsafe_false_positive_count"],
            "conservative_false_negative_count": manual_metrics["conservative_false_negative_count"],
            "routing_confusion": manual_metrics["routing_confusion"],
            "confusion": confusion,
            "extra_speech": extra_speech,
        },
        "face_qa": face_qa,
        "alignment": {
            "overall": boundary_overall,
            "by_sample": boundary_by_sample,
            "alignment_confidence": {
                "counts": confidence_summary.get("confidence_counts", {}),
                "audited_word_count": confidence_summary.get("audited_word_count"),
                "eligible_real_mfa_sample_count": confidence_summary.get("eligible_real_mfa_sample_count"),
                "meaning": "MFA–ASR 边界分歧不确定性，不是人工时间精度",
            },
            "timeline": timeline_payload(),
        },
        "visual": {
            "schema": read_json(R51 / "visual_feature_schema.json"),
            "feature_schema": read_json(R51 / "feature_schema.json"),
            "backend_comparison": backend_rows,
        },
        "smoke": {
            "rows": smoke_rows,
            "samples": samples,
            "summary": read_json(R51 / "alignment_confidence_summary.json")
            if (R51 / "alignment_confidence_summary.json").is_file() else {},
            "full_100_sample_feature_extraction": "NOT_RUN",
        },
        "sources": [
            "outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json",
            "outputs/q1_v2_round5/manual_qa/manual_qa_confusion_matrix.csv",
            "outputs/q1_v2_round5/qa/extra_speech_metrics.json",
            "outputs/q1_v2_round5/qa/face_qa_metrics.json",
            "outputs/q1_v2_round5/manual_boundary/manual_boundary_metrics_overall.json",
            "outputs/q1_v2_round5/alignment_confidence/alignment_confidence_summary.json",
            "outputs/q1_v2_round5/visual_backend/visual_backend_comparison.csv",
            "outputs/q1_v2_round5_1/smoke_test_summary.csv",
            "outputs/q1_v2_round5_1/feature_schema.json",
            "outputs/q1_v2_round5_1/visual_feature_schema.json",
        ],
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    target = PUBLIC / "frontend_data.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
