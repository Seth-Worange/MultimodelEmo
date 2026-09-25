"""Round-4 light correspondence audit; never launches 100-sample full extraction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .config import Q1Config, collect_environment
from .correspondence_qa import QAConfig, inspect_audio_correspondence, inspect_visual_scene
from .data_loader import SampleRecord, load_samples
from .io_utils import read_json, sha256_file, write_csv, write_json
from .manual_sample_audit import create_manual_sample_audit
from .media import inspect_and_extract_media


AUDIT_FIELDS = (
    "sample_id", "video_id", "clip_id", "duration_s", "audio_signal_present",
    "speech_present", "speech_duration_s", "detected_language", "language_probability",
    "english_speech_likely", "official_word_count", "asr_word_count",
    "official_text_present", "text_audio_match_score", "matched_audio_start_s",
    "matched_audio_end_s", "extra_speech_before", "extra_speech_after",
    "video_signal_present", "face_present", "face_detection_rate",
    "static_visual_score", "static_or_near_static_visual", "decoded_frame_count",
    "sampled_frame_count", "face_detected_frame_count", "multiple_face_frame_count",
    "correspondence_status", "route", "automatic_reason", "manual_review_required",
    "qa_elapsed_s", "qa_error",
)


def _audit_row(record: SampleRecord, sample_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {field: None for field in AUDIT_FIELDS}
    row.update({
        "sample_id": record.sample_id,
        "video_id": record.video_id,
        "clip_id": record.clip_id,
        "official_word_count": record.word_count,
        "correspondence_status": "NOT_EVALUATED",
        "route": "NOT_EVALUATED",
        "manual_review_required": None,
    })
    qa_path = sample_dir / "correspondence_qa.json"
    if qa_path.is_file():
        qa = read_json(qa_path)
        row.update({field: qa.get(field) for field in AUDIT_FIELDS if field in qa})
    failed = sample_dir / "_QA_FAILED.json"
    if failed.is_file():
        detail = read_json(failed)
        row.update({
            "correspondence_status": "UNRESOLVED",
            "route": "REVIEW_REQUIRED",
            "automatic_reason": "qa_execution_failed",
            "manual_review_required": True,
            "qa_error": detail.get("error"),
            "qa_elapsed_s": detail.get("elapsed_s"),
        })
    return row


def write_audit_reports(records: list[SampleRecord], output_dir: Path) -> dict[str, Any]:
    rows = [_audit_row(record, output_dir / "samples" / record.sample_id) for record in records]
    write_csv(output_dir / "correspondence_audit.csv", rows, AUDIT_FIELDS)
    status_counts = {
        status: sum(row["correspondence_status"] == status for row in rows)
        for status in (
            "MATCHED", "PARTIAL_MATCH", "TEXT_AUDIO_MISMATCH", "NO_SPEECH",
            "NON_ENGLISH_SPEECH", "NO_AUDIO", "UNRESOLVED", "NOT_EVALUATED",
        )
    }
    evaluated = [row for row in rows if row["correspondence_status"] != "NOT_EVALUATED"]

    def rate(predicate: Any, eligible: Any | None = None) -> float | None:
        denominator = [row for row in evaluated if eligible is None or eligible(row)]
        return sum(bool(predicate(row)) for row in denominator) / len(denominator) if denominator else None

    manual_path = output_dir / "manual_sample_audit_template.csv"
    automatic = {
        row["sample_id"]: {
            "automatic_correspondence_status": row["correspondence_status"],
            "automatic_speech_present": row["speech_present"],
            "automatic_language": row["detected_language"],
            "automatic_face_present": row["face_present"],
            "automatic_match_score": row["text_audio_match_score"],
        }
        for row in evaluated
    }
    create_manual_sample_audit(records, manual_path, automatic_by_id=automatic)
    with manual_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manual_rows = list(csv.DictReader(handle))
    manual_count = sum(bool(row["manual_correspondence_class"].strip()) for row in manual_rows)
    confusion: dict[str, dict[str, int]] = {}
    for row in manual_rows:
        human = row["manual_correspondence_class"].strip()
        automatic_status = row["automatic_correspondence_status"].strip()
        if human and automatic_status != "NOT_EVALUATED":
            confusion.setdefault(automatic_status, {})[human] = confusion.setdefault(automatic_status, {}).get(human, 0) + 1
    manual_metrics_path = output_dir / "manual_validation" / "manual_alignment_metrics.json"
    manual_metrics = read_json(manual_metrics_path) if manual_metrics_path.is_file() else None
    def alignment_succeeded(sample_id: str) -> bool:
        sample_dir = output_dir / "samples" / sample_id
        if not (sample_dir / "mfa_raw.json").is_file():
            return False
        alignment_path = sample_dir / "word_alignment.csv"
        if not alignment_path.is_file():
            return False
        with alignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
            alignment_rows = list(csv.DictReader(handle))
        return bool(alignment_rows) and any(row["alignment_mask"] == "1" for row in alignment_rows)

    summary = {
        "official_sample_count": len(records),
        "qa_sample_count": len(evaluated),
        "qa_unevaluated_count": status_counts["NOT_EVALUATED"],
        "status_counts": status_counts,
        "speech_present_rate": rate(lambda row: row["speech_present"] is True, lambda row: row["speech_present"] is not None),
        "english_speech_rate": rate(lambda row: row["english_speech_likely"] is True, lambda row: row["english_speech_likely"] is not None),
        "official_text_match_rate": rate(lambda row: row["correspondence_status"] == "MATCHED"),
        "partial_match_rate": rate(lambda row: row["correspondence_status"] == "PARTIAL_MATCH"),
        "mismatch_rate": rate(lambda row: row["correspondence_status"] == "TEXT_AUDIO_MISMATCH"),
        "no_speech_rate": rate(lambda row: row["correspondence_status"] == "NO_SPEECH"),
        "no_face_rate": rate(lambda row: row["face_present"] is False, lambda row: row["face_present"] is not None),
        "conditional_alignment_attempt_count": sum(
            (output_dir / "samples" / row["sample_id"] / "mfa_command.json").is_file()
            or (output_dir / "samples" / row["sample_id"] / "mfa_probe.json").is_file()
            for row in rows
        ),
        "conditional_alignment_success_count": sum(
            alignment_succeeded(row["sample_id"])
            for row in rows
        ),
        "manual_sample_audit_count": manual_count,
        "manual_correspondence_confusion_matrix": confusion or None,
        "manual_alignment_sample_count": len(manual_metrics["sample_ids"]) if manual_metrics else 0,
        "manual_alignment_word_count": manual_metrics["new_local_crop_mfa"]["compared_word_count"] if manual_metrics else 0,
        "manual_boundary_metrics": manual_metrics["new_local_crop_mfa"] if manual_metrics else None,
        "accuracy_note": (
            "Automatic text/audio match is neither word-boundary accuracy nor sentiment accuracy; "
            "unreviewed automatic statuses are hypotheses."
        ),
        "full_100_sample_feature_extraction": "NOT_RUN",
    }
    write_json(output_dir / "quality_summary.json", summary)
    return summary


def audit_one(
    record: SampleRecord, output_dir: Path, *, asr_python: Path,
    asr_model_path: Path,
    face_model_path: Path, config: QAConfig,
) -> dict[str, Any]:
    sample_dir = output_dir / "samples" / record.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    qa_path = sample_dir / "correspondence_qa.json"
    if qa_path.is_file():
        return {"sample_id": record.sample_id, "action": "skipped_existing", "status": read_json(qa_path)["correspondence_status"]}
    started = time.perf_counter()
    try:
        if record.read_status != "ok":
            raise ValueError("official_sample_read_error:" + "|".join(record.errors))
        media = inspect_and_extract_media(
            record.video_path, sample_dir / "audio_mfa_16k_mono.wav",
            metadata_path=sample_dir / "media_metadata.json",
        )
        audio = inspect_audio_correspondence(
            media, record.text, asr_python=asr_python,
            asr_model_path=asr_model_path, config=config, output_dir=sample_dir,
        )
        visual = inspect_visual_scene(
            media, video_path=record.video_path, face_model_path=face_model_path,
            config=config,
        )
        if audio["route"] == "HIGH_CONFIDENCE_MATCH" and not media.audio_timestamp_reliable:
            audio.update({
                "correspondence_status": "UNRESOLVED",
                "route": "REVIEW_REQUIRED",
                "correspondence_reason": "audio_pts_mapping_unreliable",
                "automatic_reason": "audio_pts_mapping_unreliable",
                "manual_review_required": True,
                "official_text_present_in_audio": None,
                "official_text_present": None,
            })
        qa = {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "duration_s": media.duration_s,
            "source_video_sha256": sha256_file(record.video_path),
            "official_text_source": "label-100.xlsx",
            "official_text": record.text,
            "audio_timestamp_reliable": media.audio_timestamp_reliable,
            "video_timestamp_reliable": media.video_timestamp_reliable,
            "media_errors": media.errors,
            "media_warnings": media.warnings,
            **audio,
            **visual,
            "qa_elapsed_s": time.perf_counter() - started,
            "qa_error": None,
        }
        write_json(qa_path, qa)
        return {"sample_id": record.sample_id, "action": "processed", "status": qa["correspondence_status"]}
    except Exception as exc:
        failure = {
            "sample_id": record.sample_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": time.perf_counter() - started,
        }
        write_json(sample_dir / "_QA_FAILED.json", failure)
        return {"sample_id": record.sample_id, "action": "failed", "error": failure["error"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-id", action="append")
    selection.add_argument("--all-qa", action="store_true")
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--asr-python", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, load_summary = load_samples(args.data_root, args.labels, probe_decode=False)
    if len(records) != 100 or len({item.sample_id for item in records}) != 100:
        raise ValueError(f"Official sample set is not 100 unique items: {load_summary}")
    create_manual_sample_audit(records, output / "manual_sample_audit_template.csv")
    chosen = records if args.all_qa else [item for item in records if item.sample_id in set(args.sample_id)]
    if not chosen or (args.sample_id and len(chosen) != len(set(args.sample_id))):
        raise ValueError("Requested sample_id is absent from the official workbook")
    if not args.asr_model.is_dir() or not args.face_model.is_file() or not args.asr_python.is_file():
        raise FileNotFoundError("Local ASR model, isolated Python, and Face Landmarker model are required")

    config = QAConfig(asr_model=str(args.asr_model.resolve()))
    write_json(output / "experiment_config.json", {
        "qa_config": asdict(config),
        "asr_python": str(args.asr_python.resolve()),
        "q1_default_visual_fps": Q1Config().visual_fps,
        "asr_model_source": "https://huggingface.co/Systran/faster-whisper-base",
        "asr_model_sha256": sha256_file(args.asr_model / "model.bin"),
        "face_model_path": str(args.face_model.resolve()),
        "face_model_sha256": sha256_file(args.face_model),
        "selected_sample_ids": [item.sample_id for item in chosen],
        "full_feature_extraction": False,
        "environment": collect_environment(),
    })
    for number, record in enumerate(chosen, 1):
        failed = output / "samples" / record.sample_id / "_QA_FAILED.json"
        if failed.is_file() and not args.retry_failed:
            print(json.dumps({"sample_id": record.sample_id, "action": "skipped_failed_use_--retry-failed"}), flush=True)
            continue
        result = audit_one(
            record, output, asr_python=args.asr_python, asr_model_path=args.asr_model,
            face_model_path=args.face_model, config=config,
        )
        print(json.dumps({"progress": f"{number}/{len(chosen)}", **result}, ensure_ascii=False), flush=True)
        write_audit_reports(records, output)
    write_audit_reports(records, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
