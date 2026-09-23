"""Coverage, timestamp, modality, and optional manual-reference quality reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data_loader import MANIFEST_FIELDS, SampleRecord
from .io_utils import directory_size, read_json, write_csv, write_json


QUALITY_FIELDS = (
    "sample_id", "read_status", "processing_status", "original_word_count",
    "aligned_word_count", "alignment_coverage", "text_valid_count", "text_valid_rate",
    "audio_valid_count", "audio_valid_rate", "visual_valid_count", "visual_valid_rate",
    "visual_no_sample_word_count", "visual_sampling_gap_rate", "sampled_visual_frame_count",
    "face_detection_failure_count", "face_detection_failure_rate",
    "short_word_count", "short_word_visual_coverage", "medium_word_count",
    "medium_word_visual_coverage", "long_word_count", "long_word_visual_coverage",
    "timestamp_out_of_bounds_count", "timestamp_non_monotonic_count",
    "timestamp_abnormal_overlap_count", "timestamp_zero_duration_count",
    "processing_elapsed_s", "output_size_bytes", "failure_reason",
)

REVIEW_FIELDS = (
    "sample_id", "automatic_reason", "alignment_coverage", "visual_sampling_gap_rate",
    "face_detection_failure_rate", "manual_reference_present",
)

MANUAL_FIELDS = (
    "sample_id", "word_index", "reference_start_s", "reference_end_s", "reviewer_id",
)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _count_contains(values: Iterable[str], needle: str) -> int:
    return sum(needle in str(value) for value in values)


def _visual_coverage_by_duration(
    starts: np.ndarray, ends: np.ndarray, alignment: np.ndarray, visual: np.ndarray,
) -> dict[str, tuple[int, float | None]]:
    duration = ends - starts
    definitions = {
        "short": (duration < 0.2),
        "medium": ((duration >= 0.2) & (duration < 0.5)),
        "long": (duration >= 0.5),
    }
    output: dict[str, tuple[int, float | None]] = {}
    aligned = alignment.astype(bool) & np.isfinite(duration)
    for name, selector in definitions.items():
        selected = aligned & selector
        count = int(selected.sum())
        output[name] = (count, _safe_rate(int((visual.astype(bool) & selected).sum()), count))
    return output


def _failed_row(record: SampleRecord, sample_dir: Path) -> dict[str, Any]:
    failed_path = sample_dir / "_FAILED.json"
    reason = "not_processed"
    elapsed = None
    if failed_path.is_file():
        failed = read_json(failed_path)
        reason = str(failed.get("error", "processing_failed"))
        elapsed = failed.get("elapsed_s")
    return {
        "sample_id": record.sample_id,
        "read_status": record.read_status,
        "processing_status": "failed" if failed_path.is_file() else "not_processed",
        "original_word_count": record.word_count,
        "processing_elapsed_s": elapsed,
        "output_size_bytes": directory_size(sample_dir) if sample_dir.is_dir() else 0,
        "failure_reason": reason,
    }


def sample_quality(record: SampleRecord, sample_dir: Path) -> dict[str, Any]:
    success_path = sample_dir / "_SUCCESS.json"
    fused_path = sample_dir / "fused_features.npz"
    if not success_path.is_file() or not fused_path.is_file():
        return _failed_row(record, sample_dir)
    success = read_json(success_path)
    with np.load(fused_path, allow_pickle=False) as arrays:
        alignment = arrays["alignment_mask"].astype(bool)
        text = arrays["text_mask"].astype(bool)
        audio = arrays["audio_mask"].astype(bool)
        visual = arrays["visual_mask"].astype(bool)
        starts = arrays["word_start_s"]
        ends = arrays["word_end_s"]
        visual_frame_count = len(arrays["visual_frame_times_s"])
        visual_detection = arrays["visual_detection_mask"].astype(bool)
        visual_word_frame_counts = arrays["visual_word_frame_counts"]
    n_words = len(alignment)
    aligned_count = int(alignment.sum())
    aligned_with_no_visual_sample = int(((visual_word_frame_counts == 0) & alignment).sum())
    coverage = _visual_coverage_by_duration(starts, ends, alignment, visual)
    alignment_metadata = read_json(sample_dir / "alignment_metadata.json")
    failure_reasons = [str(word.get("failure_reason", "")) for word in alignment_metadata.get("words", [])]
    media_metadata = read_json(sample_dir / "media_metadata.json")
    media_errors = [str(value) for value in media_metadata.get("errors", [])]
    return {
        "sample_id": record.sample_id,
        "read_status": record.read_status,
        "processing_status": "success",
        "original_word_count": n_words,
        "aligned_word_count": aligned_count,
        "alignment_coverage": _safe_rate(aligned_count, n_words),
        "text_valid_count": int(text.sum()),
        "text_valid_rate": _safe_rate(int(text.sum()), n_words),
        "audio_valid_count": int(audio.sum()),
        "audio_valid_rate": _safe_rate(int(audio.sum()), n_words),
        "visual_valid_count": int(visual.sum()),
        "visual_valid_rate": _safe_rate(int(visual.sum()), n_words),
        "visual_no_sample_word_count": aligned_with_no_visual_sample,
        "visual_sampling_gap_rate": _safe_rate(aligned_with_no_visual_sample, aligned_count),
        "sampled_visual_frame_count": visual_frame_count,
        "face_detection_failure_count": int((~visual_detection).sum()),
        "face_detection_failure_rate": _safe_rate(int((~visual_detection).sum()), visual_frame_count),
        "short_word_count": coverage["short"][0],
        "short_word_visual_coverage": coverage["short"][1],
        "medium_word_count": coverage["medium"][0],
        "medium_word_visual_coverage": coverage["medium"][1],
        "long_word_count": coverage["long"][0],
        "long_word_visual_coverage": coverage["long"][1],
        "timestamp_out_of_bounds_count": _count_contains(failure_reasons, "out_of_bounds"),
        "timestamp_non_monotonic_count": (
            _count_contains(media_errors, "non_monotonic")
            + _count_contains(failure_reasons, "non_monotonic")
        ),
        "timestamp_abnormal_overlap_count": _count_contains(failure_reasons, "abnormal_overlap"),
        "timestamp_zero_duration_count": _count_contains(failure_reasons, "zero_or_negative_duration"),
        "processing_elapsed_s": success.get("elapsed_s"),
        "output_size_bytes": directory_size(sample_dir),
        "failure_reason": "",
    }


def _load_manual_reference(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in MANUAL_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manual reference CSV missing columns: {missing}")
        return list(reader)


def evaluate_manual_references(
    output_dir: Path, references: Sequence[dict[str, str]],
) -> dict[str, Any] | None:
    if not references:
        return None
    start_errors: list[float] = []
    end_errors: list[float] = []
    reviewed_rows = 0
    missing_predictions = 0
    cache: dict[str, Any] = {}
    for row in references:
        sample_id = str(row["sample_id"])
        try:
            word_index = int(row["word_index"])
            reference_start = float(row["reference_start_s"])
            reference_end = float(row["reference_end_s"])
        except (TypeError, ValueError):
            continue
        reviewed_rows += 1
        if sample_id not in cache:
            path = output_dir / "samples" / sample_id / "fused_features.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as arrays:
                    cache[sample_id] = {
                        "starts": arrays["word_start_s"].copy(),
                        "ends": arrays["word_end_s"].copy(),
                        "mask": arrays["alignment_mask"].copy(),
                    }
            else:
                cache[sample_id] = None
        prediction = cache[sample_id]
        if prediction is None or word_index < 0 or word_index >= len(prediction["mask"]) or not prediction["mask"][word_index]:
            missing_predictions += 1
            continue
        start_errors.append(abs(float(prediction["starts"][word_index]) - reference_start))
        end_errors.append(abs(float(prediction["ends"][word_index]) - reference_end))
    if not start_errors:
        return {
            "manual_reference_rows": reviewed_rows,
            "matched_prediction_count": 0,
            "missing_prediction_count": missing_predictions,
            "metrics": None,
        }
    starts = np.asarray(start_errors)
    ends = np.asarray(end_errors)
    maximum = np.maximum(starts, ends)
    return {
        "manual_reference_rows": reviewed_rows,
        "matched_prediction_count": len(starts),
        "missing_prediction_count": missing_predictions,
        "start_mae_s": float(starts.mean()),
        "end_mae_s": float(ends.mean()),
        "mean_boundary_error_s": float(np.concatenate([starts, ends]).mean()),
        "pass_rate_50ms": float((maximum <= 0.05).mean()),
        "pass_rate_100ms": float((maximum <= 0.10).mean()),
        "pass_rate_200ms": float((maximum <= 0.20).mean()),
    }


def _aggregate(rows: Sequence[dict[str, Any]], manual: dict[str, Any] | None) -> dict[str, Any]:
    successful = [row for row in rows if row.get("processing_status") == "success"]
    total_words = sum(int(row.get("original_word_count") or 0) for row in rows)
    aligned_words = sum(int(row.get("aligned_word_count") or 0) for row in successful)
    total_visual_frames = sum(int(row.get("sampled_visual_frame_count") or 0) for row in successful)
    failed_faces = sum(int(row.get("face_detection_failure_count") or 0) for row in successful)
    return {
        "sample_file_coverage": _safe_rate(sum(record.get("read_status") == "ok" for record in rows), len(rows)),
        "sample_processing_success_rate": _safe_rate(len(successful), len(rows)),
        "sample_count": len(rows),
        "successful_sample_count": len(successful),
        "original_word_count": total_words,
        "successful_output_original_word_count": sum(
            int(row.get("original_word_count") or 0) for row in successful
        ),
        "aligned_word_count": aligned_words,
        "automatic_alignment_coverage_not_accuracy": _safe_rate(aligned_words, total_words),
        "text_word_valid_rate": _safe_rate(sum(int(row.get("text_valid_count") or 0) for row in successful), total_words),
        "audio_word_valid_rate": _safe_rate(sum(int(row.get("audio_valid_count") or 0) for row in successful), total_words),
        "visual_word_valid_rate": _safe_rate(sum(int(row.get("visual_valid_count") or 0) for row in successful), total_words),
        "face_detection_failure_rate": _safe_rate(failed_faces, total_visual_frames),
        "total_output_size_bytes": sum(int(row.get("output_size_bytes") or 0) for row in rows),
        "manual_alignment_evaluation": manual,
        "accuracy_note": (
            "MFA completion/coverage and nonzero feature rates are not alignment accuracy or sentiment accuracy. "
            "Boundary accuracy is reported only when human references are supplied."
        ),
    }


def build_quality_reports(
    output_dir: Path,
    records: Sequence[SampleRecord],
    *,
    manual_reference_csv: Path | None = None,
    selected_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "manual_alignment_reference_template.csv"
    if not template_path.exists():
        write_csv(template_path, [], MANUAL_FIELDS)
    references = _load_manual_reference(manual_reference_csv)
    rows = [sample_quality(record, output_dir / "samples" / record.sample_id) for record in records]
    manual = evaluate_manual_references(output_dir, references)
    selected_sample_ids = selected_sample_ids or {record.sample_id for record in records}
    selected_rows = [row for row in rows if row["sample_id"] in selected_sample_ids]
    summary = _aggregate(selected_rows, manual)
    summary["dataset_manifest_sample_count"] = len(records)
    summary["selected_sample_count"] = len(selected_rows)
    summary["output_tree_size_bytes_before_current_report"] = directory_size(output_dir)
    cache_dir = output_dir / "cache"
    summary["cache_size_bytes"] = directory_size(cache_dir) if cache_dir.is_dir() else 0
    summary["submission_size_note"] = (
        "Use per-sample output_size_bytes and exclude cache/model weights when preparing the 50 MB submission."
    )
    write_csv(output_dir / "quality_by_sample.csv", rows, QUALITY_FIELDS)

    reference_samples = {str(row["sample_id"]) for row in references}
    candidates: list[dict[str, Any]] = []
    for row in selected_rows:
        reasons: list[str] = []
        if row.get("processing_status") != "success":
            reasons.append(str(row.get("failure_reason") or "not_successful"))
        if row.get("alignment_coverage") is not None and float(row["alignment_coverage"]) < 0.9:
            reasons.append("automatic_alignment_coverage_below_0.9")
        if row.get("visual_sampling_gap_rate") is not None and float(row["visual_sampling_gap_rate"]) > 0.25:
            reasons.append("visual_sampling_gap_rate_above_0.25")
        if row.get("face_detection_failure_rate") is not None and float(row["face_detection_failure_rate"]) > 0.5:
            reasons.append("face_detection_failure_rate_above_0.5")
        if sum(int(row.get(name) or 0) for name in (
            "timestamp_out_of_bounds_count", "timestamp_non_monotonic_count",
            "timestamp_abnormal_overlap_count", "timestamp_zero_duration_count",
        )):
            reasons.append("timestamp_integrity_error")
        if reasons:
            candidates.append({
                "sample_id": row["sample_id"],
                "automatic_reason": " | ".join(reasons),
                "alignment_coverage": row.get("alignment_coverage"),
                "visual_sampling_gap_rate": row.get("visual_sampling_gap_rate"),
                "face_detection_failure_rate": row.get("face_detection_failure_rate"),
                "manual_reference_present": int(row["sample_id"] in reference_samples),
            })
    write_csv(output_dir / "review_candidates.csv", candidates, REVIEW_FIELDS)
    automatic_samples = {str(row["sample_id"]) for row in candidates}
    summary["manual_reference_sample_count"] = len(reference_samples)
    summary["automatic_review_candidate_count"] = len(automatic_samples)
    summary["manual_and_automatic_overlap_count"] = len(reference_samples & automatic_samples)
    summary["review_set_note"] = (
        "Manual-reference samples and automatically selected anomaly candidates are counted separately."
    )
    write_json(output_dir / "quality_summary.json", summary)

    quality_by_id = {row["sample_id"]: row for row in rows}
    manifest_rows: list[dict[str, Any]] = []
    manifest_fields = list(MANIFEST_FIELDS) + [field for field in QUALITY_FIELDS if field not in MANIFEST_FIELDS]
    for record in records:
        merged = record.manifest_row()
        merged.update(quality_by_id[record.sample_id])
        manifest_rows.append(merged)
    write_csv(output_dir / "manifest.csv", manifest_rows, manifest_fields)
    return summary
