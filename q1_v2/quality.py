"""Coverage, timestamp, modality, and optional manual-reference quality reports."""

from __future__ import annotations

import csv
import argparse
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .alignment import locate_mfa_unknowns, parse_mfa_output
from .data_loader import MANIFEST_FIELDS, SampleRecord, load_samples, split_original_words
from .io_utils import directory_size, read_json, write_csv, write_json


QUALITY_FIELDS = (
    "sample_id", "read_status", "processing_status", "original_word_count",
    "alignment_status", "text_status", "audio_status", "visual_status",
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
WORD_REVIEW_FIELDS = (
    "sample_id", "word_index", "original_word", "start_s", "end_s",
    "alignment_mask", "audio_mask", "visual_mask", "issue_type",
    "review_priority", "review_reason", "mfa_unknown_start_s", "mfa_unknown_end_s",
    "visual_frame_count", "visual_status", "revised_start_s", "revised_end_s",
    "revised_alignment_mask", "manual_boundary_review_status",
    "realignment_max_boundary_shift_s", "source_variant",
)
MANUAL_TEMPLATE_FIELDS = (
    "sample_id", "word_index", "original_word", "reference_start_s",
    "reference_end_s", "reviewer_id", "boundary_status", "review_note",
    "automatic_revised_start_s", "automatic_revised_end_s",
)


def word_review_rows(sample_dir: Path) -> list[dict[str, Any]]:
    """Generate word-level issues independently of sample-level thresholds."""
    fused_path = sample_dir / "fused_features.npz"
    raw_path = sample_dir / "mfa_raw.json"
    if not fused_path.is_file():
        failed_path = sample_dir / "_FAILED.json"
        input_path = sample_dir / "sample_input.json"
        if not failed_path.is_file() or not input_path.is_file():
            return []
        failed = read_json(failed_path)
        sample = read_json(input_path)
        reason = f"{failed.get('failure_stage', 'unknown_stage')}: {failed.get('error', 'processing_failed')}"
        return [{
            "sample_id": sample["sample_id"], "word_index": index,
            "original_word": word, "start_s": "", "end_s": "",
            "alignment_mask": 0 if failed.get("failure_stage") == "alignment" else "",
            "audio_mask": "", "visual_mask": "",
            "issue_type": "processing_failed | word_time_missing" if failed.get("failure_stage") == "alignment" else "processing_failed",
            "review_priority": "high", "review_reason": reason,
            "visual_frame_count": "", "visual_status": "not_evaluated",
        } for index, word in enumerate(sample["original_words"])]
    with np.load(fused_path, allow_pickle=False) as arrays:
        sample_id = str(arrays["sample_id"].item())
        words = arrays["original_words"].astype(str).tolist()
        starts = arrays["word_start_s"].copy()
        ends = arrays["word_end_s"].copy()
        alignment = arrays["alignment_mask"].astype(bool).copy()
        audio = arrays["audio_mask"].astype(bool).copy()
        visual = arrays["visual_mask"].astype(bool).copy()
        frame_counts = arrays["visual_word_frame_counts"].copy()
    metadata = read_json(sample_dir / "fused_metadata.json")
    visual_statuses = metadata.get("visual_word_status", [])
    unknowns = {}
    if raw_path.is_file():
        transcript_path = sample_dir / "transcript_original.txt"
        original_text = (
            transcript_path.read_text(encoding="utf-8") if transcript_path.is_file()
            else read_json(sample_dir / "sample_input.json")["text"]
        )
        original_words = split_original_words(original_text)
        if [word.text for word in original_words] != words:
            raise ValueError(f"Original word mismatch in {sample_dir}")
        unknowns = locate_mfa_unknowns(original_words, parse_mfa_output(raw_path))
    rows: list[dict[str, Any]] = []
    for index, original_word in enumerate(words):
        issues: list[str] = []
        reasons: list[str] = []
        if index in unknowns:
            issues.append("mfa_unknown")
            reasons.append("MFA emitted <unk> at this uniquely anchored original position")
        if not alignment[index]:
            issues.append("word_time_missing")
            reasons.append("Original word has no accepted MFA interval")
        elif int(frame_counts[index]) == 0:
            issues.append("no_video_frame_in_interval")
            reasons.append("No actually sampled video frame fell within [start,end)")
        elif not visual[index]:
            issues.append("sampled_frames_without_valid_face")
            reasons.append("Video frames exist in the word interval, but no valid face feature was extracted")
        if alignment[index] and float(ends[index] - starts[index]) > 1.0:
            issues.append("long_duration_review")
            reasons.append("Duration exceeds 1 s; review suggested, not an error verdict")
        if not issues:
            continue
        unknown = unknowns.get(index)
        rows.append({
            "sample_id": sample_id, "word_index": index,
            "original_word": original_word,
            "start_s": float(starts[index]) if alignment[index] else "",
            "end_s": float(ends[index]) if alignment[index] else "",
            "alignment_mask": int(alignment[index]),
            "audio_mask": int(audio[index]), "visual_mask": int(visual[index]),
            "issue_type": " | ".join(issues),
            "review_priority": "high" if "mfa_unknown" in issues or "word_time_missing" in issues else "medium" if ("no_video_frame_in_interval" in issues or "sampled_frames_without_valid_face" in issues) else "low",
            "review_reason": " | ".join(reasons),
            "mfa_unknown_start_s": unknown.start_s if unknown else "",
            "mfa_unknown_end_s": unknown.end_s if unknown else "",
            "visual_frame_count": int(frame_counts[index]),
            "visual_status": visual_statuses[index] if index < len(visual_statuses) else "",
            "source_variant": "fused_output",
        })
    return rows


def write_word_review_reports(
    sample_dirs: Sequence[Path], output_dir: Path,
    *, revised_alignment_root: Path | None = None,
) -> list[dict[str, Any]]:
    rows = [row for sample_dir in sample_dirs for row in word_review_rows(sample_dir)]
    if revised_alignment_root is not None:
        rows_by_key = {(str(row["sample_id"]), int(row["word_index"])): row for row in rows}
        revised_by_sample: dict[str, dict[int, dict[str, str]]] = {}
        for sample_dir in sample_dirs:
            sample_id = sample_dir.name
            path = revised_alignment_root / "samples" / sample_id / "word_alignment.csv"
            if path.is_file():
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    revised_by_sample[sample_id] = {
                        int(item["word_index"]): item for item in csv.DictReader(handle)
                    }
                with np.load(sample_dir / "fused_features.npz", allow_pickle=False) as arrays:
                    words = arrays["original_words"].astype(str).tolist()
                    old_starts = arrays["word_start_s"].copy()
                    old_ends = arrays["word_end_s"].copy()
                    old_alignment = arrays["alignment_mask"].astype(bool).copy()
                    old_audio = arrays["audio_mask"].copy()
                    old_visual = arrays["visual_mask"].copy()
                for index, revised in revised_by_sample[sample_id].items():
                    if index >= len(words) or revised["original_word"] != words[index]:
                        raise ValueError("Revised alignment changed original word order")
                    if not old_alignment[index] or revised["alignment_mask"] != "1":
                        continue
                    shift = max(
                        abs(float(old_starts[index]) - float(revised["start_s"])),
                        abs(float(old_ends[index]) - float(revised["end_s"])),
                    )
                    if shift <= 0.2:
                        continue
                    key = (sample_id, index)
                    if key not in rows_by_key:
                        row = {
                            "sample_id": sample_id, "word_index": index,
                            "original_word": words[index],
                            "start_s": float(old_starts[index]),
                            "end_s": float(old_ends[index]),
                            "alignment_mask": 1,
                            "audio_mask": int(old_audio[index]),
                            "visual_mask": int(old_visual[index]),
                            "issue_type": "", "review_priority": "high",
                            "review_reason": "", "visual_frame_count": "",
                            "visual_status": "baseline masks; revised features not yet aggregated",
                        }
                        rows.append(row)
                        rows_by_key[key] = row
                    row = rows_by_key[key]
                    row["issue_type"] = (row["issue_type"] + " | realignment_boundary_shift").strip(" |")
                    row["review_priority"] = "high"
                    row["review_reason"] = (
                        str(row["review_reason"]) + f" | Re-alignment moved a boundary by {shift:.3f} s; manual review required"
                    ).strip(" |")
                    row["realignment_max_boundary_shift_s"] = shift
        for row in rows:
            revised = revised_by_sample.get(str(row["sample_id"]), {}).get(int(row["word_index"]))
            if revised:
                if revised["original_word"] != row["original_word"]:
                    raise ValueError("Revised alignment changed original word order")
                row["revised_start_s"] = revised["start_s"]
                row["revised_end_s"] = revised["end_s"]
                row["revised_alignment_mask"] = revised["alignment_mask"]
                row["manual_boundary_review_status"] = "pending"
    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["word_index"])))
    write_csv(output_dir / "word_review_candidates.csv", rows, WORD_REVIEW_FIELDS)
    template = [
        {"sample_id": row["sample_id"], "word_index": row["word_index"],
         "original_word": row["original_word"], "reference_start_s": "",
         "reference_end_s": "", "reviewer_id": "", "boundary_status": "unreviewed",
         "review_note": "",
         "automatic_revised_start_s": row.get("revised_start_s", ""),
         "automatic_revised_end_s": row.get("revised_end_s", "")}
        for row in rows
    ]
    write_csv(output_dir / "word_review_reference_template.csv", template, MANUAL_TEMPLATE_FIELDS)
    return rows


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
    else:
        failed = {}

    stage_outputs = {
        "alignment": sample_dir / "alignment_metadata.json",
        "text": sample_dir / "text_features.npz",
        "audio": sample_dir / "audio_features.npz",
        "visual": sample_dir / "visual_features.npz",
    }
    failure_stage = str(failed.get("failure_stage", ""))
    if not failure_stage and failed_path.is_file():
        # Compatibility with first-round failures, which predate the explicit
        # failure_stage field.  The first absent artifact is the stage that was
        # attempted; subsequent stages were not evaluated.
        if not (sample_dir / "media_metadata.json").is_file():
            failure_stage = "media"
        else:
            failure_stage = next(
                (stage for stage, path in stage_outputs.items() if not path.is_file()),
                "fusion",
            )

    stage_status: dict[str, str] = {}
    reached_failure = False
    for stage, path in stage_outputs.items():
        if path.is_file():
            stage_status[stage] = "evaluated"
        elif failed_path.is_file() and stage == failure_stage:
            stage_status[stage] = "failed"
            reached_failure = True
        else:
            stage_status[stage] = "not_evaluated" if reached_failure or failed_path.is_file() else "not_evaluated"

    counts: dict[str, int | None] = {
        "alignment": None,
        "text": None,
        "audio": None,
        "visual": None,
    }
    alignment_path = stage_outputs["alignment"]
    if alignment_path.is_file():
        words = read_json(alignment_path).get("words", [])
        counts["alignment"] = sum(bool(word.get("alignment_mask")) for word in words)
    for stage, mask_name in (("text", "mask"), ("audio", "word_mask"), ("visual", "word_mask")):
        path = stage_outputs[stage]
        if path.is_file():
            with np.load(path, allow_pickle=False) as arrays:
                counts[stage] = int(np.asarray(arrays[mask_name], dtype=bool).sum())
    for stage in counts:
        if counts[stage] is None and stage_status[stage] == "failed":
            counts[stage] = 0

    def stage_rate(stage: str) -> float | None:
        count = counts[stage]
        return None if count is None else _safe_rate(count, record.word_count)

    return {
        "sample_id": record.sample_id,
        "read_status": record.read_status,
        "processing_status": "failed" if failed_path.is_file() else "not_processed",
        "original_word_count": record.word_count,
        "alignment_status": stage_status["alignment"],
        "text_status": stage_status["text"],
        "audio_status": stage_status["audio"],
        "visual_status": stage_status["visual"],
        "aligned_word_count": counts["alignment"],
        "alignment_coverage": stage_rate("alignment"),
        "text_valid_count": counts["text"],
        "text_valid_rate": stage_rate("text"),
        "audio_valid_count": counts["audio"],
        "audio_valid_rate": stage_rate("audio"),
        "visual_valid_count": counts["visual"],
        "visual_valid_rate": stage_rate("visual"),
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
        "alignment_status": "evaluated",
        "text_status": "evaluated",
        "audio_status": "evaluated",
        "visual_status": "evaluated",
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
        if str(row.get("boundary_status", "")).lower() in {
            "uncertain", "undeterminable", "cannot_determine", "unreviewed",
        }:
            continue
        sample_id = str(row["sample_id"])
        try:
            word_index = int(row["word_index"])
            reference_start = float(row["reference_start_s"])
            reference_end = float(row["reference_end_s"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(reference_start) and math.isfinite(reference_end)
                and 0 <= reference_start < reference_end):
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
    stage_rows = {
        stage: [row for row in rows if row.get(f"{stage}_status") in {"evaluated", "failed"}]
        for stage in ("alignment", "text", "audio", "visual")
    }

    def stage_total_words(stage: str) -> int:
        return sum(int(row.get("original_word_count") or 0) for row in stage_rows[stage])

    def stage_rate(stage: str, count_field: str) -> float | None:
        selected = stage_rows[stage]
        if not selected:
            return None
        return _safe_rate(
            sum(int(row.get(count_field) or 0) for row in selected),
            stage_total_words(stage),
        )

    aligned_words = sum(int(row.get("aligned_word_count") or 0) for row in stage_rows["alignment"])
    total_visual_frames = sum(int(row.get("sampled_visual_frame_count") or 0) for row in stage_rows["visual"])
    failed_faces = sum(int(row.get("face_detection_failure_count") or 0) for row in stage_rows["visual"])
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
        "alignment_evaluated_sample_count": len(stage_rows["alignment"]),
        "text_evaluated_sample_count": len(stage_rows["text"]),
        "audio_evaluated_sample_count": len(stage_rows["audio"]),
        "visual_evaluated_sample_count": len(stage_rows["visual"]),
        "automatic_alignment_coverage_not_accuracy": stage_rate("alignment", "aligned_word_count"),
        "text_word_valid_rate": stage_rate("text", "text_valid_count"),
        "audio_word_valid_rate": stage_rate("audio", "audio_valid_count"),
        "visual_word_valid_rate": stage_rate("visual", "visual_valid_count"),
        "face_detection_failure_rate": (
            _safe_rate(failed_faces, total_visual_frames) if stage_rows["visual"] else None
        ),
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
    word_rows = write_word_review_reports(
        [output_dir / "samples" / row["sample_id"] for row in selected_rows], output_dir,
    )
    automatic_samples = {str(row["sample_id"]) for row in candidates}
    summary["manual_reference_sample_count"] = len(reference_samples)
    summary["automatic_review_candidate_count"] = len(automatic_samples)
    summary["automatic_word_review_candidate_count"] = len(word_rows)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild Q1 quality reports from existing sample outputs")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--manual-reference-csv", type=Path, default=None)
    args = parser.parse_args()
    records, _ = load_samples(
        args.data_root, args.labels, expected_count=100,
        probe_decode=False, output_dir=args.output_dir,
    )
    known = {record.sample_id for record in records}
    missing = set(args.sample_id) - known
    if missing:
        raise ValueError(f"Unknown sample IDs: {sorted(missing)}")
    build_quality_reports(
        args.output_dir, records,
        manual_reference_csv=args.manual_reference_csv,
        selected_sample_ids=set(args.sample_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
