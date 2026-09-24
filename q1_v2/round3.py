"""Targeted OOV repair and fixed-alignment 5/10 FPS visual comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio_features import AudioFeatureResult, aggregate_audio_to_words
from .alignment import WordAlignment, locate_mfa_unknowns, parse_mfa_output, run_mfa_alignment
from .config import Q1Config
from .data_loader import split_original_words
from .io_utils import directory_size, read_json, sha256_file, write_csv, write_json
from .media import VideoFrameTiming
from .fusion_alignment import build_fused_sample, validate_fused_output
from .quality import (
    MANUAL_TEMPLATE_FIELDS, WORD_REVIEW_FIELDS, word_review_rows,
    write_word_review_reports,
)
from .text_features import TextFeatureResult
from .visual_features import VisualFeatureResult, extract_visual_features


TARGETS = {"-3g5yACwYnA__2": "polymer", "-3g5yACwYnA__3": "adhesives"}
SAMPLES = ("-3g5yACwYnA__13", *TARGETS)
PRONUNCIATIONS = (
    ("polymer", "P AA1 L AH0 M ER0"),
    ("adhesives", "AE0 D HH IY1 S IH0 V Z"),
    ("adhesives", "AH0 D HH IY1 S IH0 V Z"),
)
CMU_SOURCE = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
WORD_VISUAL_FIELDS = (
    "sample_id", "sampling_fps", "word_index", "original_word", "start_s",
    "end_s", "alignment_mask", "visual_mask", "interval_frame_count",
    "valid_face_count", "visual_status", "sampled_frame_times_s",
    "sampled_original_frame_indices", "alignment_source_sha256",
)
COMPARISON_FIELDS = (
    "sample_id", "sampling_fps", "original_word_count", "aligned_word_count",
    "visual_valid_word_count", "no_sampled_frame_word_count",
    "face_detection_failed_frame_count", "sampled_video_frame_count",
    "word_interval_frame_count_sum", "short_word_count", "short_word_visual_coverage",
    "medium_word_count", "medium_word_visual_coverage", "long_word_count",
    "long_word_visual_coverage", "elapsed_s", "alignment_source_sha256",
    "face_model_sha256",
)
DELTA_FIELDS = (
    "sample_id", "word_index", "original_word", "before_start_s", "before_end_s",
    "after_start_s", "after_end_s", "before_alignment_mask", "after_alignment_mask",
    "max_boundary_shift_s", "target_oov_word", "manual_review_required",
)
EXPORT_SIZE_FIELDS = (
    "sample_id", "baseline_research_bytes", "revised_mfa_bytes",
    "visual_10fps_bytes", "revised_fused_10fps_bytes",
    "full_research_components_bytes", "submission_export_bytes",
    "saved_bytes", "export_fraction", "full_dataset_limit_verified",
)


def build_target_dictionary(base_dictionary: Path, output_dir: Path) -> Path:
    """Copy the immutable MFA lexicon and append only documented CMU entries."""
    if not base_dictionary.is_file():
        raise FileNotFoundError(base_dictionary)
    words = set()
    with base_dictionary.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                words.add(line.split("\t", 1)[0].lower())
    already_present = [word for word in TARGETS.values() if word in words]
    if already_present:
        raise ValueError(f"Target words already in base dictionary: {already_present}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "english_us_arpa_plus_two.dict"
    if target.exists():
        raise FileExistsError(target)
    with base_dictionary.open("rb") as source, target.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
        destination.write(b"\n")
        for word, phones in PRONUNCIATIONS:
            destination.write(f"{word}\t{phones}\n".encode("ascii"))
    write_json(output_dir / "dictionary_change.json", {
        "base_dictionary": str(base_dictionary.resolve()),
        "base_sha256": sha256_file(base_dictionary),
        "supplemented_dictionary": str(target.resolve()),
        "supplemented_sha256": sha256_file(target),
        "target_words_absent_from_base": sorted(TARGETS.values()),
        "added_entries": [{"word": word, "arpabet": phones} for word, phones in PRONUNCIATIONS],
        "pronunciation_source": CMU_SOURCE,
        "transcript_policy": "Original contest transcription is unchanged",
        "time_accuracy_policy": "Re-aligned intervals require human listening review",
    })
    return target


def _alignment_from_csv(path: Path) -> list[WordAlignment]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [WordAlignment(
        sample_id=row["sample_id"], word_index=int(row["word_index"]),
        original_word=row["original_word"], normalized_word=row["normalized_word"],
        start_s=float(row["start_s"]) if row["start_s"] else float("nan"),
        end_s=float(row["end_s"]) if row["end_s"] else float("nan"),
        alignment_mask=int(row["alignment_mask"]), failure_reason=row["failure_reason"],
    ) for row in rows]


def realign_targets(baseline: Path, output: Path, dictionary: Path, mfa_work: Path) -> list[dict]:
    records = []
    for sample_id, target_word in TARGETS.items():
        source = baseline / "samples" / sample_id
        destination = output / "realigned" / "samples" / sample_id
        if destination.exists():
            diagnostic = destination / "target_word_diagnostic.json"
            if diagnostic.is_file() and (destination / "mfa_raw.json").is_file():
                previous = read_json(diagnostic)
                if previous.get("dictionary_sha256") != sha256_file(dictionary):
                    raise ValueError(f"Existing alignment dictionary differs for {sample_id}")
                records.append(previous)
                continue
            if not (destination / "_FAILED.json").is_file():
                raise FileExistsError(destination)
            if not destination.resolve().is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe archive target: {destination}")
            archive = output / "realigned" / "failed_attempts" / f"{sample_id}-{uuid.uuid4().hex[:8]}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(archive)
        destination.mkdir(parents=True, exist_ok=False)
        before_raw = source / "mfa_raw.json"
        before_csv = source / "word_alignment.csv"
        (destination / "before_mfa_raw.json").write_bytes(before_raw.read_bytes())
        (destination / "before_word_alignment.csv").write_bytes(before_csv.read_bytes())
        text = (source / "transcript_original.txt").read_text(encoding="utf-8")
        words = split_original_words(text)
        positions = [i for i, word in enumerate(words) if word.text.lower().strip(".,!?;") == target_word]
        if len(positions) != 1:
            raise ValueError(f"Expected one {target_word} in {sample_id}, got {positions}")
        target_index = positions[0]
        unknowns = locate_mfa_unknowns(words, parse_mfa_output(before_raw))
        if target_index not in unknowns:
            raise ValueError(f"Before MFA <unk> could not be uniquely anchored to {target_word}")
        media = read_json(source / "media_metadata.json")
        try:
            result = run_mfa_alignment(
                sample_id=sample_id, original_text=text, original_words=words,
                wav_path=source / "audio_mfa_16k_mono.wav", output_dir=destination,
                acoustic_model="english_us_arpa", dictionary=str(dictionary.resolve()),
                wav_origin_media_s=float(media["wav_origin_media_s"]),
                timeline_origin_media_s=float(media["timeline_origin_media_s"]),
                duration_s=float(media["duration_s"]), temporary_directory=mfa_work,
            )
        except Exception as exc:
            write_json(destination / "_FAILED.json", {"sample_id": sample_id, "error": repr(exc)})
            raise
        before = _alignment_from_csv(before_csv)[target_index]
        after = result.words[target_index]
        after_unknowns = locate_mfa_unknowns(words, result.mfa_words)
        row = {
            "sample_id": sample_id, "word_index": target_index,
            "original_word": words[target_index].text,
            "before_mfa_unknown_start_s": unknowns[target_index].start_s,
            "before_mfa_unknown_end_s": unknowns[target_index].end_s,
            "before_alignment_mask": before.alignment_mask,
            "after_start_s": after.start_s if after.alignment_mask else None,
            "after_end_s": after.end_s if after.alignment_mask else None,
            "after_alignment_mask": after.alignment_mask,
            "after_failure_reason": after.failure_reason,
            "after_mfa_unknown_for_word": target_index in after_unknowns,
            "original_word_count": len(words),
            "after_word_count": len(result.words),
            "manual_boundary_review": "pending",
            "dictionary_sha256": sha256_file(dictionary),
            "before_raw_sha256": sha256_file(before_raw),
            "after_raw_sha256": sha256_file(destination / "mfa_raw.json"),
        }
        write_json(destination / "target_word_diagnostic.json", row)
        records.append(row)
    write_csv(output / "oov_before_after.csv", records, tuple(records[0]))
    return records


def _coverage_by_duration(alignments: Sequence[WordAlignment], visual_mask: np.ndarray) -> dict:
    ranges = {"short": (0.0, 0.2), "medium": (0.2, 0.5), "long": (0.5, float("inf"))}
    output = {}
    for label, (lower, upper) in ranges.items():
        indices = [i for i, word in enumerate(alignments)
                   if word.alignment_mask and lower <= word.end_s - word.start_s < upper]
        output[f"{label}_word_count"] = len(indices)
        output[f"{label}_word_visual_coverage"] = (
            float(np.mean(visual_mask[indices])) if indices else None
        )
    return output


def compare_visual(baseline: Path, output: Path, face_model: Path) -> list[dict]:
    config = Q1Config()
    comparison = []
    model_sha = sha256_file(face_model)
    for sample_id in SAMPLES:
        source = baseline / "samples" / sample_id
        alignment_path = (
            output / "realigned" / "samples" / sample_id / "word_alignment.csv"
            if sample_id in TARGETS else source / "word_alignment.csv"
        )
        alignments = _alignment_from_csv(alignment_path)
        input_audit = read_json(source / "sample_input.json")
        video_path = Path(input_audit["video_path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        frame_timings = [VideoFrameTiming(**row) for row in read_json(source / "media_metadata.json")["video_frames"]]
        alignment_sha = sha256_file(alignment_path)
        for fps in (5, 10):
            sample_dir = output / f"visual_{fps}fps" / "samples" / sample_id
            if sample_dir.exists():
                raise FileExistsError(sample_dir)
            started = time.perf_counter()
            try:
                visual = extract_visual_features(
                    video_path, frame_timings, alignments,
                    face_model_path=face_model,
                    selected_landmarks=config.selected_landmarks,
                    sampling_fps=float(fps),
                    max_faces=config.max_faces,
                    min_face_detection_confidence=config.min_face_detection_confidence,
                    min_face_presence_confidence=config.min_face_presence_confidence,
                    min_tracking_confidence=config.min_tracking_confidence,
                    nearest_max_distance_s=None, output_dir=sample_dir,
                )
            except Exception as exc:
                write_json(sample_dir / "_FAILED.json", {"sample_id": sample_id, "fps": fps, "error": repr(exc)})
                raise
            elapsed = time.perf_counter() - started
            per_word = []
            for i, word in enumerate(alignments):
                indices = visual.word_frame_indices[i]
                per_word.append({
                    "sample_id": sample_id, "sampling_fps": fps,
                    "word_index": i, "original_word": word.original_word,
                    "start_s": word.start_s if word.alignment_mask else "",
                    "end_s": word.end_s if word.alignment_mask else "",
                    "alignment_mask": word.alignment_mask,
                    "visual_mask": int(visual.word_mask[i]),
                    "interval_frame_count": int(visual.word_frame_counts[i]),
                    "valid_face_count": int(visual.word_valid_face_counts[i]),
                    "visual_status": visual.word_status[i],
                    "sampled_frame_times_s": json.dumps([float(visual.frame_times_s[j]) for j in indices]),
                    "sampled_original_frame_indices": json.dumps(visual.word_original_frame_indices[i]),
                    "alignment_source_sha256": alignment_sha,
                })
            write_csv(sample_dir / "visual_word_audit.csv", per_word, WORD_VISUAL_FIELDS)
            row = {
                "sample_id": sample_id, "sampling_fps": fps,
                "original_word_count": len(alignments),
                "aligned_word_count": sum(word.alignment_mask for word in alignments),
                "visual_valid_word_count": int(visual.word_mask.sum()),
                "no_sampled_frame_word_count": sum(
                    word.alignment_mask and count == 0
                    for word, count in zip(alignments, visual.word_frame_counts)
                ),
                "face_detection_failed_frame_count": int((visual.detection_mask == 0).sum()),
                "sampled_video_frame_count": len(visual.frame_times_s),
                "word_interval_frame_count_sum": int(visual.word_frame_counts.sum()),
                "elapsed_s": elapsed,
                "alignment_source_sha256": alignment_sha,
                "face_model_sha256": model_sha,
                **_coverage_by_duration(alignments, visual.word_mask),
            }
            write_json(sample_dir / "comparison_metrics.json", row)
            comparison.append(row)
    write_csv(output / "visual_5_10_comparison.csv", comparison, COMPARISON_FIELDS)
    return comparison


def write_round3_quality(baseline: Path, output: Path, review_rows: Sequence[dict]) -> dict:
    deltas = []
    for sample_id in TARGETS:
        before = _alignment_from_csv(baseline / "samples" / sample_id / "word_alignment.csv")
        after = _alignment_from_csv(output / "realigned" / "samples" / sample_id / "word_alignment.csv")
        if len(before) != len(after):
            raise ValueError(f"Re-alignment lost words in {sample_id}")
        for old, new in zip(before, after):
            if old.original_word != new.original_word or old.word_index != new.word_index:
                raise ValueError(f"Re-alignment changed word order in {sample_id}")
            shift = (max(abs(old.start_s - new.start_s), abs(old.end_s - new.end_s))
                     if old.alignment_mask and new.alignment_mask else None)
            deltas.append({
                "sample_id": sample_id, "word_index": old.word_index,
                "original_word": old.original_word,
                "before_start_s": old.start_s if old.alignment_mask else "",
                "before_end_s": old.end_s if old.alignment_mask else "",
                "after_start_s": new.start_s if new.alignment_mask else "",
                "after_end_s": new.end_s if new.alignment_mask else "",
                "before_alignment_mask": old.alignment_mask,
                "after_alignment_mask": new.alignment_mask,
                "max_boundary_shift_s": shift if shift is not None else "",
                "target_oov_word": int(old.normalized_word == TARGETS[sample_id]),
                "manual_review_required": int(old.normalized_word == TARGETS[sample_id] or (shift is not None and shift > .2)),
            })
    write_csv(output / "realignment_word_deltas.csv", deltas, DELTA_FIELDS)
    visual_path = output / "visual_5_10_comparison.csv"
    visual_rows = []
    if visual_path.is_file():
        with visual_path.open("r", encoding="utf-8-sig", newline="") as handle:
            visual_rows = list(csv.DictReader(handle))
    unchanged = _alignment_from_csv(baseline / "samples" / "-3g5yACwYnA__13" / "word_alignment.csv")
    target_word_count = len(deltas)
    target_before_aligned = sum(int(row["before_alignment_mask"]) for row in deltas)
    target_after_aligned = sum(int(row["after_alignment_mask"]) for row in deltas)
    summary = {
        "experiment_scope": "3 original samples; only 2 OOV samples were re-aligned",
        "original_word_count": len(unchanged) + target_word_count,
        "baseline_aligned_word_count": sum(word.alignment_mask for word in unchanged) + target_before_aligned,
        "revised_aligned_word_count": sum(word.alignment_mask for word in unchanged) + target_after_aligned,
        "baseline_mfa_unknown_word_count": 2,
        "word_review_candidate_count": len(review_rows),
        "boundary_shift_over_200ms_count": sum(
            isinstance(row["max_boundary_shift_s"], float) and row["max_boundary_shift_s"] > .2
            for row in deltas
        ),
        "manual_boundary_accuracy": None,
        "manual_reference_word_count": 0,
        "visual_by_fps": {
            str(fps): {
                "sample_count": sum(int(row["sampling_fps"]) == fps for row in visual_rows),
                "visual_valid_words": sum(int(row["visual_valid_word_count"]) for row in visual_rows if int(row["sampling_fps"]) == fps),
                "no_sampled_frame_words": sum(int(row["no_sampled_frame_word_count"]) for row in visual_rows if int(row["sampling_fps"]) == fps),
                "face_detection_failed_frames": sum(int(row["face_detection_failed_frame_count"]) for row in visual_rows if int(row["sampling_fps"]) == fps),
            }
            for fps in (5, 10)
        } if visual_rows else None,
        "accuracy_note": "Automatic MFA coverage is not boundary accuracy; revised OOV words and shifted boundaries await human listening review.",
    }
    write_json(output / "quality_summary.json", summary)
    baseline_experiment = read_json(baseline / "experiment_config.json")
    dictionary_change = read_json(output / "dictionary" / "dictionary_change.json")
    probe = read_json(output / "realigned" / "samples" / "-3g5yACwYnA__2" / "mfa_probe.json")
    write_json(output / "round3_experiment_config.json", {
        "baseline_root": str(baseline.resolve()),
        "baseline_config_sha256": baseline_experiment.get("config_sha256"),
        "environment": baseline_experiment.get("environment"),
        "mfa_executable": probe.get("executable"),
        "mfa_version": str(probe.get("version", {}).get("stdout", "")).strip(),
        "acoustic_model": "english_us_arpa",
        "base_dictionary_sha256": dictionary_change["base_sha256"],
        "supplemented_dictionary_sha256": dictionary_change["supplemented_sha256"],
        "face_model_sha256": visual_rows[0]["face_model_sha256"] if visual_rows else None,
        "sampling_fps": [5, 10],
        "nearest_frame_estimation": False,
        "comparison_alignment_policy": "Both FPS use the identical revised word_alignment.csv for target samples and unchanged baseline alignment for sample 13",
        "manual_boundary_reference_available": False,
    })
    return summary


def write_round3_word_review(baseline: Path, output: Path) -> list[dict]:
    baseline_rows = write_word_review_reports(
        [baseline / "samples" / sample_id for sample_id in SAMPLES],
        output / "baseline_review", revised_alignment_root=output / "realigned",
    )
    for row in baseline_rows:
        row["source_variant"] = "baseline_5fps"
    write_csv(output / "baseline_review" / "word_review_candidates.csv", baseline_rows, WORD_REVIEW_FIELDS)
    current_dirs = [output / "revised_fused_5fps" / "samples" / sample_id for sample_id in SAMPLES]
    rows = [row for sample_dir in current_dirs for row in word_review_rows(sample_dir)]
    by_key = {(str(row["sample_id"]), int(row["word_index"])): row for row in rows}
    current_arrays = {}
    current_metadata = {}
    for sample_id in SAMPLES:
        source = output / "revised_fused_5fps" / "samples" / sample_id
        with np.load(source / "fused_features.npz", allow_pickle=False) as arrays:
            current_arrays[sample_id] = {
                key: arrays[key].copy() for key in (
                    "original_words", "word_start_s", "word_end_s", "alignment_mask",
                    "audio_mask", "visual_mask", "visual_word_frame_counts",
                )
            }
        current_metadata[sample_id] = read_json(source / "fused_metadata.json")
    for old in baseline_rows:
        issues = str(old["issue_type"])
        if "mfa_unknown" not in issues and "realignment_boundary_shift" not in issues:
            continue
        sample_id, index = str(old["sample_id"]), int(old["word_index"])
        key = (sample_id, index)
        if key not in by_key:
            values = current_arrays[sample_id]
            mask = bool(values["alignment_mask"][index])
            row = {
                "sample_id": sample_id, "word_index": index,
                "original_word": str(values["original_words"][index]),
                "start_s": float(values["word_start_s"][index]) if mask else "",
                "end_s": float(values["word_end_s"][index]) if mask else "",
                "alignment_mask": int(mask),
                "audio_mask": int(values["audio_mask"][index]),
                "visual_mask": int(values["visual_mask"][index]),
                "issue_type": "", "review_priority": "high", "review_reason": "",
                "visual_frame_count": int(values["visual_word_frame_counts"][index]),
                "visual_status": current_metadata[sample_id]["visual_word_status"][index],
            }
            rows.append(row)
            by_key[key] = row
        row = by_key[key]
        additions = []
        if "mfa_unknown" in issues:
            additions.append("prior_mfa_unknown_requires_manual_review")
            row["mfa_unknown_start_s"] = old.get("mfa_unknown_start_s", "")
            row["mfa_unknown_end_s"] = old.get("mfa_unknown_end_s", "")
        if "realignment_boundary_shift" in issues:
            additions.append("realignment_boundary_shift")
            row["realignment_max_boundary_shift_s"] = old.get("realignment_max_boundary_shift_s", "")
        row["issue_type"] = " | ".join(filter(None, [row.get("issue_type", ""), *additions]))
        row["review_priority"] = "high"
        row["review_reason"] = " | ".join(filter(None, [
            row.get("review_reason", ""),
            "Prior <unk> or >200 ms re-alignment boundary shift; current interval needs human listening review",
        ]))
    for row in rows:
        row["source_variant"] = "revised_5fps"
        row["revised_start_s"] = row.get("start_s", "")
        row["revised_end_s"] = row.get("end_s", "")
        row["revised_alignment_mask"] = row.get("alignment_mask", "")
        row["manual_boundary_review_status"] = "pending"
    rows.sort(key=lambda row: (str(row["sample_id"]), int(row["word_index"])))
    write_csv(output / "word_review_candidates.csv", rows, WORD_REVIEW_FIELDS)
    template = [{
        "sample_id": row["sample_id"], "word_index": row["word_index"],
        "original_word": row["original_word"], "reference_start_s": "",
        "reference_end_s": "", "reviewer_id": "", "boundary_status": "unreviewed",
        "review_note": "", "automatic_revised_start_s": row.get("start_s", ""),
        "automatic_revised_end_s": row.get("end_s", ""),
    } for row in rows]
    write_csv(output / "word_review_reference_template.csv", template, MANUAL_TEMPLATE_FIELDS)
    return rows


def rebuild_fused_samples(baseline: Path, output: Path) -> list[dict]:
    """Re-aggregate unchanged text/audio frames using revised word intervals."""
    rows = []
    for fps in (5, 10):
        for sample_id in SAMPLES:
            source = baseline / "samples" / sample_id
            alignment_path = (
                output / "realigned" / "samples" / sample_id / "word_alignment.csv"
                if sample_id in TARGETS else source / "word_alignment.csv"
            )
            raw_path = alignment_path.parent / "mfa_raw.json"
            visual_dir = output / f"visual_{fps}fps" / "samples" / sample_id
            target = output / f"revised_fused_{fps}fps" / "samples" / sample_id
            if target.exists():
                raise FileExistsError(target)
            words = split_original_words((source / "transcript_original.txt").read_text(encoding="utf-8"))
            alignments = _alignment_from_csv(alignment_path)
            with np.load(source / "text_features.npz", allow_pickle=False) as arrays:
                text_features = arrays["features"].copy()
                text_mask = arrays["mask"].copy()
            text_metadata = read_json(source / "text_features_metadata.json")
            text = TextFeatureResult(
                features=text_features, mask=text_mask, token_mappings=[],
                feature_dim=text_features.shape[1], valid_length=int(text_mask.sum()),
                model_name=text_metadata.get("model_name", ""),
                model_revision=text_metadata.get("model_revision"),
                pooling=text_metadata.get("pooling", "mean"), windows=[],
            )
            with np.load(source / "audio_features.npz", allow_pickle=False) as arrays:
                audio_arrays = {name: arrays[name].copy() for name in arrays.files}
            audio_metadata = read_json(source / "audio_features_metadata.json")
            aggregated = aggregate_audio_to_words(
                audio_arrays["frame_features"], audio_arrays["frame_times_s"],
                audio_arrays["f0_valid_mask"], alignments,
            )
            audio = AudioFeatureResult(
                frame_features=audio_arrays["frame_features"],
                frame_times_s=audio_arrays["frame_times_s"],
                frame_start_samples=audio_arrays["frame_start_samples"],
                f0_valid_mask=audio_arrays["f0_valid_mask"],
                f0_analysis_times_s=audio_arrays["f0_analysis_times_s"],
                f0_analysis_hz=audio_arrays["f0_analysis_hz"],
                f0_analysis_valid_mask=audio_arrays["f0_analysis_valid_mask"],
                feature_names=audio_metadata["feature_names"],
                word_features=aggregated[0], word_mask=aggregated[1],
                word_frame_indices=aggregated[2], word_frame_counts=aggregated[3],
                word_f0_valid_counts=aggregated[4], word_feature_valid_counts=aggregated[5],
                word_failure_reasons=aggregated[6],
                sample_rate=int(audio_metadata["sample_rate"]),
                window_samples=int(audio_metadata["window_samples"]),
                hop_samples=int(audio_metadata["hop_samples"]),
                f0_window_samples=int(audio_metadata["f0_window_samples"]),
            )
            with np.load(visual_dir / "visual_features.npz", allow_pickle=False) as arrays:
                visual_arrays = {name: arrays[name].copy() for name in arrays.files}
            visual_metadata = read_json(visual_dir / "visual_features_metadata.json")
            visual = VisualFeatureResult(
                frame_features=visual_arrays["frame_features"],
                frame_times_s=visual_arrays["frame_times_s"],
                original_frame_indices=visual_arrays["original_frame_indices"],
                target_times_s=visual_arrays["target_times_s"],
                target_time_errors_s=visual_arrays["target_time_errors_s"],
                mediapipe_timestamps_ms=visual_arrays["mediapipe_timestamps_ms"],
                detection_mask=visual_arrays["detection_mask"],
                face_counts=visual_arrays["face_counts"],
                speaker_identity_mask=visual_arrays["speaker_identity_mask"],
                frame_status=visual_metadata["frame_status"],
                word_features=visual_arrays["word_features"],
                word_mask=visual_arrays["word_mask"],
                word_identity_mask=visual_arrays["word_identity_mask"],
                word_estimated_mask=visual_arrays["word_estimated_mask"],
                word_frame_indices=visual_metadata["word_frame_indices"],
                word_original_frame_indices=visual_metadata["word_original_frame_indices"],
                word_frame_counts=visual_arrays["word_frame_counts"],
                word_valid_face_counts=visual_arrays["word_valid_face_counts"],
                word_status=visual_metadata["word_status"],
                selected_landmarks=visual_metadata["selected_landmarks"],
                blendshape_names=visual_metadata["blendshape_names"],
                sampling_fps=float(fps),
            )
            build_fused_sample(sample_id, words, alignments, text, audio, visual, output_dir=target)
            for name, original in (
                ("word_alignment.csv", alignment_path),
                ("mfa_raw.json", raw_path),
                ("sample_input.json", source / "sample_input.json"),
            ):
                shutil.copy2(original, target / name)
            write_json(target / "rebuild_provenance.json", {
                "sample_id": sample_id, "sampling_fps": fps,
                "text_source_sha256": sha256_file(source / "text_features.npz"),
                "audio_frame_source_sha256": sha256_file(source / "audio_features.npz"),
                "alignment_source_sha256": sha256_file(alignment_path),
                "visual_source_sha256": sha256_file(visual_dir / "visual_features.npz"),
                "method": "Text unchanged; audio word statistics re-aggregated from original timestamped frames; visual words use measured in-interval frames",
                "manual_boundary_review": "pending for revised OOV and shifted boundaries",
            })
            write_json(target / "_SUCCESS.json", {
                "sample_id": sample_id, "status": "success", "schema_version": "q1_v2.1-round3",
                "elapsed_s": None,
            })
            valid, errors = validate_fused_output(target, sample_id)
            if not valid:
                raise ValueError(f"Rebuilt fused output invalid: {sample_id}, {fps}, {errors}")
            rows.append({
                "sample_id": sample_id, "sampling_fps": fps,
                "word_count": len(words),
                "alignment_valid_count": int(sum(word.alignment_mask for word in alignments)),
                "text_valid_count": int(text.mask.sum()),
                "audio_valid_count": int(audio.word_mask.sum()),
                "visual_valid_count": int(visual.word_mask.sum()),
                "fused_sha256": sha256_file(target / "fused_features.npz"),
            })
    write_csv(output / "revised_fused_integrity.csv", rows, tuple(rows[0]))
    return rows


def write_revision_export_sizes(baseline: Path, output: Path) -> list[dict]:
    export_root = output / "submission_export_10fps" / "samples"
    if not export_root.is_dir():
        return []
    rows = []
    for sample_id in SAMPLES:
        parts = {
            "baseline_research_bytes": directory_size(baseline / "samples" / sample_id),
            "revised_mfa_bytes": (
                directory_size(output / "realigned" / "samples" / sample_id)
                if sample_id in TARGETS else 0
            ),
            "visual_10fps_bytes": directory_size(output / "visual_10fps" / "samples" / sample_id),
            "revised_fused_10fps_bytes": directory_size(output / "revised_fused_10fps" / "samples" / sample_id),
        }
        export_bytes = directory_size(export_root / sample_id)
        if export_bytes == 0:
            raise FileNotFoundError(export_root / sample_id)
        full_bytes = sum(parts.values())
        rows.append({
            "sample_id": sample_id, **parts,
            "full_research_components_bytes": full_bytes,
            "submission_export_bytes": export_bytes,
            "saved_bytes": full_bytes - export_bytes,
            "export_fraction": export_bytes / full_bytes,
            "full_dataset_limit_verified": False,
        })
    write_csv(output / "submission_export_10fps" / "composite_size_report.csv", rows, EXPORT_SIZE_FIELDS)
    return rows


def verify_revised_outputs(baseline: Path, output: Path) -> dict:
    checks = []
    errors = []
    for sample_id in SAMPLES:
        media = read_json(baseline / "samples" / sample_id / "media_metadata.json")
        duration = float(media["duration_s"])
        source_words = [word.text for word in split_original_words(
            (baseline / "samples" / sample_id / "transcript_original.txt").read_text(encoding="utf-8")
        )]
        for fps in (5, 10):
            sample_dir = output / f"revised_fused_{fps}fps" / "samples" / sample_id
            valid, validation_errors = validate_fused_output(sample_dir, sample_id)
            errors.extend(f"{sample_id}/{fps}:{error}" for error in validation_errors)
            if not valid:
                continue
            metadata = read_json(sample_dir / "fused_metadata.json")
            with np.load(sample_dir / "fused_features.npz", allow_pickle=False) as arrays:
                words = arrays["original_words"].astype(str).tolist()
                n = len(words)
                if words != source_words:
                    errors.append(f"{sample_id}/{fps}:original_word_order_changed")
                expected_dims = {"text_word_features": 768, "audio_word_features": 138, "visual_word_features": 332}
                for name, dimension in expected_dims.items():
                    if arrays[name].shape != (n, dimension):
                        errors.append(f"{sample_id}/{fps}:{name}_shape={arrays[name].shape}")
                alignment_mask = arrays["alignment_mask"].astype(bool)
                starts = arrays["word_start_s"]
                ends = arrays["word_end_s"]
                audio_times = arrays["audio_frame_times_s"]
                visual_times = arrays["visual_frame_times_s"]
                original_video_indices = arrays["visual_original_frame_indices"]
                for i in range(n):
                    if alignment_mask[i]:
                        if not (np.isfinite(starts[i]) and np.isfinite(ends[i]) and
                                -.05 <= starts[i] < ends[i] <= duration + .05):
                            errors.append(f"{sample_id}/{fps}:invalid_word_time:{i}")
                    elif np.isfinite(starts[i]) or np.isfinite(ends[i]):
                        errors.append(f"{sample_id}/{fps}:failed_word_has_time:{i}")
                    for modality, key in (("text", "text_word_features"), ("audio", "audio_word_features"), ("visual", "visual_word_features")):
                        mask = bool(arrays[f"{modality}_mask"][i])
                        values = arrays[key][i]
                        if mask and not np.any(np.isfinite(values)):
                            errors.append(f"{sample_id}/{fps}:{modality}_masked_valid_but_all_nan:{i}")
                        if not mask and modality != "text" and np.any(np.isfinite(values)):
                            errors.append(f"{sample_id}/{fps}:{modality}_masked_invalid_but_finite:{i}")
                    for modality, times in (("audio", audio_times), ("visual", visual_times)):
                        indices = metadata[f"{modality}_word_frame_indices"][i]
                        if len(indices) != int(arrays[f"{modality}_word_frame_counts"][i]):
                            errors.append(f"{sample_id}/{fps}:{modality}_frame_count_mismatch:{i}")
                        for index in indices:
                            if not 0 <= index < len(times) or not starts[i] <= times[index] < ends[i]:
                                errors.append(f"{sample_id}/{fps}:{modality}_frame_time_mismatch:{i}:{index}")
                    visual_local = metadata["visual_word_frame_indices"][i]
                    visual_original = metadata["visual_word_original_frame_indices"][i]
                    if visual_original != [int(original_video_indices[j]) for j in visual_local]:
                        errors.append(f"{sample_id}/{fps}:video_source_index_mismatch:{i}")
                if np.any(arrays["visual_word_estimated_mask"]):
                    errors.append(f"{sample_id}/{fps}:estimated_video_frames_present")
                checks.append({
                    "sample_id": sample_id, "fps": fps, "word_count": n,
                    "alignment_valid_count": int(alignment_mask.sum()),
                    "text_valid_count": int(arrays["text_mask"].sum()),
                    "audio_valid_count": int(arrays["audio_mask"].sum()),
                    "visual_valid_count": int(arrays["visual_mask"].sum()),
                    "audio_frame_count": len(audio_times),
                    "video_frame_count": len(visual_times),
                    "npz_allow_pickle_false": True,
                })
    result = {"checked_outputs": len(checks), "errors": errors, "checks": checks,
              "status": "pass" if len(checks) == 6 and not errors else "fail"}
    write_json(output / "revised_fused_integrity_audit.json", result)
    if errors or len(checks) != 6:
        raise ValueError(f"Revised fused integrity failed: {errors}; checked={len(checks)}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dictionary", "realign", "visual", "fusion", "review", "verify"), required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-dictionary", type=Path)
    parser.add_argument("--face-model", type=Path)
    parser.add_argument("--mfa-root-dir", type=Path)
    parser.add_argument("--mfa-work-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mfa_root_dir:
        os.environ["MFA_ROOT_DIR"] = str(args.mfa_root_dir.resolve())
    if args.phase == "dictionary":
        if not args.base_dictionary:
            parser.error("--base-dictionary is required")
        build_target_dictionary(args.base_dictionary, output / "dictionary")
    elif args.phase == "realign":
        if not args.mfa_work_dir:
            parser.error("--mfa-work-dir is required")
        realign_targets(args.baseline_root, output, output / "dictionary" / "english_us_arpa_plus_two.dict", args.mfa_work_dir)
    elif args.phase == "visual":
        if not args.face_model:
            parser.error("--face-model is required")
        compare_visual(args.baseline_root, output, args.face_model)
    elif args.phase == "fusion":
        rebuild_fused_samples(args.baseline_root, output)
    elif args.phase == "verify":
        verify_revised_outputs(args.baseline_root, output)
    else:
        review_rows = write_round3_word_review(args.baseline_root, output)
        write_round3_quality(args.baseline_root, output, review_rows)
        write_revision_export_sizes(args.baseline_root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
