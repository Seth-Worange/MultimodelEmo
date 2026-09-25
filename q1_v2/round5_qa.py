"""Re-score saved Round-4 ASR/VAD evidence and run 5-FPS visual scene QA.

This command never executes MFA, BERT, full acoustic features, or 100-sample
visual feature extraction. Round-4 stage-1 text/audio routes remain frozen.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .config import Q1Config, collect_environment
from .correspondence_qa import vad_extra_speech
from .data_loader import SampleRecord, load_samples
from .io_utils import read_json, sha256_file, write_csv, write_json
from .visual_scene_qa import probe_visual_scene


AUDIT_FIELDS = (
    "sample_id", "stage1_correspondence_status", "stage1_route",
    "extra_speech_before", "extra_speech_after",
    "extra_speech_before_duration_s", "extra_speech_after_duration_s",
    "round4_extra_speech_before", "round4_extra_speech_after",
    "video_signal_present", "sampled_frame_count", "any_face_detected",
    "face_detection_rate", "multiple_face_frame_count", "usable_face_present",
    "usable_face_coverage", "static_visual", "static_visual_score",
    "nonhuman_or_no_face_scene", "visual_scene_status", "visual_probe_fps",
    "qa_elapsed_s", "qa_error",
)


def _manual_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate normalized manual QA IDs")
    return result


def _boolean(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    return None


def _direction(before: bool | None, after: bool | None) -> str | None:
    if before is None or after is None:
        return None
    return "BOTH" if before and after else "BEFORE" if before else "AFTER" if after else "NONE"


def _binary_metrics(pairs: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(human and predicted for human, predicted in pairs)
    fp = sum(not human and predicted for human, predicted in pairs)
    fn = sum(human and not predicted for human, predicted in pairs)
    tn = sum(not human and not predicted for human, predicted in pairs)
    return {
        "evaluated_count": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "accuracy": (tp + tn) / len(pairs) if pairs else None,
    }


def _extra_metrics(manual: dict[str, dict[str, str]], qa: dict[str, dict[str, Any]], target: Path) -> dict[str, Any]:
    eligible = {sample_id: row for sample_id, row in manual.items() if row["manual_extra_speech"] in {"NONE", "BEFORE", "AFTER", "BOTH"}}
    matched = {
        sample_id: row for sample_id, row in eligible.items()
        if sample_id in qa
        and qa[sample_id].get("stage1_route") == "HIGH_CONFIDENCE_MATCH"
        and _direction(qa[sample_id].get("extra_speech_before"), qa[sample_id].get("extra_speech_after")) is not None
    }
    confusion: Counter[tuple[str, str]] = Counter()
    old_confusion: Counter[tuple[str, str]] = Counter()
    before_pairs = []
    after_pairs = []
    old_before_pairs = []
    old_after_pairs = []
    for sample_id, row in matched.items():
        human = row["manual_extra_speech"]
        record = qa[sample_id]
        predicted = _direction(record["extra_speech_before"], record["extra_speech_after"])
        old = _direction(record["round4_extra_speech_before"], record["round4_extra_speech_after"])
        confusion[human, predicted] += 1
        if old is not None:
            old_confusion[human, old] += 1
        before_pairs.append((human in {"BEFORE", "BOTH"}, record["extra_speech_before"]))
        after_pairs.append((human in {"AFTER", "BOTH"}, record["extra_speech_after"]))
        if old is not None:
            old_before_pairs.append((human in {"BEFORE", "BOTH"}, record["round4_extra_speech_before"]))
            old_after_pairs.append((human in {"AFTER", "BOTH"}, record["round4_extra_speech_after"]))
    write_csv(target / "extra_speech_confusion_matrix.csv", (
        {"manual_direction": human, "round5_vad_direction": predicted, "count": count}
        for (human, predicted), count in sorted(confusion.items())
    ), ("manual_direction", "round5_vad_direction", "count"))
    write_csv(target / "extra_speech_round4_confusion_matrix.csv", (
        {"manual_direction": human, "round4_asr_direction": predicted, "count": count}
        for (human, predicted), count in sorted(old_confusion.items())
    ), ("manual_direction", "round4_asr_direction", "count"))
    summary = {
        "manual_definite_direction_count": len(eligible),
        "manual_uncertain_or_blank_count": len(manual) - len(eligible),
        "round5_comparable_matched_span_count": len(matched),
        "excluded_no_high_confidence_match_span_count": len(eligible) - len(matched),
        "round5_before": _binary_metrics(before_pairs),
        "round5_after": _binary_metrics(after_pairs),
        "round5_exact_direction_accuracy": sum(h == p for (h, p), count in confusion.items() for _ in range(count)) / len(matched) if matched else None,
        "round4_before_on_same_samples": _binary_metrics(old_before_pairs),
        "round4_after_on_same_samples": _binary_metrics(old_after_pairs),
        "round4_exact_direction_accuracy_on_same_samples": sum(h == p for (h, p), count in old_confusion.items() for _ in range(count)) / sum(old_confusion.values()) if old_confusion else None,
        "rule": "VAD cumulative speech duration outside matched span, ASR unmatched tokens supplementary",
        "not_ground_truth": "VAD and ASR predictions are compared against independent manual categorical review; no word-boundary accuracy implied",
    }
    write_json(target / "extra_speech_metrics.json", summary)
    return summary


def _face_metrics(manual: dict[str, dict[str, str]], qa: dict[str, dict[str, Any]], old_root: Path, target: Path) -> dict[str, Any]:
    pairs2: list[tuple[bool, bool]] = []
    pairs5: list[tuple[bool, bool]] = []
    usable_pairs: list[tuple[bool, bool]] = []
    disagreements: list[dict[str, Any]] = []
    eligible = 0
    for sample_id, row in manual.items():
        human = _boolean(row["manual_face_present"])
        if human is None:
            continue
        eligible += 1
        new = qa.get(sample_id)
        old_path = old_root / "samples" / sample_id / "correspondence_qa.json"
        if not new or not old_path.is_file():
            continue
        old = read_json(old_path)
        old_face = _boolean(old.get("face_present"))
        new_face = _boolean(new.get("any_face_detected"))
        usable = _boolean(new.get("usable_face_present"))
        if old_face is None or new_face is None or usable is None:
            continue
        pairs2.append((human, old_face))
        pairs5.append((human, new_face))
        usable_pairs.append((human, usable))
        if old_face != human or new_face != human or old_face != new_face:
            disagreements.append({
                "sample_id": sample_id, "manual_face_present": human,
                "round4_2fps_any_face": old_face, "round5_5fps_any_face": new_face,
                "round5_5fps_usable_face": usable,
                "round4_sampled_frame_count": old.get("sampled_frame_count"),
                "round5_sampled_frame_count": new.get("sampled_frame_count"),
                "round5_face_detection_rate": new.get("face_detection_rate"),
                "round5_visual_scene_status": new.get("visual_scene_status"),
            })
    write_csv(target / "face_qa_disagreements.csv", disagreements, (
        "sample_id", "manual_face_present", "round4_2fps_any_face",
        "round5_5fps_any_face", "round5_5fps_usable_face",
        "round4_sampled_frame_count", "round5_sampled_frame_count",
        "round5_face_detection_rate", "round5_visual_scene_status",
    ))
    metrics = {
        "manual_face_eligible_count": eligible,
        "same_sample_comparison_count": len(pairs5),
        "round4_2fps_any_face": _binary_metrics(pairs2),
        "round5_5fps_any_face": _binary_metrics(pairs5),
        "round5_5fps_usable_face_diagnostic_not_same_semantics": _binary_metrics(usable_pairs),
        "round5_scene_status_counts": dict(Counter(row["visual_scene_status"] for row in qa.values() if row.get("visual_scene_status"))),
        "static_vs_manual_static_or_nonhuman_accuracy": None,
        "reason_no_static_accuracy": "manual_static_or_nonhuman combines two concepts; static_visual is motion only",
    }
    write_json(target / "face_qa_metrics.json", metrics)
    return metrics


def audit_one(record: SampleRecord, *, round4_root: Path, output_root: Path, face_model: Path, config: Q1Config) -> dict[str, Any]:
    old_dir = round4_root / "samples" / record.sample_id
    new_dir = output_root / "samples" / record.sample_id
    new_dir.mkdir(parents=True, exist_ok=True)
    old_path = old_dir / "correspondence_qa.json"
    old_media = old_dir / "media_metadata.json"
    if not old_path.is_file() or not old_media.is_file():
        raise FileNotFoundError(f"Round-4 QA/media evidence missing for {record.sample_id}")
    old, media = read_json(old_path), read_json(old_media)
    if old["sample_id"] != record.sample_id or old["source_video_sha256"] != sha256_file(record.video_path):
        raise ValueError(f"Round-4 source mismatch for {record.sample_id}")
    started = time.perf_counter()
    extra = vad_extra_speech(
        old["vad_regions"], matched_start_s=old.get("matched_audio_start_s"),
        matched_end_s=old.get("matched_audio_end_s"),
        audio_start_s=float(media["wav_origin_sample_s"]),
        audio_end_s=float(media["decoded_audio_end_sample_s"]),
        tolerance_s=config.extra_speech_tolerance_s,
        min_duration_s=config.extra_speech_min_duration_s,
        asr_unmatched_before=old.get("local_match", {}).get("extra_speech_before"),
        asr_unmatched_after=old.get("local_match", {}).get("extra_speech_after"),
    )
    visual = probe_visual_scene(
        record.video_path, media["video_frames"], face_model_path=face_model,
        config=config, output_csv=new_dir / "visual_probe_frames_5fps.csv",
    )
    result = {
        "sample_id": record.sample_id,
        "source_video_sha256": old["source_video_sha256"],
        "stage1_correspondence_status": old["correspondence_status"],
        "stage1_route": old["route"],
        "stage1_text_audio_match_score": old["text_audio_match_score"],
        "matched_audio_start_s": old.get("matched_audio_start_s"),
        "matched_audio_end_s": old.get("matched_audio_end_s"),
        "round4_extra_speech_before": old.get("extra_speech_before"),
        "round4_extra_speech_after": old.get("extra_speech_after"),
        "round4_face_present": old.get("face_present"),
        **extra, **visual,
        "qa_elapsed_s": time.perf_counter() - started,
        "qa_error": None,
        "evidence_source": str(old_path),
    }
    write_json(new_dir / "correspondence_qa_round5.json", result)
    return result


def write_reports(records: Sequence[SampleRecord], *, round4_root: Path, output_root: Path, manual_path: Path) -> dict[str, Any]:
    rows = []
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        path = output_root / "samples" / record.sample_id / "correspondence_qa_round5.json"
        failure = output_root / "samples" / record.sample_id / "_QA_FAILED.json"
        qa = read_json(path) if path.is_file() else {}
        if qa:
            by_id[record.sample_id] = qa
        row = {field: qa.get(field) for field in AUDIT_FIELDS}
        row["sample_id"] = record.sample_id
        if failure.is_file():
            row["qa_error"] = read_json(failure)["error"]
        rows.append(row)
    target = output_root / "qa"
    write_csv(target / "correspondence_audit_round5.csv", rows, AUDIT_FIELDS)
    manual = _manual_rows(manual_path)
    extra = _extra_metrics(manual, by_id, target)
    face = _face_metrics(manual, by_id, round4_root, target)
    summary = {
        "official_sample_count": len(records), "round5_qa_completed_count": len(by_id),
        "round5_qa_failed_count": sum((output_root / "samples" / record.sample_id / "_QA_FAILED.json").is_file() for record in records),
        "stage1_route_unchanged": True,
        "extra_speech_evaluation": extra,
        "face_evaluation": face,
        "full_100_sample_feature_extraction": "NOT_RUN",
    }
    write_json(output_root / "quality_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-normalized-csv", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-id", action="append")
    group.add_argument("--all-qa", action="store_true")
    args = parser.parse_args(argv)
    records, _ = load_samples(args.data_root, probe_decode=False)
    if len(records) != 100 or len({record.sample_id for record in records}) != 100:
        raise ValueError("Official sample list is not 100 unique rows")
    config = Q1Config()
    if not args.face_model.is_file():
        raise FileNotFoundError(args.face_model)
    chosen = records if args.all_qa else [record for record in records if record.sample_id in set(args.sample_id)]
    if not chosen or args.sample_id and len(chosen) != len(set(args.sample_id)):
        raise ValueError("Requested sample ID absent from official set")
    write_json(args.output_dir / "experiment_config.json", {
        "visual_probe_fps": config.visual_probe_fps,
        "visual_feature_fps_not_executed_here": config.visual_fps,
        "visual_probe_backend": "mediapipe478",
        "visual_qa_min_face_rate": config.visual_qa_min_face_rate,
        "visual_qa_min_continuous_s": config.visual_qa_min_continuous_s,
        "visual_qa_min_face_area_ratio": config.visual_qa_min_face_area_ratio,
        "visual_qa_static_mean_difference_max": config.visual_qa_static_mean_difference_max,
        "extra_speech_tolerance_s": config.extra_speech_tolerance_s,
        "extra_speech_min_duration_s": config.extra_speech_min_duration_s,
        "face_model_sha256": sha256_file(args.face_model),
        "round4_root": str(args.round4_root.resolve()),
        "manual_normalized_source": str(args.manual_normalized_csv.resolve()),
        "environment": collect_environment(),
    })
    for number, record in enumerate(chosen, 1):
        target = args.output_dir / "samples" / record.sample_id
        if (target / "correspondence_qa_round5.json").is_file():
            print({"progress": f"{number}/{len(chosen)}", "sample_id": record.sample_id, "action": "skipped_existing"}, flush=True)
            continue
        try:
            result = audit_one(record, round4_root=args.round4_root, output_root=args.output_dir,
                               face_model=args.face_model, config=config)
            print({"progress": f"{number}/{len(chosen)}", "sample_id": record.sample_id,
                   "action": "processed", "scene": result["visual_scene_status"]}, flush=True)
        except Exception as exc:
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "_QA_FAILED.json", {"sample_id": record.sample_id,
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            print({"progress": f"{number}/{len(chosen)}", "sample_id": record.sample_id,
                   "action": "failed", "error": str(exc)}, flush=True)
    summary = write_reports(records, round4_root=args.round4_root, output_root=args.output_dir,
                            manual_path=args.manual_normalized_csv)
    print({"round5_qa_completed_count": summary["round5_qa_completed_count"],
           "round5_qa_failed_count": summary["round5_qa_failed_count"]})
    return int(summary["round5_qa_failed_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
