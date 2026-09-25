"""Selected full-feature quality, separate from the 100-sample light QA."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_loader import SampleRecord, load_samples
from .io_utils import read_json, sha256_file, write_csv, write_json
from .quality import MANUAL_TEMPLATE_FIELDS, QUALITY_FIELDS, WORD_REVIEW_FIELDS, sample_quality, word_review_rows


SAMPLE_REVIEW_FIELDS = (
    "sample_id", "correspondence_status", "route", "face_present",
    "extra_speech_before", "extra_speech_after", "manual_review_required",
    "review_priority", "review_reason",
)


def _backfill_metadata(record: SampleRecord, sample_dir: Path) -> None:
    input_path = sample_dir / "sample_input.json"
    if not input_path.is_file():
        write_json(input_path, {
            "sample_id": record.sample_id, "video_id": record.video_id,
            "clip_id": record.clip_id, "source_row": record.source_row,
            "text": record.text, "video_path": str(record.video_path.resolve()),
            "video_sha256": sha256_file(record.video_path),
            "original_words": [word.text for word in record.original_words],
            "label_usage": "audit-only",
        })
    alignment_meta = sample_dir / "alignment_metadata.json"
    if not alignment_meta.is_file():
        with (sample_dir / "word_alignment.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != record.word_count:
            raise ValueError(f"Word length mismatch in {sample_dir}")
        write_json(alignment_meta, {
            "sample_id": record.sample_id,
            "status": "ok" if all(row["alignment_mask"] == "1" for row in rows) else (
                "partial" if any(row["alignment_mask"] == "1" for row in rows) else "unavailable"
            ),
            "source": "real_local_mfa" if (sample_dir / "mfa_raw.json").is_file() else "no_trusted_alignment",
            "words": rows,
        })


def write_selected_quality(records: Sequence[SampleRecord], output_dir: Path) -> dict:
    selected = [record for record in records if (output_dir / "samples" / record.sample_id / "_SUCCESS.json").is_file()]
    rows = []
    reviews = []
    template = []
    for record in selected:
        sample_dir = output_dir / "samples" / record.sample_id
        _backfill_metadata(record, sample_dir)
        quality = sample_quality(record, sample_dir)
        quality["processing_elapsed_s"] = read_json(sample_dir / "_SUCCESS.json").get("feature_elapsed_s")
        rows.append(quality)
        qa = read_json(sample_dir / "correspondence_qa.json")
        if qa["route"] != "HIGH_CONFIDENCE_MATCH":
            continue  # No artificial word-level boundary review for mismatched media.
        reviews.extend(word_review_rows(sample_dir))
        with (sample_dir / "word_alignment.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            alignment_rows = list(csv.DictReader(handle))
        for row in alignment_rows:
            template.append({
                "sample_id": record.sample_id, "word_index": row["word_index"],
                "original_word": row["original_word"],
                "reference_start_s": "", "reference_end_s": "", "reviewer_id": "",
                "boundary_status": "unreviewed", "review_note": "",
                "automatic_revised_start_s": row["start_s"] if row["alignment_mask"] == "1" else "",
                "automatic_revised_end_s": row["end_s"] if row["alignment_mask"] == "1" else "",
            })
    manual_detail_path = output_dir / "manual_validation" / "manual_alignment_detail.csv"
    if manual_detail_path.is_file():
        with manual_detail_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manual_detail = list(csv.DictReader(handle))
        by_key = {(row["sample_id"], int(row["word_index"])): row for row in reviews}
        for manual in manual_detail:
            if not manual["max_error"] or float(manual["max_error"]) <= .2:
                continue
            key = (manual["sample_id"], int(manual["word_index"]))
            issue = "manual_boundary_error_over_200ms"
            row = by_key.get(key)
            if row is None:
                with np.load(output_dir / "samples" / key[0] / "fused_features.npz", allow_pickle=False) as arrays:
                    alignment_mask = int(arrays["alignment_mask"][key[1]])
                    audio_mask = int(arrays["audio_mask"][key[1]])
                    visual_mask = int(arrays["visual_mask"][key[1]])
                row = {
                    "sample_id": key[0], "word_index": key[1],
                    "original_word": manual["original_word"],
                    "start_s": manual["auto_start"], "end_s": manual["auto_end"],
                    "alignment_mask": alignment_mask,
                    "audio_mask": audio_mask, "visual_mask": visual_mask,
                    "issue_type": issue, "review_priority": "high",
                    "review_reason": "Provided human boundary differs by >200 ms; listening recheck requested",
                    "manual_boundary_review_status": "provided_numeric_reference_reviewer_id_blank",
                    "source_variant": "round4_local_mfa",
                }
                reviews.append(row)
                by_key[key] = row
            else:
                row["issue_type"] = " | ".join(filter(None, [row.get("issue_type", ""), issue]))
                row["review_priority"] = "high"
                row["review_reason"] = " | ".join(filter(None, [row.get("review_reason", ""), "Provided human boundary differs by >200 ms"]))
    reviews.sort(key=lambda row: (str(row["sample_id"]), int(row["word_index"])))
    write_csv(output_dir / "quality_by_sample.csv", rows, QUALITY_FIELDS)
    write_csv(output_dir / "word_review_candidates.csv", reviews, WORD_REVIEW_FIELDS)
    write_csv(output_dir / "manual_validation" / "word_reference_template.csv", template, MANUAL_TEMPLATE_FIELDS)
    sample_reviews = []
    for record in records:
        qa_path = output_dir / "samples" / record.sample_id / "correspondence_qa.json"
        if not qa_path.is_file():
            continue
        qa = read_json(qa_path)
        issues = []
        if qa["route"] != "HIGH_CONFIDENCE_MATCH":
            issues.append(qa["correspondence_reason"])
        if qa.get("extra_speech_before"):
            issues.append("extra_speech_before_official_text")
        if qa.get("extra_speech_after"):
            issues.append("extra_speech_after_official_text")
        if qa.get("face_present") is False:
            issues.append("no_face_in_low_rate_probe")
        if qa.get("static_or_near_static_visual") is True:
            issues.append("static_or_near_static_visual")
        if issues:
            sample_reviews.append({
                "sample_id": record.sample_id,
                "correspondence_status": qa["correspondence_status"],
                "route": qa["route"],
                "face_present": qa.get("face_present"),
                "extra_speech_before": qa.get("extra_speech_before"),
                "extra_speech_after": qa.get("extra_speech_after"),
                "manual_review_required": True,
                "review_priority": "high" if qa["route"] != "HIGH_CONFIDENCE_MATCH" else "medium",
                "review_reason": " | ".join(issues),
            })
    write_csv(output_dir / "sample_review_candidates.csv", sample_reviews, SAMPLE_REVIEW_FIELDS)
    feature_manifest = []
    for record in selected:
        sample_dir = output_dir / "samples" / record.sample_id
        visual_metadata = read_json(sample_dir / "visual_features_metadata.json")
        audio_metadata = read_json(sample_dir / "audio_features_metadata.json")
        text_metadata = read_json(sample_dir / "text_features_metadata.json")
        native_path = sample_dir / "native_feature_reuse_provenance.json"
        feature_manifest.append({
            "sample_id": record.sample_id,
            "correspondence_status": read_json(sample_dir / "correspondence_qa.json")["correspondence_status"],
            "visual_fps": visual_metadata["sampling_fps"],
            "visual_topology": "MediaPipe-native-478-not-OpenFace68",
            "visual_frame_dim": visual_metadata["frame_feature_dim"],
            "text_model": text_metadata["model_name"],
            "audio_sample_rate": audio_metadata["sample_rate"],
            "audio_frame_dim": audio_metadata["frame_feature_dim"],
            "native_feature_reuse": read_json(native_path) if native_path.is_file() else None,
            "local_crop": read_json(sample_dir / "mfa_crop_metadata.json") if (sample_dir / "mfa_crop_metadata.json").is_file() else None,
            "feature_elapsed_s": read_json(sample_dir / "_SUCCESS.json").get("feature_elapsed_s"),
        })
    write_json(output_dir / "feature_experiment_manifest.json", {
        "selected_sample_count": len(feature_manifest),
        "full_100_sample_extraction": "NOT_RUN",
        "samples": feature_manifest,
    })
    summary = {
        "scope": "only selected real samples with completed full features",
        "sample_count": len(rows),
        "original_word_count": sum(row["original_word_count"] for row in rows),
        "aligned_word_count": sum(row["aligned_word_count"] for row in rows),
        "text_valid_word_count": sum(row["text_valid_count"] for row in rows),
        "audio_valid_word_count": sum(row["audio_valid_count"] for row in rows),
        "visual_valid_word_count": sum(row["visual_valid_count"] for row in rows),
        "word_review_candidate_count": len(reviews),
        "sample_review_candidate_count": len(sample_reviews),
        "boundary_accuracy_not_inferred_from_coverage": True,
    }
    write_json(output_dir / "selected_feature_quality_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, _ = load_samples(args.data_root, probe_decode=False)
    print(write_selected_quality(records, args.output_dir))


if __name__ == "__main__":
    main()
