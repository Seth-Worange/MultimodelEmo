"""Read back selected real round-4 NPZ files without pickle and trace frames."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_loader import SampleRecord, load_samples
from .io_utils import read_json, write_csv, write_json


FIELDS = (
    "sample_id", "status", "route", "word_count", "alignment_valid_count",
    "text_valid_count", "audio_valid_count", "visual_valid_count",
    "audio_frame_count", "visual_frame_count", "text_dim", "audio_frame_dim",
    "audio_word_dim", "visual_frame_dim", "visual_word_dim", "error_count", "errors",
)


def check_one(record: SampleRecord, sample_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    values: dict[str, Any] = {"sample_id": record.sample_id, "status": "failed", "errors": errors}
    try:
        qa = read_json(sample_dir / "correspondence_qa.json")
        media = read_json(sample_dir / "media_metadata.json")
        metadata = read_json(sample_dir / "fused_metadata.json")
        if not (sample_dir / "_SUCCESS.json").is_file():
            errors.append("missing_success_marker")
        with np.load(sample_dir / "fused_features.npz", allow_pickle=False) as arrays:
            data = {name: arrays[name].copy() for name in arrays.files}
        n = record.word_count
        original = [word.text for word in record.original_words]
        if str(data["sample_id"].item()) != record.sample_id:
            errors.append("sample_id_mismatch")
        if data["original_words"].tolist() != original:
            errors.append("official_word_sequence_mismatch")
        for name in (
            "word_start_s", "word_end_s", "text_word_features", "audio_word_features",
            "visual_word_features", "text_mask", "audio_mask", "visual_mask", "alignment_mask",
        ):
            if len(data[name]) != n:
                errors.append(f"word_length_mismatch:{name}")
        if errors:
            raise ValueError("Basic word length/ID integrity failed")
        if "correspondence_status" not in data or str(data["correspondence_status"].item()) != qa["correspondence_status"]:
            errors.append("qa_status_mismatch")
        alignment = data["alignment_mask"].astype(bool)
        text_mask = data["text_mask"].astype(bool)
        audio_mask = data["audio_mask"].astype(bool)
        visual_mask = data["visual_mask"].astype(bool)
        if qa["route"] != "HIGH_CONFIDENCE_MATCH" and np.any(alignment):
            errors.append("invalid_route_has_word_times")
        if np.any(audio_mask & ~alignment) or np.any(visual_mask & ~alignment):
            errors.append("word_modality_mask_without_time")
        audio_times = data["audio_frame_times_s"]
        visual_times = data["visual_frame_times_s"]
        source_video_frames = {int(frame["frame_index"]): frame for frame in media["video_frames"]}
        if len(audio_times) and (not np.all(np.isfinite(audio_times)) or np.any(np.diff(audio_times) <= 0)):
            errors.append("bad_audio_frame_times")
        if len(visual_times) and (not np.all(np.isfinite(visual_times)) or np.any(np.diff(visual_times) <= 0)):
            errors.append("bad_visual_frame_times")
        if len(visual_times) != len(data["visual_original_frame_indices"]):
            errors.append("visual_source_index_length_mismatch")
        else:
            for index, (source_index, sample_time) in enumerate(zip(data["visual_original_frame_indices"], visual_times)):
                frame = source_video_frames.get(int(source_index))
                if frame is None or abs(float(frame["sample_time_s"]) - float(sample_time)) > 1e-6:
                    errors.append(f"visual_frame_pts_mapping_error:{index}")
        for i in range(n):
            start, end = float(data["word_start_s"][i]), float(data["word_end_s"][i])
            if alignment[i]:
                if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= float(media["duration_s"]) + .05):
                    errors.append(f"bad_word_interval:{i}")
            elif np.isfinite(start) or np.isfinite(end):
                errors.append(f"unaligned_word_has_time:{i}")
            for modality, mask, feature_key in (
                ("text", text_mask, "text_word_features"),
                ("audio", audio_mask, "audio_word_features"),
                ("visual", visual_mask, "visual_word_features"),
            ):
                feature = data[feature_key][i]
                if mask[i] and not np.any(np.isfinite(feature)):
                    errors.append(f"valid_mask_no_finite_feature:{modality}:{i}")
                if not mask[i] and modality != "text" and np.any(np.isfinite(feature)):
                    errors.append(f"invalid_mask_has_feature:{modality}:{i}")
            for modality, times in (("audio", audio_times), ("visual", visual_times)):
                indices = metadata[f"{modality}_word_frame_indices"][i]
                count = int(data[f"{modality}_word_frame_counts"][i])
                if len(indices) != count:
                    errors.append(f"word_frame_count_mismatch:{modality}:{i}")
                for index in indices:
                    if not (0 <= index < len(times) and alignment[i] and start <= float(times[index]) < end):
                        errors.append(f"word_frame_outside_interval:{modality}:{i}:{index}")
            source_indices = metadata["visual_word_original_frame_indices"][i]
            if source_indices != [int(data["visual_original_frame_indices"][j]) for j in metadata["visual_word_frame_indices"][i]]:
                errors.append(f"word_visual_source_indices_mismatch:{i}")
        if np.any(data["visual_word_estimated_mask"]):
            errors.append("estimated_visual_frames_present")
        values.update({
            "route": qa["route"], "word_count": n,
            "alignment_valid_count": int(alignment.sum()),
            "text_valid_count": int(text_mask.sum()),
            "audio_valid_count": int(audio_mask.sum()),
            "visual_valid_count": int(visual_mask.sum()),
            "audio_frame_count": len(audio_times),
            "visual_frame_count": len(visual_times),
            "text_dim": int(data["text_word_features"].shape[1]),
            "audio_frame_dim": int(data["audio_frame_features"].shape[1]),
            "audio_word_dim": int(data["audio_word_features"].shape[1]),
            "visual_frame_dim": int(data["visual_frame_features"].shape[1]),
            "visual_word_dim": int(data["visual_word_features"].shape[1]),
            "npz_allow_pickle_false": True,
        })
    except Exception as exc:
        errors.append(f"integrity_exception:{type(exc).__name__}:{exc}")
    values["status"] = "pass" if not errors else "fail"
    values["error_count"] = len(errors)
    values["errors"] = errors
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args(argv)
    records, _ = load_samples(args.data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in records}
    rows = [check_one(by_id[sample_id], args.output_dir / "samples" / sample_id) for sample_id in args.sample_id]
    write_csv(args.output_dir / "integrity_by_sample.csv", (
        {**row, "errors": " | ".join(row["errors"])} for row in rows
    ), FIELDS)
    write_json(args.output_dir / "integrity_report.json", {
        "checked_sample_count": len(rows),
        "pass_count": sum(row["status"] == "pass" for row in rows),
        "rows": rows,
    })
    for row in rows:
        print(row)
    return int(any(row["status"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
