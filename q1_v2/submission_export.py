"""Small, auditable Q1 submission view of complete research outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .fusion_alignment import validate_fused_output
from .io_utils import directory_size, read_json, sha256_file, write_csv, write_json


SIZE_FIELDS = (
    "sample_id", "research_bytes", "export_bytes", "saved_bytes", "size_ratio",
    "word_count", "text_dim", "audio_word_dim", "visual_word_dim",
    "audio_frame_count", "visual_frame_count", "status",
)


def export_sample(source: Path, target: Path) -> dict:
    sample_id = source.name
    valid, errors = validate_fused_output(source, sample_id)
    if not valid:
        raise ValueError(f"Invalid source {sample_id}: {errors}")
    if target.exists():
        raise FileExistsError(f"Export target already exists: {target}")
    with np.load(source / "fused_features.npz", allow_pickle=False) as arrays:
        mandatory = (
            "sample_id", "original_words", "word_start_s", "word_end_s",
            "text_word_features", "audio_word_features", "visual_word_features",
            "text_mask", "audio_mask", "visual_mask", "alignment_mask",
            "audio_frame_features", "audio_frame_times_s", "audio_frame_f0_valid_mask",
            "visual_frame_features", "visual_frame_times_s",
            "visual_original_frame_indices", "visual_detection_mask",
            "audio_word_frame_counts", "visual_word_frame_counts",
        )
        missing = set(mandatory) - set(arrays.files)
        if missing:
            raise ValueError(f"Missing essential arrays: {sorted(missing)}")
        word_count = len(arrays["original_words"])
        if str(arrays["sample_id"].item()) != sample_id:
            raise ValueError("NPZ sample_id mismatch")
        for key in ("word_start_s", "word_end_s", "text_mask", "audio_mask", "visual_mask", "alignment_mask"):
            if len(arrays[key]) != word_count:
                raise ValueError(f"{key} length mismatch")
        info = {
            "word_count": word_count,
            "text_dim": int(arrays["text_word_features"].shape[1]),
            "audio_word_dim": int(arrays["audio_word_features"].shape[1]),
            "visual_word_dim": int(arrays["visual_word_features"].shape[1]),
            "audio_frame_count": len(arrays["audio_frame_times_s"]),
            "visual_frame_count": len(arrays["visual_frame_times_s"]),
            "source_npz_members": arrays.files,
        }
    target.mkdir(parents=True, exist_ok=False)
    for name in ("fused_features.npz", "fused_metadata.json", "word_alignment.csv"):
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, target / name)
    input_audit = read_json(source / "sample_input.json")
    write_json(target / "provenance.json", {
        "sample_id": sample_id,
        "source_video_id": input_audit["video_id"],
        "source_clip_id": input_audit["clip_id"],
        "source_video_sha256": input_audit["video_sha256"],
        "source_fused_sha256": sha256_file(source / "fused_features.npz"),
        "source_mfa_raw_sha256": sha256_file(source / "mfa_raw.json"),
        "frame_index_semantics": "audio local decoded frame index; visual_original_frame_indices are decoded video frame indices",
        "research_output_retained_at": str(source.resolve()),
        "omitted": ["WAV", "model caches", "modality-specific NPZ copies", "raw MFA JSON and logs"],
        "array_members": info["source_npz_members"],
    })
    with np.load(target / "fused_features.npz", allow_pickle=False) as check:
        if str(check["sample_id"].item()) != sample_id or len(check["original_words"]) != word_count:
            raise ValueError("Export re-read validation failed")
    research_bytes = directory_size(source)
    export_bytes = directory_size(target)
    return {
        "sample_id": sample_id, "research_bytes": research_bytes,
        "export_bytes": export_bytes, "saved_bytes": research_bytes - export_bytes,
        "size_ratio": export_bytes / research_bytes if research_bytes else None,
        "status": "ok", **{key: info[key] for key in SIZE_FIELDS if key in info},
    }


def export_samples(source_root: Path, output_root: Path, sample_ids: Sequence[str]) -> list[dict]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = [
        export_sample(source_root / "samples" / sample_id, output_root / "samples" / sample_id)
        for sample_id in sample_ids
    ]
    experiment = source_root / "experiment_config.json"
    if experiment.is_file():
        shutil.copy2(experiment, output_root / "experiment_config.json")
    write_csv(output_root / "size_report.csv", rows, SIZE_FIELDS)
    write_json(output_root / "export_summary.json", {
        "sample_count": len(rows),
        "research_sample_bytes": sum(row["research_bytes"] for row in rows),
        "export_sample_bytes": sum(row["export_bytes"] for row in rows),
        "full_100_sample_limit_checked": False,
        "note": "These measured sizes describe only the selected samples.",
    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args()
    export_samples(args.source_root, args.output_dir, args.sample_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
