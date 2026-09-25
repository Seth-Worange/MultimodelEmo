"""Run a timestamp-matched MediaPipe478 baseline and compare real visual backends."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .alignment import WordAlignment
from .config import Q1Config
from .data_loader import load_samples
from .io_utils import read_json, write_csv, write_json
from .media import VideoFrameTiming
from .openface_backend import aggregate_openface_words
from .visual_backend import MediaPipe478Backend


FIELDS = (
    "sample_id", "backend", "sampling_fps", "sampled_frame_count",
    "face_success_frame_count", "face_success_rate", "feature_dimension",
    "runtime_s", "compressed_output_bytes", "frame_npz_bytes", "word_npz_bytes", "alignment_word_count",
    "word_visual_valid_count", "word_visual_coverage", "timestamp_mapping_reliable_count",
)


def _alignments(path: Path) -> list[WordAlignment]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [WordAlignment(
            sample_id=row["sample_id"], word_index=int(row["word_index"]),
            original_word=row["original_word"], normalized_word=row["normalized_word"],
            start_s=float(row["start_s"]), end_s=float(row["end_s"]),
            alignment_mask=int(row["alignment_mask"]), failure_reason=row.get("failure_reason", ""),
        ) for row in csv.DictReader(handle)]


def _word_counts(times: np.ndarray, valid: np.ndarray, alignments: Sequence[WordAlignment]) -> tuple[int, int]:
    if not alignments:
        return 0, 0
    count = 0
    for word in alignments:
        if word.alignment_mask and np.isfinite(word.start_s) and np.isfinite(word.end_s):
            count += int(np.any(valid & (times >= word.start_s) & (times <= word.end_s)))
    return len(alignments), count


def run_comparison(*, data_root: Path, round4_root: Path, output_dir: Path,
                   face_model: Path, sample_ids: Sequence[str], config: Q1Config) -> list[dict]:
    records, _ = load_samples(data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in records}
    rows = []
    for sample_id in sample_ids:
        if sample_id not in by_id:
            raise ValueError(f"Not an official sample: {sample_id}")
        source = round4_root / "samples" / sample_id
        target = output_dir / "samples" / sample_id / "mediapipe478"
        target.mkdir(parents=True, exist_ok=True)
        alignments = _alignments(source / "word_alignment.csv") if (source / "mfa_raw.json").is_file() else []
        result_path = target / "mediapipe_result.json"
        if not result_path.is_file():
            media = read_json(source / "media_metadata.json")
            frames = [VideoFrameTiming(**item) for item in media["video_frames"]]
            started = time.perf_counter()
            result = MediaPipe478Backend(face_model).extract_native(
                by_id[sample_id].video_path, frames, alignments, config, target,
            )
            write_json(result_path, {
                "runtime_s": time.perf_counter() - started,
                "sampled_frame_count": len(result.frame_times_s),
                "face_success_frame_count": int(result.detection_mask.sum()),
                "visual_fps": config.visual_fps,
            })
        mp_result = read_json(result_path)
        mp_npz = target / "visual_features.npz"
        with np.load(mp_npz, allow_pickle=False) as archive:
            mp_times = archive["frame_times_s"]
            mp_valid = archive["detection_mask"].astype(bool)
            mp_dim = int(archive["frame_features"].shape[1])
        n_words, n_valid = _word_counts(mp_times, mp_valid, alignments)
        rows.append({
            "sample_id": sample_id, "backend": "mediapipe478", "sampling_fps": config.visual_fps,
            "sampled_frame_count": len(mp_times), "face_success_frame_count": int(mp_valid.sum()),
            "face_success_rate": float(mp_valid.mean()) if len(mp_valid) else None,
            "feature_dimension": mp_dim, "runtime_s": mp_result["runtime_s"],
            "compressed_output_bytes": mp_npz.stat().st_size,
            "frame_npz_bytes": mp_npz.stat().st_size, "word_npz_bytes": 0,
            "alignment_word_count": n_words if n_words else None,
            "word_visual_valid_count": n_valid if n_words else None,
            "word_visual_coverage": n_valid / n_words if n_words else None,
            "timestamp_mapping_reliable_count": len(mp_times),
        })
        of_dir = output_dir / "samples" / sample_id / "openface68"
        of_result = read_json(of_dir / "openface_result.json")
        aggregate_elapsed = 0.0
        if alignments:
            aggregate_started = time.perf_counter()
            aggregate_openface_words(of_dir / "openface68_frames.npz", alignments, of_dir)
            aggregate_elapsed = time.perf_counter() - aggregate_started
        with np.load(of_dir / "openface68_frames.npz", allow_pickle=False) as archive:
            of_times = archive["sample_times_s"]
            of_valid = archive["face_valid_mask"].astype(bool)
            of_dim = int(archive["landmark_geometry"].shape[1])
        of_n, of_w = _word_counts(of_times, of_valid, alignments)
        rows.append({
            "sample_id": sample_id, "backend": "openface68", "sampling_fps": config.visual_fps,
            "sampled_frame_count": len(of_times), "face_success_frame_count": int(of_valid.sum()),
            "face_success_rate": float(of_valid.mean()) if len(of_valid) else None,
            "feature_dimension": of_dim, "runtime_s": of_result["runtime_s"] + aggregate_elapsed,
            "compressed_output_bytes": of_result["compressed_output_bytes"] +
                ((of_dir / "openface68_word_features.npz").stat().st_size if alignments else 0),
            "frame_npz_bytes": of_result["compressed_output_bytes"],
            "word_npz_bytes": (of_dir / "openface68_word_features.npz").stat().st_size if alignments else 0,
            "alignment_word_count": of_n if of_n else None,
            "word_visual_valid_count": of_w if of_n else None,
            "word_visual_coverage": of_w / of_n if of_n else None,
            "timestamp_mapping_reliable_count": of_result["timestamp_mapping_reliable_count"],
        })
        print({"sample_id": sample_id, "mediapipe_face_success": int(mp_valid.sum()),
               "openface_face_success": int(of_valid.sum())}, flush=True)
    out = output_dir / "visual_backend"
    write_csv(out / "visual_backend_comparison.csv", rows, FIELDS)
    write_json(out / "visual_backend_comparison_notes.json", {
        "sample_count": len(sample_ids), "backend_sample_rows": len(rows),
        "frame_sampling": "Both use the same original PyAV PTS frame selection at 10 FPS",
        "timing": "Runtime measured independently for each backend on this machine, not isolated benchmark",
        "word_coverage": "Only real Round-4 MFA words; no alignment means null, not zero",
        "identity": "A detected face is not verified as the speaker",
    })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    args = parser.parse_args(argv)
    if len(args.sample_id) > 12:
        parser.error("Representative comparison capped at 12 samples")
    run_comparison(data_root=args.data_root, round4_root=args.round4_root,
                   output_dir=args.output_dir, face_model=args.face_model,
                   sample_ids=args.sample_id, config=Q1Config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
