"""Explicit visual-backend selector for representative Q1 extraction.

This isolated selector does not replace the Round-4 fused archive schema. A
future fusion version must name its chosen backend and dimensions explicitly.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Sequence

from .alignment import WordAlignment
from .config import Q1Config
from .data_loader import load_samples
from .io_utils import read_json, write_json
from .media import VideoFrameTiming
from .openface_backend import aggregate_openface_words, run_one
from .visual_features import extract_visual_features


BACKENDS = ("mediapipe478", "openface68")


def _read_real_alignments(source: Path) -> list[WordAlignment]:
    if not (source / "mfa_raw.json").is_file():
        return []
    path = source / "word_alignment.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [WordAlignment(
            sample_id=row["sample_id"], word_index=int(row["word_index"]),
            original_word=row["original_word"], normalized_word=row["normalized_word"],
            start_s=float(row["start_s"]), end_s=float(row["end_s"]),
            alignment_mask=int(row["alignment_mask"]), failure_reason=row.get("failure_reason", ""),
        ) for row in csv.DictReader(handle)]


def extract_with_backend(
    *, backend: str, video_path: Path, media_metadata: Path,
    output_dir: Path, alignments: Sequence[WordAlignment], config: Q1Config,
    face_model: Path | None = None, openface_executable: Path | None = None,
) -> dict:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown visual backend: {backend}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "mediapipe478":
        if face_model is None:
            raise ValueError("MediaPipe backend requires --face-model")
        result_path = output_dir / "mediapipe_result.json"
        if result_path.is_file() and not (output_dir / "visual_features.npz").is_file():
            raise RuntimeError("Existing MediaPipe result lacks NPZ; retained, not overwritten")
        if not result_path.is_file():
            media = read_json(media_metadata)
            frames = [VideoFrameTiming(**row) for row in media["video_frames"]]
            started = time.perf_counter()
            result = extract_visual_features(
                video_path, frames, alignments, face_model_path=face_model,
                selected_landmarks=config.mediapipe_native_landmark_indices,
                sampling_fps=config.visual_fps, output_dir=output_dir,
            )
            write_json(result_path, {
                "backend": backend, "frame_dimension": int(result.frame_features.shape[1]),
                "sampled_frame_count": len(result.frame_times_s),
                "face_success_frame_count": int(result.detection_mask.sum()),
                "runtime_s": time.perf_counter() - started, "visual_fps": config.visual_fps,
            })
        return read_json(result_path)
    if openface_executable is None:
        raise ValueError("OpenFace backend requires --openface-executable")
    result_path = output_dir / "openface_result.json"
    if result_path.is_file() and not (output_dir / "openface68_frames.npz").is_file():
        raise RuntimeError("Existing OpenFace result lacks NPZ; retained, not overwritten")
    if not result_path.is_file():
        run_one(video_path=video_path, media_metadata=media_metadata,
                output_dir=output_dir, executable=openface_executable,
                visual_fps=config.visual_fps)
    if alignments:
        aggregate_openface_words(output_dir / "openface68_frames.npz", alignments, output_dir)
    return read_json(result_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default=Q1Config().final_visual_backend_candidate)
    parser.add_argument("--face-model", type=Path)
    parser.add_argument("--openface-executable", type=Path)
    args = parser.parse_args(argv)
    if len(args.sample_id) > 12:
        parser.error("Representative backend extraction capped at 12 samples")
    official, _ = load_samples(args.data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in official}
    for sample_id in args.sample_id:
        if sample_id not in by_id:
            parser.error(f"Not an official sample: {sample_id}")
        source = args.round4_root / "samples" / sample_id
        target = args.output_dir / "samples" / sample_id / args.backend
        result = extract_with_backend(
            backend=args.backend, video_path=by_id[sample_id].video_path,
            media_metadata=source / "media_metadata.json", output_dir=target,
            alignments=_read_real_alignments(source), config=Q1Config(),
            face_model=args.face_model, openface_executable=args.openface_executable,
        )
        print({"sample_id": sample_id, "backend": args.backend,
               "sampled_frame_count": result["sampled_frame_count"]}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
