"""Explicit visual-backend selector for representative Q1 extraction.

This isolated selector does not replace the Round-4 fused archive schema. A
future fusion version must name its chosen backend and dimensions explicitly.
"""

from __future__ import annotations

import argparse
import csv
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .alignment import WordAlignment
from .config import Q1Config
from .data_loader import load_samples
from .io_utils import read_json, write_json
from .media import VideoFrameTiming
from .openface_backend import aggregate_openface_words, run_one
from .visual_features import VisualFeatureResult, extract_visual_features


BACKENDS = ("mediapipe478", "openface68")


@dataclass
class VisualBackendResult:
    backend: str
    frame_blocks: dict[str, np.ndarray]
    word_blocks: dict[str, np.ndarray]
    frame_indices: np.ndarray
    frame_times_s: np.ndarray
    frame_mask: np.ndarray
    word_mask: np.ndarray
    word_sampled_frame_counts: np.ndarray
    word_valid_face_counts: np.ndarray
    word_original_frame_indices: list[list[int]]
    source_dir: Path


class VisualBackend(ABC):
    """Only entry point used by the final feature pipeline for visual work."""

    @abstractmethod
    def extract(
        self, video_path: Path, frame_timings: Sequence[VideoFrameTiming],
        alignments: Sequence[WordAlignment], config: Q1Config, output_dir: Path,
        *, media_metadata: Path,
    ) -> VisualBackendResult:
        raise NotImplementedError


class MediaPipe478Backend(VisualBackend):
    def __init__(self, face_model: Path):
        self.face_model = face_model

    def extract_native(
        self, video_path: Path, frame_timings: Sequence[VideoFrameTiming],
        alignments: Sequence[WordAlignment], config: Q1Config, output_dir: Path,
        *, selected_landmarks: Sequence[int] | None = None,
    ) -> VisualFeatureResult:
        """Compatibility bridge for the archived Q1 MediaPipe fusion schema."""
        return extract_visual_features(
            video_path, frame_timings, alignments,
            face_model_path=self.face_model,
            selected_landmarks=selected_landmarks or config.mediapipe_native_landmark_indices,
            sampling_fps=config.visual_fps, max_faces=config.max_faces,
            min_face_detection_confidence=config.face_validity_threshold,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            nearest_max_distance_s=None, output_dir=output_dir,
        )

    def extract(
        self, video_path: Path, frame_timings: Sequence[VideoFrameTiming],
        alignments: Sequence[WordAlignment], config: Q1Config, output_dir: Path,
        *, media_metadata: Path,
    ) -> VisualBackendResult:
        native = self.extract_native(video_path, frame_timings, alignments, config, output_dir)
        geometry_dim = len(config.mediapipe_native_landmark_indices) * 3
        frame = native.frame_features
        word = native.word_features
        return VisualBackendResult(
            backend="mediapipe478",
            frame_blocks={"landmark_geometry": frame[:, :geometry_dim],
                          "blendshape": frame[:, geometry_dim:]},
            word_blocks={
                "landmark_geometry": np.concatenate(
                    [word[:, :geometry_dim], word[:, frame.shape[1]:frame.shape[1] + geometry_dim]], axis=1),
                "blendshape": np.concatenate(
                    [word[:, geometry_dim:frame.shape[1]], word[:, frame.shape[1] + geometry_dim:]], axis=1),
            },
            frame_indices=native.original_frame_indices,
            frame_times_s=native.frame_times_s,
            frame_mask=native.detection_mask.astype(bool),
            word_mask=native.word_mask.astype(bool),
            word_sampled_frame_counts=native.word_frame_counts,
            word_valid_face_counts=native.word_valid_face_counts,
            word_original_frame_indices=native.word_original_frame_indices,
            source_dir=output_dir,
        )


class OpenFace68Backend(VisualBackend):
    def __init__(self, executable: Path):
        self.executable = executable

    def extract(
        self, video_path: Path, frame_timings: Sequence[VideoFrameTiming],
        alignments: Sequence[WordAlignment], config: Q1Config, output_dir: Path,
        *, media_metadata: Path,
    ) -> VisualBackendResult:
        if not output_dir.joinpath("openface_result.json").is_file():
            run_one(video_path=video_path, media_metadata=media_metadata,
                    output_dir=output_dir, executable=self.executable,
                    visual_fps=config.visual_fps,
                    confidence_min=config.openface_confidence_threshold,
                    timestamp_tolerance_s=config.openface_timestamp_tolerance_s)
        elif not output_dir.joinpath("openface68_frames.npz").is_file():
            raise RuntimeError("Existing OpenFace result lacks frame NPZ")
        aggregate_openface_words(output_dir / "openface68_frames.npz", alignments, output_dir)
        with np.load(output_dir / "openface68_frames.npz", allow_pickle=False) as archive:
            frame = {name: archive[name].copy() for name in
                     ("landmark_geometry", "action_units", "head_pose", "gaze")}
            indices = archive["original_frame_indices"].copy()
            times = archive["sample_times_s"].copy()
            frame_mask = (archive["face_valid_mask"] & archive["timestamp_mapping_mask"]).copy()
        with np.load(output_dir / "openface68_word_features.npz", allow_pickle=False) as archive:
            word = {name: archive[f"word_{name}"].copy() for name in frame}
            word_mask = archive["word_visual_mask"].copy()
            sampled = archive["word_sampled_frame_counts"].copy()
            valid_counts = archive["word_valid_face_frame_counts"].copy()
        with (output_dir / "openface68_word_trace.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            trace = list(csv.DictReader(handle))
        original_indices = [
            [int(item) for item in row["original_frame_indices"].split(",") if item]
            for row in trace
        ]
        return VisualBackendResult(
            backend="openface68", frame_blocks=frame, word_blocks=word,
            frame_indices=indices, frame_times_s=times, frame_mask=frame_mask,
            word_mask=word_mask, word_sampled_frame_counts=sampled,
            word_valid_face_counts=valid_counts,
            word_original_frame_indices=original_indices, source_dir=output_dir,
        )


def make_visual_backend(config: Q1Config, *, face_model: Path | None,
                        openface_executable: Path | None) -> VisualBackend:
    if config.visual_backend == "openface68":
        if openface_executable is None:
            raise ValueError("OpenFace68 requires an executable")
        return OpenFace68Backend(openface_executable)
    if config.visual_backend == "mediapipe478":
        if face_model is None:
            raise ValueError("MediaPipe478 requires a face model")
        return MediaPipe478Backend(face_model)
    raise ValueError(f"Unknown visual backend: {config.visual_backend}")


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
                sampling_fps=config.visual_fps,
                min_face_detection_confidence=config.face_validity_threshold,
                output_dir=output_dir,
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
                visual_fps=config.visual_fps,
                confidence_min=config.openface_confidence_threshold,
                timestamp_tolerance_s=config.openface_timestamp_tolerance_s)
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
