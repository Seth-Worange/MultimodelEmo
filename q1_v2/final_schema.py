"""Frozen, explicit Q1 Round-5.1 package schemas and integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Q1Config
from .io_utils import read_json, write_json


SCHEMA_VERSION = "q1_v2.5.1-openface-blocks-v1"
OPENFACE_FRAME_DIMS = {"landmark_geometry": 136, "action_units": 35, "head_pose": 6, "gaze": 8}
OPENFACE_WORD_DIMS = {name: dimension * 2 for name, dimension in OPENFACE_FRAME_DIMS.items()}


def visual_schema(backend: str, config: Q1Config) -> dict[str, Any]:
    if backend == "openface68":
        blocks = [
            {"feature_name": name, "frame_dimension": OPENFACE_FRAME_DIMS[name],
             "word_dimension": OPENFACE_WORD_DIMS[name], "source_backend": backend,
             "pooling": "in-interval valid frames; mean then population std"}
            for name in OPENFACE_FRAME_DIMS
        ]
        blocks[0]["normalization"] = (
            "2-D pixels translated to midpoint of OpenFace eye centres 36..41/42..47 "
            "and divided by interocular Euclidean distance; no roll rotation"
        )
        blocks[1]["normalization"] = "native OpenFace AU intensity (_r) and presence (_c); no MediaPipe blendshape"
        blocks[2]["normalization"] = "native OpenFace pose_Tx/Ty/Tz/Rx/Ry/Rz"
        blocks[3]["normalization"] = "native OpenFace gaze_0/1_xyz and gaze_angle_xy"
        return {"schema_version": SCHEMA_VERSION, "backend": backend, "visual_fps": config.visual_fps,
                "blocks": blocks, "total_word_dimension_if_ordered_concatenation_is_needed": 370,
                "stored_as_separate_blocks": True,
                "word_mask_meaning": "at least one successful true in-interval face frame; not verified speaker",
                "frame_time_source": "original PyAV PTS cross-checked with OpenFace timestamp"}
    if backend == "mediapipe478":
        return {"schema_version": SCHEMA_VERSION, "backend": backend, "visual_fps": config.visual_fps,
                "blocks": [
                    {"feature_name": "landmark_geometry", "frame_dimension": 1434, "word_dimension": 2868,
                     "normalization": "native MediaPipe 478 xyz after face translation/scale", "source_backend": backend},
                    {"feature_name": "blendshape", "frame_dimension": 52, "word_dimension": 104,
                     "normalization": "native MediaPipe Face Landmarker coefficients", "source_backend": backend},
                ], "stored_as_separate_blocks": True,
                "word_mask_meaning": "true in-interval detected face; not verified speaker",
                "frame_time_source": "original PyAV PTS"}
    raise ValueError(f"Unknown backend: {backend}")


def feature_schema(backend: str, config: Q1Config) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "sample_unit": "official (video_id,clip_id)",
        "word_axis": "all original official words in order; never truncated",
        "quality_router": "prespecified Round-4 HIGH_CONFIDENCE_MATCH only permits local ASR+MFA",
        "alignment_confidence": "MFA-ASR boundary disagreement uncertainty; not human accuracy",
        "text_word_dimension": 768, "audio_frame_dimension": config.n_mels + 5,
        "audio_word_dimension": 2 * (config.n_mels + 5),
        "visual_backend": backend, "visual": visual_schema(backend, config),
        "package_arrays": {
            "sample_id": "scalar Unicode", "quality_status": "scalar Unicode",
            "speech_status": "scalar Unicode: SPEECH_PRESENT/NO_SPEECH/UNKNOWN",
            "language_status": "scalar Unicode: ENGLISH/NON_ENGLISH/NOT_APPLICABLE/UNKNOWN",
            "face_status": "scalar Unicode: FACE_PRESENT/NO_FACE/UNKNOWN",
            "quality_metadata_block": "metadata JSON quality{correspondence_status,speech_status,language_status,face_status}",
            "word_index": "int32 [W]", "word_text": "Unicode [W]",
            "word_start_s": "float64 [W], NaN when unavailable",
            "word_end_s": "float64 [W], NaN when unavailable",
            "alignment_confidence": "Unicode [W]: HIGH/MEDIUM/LOW/UNAVAILABLE",
            "text_features": "float32 [W,768]", "audio_features": "float32 [W,138]",
            "visual_word_<block>": "float32 [W,block_word_dimension]",
            "visual_frame_<block>": "float32 [F,block_frame_dimension]",
            "audio_frame_features": "float32 [A,69]",
            "audio_frame_times_s": "float64 [A]",
            "visual_frame_indices": "int32/int64 [F] original decoded frame indices",
            "visual_frame_times_s": "float64 [F] sample-relative PyAV PTS",
            "text_available": "uint8 [W]", "audio_available": "uint8 [W]",
            "visual_available": "uint8 [W]", "alignment_available": "uint8 [W]",
            "visual_frame_valid": "uint8 [F]",
            "has_face": "uint8 scalar; at least one valid selected face frame",
            "usable_face": "int8 scalar, -1 means not evaluated",
            "extra_speech_before/after": "int8 scalar, -1 means no reliable evaluation",
        },
        "trace": "metadata JSON stores per-word audio frame indices and original video frame indices",
        "npz_loading": "np.load(path, allow_pickle=False)",
        "missing_policy": "NaN feature values and zero masks; no synthetic time or nearest-frame coverage",
        "speaker_identity_verified": False,
    }


def write_root_schemas(output_dir: Path, backend: str, config: Q1Config) -> None:
    write_json(output_dir / "visual_feature_schema.json", visual_schema(backend, config))
    write_json(output_dir / "feature_schema.json", feature_schema(backend, config))


def validate_package(sample_dir: Path, sample_id: str, *, backend: str) -> list[str]:
    path = sample_dir / "feature_package.npz"
    meta_path = sample_dir / "feature_package_metadata.json"
    errors = []
    if not path.is_file() or not meta_path.is_file():
        return ["missing_package_or_metadata"]
    try:
        meta = read_json(meta_path)
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["sample_id"].item()) != sample_id:
                errors.append("sample_id_mismatch")
            if str(archive["visual_backend"].item()) != backend:
                errors.append("visual_backend_mismatch")
            for name in ("quality_status", "speech_status", "language_status", "face_status"):
                if name not in archive.files:
                    errors.append(f"missing_quality_scalar:{name}")
            words = archive["word_text"]
            frames = archive["visual_frame_times_s"]
            for name in ("word_index", "word_start_s", "word_end_s", "alignment_confidence",
                         "text_features", "audio_features", "text_available", "audio_available",
                         "visual_available", "alignment_available"):
                if len(archive[name]) != len(words):
                    errors.append(f"word_length_mismatch:{name}")
            for name in ("visual_frame_indices", "visual_frame_valid"):
                if len(archive[name]) != len(frames):
                    errors.append(f"frame_length_mismatch:{name}")
            for block in visual_schema(backend, Q1Config())["blocks"]:
                name = block["feature_name"]
                if archive[f"visual_frame_{name}"].shape != (len(frames), block["frame_dimension"]):
                    errors.append(f"visual_frame_block_shape:{name}")
                if archive[f"visual_word_{name}"].shape != (len(words), block["word_dimension"]):
                    errors.append(f"visual_word_block_shape:{name}")
            if len(frames) and np.any(np.diff(frames) < 0):
                errors.append("nonmonotone_visual_frame_times")
            if meta["sample_id"] != sample_id or meta["word_count"] != len(words):
                errors.append("metadata_identity_or_word_count")
            quality = meta.get("quality")
            if not isinstance(quality, dict) or set(quality) != {
                    "correspondence_status", "speech_status", "language_status", "face_status"}:
                errors.append("metadata_quality_block")
            if len(meta["visual_word_original_frame_indices"]) != len(words):
                errors.append("visual_word_trace_length")
            if len(meta["audio_word_frame_indices"]) != len(words):
                errors.append("audio_word_trace_length")
    except Exception as exc:
        errors.append(f"unreadable_package:{type(exc).__name__}:{exc}")
    return errors
