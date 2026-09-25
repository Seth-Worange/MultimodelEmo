"""Configuration and reproducibility helpers for the Q1 v2 pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Q1Config:
    """All parameters that can affect generated features.

    Labels deliberately do not appear here: they are metadata used only to
    validate the contest sample mapping.
    """

    sample_rate: int = 16_000
    audio_window_ms: float = 25.0
    audio_hop_ms: float = 10.0
    f0_window_ms: float = 64.0
    n_mels: int = 64
    fmin_hz: float = 50.0
    fmax_hz: float = 500.0
    visual_fps: float = 10.0
    visual_probe_fps: float = 5.0
    # Existing fused archives remain MediaPipe-native; the separate selector's
    # candidate default is OpenFace68 and does not relabel historical outputs.
    visual_backend: str = "openface68"
    final_visual_backend_candidate: str = "openface68"
    visual_feature_schema: str = "openface68_xy_au_pose_gaze_v1"
    alignment_margin_s: float = 0.30
    openface_confidence_threshold: float = 0.50
    openface_timestamp_tolerance_s: float = 0.06
    face_validity_threshold: float = 0.50
    extra_speech_tolerance_s: float = 0.15
    extra_speech_min_duration_s: float = 0.20
    asr_mfa_high_max_disagreement_s: float = 0.10
    asr_mfa_medium_max_disagreement_s: float = 0.20
    visual_qa_min_face_rate: float = 0.20
    visual_qa_min_continuous_s: float = 0.40
    visual_qa_min_face_area_ratio: float = 0.01
    visual_qa_static_mean_difference_max: float = 0.006
    text_model: str = "google-bert/bert-base-uncased"
    text_pooling: str = "mean"
    text_window_overlap_words: int = 32
    mfa_acoustic_model: str = "english_us_arpa"
    mfa_dictionary: str = "english_us_arpa"
    mfa_output_format: str = "json"
    mfa_timeout_s: int = 900
    mfa_beam: int | None = None
    mfa_root_dir: str | None = None
    mfa_work_root: str | None = None
    face_model_path: str | None = None
    face_model_sha256: str | None = None
    inference_device: str = "cpu"
    max_faces: int = 2
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    nearest_visual_max_distance_s: float | None = None
    # Legacy 38-point schema for q1_v2.run; not an OpenFace feature.
    selected_landmarks: tuple[int, ...] = (
        0, 1, 4, 10, 13, 14, 17, 33, 46, 52, 55, 61, 63, 65, 66, 70,
        78, 105, 107, 133, 145, 152, 159, 263, 276, 282, 285, 291,
        293, 295, 296, 300, 308, 334, 336, 362, 374, 386,
    )
    expected_sample_count: int = 100
    schema_version: str = "q1_v2.1"

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.audio_window_ms / 1000.0))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.audio_hop_ms / 1000.0))

    @property
    def f0_window_samples(self) -> int:
        return int(round(self.sample_rate * self.f0_window_ms / 1000.0))

    @property
    def mediapipe_native_landmark_indices(self) -> tuple[int, ...]:
        """One authoritative native MediaPipe topology for Round-4/5 extraction."""
        return tuple(range(478))

    @property
    def crop_margin_s(self) -> float:
        """Compatibility alias; alignment_margin_s is the single source."""
        return self.alignment_margin_s

    @property
    def min_face_detection_confidence(self) -> float:
        """Compatibility alias for the MediaPipe detection threshold."""
        return self.face_validity_threshold

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if value["mfa_beam"] is None:
            # Preserve pre-round-3 fingerprints for unchanged MFA defaults.
            value.pop("mfa_beam")
        value["selected_landmarks"] = list(self.selected_landmarks)
        return value

    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def package_versions(names: tuple[str, ...] | None = None) -> dict[str, str | None]:
    names = names or (
        "numpy", "openpyxl", "av", "soundfile", "librosa", "torch",
        "transformers", "tokenizers", "mediapipe", "matplotlib",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _command_version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=False,
            encoding="utf-8", errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "detail": type(exc).__name__}
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "version": output[0] if output else "",
    }


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "executables": {
            "mfa": _command_version(["mfa", "version"]),
            "ffmpeg": _command_version(["ffmpeg", "-version"]),
            "ffprobe": _command_version(["ffprobe", "-version"]),
        },
    }


def save_experiment_config(output_dir: Path, config: Q1Config, cli: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "config_sha256": config.fingerprint(),
        "cli": {key: str(value) if isinstance(value, Path) else value for key, value in cli.items()},
        "environment": collect_environment(),
    }
    path = output_dir / "experiment_config.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
