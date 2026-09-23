"""PTS-based MediaPipe Face Landmarker features and word associations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .alignment import WordAlignment
from .io_utils import write_json
from .media import VideoFrameTiming


# Canonical names emitted by the MediaPipe Face Landmarker blendshape model.
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)


@dataclass
class VisualFeatureResult:
    frame_features: np.ndarray
    frame_times_s: np.ndarray
    original_frame_indices: np.ndarray
    target_times_s: np.ndarray
    target_time_errors_s: np.ndarray
    mediapipe_timestamps_ms: np.ndarray
    detection_mask: np.ndarray
    face_counts: np.ndarray
    speaker_identity_mask: np.ndarray
    frame_status: list[str]
    word_features: np.ndarray
    word_mask: np.ndarray
    word_identity_mask: np.ndarray
    word_estimated_mask: np.ndarray
    word_frame_indices: list[list[int]]
    word_original_frame_indices: list[list[int]]
    word_frame_counts: np.ndarray
    word_valid_face_counts: np.ndarray
    word_status: list[str]
    selected_landmarks: list[int]
    blendshape_names: list[str]
    sampling_fps: float

    def metadata(self) -> dict[str, Any]:
        return {
            "frame_feature_dim": int(self.frame_features.shape[1]),
            "word_feature_dim": int(self.word_features.shape[1]),
            "selected_landmarks": self.selected_landmarks,
            "landmark_coordinate_order": "for each selected index: x_centered/interocular, y_centered/interocular, z/interocular",
            "landmark_anchor": "landmark 1 (nose tip)",
            "landmark_scale": "Euclidean xy distance between landmarks 33 and 263",
            "blendshape_names": self.blendshape_names,
            "sampling_fps": self.sampling_fps,
            "sampling_rule": "nearest decoded frame to each sample-time grid point, deduplicated, using decoded PTS",
            "mediapipe_time_mapping": "timestamp_ms = rounded((actual_sample_time-first_sample_time)*1000), made strictly increasing after rounding",
            "speaker_rule": "largest detected face is represented, but no face is asserted to be the speaker; identity mask stays 0",
            "word_frame_indices": self.word_frame_indices,
            "word_original_frame_indices": self.word_original_frame_indices,
            "word_status": self.word_status,
        }


def choose_sampled_frames(
    frame_timings: Sequence[VideoFrameTiming], sampling_fps: float,
) -> list[dict[str, float | int]]:
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be positive")
    valid = [frame for frame in frame_timings if frame.timestamp_valid and not frame.corrupt]
    if not valid:
        return []
    times = np.asarray([frame.sample_time_s for frame in valid], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Video frame timestamps are not strictly increasing")
    grid_start = math.ceil(max(0.0, float(times[0])) * sampling_fps - 1e-9) / sampling_fps
    targets = np.arange(grid_start, float(times[-1]) + 1e-9, 1.0 / sampling_fps)
    selected: list[dict[str, float | int]] = []
    used: set[int] = set()
    for target in targets:
        right = int(np.searchsorted(times, target, side="left"))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(valid)]
        best = min(candidates, key=lambda index: (abs(times[index] - target), index))
        source_index = valid[best].frame_index
        if source_index in used:
            continue
        used.add(source_index)
        selected.append({
            "frame_index": source_index,
            "sample_time_s": float(times[best]),
            "target_time_s": float(target),
            "target_error_s": float(times[best] - target),
        })
    return selected


def _decode_selected_rgb(video_path: Path, selected_indices: set[int]) -> dict[int, np.ndarray]:
    import av

    frames: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return frames
        for frame_index, frame in enumerate(container.decode(container.streams.video[0])):
            if frame_index in selected_indices:
                frames[frame_index] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(selected_indices):
                break
    return frames


def _face_area(landmarks: Sequence[Any]) -> float:
    xs = [float(point.x) for point in landmarks]
    ys = [float(point.y) for point in landmarks]
    return (max(xs) - min(xs)) * (max(ys) - min(ys)) if xs else 0.0


def normalize_landmarks(landmarks: Sequence[Any], selected: Sequence[int]) -> np.ndarray:
    if max((*selected, 263, 33, 1)) >= len(landmarks):
        raise ValueError(f"Face model returned only {len(landmarks)} landmarks")
    anchor = landmarks[1]
    left, right = landmarks[33], landmarks[263]
    scale = math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y))
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Invalid interocular landmark scale")
    values: list[float] = []
    for index in selected:
        point = landmarks[index]
        values.extend([
            (float(point.x) - float(anchor.x)) / scale,
            (float(point.y) - float(anchor.y)) / scale,
            float(point.z) / scale,
        ])
    return np.asarray(values, dtype=np.float32)


def visual_frame_indices_in_interval(frame_times_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((frame_times_s >= start_s) & (frame_times_s < end_s)).astype(np.int64)


def aggregate_visual_to_words(
    frame_features: np.ndarray,
    frame_times_s: np.ndarray,
    detection_mask: np.ndarray,
    face_counts: np.ndarray,
    original_frame_indices: np.ndarray,
    alignments: Sequence[WordAlignment],
    *,
    nearest_max_distance_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[list[int]], np.ndarray, np.ndarray, list[str]]:
    feature_dim = int(frame_features.shape[1])
    word_features = np.full((len(alignments), feature_dim * 2), np.nan, dtype=np.float32)
    word_mask = np.zeros(len(alignments), dtype=np.uint8)
    identity_mask = np.zeros(len(alignments), dtype=np.uint8)
    estimated_mask = np.zeros(len(alignments), dtype=np.uint8)
    local_lists: list[list[int]] = []
    original_lists: list[list[int]] = []
    observed_counts = np.zeros(len(alignments), dtype=np.int32)
    valid_counts = np.zeros(len(alignments), dtype=np.int32)
    statuses: list[str] = []
    for word_index, word in enumerate(alignments):
        if not word.alignment_mask:
            indices = np.empty(0, dtype=np.int64)
            status = "word_time_missing"
        else:
            indices = visual_frame_indices_in_interval(frame_times_s, word.start_s, word.end_s)
            status = "" if len(indices) else "no_sampled_frame_in_interval"
        true_interval_count = len(indices)
        observed_counts[word_index] = true_interval_count
        if not len(indices) and word.alignment_mask and nearest_max_distance_s is not None and len(frame_times_s):
            midpoint = (word.start_s + word.end_s) / 2.0
            nearest = int(np.argmin(np.abs(frame_times_s - midpoint)))
            distance = float(abs(frame_times_s[nearest] - midpoint))
            if distance <= nearest_max_distance_s:
                indices = np.asarray([nearest], dtype=np.int64)
                estimated_mask[word_index] = 1
                status = f"nearest_frame_estimate:distance_s={distance:.6f}"
        local_lists.append(indices.tolist())
        original_lists.append(np.asarray(original_frame_indices)[indices].astype(int).tolist())
        if not len(indices):
            statuses.append(status)
            continue
        valid = indices[np.asarray(detection_mask)[indices].astype(bool)]
        valid_counts[word_index] = len(valid)
        if not len(valid):
            statuses.append("sampled_frames_present_but_all_face_detections_failed" if true_interval_count else status + ";face_detection_failed")
            continue
        selected = frame_features[valid]
        means = np.nanmean(selected, axis=0)
        stds = np.nanstd(selected, axis=0)
        word_features[word_index] = np.concatenate([means, stds]).astype(np.float32)
        if not estimated_mask[word_index]:
            word_mask[word_index] = 1
        if np.any(np.asarray(face_counts)[valid] > 1):
            statuses.append("face_features_observed_multiple_faces_speaker_unresolved")
        else:
            statuses.append("face_features_observed_single_face_speaker_unverified")
        # No audiovisual diarization is performed, so identity_mask remains zero.
    return (
        word_features, word_mask, identity_mask, estimated_mask, local_lists,
        original_lists, observed_counts, valid_counts, statuses,
    )


def extract_visual_features(
    video_path: Path,
    frame_timings: Sequence[VideoFrameTiming],
    alignments: Sequence[WordAlignment],
    *,
    face_model_path: Path,
    selected_landmarks: Sequence[int],
    sampling_fps: float = 5.0,
    max_faces: int = 2,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    nearest_max_distance_s: float | None = None,
    output_dir: Path | None = None,
) -> VisualFeatureResult:
    if not face_model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe Face Landmarker model missing: {face_model_path}; "
            "download the official face_landmarker.task asset"
        )
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "Could not import mediapipe; use the clean environment from q1_v2/README.md"
        ) from exc
    selections = choose_sampled_frames(frame_timings, sampling_fps)
    decoded = _decode_selected_rgb(video_path, {int(item["frame_index"]) for item in selections})
    landmark_dim = len(selected_landmarks) * 3
    feature_dim = landmark_dim + len(BLENDSHAPE_NAMES)
    features = np.full((len(selections), feature_dim), np.nan, dtype=np.float32)
    detection = np.zeros(len(selections), dtype=np.uint8)
    face_counts = np.zeros(len(selections), dtype=np.int16)
    identity = np.zeros(len(selections), dtype=np.uint8)
    statuses: list[str] = []
    mp_timestamps: list[int] = []
    unknown_blendshapes: set[str] = set()
    first_time = float(selections[0]["sample_time_s"]) if selections else 0.0
    previous_ms = -1

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    # MediaPipe's native Windows file loader can fail on non-ASCII absolute
    # paths even though Python can read the file.  Supplying the documented
    # in-memory asset keeps paths with Chinese characters fully supported.
    model_asset = face_model_path.read_bytes()
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_asset),
        running_mode=RunningMode.VIDEO,
        num_faces=max_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
    )
    name_to_index = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}
    with FaceLandmarker.create_from_options(options) as landmarker:
        for local_index, selection in enumerate(selections):
            timestamp_ms = int(round((float(selection["sample_time_s"]) - first_time) * 1000.0))
            timestamp_ms = max(timestamp_ms, previous_ms + 1)
            previous_ms = timestamp_ms
            mp_timestamps.append(timestamp_ms)
            frame_index = int(selection["frame_index"])
            rgb = decoded.get(frame_index)
            if rgb is None:
                statuses.append("selected_frame_decode_failed")
                continue
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, timestamp_ms)
            count = len(result.face_landmarks)
            face_counts[local_index] = count
            if count == 0:
                statuses.append("face_detection_failed")
                continue
            primary = max(range(count), key=lambda index: _face_area(result.face_landmarks[index]))
            try:
                normalized = normalize_landmarks(result.face_landmarks[primary], selected_landmarks)
            except ValueError as exc:
                statuses.append(f"landmark_normalization_failed:{exc}")
                continue
            blendshape_vector = np.full(len(BLENDSHAPE_NAMES), np.nan, dtype=np.float32)
            if primary < len(result.face_blendshapes):
                for category in result.face_blendshapes[primary]:
                    name = str(category.category_name)
                    if name in name_to_index:
                        blendshape_vector[name_to_index[name]] = float(category.score)
                    else:
                        unknown_blendshapes.add(name)
            features[local_index] = np.concatenate([normalized, blendshape_vector])
            detection[local_index] = 1
            statuses.append("multiple_faces_speaker_unresolved" if count > 1 else "single_face_speaker_unverified")

    frame_times = np.asarray([item["sample_time_s"] for item in selections], dtype=np.float64)
    original_indices = np.asarray([item["frame_index"] for item in selections], dtype=np.int64)
    target_times = np.asarray([item["target_time_s"] for item in selections], dtype=np.float64)
    target_errors = np.asarray([item["target_error_s"] for item in selections], dtype=np.float64)
    aggregated = aggregate_visual_to_words(
        features, frame_times, detection, face_counts, original_indices, alignments,
        nearest_max_distance_s=nearest_max_distance_s,
    )
    result = VisualFeatureResult(
        frame_features=features,
        frame_times_s=frame_times,
        original_frame_indices=original_indices,
        target_times_s=target_times,
        target_time_errors_s=target_errors,
        mediapipe_timestamps_ms=np.asarray(mp_timestamps, dtype=np.int64),
        detection_mask=detection,
        face_counts=face_counts,
        speaker_identity_mask=identity,
        frame_status=statuses,
        word_features=aggregated[0],
        word_mask=aggregated[1],
        word_identity_mask=aggregated[2],
        word_estimated_mask=aggregated[3],
        word_frame_indices=aggregated[4],
        word_original_frame_indices=aggregated[5],
        word_frame_counts=aggregated[6],
        word_valid_face_counts=aggregated[7],
        word_status=aggregated[8],
        selected_landmarks=list(selected_landmarks),
        blendshape_names=list(BLENDSHAPE_NAMES),
        sampling_fps=sampling_fps,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "visual_features.npz",
            frame_features=result.frame_features,
            frame_times_s=result.frame_times_s,
            original_frame_indices=result.original_frame_indices,
            target_times_s=result.target_times_s,
            target_time_errors_s=result.target_time_errors_s,
            mediapipe_timestamps_ms=result.mediapipe_timestamps_ms,
            detection_mask=result.detection_mask,
            face_counts=result.face_counts,
            speaker_identity_mask=result.speaker_identity_mask,
            word_features=result.word_features,
            word_mask=result.word_mask,
            word_identity_mask=result.word_identity_mask,
            word_estimated_mask=result.word_estimated_mask,
            word_frame_counts=result.word_frame_counts,
            word_valid_face_counts=result.word_valid_face_counts,
        )
        metadata = result.metadata()
        metadata["frame_status"] = result.frame_status
        metadata["unknown_blendshape_names"] = sorted(unknown_blendshapes)
        write_json(output_dir / "visual_features_metadata.json", metadata)
    return result
