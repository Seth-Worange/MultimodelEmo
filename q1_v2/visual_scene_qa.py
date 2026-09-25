"""Timestamped 5-FPS MediaPipe scene probe with distinct face/static concepts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import Q1Config
from .io_utils import write_csv
from .media import VideoFrameTiming
from .visual_features import _decode_selected_rgb, choose_sampled_frames


PROBE_FIELDS = (
    "frame_index", "sample_time_s", "decoded", "face_count",
    "largest_face_area_ratio", "area_qualified_face", "frame_mean_absolute_difference",
)


def visual_scene_decision(
    frame_rows: Sequence[dict[str, Any]], *, fps: float,
    config: Q1Config,
) -> dict[str, Any]:
    observed = [row for row in frame_rows if row["decoded"]]
    count = len(observed)
    detected = sum(row["face_count"] > 0 for row in observed)
    qualified = [row for row in observed if row["area_qualified_face"]]
    multiple = sum(row["face_count"] > 1 for row in observed)
    differences = [row["frame_mean_absolute_difference"] for row in observed if row["frame_mean_absolute_difference"] is not None]
    mean_diff = float(np.mean(differences)) if differences else None
    static = mean_diff <= config.visual_qa_static_mean_difference_max if mean_diff is not None else None
    longest = 0.0
    run_start = last_time = None
    for row in observed:
        time = float(row["sample_time_s"])
        if row["area_qualified_face"]:
            if last_time is None or time - last_time > 1.6 / fps:
                run_start = time
            last_time = time
            longest = max(longest, time - run_start + 1.0 / fps)
        else:
            last_time = None
            run_start = None
    rate = detected / count if count else None
    coverage = len(qualified) / count if count else None
    any_face = detected > 0
    usable = bool(
        count and not static and coverage is not None
        and coverage >= config.visual_qa_min_face_rate
        and longest >= config.visual_qa_min_continuous_s
    )
    if not count:
        scene = "MIXED_OR_UNRESOLVED"
    elif static and any_face:
        # A face in a near-static frame can be a photo, still video, or speaker.
        scene = "MIXED_OR_UNRESOLVED"
    elif usable:
        scene = "NORMAL_FACE_VIDEO"
    elif not any_face and static:
        scene = "NON_FACE_STATIC_SCENE"
    elif not any_face:
        scene = "NON_FACE_DYNAMIC_SCENE"
    else:
        scene = "MIXED_OR_UNRESOLVED"
    return {
        "video_signal_present": count > 0,
        "sampled_frame_count": count,
        "any_face_detected": any_face,
        "face_detected_frame_count": detected,
        "face_detection_rate": rate,
        "multiple_face_frame_count": multiple,
        "usable_face_present": usable,
        "usable_face_frame_count": len(qualified),
        "usable_face_coverage": coverage,
        "longest_qualified_face_visible_s": longest,
        "static_visual": static,
        "static_visual_score": 1.0 - mean_diff if mean_diff is not None else None,
        "nonhuman_or_no_face_scene": not any_face if count else None,
        "visual_scene_status": scene,
        "speaker_identity_uncertain": multiple > 0,
        "visual_probe_fps": fps,
        "face_usability_rule": "area>=min_ratio, coverage>=min_rate, continuous>=min_s, dynamic; no speaker identification",
    }


def probe_visual_scene(
    video_path: Path, video_frames: Sequence[dict[str, Any]],
    *, face_model_path: Path, config: Q1Config, output_csv: Path,
) -> dict[str, Any]:
    import cv2
    import mediapipe as mp

    timings = [VideoFrameTiming(**row) for row in video_frames]
    selected = choose_sampled_frames(timings, config.visual_probe_fps)
    decoded = _decode_selected_rgb(video_path, {int(row["frame_index"]) for row in selected})
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=face_model_path.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=config.max_faces,
        output_face_blendshapes=False,
    )
    rows: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for item in selected:
            index, sample_time = int(item["frame_index"]), float(item["sample_time_s"])
            rgb = decoded.get(index)
            if rgb is None:
                rows.append({"frame_index": index, "sample_time_s": sample_time,
                             "decoded": False, "face_count": 0,
                             "largest_face_area_ratio": None, "area_qualified_face": False,
                             "frame_mean_absolute_difference": None})
                continue
            gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32) / 255.0
            difference = float(np.mean(np.abs(gray - previous))) if previous is not None else None
            previous = gray
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            faces = landmarker.detect(image).face_landmarks
            areas = []
            for face in faces:
                xs, ys = [point.x for point in face], [point.y for point in face]
                areas.append(max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys))))
            area = max(areas, default=0.0)
            rows.append({
                "frame_index": index, "sample_time_s": sample_time,
                "decoded": True, "face_count": len(faces),
                "largest_face_area_ratio": area,
                "area_qualified_face": area >= config.visual_qa_min_face_area_ratio,
                "frame_mean_absolute_difference": difference,
            })
    write_csv(output_csv, rows, PROBE_FIELDS)
    result = visual_scene_decision(rows, fps=config.visual_probe_fps, config=config)
    result["selected_frame_count"] = len(selected)
    result["selected_original_frame_indices"] = [int(row["frame_index"]) for row in selected]
    result["selected_frame_times_s"] = [float(row["sample_time_s"]) for row in selected]
    return result
