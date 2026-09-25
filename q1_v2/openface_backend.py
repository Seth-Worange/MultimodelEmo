"""Official OpenFace 2.2.0 68-point backend; no MediaPipe index substitution.

OpenFace 2-D x_0..x_67/y_0..y_67 are pixel coordinates. Geometry is
translated to the midpoint of eye centres (36:41, 42:47) and scaled by their
distance. No roll/head-pose rotation is applied. A failed frame remains NaN.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_loader import load_samples
from .alignment import WordAlignment
from .io_utils import directory_size, read_json, sha256_file, write_csv, write_json
from .media import VideoFrameTiming
from .visual_features import choose_sampled_frames


SOURCE_URL = "https://github.com/TadasBaltrusaitis/OpenFace"
DOC_URL = "https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format"
RELEASE_URL = "https://github.com/TadasBaltrusaitis/OpenFace/releases/tag/OpenFace_2.2.0"


def feature_schema(*, au_names: Sequence[str] = (), pose_names: Sequence[str] = (), gaze_names: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "backend": "openface68",
        "status": "parser_available_executable_required_for_real_extraction",
        "release": "2.2.0",
        "source_url": SOURCE_URL,
        "output_format_url": DOC_URL,
        "geometry_source_columns": [*[f"x_{i}" for i in range(68)], *[f"y_{i}" for i in range(68)]],
        "geometry_type": "68 OpenFace 2-D landmarks in pixels, converted to normalized xy",
        "geometry_frame_dimension": 136,
        "anchor": "midpoint of mean landmark indices 36..41 and 42..47 (eye centres)",
        "scale": "Euclidean interocular distance between those eye centres in source pixels",
        "roll_rotation_applied": False,
        "validity": "OpenFace success==1, confidence>=0.5, finite coordinates, interocular>1 pixel",
        "sample_timestamps": "PyAV PTS sample-relative times for same original frame index, cross-checked with OpenFace timestamp",
        "action_unit_names": list(au_names),
        "head_pose_names": list(pose_names),
        "gaze_names": list(gaze_names),
        "mediapipe_blendshape_concatenated": False,
        "speaker_identity_claim": False,
    }


def normalize_openface68(xy: np.ndarray) -> np.ndarray:
    if xy.shape != (68, 2) or not np.all(np.isfinite(xy)):
        raise ValueError("OpenFace geometry must have 68 finite xy pixel positions")
    left = xy[36:42].mean(axis=0)
    right = xy[42:48].mean(axis=0)
    scale = float(np.linalg.norm(right - left))
    if not math.isfinite(scale) or scale <= 1.0:
        raise ValueError("Invalid OpenFace interocular scale")
    anchor = (left + right) / 2
    return ((xy - anchor) / scale).astype(np.float32).reshape(136)


def _names(header: Sequence[str], prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(68) if f"{prefix}{i}" in header]


def parse_openface_csv(path: Path, *, confidence_min: float = 0.5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse one official OpenFace per-frame CSV, including failed frames."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("OpenFace CSV has no header")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        header = reader.fieldnames
        required = {"frame", "timestamp", "confidence", "success"} | {f"x_{i}" for i in range(68)} | {f"y_{i}" for i in range(68)}
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"OpenFace 68 CSV columns missing: {missing[:12]}")
        au_names = [name for name in header if re.fullmatch(r"AU\d+_[rc]", name)]
        pose_names = [name for name in header if name.startswith("pose_")]
        gaze_names = [name for name in header if name.startswith("gaze_")]
        rows = []
        seen_frames: set[int] = set()
        for source_row, raw in enumerate(reader, 2):
            if not any(value and value.strip() for value in raw.values()):
                continue
            frame = int(float(raw["frame"]))
            if frame < 1:
                raise ValueError(f"OpenFace frame must be 1-based positive at row {source_row}")
            # FeatureExtraction should emit one tracked face per frame. Multi-face
            # files need a separate identity policy and are never silently collapsed.
            if frame in seen_frames:
                raise ValueError(f"Multiple OpenFace rows for frame {frame}; identity unresolved")
            seen_frames.add(frame)
            timestamp = float(raw["timestamp"])
            confidence = float(raw["confidence"])
            success = float(raw["success"]) >= .5
            valid = bool(success and confidence >= confidence_min)
            geometry = np.full(136, np.nan, dtype=np.float32)
            reason = "openface_success_false" if not success else "openface_confidence_below_threshold" if confidence < confidence_min else ""
            if valid:
                try:
                    xy = np.array([[float(raw[f"x_{i}"]), float(raw[f"y_{i}"])] for i in range(68)], dtype=np.float64)
                    geometry = normalize_openface68(xy)
                except (TypeError, ValueError) as exc:
                    valid = False
                    reason = f"geometry_invalid:{type(exc).__name__}"
            def optional(names: Sequence[str]) -> np.ndarray:
                if not valid:
                    return np.full(len(names), np.nan, dtype=np.float32)
                try:
                    return np.array([float(raw[name]) for name in names], dtype=np.float32)
                except (TypeError, ValueError):
                    return np.full(len(names), np.nan, dtype=np.float32)
            rows.append({
                "openface_frame": frame, "openface_timestamp_s": timestamp,
                "confidence": confidence, "success": success,
                "face_valid": valid, "failure_reason": reason,
                "geometry": geometry, "action_units": optional(au_names),
                "head_pose": optional(pose_names), "gaze": optional(gaze_names),
            })
    if not rows:
        raise ValueError("OpenFace CSV has no data rows")
    return rows, feature_schema(au_names=au_names, pose_names=pose_names, gaze_names=gaze_names)


def associate_openface_with_pts(
    parsed: Sequence[dict[str, Any]], video_frames: Sequence[dict[str, Any]],
    *, fps: float, timestamp_tolerance_s: float = .06,
) -> list[dict[str, Any]]:
    """Sample by PyAV PTS, preserving both OpenFace and original frame clocks."""
    timings = [VideoFrameTiming(**item) for item in video_frames]
    selected = choose_sampled_frames(timings, fps)
    by_frame = {row["openface_frame"]: row for row in parsed}
    output = []
    for item in selected:
        index = int(item["frame_index"])
        found = by_frame.get(index + 1)
        pyav_time = float(item["sample_time_s"])
        if found is None:
            output.append({"original_frame_index": index, "sample_time_s": pyav_time,
                           "openface_frame": None, "openface_timestamp_s": None,
                           "timestamp_difference_s": None, "timestamp_mapping_reliable": False,
                           "face_valid": False, "failure_reason": "openface_frame_missing",
                           "geometry": np.full(136, np.nan, dtype=np.float32)})
            continue
        difference = abs(float(found["openface_timestamp_s"]) - pyav_time)
        reliable = math.isfinite(difference) and difference <= timestamp_tolerance_s
        result = dict(found)
        result.update({
            "original_frame_index": index, "sample_time_s": pyav_time,
            "timestamp_difference_s": difference,
            "timestamp_mapping_reliable": reliable,
        })
        if not reliable:
            result["face_valid"] = False
            result["failure_reason"] = "openface_vs_pyav_timestamp_disagreement"
            result["geometry"] = np.full(136, np.nan, dtype=np.float32)
            for name in ("action_units", "head_pose", "gaze"):
                result[name] = np.full_like(result[name], np.nan)
        output.append(result)
    return output


def _archive(path: Path, rows: Sequence[dict[str, Any]], schema: dict[str, Any]) -> None:
    n = len(rows)
    au_n, pose_n, gaze_n = (len(schema[key]) for key in ("action_unit_names", "head_pose_names", "gaze_names"))
    def stack(key: str, width: int) -> np.ndarray:
        return np.stack([row.get(key, np.full(width, np.nan, dtype=np.float32)) for row in rows]).astype(np.float32) if n else np.empty((0, width), dtype=np.float32)
    np.savez_compressed(
        path,
        landmark_geometry=stack("geometry", 136),
        action_units=stack("action_units", au_n),
        head_pose=stack("head_pose", pose_n),
        gaze=stack("gaze", gaze_n),
        face_valid_mask=np.array([bool(row["face_valid"]) for row in rows], dtype=np.bool_),
        original_frame_indices=np.array([int(row["original_frame_index"]) for row in rows], dtype=np.int32),
        sample_times_s=np.array([float(row["sample_time_s"]) for row in rows], dtype=np.float64),
        openface_frame_indices=np.array([int(row["openface_frame"]) if row.get("openface_frame") is not None else -1 for row in rows], dtype=np.int32),
        openface_timestamps_s=np.array([float(row["openface_timestamp_s"]) if row.get("openface_timestamp_s") is not None else np.nan for row in rows], dtype=np.float64),
        timestamp_mapping_mask=np.array([bool(row["timestamp_mapping_reliable"]) for row in rows], dtype=np.bool_),
    )


def aggregate_openface_words(
    frame_npz: Path, alignments: Sequence[WordAlignment], output_dir: Path,
) -> dict[str, Any]:
    """Aggregate true in-interval OpenFace frames without estimating missing words.

    The four blocks are kept distinct. A face detection is not a verified
    speaker identity; this backend never sets a positive identity mask.
    """
    with np.load(frame_npz, allow_pickle=False) as archive:
        times = archive["sample_times_s"]
        indices = archive["original_frame_indices"]
        valid = archive["face_valid_mask"].astype(bool) & archive["timestamp_mapping_mask"].astype(bool)
        blocks = {name: archive[name] for name in ("landmark_geometry", "action_units", "head_pose", "gaze")}
    n = len(alignments)
    word_blocks = {name: np.full((n, values.shape[1] * 2), np.nan, dtype=np.float32)
                   for name, values in blocks.items()}
    word_mask = np.zeros(n, dtype=np.bool_)
    observed_count = np.zeros(n, dtype=np.int32)
    valid_count = np.zeros(n, dtype=np.int32)
    trace = []
    for i, word in enumerate(alignments):
        if not word.alignment_mask or not np.isfinite(word.start_s) or not np.isfinite(word.end_s):
            local = np.empty(0, dtype=np.int64)
            status = "word_time_missing"
        else:
            local = np.flatnonzero((times >= word.start_s) & (times <= word.end_s))
            status = "no_sampled_frame_in_interval" if not len(local) else ""
        observed_count[i] = len(local)
        good = local[valid[local]]
        valid_count[i] = len(good)
        if len(good):
            word_mask[i] = True
            status = "face_features_observed_speaker_unverified"
            for name, values in blocks.items():
                selected = values[good]
                if selected.shape[1]:
                    with np.errstate(invalid="ignore"):
                        finite_count = np.isfinite(selected).sum(axis=0)
                        safe = np.where(np.isfinite(selected), selected, 0.)
                        mean = np.divide(safe.sum(axis=0), finite_count,
                                         out=np.full(selected.shape[1], np.nan), where=finite_count > 0)
                        squared = np.where(np.isfinite(selected), (selected - mean) ** 2, 0.)
                        variance = np.divide(squared.sum(axis=0), finite_count,
                                             out=np.full(selected.shape[1], np.nan), where=finite_count > 0)
                    word_blocks[name][i] = np.concatenate([mean, np.sqrt(variance)]).astype(np.float32)
        elif len(local):
            status = "sampled_frames_present_but_no_valid_openface_face"
        trace.append({
            "sample_id": word.sample_id, "word_index": word.word_index,
            "original_word": word.original_word, "start_s": word.start_s,
            "end_s": word.end_s, "alignment_mask": word.alignment_mask,
            "sampled_frame_count": len(local), "valid_face_frame_count": len(good),
            "original_frame_indices": ",".join(str(int(indices[j])) for j in local),
            "valid_original_frame_indices": ",".join(str(int(indices[j])) for j in good),
            "word_visual_mask": bool(word_mask[i]), "speaker_identity_verified": False,
            "status": status,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "openface68_word_features.npz",
        **{f"word_{name}": values for name, values in word_blocks.items()},
        word_visual_mask=word_mask,
        word_alignment_mask=np.asarray([bool(word.alignment_mask) for word in alignments]),
        word_sampled_frame_counts=observed_count,
        word_valid_face_frame_counts=valid_count,
        word_indices=np.asarray([word.word_index for word in alignments], dtype=np.int32),
    )
    write_csv(output_dir / "openface68_word_trace.csv", trace, (
        "sample_id", "word_index", "original_word", "start_s", "end_s",
        "alignment_mask", "sampled_frame_count", "valid_face_frame_count",
        "original_frame_indices", "valid_original_frame_indices", "word_visual_mask",
        "speaker_identity_verified", "status",
    ))
    metadata = {"word_count": n, "word_visual_valid_count": int(word_mask.sum()),
                "source_frame_npz": str(frame_npz),
                "block_dimensions": {name: list(values.shape) for name, values in word_blocks.items()},
                "aggregation": "in-interval valid frames only; per-coordinate mean and population std; no nearest-frame estimate",
                "speaker_identity_verified": False}
    write_json(output_dir / "openface68_word_metadata.json", metadata)
    return metadata


def run_one(
    *, video_path: Path, media_metadata: Path, output_dir: Path,
    executable: Path, visual_fps: float = 10.0, timeout_s: int = 900,
    confidence_min: float = 0.5, timestamp_tolerance_s: float = .06,
) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_dir = output_dir / "openface_raw"
    native_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable.resolve()), "-f", str(video_path.resolve()),
               "-out_dir", str(native_dir.resolve()), "-of", "openface68",
               "-2Dfp", "-aus", "-pose", "-gaze"]
    write_json(output_dir / "openface_command.json", {"argv": command, "cwd": str(executable.parent.resolve()), "timeout_s": timeout_s})
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=executable.parent, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    (output_dir / "openface_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "openface_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"OpenFace exit {completed.returncode}; see raw stdout/stderr")
    candidates = sorted(native_dir.glob("*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one OpenFace CSV, got {candidates}")
    parsed, schema = parse_openface_csv(candidates[0], confidence_min=confidence_min)
    media = read_json(media_metadata)
    associated = associate_openface_with_pts(parsed, media["video_frames"], fps=visual_fps,
                                              timestamp_tolerance_s=timestamp_tolerance_s)
    _archive(output_dir / "openface68_frames.npz", associated, schema)
    trace = [{
        "original_frame_index": row["original_frame_index"],
        "sample_time_s": row["sample_time_s"],
        "openface_frame": row.get("openface_frame"),
        "openface_timestamp_s": row.get("openface_timestamp_s"),
        "timestamp_difference_s": row.get("timestamp_difference_s"),
        "timestamp_mapping_reliable": row["timestamp_mapping_reliable"],
        "face_valid": row["face_valid"], "failure_reason": row["failure_reason"],
    } for row in associated]
    write_csv(output_dir / "openface68_frame_trace.csv", trace, (
        "original_frame_index", "sample_time_s", "openface_frame", "openface_timestamp_s",
        "timestamp_difference_s", "timestamp_mapping_reliable", "face_valid", "failure_reason",
    ))
    schema["status"] = "implemented_and_real_extraction_executed"
    schema["openface_executable"] = str(executable.resolve())
    schema["openface_executable_sha256"] = sha256_file(executable)
    schema["source_video_sha256"] = sha256_file(video_path)
    schema["openface_raw_csv"] = str(candidates[0])
    schema["visual_fps"] = visual_fps
    schema["openface_confidence_threshold"] = confidence_min
    schema["timestamp_tolerance_s"] = timestamp_tolerance_s
    write_json(output_dir / "openface_feature_schema.json", schema)
    result = {
        "status": "success", "sampled_frame_count": len(associated),
        "face_success_frame_count": sum(row["face_valid"] for row in associated),
        "face_success_rate": sum(row["face_valid"] for row in associated) / len(associated) if associated else None,
        "timestamp_mapping_reliable_count": sum(row["timestamp_mapping_reliable"] for row in associated),
        "geometry_frame_dim": 136,
        "runtime_s": time.perf_counter() - started,
        "compressed_output_bytes": (output_dir / "openface68_frames.npz").stat().st_size,
        "raw_output_bytes": directory_size(native_dir),
    }
    write_json(output_dir / "openface_result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--openface-executable", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--visual-fps", type=float, choices=(5.0, 10.0), default=10.0)
    args = parser.parse_args(argv)
    official, _ = load_samples(args.data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in official}
    if len(args.sample_id) > 12:
        raise ValueError("Representative OpenFace experiment capped at 12 samples")
    failed = False
    for sample_id in args.sample_id:
        if sample_id not in by_id:
            raise ValueError(f"Not an official sample: {sample_id}")
        target = args.output_dir / "samples" / sample_id / "openface68"
        if (target / "openface_result.json").is_file():
            print({"sample_id": sample_id, "action": "skipped_existing"}, flush=True)
            continue
        try:
            result = run_one(video_path=by_id[sample_id].video_path,
                             media_metadata=args.round4_root / "samples" / sample_id / "media_metadata.json",
                             output_dir=target, executable=args.openface_executable,
                             visual_fps=args.visual_fps)
            print({"sample_id": sample_id, **result}, flush=True)
        except Exception as exc:
            failed = True
            write_json(target / "_OPENFACE_FAILED.json", {
                "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print({"sample_id": sample_id, "action": "failed", "error": str(exc)}, flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
