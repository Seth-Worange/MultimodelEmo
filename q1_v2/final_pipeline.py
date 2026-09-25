"""Round-5.1 frozen Q1 feature pipeline; explicitly limited to selected samples.

Stage-1 correspondence QA is verified against the immutable MP4/text before
use. Only HIGH_CONFIDENCE_MATCH can invoke local-crop MFA. All original word
positions survive even when the trusted time interval is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .alignment import ALIGNMENT_FIELDS, WordAlignment, parse_mfa_output, remap_mfa_words, run_mfa_alignment
from .alignment_confidence import audit_one as audit_disagreement
from .audio_features import extract_audio_features
from .config import Q1Config, collect_environment
from .data_loader import SampleRecord, load_samples
from .final_schema import SCHEMA_VERSION, validate_package, write_root_schemas
from .io_utils import directory_size, read_json, sha256_file, write_csv, write_json
from .media import inspect_and_extract_media
from .round4_features import _empty_audio, crop_for_local_mfa, reuse_verified_native_features, unavailable_alignment
from .text_features import extract_text_features
from .visual_backend import BACKENDS, make_visual_backend


# Six spec-named ids plus category roles NO_SPEECH + no-face (-NFrJFQijFE__1)
# and static figure (-UuX1xuaiiE__1). A NON_ENGLISH case and one ordinary
# MATCHED case are selected from automatic evidence in select_smoke_ids.
REQUIRED_SMOKE = (
    "-3g5yACwYnA__13", "-3g5yACwYnA__2", "-3g5yACwYnA__3",
    "-THoVjtIkeU__2", "-s9qJ7ATP7w__6", "-AUZQgSxyPQ__2",
    "-NFrJFQijFE__1", "-UuX1xuaiiE__1",
)
SUMMARY_FIELDS = (
    "sample_id", "processing_status", "quality_status", "quality_route", "mfa_policy",
    "mfa_executed_this_run", "mfa_real_output_available", "original_word_count",
    "aligned_word_count", "text_word_count", "audio_word_count", "visual_word_count",
    "text_word_dim", "audio_frame_dim", "audio_word_dim", "visual_word_dim_total",
    "visual_backend", "openface_sampled_frame_count", "openface_valid_frame_count",
    "visual_word_coverage", "npz_allow_pickle_false", "feature_package_bytes",
    "sample_output_bytes", "elapsed_s", "failure_stage", "error",
)


def select_smoke_ids(qa_root: Path, round5_root: Path, *, seed: int = 2026) -> list[str]:
    """Choose NON_ENGLISH and ordinary cases from automatic evidence, never human labels."""
    excluded = set(REQUIRED_SMOKE)
    with (qa_root / "correspondence_audit.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        stage1 = list(csv.DictReader(handle))
    with (round5_root / "qa" / "correspondence_audit_round5.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        scenes = {row["sample_id"]: row for row in csv.DictReader(handle)}
    non_english = sorted(row["sample_id"] for row in stage1
                         if row["sample_id"] not in excluded
                         and row["correspondence_status"] == "NON_ENGLISH_SPEECH")
    if not non_english:
        raise ValueError("No automatic NON_ENGLISH_SPEECH candidate for smoke sample")
    pool = sorted(row["sample_id"] for row in stage1
                  if row["sample_id"] not in excluded
                  and row["sample_id"] != non_english[0]
                  and row["route"] == "HIGH_CONFIDENCE_MATCH"
                  and scenes.get(row["sample_id"], {}).get("visual_scene_status") == "NORMAL_FACE_VIDEO")
    if not pool:
        raise ValueError("No ordinary auto-MATCHED candidate for tenth smoke sample")
    return [*REQUIRED_SMOKE, non_english[0], random.Random(seed).choice(pool)]


def _verified_qa(record: SampleRecord, qa_root: Path, target: Path) -> tuple[dict[str, Any], Path]:
    source = qa_root / "samples" / record.sample_id
    qa_path = source / "correspondence_qa.json"
    qa = read_json(qa_path)
    source_sha = sha256_file(record.video_path)
    if qa.get("sample_id") != record.sample_id or qa.get("official_text") != record.text:
        raise ValueError("Stage-1 QA sample ID or official text mismatch")
    if qa.get("source_video_sha256") != source_sha:
        raise ValueError("Stage-1 QA video SHA256 mismatch")
    if qa.get("route") not in {"HIGH_CONFIDENCE_MATCH", "REVIEW_REQUIRED", "INVALID_CORRESPONDENCE"}:
        raise ValueError("Stage-1 QA route missing or unknown")
    for name in ("asr_words.csv", "local_match.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
    write_json(target / "correspondence_qa.json", qa)
    write_json(target / "qa_source_provenance.json", {
        "source_qa": str(qa_path.resolve()), "source_qa_sha256": sha256_file(qa_path),
        "source_video_sha256": source_sha, "route_unchanged": True,
        "thresholds_unchanged": True, "manual_label_used_for_routing": False,
    })
    return qa, source


def _align(
    record: SampleRecord, media: Any, qa: dict[str, Any], qa_source: Path,
    target: Path, config: Q1Config, *, dictionary: Path, mfa_work_dir: Path,
) -> tuple[list[WordAlignment], str, bool]:
    if qa["route"] != "HIGH_CONFIDENCE_MATCH":
        words = unavailable_alignment(record, "correspondence_not_high_confidence")
        write_csv(target / "word_alignment.csv", (asdict(word) for word in words), ALIGNMENT_FIELDS)
        write_json(target / "alignment_metadata.json", {
            "source": "no_trusted_alignment", "mfa_policy": "PROHIBITED_BY_ROUTER",
            "reason": qa.get("correspondence_status"), "original_word_count": len(words),
        })
        return words, "PROHIBITED_BY_ROUTER", False
    if not media.audio_timestamp_reliable:
        raise RuntimeError("HIGH_CONFIDENCE_MATCH but decoded audio timestamp unreliable")
    crop = crop_for_local_mfa(
        media, float(qa["matched_audio_start_s"]), float(qa["matched_audio_end_s"]),
        margin_s=config.alignment_margin_s, output_path=target / "audio_mfa_local_crop.wav",
    )
    write_json(target / "mfa_crop_metadata.json", crop)
    raw_source = qa_source / "mfa_raw.json"
    if raw_source.is_file():
        source_input = read_json(qa_source / "sample_input.json")
        if source_input.get("video_sha256") != sha256_file(record.video_path) or source_input.get("text") != record.text:
            raise ValueError("Previous real MFA source no longer matches the official sample")
        words = remap_mfa_words(
            record.sample_id, record.original_words, parse_mfa_output(raw_source),
            wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
            timeline_origin_media_s=media.timeline_origin_media_s,
            duration_s=media.duration_s,
        )
        shutil.copy2(raw_source, target / "mfa_raw.json")
        write_csv(target / "word_alignment.csv", (asdict(word) for word in words), ALIGNMENT_FIELDS)
        write_json(target / "mfa_source_provenance.json", {
            "source": "reused_verified_real_mfa", "source_path": str(raw_source.resolve()),
            "source_sha256": sha256_file(raw_source), "source_word_count": len(words),
            "new_mfa_executed": False,
        })
        policy, executed = "REUSED_VERIFIED_REAL_MFA", False
    else:
        result = run_mfa_alignment(
            sample_id=record.sample_id, original_text=record.text,
            original_words=record.original_words,
            wav_path=target / "audio_mfa_local_crop.wav", output_dir=target,
            acoustic_model=config.mfa_acoustic_model, dictionary=str(dictionary),
            wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
            timeline_origin_media_s=media.timeline_origin_media_s,
            duration_s=media.duration_s, timeout_s=config.mfa_timeout_s,
            temporary_directory=mfa_work_dir, beam=config.mfa_beam,
        )
        words = result.words
        policy, executed = "NEW_LOCAL_MFA", True
    if len(words) != record.word_count:
        raise ValueError("MFA lost official word positions")
    write_json(target / "alignment_metadata.json", {
        "source": "real_local_mfa", "mfa_policy": policy,
        "original_word_count": record.word_count,
        "aligned_word_count": sum(bool(word.alignment_mask) for word in words),
        "numerical_algorithm_changed_from_round4": False,
    })
    return words, policy, executed


def _alignment_confidence(
    record: SampleRecord, target: Path, alignments: Sequence[WordAlignment], config: Q1Config,
) -> list[str]:
    if (target / "mfa_raw.json").is_file() and (target / "asr_words.csv").is_file():
        details = audit_disagreement(record.sample_id, target, config, {})
        grades = [item["alignment_confidence"] for item in details]
    else:
        details = [{"sample_id": record.sample_id, "word_index": i,
                    "alignment_confidence": "UNAVAILABLE", "max_disagreement_s": None,
                    "reason": "no_real_mfa_or_asr_word_mapping"}
                   for i in range(record.word_count)]
        grades = ["UNAVAILABLE"] * record.word_count
    if len(grades) != record.word_count:
        raise ValueError("ASR/MFA confidence lost word positions")
    write_json(target / "alignment_confidence.json", {
        "sample_id": record.sample_id, "meaning": "MFA-ASR disagreement uncertainty, not MFA accuracy",
        "counts": dict(Counter(grades)), "words": details,
    })
    return grades


def _nullable_bool(value: Any) -> np.int8:
    if value is None or value == "":
        return np.int8(-1)
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return np.int8(1)
        if value.lower() in {"false", "0"}:
            return np.int8(0)
        return np.int8(-1)
    return np.int8(bool(value))


def _quality_block(qa: dict[str, Any]) -> dict[str, str]:
    """Four orthogonal automatic statuses; human labels never enter this block."""

    def flag(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (1, "1", "true", "True"):
            return True
        if value in (0, "0", "false", "False"):
            return False
        return None

    speech_flag = flag(qa.get("speech_present"))
    speech = "UNKNOWN" if speech_flag is None else ("SPEECH_PRESENT" if speech_flag else "NO_SPEECH")
    english_flag = flag(qa.get("english_speech_likely"))
    if speech_flag is False:
        language = "NOT_APPLICABLE"
    elif english_flag is None:
        language = "UNKNOWN"
    else:
        language = "ENGLISH" if english_flag else "NON_ENGLISH"
    face_flag = flag(qa.get("face_present"))
    face = "UNKNOWN" if face_flag is None else ("FACE_PRESENT" if face_flag else "NO_FACE")
    return {
        "correspondence_status": str(qa.get("correspondence_status", "UNKNOWN")),
        "speech_status": speech,
        "language_status": language,
        "face_status": face,
    }


def _package(
    record: SampleRecord, target: Path, qa: dict[str, Any], round5_qa: dict[str, Any],
    alignments: Sequence[WordAlignment], grades: Sequence[str], text: Any, audio: Any,
    visual: Any, *, mfa_policy: str,
) -> None:
    n = record.word_count
    if not all(len(item) == n for item in (alignments, grades, text.features, audio.word_features,
                                           visual.word_mask, visual.word_original_frame_indices)):
        raise ValueError("Final package word axes disagree")
    quality = _quality_block(qa)
    arrays: dict[str, Any] = {
        "sample_id": np.asarray(record.sample_id),
        "quality_status": np.asarray(qa["correspondence_status"]),
        "quality_route": np.asarray(qa["route"]),
        "speech_status": np.asarray(quality["speech_status"]),
        "language_status": np.asarray(quality["language_status"]),
        "face_status": np.asarray(quality["face_status"]),
        "visual_backend": np.asarray(visual.backend),
        "word_index": np.arange(n, dtype=np.int32),
        "word_text": np.asarray([word.text for word in record.original_words], dtype=np.str_),
        "word_start_s": np.asarray([word.start_s for word in alignments], dtype=np.float64),
        "word_end_s": np.asarray([word.end_s for word in alignments], dtype=np.float64),
        "alignment_confidence": np.asarray(grades, dtype=np.str_),
        "text_features": text.features.astype(np.float32),
        "audio_features": audio.word_features.astype(np.float32),
        "text_available": text.mask.astype(np.uint8),
        "audio_available": audio.word_mask.astype(np.uint8),
        "visual_available": visual.word_mask.astype(np.uint8),
        "alignment_available": np.asarray([word.alignment_mask for word in alignments], dtype=np.uint8),
        "audio_frame_features": audio.frame_features.astype(np.float32),
        "audio_frame_times_s": audio.frame_times_s.astype(np.float64),
        "audio_frame_start_samples": audio.frame_start_samples.astype(np.int64),
        "audio_frame_f0_valid": audio.f0_valid_mask.astype(np.uint8),
        "visual_frame_indices": visual.frame_indices.astype(np.int32),
        "visual_frame_times_s": visual.frame_times_s.astype(np.float64),
        "visual_frame_valid": visual.frame_mask.astype(np.uint8),
        "visual_word_sampled_frame_counts": visual.word_sampled_frame_counts.astype(np.int32),
        "visual_word_valid_face_counts": visual.word_valid_face_counts.astype(np.int32),
        "has_face": np.asarray(int(visual.frame_mask.any()), dtype=np.uint8),
        "usable_face": np.asarray(_nullable_bool(round5_qa.get("usable_face_present"))),
        "extra_speech_before": np.asarray(_nullable_bool(round5_qa.get("extra_speech_before"))),
        "extra_speech_after": np.asarray(_nullable_bool(round5_qa.get("extra_speech_after"))),
    }
    for name, values in visual.frame_blocks.items():
        arrays[f"visual_frame_{name}"] = values.astype(np.float32)
    for name, values in visual.word_blocks.items():
        arrays[f"visual_word_{name}"] = values.astype(np.float32)
    np.savez_compressed(target / "feature_package.npz", **arrays)
    write_json(target / "feature_package_metadata.json", {
        "sample_id": record.sample_id, "schema_version": SCHEMA_VERSION,
        "source_video_sha256": sha256_file(record.video_path),
        "official_text": record.text, "word_count": n,
        "quality": quality,
        "quality_status": qa["correspondence_status"], "quality_route": qa["route"],
        "mfa_policy": mfa_policy, "visual_backend": visual.backend,
        "visual_word_original_frame_indices": visual.word_original_frame_indices,
        "audio_word_frame_indices": audio.word_frame_indices,
        "alignment_failure_reasons": [word.failure_reason for word in alignments],
        "audio_word_failure_reasons": audio.word_failure_reasons,
        "word_visual_sampled_frame_counts": visual.word_sampled_frame_counts.tolist(),
        "word_visual_valid_face_counts": visual.word_valid_face_counts.tolist(),
        "quality_flags_source": "verified Round4 router and independent Round5 visual/VAD QA",
        "speaker_identity_verified": False,
        "original_word_positions_preserved": True,
        "npz_requires_allow_pickle": False,
    })


def process_one(
    record: SampleRecord, *, output_root: Path, qa_root: Path, round5_root: Path,
    config: Q1Config, dictionary: Path, mfa_work_dir: Path,
    face_model: Path, openface_executable: Path,
    details_dir: str = "smoke_test_details",
) -> dict[str, Any]:
    target = output_root / details_dir / record.sample_id
    target.mkdir(parents=True, exist_ok=True)
    success = target / "_SUCCESS.json"
    if success.is_file():
        problems = validate_package(target, record.sample_id, backend=config.visual_backend)
        if problems:
            raise RuntimeError(f"Existing output corrupt and retained: {problems}")
        return read_json(success)
    started = time.perf_counter()
    stage = "source_qa"
    mfa_executed = False
    mfa_policy = "NOT_REACHED"
    qa: dict[str, Any] = {}
    try:
        qa, qa_source = _verified_qa(record, qa_root, target)
        round5_qa_path = round5_root / "samples" / record.sample_id / "correspondence_qa_round5.json"
        round5_qa = read_json(round5_qa_path) if round5_qa_path.is_file() else {}
        if round5_qa and round5_qa.get("source_video_sha256") != sha256_file(record.video_path):
            raise ValueError("Round5 QA source MP4 SHA256 mismatch")
        stage = "media"
        media = inspect_and_extract_media(record.video_path, target / "audio_mfa_16k_mono.wav",
                                          sample_rate=config.sample_rate,
                                          metadata_path=target / "media_metadata.json")
        stage = "alignment"
        try:
            alignment, mfa_policy, mfa_executed = _align(
                record, media, qa, qa_source, target, config,
                dictionary=dictionary, mfa_work_dir=mfa_work_dir)
        except Exception as exc:
            if qa["route"] != "HIGH_CONFIDENCE_MATCH":
                raise
            write_json(target / "_MFA_FAILED.json", {
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                "policy": "retain all original words with NaN times; do not fabricate alignment",
            })
            alignment = unavailable_alignment(record, "local_mfa_failed")
            write_csv(target / "word_alignment.csv", (asdict(word) for word in alignment), ALIGNMENT_FIELDS)
            mfa_policy = "MFA_FAILED_NO_PSEUDO_ALIGNMENT"
            write_json(target / "alignment_metadata.json", {
                "source": "local_mfa_failed", "mfa_policy": mfa_policy,
                "original_word_count": record.word_count, "aligned_word_count": 0,
                "numerical_algorithm_changed_from_round4": False,
                "failure": f"{type(exc).__name__}: {exc}",
            })
        stage = "text_audio"
        native_source = qa_source if (qa_source / "text_features.npz").is_file() and (qa_source / "audio_features.npz").is_file() else None
        if native_source is not None:
            text, audio = reuse_verified_native_features(record, media, alignment, native_source, target)
        else:
            text = extract_text_features(
                record.original_words, model_name=config.text_model,
                pooling=config.text_pooling, device=config.inference_device,
                overlap_words=config.text_window_overlap_words, output_dir=target)
            audio = (
                extract_audio_features(
                    media.waveform, sample_rate=config.sample_rate,
                    wav_origin_sample_s=media.wav_origin_sample_s,
                    alignments=alignment, window_samples=config.window_samples,
                    hop_samples=config.hop_samples, f0_window_samples=config.f0_window_samples,
                    n_mels=config.n_mels, fmin_hz=config.fmin_hz, fmax_hz=config.fmax_hz,
                    output_dir=target)
                if media.audio_timestamp_reliable and len(media.waveform) else _empty_audio(record, config)
            )
        stage = "visual"
        backend = make_visual_backend(config, face_model=face_model,
                                      openface_executable=openface_executable)
        visual = backend.extract(
            record.video_path, media.video_frames, alignment, config,
            target / "visual" / config.visual_backend,
            media_metadata=target / "media_metadata.json")
        stage = "confidence"
        grades = _alignment_confidence(record, target, alignment, config)
        stage = "package"
        _package(record, target, qa, round5_qa, alignment, grades, text, audio, visual,
                 mfa_policy=mfa_policy)
        stage = "integrity"
        errors = validate_package(target, record.sample_id, backend=config.visual_backend)
        if errors:
            raise ValueError("Feature package integrity failure: " + ";".join(errors))
        package_path = target / "feature_package.npz"
        result = {
            "sample_id": record.sample_id, "processing_status": "completed" if not (target / "_MFA_FAILED.json").is_file() else "completed_with_mfa_failure",
            "quality_status": qa["correspondence_status"], "quality_route": qa["route"],
            "mfa_policy": mfa_policy, "mfa_executed_this_run": mfa_executed,
            "mfa_real_output_available": (target / "mfa_raw.json").is_file(),
            "original_word_count": record.word_count,
            "aligned_word_count": sum(bool(word.alignment_mask) for word in alignment),
            "text_word_count": int(text.mask.sum()), "audio_word_count": int(audio.word_mask.sum()),
            "visual_word_count": int(visual.word_mask.sum()),
            "text_word_dim": int(text.features.shape[1]),
            "audio_frame_dim": int(audio.frame_features.shape[1]),
            "audio_word_dim": int(audio.word_features.shape[1]),
            "visual_word_dim_total": sum(values.shape[1] for values in visual.word_blocks.values()),
            "visual_backend": visual.backend,
            "openface_sampled_frame_count": len(visual.frame_indices) if visual.backend == "openface68" else None,
            "openface_valid_frame_count": int(visual.frame_mask.sum()) if visual.backend == "openface68" else None,
            "visual_word_coverage": float(visual.word_mask.mean()) if record.word_count else None,
            "npz_allow_pickle_false": True,
            "feature_package_bytes": package_path.stat().st_size,
            "sample_output_bytes": directory_size(target),
            "elapsed_s": time.perf_counter() - started,
            "failure_stage": None, "error": None,
        }
        write_json(success, result)
        return result
    except Exception as exc:
        failure = {"sample_id": record.sample_id, "processing_status": "failed",
                   "quality_status": qa.get("correspondence_status"),
                   "quality_route": qa.get("route"), "mfa_policy": mfa_policy,
                   "mfa_executed_this_run": mfa_executed,
                   "failure_stage": stage, "error": f"{type(exc).__name__}: {exc}",
                   "elapsed_s": time.perf_counter() - started}
        write_json(target / "_FAILED.json", {**failure, "traceback": traceback.format_exc()})
        return failure


def write_smoke_summary(output_root: Path, selected: Sequence[str], *, details_dir: str = "smoke_test_details",
                        summary_name: str = "smoke_test_summary.csv",
                        full100_status: str = "NOT_RUN") -> dict[str, Any]:
    rows = []
    for sample_id in selected:
        target = output_root / details_dir / sample_id
        success, failed = target / "_SUCCESS.json", target / "_FAILED.json"
        rows.append(read_json(success if success.is_file() else failed) if success.is_file() or failed.is_file()
                    else {"sample_id": sample_id, "processing_status": "not_run"})
    write_csv(output_root / summary_name, rows, SUMMARY_FIELDS)
    grades = Counter()
    for row in rows:
        path = output_root / details_dir / row["sample_id"] / "alignment_confidence.json"
        if path.is_file():
            grades.update(read_json(path)["counts"])
    summary = {
        "selected_count": len(selected), "completed_count": sum(row.get("processing_status") == "completed" for row in rows),
        "completed_with_mfa_failure_count": sum(row.get("processing_status") == "completed_with_mfa_failure" for row in rows),
        "failed_count": sum(row.get("processing_status") == "failed" for row in rows),
        "confidence_counts": dict(grades),
        "meaning": "MFA-ASR disagreement uncertainty, not time-boundary accuracy",
        "full_100_sample_feature_extraction": full100_status,
    }
    write_json(output_root / "alignment_confidence_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--round5-qa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mfa-dictionary", type=Path, required=True)
    parser.add_argument("--mfa-executable", type=Path, required=True)
    parser.add_argument("--mfa-root-dir", type=Path, required=True)
    parser.add_argument("--mfa-work-dir", type=Path, required=True)
    parser.add_argument("--openface-executable", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--visual-backend", choices=BACKENDS, default="openface68")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-id", action="append")
    group.add_argument("--smoke10", action="store_true")
    group.add_argument("--all", action="store_true",
                       help="Extract all 100 official samples (authorized full run; details under samples/)")
    args = parser.parse_args(argv)
    if not all(path.is_file() for path in (args.mfa_dictionary, args.mfa_executable,
                                           args.openface_executable, args.face_model)):
        parser.error("Dictionary, MFA executable, OpenFace executable, or face model absent")
    os.environ["MFA_ROOT_DIR"] = str(args.mfa_root_dir.resolve())
    os.environ["PATH"] = str(args.mfa_executable.resolve().parent) + os.pathsep + os.environ.get("PATH", "")
    # Propagate an activated conda environment into this process. MFA's
    # soundfile import resolves libsndfile from Library\bin, which only lands
    # on PATH via `conda activate`; transformers must never fall back to the
    # un-writable default ~/.cache/huggingface. Both failed before without them.
    env_root = args.mfa_executable.resolve().parent.parent
    for directory in (env_root / "Library" / "bin", env_root):
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("HF_HOME", str(args.output_dir.resolve() / "cache" / "huggingface"))
    config = replace(Q1Config(), visual_backend=args.visual_backend,
                     visual_feature_schema=("openface68_xy_au_pose_gaze_v1" if args.visual_backend == "openface68" else "mediapipe478_xyz52_blendshape"),
                     mfa_work_root=str(args.mfa_work_dir.resolve()))
    official, _ = load_samples(args.data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in official}
    if args.all:
        selected = sorted(by_id)
        details_dir, summary_name = "samples", "feature_summary.csv"
    else:
        selected = select_smoke_ids(args.qa_root, args.round5_qa_root) if args.smoke10 else args.sample_id
        details_dir, summary_name = "smoke_test_details", "smoke_test_summary.csv"
    if not args.all and (len(selected) > 10 or len(set(selected)) != len(selected)
                         or any(item not in by_id for item in selected)):
        parser.error("At most 10 unique official sample IDs allowed; use --all for the authorized full run")
    if args.all and (len(set(selected)) != len(selected) or any(item not in by_id for item in selected)):
        parser.error("Full run selection must map 1:1 onto official samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_root_schemas(args.output_dir, config.visual_backend, config)
    write_json(args.output_dir / "experiment_config.json", {
        "config": config.to_dict(), "schema_version": SCHEMA_VERSION,
        "selected_sample_ids": selected, "tenth_selection": "NON_ENGLISH = alphabetically first automatic NON_ENGLISH_SPEECH; ordinary = seed=2026 pick from automatically high-confidence NORMAL_FACE_VIDEO pool; no human labels" if args.smoke10 else None,
        "qa_root": str(args.qa_root.resolve()), "round5_qa_root": str(args.round5_qa_root.resolve()),
        "mfa_executable": str(args.mfa_executable.resolve()),
        "mfa_dictionary_sha256": sha256_file(args.mfa_dictionary),
        "openface_executable_sha256": sha256_file(args.openface_executable),
        "face_model_sha256": sha256_file(args.face_model),
        "environment": collect_environment(),
        "full_100_sample_feature_extraction": "RUN" if args.all else "NOT_RUN",
    })
    for index, sample_id in enumerate(selected, 1):
        result = process_one(
            by_id[sample_id], output_root=args.output_dir, qa_root=args.qa_root,
            round5_root=args.round5_qa_root, config=config,
            dictionary=args.mfa_dictionary, mfa_work_dir=args.mfa_work_dir,
            face_model=args.face_model, openface_executable=args.openface_executable,
            details_dir=details_dir)
        print({"progress": f"{index}/{len(selected)}", **result}, flush=True)
        write_smoke_summary(args.output_dir, selected, details_dir=details_dir,
                            summary_name=summary_name,
                            full100_status="RUN" if args.all else "NOT_RUN")
        if result.get("processing_status") == "failed":
            # Isolated failure: continue only on an explicitly selected batch.
            if not (args.smoke10 or args.all):
                break
    final = write_smoke_summary(args.output_dir, selected, details_dir=details_dir,
                                summary_name=summary_name,
                                full100_status="RUN" if args.all else "NOT_RUN")
    print(final, flush=True)
    return int(final["failed_count"] > 0 or final["completed_with_mfa_failure_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
