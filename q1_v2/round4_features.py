"""Selective round-4 feature extraction routed by persisted correspondence QA.

There is intentionally no --all option: 100-sample full extraction needs a
separate review after the light QA and representative real experiments.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .alignment import ALIGNMENT_FIELDS, WordAlignment, parse_mfa_output, remap_mfa_words, run_mfa_alignment
from .audio_features import AudioFeatureResult, aggregate_audio_to_words, extract_audio_features
from .config import Q1Config
from .data_loader import SampleRecord, load_samples
from .fusion_alignment import build_fused_sample, validate_fused_output
from .io_utils import read_json, sha256_file, write_csv, write_json
from .media import MediaResult, inspect_and_extract_media, write_pcm16_wav
from .text_features import TextFeatureResult, extract_text_features
from .visual_features import extract_visual_features


def unavailable_alignment(record: SampleRecord, reason: str) -> list[WordAlignment]:
    from .alignment import normalize_word

    return [WordAlignment(
        record.sample_id, word.index, word.text, normalize_word(word.text),
        float("nan"), float("nan"), 0, reason,
    ) for word in record.original_words]


def crop_for_local_mfa(
    media: MediaResult, match_start_s: float, match_end_s: float,
    *, margin_s: float, output_path: Path,
) -> dict[str, float | int | str]:
    if not media.audio_timestamp_reliable or not np.isfinite(media.wav_origin_sample_s):
        raise ValueError("Cannot crop MFA input without reliable audio PTS")
    if not (np.isfinite(match_start_s) and np.isfinite(match_end_s) and match_end_s > match_start_s):
        raise ValueError("Invalid ASR local-match candidate interval")
    wav_start = media.wav_origin_sample_s
    wav_end = wav_start + len(media.waveform) / media.sample_rate
    start_global = max(wav_start, match_start_s - margin_s)
    end_global = min(wav_end, match_end_s + margin_s)
    start_index = max(0, int(round((start_global - wav_start) * media.sample_rate)))
    end_index = min(len(media.waveform), int(round((end_global - wav_start) * media.sample_rate)))
    if end_index <= start_index:
        raise ValueError("Empty MFA crop after mapping to source WAV samples")
    actual_start = wav_start + start_index / media.sample_rate
    actual_end = wav_start + end_index / media.sample_rate
    write_pcm16_wav(output_path, media.waveform[start_index:end_index], media.sample_rate)
    return {
        "matched_audio_start_s": match_start_s,
        "matched_audio_end_s": match_end_s,
        "margin_s": margin_s,
        "crop_start_sample_index_in_full_wav": start_index,
        "crop_end_sample_index_in_full_wav": end_index,
        "crop_start_global_s": actual_start,
        "crop_end_global_s": actual_end,
        "global_mapping": "t_global_sample_s = crop_start_global_s + t_mfa_local_s",
        "full_wav_origin_media_s": media.wav_origin_media_s,
        "crop_wav_origin_media_s": media.timeline_origin_media_s + actual_start,
        "source_wav": media.wav_path,
        "crop_wav": output_path.name,
    }


def _empty_audio(record: SampleRecord, config: Q1Config) -> AudioFeatureResult:
    n = record.word_count
    frame_dim = config.n_mels + 5
    return AudioFeatureResult(
        frame_features=np.empty((0, frame_dim), dtype=np.float32),
        frame_times_s=np.empty(0, dtype=np.float64),
        frame_start_samples=np.empty(0, dtype=np.int64),
        f0_valid_mask=np.empty(0, dtype=np.uint8),
        f0_analysis_times_s=np.empty(0, dtype=np.float64),
        f0_analysis_hz=np.empty(0, dtype=np.float32),
        f0_analysis_valid_mask=np.empty(0, dtype=np.uint8),
        feature_names=[f"log_mel_{i:02d}" for i in range(config.n_mels)] + [
            "rms", "f0_hz", "zero_crossing_rate", "spectral_centroid_hz", "spectral_bandwidth_hz",
        ],
        word_features=np.full((n, frame_dim * 2), np.nan, dtype=np.float32),
        word_mask=np.zeros(n, dtype=np.uint8),
        word_frame_indices=[[] for _ in range(n)],
        word_frame_counts=np.zeros(n, dtype=np.int32),
        word_f0_valid_counts=np.zeros(n, dtype=np.int32),
        word_feature_valid_counts=np.zeros((n, frame_dim), dtype=np.int32),
        word_failure_reasons=["audio_time_axis_unavailable" for _ in range(n)],
        sample_rate=config.sample_rate,
        window_samples=config.window_samples,
        hop_samples=config.hop_samples,
        f0_window_samples=config.f0_window_samples,
    )


def reuse_verified_native_features(
    record: SampleRecord, media: MediaResult, alignments: Sequence[WordAlignment],
    source_dir: Path, output_dir: Path,
) -> tuple[TextFeatureResult, AudioFeatureResult]:
    """Reuse *measured* BERT/audio frames from the same immutable MP4/text.

Only word-level audio statistics are recomputed. A previous forced alignment's
word arrays are never reused, particularly for INVALID_CORRESPONDENCE.
    """
    source_input = read_json(source_dir / "sample_input.json")
    source_media = read_json(source_dir / "media_metadata.json")
    if source_input["sample_id"] != record.sample_id or source_input["text"] != record.text:
        raise ValueError("Native feature source sample ID or official transcript differs")
    video_sha = sha256_file(record.video_path)
    if source_input["video_sha256"] != video_sha:
        raise ValueError("Native feature source MP4 SHA-256 differs")
    if int(source_media["sample_rate"]) != media.sample_rate:
        raise ValueError("Native audio source sample rate differs")
    if abs(float(source_media["wav_origin_sample_s"]) - media.wav_origin_sample_s) > 1 / media.sample_rate:
        raise ValueError("Native audio source time origin differs")
    if abs(float(source_media["decoded_audio_duration_s"]) - media.decoded_audio_duration_s) > 1 / media.sample_rate:
        raise ValueError("Native audio duration differs")

    with np.load(source_dir / "text_features.npz", allow_pickle=False) as arrays:
        text_features = arrays["features"].copy()
        text_mask = arrays["mask"].copy()
    if text_features.shape[0] != record.word_count or text_mask.shape[0] != record.word_count:
        raise ValueError("Native text features lost or added official words")
    text_meta = read_json(source_dir / "text_features_metadata.json")
    text = TextFeatureResult(
        text_features, text_mask, [], int(text_features.shape[1]), int(text_mask.sum()),
        text_meta["model_name"], text_meta.get("model_revision"), text_meta.get("pooling", "mean"),
        text_meta.get("windows", []), text_meta.get("tokenizer_class", ""),
    )
    shutil.copy2(source_dir / "text_features.npz", output_dir / "text_features.npz")
    shutil.copy2(source_dir / "text_features_metadata.json", output_dir / "text_features_metadata.json")

    with np.load(source_dir / "audio_features.npz", allow_pickle=False) as arrays:
        source_audio = {name: arrays[name].copy() for name in arrays.files}
    audio_meta = read_json(source_dir / "audio_features_metadata.json")
    frame_times = source_audio["frame_times_s"]
    expected_times = media.wav_origin_sample_s + (
        source_audio["frame_start_samples"] + int(audio_meta["window_samples"]) / 2
    ) / media.sample_rate
    if not np.allclose(frame_times, expected_times, rtol=0, atol=1 / media.sample_rate):
        raise ValueError("Source acoustic frame times do not map to current media time axis")
    aggregated = aggregate_audio_to_words(
        source_audio["frame_features"], frame_times,
        source_audio["f0_valid_mask"], alignments,
    )
    audio = AudioFeatureResult(
        frame_features=source_audio["frame_features"],
        frame_times_s=frame_times,
        frame_start_samples=source_audio["frame_start_samples"],
        f0_valid_mask=source_audio["f0_valid_mask"],
        f0_analysis_times_s=source_audio["f0_analysis_times_s"],
        f0_analysis_hz=source_audio["f0_analysis_hz"],
        f0_analysis_valid_mask=source_audio["f0_analysis_valid_mask"],
        feature_names=audio_meta["feature_names"],
        word_features=aggregated[0], word_mask=aggregated[1],
        word_frame_indices=aggregated[2], word_frame_counts=aggregated[3],
        word_f0_valid_counts=aggregated[4], word_feature_valid_counts=aggregated[5],
        word_failure_reasons=aggregated[6],
        sample_rate=int(audio_meta["sample_rate"]),
        window_samples=int(audio_meta["window_samples"]),
        hop_samples=int(audio_meta["hop_samples"]),
        f0_window_samples=int(audio_meta["f0_window_samples"]),
    )
    np.savez_compressed(output_dir / "audio_features.npz", **{
        key: getattr(audio, key) for key in (
            "frame_features", "frame_times_s", "frame_start_samples", "f0_valid_mask",
            "f0_analysis_times_s", "f0_analysis_hz", "f0_analysis_valid_mask",
            "word_features", "word_mask", "word_frame_counts", "word_f0_valid_counts",
            "word_feature_valid_counts",
        )
    })
    write_json(output_dir / "audio_features_metadata.json", audio.metadata())
    write_json(output_dir / "native_feature_reuse_provenance.json", {
        "source_dir": str(source_dir.resolve()),
        "source_video_sha256": video_sha,
        "source_text_features_sha256": sha256_file(source_dir / "text_features.npz"),
        "source_audio_features_sha256": sha256_file(source_dir / "audio_features.npz"),
        "reused": "real BERT word features and real timestamped acoustic frames",
        "recomputed": "audio word mean/std and masks on round-4 word intervals",
        "not_reused": "source forced-alignment word statistics or source word masks",
    })
    return text, audio


def process_one(
    record: SampleRecord, output_dir: Path, *, face_model: Path,
    dictionary: str, mfa_work_dir: Path,
    config: Q1Config,
    alignment_source: Path | None = None,
    reuse_native_source: Path | None = None,
) -> dict[str, object]:
    sample_dir = output_dir / "samples" / record.sample_id
    qa_path = sample_dir / "correspondence_qa.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"QA must precede feature extraction: {qa_path}")
    qa = read_json(qa_path)
    if (sample_dir / "_SUCCESS.json").is_file():
        valid, errors = validate_fused_output(sample_dir, record.sample_id)
        if valid:
            return {"sample_id": record.sample_id, "action": "skipped_complete", "route": qa["route"]}
        raise RuntimeError(f"Existing output is corrupt; retained for diagnosis: {errors}")
    start = time.perf_counter()
    stage = "media"
    try:
        write_json(sample_dir / "sample_input.json", {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "source_row": record.source_row,
            "text": record.text,
            "video_path": str(record.video_path.resolve()),
            "video_sha256": sha256_file(record.video_path),
            "original_words": [word.text for word in record.original_words],
            "label_usage": "audit-only; excluded from ASR, MFA, BERT, audio, visual and fusion",
        })
        media = inspect_and_extract_media(
            record.video_path, sample_dir / "audio_mfa_16k_mono.wav",
            sample_rate=config.sample_rate,
            metadata_path=sample_dir / "media_metadata.json",
        )
        stage = "alignment"
        alignment = unavailable_alignment(record, "correspondence_not_high_confidence")
        if qa["route"] == "HIGH_CONFIDENCE_MATCH":
            crop_path = sample_dir / "audio_mfa_local_crop.wav"
            crop = crop_for_local_mfa(
                media, float(qa["matched_audio_start_s"]), float(qa["matched_audio_end_s"]),
                margin_s=0.30, output_path=crop_path,
            )
            write_json(sample_dir / "mfa_crop_metadata.json", crop)
            try:
                if alignment_source is not None:
                    raw = alignment_source / "mfa_raw.json"
                    if not raw.is_file():
                        raise FileNotFoundError(raw)
                    alignment = remap_mfa_words(
                        record.sample_id, record.original_words, parse_mfa_output(raw),
                        wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
                        timeline_origin_media_s=media.timeline_origin_media_s,
                        duration_s=media.duration_s,
                    )
                    shutil.copy2(raw, sample_dir / "mfa_raw.json")
                    write_csv(sample_dir / "word_alignment.csv", (asdict(word) for word in alignment), ALIGNMENT_FIELDS)
                    write_json(sample_dir / "alignment_import_provenance.json", {
                        "source_directory": str(alignment_source.resolve()),
                        "source_raw_sha256": sha256_file(raw),
                        "method": "Re-map real MFA local intervals through freshly computed crop origin",
                    })
                    for name in ("mfa_stdout.log", "mfa_stderr.log", "mfa_command.json"):
                        source_file = alignment_source / name
                        if source_file.is_file():
                            shutil.copy2(source_file, sample_dir / name)
                else:
                    result = run_mfa_alignment(
                        sample_id=record.sample_id,
                        original_text=record.text,
                        original_words=record.original_words,
                        wav_path=crop_path,
                        output_dir=sample_dir,
                        acoustic_model=config.mfa_acoustic_model,
                        dictionary=dictionary,
                        wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
                        timeline_origin_media_s=media.timeline_origin_media_s,
                        duration_s=media.duration_s,
                        timeout_s=config.mfa_timeout_s,
                        temporary_directory=mfa_work_dir,
                        beam=config.mfa_beam,
                    )
                    alignment = result.words
            except Exception as exc:
                write_json(sample_dir / "_MFA_FAILED.json", {
                    "sample_id": record.sample_id, "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "policy": "No word timestamps retained after failed MFA; native modalities continue",
                })
                alignment = unavailable_alignment(record, "local_mfa_failed")
                write_csv(sample_dir / "word_alignment.csv", (asdict(word) for word in alignment), ALIGNMENT_FIELDS)
        else:
            write_csv(sample_dir / "word_alignment.csv", (asdict(word) for word in alignment), ALIGNMENT_FIELDS)
        if not (sample_dir / "alignment_metadata.json").is_file():
            write_json(sample_dir / "alignment_metadata.json", {
                "sample_id": record.sample_id,
                "status": "ok" if all(item.alignment_mask for item in alignment) else (
                    "partial" if any(item.alignment_mask for item in alignment) else "unavailable"
                ),
                "source": "real_local_mfa" if (sample_dir / "mfa_raw.json").is_file() else "no_trusted_alignment",
                "words": [asdict(item) for item in alignment],
            })
        stage = "text_audio"
        if reuse_native_source is not None:
            text, audio = reuse_verified_native_features(
                record, media, alignment, reuse_native_source, sample_dir,
            )
        else:
            text = extract_text_features(
                record.original_words, model_name=config.text_model,
                device="cpu", output_dir=sample_dir,
            )
            audio = (
                extract_audio_features(
                    media.waveform, sample_rate=config.sample_rate,
                    wav_origin_sample_s=media.wav_origin_sample_s,
                    alignments=alignment, window_samples=config.window_samples,
                    hop_samples=config.hop_samples,
                    f0_window_samples=config.f0_window_samples,
                    n_mels=config.n_mels, fmin_hz=config.fmin_hz, fmax_hz=config.fmax_hz,
                    output_dir=sample_dir,
                )
                if media.audio_timestamp_reliable and len(media.waveform) else _empty_audio(record, config)
            )
        stage = "visual"
        visual = extract_visual_features(
            record.video_path, media.video_frames, alignment,
            face_model_path=face_model,
            selected_landmarks=tuple(range(478)),  # Native MediaPipe topology, NOT OpenFace/iBUG 68.
            sampling_fps=config.visual_fps,
            max_faces=config.max_faces,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            nearest_max_distance_s=None,
            output_dir=sample_dir,
        )
        stage = "fusion"
        build_fused_sample(
            record.sample_id, record.original_words, alignment, text, audio, visual,
            correspondence_qa=qa, output_dir=sample_dir,
        )
        write_json(sample_dir / "_SUCCESS.json", {
            "sample_id": record.sample_id,
            "schema_version": "q1_v2.1-round4",
            "route": qa["route"],
            "feature_elapsed_s": time.perf_counter() - start,
            "elapsed_s": time.perf_counter() - start,
            "visual_topology": "MediaPipe-native-478-not-OpenFace68",
        })
        valid, errors = validate_fused_output(sample_dir, record.sample_id)
        if not valid:
            raise RuntimeError(f"Fused output integrity failed: {errors}")
        return {
            "sample_id": record.sample_id, "action": "processed", "route": qa["route"],
            "word_count": record.word_count,
            "aligned_word_count": sum(item.alignment_mask for item in alignment),
            "text_word_count": int(text.mask.sum()),
            "audio_word_count": int(audio.word_mask.sum()),
            "visual_word_count": int(visual.word_mask.sum()),
            "visual_frame_dim": int(visual.frame_features.shape[1]),
            "visual_word_dim": int(visual.word_features.shape[1]),
            "feature_elapsed_s": time.perf_counter() - start,
        }
    except Exception as exc:
        (sample_dir / "_SUCCESS.json").unlink(missing_ok=True)
        write_json(sample_dir / "_FEATURE_FAILED.json", {
            "sample_id": record.sample_id, "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": time.perf_counter() - start,
        })
        return {"sample_id": record.sample_id, "action": "failed", "stage": stage, "error": str(exc)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--mfa-dictionary", required=True)
    parser.add_argument("--mfa-executable", type=Path, required=True)
    parser.add_argument("--mfa-root-dir", type=Path, required=True)
    parser.add_argument("--mfa-work-dir", type=Path, required=True)
    parser.add_argument("--visual-fps", type=float, choices=(5.0, 10.0), default=10.0)
    parser.add_argument("--retry-feature", action="store_true", help="Archive an existing attempt before rebuilding")
    parser.add_argument("--alignment-source", type=Path, help="Reuse an independently verified real MFA raw output for one selected sample")
    parser.add_argument("--reuse-native-source", type=Path, help="Reuse verified real BERT/acoustic frames from a prior run of the same immutable MP4 and text")
    args = parser.parse_args(argv)
    os.environ["MFA_ROOT_DIR"] = str(args.mfa_root_dir.resolve())
    mfa_exe = args.mfa_executable.resolve()
    if not mfa_exe.is_file():
        raise FileNotFoundError(mfa_exe)
    os.environ["PATH"] = str(mfa_exe.parent) + os.pathsep + os.environ.get("PATH", "")
    records, _ = load_samples(args.data_root, args.labels, probe_decode=False)
    by_id = {record.sample_id: record for record in records}
    if len(args.sample_id) > 12:
        raise ValueError("Representative round-4 validation is capped at 12 samples")
    if args.alignment_source is not None and len(args.sample_id) != 1:
        raise ValueError("--alignment-source requires exactly one --sample-id")
    if args.reuse_native_source is not None and len(args.sample_id) != 1:
        raise ValueError("--reuse-native-source requires exactly one --sample-id")
    config = Q1Config(visual_fps=args.visual_fps, mfa_work_root=str(args.mfa_work_dir.resolve()))
    write_json(args.output_dir.resolve() / "feature_experiment_config.json", {
        "config": config.to_dict(),
        "text_model": config.text_model,
        "visual_topology": "MediaPipe-native-478-not-OpenFace68",
        "visual_landmark_count": 478,
        "visual_fps": args.visual_fps,
        "mfa_acoustic_model": config.mfa_acoustic_model,
        "mfa_dictionary": args.mfa_dictionary,
        "mfa_dictionary_sha256": sha256_file(Path(args.mfa_dictionary)) if Path(args.mfa_dictionary).is_file() else None,
        "mfa_executable": str(mfa_exe),
        "mfa_root_dir": str(args.mfa_root_dir.resolve()),
        "mfa_work_dir": str(args.mfa_work_dir.resolve()),
        "face_model": str(args.face_model.resolve()),
        "face_model_sha256": sha256_file(args.face_model),
        "selected_samples_only": list(args.sample_id),
        "all_100_full_extraction": False,
    })
    failed = False
    for sample_id in args.sample_id:
        if sample_id not in by_id:
            raise ValueError(f"Not an official sample: {sample_id}")
        alignment_source = args.alignment_source.resolve() if args.alignment_source else None
        if args.retry_feature:
            existing = args.output_dir.resolve() / "samples" / sample_id
            if existing.is_dir():
                if not existing.resolve().is_relative_to(args.output_dir.resolve()):
                    raise ValueError(f"Unsafe archive target: {existing}")
                if alignment_source is not None and alignment_source.is_relative_to(existing.resolve()):
                    retained = args.output_dir.resolve() / "mfa_verified" / f"{sample_id}-{uuid.uuid4().hex[:8]}"
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(alignment_source, retained)
                    alignment_source = retained
                archive = args.output_dir.resolve() / "attempts" / f"{sample_id}-{uuid.uuid4().hex[:8]}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                existing.replace(archive)
                existing.mkdir(parents=True, exist_ok=False)
                for name in (
                    "correspondence_qa.json", "asr_segments.json", "asr_words.csv",
                    "asr_worker_raw.json", "asr_command.json", "asr_stdout.log",
                    "asr_stderr.log", "local_match.json",
                ):
                    source = archive / name
                    if source.is_file():
                        shutil.copy2(source, existing / name)
                write_json(existing / "previous_attempt.json", {"archive_path": str(archive)})
        result = process_one(
            by_id[sample_id], args.output_dir.resolve(), face_model=args.face_model.resolve(),
            dictionary=args.mfa_dictionary, mfa_work_dir=args.mfa_work_dir.resolve(),
            config=config, alignment_source=alignment_source,
            reuse_native_source=args.reuse_native_source.resolve() if args.reuse_native_source else None,
        )
        print(result, flush=True)
        failed |= result["action"] == "failed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
