"""End-to-end, resumable Question 1 v2 feature pipeline."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .alignment import run_mfa_alignment
from .audio_features import extract_audio_features
from .config import Q1Config, save_experiment_config
from .data_loader import SampleRecord, load_samples
from .fusion_alignment import build_fused_sample, validate_fused_output
from .io_utils import directory_size, read_json, sha256_file, write_json
from .media import inspect_and_extract_media
from .quality import build_quality_reports
from .text_features import extract_text_features
from .visual_features import extract_visual_features


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _archive_existing(path: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = archive_root / f"{path.name}-{timestamp}-{uuid.uuid4().hex[:8]}"
    path.replace(destination)
    return destination


def _resume_decision(
    sample_dir: Path,
    sample_id: str,
    config_hash: str,
    *,
    retry_failed: bool,
) -> tuple[str, str]:
    if not sample_dir.exists():
        return "run", "new_sample"
    success_path = sample_dir / "_SUCCESS.json"
    if success_path.is_file():
        success = read_json(success_path)
        valid, errors = validate_fused_output(sample_dir, sample_id)
        if success.get("config_sha256") == config_hash and valid:
            return "skip", "valid_completed_output"
        return "archive_run", "completed_output_config_or_integrity_mismatch:" + "|".join(errors)
    if (sample_dir / "_FAILED.json").is_file():
        return ("archive_run", "retry_failed_requested") if retry_failed else ("skip_failed", "previous_failure_use_--retry-failed")
    return "archive_run", "incomplete_existing_output"


def process_sample(
    record: SampleRecord,
    output_dir: Path,
    config: Q1Config,
    *,
    device: str,
    retry_failed: bool,
) -> dict[str, Any]:
    samples_root = output_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    sample_dir = samples_root / record.sample_id
    action, reason = _resume_decision(
        sample_dir, record.sample_id, config.fingerprint(), retry_failed=retry_failed,
    )
    if action == "skip":
        return {"sample_id": record.sample_id, "status": "skipped_complete", "reason": reason}
    if action == "skip_failed":
        return {"sample_id": record.sample_id, "status": "skipped_failed", "reason": reason}
    if action == "archive_run":
        _archive_existing(sample_dir, output_dir / "archived_attempts")

    staging = samples_root / f".{record.sample_id}.inprogress-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    current_stage = "input_validation"
    try:
        write_json(staging / "sample_input.json", {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "source_row": record.source_row,
            "text": record.text,
            "video_path": str(record.video_path.resolve()),
            "video_sha256": sha256_file(record.video_path),
            "original_words": [word.text for word in record.original_words],
            "label_usage": "audit-only; label and annotation intentionally omitted from feature stages",
        })
        if record.read_status != "ok":
            raise RuntimeError("Input validation failed: " + " | ".join(record.errors))

        current_stage = "media"
        media = inspect_and_extract_media(
            record.video_path,
            staging / "audio_mfa_16k_mono.wav",
            sample_rate=config.sample_rate,
            metadata_path=staging / "media_metadata.json",
        )
        if not media.audio_timestamp_reliable:
            raise RuntimeError("Audio PTS mapping is unreliable; refusing forced alignment")

        current_stage = "alignment"
        alignment = run_mfa_alignment(
            sample_id=record.sample_id,
            original_text=record.text,
            original_words=record.original_words,
            wav_path=staging / "audio_mfa_16k_mono.wav",
            output_dir=staging,
            acoustic_model=config.mfa_acoustic_model,
            dictionary=config.mfa_dictionary,
            wav_origin_media_s=media.wav_origin_media_s,
            timeline_origin_media_s=media.timeline_origin_media_s,
            duration_s=media.duration_s,
            timeout_s=config.mfa_timeout_s,
            temporary_directory=Path(config.mfa_work_root) if config.mfa_work_root else None,
            beam=config.mfa_beam,
        )
        current_stage = "text"
        text = extract_text_features(
            record.original_words,
            model_name=config.text_model,
            pooling=config.text_pooling,
            device=device,
            overlap_words=config.text_window_overlap_words,
            output_dir=staging,
        )
        current_stage = "audio"
        audio = extract_audio_features(
            media.waveform,
            sample_rate=config.sample_rate,
            wav_origin_sample_s=media.wav_origin_sample_s,
            alignments=alignment.words,
            window_samples=config.window_samples,
            hop_samples=config.hop_samples,
            f0_window_samples=config.f0_window_samples,
            n_mels=config.n_mels,
            fmin_hz=config.fmin_hz,
            fmax_hz=config.fmax_hz,
            output_dir=staging,
        )
        if config.face_model_path is None:
            raise FileNotFoundError("--face-model is required for MediaPipe Face Landmarker")
        current_stage = "visual"
        visual = extract_visual_features(
            record.video_path,
            media.video_frames,
            alignment.words,
            face_model_path=Path(config.face_model_path),
            selected_landmarks=config.selected_landmarks,
            sampling_fps=config.visual_fps,
            max_faces=config.max_faces,
            min_face_detection_confidence=config.min_face_detection_confidence,
            min_face_presence_confidence=config.min_face_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            nearest_max_distance_s=config.nearest_visual_max_distance_s,
            output_dir=staging,
        )
        current_stage = "fusion"
        build_fused_sample(
            record.sample_id, record.original_words, alignment.words, text, audio, visual,
            output_dir=staging,
        )
        elapsed = time.perf_counter() - start
        current_stage = "output_validation"
        write_json(staging / "_SUCCESS.json", {
            "sample_id": record.sample_id,
            "status": "success",
            "completed_at_utc": _utc_now(),
            "elapsed_s": elapsed,
            "config_sha256": config.fingerprint(),
            "schema_version": config.schema_version,
        })
        valid, errors = validate_fused_output(staging, record.sample_id)
        if not valid:
            raise RuntimeError("Output integrity validation failed: " + " | ".join(errors))
        staging.replace(sample_dir)
        return {
            "sample_id": record.sample_id,
            "status": "success",
            "elapsed_s": elapsed,
            "output_size_bytes": directory_size(sample_dir),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - start
        success_marker = staging / "_SUCCESS.json"
        if success_marker.exists():
            success_marker.unlink()
        write_json(staging / "_FAILED.json", {
            "sample_id": record.sample_id,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "elapsed_s": elapsed,
            "config_sha256": config.fingerprint(),
            "failure_stage": current_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        staging.replace(sample_dir)
        return {
            "sample_id": record.sample_id,
            "status": "failed",
            "elapsed_s": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Attachment 1 root containing video_id folders")
    parser.add_argument("--labels", type=Path, default=None, help="label-100.xlsx; auto-detected below data root if omitted")
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--max-samples", type=int)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--sample-id", action="append", help="Process one or more exact video_id__clip_id values")
    parser.add_argument("--visual-fps", type=float, choices=(5.0, 10.0), default=10.0)
    parser.add_argument("--face-model", type=Path, default=None)
    parser.add_argument("--text-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mfa-acoustic-model", default="english_us_arpa")
    parser.add_argument("--mfa-dictionary", default="english_us_arpa")
    parser.add_argument("--mfa-beam", type=int, default=None, help="Optional diagnostic beam width for MFA align_one")
    parser.add_argument(
        "--mfa-root-dir", type=Path, default=None,
        help="ASCII-only MFA model/config root (sets MFA_ROOT_DIR for child processes)",
    )
    parser.add_argument(
        "--mfa-work-dir", type=Path, default=None,
        help="ASCII-only native MFA/Kaldi work root; defaults to the system temp directory",
    )
    parser.add_argument("--manual-reference-csv", type=Path, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-decode-probe", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.mfa_beam is not None and args.mfa_beam <= 0:
        raise SystemExit("--mfa-beam must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mfa_work_dir = (
        args.mfa_work_dir.resolve()
        if args.mfa_work_dir is not None
        else (Path(tempfile.gettempdir()) / "q1_v2_mfa_work").resolve()
    )
    if not str(mfa_work_dir).isascii():
        raise SystemExit(f"--mfa-work-dir must be ASCII-only for Windows Kaldi/OpenFST: {mfa_work_dir}")
    inherited_mfa_root = os.environ.get("MFA_ROOT_DIR")
    mfa_root_dir = (
        args.mfa_root_dir.resolve()
        if args.mfa_root_dir is not None
        else Path(inherited_mfa_root).resolve() if inherited_mfa_root else None
    )
    if mfa_root_dir is not None:
        if not str(mfa_root_dir).isascii():
            raise SystemExit(f"--mfa-root-dir must be ASCII-only for Windows Kaldi: {mfa_root_dir}")
        os.environ["MFA_ROOT_DIR"] = str(mfa_root_dir)
    # Keep model downloads under the selected output tree, which is expected to
    # be git-ignored/external, rather than inside tracked source directories.
    os.environ.setdefault("HF_HOME", str(output_dir / "cache" / "huggingface"))
    device = _resolve_device(args.device)
    face_model_hash = (
        sha256_file(args.face_model.resolve())
        if args.face_model is not None and args.face_model.is_file() else None
    )
    config = Q1Config(
        visual_fps=args.visual_fps,
        face_model_path=str(args.face_model.resolve()) if args.face_model is not None else None,
        face_model_sha256=face_model_hash,
        inference_device=device,
        text_model=args.text_model,
        mfa_acoustic_model=args.mfa_acoustic_model,
        mfa_dictionary=args.mfa_dictionary,
        mfa_beam=args.mfa_beam,
        mfa_root_dir=str(mfa_root_dir) if mfa_root_dir is not None else None,
        mfa_work_root=str(mfa_work_dir),
    )
    save_experiment_config(output_dir, config, vars(args))
    records, loading_summary = load_samples(
        args.data_root,
        args.labels,
        expected_count=config.expected_sample_count,
        probe_decode=not args.skip_decode_probe,
        output_dir=output_dir,
    )
    if args.all and len(records) != config.expected_sample_count:
        raise RuntimeError(
            f"--all refuses a noncanonical dataset: expected {config.expected_sample_count}, found {len(records)}"
        )
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [record for record in records if record.sample_id in requested]
        missing = requested - {record.sample_id for record in selected}
        if missing:
            raise ValueError(f"Unknown --sample-id values: {sorted(missing)}")
    else:
        selected = records if args.all else records[:args.max_samples]
    write_json(output_dir / "run_selection.json", {
        "selected_sample_ids": [record.sample_id for record in selected],
        "selected_count": len(selected),
        "dataset_count": len(records),
        "all_requested": args.all,
    })
    outcomes: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {record.sample_id}", flush=True)
        outcome = process_sample(
            record, output_dir, config, device=device, retry_failed=args.retry_failed,
        )
        outcomes.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False), flush=True)
        write_json(output_dir / "run_progress.json", {
            "updated_at_utc": _utc_now(),
            "outcomes": outcomes,
        })
    summary = build_quality_reports(
        output_dir,
        records,
        manual_reference_csv=args.manual_reference_csv,
        selected_sample_ids={record.sample_id for record in selected},
    )
    write_json(output_dir / "run_complete.json", {
        "completed_at_utc": _utc_now(),
        "selected_count": len(selected),
        "success_count": sum(item["status"] in {"success", "skipped_complete"} for item in outcomes),
        "failed_count": sum(item["status"] in {"failed", "skipped_failed"} for item in outcomes),
        "quality_summary": summary,
    })
    return 0 if not any(item["status"] in {"failed", "skipped_failed"} for item in outcomes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
