"""Run an auditable MFA-only smoke test on one immutable real sample."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .alignment import run_mfa_alignment
from .data_loader import load_samples
from .io_utils import write_json
from .media import inspect_and_extract_media


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mfa-acoustic-model", default="english_us_arpa")
    parser.add_argument("--mfa-dictionary", default="english_us_arpa")
    parser.add_argument("--mfa-root-dir", type=Path, default=None)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--mfa-work-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mfa_root_dir is not None:
        mfa_root_dir = args.mfa_root_dir.resolve()
        if not str(mfa_root_dir).isascii():
            raise RuntimeError(f"MFA root directory must be ASCII-only: {mfa_root_dir}")
        os.environ["MFA_ROOT_DIR"] = str(mfa_root_dir)
    records, _ = load_samples(
        args.data_root,
        args.labels,
        expected_count=100,
        probe_decode=False,
        output_dir=output_dir,
    )
    matches = [record for record in records if record.sample_id == args.sample_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one sample_id={args.sample_id!r}, found {len(matches)}")
    record = matches[0]
    sample_dir = output_dir / "samples" / record.sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    media = inspect_and_extract_media(
        record.video_path,
        sample_dir / "audio_mfa_16k_mono.wav",
        sample_rate=16000,
        metadata_path=sample_dir / "media_metadata.json",
    )
    mfa_work_dir = (
        args.mfa_work_dir.resolve()
        if args.mfa_work_dir is not None
        else (Path(tempfile.gettempdir()) / "q1_v2_mfa_work").resolve()
    )
    if not str(mfa_work_dir).isascii():
        raise RuntimeError(f"MFA native work directory must be ASCII-only: {mfa_work_dir}")
    result = run_mfa_alignment(
        sample_id=record.sample_id,
        original_text=record.text,
        original_words=record.original_words,
        wav_path=sample_dir / "audio_mfa_16k_mono.wav",
        output_dir=sample_dir,
        acoustic_model=args.mfa_acoustic_model,
        dictionary=args.mfa_dictionary,
        wav_origin_media_s=media.wav_origin_media_s,
        timeline_origin_media_s=media.timeline_origin_media_s,
        duration_s=media.duration_s,
        timeout_s=args.timeout_s,
        temporary_directory=mfa_work_dir,
    )
    summary = {
        "sample_id": record.sample_id,
        "original_word_count": len(record.original_words),
        "output_word_count": len(result.words),
        "aligned_word_count": sum(word.alignment_mask for word in result.words),
        "status": result.status,
        "mfa_version": result.mfa_version,
    }
    write_json(output_dir / "mfa_smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
