"""Write a machine-readable validation report for the isolated Q1 environment."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from .alignment import probe_mfa
from .io_utils import sha256_file, write_json


PACKAGE_NAMES = (
    "numpy", "scipy", "openpyxl", "av", "librosa", "soundfile", "torch",
    "transformers", "tokenizers", "huggingface-hub", "mediapipe",
    "opencv-contrib-python", "matplotlib", "seaborn", "pandas",
    "montreal-forced-aligner", "kalpy",
)


def _version_command(argv: list[str]) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"available": False, "argv": argv, "error": "executable_not_found"}
    completed = subprocess.run(
        [executable, *argv[1:]], capture_output=True, text=True, check=False,
        encoding="utf-8", errors="replace", timeout=60,
    )
    return {
        "available": completed.returncode == 0,
        "executable": executable,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mfa-root-dir", type=Path, required=True)
    parser.add_argument("--mfa-work-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    args = parser.parse_args(argv)
    mfa_root = args.mfa_root_dir.resolve()
    mfa_work = args.mfa_work_dir.resolve()
    os.environ["MFA_ROOT_DIR"] = str(mfa_root)
    import torch

    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "mfa_root_dir": str(mfa_root),
        "mfa_root_ascii_only": str(mfa_root).isascii(),
        "mfa_work_dir": str(mfa_work),
        "mfa_work_ascii_only": str(mfa_work).isascii(),
        "mfa_probe": probe_mfa("english_us_arpa", "english_us_arpa"),
        "commands": {
            "mfa_help": _version_command(["mfa", "--help"]),
            "ffmpeg_version": _version_command(["ffmpeg", "-version"]),
        },
        "face_model": {
            "path": str(args.face_model.resolve()),
            "size_bytes": args.face_model.stat().st_size,
            "sha256": sha256_file(args.face_model),
        },
        "pip_check_command": f"{sys.executable} -m pip check",
        "pip_check_result": _version_command([sys.executable, "-m", "pip", "check"]),
    }
    write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
