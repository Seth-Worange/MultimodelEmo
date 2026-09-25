"""Download official OpenFace 2.2.0 CEN patch experts into an unpacked binary.

Source URLs are those in the official release's download_models.ps1. Binary,
models, and cache belong under ignored outputs, never in Git.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Sequence

from .io_utils import sha256_file, write_json


MODEL_URLS = {
    "cen_patches_0.25_of.dat": "https://www.dropbox.com/s/7na5qsjzz8yfoer/cen_patches_0.25_of.dat?dl=1",
    "cen_patches_0.35_of.dat": "https://www.dropbox.com/s/k7bj804cyiu474t/cen_patches_0.35_of.dat?dl=1",
    "cen_patches_0.50_of.dat": "https://www.dropbox.com/s/ixt4vkbmxgab1iu/cen_patches_0.50_of.dat?dl=1",
    "cen_patches_1.00_of.dat": "https://www.dropbox.com/s/2t5t1sdpshzfhpj/cen_patches_1.00_of.dat?dl=1",
}
SOURCE_SCRIPT_URL = "https://github.com/TadasBaltrusaitis/OpenFace/blob/OpenFace_2.2.0/download_models.ps1"


def install_models(root: Path) -> dict:
    import requests

    root = root.resolve()
    executable = root / "FeatureExtraction.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    directory = root / "model" / "patch_experts"
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    report = {
        "openface_release": "2.2.0",
        "root": str(root), "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "official_download_script": SOURCE_SCRIPT_URL,
        "models": [],
    }
    failed = False
    for name, url in MODEL_URLS.items():
        target = directory / name
        item = {"name": name, "source_url": url, "target": str(target)}
        try:
            if target.is_file() and target.stat().st_size >= 1_000_000:
                item["action"] = "retained_existing"
            else:
                if target.exists():
                    raise ValueError(f"Existing model too small; refusing overwrite: {target}")
                partial = directory / f"{name}.partial"
                if partial.exists():
                    raise FileExistsError(f"Existing interrupted download requires inspection: {partial}")
                with requests.get(url, stream=True, timeout=(20, 180)) as response:
                    response.raise_for_status()
                    if "html" in response.headers.get("Content-Type", "").lower():
                        raise ValueError("Received HTML instead of a model binary")
                    with partial.open("xb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if partial.stat().st_size < 1_000_000:
                    raise ValueError(f"Downloaded model unexpectedly small: {partial.stat().st_size}")
                partial.replace(target)
                item["action"] = "downloaded"
            item["size_bytes"] = target.stat().st_size
            item["sha256"] = sha256_file(target)
            item["status"] = "ok"
        except Exception as exc:
            failed = True
            item["status"] = "failed"
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc()
        report["models"].append(item)
        write_json(root / "model_download_report.json", report)
    report["status"] = "failed" if failed else "ok"
    write_json(root / "model_download_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openface-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = install_models(args.openface_root)
    for item in report["models"]:
        print({key: item.get(key) for key in ("name", "status", "size_bytes", "error")}, flush=True)
    return int(report["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
