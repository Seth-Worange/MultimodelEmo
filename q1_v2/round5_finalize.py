"""Verify and summarize executed Round-5 outputs without launching new features."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from .io_utils import read_json, write_json


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finalize(root: Path) -> dict:
    manual_report = read_json(root / "manual_qa" / "manual_sample_audit_import_report.json")
    if (manual_report["row_count"], manual_report["unique_sample_count"],
            manual_report["matched_official_sample_count"]) != (100, 100, 100):
        raise ValueError("Manual QA 100-row source coverage invalid")
    boundary_import = read_json(root / "manual_boundary" / "import_report.json")
    boundary = read_json(root / "manual_boundary" / "manual_boundary_metrics_overall.json")
    qa = read_json(root / "quality_summary.json")
    stage2 = read_json(root / "qa" / "stage2_qa_metrics.json")
    confidence = read_json(root / "alignment_confidence" / "alignment_confidence_summary.json")
    comparison = _csv(root / "visual_backend" / "visual_backend_comparison.csv")
    samples = {row["sample_id"] for row in comparison}
    if len(comparison) != len(samples) * 2:
        raise ValueError("Visual comparison must have two backend rows per sample")
    checks = []
    for row in comparison:
        sample_id, backend = row["sample_id"], row["backend"]
        target = root / "samples" / sample_id / backend
        path = target / ("openface68_frames.npz" if backend == "openface68" else "visual_features.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            times = archive["sample_times_s" if backend == "openface68" else "frame_times_s"]
            indices = archive["original_frame_indices"]
            mask = archive["face_valid_mask" if backend == "openface68" else "detection_mask"]
            geometry = archive["landmark_geometry" if backend == "openface68" else "frame_features"]
            if len(times) != len(indices) or len(times) != len(mask) or len(times) != len(geometry):
                raise ValueError(f"Frame trace dimension mismatch: {sample_id} {backend}")
            if len(times) and not np.all(np.diff(times) >= 0):
                raise ValueError(f"Frame time non-monotonic: {sample_id} {backend}")
            if int(mask.astype(bool).sum()) != int(row["face_success_frame_count"]):
                raise ValueError(f"Frame validity count mismatch: {sample_id} {backend}")
            checks.append({"sample_id": sample_id, "backend": backend, "frame_count": len(times),
                           "feature_dim": int(geometry.shape[1]), "npz_allow_pickle_false": True})
        if backend == "openface68":
            word_path = target / "openface68_word_features.npz"
            if row["alignment_word_count"] and not word_path.is_file():
                raise FileNotFoundError(word_path)
            if word_path.is_file():
                with np.load(word_path, allow_pickle=False) as archive:
                    if len(archive["word_visual_mask"]) != int(row["alignment_word_count"]):
                        raise ValueError(f"Word sequence length mismatch: {sample_id}")
                    if int(archive["word_visual_mask"].sum()) != int(row["word_visual_valid_count"]):
                        raise ValueError(f"Word visual count mismatch: {sample_id}")
    schema = read_json(root / "samples" / sorted(samples)[0] / "openface68" / "openface_feature_schema.json")
    if schema["geometry_frame_dimension"] != 136 or schema["backend"] != "openface68":
        raise ValueError("OpenFace schema invalid")
    write_json(root / "visual_backend" / "openface_feature_schema.json", schema)
    write_json(root / "visual_backend" / "openface_version.json", {
        "release": schema["release"], "source_url": schema["source_url"],
        "executable_path": schema["openface_executable"],
        "executable_sha256": schema["openface_executable_sha256"],
        "model_download_report": str(root / "visual_backend" / "openface_download" / "extracted" / "OpenFace_2.2.0_win_x64" / "model_download_report.json"),
    })
    write_json(root / "visual_backend" / "openface_command_examples.json", {
        sample_id: read_json(root / "samples" / sample_id / "openface68" / "openface_command.json")
        for sample_id in sorted(samples)
    })
    experiment = read_json(root / "experiment_config.json")
    experiment.update({
        "manual_qa_source_sha256": manual_report["source_sha256"],
        "manual_boundary_source_sha256": boundary_import["source_sha256"],
        "representative_visual_backends": ["mediapipe478", "openface68"],
        "existing_fused_backend": "mediapipe478",
        "future_visual_backend_candidate": "openface68",
        "representative_visual_fps": 10.0,
        "openface_release": schema["release"],
        "openface_executable_sha256": schema["openface_executable_sha256"],
        "stage2_asr_beam_size": stage2["beam_size_stage2"],
        "stage2_model_independent_from_stage1": False,
        "full_100_sample_feature_extraction": "NOT_RUN",
    })
    write_json(root / "experiment_config.json", experiment)
    qa.update({
        "manual_100_qa": read_json(root / "manual_qa" / "manual_qa_metrics.json"),
        "manual_boundary_source": {"row_count": boundary_import["row_count"],
                                   "distinct_sample_count": boundary_import["distinct_sample_count"],
                                   "valid_interval_count": boundary_import["valid_interval_count"],
                                   "word_mapping_error_count": boundary_import["word_mapping_error_count"]},
        "manual_boundary_evaluation": boundary,
        "asr_mfa_disagreement_audit": {"eligible_real_mfa_sample_count": confidence["eligible_real_mfa_sample_count"],
                                       "audited_word_count": confidence["audited_word_count"],
                                       "confidence_counts": confidence["confidence_counts"]},
        "stage2_review": stage2,
        "visual_backend_comparison": {"sample_count": len(samples),
                                      "backend_sample_rows": len(comparison),
                                      "verified_npz_count": len(checks),
                                      "backend_counts": dict(Counter(row["backend"] for row in comparison))},
        "full_100_sample_feature_extraction": "NOT_RUN",
    })
    write_json(root / "quality_summary.json", qa)
    write_json(root / "visual_backend" / "npz_integrity_checks.json", checks)
    return qa


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = finalize(args.output_dir)
    print({"qa_completed": result["round5_qa_completed_count"],
           "manual_boundary_sample_count": result["manual_boundary_source"]["distinct_sample_count"],
           "visual_comparison_sample_count": result["visual_backend_comparison"]["sample_count"],
           "full_features": result["full_100_sample_feature_extraction"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
