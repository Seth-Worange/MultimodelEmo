"""Document the measured native MediaPipe geometry and absent 68-map claim."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_json
from .visual_features import BLENDSHAPE_NAMES


def write_landmark_definition(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "landmark68_mapping.json", {
        "mapping_available": False,
        "mapping_name": None,
        "reason": "No publicly documented, validated one-to-one MediaPipe Face Landmarker 478 to iBUG/OpenFace 68 correspondence was identified.",
        "current_backend": "MediaPipe Face Landmarker",
        "current_topology": "native_478",
        "selected_landmark_indices": list(range(478)),
        "normalization": "subtract landmark 1 x/y; divide x/y/z by xy distance between landmarks 33 and 263",
        "geometry_dim": 478 * 3,
        "blendshape_names": list(BLENDSHAPE_NAMES),
        "blendshape_dim": len(BLENDSHAPE_NAMES),
        "frame_feature_dim": 478 * 3 + len(BLENDSHAPE_NAMES),
        "word_feature_dim": 2 * (478 * 3 + len(BLENDSHAPE_NAMES)),
        "not_openface_compatible": True,
        "sources": [
            "https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker",
            "https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python",
            "https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format",
            "https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK",
        ],
    })
    (output_dir / "landmark68_source.md").write_text(
        "# Facial geometry provenance\n\n"
        "The requested MediaPipe-to-iBUG/OpenFace 68-point mapping is **not available** "
        "as an authoritative mapping in the reviewed primary documentation. "
        "The round-4 backend therefore retains all 478 native MediaPipe Face "
        "Landmarker points, not an invented 68-point subset. Its 3-D coordinate "
        "topology and normalization are specified in `landmark68_mapping.json`.\n\n"
        "MediaPipe documents an estimate of 478 3-D face landmarks: "
        "https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker\n\n"
        "OpenFace documents its own 68-landmark output and model: "
        "https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format\n\n"
        "The CMU Multimodal SDK repository was also reviewed: "
        "https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK\n\n"
        "These are different topologies. Round-4 results must not be described "
        "as OpenFace-compatible 68-point features. An actual OpenFace backend "
        "remains unimplemented and untested.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_landmark_definition(args.output_dir)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
