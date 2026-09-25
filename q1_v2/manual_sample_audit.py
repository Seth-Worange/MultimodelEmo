"""Create the non-destructive, 100-sample correspondence review worksheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping

from .data_loader import SampleRecord, load_samples
from .io_utils import write_csv


IDENTITY_FIELDS = ("sample_id", "video_id", "clip_id")
AUTOMATIC_FIELDS = (
    "automatic_correspondence_status",
    "automatic_speech_present",
    "automatic_language",
    "automatic_face_present",
    "automatic_match_score",
)
MANUAL_FIELDS = (
    "manual_speech_present",
    "manual_language",
    "manual_official_text_present",
    "manual_extra_speech",
    "manual_face_present",
    "manual_static_or_nonhuman",
    "manual_correspondence_class",
    "reviewer_id",
    "review_note",
)
FIELDS = IDENTITY_FIELDS + AUTOMATIC_FIELDS + MANUAL_FIELDS


def create_manual_sample_audit(
    records: list[SampleRecord],
    path: Path,
    *,
    automatic_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Write once, or refresh only automatic cells while preserving manual entries.

    A template may be created before the automatic QA exists. Such cells stay
    explicitly unevaluated, never masquerading as a negative finding.
    """
    if len(records) != 100 or len({record.sample_id for record in records}) != 100:
        raise ValueError("Manual audit requires exactly 100 unique official samples")
    previous: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"Existing audit file has incompatible columns: {path}")
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in previous:
                    raise ValueError(f"Duplicate sample in existing audit: {sample_id}")
                previous[sample_id] = row
        if set(previous) != {record.sample_id for record in records}:
            raise ValueError("Existing audit contains a different sample set; not overwriting it")

    rows: list[dict[str, object]] = []
    for record in records:
        old = previous.get(record.sample_id, {})
        row: dict[str, object] = {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "clip_id": record.clip_id,
            "automatic_correspondence_status": old.get("automatic_correspondence_status", "NOT_EVALUATED"),
            **{field: old.get(field, "") for field in AUTOMATIC_FIELDS[1:]},
            **{field: old.get(field, "") for field in MANUAL_FIELDS},
        }
        if automatic_by_id is not None and record.sample_id in automatic_by_id:
            update = automatic_by_id[record.sample_id]
            for field in AUTOMATIC_FIELDS:
                if field in update:
                    row[field] = update[field]
        rows.append(row)
    write_csv(path, rows, FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, summary = load_samples(args.data_root, args.labels, probe_decode=False)
    if summary["invalid_sample_count"] or summary["extra_videos"]:
        raise ValueError(f"Input sample audit failed: {summary}")
    output = args.output_dir / "manual_sample_audit_template.csv"
    create_manual_sample_audit(records, output)
    print(f"Created or refreshed {output.resolve()} with {len(records)} official samples")


if __name__ == "__main__":
    main()
