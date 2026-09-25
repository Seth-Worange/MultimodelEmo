"""Read the untouched 100-row human QA workbook and compare it with automatic QA.

The workbook is evaluation evidence only. Its labels never enter QA routing or
feature extraction.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .data_loader import SampleRecord, canonical_id, load_samples, make_sample_id
from .io_utils import sha256_file, write_csv, write_json


MANUAL_FIELDS = (
    "sample_id", "video_id", "clip_id", "manual_speech_present",
    "manual_language", "manual_official_text_present", "manual_extra_speech",
    "manual_face_present", "manual_static_or_nonhuman",
    "manual_correspondence_class", "reviewer_id", "source_sheet", "source_row",
)
CLASSES = (
    "MATCHED", "PARTIAL_MATCH", "TEXT_AUDIO_MISMATCH", "NO_SPEECH",
    "NON_ENGLISH_SPEECH", "NO_AUDIO", "UNRESOLVED",
)
DISAGREEMENT_FIELDS = (
    "sample_id", "manual_correspondence_class", "automatic_correspondence_status",
    "automatic_route", "exact_class_agreement", "safe_route_agreement",
    "manual_extra_speech", "round4_extra_before", "round4_extra_after",
    "manual_face_present", "round4_face_present", "reviewer_id",
)


def _header(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _category(value: Any, allowed: set[str], field: str, *, allow_blank: bool = False) -> str | None:
    if value is None or str(value).strip() == "":
        if allow_blank:
            return None
        raise ValueError(f"Missing {field}")
    normalized = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    if normalized not in allowed:
        raise ValueError(f"Invalid {field}: {value!r}")
    return normalized


def _boolean(value: Any, field: str) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "yes", "y"}:
        return True
    if normalized in {"0", "0.0", "false", "no", "n"}:
        return False
    if normalized in {"uncertain", "unknown", "not_sure"}:
        return None
    raise ValueError(f"Invalid {field}: {value!r}")


def normalize_rows(
    workbook_rows: Sequence[Sequence[Any]], official: Sequence[SampleRecord],
    *, source_sheet: str = "Sheet1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not workbook_rows:
        raise ValueError("Manual QA workbook is empty")
    header = [_header(cell) for cell in workbook_rows[0]]
    if len(header) != len(set(header)):
        raise ValueError("Duplicate manual QA column names after normalization")
    required = set(MANUAL_FIELDS[3:11])
    missing_columns = sorted(required - set(header))
    if missing_columns:
        raise ValueError(f"Missing manual QA columns: {missing_columns}")
    if "sample_id" not in header and not {"video_id", "clip_id"}.issubset(header):
        raise ValueError("Manual QA needs sample_id or video_id + clip_id")
    pos = {name: header.index(name) for name in header}
    by_id = {record.sample_id: record for record in official}
    if len(by_id) != 100 or len(official) != 100:
        raise ValueError("Official sample list is not exactly 100 unique rows")
    normalized: list[dict[str, Any]] = []
    for excel_row, values in enumerate(workbook_rows[1:], 2):
        if not any(cell is not None and str(cell).strip() for cell in values):
            continue
        cell = lambda name: values[pos[name]] if name in pos and pos[name] < len(values) else None
        video_id, clip_id = canonical_id(cell("video_id")), canonical_id(cell("clip_id"))
        given_id = canonical_id(cell("sample_id"))
        composite = make_sample_id(video_id, clip_id) if video_id and clip_id else ""
        if given_id and composite and given_id != composite:
            raise ValueError(f"sample_id/video_id/clip_id disagree at row {excel_row}")
        sample_id = given_id or composite
        if not sample_id:
            raise ValueError(f"Missing sample ID at row {excel_row}")
        if sample_id in by_id:
            record = by_id[sample_id]
            if "text" in pos and cell("text") is not None and str(cell("text")) != record.text:
                raise ValueError(f"Official text mismatch at row {excel_row}: {sample_id}")
            video_id, clip_id = record.video_id, record.clip_id
        reviewer = canonical_id(cell("reviewer_id"))
        normalized.append({
            "sample_id": sample_id, "video_id": video_id, "clip_id": clip_id,
            "manual_speech_present": _boolean(cell("manual_speech_present"), "manual_speech_present"),
            "manual_language": _category(cell("manual_language"), {"EN", "NON_EN", "NONE", "UNCERTAIN"}, "manual_language"),
            "manual_official_text_present": _category(cell("manual_official_text_present"), {"FULL", "PARTIAL", "NONE", "UNCERTAIN"}, "manual_official_text_present", allow_blank=True),
            "manual_extra_speech": _category(cell("manual_extra_speech"), {"NONE", "BEFORE", "AFTER", "BOTH", "UNCERTAIN"}, "manual_extra_speech", allow_blank=True),
            "manual_face_present": _boolean(cell("manual_face_present"), "manual_face_present"),
            "manual_static_or_nonhuman": _boolean(cell("manual_static_or_nonhuman"), "manual_static_or_nonhuman"),
            "manual_correspondence_class": _category(cell("manual_correspondence_class"), set(CLASSES), "manual_correspondence_class"),
            "reviewer_id": reviewer, "source_sheet": source_sheet, "source_row": excel_row,
        })
    counts = Counter(row["sample_id"] for row in normalized)
    expected, actual = set(by_id), set(counts)
    report = {
        "row_count": len(normalized), "unique_sample_count": len(counts),
        "matched_official_sample_count": len(expected & actual),
        "missing_sample_ids": sorted(expected - actual),
        "extra_sample_ids": sorted(actual - expected),
        "duplicate_sample_ids": sorted(key for key, count in counts.items() if count > 1),
        "reviewer_id_blank_count": sum(not row["reviewer_id"] for row in normalized),
        "manual_official_text_present_blank_count": sum(row["manual_official_text_present"] is None for row in normalized),
        "manual_extra_speech_blank_count": sum(row["manual_extra_speech"] is None for row in normalized),
        "normalization_rules": {
            "identifiers": "strip; integral Excel numeric IDs become decimal integer strings",
            "categories": "strip, uppercase, spaces/hyphens to underscores; unknown values rejected; genuinely blank optional fields remain null",
            "booleans": "Excel boolean, 1/0 numeric and true/false/yes/no strings accepted; uncertain -> null",
            "reviewer_id": "string conversion and strip; blank remains blank",
            "raw_file": "read_only; never modified",
        },
    }
    return normalized, report


def import_manual_qa(path: Path, official: Sequence[SampleRecord], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from openpyxl import load_workbook

    if not path.is_file():
        raise FileNotFoundError(path)
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = wb.active
        rows, report = normalize_rows(list(sheet.values), official, source_sheet=sheet.title)
    finally:
        wb.close()
    report.update({"source_path": str(path.resolve()), "source_sha256": sha256_file(path)})
    target = output_dir / "manual_qa"
    write_json(target / "manual_sample_audit_import_report.json", report)
    if (report["row_count"], report["unique_sample_count"], report["matched_official_sample_count"]) != (100, 100, 100) or any(report[k] for k in ("missing_sample_ids", "extra_sample_ids", "duplicate_sample_ids")):
        raise ValueError("Manual QA does not map 1:1 to the official 100; see import report")
    write_csv(target / "manual_sample_audit_normalized.csv", rows, MANUAL_FIELDS)
    return rows, report


def _read_auto(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate sample IDs in automatic QA")
    return by_id


def _coarse(value: str) -> str:
    return "NO_VALID_SPEECH" if value in {"NO_AUDIO", "NO_SPEECH"} else value


def evaluate_manual_qa(rows: Sequence[dict[str, Any]], auto_by_id: dict[str, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    if set(row["sample_id"] for row in rows) != set(auto_by_id):
        raise ValueError("Manual and automatic QA sample sets differ")
    joint: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    coarse: Counter[tuple[str, str]] = Counter()
    tp = fp = fn = tn = 0
    for manual in rows:
        sample_id = manual["sample_id"]
        auto = auto_by_id[sample_id]
        human = manual["manual_correspondence_class"]
        predicted = str(auto.get("correspondence_status") or "NOT_EVALUATED").strip().upper()
        route = str(auto.get("route") or "NOT_EVALUATED").strip().upper()
        confusion[human, predicted] += 1
        coarse[_coarse(human), _coarse(predicted)] += 1
        safe = human == "MATCHED"
        allowed = route == "HIGH_CONFIDENCE_MATCH"
        if safe and allowed:
            tp += 1
        elif not safe and allowed:
            fp += 1
        elif safe and not allowed:
            fn += 1
        else:
            tn += 1
        joint.append({
            "sample_id": sample_id, "manual_correspondence_class": human,
            "automatic_correspondence_status": predicted, "automatic_route": route,
            "exact_class_agreement": human == predicted,
            "safe_route_agreement": safe == allowed,
            "manual_extra_speech": manual["manual_extra_speech"],
            "round4_extra_before": auto.get("extra_speech_before"),
            "round4_extra_after": auto.get("extra_speech_after"),
            "manual_face_present": manual["manual_face_present"],
            "round4_face_present": auto.get("face_present"),
            "reviewer_id": manual["reviewer_id"],
        })
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    metrics = {
        "sample_count": len(rows),
        "manual_class_counts": dict(Counter(row["manual_correspondence_class"] for row in rows)),
        "automatic_class_counts": dict(Counter(str(auto_by_id[row["sample_id"]].get("correspondence_status") or "NOT_EVALUATED").upper() for row in rows)),
        "exact_class_accuracy": sum(row["exact_class_agreement"] for row in joint) / len(joint) if joint else None,
        "coarse_class_accuracy": sum(_coarse(row["manual_correspondence_class"]) == _coarse(row["automatic_correspondence_status"]) for row in joint) / len(joint) if joint else None,
        "coarse_mapping": {"NO_AUDIO": "NO_VALID_SPEECH", "NO_SPEECH": "NO_VALID_SPEECH"},
        "high_confidence_precision": precision,
        "high_confidence_recall": recall,
        "high_confidence_f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None,
        "unsafe_false_positive_count": fp,
        "conservative_false_negative_count": fn,
        "routing_confusion": {"safe_allowed_tp": tp, "unsafe_allowed_fp": fp, "safe_held_fn": fn, "unsafe_held_tn": tn},
        "evaluation_only_no_training": True,
        "thresholds_unchanged_from_round4": True,
    }
    target = output_dir / "manual_qa"
    write_csv(target / "manual_qa_confusion_matrix.csv", (
        {"manual_class": manual, "automatic_class": automatic, "count": count}
        for (manual, automatic), count in sorted(confusion.items())
    ), ("manual_class", "automatic_class", "count"))
    write_csv(target / "manual_qa_coarse_confusion_matrix.csv", (
        {"manual_coarse_class": manual, "automatic_coarse_class": automatic, "count": count}
        for (manual, automatic), count in sorted(coarse.items())
    ), ("manual_coarse_class", "automatic_coarse_class", "count"))
    write_csv(target / "manual_auto_disagreements.csv", (
        row for row in joint if not row["exact_class_agreement"] or not row["safe_route_agreement"]
    ), DISAGREEMENT_FIELDS)
    write_json(target / "manual_qa_metrics.json", metrics)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-xlsx", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auto-audit", type=Path, help="Existing immutable Round-4/5 correspondence_audit.csv")
    args = parser.parse_args(argv)
    official, _ = load_samples(args.data_root, probe_decode=False)
    rows, report = import_manual_qa(args.manual_xlsx, official, args.output_dir)
    print({key: report[key] for key in ("row_count", "unique_sample_count", "matched_official_sample_count", "reviewer_id_blank_count")})
    if args.auto_audit:
        print(evaluate_manual_qa(rows, _read_auto(args.auto_audit), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
