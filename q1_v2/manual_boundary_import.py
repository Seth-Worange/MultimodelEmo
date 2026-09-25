"""Validate untouched check5-style human boundaries and evaluate available MFA.

Missing endpoints are retained as incomplete reference, never inferred from the
next word. Numerical alignment scores are restricted to complete reference
intervals and actual MFA intervals for the same official word position.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from .alignment import normalize_word
from .data_loader import SampleRecord, canonical_id, load_samples
from .io_utils import read_json, sha256_file, write_csv, write_json


REFERENCE_FIELDS = (
    "sample_id", "word_index", "original_word", "reference_start_s",
    "reference_end_s", "reviewer_id", "boundary_status", "word_mapping_status",
    "official_word_at_index", "source_sheet", "source_row",
)
DETAIL_FIELDS = (
    "sample_id", "word_index", "original_word", "reference_start_s", "reference_end_s",
    "reference_status", "round5_alignment_source", "round5_mfa_start_s", "round5_mfa_end_s",
    "round4_mfa_start_s", "round4_mfa_end_s", "start_error_s", "end_error_s",
    "center_error_s", "duration_error_s", "max_boundary_error_s", "temporal_iou",
    "round4_start_error_s", "round4_end_error_s", "round4_max_boundary_error_s",
    "round4_temporal_iou", "comparison_status",
)
BY_SAMPLE_FIELDS = (
    "sample_id", "official_word_count", "manual_row_count", "reference_word_count",
    "manual_reference_coverage", "compared_word_count", "coverage",
    "start_mae_s", "end_mae_s", "mean_boundary_error_s", "median_boundary_error_s",
    "p90_boundary_error_s", "max_boundary_error_s", "mean_temporal_iou",
    "mean_center_error_s", "mean_duration_error_s",
    "pass_rate_50ms", "pass_rate_100ms", "pass_rate_200ms", "alignment_source",
)


def _float_or_none(value: Any, field: str, source_row: int) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Boolean {field} at row {source_row}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {field} at row {source_row}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} at row {source_row}")
    return result


def _duration(record: SampleRecord, round4_root: Path | None) -> float:
    if round4_root:
        path = round4_root / "samples" / record.sample_id / "correspondence_qa.json"
        if path.is_file():
            qa = read_json(path)
            if qa.get("source_video_sha256") == sha256_file(record.video_path):
                return float(qa["duration_s"])
    import av

    with av.open(str(record.video_path)) as container:
        if container.duration is None:
            raise ValueError(f"No usable media duration: {record.sample_id}")
        return float(container.duration / 1_000_000)


def normalize_boundaries(
    workbook_rows: Sequence[Sequence[Any]], official: Sequence[SampleRecord],
    duration_by_id: dict[str, float], *, source_sheet: str = "Sheet1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not workbook_rows:
        raise ValueError("Empty manual boundary workbook")
    header = [str(cell or "").strip().lower().replace(" ", "_") for cell in workbook_rows[0]]
    required = {"sample_id", "word_index", "reference_start_s", "reference_end_s"}
    if not required.issubset(header):
        raise ValueError(f"Missing boundary columns: {sorted(required - set(header))}")
    word_header = "original_words" if "original_words" in header else "original_word" if "original_word" in header else None
    if word_header is None:
        raise ValueError("Missing original words/original_words column")
    if len(header) != len(set(header)):
        raise ValueError("Duplicate boundary headers")
    pos = {name: header.index(name) for name in header}
    by_id = {record.sample_id: record for record in official}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    last_start: dict[str, float] = {}
    last_end: dict[str, float] = {}
    for source_row, values in enumerate(workbook_rows[1:], 2):
        if not any(cell is not None and str(cell).strip() for cell in values):
            continue
        cell = lambda name: values[pos[name]] if name in pos and pos[name] < len(values) else None
        sample_id = canonical_id(cell("sample_id"))
        if sample_id not in by_id:
            raise ValueError(f"Unknown official sample_id at row {source_row}: {sample_id}")
        raw_index = cell("word_index")
        if isinstance(raw_index, bool) or raw_index is None or str(raw_index).strip() == "":
            raise ValueError(f"Missing/invalid word_index at row {source_row}")
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid word_index at row {source_row}: {raw_index!r}") from exc
        if str(raw_index).strip() not in {str(index), f"{index}.0"}:
            raise ValueError(f"Non-integral word_index at row {source_row}: {raw_index!r}")
        record = by_id[sample_id]
        if (sample_id, index) in seen:
            raise ValueError(f"Duplicate reference word_index for {sample_id}:{index}")
        seen.add((sample_id, index))
        original = canonical_id(cell(word_header))
        official_word = record.original_words[index].text if 0 <= index < record.word_count else ""
        mapping_status = (
            "WORD_INDEX_OUT_OF_RANGE" if not official_word else
            "WORD_MISMATCH" if normalize_word(original) != normalize_word(official_word) else
            "MATCHED_OFFICIAL_WORD"
        )
        start = _float_or_none(cell("reference_start_s"), "reference_start_s", source_row)
        end = _float_or_none(cell("reference_end_s"), "reference_end_s", source_row)
        duration = duration_by_id[sample_id]
        for field, value in (("start", start), ("end", end)):
            if value is not None and not 0 <= value <= duration + .05:
                raise ValueError(f"{field} boundary outside media duration at {sample_id}:{index}: {value} > {duration}")
        if start is not None and end is not None and not start < end:
            raise ValueError(f"Invalid start/end order at {sample_id}:{index}")
        if start is not None:
            if sample_id in last_start and start < last_start[sample_id] - 1e-6:
                raise ValueError(f"Non-monotonic reference start at {sample_id}:{index}")
            if sample_id in last_end and start < last_end[sample_id] - .05:
                raise ValueError(f"Overlapping/non-monotonic reference boundary at {sample_id}:{index}")
            last_start[sample_id] = start
        if end is not None:
            if sample_id in last_end and end < last_end[sample_id] - 1e-6:
                raise ValueError(f"Non-monotonic reference end at {sample_id}:{index}")
            last_end[sample_id] = end
        status = (
            mapping_status if mapping_status != "MATCHED_OFFICIAL_WORD" else
            "VALID" if start is not None and end is not None else
            "MISSING_END" if end is None and start is not None else
            "MISSING_START" if start is None and end is not None else "MISSING_BOTH"
        )
        rows.append({
            "sample_id": sample_id, "word_index": index,
            "original_word": original,
            "reference_start_s": start, "reference_end_s": end,
            "reviewer_id": canonical_id(cell("reviewer_id")),
            "boundary_status": status, "word_mapping_status": mapping_status,
            "official_word_at_index": official_word,
            "source_sheet": source_sheet,
            "source_row": source_row,
        })
    by_sample: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row["word_index"])
    for sample_id, indices in by_sample.items():
        if indices != sorted(indices):
            raise ValueError(f"Manual rows are out of word order: {sample_id}")
    report = {
        "row_count": len(rows), "distinct_sample_count": len(by_sample),
        "sample_ids": sorted(by_sample),
        "valid_interval_count": sum(row["boundary_status"] == "VALID" for row in rows),
        "incomplete_interval_count": sum(row["boundary_status"] != "VALID" for row in rows),
        "missing_start_count": sum(row["reference_start_s"] is None for row in rows),
        "missing_end_count": sum(row["reference_end_s"] is None for row in rows),
        "word_mapping_error_count": sum(row["word_mapping_status"] != "MATCHED_OFFICIAL_WORD" for row in rows),
        "word_mapping_errors": [
            {"sample_id": row["sample_id"], "word_index": row["word_index"],
             "source_row": row["source_row"], "original_word": row["original_word"],
             "official_word_at_index": row["official_word_at_index"],
             "issue": row["word_mapping_status"]}
            for row in rows if row["word_mapping_status"] != "MATCHED_OFFICIAL_WORD"
        ],
        "sample_ids_not_starting_at_zero": sorted(sample_id for sample_id, indices in by_sample.items() if min(indices) != 0),
        "incomplete_by_sample": {
            sample_id: sum(row["sample_id"] == sample_id and row["boundary_status"] != "VALID" for row in rows)
            for sample_id in sorted(by_sample)
        },
        "reviewer_id_blank_count": sum(not row["reviewer_id"] for row in rows),
        "official_word_count_by_sample": {sample_id: by_id[sample_id].word_count for sample_id in sorted(by_sample)},
        "manual_row_coverage_by_sample": {sample_id: len(indices) / by_id[sample_id].word_count for sample_id, indices in sorted(by_sample.items())},
        "normalization_rules": "sample_id strip, original word checked against official sequence, original words/original_words accepted; invalid word indices and incomplete endpoints retained and flagged, never silently remapped or inferred",
        "source_file_modified": False,
    }
    return rows, report


def import_boundaries(path: Path, official: Sequence[SampleRecord], output_dir: Path, round4_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from openpyxl import load_workbook

    if not path.is_file():
        raise FileNotFoundError(path)
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = wb.active
        raw = list(sheet.values)
    finally:
        wb.close()
    header = [str(cell or "").strip().lower().replace(" ", "_") for cell in raw[0]]
    id_index = header.index("sample_id") if "sample_id" in header else None
    if id_index is None:
        raise ValueError("Missing sample_id column")
    by_id = {record.sample_id: record for record in official}
    listed = {canonical_id(row[id_index]) for row in raw[1:] if any(cell is not None for cell in row)}
    for sample_id in listed:
        if sample_id not in by_id:
            raise ValueError(f"Unknown official sample_id: {sample_id}")
    durations = {sample_id: _duration(by_id[sample_id], round4_root) for sample_id in listed}
    rows, report = normalize_boundaries(raw, official, durations, source_sheet=sheet.title)
    report.update({"source_path": str(path.resolve()), "source_sha256": sha256_file(path), "media_duration_s_by_sample": durations})
    target = output_dir / "manual_boundary"
    write_csv(target / "normalized_manual_boundaries.csv", rows, REFERENCE_FIELDS)
    write_json(target / "import_report.json", report)
    return rows, report


def _read_alignment(path: Path | None, sample_id: str) -> dict[int, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        if row["sample_id"] != sample_id:
            raise ValueError(f"Alignment sample ID mismatch in {path}")
        index = int(row["word_index"])
        if index in by_index:
            raise ValueError(f"Duplicate alignment index in {path}: {index}")
        by_index[index] = row
    return by_index


def _interval(row: dict[str, str] | None, official_word: str) -> tuple[float, float] | None:
    if not row or row["alignment_mask"] != "1":
        return None
    if normalize_word(row["original_word"]) != normalize_word(official_word):
        raise ValueError("Alignment word does not match official word")
    start, end = float(row["start_s"]), float(row["end_s"])
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise ValueError("Invalid MFA interval")
    return start, end


def _p90(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = .9 * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _metrics(detail: Sequence[dict[str, Any]], *, official_word_count: int, source: str) -> dict[str, Any]:
    valid_reference = [row for row in detail if row["reference_status"] == "VALID"]
    compared = [row for row in valid_reference if row["comparison_status"] == "COMPARED"]
    starts = [row["start_error_s"] for row in compared]
    ends = [row["end_error_s"] for row in compared]
    boundaries = starts + ends
    max_errors = [row["max_boundary_error_s"] for row in compared]
    result = {
        "official_word_count": official_word_count,
        "manual_row_count": len(detail),
        "reference_word_count": len(valid_reference),
        "manual_reference_coverage": len(valid_reference) / official_word_count if official_word_count else None,
        "compared_word_count": len(compared),
        "coverage": len(compared) / len(valid_reference) if valid_reference else None,
        "start_mae_s": statistics.mean(starts) if starts else None,
        "end_mae_s": statistics.mean(ends) if ends else None,
        "mean_boundary_error_s": statistics.mean(boundaries) if boundaries else None,
        "median_boundary_error_s": statistics.median(boundaries) if boundaries else None,
        "p90_boundary_error_s": _p90(boundaries),
        "max_boundary_error_s": max(boundaries) if boundaries else None,
        "mean_temporal_iou": statistics.mean(row["temporal_iou"] for row in compared) if compared else None,
        "mean_center_error_s": statistics.mean(row["center_error_s"] for row in compared) if compared else None,
        "mean_duration_error_s": statistics.mean(row["duration_error_s"] for row in compared) if compared else None,
        "pass_rate_50ms": sum(error <= .05 for error in max_errors) / len(compared) if compared else None,
        "pass_rate_100ms": sum(error <= .10 for error in max_errors) / len(compared) if compared else None,
        "pass_rate_200ms": sum(error <= .20 for error in max_errors) / len(compared) if compared else None,
        "alignment_source": source,
    }
    return result


def evaluate_boundaries(
    reference: Sequence[dict[str, Any]], official: Sequence[SampleRecord],
    output_dir: Path, round4_root: Path | None,
) -> dict[str, Any]:
    by_id = {record.sample_id: record for record in official}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference:
        grouped[row["sample_id"]].append(row)
    details: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for sample_id, rows in grouped.items():
        new_path = output_dir / "samples" / sample_id / "word_alignment.csv"
        old_path = round4_root / "samples" / sample_id / "word_alignment.csv" if round4_root else None
        source = "round5_local_mfa" if new_path.is_file() else "unchanged_round4_real_mfa" if old_path and old_path.is_file() else "NO_ALIGNMENT"
        new = _read_alignment(new_path if new_path.is_file() else old_path, sample_id)
        old = _read_alignment(old_path, sample_id)
        sample_details: list[dict[str, Any]] = []
        for row in rows:
            index = row["word_index"]
            mapping_valid = row["word_mapping_status"] == "MATCHED_OFFICIAL_WORD"
            auto = _interval(new.get(index), row["original_word"]) if mapping_valid else None
            previous = _interval(old.get(index), row["original_word"]) if mapping_valid else None
            start_ref, end_ref = row["reference_start_s"], row["reference_end_s"]
            detail: dict[str, Any] = {
                "sample_id": sample_id, "word_index": index,
                "original_word": row["original_word"],
                "reference_start_s": start_ref, "reference_end_s": end_ref,
                "reference_status": row["boundary_status"],
                "round5_alignment_source": source,
                "round5_mfa_start_s": auto[0] if auto else None,
                "round5_mfa_end_s": auto[1] if auto else None,
                "round4_mfa_start_s": previous[0] if previous else None,
                "round4_mfa_end_s": previous[1] if previous else None,
                "comparison_status": row["boundary_status"] if row["boundary_status"] != "VALID" else "NO_MFA_ALIGNMENT" if auto is None else "COMPARED",
            }
            if detail["comparison_status"] == "COMPARED":
                a_start, a_end = auto
                detail.update({
                    "start_error_s": abs(a_start - start_ref),
                    "end_error_s": abs(a_end - end_ref),
                    "center_error_s": abs((a_start + a_end - start_ref - end_ref) / 2),
                    "duration_error_s": abs((a_end - a_start) - (end_ref - start_ref)),
                    "max_boundary_error_s": max(abs(a_start - start_ref), abs(a_end - end_ref)),
                    "temporal_iou": max(0, min(a_end, end_ref) - max(a_start, start_ref)) / (max(a_end, end_ref) - min(a_start, start_ref)),
                })
            if row["boundary_status"] == "VALID" and previous:
                p_start, p_end = previous
                detail.update({
                    "round4_start_error_s": abs(p_start - start_ref),
                    "round4_end_error_s": abs(p_end - end_ref),
                    "round4_max_boundary_error_s": max(abs(p_start - start_ref), abs(p_end - end_ref)),
                    "round4_temporal_iou": max(0, min(p_end, end_ref) - max(p_start, start_ref)) / (max(p_end, end_ref) - min(p_start, start_ref)),
                })
            sample_details.append(detail)
            details.append(detail)
        metrics = _metrics(sample_details, official_word_count=by_id[sample_id].word_count, source=source)
        per_sample.append({"sample_id": sample_id, **metrics})
        comparable = [row for row in sample_details if row["comparison_status"] == "COMPARED" and row.get("round4_max_boundary_error_s") is not None]
        comparison_rows.append({
            "sample_id": sample_id, "compared_word_count": len(comparable),
            "round4_mean_boundary_error_s": statistics.mean([x for row in comparable for x in (row["round4_start_error_s"], row["round4_end_error_s"])]) if comparable else None,
            "round5_mean_boundary_error_s": statistics.mean([x for row in comparable for x in (row["start_error_s"], row["end_error_s"])]) if comparable else None,
            "round4_mean_iou": statistics.mean(row["round4_temporal_iou"] for row in comparable) if comparable else None,
            "round5_mean_iou": statistics.mean(row["temporal_iou"] for row in comparable) if comparable else None,
            "numeric_output_unchanged": all(row["round5_mfa_start_s"] == row["round4_mfa_start_s"] and row["round5_mfa_end_s"] == row["round4_mfa_end_s"] for row in comparable) if comparable else None,
        })
    target = output_dir / "manual_boundary"
    write_csv(target / "manual_alignment_detail.csv", details, DETAIL_FIELDS)
    write_csv(target / "manual_boundary_metrics_by_sample.csv", per_sample, BY_SAMPLE_FIELDS)
    write_csv(target / "round4_round5_alignment_comparison.csv", comparison_rows, (
        "sample_id", "compared_word_count", "round4_mean_boundary_error_s", "round5_mean_boundary_error_s",
        "round4_mean_iou", "round5_mean_iou", "numeric_output_unchanged",
    ))
    valid_sample_metrics = [row for row in per_sample if row["compared_word_count"]]
    compared_all = [row for row in details if row["comparison_status"] == "COMPARED"]
    macro_keys = (
        "start_mae_s", "end_mae_s", "mean_boundary_error_s", "median_boundary_error_s",
        "p90_boundary_error_s", "max_boundary_error_s", "mean_temporal_iou",
        "pass_rate_50ms", "pass_rate_100ms", "pass_rate_200ms",
    )
    macro = {key: statistics.mean(row[key] for row in valid_sample_metrics) if valid_sample_metrics else None for key in macro_keys}
    micro = _metrics(details, official_word_count=sum(by_id[sample_id].word_count for sample_id in grouped), source="pooled_actual_mfa")
    overall = {
        "manual_sample_count": len(grouped), "manual_row_count": len(reference),
        "complete_reference_word_count": sum(row["boundary_status"] == "VALID" for row in reference),
        "incomplete_reference_word_count": sum(row["boundary_status"] != "VALID" for row in reference),
        "evaluated_sample_count": len(valid_sample_metrics),
        "compared_word_count": len(compared_all),
        "macro_by_sample": macro, "micro_by_word_boundary": micro,
        "pass_rule": "both boundaries must be within tolerance",
        "round5_alignment_algorithm_changed": False,
        "round5_only_adds_confidence_auditing": True,
        "note": "Reused Round-4 real MFA numerical intervals are not reported as a Round-5 improvement. Incomplete human intervals are excluded, not imputed.",
        "inter_rater_agreement": "NOT_EVALUATED",
    }
    write_json(target / "manual_boundary_metrics_overall.json", overall)
    return overall


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-xlsx", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path)
    args = parser.parse_args(argv)
    official, _ = load_samples(args.data_root, probe_decode=False)
    reference, report = import_boundaries(args.manual_xlsx, official, args.output_dir, args.round4_root)
    print({key: report[key] for key in ("row_count", "distinct_sample_count", "valid_interval_count", "incomplete_interval_count")})
    print(evaluate_boundaries(reference, official, args.output_dir, args.round4_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
