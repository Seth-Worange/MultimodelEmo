"""Audit MFA–ASR boundary disagreement without treating ASR as ground truth."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from .alignment import normalize_word
from .config import Q1Config
from .data_loader import load_samples
from .io_utils import read_json, write_csv, write_json


FIELDS = (
    "sample_id", "word_index", "original_word", "mfa_start_s", "mfa_end_s",
    "asr_start_s", "asr_end_s", "asr_word_indices", "asr_mapping_status",
    "start_disagreement_s", "end_disagreement_s", "max_disagreement_s",
    "alignment_confidence", "manual_review_required", "human_max_boundary_error_s",
)


def _tokens(word: str) -> list[str]:
    from .correspondence_qa import _tokens as qa_tokens

    return qa_tokens(word)


def map_official_to_asr(
    official_words: Sequence[str], asr_words: Sequence[dict[str, Any]],
    *, first_asr_index: int, last_asr_index: int,
) -> list[dict[str, Any]]:
    """Monotone exact-token subsequence map; unmatched words remain unavailable."""
    flat_official: list[str] = []
    official_owner: list[int] = []
    for index, word in enumerate(official_words):
        for token in _tokens(word):
            flat_official.append(token)
            official_owner.append(index)
    flat_asr: list[str] = []
    asr_owner: list[int] = []
    for index in range(first_asr_index, last_asr_index + 1):
        for token in _tokens(str(asr_words[index]["word"])):
            flat_asr.append(token)
            asr_owner.append(index)
    matched: dict[int, int] = {}
    matcher = SequenceMatcher(None, flat_official, flat_asr, autojunk=False)
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            matched[block.a + offset] = asr_owner[block.b + offset]
    by_word: dict[int, list[int]] = defaultdict(list)
    token_count = Counter(official_owner)
    for token_index, asr_index in matched.items():
        by_word[official_owner[token_index]].append(asr_index)
    output = []
    for index in range(len(official_words)):
        indices = by_word.get(index, [])
        if not token_count[index] or len(indices) != token_count[index]:
            output.append({"status": "UNMATCHED_ASR_TOKEN", "indices": []})
        else:
            output.append({"status": "EXACT_MONOTONE_TOKEN_MATCH", "indices": sorted(set(indices))})
    return output


def confidence(disagreement_s: float | None, config: Q1Config) -> str:
    if disagreement_s is None:
        return "UNAVAILABLE"
    if disagreement_s <= config.asr_mfa_high_max_disagreement_s:
        return "HIGH"
    if disagreement_s <= config.asr_mfa_medium_max_disagreement_s:
        return "MEDIUM"
    return "LOW"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_one(sample_id: str, sample_dir: Path, config: Q1Config, human_errors: dict[tuple[str, int], float]) -> list[dict[str, Any]]:
    alignment = _csv_rows(sample_dir / "word_alignment.csv")
    asr = _csv_rows(sample_dir / "asr_words.csv")
    qa = read_json(sample_dir / "correspondence_qa.json")
    original = [row["original_word"] for row in alignment]
    first = qa["local_match"].get("matched_asr_word_start_index")
    last = qa["local_match"].get("matched_asr_word_end_index")
    mapping = map_official_to_asr(original, asr, first_asr_index=int(first), last_asr_index=int(last)) if first is not None and last is not None else [{"status": "NO_MATCH_SPAN", "indices": []} for _ in original]
    rows = []
    for index, (word, mapped) in enumerate(zip(alignment, mapping)):
        if int(word["word_index"]) != index or word["sample_id"] != sample_id:
            raise ValueError(f"Alignment word sequence mismatch in {sample_id}")
        if normalize_word(word["original_word"]) != normalize_word(original[index]):
            raise ValueError(f"Official word mismatch at {sample_id}:{index}")
        indices = mapped["indices"]
        mfa_valid = word["alignment_mask"] == "1"
        mfa_start = float(word["start_s"]) if mfa_valid else None
        mfa_end = float(word["end_s"]) if mfa_valid else None
        asr_start = min(float(asr[i]["start_s"]) for i in indices) if indices else None
        asr_end = max(float(asr[i]["end_s"]) for i in indices) if indices else None
        ds = abs(mfa_start - asr_start) if mfa_start is not None and asr_start is not None else None
        de = abs(mfa_end - asr_end) if mfa_end is not None and asr_end is not None else None
        maximum = max(ds, de) if ds is not None and de is not None else None
        grade = confidence(maximum, config)
        rows.append({
            "sample_id": sample_id, "word_index": index,
            "original_word": word["original_word"],
            "mfa_start_s": mfa_start, "mfa_end_s": mfa_end,
            "asr_start_s": asr_start, "asr_end_s": asr_end,
            "asr_word_indices": "|".join(str(i) for i in indices),
            "asr_mapping_status": mapped["status"],
            "start_disagreement_s": ds, "end_disagreement_s": de,
            "max_disagreement_s": maximum,
            "alignment_confidence": grade,
            "manual_review_required": grade in {"LOW", "UNAVAILABLE"},
            "human_max_boundary_error_s": human_errors.get((sample_id, index)),
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--round4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-detail-csv", type=Path)
    args = parser.parse_args(argv)
    official, _ = load_samples(args.data_root, probe_decode=False)
    human_errors: dict[tuple[str, int], float] = {}
    if args.manual_detail_csv and args.manual_detail_csv.is_file():
        for row in _csv_rows(args.manual_detail_csv):
            if row["comparison_status"] == "COMPARED" and row.get("max_boundary_error_s"):
                human_errors[row["sample_id"], int(row["word_index"])] = float(row["max_boundary_error_s"])
    config = Q1Config()
    rows = []
    unavailable_samples = []
    for record in official:
        sample_dir = args.round4_root / "samples" / record.sample_id
        if ((sample_dir / "mfa_raw.json").is_file()
                and (sample_dir / "word_alignment.csv").is_file()
                and (sample_dir / "asr_words.csv").is_file()):
            rows.extend(audit_one(record.sample_id, sample_dir, config, human_errors))
        else:
            unavailable_samples.append(record.sample_id)
    target = args.output_dir / "alignment_confidence"
    write_csv(target / "word_alignment_confidence.csv", rows, FIELDS)
    counts = Counter(row["alignment_confidence"] for row in rows)
    human_rows = [row for row in rows if row["human_max_boundary_error_s"] is not None]
    large_human = [row for row in human_rows if row["human_max_boundary_error_s"] > .20]
    summary = {
        "eligible_real_mfa_sample_count": len({row["sample_id"] for row in rows}),
        "audited_word_count": len(rows),
        "confidence_counts": dict(counts),
        "unavailable_sample_count": len(unavailable_samples),
        "unavailable_sample_ids": unavailable_samples,
        "high_max_disagreement_s": config.asr_mfa_high_max_disagreement_s,
        "medium_max_disagreement_s": config.asr_mfa_medium_max_disagreement_s,
        "manual_word_overlap_count": len(human_rows),
        "manual_large_error_over_200ms_count": len(large_human),
        "manual_large_error_flagged_low_or_unavailable_count": sum(row["alignment_confidence"] in {"LOW", "UNAVAILABLE"} for row in large_human),
        "manual_large_error_word_indices": [{"sample_id": row["sample_id"], "word_index": row["word_index"], "confidence": row["alignment_confidence"]} for row in large_human],
        "meaning": "MFA-ASR disagreement is an uncertainty signal, not a measured MFA error or time-alignment accuracy",
    }
    write_json(target / "alignment_confidence_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
