"""Second decoding audit only for stage-1 REVIEW_REQUIRED samples.

The independent human 100-sample audit is read only after all routing decisions.
No MFA is launched from this module, and stage-1 files remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from .correspondence_qa import ASRWord, QAConfig, decide_correspondence, local_word_match
from .data_loader import load_samples
from .io_utils import read_json, write_csv, write_json


FIELDS = (
    "sample_id", "stage1_status", "stage1_route", "stage1_match_start_s", "stage1_match_end_s",
    "stage2_status", "stage2_route", "stage2_match_start_s", "stage2_match_end_s",
    "span_agreement_start_s", "span_agreement_end_s", "span_agreement_max_s",
    "stage1_token_recall", "stage2_token_recall", "stage2_language_probability",
    "promotion_status", "promotion_reason", "manual_correspondence_class",
)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def second_stage_decision(stage1: dict[str, Any], stage2_status: str,
                          stage2_route: str, stage2_match: dict[str, Any],
                          *, max_span_difference_s: float = .20,
                          min_stage1_recall: float = .65) -> tuple[bool, str, float | None, float | None, float | None]:
    """Conservative promotion gate, defined without reference to human labels."""
    if stage1.get("route") != "REVIEW_REQUIRED":
        return False, "stage1_not_review_required", None, None, None
    first_start = _number(stage1.get("matched_audio_start_s"))
    first_end = _number(stage1.get("matched_audio_end_s"))
    second_start = _number(stage2_match.get("matched_audio_start_s"))
    second_end = _number(stage2_match.get("matched_audio_end_s"))
    if None in (first_start, first_end, second_start, second_end):
        return False, "matched_span_missing", None, None, None
    ds, de = abs(first_start - second_start), abs(first_end - second_end)
    difference = max(ds, de)
    if stage1.get("correspondence_status") != "PARTIAL_MATCH":
        return False, "stage1_not_partial_match", ds, de, difference
    if float(stage1.get("token_recall", 0)) < min_stage1_recall:
        return False, "stage1_evidence_too_weak", ds, de, difference
    if stage1.get("local_match_ambiguous"):
        return False, "stage1_local_span_ambiguous", ds, de, difference
    if stage2_status != "MATCHED" or stage2_route != "HIGH_CONFIDENCE_MATCH":
        return False, "stage2_not_high_confidence_match", ds, de, difference
    if difference > max_span_difference_s:
        return False, "two_decodes_disagree_on_span", ds, de, difference
    return True, "two_decoding_settings_support_same_local_interval", ds, de, difference


def run(*, data_root: Path, round4_root: Path, output_dir: Path,
        asr_python: Path, asr_model: Path, manual_csv: Path,
        beam_size: int = 10) -> dict[str, Any]:
    if beam_size <= 5:
        raise ValueError("Stage-2 beam must exceed prespecified stage-1 beam=5")
    records, _ = load_samples(data_root, probe_decode=False)
    by_id = {record.sample_id: record for record in records}
    stage1_rows = _csv(round4_root / "correspondence_audit.csv")
    manual = {row["sample_id"]: row["manual_correspondence_class"] for row in _csv(manual_csv)}
    candidates = [row for row in stage1_rows if row["route"] == "REVIEW_REQUIRED"]
    output_rows = []
    for first in candidates:
        sample_id = first["sample_id"]
        sample_source = round4_root / "samples" / sample_id
        target = output_dir / "samples" / sample_id / "stage2_qa"
        target.mkdir(parents=True, exist_ok=True)
        raw_path = target / "asr_beam10_raw.json"
        command = [str(asr_python.resolve()), "-m", "q1_v2.asr_worker",
                   "--wav-path", str((sample_source / "audio_mfa_16k_mono.wav").resolve()),
                   "--model", str(asr_model.resolve()), "--output-json", str(raw_path.resolve()),
                   "--beam-size", str(beam_size)]
        write_json(target / "asr_beam10_command.json", {"argv": command, "source_stage1": str(sample_source)})
        started = time.perf_counter()
        try:
            if not raw_path.is_file():
                completed = subprocess.run(command, cwd=Path.cwd(), capture_output=True,
                                           text=True, encoding="utf-8", errors="replace", timeout=1200)
                (target / "asr_beam10_stdout.log").write_text(completed.stdout, encoding="utf-8")
                (target / "asr_beam10_stderr.log").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise RuntimeError(f"Stage-2 ASR exit {completed.returncode}")
            raw = read_json(raw_path)
            observed = [ASRWord(**word) for word in raw["words"]]
            match = local_word_match(by_id[sample_id].text, observed)
            status, route, reason, _ = decide_correspondence(
                audio_signal_present=raw["audio_signal_present"],
                speech_present=raw["speech_duration_s"] >= QAConfig().min_speech_s,
                detected_language=raw["detected_language"],
                language_probability=raw["language_probability"],
                match=match, config=QAConfig(),
            )
            stage1_match = read_json(sample_source / "local_match.json")
            gate_input = {**first, "token_recall": stage1_match["token_recall"],
                          "local_match_ambiguous": stage1_match["local_match_ambiguous"]}
            promoted, gate_reason, ds, de, maximum = second_stage_decision(gate_input, status, route, match)
            row = {
                "sample_id": sample_id, "stage1_status": first["correspondence_status"],
                "stage1_route": first["route"],
                "stage1_match_start_s": first["matched_audio_start_s"],
                "stage1_match_end_s": first["matched_audio_end_s"],
                "stage2_status": status, "stage2_route": route,
                "stage2_match_start_s": match["matched_audio_start_s"],
                "stage2_match_end_s": match["matched_audio_end_s"],
                "span_agreement_start_s": ds, "span_agreement_end_s": de,
                "span_agreement_max_s": maximum,
                "stage1_token_recall": stage1_match["token_recall"],
                "stage2_token_recall": match["token_recall"],
                "stage2_language_probability": raw["language_probability"],
                "promotion_status": "HIGH_CONFIDENCE_MATCH_SECOND_STAGE" if promoted else "REVIEW_REQUIRED",
                "promotion_reason": gate_reason,
                "manual_correspondence_class": manual.get(sample_id),
            }
            write_json(target / "stage2_decision.json", {**row, "stage2_match": match,
                "stage2_automatic_reason": reason, "runtime_s": time.perf_counter() - started})
        except Exception as exc:
            row = {"sample_id": sample_id, "stage1_status": first["correspondence_status"],
                   "stage1_route": first["route"], "promotion_status": "REVIEW_REQUIRED",
                   "promotion_reason": f"stage2_failed:{type(exc).__name__}",
                   "manual_correspondence_class": manual.get(sample_id)}
            write_json(target / "_STAGE2_FAILED.json", {"error": str(exc), "traceback": traceback.format_exc()})
        output_rows.append(row)
        print({"sample_id": sample_id, "promotion_status": row["promotion_status"],
               "reason": row["promotion_reason"]}, flush=True)
    out = output_dir / "qa"
    write_csv(out / "stage2_promotions.csv", output_rows, FIELDS)
    promoted_ids = {row["sample_id"] for row in output_rows if row["promotion_status"] == "HIGH_CONFIDENCE_MATCH_SECOND_STAGE"}
    stage1_safe = {row["sample_id"] for row in stage1_rows if row["route"] == "HIGH_CONFIDENCE_MATCH"}
    manual_safe = {sample_id for sample_id, label in manual.items() if label == "MATCHED"}
    def metrics(safe: set[str]) -> dict[str, Any]:
        tp = len(safe & manual_safe)
        fp = len(safe - manual_safe)
        fn = len(manual_safe - safe)
        return {"high_confidence_precision": tp / (tp + fp) if tp + fp else None,
                "high_confidence_recall": tp / (tp + fn) if tp + fn else None,
                "unsafe_false_positive_count": fp, "conservative_false_negative_count": fn,
                "released_count": len(safe)}
    result = {
        "candidate_count": len(candidates), "completed_stage2_count": sum("stage2_status" in row for row in output_rows),
        "promotion_count": len(promoted_ids), "beam_size_stage1": 5, "beam_size_stage2": beam_size,
        "stage2_is_independent_model": False,
        "stage1_prespecified": metrics(stage1_safe),
        "stage1_plus_exploratory_stage2_same_100_manual_set": metrics(stage1_safe | promoted_ids),
        "unsafe_new_promotions": sorted(promoted_ids - manual_safe),
        "human_labels_used_in_decision": False,
        "caution": "Stage-2 is a second decode of the same base model, not an independent test-set estimate or permission for MFA by itself",
    }
    write_json(out / "stage2_qa_metrics.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "round4-root", "output-dir", "asr-python", "asr-model", "manual-csv"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args(argv)
    print(run(data_root=args.data_root, round4_root=args.round4_root,
              output_dir=args.output_dir, asr_python=args.asr_python,
              asr_model=args.asr_model, manual_csv=args.manual_csv, beam_size=args.beam_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
