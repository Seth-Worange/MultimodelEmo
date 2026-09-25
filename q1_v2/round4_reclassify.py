"""Re-evaluate saved ASR evidence after a routing-code revision; no ASR rerun."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from .correspondence_qa import ASRWord, QAConfig, decide_correspondence, local_word_match
from .data_loader import load_samples
from .io_utils import read_json, write_json
from .round4 import write_audit_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, _ = load_samples(args.data_root, probe_decode=False)
    if len(records) != 100:
        raise ValueError("Expected 100 official samples")
    changed = []
    for record in records:
        sample_dir = args.output_dir / "samples" / record.sample_id
        qa_path = sample_dir / "correspondence_qa.json"
        words_path = sample_dir / "asr_words.csv"
        if not qa_path.is_file() or not words_path.is_file():
            continue
        qa = read_json(qa_path)
        with words_path.open("r", encoding="utf-8-sig", newline="") as handle:
            words = [ASRWord(
                row["word"], float(row["start_s"]), float(row["end_s"]),
                float(row["probability"]) if row["probability"] else None,
            ) for row in csv.DictReader(handle)]
        match = local_word_match(record.text, words)
        status, route, reason, review = decide_correspondence(
            audio_signal_present=bool(qa["audio_signal_present"]),
            speech_present=bool(qa["speech_present"]),
            detected_language=qa.get("detected_language"),
            language_probability=qa.get("language_probability"),
            match=match, config=QAConfig(),
        )
        if not qa.get("audio_timestamp_reliable") and route == "HIGH_CONFIDENCE_MATCH":
            status, route, reason, review = "UNRESOLVED", "REVIEW_REQUIRED", "audio_pts_mapping_unreliable", True
        if qa["correspondence_status"] != status or qa["route"] != route:
            before = sample_dir / "correspondence_qa_before_route_revision.json"
            if not before.exists():
                shutil.copy2(qa_path, before)
            changed.append({"sample_id": record.sample_id, "before": qa["correspondence_status"], "after": status})
        qa.update({
            "local_match": match,
            "text_audio_match_score": match["normalized_edit_similarity"],
            "matched_audio_start_s": match["matched_audio_start_s"],
            "matched_audio_end_s": match["matched_audio_end_s"],
            "extra_speech_before": match["extra_speech_before"],
            "extra_speech_after": match["extra_speech_after"],
            "correspondence_status": status,
            "route": route,
            "correspondence_reason": reason,
            "automatic_reason": reason,
            "manual_review_required": review,
            "official_text_present_in_audio": status == "MATCHED",
            "official_text_present": status == "MATCHED",
            "route_revision": "ambiguity_abstention_v2_saved_asr_evidence",
        })
        write_json(sample_dir / "local_match.json", match)
        write_json(qa_path, qa)
    summary = write_audit_reports(records, args.output_dir)
    write_json(args.output_dir / "route_reclassification.json", {
        "method": "Re-score untouched saved ASR word times against untouched official text; no audio inference rerun",
        "changed": changed,
        "evaluated_sample_count": summary["qa_sample_count"],
    })
    print(f"Reclassified {summary['qa_sample_count']} evaluated samples; changed={len(changed)}")


if __name__ == "__main__":
    main()
