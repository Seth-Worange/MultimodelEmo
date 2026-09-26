"""Second-stage review for stage-1 PARTIAL_MATCH samples only.

Scope (frozen by design): stage-1 already rejects TEXT_AUDIO_MISMATCH perfectly,
so this layer never re-opens INVALID_CORRESPONDENCE samples and never changes
stage-1 verdict fields. It re-examines PARTIAL_MATCH evidence with two tools:

  A. normalization — number words <-> digits only (e.g. "twelve hundred"=1200);
  B. block chain + gap annotation — multiple ASR-anchored segments with the
     in-between words labelled, so inserted extra speech ("and even as the")
     is separable from missing text.

Decision keeps the original thresholds (recall>=0.80, precision>=0.75,
edit>=0.70, exact units>=3, unambiguous). Two pass paths:

  A-pass: normalized single-region metrics meet thresholds
          -> policy REVIEWED_NORMALIZED_MFA (one crop, full text)
  B-pass: A fails, but block chain shows >= 2 segments covering the whole
          official text contiguously and EVERY inter-segment gap word is
          outside the official text (pure inserted speech)
          -> policy REVIEWED_BLOCKWISE_MFA (one MFA crop per segment)

Anything else stays blocked. Human labels are never read here.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .alignment import WordAlignment

# ---------------------------------------------------------------------------
# normalization (component A): number words <-> digits only
# ---------------------------------------------------------------------------

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES) | {"a"}


def _word_tokens(word: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", word.lower())


def _parse_number_run(run: Sequence[str]) -> int | None:
    if not run:
        return None
    if len(run) == 1 and run[0].isdigit():
        return int(run[0])
    total, current = 0, 0
    seen = False
    for token in run:
        if token.isdigit():
            current += int(token)
            seen = True
        elif token in _ONES:
            current += _ONES[token]
            seen = True
        elif token in _TENS:
            current += _TENS[token]
            seen = True
        elif token == "hundred":
            if current == 0:
                return None
            current *= 100
            seen = True
        elif token in ("thousand", "million"):
            scale = _SCALES[token]
            total += (current or 1) * scale
            current = 0
            seen = True
        else:
            return None
    return total + current if seen else None


@dataclass
class NormUnit:
    """One comparison unit: canonical text + the source word indices it spans."""
    text: str
    word_indices: tuple[int, ...]
    is_number: bool = False


def normalize_words(words: Sequence[str]) -> list[NormUnit]:
    """Word-level canonical units; number-word runs collapse to digit strings."""
    units: list[NormUnit] = []
    run_tokens: list[str] = []
    run_words: list[int] = []
    for index, word in enumerate(words):
        pieces = _word_tokens(word)
        joined = " ".join(pieces)
        number_like = bool(pieces) and all(
            token in _NUMBER_WORDS or token.isdigit() for token in pieces)
        # a number run continues while tokens are number words or digits
        if number_like and joined:
            run_tokens.extend(pieces)
            run_words.append(index)
            continue
        if run_words:
            value = _parse_number_run(run_tokens)
            if value is not None:
                units.append(NormUnit(str(value), tuple(run_words), True))
            else:
                for source in run_words:
                    units.append(NormUnit(" ".join(_word_tokens(words[source])), (source,)))
            run_tokens, run_words = [], []
        for piece in pieces:
            units.append(NormUnit(piece, (index,)))
    if run_words:
        value = _parse_number_run(run_tokens)
        if value is not None:
            units.append(NormUnit(str(value), tuple(run_words), True))
        else:
            for source in run_words:
                units.append(NormUnit(" ".join(_word_tokens(words[source])), (source,)))
    return units


def _fuzzy_unit_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and (b.startswith(a) or a.startswith(b)):
        return True
    if len(a) >= 5 and len(b) >= 5:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        # edit distance <= 1
        if longer.startswith(shorter) or longer.endswith(shorter):
            return True
        for i in range(len(shorter)):
            if shorter[:i] + shorter[i + 1:] == longer:
                return True
    return False


# ---------------------------------------------------------------------------
# matching primitives (same scoring as stage-1: +2 exact / -1 gap)
# ---------------------------------------------------------------------------

def smith_waterman(official: Sequence[str], observed: Sequence[str],
                   *, match: bool = True) -> tuple[int, int, int, list[tuple[int, int]]]:
    """Return (score, start_j, end_j, exact_pairs) of the best local block."""
    import numpy as np
    n, m = len(official), len(observed)
    scores = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.uint8)
    best = (0, 0, 0)
    compare = (lambda x, y: x == y) if match else (lambda x, y: _fuzzy_unit_match(x, y))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = int(scores[i - 1, j - 1]) + (2 if compare(official[i - 1], observed[j - 1]) else -2)
            up = int(scores[i - 1, j]) - 1
            left = int(scores[i, j - 1]) - 1
            choices = (0, diagonal, up, left)
            direction = max(range(4), key=lambda k: choices[k])
            scores[i, j] = choices[direction]
            back[i, j] = direction
            if scores[i, j] > best[0]:
                best = (int(scores[i, j]), i, j)
    if best[0] <= 0:
        return 0, 0, 0, []
    _, i, j = best
    end_j = j
    pairs: list[tuple[int, int]] = []
    while i > 0 and j > 0 and scores[i, j] > 0:
        direction = int(back[i, j])
        if direction == 1:
            if compare(official[i - 1], observed[j - 1]):
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    pairs.reverse()
    return best[0], j, end_j, pairs


def normalized_single_metrics(official: Sequence[NormUnit],
                              observed: Sequence[NormUnit]) -> dict[str, Any]:
    texts_o = [unit.text for unit in official]
    texts_a = [unit.text for unit in observed]
    score, start_j, end_j, pairs = smith_waterman(texts_o, texts_a)
    if not pairs:
        return {"score": 0.0, "matched_official": 0, "matched_observed": 0,
                "span_units": 0, "token_recall": 0.0, "token_precision": 0.0,
                "normalized_edit_similarity": 0.0, "observed_start_unit": None,
                "observed_end_unit": None}
    span = texts_a[start_j:end_j]
    matched_o = len({i for i, _ in pairs})
    matched_a = len({j for _, j in pairs})
    n, k = len(texts_o), len(span)
    # token-level levenshtein on unit texts (same metric as stage-1)
    edit = _unit_levenshtein(texts_o, span)
    similarity = max(0.0, 1.0 - edit / max(n, k, 1))
    return {
        "score": float(score), "matched_official": matched_o, "matched_observed": matched_a,
        "span_units": k, "token_recall": matched_o / n, "token_precision": matched_a / k,
        "normalized_edit_similarity": similarity,
        "observed_start_unit": start_j, "observed_end_unit": end_j,
    }


def _unit_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        current = [i]
        for j, token_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (token_a != token_b)))
        previous = current
    return previous[-1]


# ---------------------------------------------------------------------------
# component B: block chain + gap annotation
# ---------------------------------------------------------------------------

def _split_chain(chain: list[tuple[int, int]], official: Sequence[NormUnit],
                 observed: Sequence[NormUnit]) -> list[list[tuple[int, int]]]:
    pieces: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = [chain[0]]
    for (i1, j1), (i2, j2) in zip(chain, chain[1:]):
        gap_official = [unit.text for unit in official[i1 + 1:i2]]
        gap_observed = observed[j1 + 1:j2]
        pool = list(gap_official)
        unexplained = 0
        for unit in gap_observed:
            hit = None
            for index, text in enumerate(pool):
                if _fuzzy_unit_match(text, unit.text):
                    hit = index
                    break
            if hit is None:
                unexplained += 1
            else:
                pool.pop(hit)
        if gap_observed and unexplained:
            pieces.append(current)
            current = [(i2, j2)]
        else:
            current.append((i2, j2))
    pieces.append(current)
    return pieces


def build_segments(official: Sequence[NormUnit], observed: Sequence[NormUnit],
                   *, min_block_anchors: int = 1) -> list[dict[str, Any]]:
    """Greedy collinear anchor chaining (exact first, fuzzy extension inside)."""
    texts_o = [unit.text for unit in official]
    texts_a = [unit.text for unit in observed]
    anchors = [(i, j) for i, a in enumerate(texts_o) for j, b in enumerate(texts_a)
               if a == b or _fuzzy_unit_match(a, b)]
    # longest increasing chain (collinear in both sequences), greedy by length
    used_o: set[int] = set()
    used_a: set[int] = set()
    chains: list[list[tuple[int, int]]] = []
    while True:
        remaining = [(i, j) for i, j in anchors if i not in used_o and j not in used_a]
        if not remaining:
            break
        # dynamic programming over remaining anchors for the longest chain
        remaining.sort()
        lengths: dict[tuple[int, int], int] = {}
        prev: dict[tuple[int, int], tuple[int, int]] = {}
        for index, (i, j) in enumerate(remaining):
            best_key, best_len = (i, j), 1
            for index2 in range(index):
                i2, j2 = remaining[index2]
                if i2 < i and j2 < j and lengths[(i2, j2)] + 1 > best_len:
                    best_key, best_len = (i2, j2), lengths[(i2, j2)] + 1
            lengths[(i, j)] = best_len
            if best_len > 1:
                prev[(i, j)] = best_key
        if not lengths:
            break
        end_key = max(lengths, key=lambda key: lengths[key])
        if lengths[end_key] < min_block_anchors:
            break
        chain = [end_key]
        while chain[-1] in prev:
            chain.append(prev[chain[-1]])
        chain.reverse()
        exact = sum(1 for index in range(1, len(chain))
                    if texts_o[chain[index][0]] == texts_a[chain[index][1]])
        if exact + 0 < 1 and lengths[end_key] < 2:
            break
        # split the chain wherever the observed gap holds words with no
        # counterpart in the official gap (inserted extra speech)
        for piece in _split_chain(chain, official, observed):
            chains.append(piece)
            used_o.update(i for i, _ in piece)
            used_a.update(j for _, j in piece)
    chains.sort(key=lambda chain: (chain[0][0], chain[0][1]))
    segments: list[dict[str, Any]] = []
    for chain in chains:
        exact_pairs = [(i, j) for i, j in chain if texts_o[i] == texts_a[j]]
        segments.append({
            "official_unit_start": chain[0][0], "official_unit_end": chain[-1][0],
            "observed_unit_start": chain[0][1], "observed_unit_end": chain[-1][1],
            "anchor_units": len(chain), "exact_anchor_units": len(exact_pairs),
            "official_word_indices": sorted({w for i in range(chain[0][0], chain[-1][0] + 1)
                                             for w in official[i].word_indices}),
            "observed_word_indices": sorted({w for j in range(chain[0][1], chain[-1][1] + 1)
                                             for w in observed[j].word_indices}),
        })
    return segments


def annotate_gaps(official: Sequence[NormUnit], observed: Sequence[NormUnit],
                  segments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    official_texts = {unit.text for unit in official}
    gaps: list[dict[str, Any]] = []
    for left, right in zip(segments, segments[1:]):
        gap_observed = observed[left["observed_unit_end"] + 1:right["observed_unit_start"]]
        gap_official = official[left["official_unit_end"] + 1:right["official_unit_start"]]
        gaps.append({
            "between_segments": [left["official_unit_end"], right["official_unit_start"]],
            "gap_observed_units": [unit.text for unit in gap_observed],
            "gap_observed_words": sorted({w for unit in gap_observed for w in unit.word_indices}),
            "gap_official_units": [unit.text for unit in gap_official],
            "pure_inserted_speech": bool(gap_observed) and not gap_official
            and all(unit.text not in official_texts for unit in gap_observed),
        })
    return gaps


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

def review_partial_match(*, sample_id: str, official_text: str,
                         original_words: Sequence[Any], asr_rows: Sequence[dict[str, str]],
                         stage1_ambiguous: bool, thresholds: dict[str, float]) -> dict[str, Any]:
    official_words = [word.text for word in original_words]
    official_units = normalize_words(official_words)
    observed_words = [row["word"] for row in asr_rows]
    observed_units = normalize_words(observed_words)
    single = normalized_single_metrics(official_units, observed_units)
    segments = build_segments(official_units, observed_units)
    gaps = annotate_gaps(official_units, observed_units, segments)

    thresholds = {
        "match_recall_min": thresholds.get("match_recall_min", 0.80),
        "match_precision_min": thresholds.get("match_precision_min", 0.75),
        "match_edit_similarity_min": thresholds.get("match_edit_similarity_min", 0.70),
        "min_exact_tokens": thresholds.get("min_exact_tokens", 3),
    }
    path_a = (
        not stage1_ambiguous
        and single["token_recall"] >= thresholds["match_recall_min"]
        and single["token_precision"] >= thresholds["match_precision_min"]
        and single["normalized_edit_similarity"] >= thresholds["match_edit_similarity_min"]
        and single["matched_official"] >= thresholds["min_exact_tokens"]
        and single["observed_start_unit"] is not None
    )
    # path B: complete contiguous official coverage + pure-insertion gaps only
    covered = [segment["official_unit_start"] for segment in segments] + (
        [segments[-1]["official_unit_end"]] if segments else [])
    full_cover = bool(segments) and segments[0]["official_unit_start"] == 0 \
        and segments[-1]["official_unit_end"] == len(official_units) - 1 \
        and all(left["official_unit_end"] + 1 == right["official_unit_start"]
                for left, right in zip(segments, segments[1:]))
    pure_insertion = bool(gaps) and all(gap["pure_inserted_speech"] for gap in gaps)
    total_exact = sum(segment["exact_anchor_units"] for segment in segments)
    path_b = (
        not path_a and not stage1_ambiguous
        and len(segments) >= 2 and full_cover and pure_insertion
        and total_exact >= thresholds["min_exact_tokens"]
    )

    if path_a:
        decision = {"passed": True, "policy": "REVIEWED_NORMALIZED_MFA",
                    "reason": "normalized_metrics_meet_prespecified_thresholds", "plan": "single"}
    elif path_b:
        decision = {"passed": True, "policy": "REVIEWED_BLOCKWISE_MFA",
                    "reason": "block_chain_pure_insertion_gaps", "plan": "blockwise"}
    else:
        reason = "stage1_ambiguous_stays_blocked" if stage1_ambiguous else (
            "block_chain_not_pure_insertion" if len(segments) >= 2 and not pure_insertion
            else ("single_segment_only" if len(segments) < 2 else "thresholds_still_unmet"))
        decision = {"passed": False, "policy": None, "reason": reason, "plan": None}

    segments_out = []
    for segment in segments:
        indices = segment["official_word_indices"]
        observed_indices = segment["observed_word_indices"]
        segments_out.append({
            **segment,
            "official_text": " ".join(official_words[i] for i in indices),
            "audio_start_s": round(float(asr_rows[observed_indices[0]]["start_s"]), 3) if observed_indices else None,
            "audio_end_s": round(float(asr_rows[observed_indices[-1]]["end_s"]), 3) if observed_indices else None,
        })
    return {
        "sample_id": sample_id,
        "layer": "partial_match_review",
        "scope": "stage-1 PARTIAL_MATCH only; stage-1 verdict fields unchanged; human labels never read",
        "components": {
            "A_normalization": "number words <-> digits only",
            "B_block_chain": "multi-segment anchors with gap words annotated",
        },
        "thresholds": thresholds,
        "stage1_ambiguous": stage1_ambiguous,
        "normalized_single": single,
        "segments": segments_out,
        "gaps": gaps,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# diagnosis CLI
# ---------------------------------------------------------------------------

def diagnose(qa_root: Path, sample_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import numpy  # noqa: F401  (smith_waterman needs numpy at call time)
    results = []
    samples_dir = qa_root / "samples"
    audit = list(csv.DictReader((qa_root / "correspondence_audit.csv").open(encoding="utf-8-sig")))
    targets = [row for row in audit if row["correspondence_status"] == "PARTIAL_MATCH"]
    if sample_ids:
        wanted = set(sample_ids)
        targets = [row for row in targets if row["sample_id"] in wanted]
    for row in targets:
        sample_id = row["sample_id"]
        sample_dir = samples_dir / sample_id
        qa = json.loads((sample_dir / "correspondence_qa.json").read_text(encoding="utf-8"))
        match = json.loads((sample_dir / "local_match.json").read_text(encoding="utf-8"))
        asr_rows = list(csv.DictReader((sample_dir / "asr_words.csv").open(encoding="utf-8-sig")))
        from .data_loader import OriginalWord  # local import keeps CLI light
        import re as _re
        words = [OriginalWord(index, text, 0, 0) for index, text in
                 enumerate(_re.findall(r"[a-z0-9']+", qa["official_text"].lower()))]
        # NOTE: official word list here is token-level; fine for review geometry
        outcome = review_partial_match(
            sample_id=sample_id, official_text=qa["official_text"],
            original_words=words, asr_rows=asr_rows,
            stage1_ambiguous=bool(match.get("local_match_ambiguous")),
            thresholds={"match_recall_min": 0.80, "match_precision_min": 0.75,
                        "match_edit_similarity_min": 0.70, "min_exact_tokens": 3},
        )
        results.append(outcome)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--sample-id", action="append")
    args = parser.parse_args(argv)
    for outcome in diagnose(args.qa_root, args.sample_id):
        single = outcome["normalized_single"]
        decision = outcome["decision"]
        print(f"== {outcome['sample_id']}")
        print(f"   normalized: recall={single['token_recall']:.3f} "
              f"precision={single['token_precision']:.3f} "
              f"edit={single['normalized_edit_similarity']:.3f} "
              f"units={single['matched_official']} ambiguous={outcome['stage1_ambiguous']}")
        for segment in outcome["segments"]:
            print(f"   segment: official[{segment['official_unit_start']}..{segment['official_unit_end']}] "
                  f"audio {segment['audio_start_s']}~{segment['audio_end_s']}s "
                  f"anchors={segment['anchor_units']} exact={segment['exact_anchor_units']} "
                  f"\"{segment['official_text'][:60]}\"")
        for gap in outcome["gaps"]:
            print(f"   gap: observed={gap['gap_observed_units']} official={gap['gap_official_units']} "
                  f"pure_insertion={gap['pure_inserted_speech']}")
        print(f"   decision: passed={decision['passed']} policy={decision['policy']} "
              f"reason={decision['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# MFA execution for reviewed samples
# ---------------------------------------------------------------------------

def execute_reviewed_alignment(*, record: Any, media: Any, target: Path, config: Any,
                               dictionary: Path, mfa_work_dir: Path,
                               review: dict[str, Any],
                               qa: dict[str, Any]) -> tuple[list[WordAlignment], str]:
    """Run MFA under a passed review policy; never fabricate times."""
    from .alignment import ALIGNMENT_FIELDS, run_mfa_alignment
    from .io_utils import write_json
    from .round4_features import crop_for_local_mfa, unavailable_alignment

    policy = review["decision"]["policy"]
    duration_s = media.duration_s
    if policy == "REVIEWED_NORMALIZED_MFA":
        start = float(qa["matched_audio_start_s"])
        end = float(qa["matched_audio_end_s"])
        crop = crop_for_local_mfa(
            media, start, end, margin_s=config.alignment_margin_s,
            output_path=target / "audio_mfa_local_crop.wav")
        write_json(target / "mfa_crop_metadata.json", crop)
        result = run_mfa_alignment(
            sample_id=record.sample_id, original_text=record.text,
            original_words=record.original_words,
            wav_path=target / "audio_mfa_local_crop.wav", output_dir=target,
            acoustic_model=config.mfa_acoustic_model, dictionary=str(dictionary),
            wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
            timeline_origin_media_s=media.timeline_origin_media_s,
            duration_s=duration_s, timeout_s=config.mfa_timeout_s,
            temporary_directory=mfa_work_dir, beam=config.mfa_beam)
        if len(result.words) != record.word_count:
            raise ValueError("Reviewed MFA lost official word positions")
        return result.words, policy

    if policy != "REVIEWED_BLOCKWISE_MFA":
        raise ValueError(f"Unknown review policy: {policy}")
    merged = unavailable_alignment(record, "outside_reviewed_segments")
    entries: list[list[float | str]] = []
    for index, segment in enumerate(review["segments"]):
        indices = segment["official_word_indices"]
        if not indices or segment["audio_start_s"] is None:
            raise ValueError("Reviewed segment lacks anchors")
        crop = crop_for_local_mfa(
            media, float(segment["audio_start_s"]), float(segment["audio_end_s"]),
            margin_s=config.alignment_margin_s,
            output_path=target / f"segment_{index}_crop.wav")
        write_json(target / f"segment_{index}_crop_metadata.json", crop)
        segment_words = [record.original_words[i] for i in indices]
        segment_text = " ".join(word.text for word in segment_words)
        segment_dir = target / f"review_segment_{index}"
        result = run_mfa_alignment(
            sample_id=f"{record.sample_id}_seg{index}", original_text=segment_text,
            original_words=segment_words,
            wav_path=target / f"segment_{index}_crop.wav", output_dir=segment_dir,
            acoustic_model=config.mfa_acoustic_model, dictionary=str(dictionary),
            wav_origin_media_s=float(crop["crop_wav_origin_media_s"]),
            timeline_origin_media_s=media.timeline_origin_media_s,
            duration_s=duration_s, timeout_s=config.mfa_timeout_s,
            temporary_directory=mfa_work_dir, beam=config.mfa_beam)
        if len(result.words) != len(indices):
            raise ValueError("Reviewed segment MFA lost official word positions")
        for local_index, alignment in enumerate(result.words):
            global_index = indices[local_index]
            merged[global_index] = WordAlignment(
                record.sample_id, global_index, alignment.original_word,
                alignment.normalized_word, alignment.start_s, alignment.end_s,
                alignment.alignment_mask,
                alignment.failure_reason or f"reviewed_blockwise_segment_{index}")
            if alignment.alignment_mask:
                entries.append([round(float(alignment.start_s), 4),
                                round(float(alignment.end_s), 4), alignment.original_word])
    if not any(word.alignment_mask for word in merged):
        raise ValueError("Blockwise review produced no aligned words")
    entries.sort(key=lambda item: float(item[0]))
    write_json(target / "mfa_raw.json", {
        "start": 0, "end": float(duration_s),
        "tiers": {"words": {"type": "IntervalTier", "entries": entries}},
        "note": "merged blockwise review segments on global timeline; per-segment raw under review_segment_*/",
    })
    return merged, policy
