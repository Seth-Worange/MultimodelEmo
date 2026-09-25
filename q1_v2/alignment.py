"""Explicit word alignment through Montreal Forced Aligner (MFA)."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import shutil
import subprocess
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .data_loader import OriginalWord
from .io_utils import sha256_file, write_csv, write_json
from .media import wav_to_sample_time


ALIGNMENT_FIELDS = (
    "sample_id", "word_index", "original_word", "normalized_word",
    "start_s", "end_s", "alignment_mask", "failure_reason",
)


class MFAUnavailableError(RuntimeError):
    """Raised when MFA or one of its explicitly requested models is missing."""


@dataclass
class MFAWord:
    label: str
    start_s: float
    end_s: float


@dataclass
class WordAlignment:
    sample_id: str
    word_index: int
    original_word: str
    normalized_word: str
    start_s: float
    end_s: float
    alignment_mask: int
    failure_reason: str = ""


@dataclass
class AlignmentResult:
    sample_id: str
    words: list[WordAlignment]
    mfa_words: list[MFAWord]
    command: list[str] = field(default_factory=list)
    mfa_version: str = ""
    status: str = "pending"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


def normalize_word(token: str) -> str:
    """Normalize for sequence matching only; original tokens remain untouched."""
    value = unicodedata.normalize("NFKC", token)
    value = value.replace("’", "'").replace("‘", "'").lower()
    value = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", value)
    value = re.sub(r"[^a-z0-9'-]", "", value)
    return value


def _compact(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_word(token))


def _entry_to_word(entry: Any) -> MFAWord | None:
    if isinstance(entry, dict):
        label = entry.get("label", entry.get("text", entry.get("word", entry.get("mark", ""))))
        start = entry.get("begin", entry.get("start", entry.get("xmin")))
        end = entry.get("end", entry.get("stop", entry.get("xmax")))
    elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
        start, end, label = entry[0], entry[1], entry[2]
    else:
        return None
    try:
        word = MFAWord(str(label).strip(), float(start), float(end))
    except (TypeError, ValueError):
        return None
    if not word.label or normalize_word(word.label) in {"", "sil", "sp", "spn"}:
        return None
    return word


def _json_word_candidates(value: Any, path: tuple[str, ...] = ()) -> list[tuple[int, list[MFAWord]]]:
    candidates: list[tuple[int, list[MFAWord]]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key).lower(),)
            if str(key).lower() in {"entries", "intervals", "items", "words"} and isinstance(child, list):
                words = [word for item in child if (word := _entry_to_word(item)) is not None]
                if words:
                    context = " ".join(child_path)
                    score = 10 if "word" in context else (0 if "phone" in context else 1)
                    candidates.append((score, words))
            candidates.extend(_json_word_candidates(child, child_path))
    elif isinstance(value, list):
        words = [word for item in value if (word := _entry_to_word(item)) is not None]
        if words:
            context = " ".join(path)
            score = 10 if "word" in context else (0 if "phone" in context else 1)
            candidates.append((score, words))
        for child in value:
            candidates.extend(_json_word_candidates(child, path))
    return candidates


def parse_mfa_json(path: Path) -> list[MFAWord]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = _json_word_candidates(payload)
    if not candidates:
        raise ValueError(f"No word interval tier found in MFA JSON: {path}")
    # Prefer a word-labelled tier, then the candidate containing most intervals.
    return max(candidates, key=lambda item: (item[0], len(item[1])))[1]


def parse_mfa_csv(path: Path) -> list[MFAWord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    words: list[MFAWord] = []
    for row in rows:
        tier = str(row.get("tier", row.get("type", row.get("tier_name", "")))).lower()
        if "phone" in tier:
            continue
        word = _entry_to_word(row)
        if word is not None:
            words.append(word)
    if not words:
        raise ValueError(f"No word intervals found in MFA CSV: {path}")
    return words


def parse_mfa_output(path: Path) -> list[MFAWord]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_mfa_json(path)
    if suffix == ".csv":
        return parse_mfa_csv(path)
    raise ValueError(f"Unsupported MFA output {path}; configure JSON or CSV")


def _sequence_mapping(original: list[str], aligned: list[str]) -> dict[int, int]:
    """Order-preserving edit alignment; only sufficiently similar pairs map."""
    n, m = len(original), len(aligned)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
        back[i][0] = "delete"
    for j in range(1, m + 1):
        cost[0][j] = float(j)
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = original[i - 1], aligned[j - 1]
            similarity = SequenceMatcher(None, _compact(a), _compact(b)).ratio()
            substitution = 0.0 if _compact(a) == _compact(b) and _compact(a) else 1.5 - similarity
            options = (
                (cost[i - 1][j - 1] + substitution, "match"),
                (cost[i - 1][j] + 1.0, "delete"),
                (cost[i][j - 1] + 1.0, "insert"),
            )
            cost[i][j], back[i][j] = min(options, key=lambda item: item[0])
    mapping: dict[int, int] = {}
    i, j = n, m
    while i or j:
        operation = back[i][j]
        if operation == "match":
            similarity = SequenceMatcher(None, _compact(original[i - 1]), _compact(aligned[j - 1])).ratio()
            if _compact(original[i - 1]) and similarity >= 0.75:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif operation == "delete":
            i -= 1
        elif operation == "insert":
            j -= 1
        else:
            break
    return mapping


def locate_mfa_unknowns(
    original_words: list[OriginalWord], mfa_words: list[MFAWord],
) -> dict[int, MFAWord]:
    """Link an <unk> to an original position only when its anchors are unique.

    This is diagnostic evidence, not a word timing assignment.
    """
    normalized = [normalize_word(word.text) for word in original_words]
    positions = [i for i, value in enumerate(normalized) if value]
    matched = _sequence_mapping(
        [normalized[i] for i in positions],
        [normalize_word(word.label) for word in mfa_words],
    )
    anchors = sorted((positions[i], j) for i, j in matched.items())
    matched_original = {i for i, _ in anchors}
    located: dict[int, MFAWord] = {}
    for mfa_index, word in enumerate(mfa_words):
        if word.label.lower() != "<unk>":
            continue
        before = max((i for i, j in anchors if j < mfa_index), default=-1)
        after = min((i for i, j in anchors if j > mfa_index), default=len(original_words))
        candidates = [
            i for i in range(before + 1, after)
            if normalized[i] and i not in matched_original and i not in located
        ]
        if len(candidates) == 1:
            located[candidates[0]] = word
    return located


def remap_mfa_words(
    sample_id: str,
    original_words: list[OriginalWord],
    mfa_words: list[MFAWord],
    *,
    wav_origin_media_s: float,
    timeline_origin_media_s: float,
    duration_s: float,
    overlap_tolerance_s: float = 0.02,
) -> list[WordAlignment]:
    normalized = [normalize_word(word.text) for word in original_words]
    speakable_positions = [i for i, word in enumerate(normalized) if word]
    mapping_local = _sequence_mapping(
        [normalized[i] for i in speakable_positions],
        [normalize_word(word.label) for word in mfa_words],
    )
    mapping = {speakable_positions[local]: mfa_index for local, mfa_index in mapping_local.items()}
    output: list[WordAlignment] = []
    previous_start = -math.inf
    previous_end = -math.inf
    for index, original in enumerate(original_words):
        norm = normalized[index]
        if not norm:
            output.append(WordAlignment(
                sample_id, index, original.text, norm, float("nan"), float("nan"), 0,
                "punctuation_only_no_spoken_form",
            ))
            continue
        if index not in mapping:
            output.append(WordAlignment(
                sample_id, index, original.text, norm, float("nan"), float("nan"), 0,
                "not_returned_or_not_matched_by_mfa",
            ))
            continue
        mfa_word = mfa_words[mapping[index]]
        start = wav_to_sample_time(mfa_word.start_s, wav_origin_media_s, timeline_origin_media_s)
        end = wav_to_sample_time(mfa_word.end_s, wav_origin_media_s, timeline_origin_media_s)
        failures: list[str] = []
        if not math.isfinite(start) or not math.isfinite(end):
            failures.append("non_finite_interval")
        if end <= start:
            failures.append("zero_or_negative_duration")
        if start < -0.05 or end > duration_s + 0.05:
            failures.append("interval_out_of_bounds")
        if start < previous_start:
            failures.append("non_monotonic_interval")
        if start < previous_end - overlap_tolerance_s:
            failures.append("abnormal_overlap")
        if math.isfinite(start) and math.isfinite(end):
            previous_start = start
            previous_end = end
        output.append(WordAlignment(
            sample_id, index, original.text, norm,
            start if not failures else float("nan"),
            end if not failures else float("nan"),
            int(not failures), " | ".join(failures),
        ))
    return output


def _run_probe(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=180,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "executable_not_found"}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "returncode": None, "stdout": exc.stdout or "", "stderr": "probe_timeout"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _probe_installed_model(executable: str, model_type: str, model_name: str) -> dict[str, Any]:
    """Validate a registered model using MFA's official ``model list`` command.

    MFA 3.4.2's dictionary inspector raises an internal ``WindowsPath``
    exception on Windows.  Listing registered models avoids that upstream bug
    while still requiring an exact local model-name match.
    """
    listing = _run_probe([executable, "model", "list", model_type])
    installed: list[str] = []
    if listing["ok"]:
        try:
            parsed = ast.literal_eval(str(listing["stdout"]).strip())
            if isinstance(parsed, (list, tuple)):
                installed = [str(value) for value in parsed]
        except (SyntaxError, ValueError):
            installed = []
    return {
        "ok": bool(listing["ok"] and model_name in installed),
        "requested_name": model_name,
        "installed_models": installed,
        "validation_method": f"mfa model list {model_type}",
        "command_result": listing,
    }


def probe_mfa(acoustic_model: str, dictionary: str) -> dict[str, Any]:
    executable = shutil.which("mfa")
    result: dict[str, Any] = {"executable": executable, "available": bool(executable)}
    if not executable:
        result["blocking_reason"] = "mfa executable not found on PATH"
        return result
    version = _run_probe([executable, "version"])
    acoustic = _probe_installed_model(executable, "acoustic", acoustic_model)
    dictionary_path = Path(dictionary)
    if dictionary_path.is_file():
        lexicon = {
            "ok": dictionary_path.suffix.lower() == ".dict",
            "requested_name": str(dictionary_path.resolve()),
            "validation_method": "existing local .dict file; validated by align_one",
            "sha256": sha256_file(dictionary_path),
        }
    else:
        lexicon = _probe_installed_model(executable, "dictionary", dictionary)
    result.update({"version": version, "acoustic_model": acoustic, "dictionary": lexicon})
    result["available"] = bool(version["ok"] and acoustic["ok"] and lexicon["ok"])
    if not acoustic["ok"]:
        result["blocking_reason"] = f"MFA acoustic model unavailable: {acoustic_model}"
    elif not lexicon["ok"]:
        result["blocking_reason"] = f"MFA pronunciation dictionary unavailable: {dictionary}"
    return result


def run_mfa_alignment(
    *,
    sample_id: str,
    original_text: str,
    original_words: list[OriginalWord],
    wav_path: Path,
    output_dir: Path,
    acoustic_model: str,
    dictionary: str,
    wav_origin_media_s: float,
    timeline_origin_media_s: float,
    duration_s: float,
    timeout_s: int = 900,
    temporary_directory: Path | None = None,
    beam: int | None = None,
) -> AlignmentResult:
    """Invoke the verified MFA 3.x ``align_one`` CLI and retain raw output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "transcript_original.txt"
    original_path.write_text(original_text, encoding="utf-8")
    write_json(output_dir / "original_word_mapping.json", [asdict(word) for word in original_words])
    probe = probe_mfa(acoustic_model, dictionary)
    write_json(output_dir / "mfa_probe.json", probe)
    if not probe.get("available"):
        raise MFAUnavailableError(str(probe.get("blocking_reason", "MFA prerequisites unavailable")))

    executable = str(probe["executable"])
    raw_path = output_dir / "mfa_raw.json"
    if temporary_directory is None:
        raise ValueError(
            "An explicit ASCII-only MFA temporary_directory is required on Windows "
            "so Kaldi/OpenFST never receives a non-ASCII native path"
        )
    native_root = temporary_directory.resolve()
    if not str(native_root).isascii():
        raise ValueError(f"MFA native work directory must be ASCII-only: {native_root}")
    native_sample_id = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id).lstrip("-") or "sample"
    native_attempt = native_root / f"{native_sample_id}-{uuid.uuid4().hex}"
    native_attempt.mkdir(parents=True, exist_ok=False)
    native_wav = native_attempt / "input.wav"
    native_text = native_attempt / "transcript.txt"
    native_raw = native_attempt / "mfa_raw.json"
    shutil.copy2(wav_path, native_wav)
    native_text.write_text(original_text, encoding="utf-8")
    native_dictionary = dictionary
    if Path(dictionary).is_file():
        native_dictionary_path = native_attempt / "supplemented.dict"
        shutil.copy2(dictionary, native_dictionary_path)
        native_dictionary = str(native_dictionary_path)
    command = [
        executable, "align_one", str(native_wav), str(native_text), native_dictionary,
        acoustic_model, str(native_raw), "--output_format", "json",
    ]
    if beam is not None:
        if beam <= 0:
            raise ValueError("beam must be positive")
        command.extend(["--beam", str(beam)])
    command.extend(["--temporary_directory", str(native_attempt / "temp")])
    write_json(output_dir / "mfa_native_work.json", {
        "source_wav_sample_relative_path": wav_path.name,
        "audit_transcript_sample_relative_path": original_path.name,
        "native_work_root": str(native_root),
        "native_attempt_directory": str(native_attempt),
        "ascii_only": str(native_attempt).isascii(),
        "retention": "retained for failure diagnosis; raw output is copied into the sample output",
    })
    write_json(output_dir / "mfa_command.json", {"argv": command})
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=timeout_s,
        encoding="utf-8", errors="replace",
    )
    (output_dir / "mfa_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "mfa_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MFA align_one failed with exit code {completed.returncode}; see mfa_stderr.log")
    if not native_raw.is_file():
        raise RuntimeError(f"MFA reported success but did not create {native_raw.name}")
    shutil.copy2(native_raw, raw_path)
    mfa_words = parse_mfa_output(raw_path)
    words = remap_mfa_words(
        sample_id, original_words, mfa_words,
        wav_origin_media_s=wav_origin_media_s,
        timeline_origin_media_s=timeline_origin_media_s,
        duration_s=duration_s,
    )
    result = AlignmentResult(
        sample_id=sample_id,
        words=words,
        mfa_words=mfa_words,
        command=command,
        mfa_version=str(probe.get("version", {}).get("stdout", "")).strip(),
        status="ok" if all(word.alignment_mask for word in words) else "partial",
    )
    write_csv(output_dir / "word_alignment.csv", (asdict(word) for word in words), ALIGNMENT_FIELDS)
    write_json(output_dir / "alignment_metadata.json", result.metadata())
    return result
