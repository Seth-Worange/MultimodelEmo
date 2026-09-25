"""Conservative, label-free QA of official text against observed media.

ASR is diagnostic evidence only. The official transcript is never replaced by
recognized text, and a low-confidence QA result never authorizes MFA.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .alignment import normalize_word
from .data_loader import split_original_words
from .io_utils import write_csv, write_json
from .media import MediaResult


ASR_WORD_FIELDS = ("asr_word_index", "word", "normalized", "start_s", "end_s", "probability")


@dataclass(frozen=True)
class ASRWord:
    word: str
    start_s: float
    end_s: float
    probability: float | None = None


@dataclass(frozen=True)
class QAConfig:
    asr_model: str = "Systran/faster-whisper-base"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = 5
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 300
    vad_speech_pad_ms: int = 100
    min_speech_s: float = 0.25
    english_probability_min: float = 0.70
    match_recall_min: float = 0.80
    match_precision_min: float = 0.75
    match_edit_similarity_min: float = 0.70
    min_exact_tokens: int = 3
    crop_margin_s: float = 0.30
    visual_probe_fps: float = 2.0
    static_mean_absolute_difference_max: float = 0.006


def _tokens(text: str) -> list[str]:
    normalized = text.lower().replace("’", "'").replace("‘", "'")
    # Contraction expansion improves matching when Whisper emits two words.
    contractions = {
        "they've": "they have", "we've": "we have", "i've": "i have",
        "you've": "you have", "it's": "it is", "that's": "that is",
        "there's": "there is", "don't": "do not", "doesn't": "does not",
        "can't": "can not", "won't": "will not", "i'm": "i am",
        "we're": "we are", "they're": "they are", "you're": "you are",
    }
    for source, replacement in contractions.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", replacement, normalized)
    return [token for part in normalized.split() if (token := normalize_word(part))]


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    costs = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        next_costs = [i]
        for j, right in enumerate(b, 1):
            next_costs.append(min(
                next_costs[j - 1] + 1,
                costs[j] + 1,
                costs[j - 1] + (left != right),
            ))
        costs = next_costs
    return costs[-1]


def local_word_match(official_text: str, asr_words: Sequence[ASRWord]) -> dict[str, Any]:
    """Smith–Waterman local token alignment with explicit conservative metrics."""
    official = _tokens(official_text)
    observed: list[str] = []
    owner: list[int] = []
    for index, word in enumerate(asr_words):
        for token in _tokens(word.word):
            observed.append(token)
            owner.append(index)
    n, m = len(official), len(observed)
    empty = {
        "official_token_count": n, "asr_token_count": m,
        "matched_official_token_count": 0, "matched_asr_token_count": 0,
        "token_recall": 0.0, "token_precision": 0.0,
        "normalized_edit_similarity": 0.0, "local_alignment_score": 0.0,
        "matched_audio_start_s": None, "matched_audio_end_s": None,
        "matched_asr_word_start_index": None, "matched_asr_word_end_index": None,
        "extra_speech_before": False, "extra_speech_after": False,
        "alternative_nonoverlap_candidate_count": 0,
        "local_match_ambiguous": False,
    }
    if not n or not m:
        return empty
    scores = np.zeros((n + 1, m + 1), dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.uint8)
    best = (0, 0, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = int(scores[i - 1, j - 1]) + (2 if official[i - 1] == observed[j - 1] else -2)
            up = int(scores[i - 1, j]) - 1
            left = int(scores[i, j - 1]) - 1
            choices = (0, diagonal, up, left)
            direction = max(range(4), key=lambda k: choices[k])
            scores[i, j] = choices[direction]
            back[i, j] = direction
            if scores[i, j] > best[0]:
                best = (int(scores[i, j]), i, j)
    if best[0] <= 0:
        return empty
    _, i, j = best
    end_j = j
    exact_official: set[int] = set()
    exact_asr: set[int] = set()
    while i > 0 and j > 0 and scores[i, j] > 0:
        direction = int(back[i, j])
        if direction == 1:
            if official[i - 1] == observed[j - 1]:
                exact_official.add(i - 1)
                exact_asr.add(j - 1)
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    start_j = j
    if not exact_asr or start_j >= end_j:
        return empty
    first_word = owner[start_j]
    last_word = owner[end_j - 1]
    alternative_spans: set[tuple[int, int]] = set()
    for end_i in range(1, n + 1):
        for alternative_end_j in range(1, m + 1):
            if scores[end_i, alternative_end_j] < .95 * best[0]:
                continue
            cursor_i, cursor_j = end_i, alternative_end_j
            while cursor_i > 0 and cursor_j > 0 and scores[cursor_i, cursor_j] > 0:
                direction = int(back[cursor_i, cursor_j])
                if direction == 1:
                    cursor_i -= 1
                    cursor_j -= 1
                elif direction == 2:
                    cursor_i -= 1
                elif direction == 3:
                    cursor_j -= 1
                else:
                    break
            if alternative_end_j <= start_j or cursor_j >= end_j:
                alternative_spans.add((cursor_j, alternative_end_j))
    span = observed[start_j:end_j]
    edit = _levenshtein(official, span)
    similarity = max(0.0, 1.0 - edit / max(n, len(span), 1))
    return {
        "official_token_count": n, "asr_token_count": m,
        "matched_official_token_count": len(exact_official),
        "matched_asr_token_count": len(exact_asr),
        "token_recall": len(exact_official) / n,
        "token_precision": len(exact_asr) / len(span),
        "normalized_edit_similarity": similarity,
        "local_alignment_score": float(best[0]),
        "matched_audio_start_s": float(asr_words[first_word].start_s),
        "matched_audio_end_s": float(asr_words[last_word].end_s),
        "matched_asr_word_start_index": first_word,
        "matched_asr_word_end_index": last_word,
        "extra_speech_before": first_word > 0,
        "extra_speech_after": last_word < len(asr_words) - 1,
        "alternative_nonoverlap_candidate_count": len(alternative_spans),
        "local_match_ambiguous": bool(alternative_spans),
    }


def decide_correspondence(
    *, audio_signal_present: bool, speech_present: bool,
    detected_language: str | None, language_probability: float | None,
    match: dict[str, Any], config: QAConfig,
) -> tuple[str, str, str, bool]:
    """Return (status, route, reason, manual_review_required)."""
    if not audio_signal_present:
        return "NO_AUDIO", "INVALID_CORRESPONDENCE", "no_decodable_audio_signal", True
    if not speech_present:
        return "NO_SPEECH", "INVALID_CORRESPONDENCE", "vad_found_no_speech", True
    if detected_language and detected_language != "en" and (language_probability or 0) >= config.english_probability_min:
        return "NON_ENGLISH_SPEECH", "INVALID_CORRESPONDENCE", "confident_non_english_asr_language", True
    if detected_language != "en" or (language_probability or 0) < config.english_probability_min:
        return "UNRESOLVED", "REVIEW_REQUIRED", "asr_language_uncertain", True
    if match["matched_official_token_count"] < config.min_exact_tokens:
        return "TEXT_AUDIO_MISMATCH", "INVALID_CORRESPONDENCE", "too_few_exact_local_token_matches", True
    if match.get("local_match_ambiguous"):
        return "PARTIAL_MATCH", "REVIEW_REQUIRED", "multiple_competing_local_asr_matches", True
    reliable = (
        match["token_recall"] >= config.match_recall_min
        and match["token_precision"] >= config.match_precision_min
        and match["normalized_edit_similarity"] >= config.match_edit_similarity_min
        and match["matched_audio_start_s"] is not None
        and match["matched_audio_end_s"] is not None
    )
    if reliable:
        # Automatic location is a candidate, not human-verified boundary truth.
        return "MATCHED", "HIGH_CONFIDENCE_MATCH", "local_asr_match_meets_prespecified_thresholds", False
    if match["token_recall"] >= 0.35:
        return "PARTIAL_MATCH", "REVIEW_REQUIRED", "partial_or_ambiguous_local_asr_match", True
    return "TEXT_AUDIO_MISMATCH", "INVALID_CORRESPONDENCE", "low_local_token_recall", True


def inspect_audio_correspondence(
    media: MediaResult, official_text: str, *, asr_python: Path,
    asr_model_path: Path, config: QAConfig, output_dir: Path,
) -> dict[str, Any]:
    """Run actual Silero VAD/ASR in a clean CPU process and map WAV times."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(asr_python.resolve()), "-m", "q1_v2.asr_worker",
        "--wav-path", str(output_dir / media.wav_path),
        "--model", str(asr_model_path.resolve()),
        "--output-json", str(output_dir / "asr_worker_raw.json"),
        "--beam-size", str(config.asr_beam_size),
        "--vad-threshold", str(config.vad_threshold),
        "--vad-min-speech-ms", str(config.vad_min_speech_ms),
        "--vad-min-silence-ms", str(config.vad_min_silence_ms),
        "--vad-speech-pad-ms", str(config.vad_speech_pad_ms),
    ]
    write_json(output_dir / "asr_command.json", {"argv": command})
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    (output_dir / "asr_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "asr_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ASR worker exited {completed.returncode}; see asr_stderr.log")
    from .io_utils import read_json

    payload = read_json(output_dir / "asr_worker_raw.json")
    signal = bool(payload["audio_signal_present"])
    speech_duration = float(payload["speech_duration_s"])
    vad_regions = [
        {"start_s": item["start_s"] + media.wav_origin_sample_s,
         "end_s": item["end_s"] + media.wav_origin_sample_s}
        for item in payload["vad_regions"]
    ]
    speech = speech_duration >= config.min_speech_s
    language = payload["detected_language"]
    language_probability = payload["language_probability"]
    asr_segments = [
        {**item, "start_s": item["start_s"] + media.wav_origin_sample_s,
         "end_s": item["end_s"] + media.wav_origin_sample_s}
        for item in payload["segments"]
    ]
    asr_words = [ASRWord(
        item["word"], item["start_s"] + media.wav_origin_sample_s,
        item["end_s"] + media.wav_origin_sample_s, item["probability"],
    ) for item in payload["words"]]
    match = local_word_match(official_text, asr_words)
    status, route, reason, manual_review = decide_correspondence(
        audio_signal_present=signal, speech_present=speech,
        detected_language=language, language_probability=language_probability,
        match=match, config=config,
    )
    write_json(output_dir / "asr_segments.json", {
        "model": config.asr_model,
        "model_role": "diagnostic_ASR_not_official_transcript",
        "detected_language": language,
        "language_probability": language_probability,
        "vad_regions": vad_regions,
        "segments": asr_segments,
        "original_asr_text": " ".join(item["text"].strip() for item in asr_segments),
    })
    write_csv(output_dir / "asr_words.csv", (
        {"asr_word_index": i, "word": word.word, "normalized": " ".join(_tokens(word.word)),
         "start_s": word.start_s, "end_s": word.end_s, "probability": word.probability}
        for i, word in enumerate(asr_words)
    ), ASR_WORD_FIELDS)
    write_json(output_dir / "local_match.json", match)
    result = {
        "audio_signal_present": signal,
        "speech_present": speech,
        "speech_duration_s": speech_duration,
        "speech_language": language,
        "detected_language": language,
        "language_probability": language_probability,
        "english_speech_likely": language == "en" and (language_probability or 0) >= config.english_probability_min,
        "official_text_present_in_audio": status == "MATCHED",
        "official_text_present": status == "MATCHED",
        "official_word_count": len(split_original_words(official_text)),
        "asr_word_count": len(asr_words),
        "text_audio_match_score": match["normalized_edit_similarity"],
        "matched_audio_start_s": match["matched_audio_start_s"],
        "matched_audio_end_s": match["matched_audio_end_s"],
        "extra_speech_before": match["extra_speech_before"],
        "extra_speech_after": match["extra_speech_after"],
        "correspondence_status": status,
        "route": route,
        "correspondence_reason": reason,
        "automatic_reason": reason,
        "manual_review_required": manual_review,
        "local_match": match,
        "vad_regions": vad_regions,
    }
    return result


def inspect_visual_scene(
    media: MediaResult, *, video_path: Path, face_model_path: Path,
    config: QAConfig,
) -> dict[str, Any]:
    """Low-rate scene/face probe; no word feature extraction or speaker claim."""
    import cv2
    import mediapipe as mp

    from .visual_features import _decode_selected_rgb, choose_sampled_frames

    selections = choose_sampled_frames(media.video_frames, config.visual_probe_fps)
    decoded = _decode_selected_rgb(video_path, {int(item["frame_index"]) for item in selections})
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=face_model_path.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=False,
    )
    differences: list[float] = []
    previous: np.ndarray | None = None
    detected = 0
    multiple = 0
    observed = 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for item in selections:
            rgb = decoded.get(int(item["frame_index"]))
            if rgb is None:
                continue
            observed += 1
            gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32) / 255.0
            if previous is not None:
                differences.append(float(np.mean(np.abs(gray - previous))))
            previous = gray
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            count = len(landmarker.detect(image).face_landmarks)
            detected += count > 0
            multiple += count > 1
    mean_difference = float(np.mean(differences)) if differences else None
    return {
        "video_signal_present": observed > 0,
        "decoded_frame_count": len(media.video_frames),
        "sampled_frame_count": observed,
        "face_detected_frame_count": detected,
        "face_detection_rate": detected / observed if observed else None,
        "multiple_face_frame_count": multiple,
        "static_visual_score": 1.0 - mean_difference if mean_difference is not None else None,
        "static_or_near_static_visual": (
            mean_difference <= config.static_mean_absolute_difference_max
            if mean_difference is not None else None
        ),
        "face_present": detected > 0,
        "visual_face_feature_valid": detected > 0,
        "visual_probe_fps": config.visual_probe_fps,
        "visual_probe_frame_indices": [int(item["frame_index"]) for item in selections],
        "visual_probe_frame_times_s": [float(item["sample_time_s"]) for item in selections],
    }
