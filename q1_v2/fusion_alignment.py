"""Organize explicit word-level alignment without learning a fusion network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment import WordAlignment
from .audio_features import AudioFeatureResult
from .data_loader import OriginalWord
from .io_utils import read_json, write_json
from .text_features import TextFeatureResult
from .visual_features import VisualFeatureResult


@dataclass
class FusedSample:
    sample_id: str
    original_words: list[str]
    word_start_s: np.ndarray
    word_end_s: np.ndarray
    text_features: np.ndarray
    audio_word_features: np.ndarray
    visual_word_features: np.ndarray
    text_mask: np.ndarray
    audio_mask: np.ndarray
    visual_mask: np.ndarray
    alignment_mask: np.ndarray
    audio_frame_features: np.ndarray
    audio_frame_times_s: np.ndarray
    visual_frame_features: np.ndarray
    visual_frame_times_s: np.ndarray
    audio_word_frame_indices: list[list[int]]
    visual_word_frame_indices: list[list[int]]
    visual_word_original_frame_indices: list[list[int]]
    audio_word_frame_counts: np.ndarray
    visual_word_frame_counts: np.ndarray


def _check_word_length(name: str, value: Any, expected: int) -> None:
    if len(value) != expected:
        raise ValueError(f"{name} length {len(value)} != original word count {expected}")


def build_fused_sample(
    sample_id: str,
    original_words: Sequence[OriginalWord],
    alignments: Sequence[WordAlignment],
    text: TextFeatureResult,
    audio: AudioFeatureResult,
    visual: VisualFeatureResult,
    *,
    output_dir: Path | None = None,
    correspondence_qa: Mapping[str, Any] | None = None,
) -> FusedSample:
    n_words = len(original_words)
    for name, value in (
        ("alignments", alignments), ("text_features", text.features),
        ("audio_word_features", audio.word_features),
        ("visual_word_features", visual.word_features),
        ("audio_word_frame_indices", audio.word_frame_indices),
        ("visual_word_frame_indices", visual.word_frame_indices),
    ):
        _check_word_length(name, value, n_words)
    if any(word.sample_id != sample_id for word in alignments):
        raise ValueError("Alignment sample_id does not match fused sample_id")
    starts = np.asarray([word.start_s for word in alignments], dtype=np.float64)
    ends = np.asarray([word.end_s for word in alignments], dtype=np.float64)
    alignment_mask = np.asarray([word.alignment_mask for word in alignments], dtype=np.uint8)
    fused = FusedSample(
        sample_id=sample_id,
        original_words=[word.text for word in original_words],
        word_start_s=starts,
        word_end_s=ends,
        text_features=text.features,
        audio_word_features=audio.word_features,
        visual_word_features=visual.word_features,
        text_mask=text.mask.astype(np.uint8),
        audio_mask=audio.word_mask.astype(np.uint8),
        visual_mask=visual.word_mask.astype(np.uint8),
        alignment_mask=alignment_mask,
        audio_frame_features=audio.frame_features,
        audio_frame_times_s=audio.frame_times_s,
        visual_frame_features=visual.frame_features,
        visual_frame_times_s=visual.frame_times_s,
        audio_word_frame_indices=audio.word_frame_indices,
        visual_word_frame_indices=visual.word_frame_indices,
        visual_word_original_frame_indices=visual.word_original_frame_indices,
        audio_word_frame_counts=audio.word_frame_counts,
        visual_word_frame_counts=visual.word_frame_counts,
    )
    if output_dir is not None:
        save_fused_sample(fused, alignments, audio, visual, output_dir, correspondence_qa=correspondence_qa)
    return fused


def save_fused_sample(
    fused: FusedSample,
    alignments: Sequence[WordAlignment],
    audio: AudioFeatureResult,
    visual: VisualFeatureResult,
    output_dir: Path,
    *,
    correspondence_qa: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    multimodal_mask = np.column_stack([fused.text_mask, fused.audio_mask, fused.visual_mask]).astype(np.uint8)
    arrays: dict[str, Any] = dict(
        sample_id=np.asarray(fused.sample_id),
        original_words=np.asarray(fused.original_words, dtype=np.str_),
        word_start_s=fused.word_start_s,
        word_end_s=fused.word_end_s,
        text_word_features=fused.text_features,
        audio_word_features=fused.audio_word_features,
        visual_word_features=fused.visual_word_features,
        text_mask=fused.text_mask,
        audio_mask=fused.audio_mask,
        visual_mask=fused.visual_mask,
        multimodal_mask=multimodal_mask,
        alignment_mask=fused.alignment_mask,
        audio_frame_features=fused.audio_frame_features,
        audio_frame_times_s=fused.audio_frame_times_s,
        audio_frame_f0_valid_mask=audio.f0_valid_mask,
        visual_frame_features=fused.visual_frame_features,
        visual_frame_times_s=fused.visual_frame_times_s,
        visual_original_frame_indices=visual.original_frame_indices,
        visual_detection_mask=visual.detection_mask,
        visual_face_counts=visual.face_counts,
        visual_speaker_identity_mask=visual.speaker_identity_mask,
        audio_word_frame_counts=fused.audio_word_frame_counts,
        visual_word_frame_counts=fused.visual_word_frame_counts,
        visual_word_valid_face_counts=visual.word_valid_face_counts,
        visual_word_estimated_mask=visual.word_estimated_mask,
    )
    if correspondence_qa is not None:
        arrays.update({
            "correspondence_status": np.asarray(str(correspondence_qa.get("correspondence_status", "UNRESOLVED"))),
            "audio_signal_present": np.asarray(int(bool(correspondence_qa.get("audio_signal_present"))), dtype=np.uint8),
            "speech_present": np.asarray(int(bool(correspondence_qa.get("speech_present"))), dtype=np.uint8),
            "video_signal_present": np.asarray(int(bool(correspondence_qa.get("video_signal_present"))), dtype=np.uint8),
            "face_present": np.asarray(int(bool(correspondence_qa.get("face_present"))), dtype=np.uint8),
        })
    np.savez_compressed(output_dir / "fused_features.npz", **arrays)
    metadata = {
        "sample_id": fused.sample_id,
        "word_count": len(fused.original_words),
        "variable_length_sequence": True,
        "padding_policy": (
            "No per-sample truncation. At batch construction, pad each modality to the maximum word count "
            "in that batch with NaN (or zero only after masking) and pad every validity mask with 0."
        ),
        "original_words": fused.original_words,
        "alignment_failure_reasons": [word.failure_reason for word in alignments],
        "audio_word_frame_indices": fused.audio_word_frame_indices,
        "visual_word_frame_indices": fused.visual_word_frame_indices,
        "visual_word_original_frame_indices": fused.visual_word_original_frame_indices,
        "audio_word_failure_reasons": audio.word_failure_reasons,
        "visual_word_status": visual.word_status,
        "dimensions": {
            "text_word": int(fused.text_features.shape[1]),
            "audio_frame": int(fused.audio_frame_features.shape[1]),
            "audio_word": int(fused.audio_word_features.shape[1]),
            "visual_frame": int(fused.visual_frame_features.shape[1]),
            "visual_word": int(fused.visual_word_features.shape[1]),
        },
    }
    if correspondence_qa is not None:
        metadata["correspondence_qa"] = {
            key: correspondence_qa.get(key) for key in (
                "correspondence_status", "route", "correspondence_reason",
                "matched_audio_start_s", "matched_audio_end_s",
                "audio_signal_present", "speech_present", "video_signal_present", "face_present",
            )
        }
    write_json(output_dir / "fused_metadata.json", metadata)


def validate_fused_output(sample_dir: Path, sample_id: str | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    npz_path = sample_dir / "fused_features.npz"
    metadata_path = sample_dir / "fused_metadata.json"
    success_path = sample_dir / "_SUCCESS.json"
    for path in (npz_path, metadata_path, success_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing_or_empty:{path.name}")
    if errors:
        return False, errors
    try:
        with np.load(npz_path, allow_pickle=False) as arrays:
            stored_id = str(arrays["sample_id"].item())
            word_count = len(arrays["original_words"])
            for key in (
                "word_start_s", "word_end_s", "text_word_features",
                "audio_word_features", "visual_word_features", "text_mask",
                "audio_mask", "visual_mask", "alignment_mask",
            ):
                if len(arrays[key]) != word_count:
                    errors.append(f"length_mismatch:{key}")
            if sample_id is not None and stored_id != sample_id:
                errors.append(f"sample_id_mismatch:{stored_id}:{sample_id}")
        metadata = read_json(metadata_path)
        if metadata.get("word_count") != word_count:
            errors.append("metadata_word_count_mismatch")
    except Exception as exc:
        errors.append(f"unreadable_output:{type(exc).__name__}:{exc}")
    return not errors, errors


def pad_word_sequences(
    sequences: Sequence[np.ndarray], masks: Sequence[np.ndarray], *, fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit future batching helper; inputs remain untruncated."""
    if not sequences:
        return np.empty((0, 0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.uint8)
    max_length = max(len(sequence) for sequence in sequences)
    feature_dim = int(sequences[0].shape[1])
    output = np.full((len(sequences), max_length, feature_dim), fill_value, dtype=np.float32)
    output_mask = np.zeros((len(sequences), max_length), dtype=np.uint8)
    for index, (sequence, mask) in enumerate(zip(sequences, masks)):
        if sequence.shape[1] != feature_dim or len(sequence) != len(mask):
            raise ValueError("Inconsistent sequence dimensions or mask length")
        output[index, :len(sequence)] = np.nan_to_num(sequence, nan=fill_value)
        output_mask[index, :len(mask)] = mask
    return output, output_mask
