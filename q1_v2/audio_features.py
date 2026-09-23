"""Timestamped low-level acoustic descriptors and word-level statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .alignment import WordAlignment
from .io_utils import write_json


@dataclass
class AudioFeatureResult:
    frame_features: np.ndarray
    frame_times_s: np.ndarray
    frame_start_samples: np.ndarray
    f0_valid_mask: np.ndarray
    f0_analysis_times_s: np.ndarray
    f0_analysis_hz: np.ndarray
    f0_analysis_valid_mask: np.ndarray
    feature_names: list[str]
    word_features: np.ndarray
    word_mask: np.ndarray
    word_frame_indices: list[list[int]]
    word_frame_counts: np.ndarray
    word_f0_valid_counts: np.ndarray
    word_feature_valid_counts: np.ndarray
    word_failure_reasons: list[str]
    sample_rate: int
    window_samples: int
    hop_samples: int
    f0_window_samples: int

    def metadata(self) -> dict[str, Any]:
        return {
            "frame_feature_dim": int(self.frame_features.shape[1]),
            "word_feature_dim": int(self.word_features.shape[1]),
            "feature_names": self.feature_names,
            "word_statistics": ["mean", "standard_deviation"],
            "frame_count": int(len(self.frame_times_s)),
            "word_valid_length": int(self.word_mask.sum()),
            "sample_rate": self.sample_rate,
            "window_samples": self.window_samples,
            "hop_samples": self.hop_samples,
            "f0_window_samples": self.f0_window_samples,
            "frame_time_definition": "wav_origin_sample_s + (frame_start + window_samples/2)/sample_rate",
            "f0_time_definition": "wav_origin_sample_s + (f0_frame_start + f0_window_samples/2)/sample_rate",
            "f0_retiming": "nearest observed pYIN center within half one acoustic hop; no interpolation across unvoiced frames",
            "f0_missing_representation": "NaN with separate f0_valid_mask; never coerced to voiced zero",
            "word_frame_indices": self.word_frame_indices,
            "word_failure_reasons": self.word_failure_reasons,
        }


def frame_indices_in_interval(frame_times_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Select observed frame centers in the half-open interval [start, end)."""
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero((frame_times_s >= start_s) & (frame_times_s < end_s)).astype(np.int64)


def _fit_length(values: np.ndarray, count: int, *, fill: float = np.nan) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if len(flat) >= count:
        return flat[:count]
    return np.pad(flat, (0, count - len(flat)), constant_values=fill)


def _retime_f0_nearest(
    source_times_s: np.ndarray,
    source_f0_hz: np.ndarray,
    source_valid: np.ndarray,
    target_times_s: np.ndarray,
    *,
    max_distance_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map observed pYIN centers to the acoustic grid without inventing F0."""
    output = np.full(len(target_times_s), np.nan, dtype=np.float32)
    valid = np.zeros(len(target_times_s), dtype=np.uint8)
    if not len(source_times_s):
        return output, valid
    right = np.searchsorted(source_times_s, target_times_s, side="left")
    left = np.clip(right - 1, 0, len(source_times_s) - 1)
    right = np.clip(right, 0, len(source_times_s) - 1)
    choose_right = np.abs(source_times_s[right] - target_times_s) < np.abs(
        source_times_s[left] - target_times_s
    )
    nearest = np.where(choose_right, right, left)
    distances = np.abs(source_times_s[nearest] - target_times_s)
    accepted = (
        (distances <= max_distance_s + 1e-12)
        & np.asarray(source_valid, dtype=bool)[nearest]
        & np.isfinite(source_f0_hz[nearest])
    )
    output[accepted] = source_f0_hz[nearest[accepted]].astype(np.float32)
    valid[accepted] = 1
    return output, valid


def aggregate_audio_to_words(
    frame_features: np.ndarray,
    frame_times_s: np.ndarray,
    f0_valid_mask: np.ndarray,
    alignments: Sequence[WordAlignment],
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_dim = int(frame_features.shape[1])
    word_features = np.full((len(alignments), feature_dim * 2), np.nan, dtype=np.float32)
    word_mask = np.zeros(len(alignments), dtype=np.uint8)
    frame_lists: list[list[int]] = []
    frame_counts = np.zeros(len(alignments), dtype=np.int32)
    f0_counts = np.zeros(len(alignments), dtype=np.int32)
    valid_counts = np.zeros((len(alignments), feature_dim), dtype=np.int32)
    reasons: list[str] = []
    for word_index, alignment in enumerate(alignments):
        if not alignment.alignment_mask:
            indices = np.empty(0, dtype=np.int64)
            reasons.append("word_time_missing")
        else:
            indices = frame_indices_in_interval(frame_times_s, alignment.start_s, alignment.end_s)
            reasons.append("" if len(indices) else "no_audio_frame_center_in_word_interval")
        frame_lists.append(indices.tolist())
        frame_counts[word_index] = len(indices)
        if not len(indices):
            continue
        selected = frame_features[indices]
        finite = np.isfinite(selected)
        valid_counts[word_index] = finite.sum(axis=0)
        means = np.full(feature_dim, np.nan, dtype=np.float32)
        stds = np.full(feature_dim, np.nan, dtype=np.float32)
        for feature_index in range(feature_dim):
            values = selected[:, feature_index]
            values = values[np.isfinite(values)]
            if len(values):
                means[feature_index] = float(np.mean(values))
                stds[feature_index] = float(np.std(values))
        word_features[word_index] = np.concatenate([means, stds])
        f0_counts[word_index] = int(np.asarray(f0_valid_mask)[indices].sum())
        word_mask[word_index] = 1
    return word_features, word_mask, frame_lists, frame_counts, f0_counts, valid_counts, reasons


def extract_audio_features(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    wav_origin_sample_s: float,
    alignments: Sequence[WordAlignment],
    window_samples: int = 400,
    hop_samples: int = 160,
    f0_window_samples: int = 1024,
    n_mels: int = 64,
    fmin_hz: float = 50.0,
    fmax_hz: float = 500.0,
    output_dir: Path | None = None,
) -> AudioFeatureResult:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required; install requirements-q1-v2.txt") from exc
    y = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not np.isfinite(wav_origin_sample_s):
        raise ValueError("Audio waveform origin is not reliable; acoustic timestamps cannot be produced")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")
    if len(y) < window_samples:
        y = np.pad(y, (0, window_samples - len(y)))
    frame_count = 1 + (len(y) - window_samples) // hop_samples
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=window_samples, win_length=window_samples,
        hop_length=hop_samples, center=False, power=2.0, n_mels=n_mels,
        fmin=0.0, fmax=sample_rate / 2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=1.0, top_db=80.0).T[:frame_count]
    rms = _fit_length(librosa.feature.rms(
        y=y, frame_length=window_samples, hop_length=hop_samples, center=False,
    ), frame_count)
    zcr = _fit_length(librosa.feature.zero_crossing_rate(
        y, frame_length=window_samples, hop_length=hop_samples, center=False,
    ), frame_count)
    centroid = _fit_length(librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, n_fft=window_samples, win_length=window_samples,
        hop_length=hop_samples, center=False,
    ), frame_count)
    bandwidth = _fit_length(librosa.feature.spectral_bandwidth(
        y=y, sr=sample_rate, n_fft=window_samples, win_length=window_samples,
        hop_length=hop_samples, center=False,
    ), frame_count)
    if f0_window_samples < window_samples:
        raise ValueError("f0_window_samples must be at least the acoustic window length")
    y_f0 = y if len(y) >= f0_window_samples else np.pad(y, (0, f0_window_samples - len(y)))
    f0_source, voiced_flag, _voiced_probability = librosa.pyin(
        y_f0, fmin=fmin_hz, fmax=min(fmax_hz, sample_rate / 2.0 - 1.0), sr=sample_rate,
        frame_length=f0_window_samples, hop_length=hop_samples, center=False,
        fill_na=np.nan,
    )
    frame_starts = np.arange(frame_count, dtype=np.int64) * hop_samples
    frame_times = wav_origin_sample_s + (frame_starts + window_samples / 2.0) / sample_rate
    f0_source = np.asarray(f0_source, dtype=np.float32)
    f0_source_valid = np.asarray(voiced_flag, dtype=bool) & np.isfinite(f0_source)
    f0_source[~f0_source_valid] = np.nan
    f0_starts = np.arange(len(f0_source), dtype=np.int64) * hop_samples
    f0_times = wav_origin_sample_s + (f0_starts + f0_window_samples / 2.0) / sample_rate
    f0, f0_valid = _retime_f0_nearest(
        f0_times, f0_source, f0_source_valid, frame_times,
        max_distance_s=hop_samples / (2.0 * sample_rate),
    )
    frame_features = np.column_stack([log_mel, rms, f0, zcr, centroid, bandwidth]).astype(np.float32)
    names = [f"log_mel_{index:02d}" for index in range(n_mels)] + [
        "rms", "f0_hz", "zero_crossing_rate", "spectral_centroid_hz", "spectral_bandwidth_hz",
    ]
    aggregated = aggregate_audio_to_words(frame_features, frame_times, f0_valid, alignments)
    result = AudioFeatureResult(
        frame_features=frame_features,
        frame_times_s=frame_times.astype(np.float64),
        frame_start_samples=frame_starts,
        f0_valid_mask=f0_valid.astype(np.uint8),
        f0_analysis_times_s=f0_times.astype(np.float64),
        f0_analysis_hz=f0_source,
        f0_analysis_valid_mask=f0_source_valid.astype(np.uint8),
        feature_names=names,
        word_features=aggregated[0],
        word_mask=aggregated[1],
        word_frame_indices=aggregated[2],
        word_frame_counts=aggregated[3],
        word_f0_valid_counts=aggregated[4],
        word_feature_valid_counts=aggregated[5],
        word_failure_reasons=aggregated[6],
        sample_rate=sample_rate,
        window_samples=window_samples,
        hop_samples=hop_samples,
        f0_window_samples=f0_window_samples,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "audio_features.npz",
            frame_features=result.frame_features,
            frame_times_s=result.frame_times_s,
            frame_start_samples=result.frame_start_samples,
            f0_valid_mask=result.f0_valid_mask,
            f0_analysis_times_s=result.f0_analysis_times_s,
            f0_analysis_hz=result.f0_analysis_hz,
            f0_analysis_valid_mask=result.f0_analysis_valid_mask,
            word_features=result.word_features,
            word_mask=result.word_mask,
            word_frame_counts=result.word_frame_counts,
            word_f0_valid_counts=result.word_f0_valid_counts,
            word_feature_valid_counts=result.word_feature_valid_counts,
        )
        write_json(output_dir / "audio_features_metadata.json", result.metadata())
    return result
