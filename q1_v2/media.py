"""Decode media while preserving FFmpeg/PyAV presentation timestamps."""

from __future__ import annotations

import math
import wave
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import write_json


AV_TIME_BASE = 1_000_000


def pts_to_seconds(pts: int | None, time_base: Fraction | None) -> float:
    """Convert a PTS to media seconds, returning NaN when it is unknowable."""
    if pts is None or time_base is None:
        return float("nan")
    return float(pts * time_base)


def media_to_sample_time(media_s: float, timeline_origin_media_s: float) -> float:
    return float(media_s - timeline_origin_media_s)


def sample_to_media_time(sample_s: float, timeline_origin_media_s: float) -> float:
    return float(sample_s + timeline_origin_media_s)


def wav_to_sample_time(
    wav_s: float, wav_origin_media_s: float, timeline_origin_media_s: float,
) -> float:
    """Map MFA/WAV-relative seconds onto the common sample-relative axis."""
    return float(wav_s + wav_origin_media_s - timeline_origin_media_s)


def sample_to_wav_time(
    sample_s: float, wav_origin_media_s: float, timeline_origin_media_s: float,
) -> float:
    return float(sample_s + timeline_origin_media_s - wav_origin_media_s)


@dataclass
class VideoFrameTiming:
    frame_index: int
    pts: int | None
    time_base: str | None
    media_time_s: float
    sample_time_s: float
    corrupt: bool
    timestamp_valid: bool


@dataclass
class AudioChunkTiming:
    chunk_index: int
    pts: int | None
    time_base: str | None
    media_start_s: float
    media_end_s: float
    wav_sample_start: int
    wav_sample_end: int
    timestamp_valid: bool


@dataclass
class MediaResult:
    video_path: str
    wav_path: str
    sample_rate: int
    duration_s: float
    container_start_media_s: float
    timeline_origin_media_s: float
    video_stream_start_media_s: float
    audio_stream_start_media_s: float
    video_stream_duration_s: float
    audio_stream_duration_s: float
    video_stream_end_media_s: float
    audio_stream_end_media_s: float
    audio_video_start_offset_s: float
    wav_origin_media_s: float
    wav_origin_sample_s: float
    audio_timestamp_reliable: bool
    video_timestamp_reliable: bool
    decoded_video_last_sample_s: float
    decoded_audio_end_sample_s: float
    decoded_audio_duration_s: float
    video_frames: list[VideoFrameTiming]
    audio_chunks: list[AudioChunkTiming]
    waveform: np.ndarray = field(repr=False)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("waveform", None)
        return data


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _stream_start_seconds(stream: Any) -> float:
    return pts_to_seconds(stream.start_time, stream.time_base)


def _stream_duration_seconds(stream: Any) -> float:
    return pts_to_seconds(stream.duration, stream.time_base)


def _container_times(video_path: Path) -> tuple[dict[str, float], list[str]]:
    import av

    warnings: list[str] = []
    with av.open(str(video_path)) as container:
        container_start = (
            float(container.start_time / AV_TIME_BASE)
            if container.start_time is not None else float("nan")
        )
        container_duration = (
            float(container.duration / AV_TIME_BASE)
            if container.duration is not None else float("nan")
        )
        video_start = _stream_start_seconds(container.streams.video[0]) if container.streams.video else float("nan")
        audio_start = _stream_start_seconds(container.streams.audio[0]) if container.streams.audio else float("nan")
        video_duration = _stream_duration_seconds(container.streams.video[0]) if container.streams.video else float("nan")
        audio_duration = _stream_duration_seconds(container.streams.audio[0]) if container.streams.audio else float("nan")
    if not _finite(container_start):
        warnings.append("missing_container_start_time")
    if not _finite(container_duration):
        warnings.append("missing_container_duration")
    if not _finite(video_start):
        warnings.append("missing_video_stream_start_time")
    if not _finite(audio_start):
        warnings.append("missing_audio_stream_start_time")
    return {
        "container_start": container_start,
        "duration": container_duration,
        "video_start": video_start,
        "audio_start": audio_start,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
    }, warnings


def _decode_video_times(video_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import av

    values: list[dict[str, Any]] = []
    errors: list[str] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return values, ["missing_video_stream"]
        stream = container.streams.video[0]
        previous = -math.inf
        for index, frame in enumerate(container.decode(stream)):
            time_s = pts_to_seconds(frame.pts, frame.time_base)
            valid = _finite(time_s)
            if not valid:
                errors.append(f"video_frame_missing_pts:{index}")
            elif time_s <= previous:
                errors.append(f"video_timestamp_non_monotonic:{index}:{previous}:{time_s}")
            if valid:
                previous = time_s
            values.append({
                "frame_index": index,
                "pts": frame.pts,
                "time_base": str(frame.time_base) if frame.time_base is not None else None,
                "media_time_s": time_s,
                "corrupt": bool(frame.is_corrupt),
                "timestamp_valid": valid,
            })
    if not values:
        errors.append("video_stream_has_no_decodable_frames")
    return values, errors


def _decode_resampled_audio(
    video_path: Path, sample_rate: int,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    import av

    chunks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            return chunks, ["missing_audio_stream"], warnings
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        source_index = -1
        for source_index, frame in enumerate(container.decode(stream)):
            outputs = resampler.resample(frame)
            if outputs is None:
                continue
            if not isinstance(outputs, list):
                outputs = [outputs]
            for output in outputs:
                array = np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1)
                media_start = pts_to_seconds(output.pts, output.time_base)
                chunks.append({
                    "source_index": source_index,
                    "pts": output.pts,
                    "time_base": str(output.time_base) if output.time_base is not None else None,
                    "media_start_s": media_start,
                    "samples": array,
                    "timestamp_valid": _finite(media_start),
                })
        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            for output in flushed:
                array = np.asarray(output.to_ndarray(), dtype=np.float32).reshape(-1)
                media_start = pts_to_seconds(output.pts, output.time_base)
                chunks.append({
                    "source_index": source_index + 1,
                    "pts": output.pts,
                    "time_base": str(output.time_base) if output.time_base is not None else None,
                    "media_start_s": media_start,
                    "samples": array,
                    "timestamp_valid": _finite(media_start),
                })
    if not chunks:
        errors.append("audio_stream_has_no_decodable_samples")
    return chunks, errors, warnings


def _assemble_timestamped_waveform(
    raw_chunks: list[dict[str, Any]], sample_rate: int,
) -> tuple[np.ndarray, float, list[AudioChunkTiming], bool, list[str]]:
    if not raw_chunks:
        return np.empty(0, dtype=np.float32), float("nan"), [], False, []
    warnings: list[str] = []
    valid_times = [chunk["media_start_s"] for chunk in raw_chunks if chunk["timestamp_valid"]]
    origin = valid_times[0] if valid_times else float("nan")
    reliable = len(valid_times) == len(raw_chunks) and bool(valid_times)
    if not reliable:
        warnings.append("audio_chunk_missing_pts_waveform_is_not_alignment_reliable")

    placements: list[tuple[int, int, dict[str, Any], float]] = []
    sequential_start = 0
    previous_media = -math.inf
    for chunk in raw_chunks:
        samples = chunk["samples"]
        media_start = chunk["media_start_s"]
        if chunk["timestamp_valid"] and _finite(origin):
            start = int(round((media_start - origin) * sample_rate))
            if media_start < previous_media:
                reliable = False
                warnings.append("audio_timestamp_non_monotonic")
            previous_media = media_start
        else:
            # The samples remain available for diagnostics, but downstream MFA
            # is disabled because this sequential placement is not observed time.
            start = sequential_start
            media_start = float("nan")
        if start < 0:
            reliable = False
            warnings.append("audio_chunk_precedes_waveform_origin")
            start = 0
        end = start + len(samples)
        placements.append((start, end, chunk, media_start))
        sequential_start = max(sequential_start, end)

    waveform = np.zeros(max((end for _, end, _, _ in placements), default=0), dtype=np.float32)
    occupied = np.zeros(len(waveform), dtype=np.uint8)
    timings: list[AudioChunkTiming] = []
    for chunk_index, (start, end, chunk, media_start) in enumerate(placements):
        overlap = bool(occupied[start:end].any())
        if overlap:
            reliable = False
            warnings.append(f"audio_chunk_overlap:{chunk_index}")
        waveform[start:end] = chunk["samples"]
        occupied[start:end] = 1
        timings.append(AudioChunkTiming(
            chunk_index=chunk_index,
            pts=chunk["pts"],
            time_base=chunk["time_base"],
            media_start_s=media_start,
            media_end_s=(media_start + len(chunk["samples"]) / sample_rate) if _finite(media_start) else float("nan"),
            wav_sample_start=start,
            wav_sample_end=end,
            timestamp_valid=bool(chunk["timestamp_valid"]),
        ))
    gap_samples = int((occupied == 0).sum())
    if gap_samples:
        warnings.append(f"audio_timestamp_gaps_zero_filled_samples:{gap_samples}")
    return waveform, origin, timings, reliable, warnings


def write_pcm16_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def inspect_and_extract_media(
    video_path: Path,
    wav_path: Path,
    *,
    sample_rate: int = 16_000,
    metadata_path: Path | None = None,
) -> MediaResult:
    """Decode a video and place audio/video on one sample-relative timeline."""
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    container_times, warnings = _container_times(video_path)
    raw_video, video_errors = _decode_video_times(video_path)
    raw_audio, audio_errors, audio_warnings = _decode_resampled_audio(video_path, sample_rate)
    errors = video_errors + audio_errors
    warnings.extend(audio_warnings)

    first_video = next((item["media_time_s"] for item in raw_video if item["timestamp_valid"]), float("nan"))
    first_audio = next((item["media_start_s"] for item in raw_audio if item["timestamp_valid"]), float("nan"))
    candidates = [
        container_times["container_start"], container_times["video_start"],
        container_times["audio_start"], first_video, first_audio,
    ]
    finite_candidates = [value for value in candidates if _finite(value)]
    if not finite_candidates:
        raise RuntimeError("No trustworthy container, stream, video-frame, or audio-frame timestamp")
    timeline_origin = min(finite_candidates)

    waveform, wav_origin, audio_chunks, audio_reliable, assembly_warnings = _assemble_timestamped_waveform(
        raw_audio, sample_rate,
    )
    warnings.extend(assembly_warnings)
    write_pcm16_wav(wav_path, waveform, sample_rate)

    video_frames = [
        VideoFrameTiming(
            frame_index=item["frame_index"],
            pts=item["pts"],
            time_base=item["time_base"],
            media_time_s=item["media_time_s"],
            sample_time_s=media_to_sample_time(item["media_time_s"], timeline_origin)
            if item["timestamp_valid"] else float("nan"),
            corrupt=item["corrupt"],
            timestamp_valid=item["timestamp_valid"],
        )
        for item in raw_video
    ]
    valid_video_times = [frame.sample_time_s for frame in video_frames if frame.timestamp_valid]
    video_reliable = bool(valid_video_times) and all(
        later > earlier for earlier, later in zip(valid_video_times, valid_video_times[1:])
    ) and not any(frame.corrupt for frame in video_frames)

    decoded_video_last = max(valid_video_times, default=float("nan"))
    decoded_audio_end = (
        wav_to_sample_time(len(waveform) / sample_rate, wav_origin, timeline_origin)
        if _finite(wav_origin) else float("nan")
    )
    observed_end = max(
        valid_video_times
        + ([decoded_audio_end] if _finite(decoded_audio_end) else [])
        + [0.0]
    )
    duration = container_times["duration"]
    if not _finite(duration):
        duration = observed_end
    video_start = container_times["video_start"]
    audio_start = container_times["audio_start"]
    offset = audio_start - video_start if _finite(audio_start) and _finite(video_start) else float("nan")
    result = MediaResult(
        video_path=str(video_path),
        # The sample directory is atomically renamed after processing, so an
        # absolute staging path would become stale.  This is sample-dir-relative.
        wav_path=wav_path.name,
        sample_rate=sample_rate,
        duration_s=duration,
        container_start_media_s=container_times["container_start"],
        timeline_origin_media_s=timeline_origin,
        video_stream_start_media_s=video_start,
        audio_stream_start_media_s=audio_start,
        video_stream_duration_s=container_times["video_duration"],
        audio_stream_duration_s=container_times["audio_duration"],
        video_stream_end_media_s=(
            video_start + container_times["video_duration"]
            if _finite(video_start) and _finite(container_times["video_duration"]) else float("nan")
        ),
        audio_stream_end_media_s=(
            audio_start + container_times["audio_duration"]
            if _finite(audio_start) and _finite(container_times["audio_duration"]) else float("nan")
        ),
        audio_video_start_offset_s=offset,
        wav_origin_media_s=wav_origin,
        wav_origin_sample_s=wav_to_sample_time(0.0, wav_origin, timeline_origin) if _finite(wav_origin) else float("nan"),
        audio_timestamp_reliable=audio_reliable,
        video_timestamp_reliable=video_reliable,
        decoded_video_last_sample_s=decoded_video_last,
        decoded_audio_end_sample_s=decoded_audio_end,
        decoded_audio_duration_s=len(waveform) / sample_rate,
        video_frames=video_frames,
        audio_chunks=audio_chunks,
        waveform=waveform,
        errors=errors,
        warnings=warnings,
    )
    if metadata_path is not None:
        write_json(metadata_path, result.metadata())
    return result
