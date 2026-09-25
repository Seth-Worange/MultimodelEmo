"""Isolated CPU faster-whisper/Silero process (no torch, MFA or MediaPipe)."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-path", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-speech-ms", type=int, default=250)
    parser.add_argument("--vad-min-silence-ms", type=int, default=300)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=100)
    args = parser.parse_args()

    from faster_whisper import WhisperModel
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    with wave.open(str(args.wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000 or handle.getsampwidth() != 2:
            raise ValueError("ASR worker requires PCM16 mono 16 kHz WAV")
        waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    signal = bool(waveform.size and np.any(np.abs(waveform) > 1e-5))
    vad_options = VadOptions(
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
    )
    regions = get_speech_timestamps(waveform, vad_options, sampling_rate=16000) if signal else []
    speech_duration = sum((item["end"] - item["start"]) / 16000 for item in regions)
    segments_out = []
    words_out = []
    language = None
    language_probability = None
    if speech_duration >= args.vad_min_speech_ms / 1000:
        model = WhisperModel(str(args.model), device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            waveform, beam_size=args.beam_size, word_timestamps=True,
            vad_filter=True, vad_parameters={
                "threshold": args.vad_threshold,
                "min_speech_duration_ms": args.vad_min_speech_ms,
                "min_silence_duration_ms": args.vad_min_silence_ms,
                "speech_pad_ms": args.vad_speech_pad_ms,
            },
        )
        language = info.language
        language_probability = float(info.language_probability)
        for segment in segments:
            segments_out.append({
                "start_s": float(segment.start), "end_s": float(segment.end),
                "text": segment.text, "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
            })
            for word in segment.words or []:
                words_out.append({
                    "word": word.word, "start_s": float(word.start),
                    "end_s": float(word.end),
                    "probability": float(word.probability) if word.probability is not None else None,
                })
    result = {
        "audio_signal_present": signal,
        "vad_regions": [{"start_s": item["start"] / 16000, "end_s": item["end"] / 16000} for item in regions],
        "speech_duration_s": speech_duration,
        "detected_language": language,
        "language_probability": language_probability,
        "segments": segments_out,
        "words": words_out,
        "model_path": str(args.model.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ASR complete: vad_regions={len(regions)} words={len(words_out)} language={language}")


if __name__ == "__main__":
    main()
