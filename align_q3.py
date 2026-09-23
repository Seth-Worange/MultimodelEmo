"""Recover token-to-time mappings for attachment 4's explanatory evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import prepare_sample, read_pickle, resolve_data_root
from features_q1 import align_words, configure_cache, decode_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs" / "runs" / "main" / "q3_alignment.json")
    parser.add_argument("--text-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    configure_cache()
    root = resolve_data_root(args.data_root)
    input_dir = args.input_dir or next(root.glob(
        "附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"))
    video_dir = args.video_dir or input_dir.parent / "未对齐版本" / "videos"
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No aligned attachment 4 files under {input_dir}")
    try:
        import whisperx
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install requirements-q1.txt in the selected PyTorch environment") from error
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    mapping = {}

    for path in files:
        raw = read_pickle(path)
        item = prepare_sample(raw)
        sample_id = item["id"] or path.stem
        video = video_dir / f"{path.stem}.mp4"
        audio = decode_audio(video)
        words = align_words(item["raw_text"], audio, align_model, metadata, whisperx, device)
        word_text = [str(word.get("word", "")).strip() for word in words]
        encoded = tokenizer(word_text, is_split_into_words=True, return_tensors="np", truncation=True,
                            max_length=50, padding="max_length")
        encoded_ids = np.asarray(encoded["input_ids"])[0]
        token_ids = item["tokens"][0]
        word_ids = encoded.word_ids(batch_index=0)
        matches = (encoded_ids == token_ids) & (item["tokens"][1] > 0)
        match_fraction = float(matches.sum() / max(1, int((item["tokens"][1] > 0).sum())))
        positions = []
        for position, word_index in enumerate(word_ids):
            if position >= len(token_ids) or not matches[position] or word_index is None or word_index >= len(words):
                positions.append(None)
                continue
            word = words[word_index]
            positions.append({"word_index": int(word_index), "word": word_text[word_index],
                              "start": float(word["start"]), "end": float(word["end"])})
        mapping[sample_id] = {
            "video": video.name,
            "token_match_fraction": match_fraction,
            "status": "mapped" if match_fraction >= 0.9 else "review_required",
            "positions": positions,
            "words": [{"word": word_text[i], "start": float(word["start"]), "end": float(word["end"])}
                      for i, word in enumerate(words)],
        }
        print(f"{sample_id}: {len(words)} words, token_match={match_fraction:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"wrote token/time mappings to {args.output}; review_required entries need manual alignment QA")


if __name__ == "__main__":
    main()
