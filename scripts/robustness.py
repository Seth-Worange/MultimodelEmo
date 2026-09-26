"""按缺失形态、模态类型、缺失率与缺失位置评估验证集表现。

覆盖四类情形，对应题面“缺失模态类型、缺失率、缺失位置对预测性能的影响规律”：
1. ``none``：完整输入基线。
2. ``whole``：整段模态缺失（``audio``、``vision``、``audio+vision``、``text``）。
3. ``local``：语音与视觉在相同词位出现多段短游程（附件3的实测形态）。
4. ``interval``：多尺度连续缺失区间，比例取自 ``--rates``，模态组合覆盖
   双模态（音视频，四位置扫描）、三模态与单模态（text/audio/vision 各自单独缺失），
   多模态缺失区间含部分重叠放置。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from utils.augmentation import mask_batch
from utils.data import load_main
from scripts.infer import load_model
from utils.config import parse_config_args
from utils.text import (DEFAULT_BERT, DEFAULT_BERT_REVISION, encode_text,
                        load_text_encoder, load_text_tokenizer)
from scripts.train import (as_tensors, attach_full_text_inputs, make_batch, metrics,
                           model_inputs, move_inputs)

DEFAULT_RATES = (0.1, 0.2, 0.4, 0.6, 0.8)
DEFAULT_LOCATIONS = ("start", "middle", "end", "random")


@torch.inference_mode()
def score_case(model, data, device, batch_size, *, pattern="none", missing_modalities=(),
               rate=None, location=None, text_encoder=None, overlap_probability=0.5):
    model.eval()
    logits_all, sentiment_all = [], []
    removed = {name: 0 for name in missing_modalities}
    totals = {name: 0 for name in missing_modalities}
    for start in range(0, len(data["tokens"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        batch = make_batch(data, ix)
        original_masks = {name: batch[f"{name}_mask"].clone() for name in missing_modalities}
        if pattern != "none":
            samples = []
            for row, sample_id in enumerate(batch["ids"]):
                identity = json.dumps([2026, pattern, missing_modalities, rate, location, str(sample_id)])
                seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                item = make_batch(batch, torch.tensor([row]))
                if pattern == "whole":
                    item = mask_batch(item, seed=seed, probability=1.0,
                                      missing_modalities=missing_modalities)
                elif pattern == "local":
                    item = mask_batch(item, seed=seed, probability=1.0,
                                      local_rate=rate, location=location)
                else:
                    item = mask_batch(item, seed=seed, probability=1.0, pattern="interval",
                                      ratios=(rate,), interval_modalities=missing_modalities,
                                      location=location, overlap_probability=overlap_probability)
                samples.append(item)
            keys = ("tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask")
            batch = {**batch, **{key: torch.cat([item[key] for item in samples]) for key in keys}}
        for name in missing_modalities:
            totals[name] += int(original_masks[name].sum())
            removed[name] += int((original_masks[name] & ~batch[f"{name}_mask"]).sum())
        batch = move_inputs(batch, device)
        if "text" in missing_modalities and text_encoder is not None:
            batch["teacher"] = encode_text(batch, text_encoder)
        output = model(*model_inputs(batch))
        logits_all.append(output["logits"].cpu().numpy())
        sentiment_all.append(output["sentiment"].cpu().numpy())
    result = metrics(data["classes"].numpy(), data["sentiment"].numpy(),
                     np.concatenate(logits_all), np.concatenate(sentiment_all))
    for name in ("text", "audio", "vision"):
        result[f"actual_rate_{name}"] = removed.get(name, 0) / max(totals.get(name, 0), 1)
    return result


def build_cases(rates, locations) -> list[dict]:
    """构造对照表：基线 + 整段缺失 + 局部短游程扫描 + 连续区间扫描。"""
    cases = [{"pattern": "none", "missing_modalities": (), "missing_rate": 0.0, "location": "none"}]
    for missing in (("audio", "vision"), ("audio",), ("vision",), ("text",)):
        cases.append({"pattern": "whole", "missing_modalities": missing,
                      "missing_rate": 1.0, "location": "none"})
    for rate in rates:
        for location in locations:
            cases.append({"pattern": "local", "missing_modalities": ("audio", "vision"),
                          "missing_rate": rate, "location": location})
            for size in (1, 2, 3):
                for names in combinations(("text", "audio", "vision"), size):
                    cases.append({"pattern": "interval", "missing_modalities": names,
                                  "missing_rate": rate, "location": location})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append",
                        help="可重复指定多个检查点并集成")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "runs" / "main" / "robustness.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neutral-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rates", type=float, nargs="+", default=list(DEFAULT_RATES),
                        help="缺失率/缺失长度比例扫描点")
    parser.add_argument("--locations", nargs="+", default=list(DEFAULT_LOCATIONS),
                        choices=list(DEFAULT_LOCATIONS), help="缺失位置")
    parser.add_argument("--overlap-probability", type=float, default=0.5,
                        help="区间族中多模态缺失区间部分重叠的概率")
    args = parse_config_args(parser, "robustness")
    if not args.checkpoint:
        parser.error("provide --checkpoint or configure checkpoints")
    if any(not 0 < rate <= 1 for rate in args.rates):
        parser.error("rates must be in (0, 1]")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device, neutral_zero=args.neutral_zero)
    bert_finetune = bool(getattr(model, "bert_finetune", False))
    bert_model_name = getattr(model, "bert_model_name", DEFAULT_BERT)
    bert_revision = getattr(model, "bert_model_revision", DEFAULT_BERT_REVISION)
    text_encoder = (load_text_encoder(device, bert_model_name, bert_revision)
                    if model.text_mode == "bert" and not bert_finetune else None)
    data = as_tensors(load_main(args.data_root, "valid",
                                need_teacher=model.text_mode == "bert" and not bert_finetune))
    if bert_finetune:
        source = getattr(model, "bert_input_source", "raw_text")
        tokenizer = load_text_tokenizer(bert_model_name, bert_revision) if source == "raw_text" else None
        attach_full_text_inputs(data, tokenizer, model.bert_max_length, source)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from tqdm.auto import tqdm
    for case in tqdm(build_cases(args.rates, args.locations), desc="Validation robustness"):
        result = score_case(model, data, device, args.batch_size,
                            pattern=case["pattern"],
                            missing_modalities=case["missing_modalities"],
                            rate=case["missing_rate"] or None,
                            location=case["location"] if case["location"] != "none" else None,
                            text_encoder=text_encoder,
                            overlap_probability=args.overlap_probability)
        row = {"pattern": case["pattern"],
               "missing_modalities": "+".join(case["missing_modalities"]) or "audio+vision",
               "missing_rate": case["missing_rate"], "location": case["location"],
               "n": len(data["tokens"]), **result}
        clean = rows[0] if rows else row
        row.update(delta_macro_f1=row["macro_f1"] - clean["macro_f1"],
                   delta_mae=row["mae"] - clean["mae"])
        rows.append(row)
        with args.output.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{row['pattern']:>9} {row['missing_modalities']:>22} "
              f"rate={row['missing_rate']:.1f} {row['location']:>7} "
              f"F1={result['macro_f1']:.4f} MAE={result['mae']:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} validation cases to {args.output}")


if __name__ == "__main__":
    main()
