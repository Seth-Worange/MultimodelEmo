"""用独立的保留窗口实验检查问题三证据的充分性。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.infer import MODALITIES, find_files, load_model, margin, run_model, tensor_item
from utils.config import parse_config_args
from utils.data import prepare_sample, read_pickle, resolve_data_root
from utils.text import encode_text, load_text_encoder


def isolate_window(batch: dict, model, encoder, row: dict) -> dict:
    start, end = int(row["position_start"]), int(row["position_end_exclusive"])
    name = row["modality"]
    masks = {key: torch.zeros_like(batch[f"{key}_mask"]) for key in MODALITIES}
    masks[name][:, start:end] = batch[f"{name}_mask"][:, start:end]
    isolated = dict(batch)
    if name == "text":
        isolated["tokens"] = batch["tokens"].clone()
        keep = masks["text"] | (batch["tokens"][:, 0] == 101) | (batch["tokens"][:, 0] == 102)
        isolated["tokens"][:, 0] *= keep.to(isolated["tokens"].dtype)
        isolated["teacher"] = encode_text(isolated, encoder, masks["text"])
    return run_model(model, isolated, masks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parse_config_args(parser, "evidence")
    if not args.checkpoint or not args.evidence_csv or not args.output:
        parser.error("checkpoint, evidence-csv and output are required")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    if model.text_mode != "bert" or getattr(model, "bert_finetune", False):
        raise ValueError("This evidence check expects cached BERT features")
    encoder = load_text_encoder(device)
    rows = defaultdict(list)
    with args.evidence_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["id"]].append(row)

    records = []
    root = resolve_data_root(args.data_root)
    for path in find_files(root, "q3", None):
        item = prepare_sample(read_pickle(path))
        sample_id = item["id"] or path.stem
        ranked = rows[sample_id]
        if not ranked:
            continue
        top = min(ranked, key=lambda row: int(row["rank_global"]))
        alternatives = [
            row for row in ranked if row["modality"] == top["modality"]
            and (int(row["position_end_exclusive"]) <= int(top["position_start"])
                 or int(row["position_start"]) >= int(top["position_end_exclusive"]))
            and int(row["valid_positions"]) > 0
        ]
        if not alternatives:
            continue
        seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "little")
        random_row = random.Random(seed).choice(alternatives)
        batch = tensor_item(item, device)
        with torch.inference_mode():
            batch["teacher"] = encode_text(batch, encoder)
            full = run_model(model, batch)
            predicted = int(full["logits"].argmax(-1).item())
            empty_masks = {name: torch.zeros_like(batch[f"{name}_mask"]) for name in MODALITIES}
            empty = margin(run_model(model, batch, empty_masks)["logits"], predicted)
            top_gain = margin(isolate_window(batch, model, encoder, top)["logits"], predicted) - empty
            random_gain = margin(isolate_window(batch, model, encoder, random_row)["logits"], predicted) - empty
        records.append({"id": sample_id, "modality": top["modality"],
                        "top_insertion_gain": top_gain, "random_insertion_gain": random_gain})
    if not records:
        raise RuntimeError("No comparable windows found")
    gaps = np.array([r["top_insertion_gain"] - r["random_insertion_gain"] for r in records])
    report = {"n_samples": len(records), "mean_top_minus_random": float(gaps.mean()),
              "fraction_top_greater": float((gaps > 0).mean()), "per_sample": records,
              "note": "保留窗口对原预测类别的margin提升；随机窗口来自同一模态且与顶部窗口不重叠。"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_sample"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
