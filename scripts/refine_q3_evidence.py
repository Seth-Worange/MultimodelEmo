"""结合删除必要性与单窗口充分性筛选问题三时间证据。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from scripts.evaluate_q3_evidence import isolate_window
from scripts.infer import MODALITIES, find_files, load_model, margin, run_model, tensor_item, write_csv
from utils.config import parse_config_args
from utils.data import prepare_sample, read_pickle, resolve_data_root
from utils.text import encode_text, load_text_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--neutral-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    args = parse_config_args(parser, "refine")
    if not args.checkpoint or not args.evidence_csv or not args.output_dir or args.top_k < 1:
        parser.error("checkpoint, evidence-csv, output-dir and positive top-k are required")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, neutral_zero=args.neutral_zero)
    if model.text_mode != "bert" or getattr(model, "bert_finetune", False):
        raise ValueError("This refinement expects cached BERT features")
    encoder = load_text_encoder(device)
    windows = defaultdict(list)
    with args.evidence_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            windows[row["id"]].append(row)
    selected = []
    for path in find_files(resolve_data_root(args.data_root), "q3", None):
        item = prepare_sample(read_pickle(path))
        sample_id = item["id"] or path.stem
        candidates = sorted(windows[sample_id], key=lambda row: int(row["rank_global"]))[:args.top_k]
        if not candidates:
            continue
        batch = tensor_item(item, device)
        with torch.inference_mode():
            batch["teacher"] = encode_text(batch, encoder)
            full = run_model(model, batch)
            predicted = int(full["logits"].argmax(-1).item())
            empty_masks = {name: torch.zeros_like(batch[f"{name}_mask"]) for name in MODALITIES}
            empty_margin = margin(run_model(model, batch, empty_masks)["logits"], predicted)
            scored = []
            for row in candidates:
                insertion = margin(isolate_window(batch, model, encoder, row)["logits"], predicted) - empty_margin
                deletion = float(row["class_margin_drop"])
                # 两种证据都为正时，该窗口既必要又有单独支撑力。
                joint = min(max(deletion, 0.0), max(insertion, 0.0))
                scored.append({**row, "insertion_margin_gain": insertion,
                               "joint_evidence_score": joint})
        winner = max(scored, key=lambda row: row["joint_evidence_score"])
        selected.append({**winner, "selection_status":
                         "necessary_and_sufficient" if winner["joint_evidence_score"] > 0
                         else "no_dual_positive_window"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "q3_selected_evidence.csv", selected)
    summary = {"n_samples": len(selected), "dual_positive": sum(
        row["selection_status"] == "necessary_and_sufficient" for row in selected),
        "top_k": args.top_k, "method": "min(positive deletion, positive insertion)"}
    (args.output_dir / "q3_selected_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
