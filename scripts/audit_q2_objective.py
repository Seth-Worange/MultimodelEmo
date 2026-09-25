"""抽取训练批次，比较各损失项的数值和编码层梯度。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model.fuse_net import build_fuse_regularization
from scripts.infer import load_model
from scripts.train import (as_tensors, consistency_loss, make_batch, model_inputs,
                           move_inputs)
from utils.augmentation import mask_batch
from utils.data import load_main
from utils.text import load_text_tokenizer
from scripts.train import attach_full_text_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint, device)
    run = json.loads((args.checkpoint.parent / "run.json").read_text(encoding="utf-8"))
    if model.bert_finetune:
        data = as_tensors(load_main(args.data_root, "train", need_teacher=False))
        tokenizer = (load_text_tokenizer(model.bert_model_name, model.bert_model_revision)
                     if model.bert_input_source == "raw_text" else None)
        attach_full_text_inputs(data, tokenizer, model.bert_max_length, model.bert_input_source)
    else:
        data = as_tensors(load_main(args.data_root, "train"))
    count = min(args.batch_size, len(data["classes"]))
    indices = torch.randperm(len(data["classes"]))[:count]
    clean_cpu = make_batch(data, indices)
    masked_cpu = mask_batch(
        clean_cpu, seed=args.seed, probability=run.get("corruption_probability", 0.8),
        pattern=run.get("corruption_mode", "auto"),
        interval_modalities=tuple(run.get("interval_modalities") or ("audio", "vision")),
        whole_probability=run.get("whole_probability", 0.3),
        local_rate_range=(run.get("local_rate_min", 0.05), run.get("local_rate_max", 0.45)),
    )
    clean, masked = move_inputs(clean_cpu, device), move_inputs(masked_cpu, device)
    labels = clean["classes"]
    counts = np.bincount(data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** run.get("class_weight_power", 0.5),
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    model.train()
    clean_out = model(*model_inputs(clean))
    masked_out = model(*model_inputs(masked))
    clean_weight = run.get("clean_supervision_weight", 0.35)
    classify = clean_weight * F.cross_entropy(clean_out["logits"], labels, weight=class_weights)
    classify += (1 - clean_weight) * F.cross_entropy(masked_out["logits"], labels,
                                                       weight=class_weights)
    regress = clean_weight * F.smooth_l1_loss(clean_out["sentiment"], clean["sentiment"])
    regress += (1 - clean_weight) * F.smooth_l1_loss(masked_out["sentiment"], clean["sentiment"])
    terms = {"classification": classify, "regression": regress,
             "consistency": 0.05 * consistency_loss(clean_out, masked_out)}
    reg = build_fuse_regularization(
        model, masked_out, clean_out, labels, class_weights,
        tau=run.get("tau", 0.1), kl_beta=run.get("kl_beta", 0.01),
        contrast_weight=run.get("contrast_weight", 0.1),
        info_weight=run.get("info_weight", 0.1),
        dual_weight=run.get("dual_weight", 0.05),
        mrc_weight=run.get("mrc_weight", 0.1),
        cross_weight=run.get("cross_weight", 0.1),
    )
    scale = run.get("auxiliary_scale", 1.0)
    for name, weight in (("contrast", "contrast_weight"), ("info", "info_weight"),
                         ("dual", "dual_weight"), ("mrc", "mrc_weight"),
                         ("kl", "kl_beta"), ("cross", "cross_weight")):
        terms[name] = scale * run.get(weight, 0.1) * reg[name]
    parameter_names = ("text_input.0.weight", "audio_input.0.weight", "vision_input.0.weight",
                       "fusion_gru.weight_ih_l0")
    named = dict(model.named_parameters())
    selected = [named[name] for name in parameter_names]
    report = {"checkpoint": str(args.checkpoint), "batch_size": count, "seed": args.seed,
              "auxiliary_scale": scale, "terms": {}}
    for name, value in terms.items():
        gradients = torch.autograd.grad(value, selected, retain_graph=True, allow_unused=True)
        report["terms"][name] = {
            "loss": float(value.detach()),
            "gradient_norms": {key: float(grad.norm().detach()) if grad is not None else 0.0
                               for key, grad in zip(parameter_names, gradients)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
