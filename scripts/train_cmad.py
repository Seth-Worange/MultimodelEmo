"""Train a missing-modality Student from an independent full-modality Teacher."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from model import AffectiveModel
from model.distillation import DistillationConfig, compute_distillation_losses
from scripts.train import (
    as_tensors,
    evaluate,
    make_batch,
    model_inputs,
    move_inputs,
    seed_everything,
    supervised_loss,
    supervised_loss_components,
)
from utils.augmentation import INTERVAL_RATIOS, mask_batch
from utils.config import parse_config_args
from utils.data import load_main


def load_teacher(path: Path, device: torch.device) -> tuple[AffectiveModel, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError(f"Teacher checkpoint has no model_config: {path}")
    if config.get("architecture", "baseline") != "baseline":
        raise ValueError("CMAD trainer currently requires a baseline Gate Teacher")
    model_keys = {"embedding_dim", "hidden_dim", "fusion", "text_mode", "audio_dynamics",
                  "regression_mode", "dropout"}
    model = AffectiveModel(**{key: value for key, value in config.items() if key in model_keys})
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Teacher checkpoint architecture mismatch: {error}") from error
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    print(f"teacher_checkpoint={path} epoch={checkpoint.get('epoch', 'unknown')} "
          f"val={checkpoint.get('val_clean', checkpoint.get('val_views', {}))}")
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "cmad")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--clean-task-weight", type=float, default=0.0)
    parser.add_argument("--clean-reg-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--corruption-probability", type=float, default=0.8)
    parser.add_argument("--whole-probability", type=float, default=0.3)
    parser.add_argument("--interval-ratios", type=float, nargs="+", default=list(INTERVAL_RATIOS))
    parser.add_argument("--interval-modalities", nargs="+", default=["audio", "vision"],
                        choices=["text", "audio", "vision"])
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--output-kd-weight", type=float, default=0.2)
    parser.add_argument("--feature-kd-weight", type=float, default=0.1)
    parser.add_argument("--correlation-kd-weight", type=float, default=0.05)
    parser.add_argument("--disable-output-kd", action="store_true")
    parser.add_argument("--disable-feature-kd", action="store_true")
    parser.add_argument("--disable-correlation-kd", action="store_true")
    args = parse_config_args(parser, "cmad")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, patience and lr must be positive")
    if not 0 <= args.corruption_probability <= 1 or not 0 <= args.whole_probability <= 1:
        parser.error("corruption and whole probabilities must be in [0, 1]")
    if args.clean_task_weight < 0 or args.clean_reg_weight < 0:
        parser.error("clean task/reg weights must be non-negative")

    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment")

    teacher, teacher_config = load_teacher(args.teacher_checkpoint, device)
    student = AffectiveModel(**{key: value for key, value in teacher_config.items()
                                if key in {"embedding_dim", "hidden_dim", "fusion", "text_mode",
                                           "audio_dynamics", "regression_mode", "dropout"}}).to(device)
    train_data = as_tensors(load_main(args.data_root, "train",
                                       need_teacher=teacher.text_mode == "bert"))
    valid_data = as_tensors(load_main(args.data_root, "valid",
                                       need_teacher=teacher.text_mode == "bert"))
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** 0.5,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    kd_config = DistillationConfig(
        temperature=args.temperature,
        output_enabled=not args.disable_output_kd,
        output_weight=args.output_kd_weight,
        feature_enabled=not args.disable_feature_kd,
        feature_weight=args.feature_kd_weight,
        correlation_enabled=not args.disable_correlation_kd,
        correlation_weight=args.correlation_kd_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = float("inf"), 0, 0
    started = time.time()
    fields = ["epoch", "task_cls_loss", "task_reg_loss", "task_loss", "kd_cls_loss",
              "kd_reg_loss", "feature_kd_loss", "correlation_kd_loss", "weighted_output_kd",
              "weighted_feature_kd", "weighted_correlation_kd", "total_kd_loss",
              "output_kd_to_task_ratio", "feature_kd_to_task_ratio",
              "correlation_kd_to_task_ratio", "total_loss", "val_score"]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            student.train()
            order = torch.randperm(len(train_data["tokens"]))
            totals = {key: 0.0 for key in fields[1:-1]}
            batches = 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                masked_cpu = mask_batch(
                    clean_cpu, seed=args.seed + epoch * 100_003 + offset,
                    probability=args.corruption_probability, pattern="auto",
                    ratios=tuple(args.interval_ratios),
                    interval_modalities=tuple(args.interval_modalities),
                    whole_probability=args.whole_probability,
                )
                clean = move_inputs(clean_cpu, device)
                masked = move_inputs(masked_cpu, device)
                if student.text_mode == "bert" and not torch.equal(
                        masked_cpu["text_mask"], clean_cpu["text_mask"]):
                    # Reuse aligned precomputed BERT features; zero positions
                    # masked from the Student rather than requiring a second
                    # transformers runtime during training.
                    masked["teacher"] = clean["teacher"].clone()
                    masked["teacher"][~masked["text_mask"]] = 0.0
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_out = teacher(*model_inputs(clean), return_features=True)
                student_out = student(*model_inputs(masked), return_features=True)
                task_cls_loss, task_reg_loss = supervised_loss_components(
                    student_out, masked, class_weights)
                task_loss = task_cls_loss + task_reg_loss
                if args.clean_task_weight > 0:
                    clean_student_out = student(*model_inputs(clean), return_features=True)
                    clean_cls, clean_reg = supervised_loss_components(
                        clean_student_out, clean, class_weights)
                    task_loss = task_loss + args.clean_task_weight * (clean_cls + clean_reg)
                elif args.clean_reg_weight > 0:
                    clean_student_out = student(*model_inputs(clean), return_features=True)
                    _, clean_reg = supervised_loss_components(clean_student_out, clean, class_weights)
                    task_loss = task_loss + args.clean_reg_weight * clean_reg
                kd = compute_distillation_losses(teacher_out, student_out, kd_config)
                total_loss = task_loss + kd["total_kd_loss"]
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                weighted_output = kd_config.output_weight * kd["output_kd_loss"]
                weighted_feature = kd_config.feature_weight * kd["feature_kd_loss"]
                weighted_correlation = kd_config.correlation_weight * kd["correlation_kd_loss"]
                epsilon = 1e-8
                values = {"task_cls_loss": task_cls_loss, "task_reg_loss": task_reg_loss,
                          "task_loss": task_loss, "kd_cls_loss": kd["kd_cls_loss"],
                          "kd_reg_loss": kd["kd_reg_loss"],
                          "feature_kd_loss": kd["feature_kd_loss"],
                          "correlation_kd_loss": kd["correlation_kd_loss"],
                          "weighted_output_kd": weighted_output,
                          "weighted_feature_kd": weighted_feature,
                          "weighted_correlation_kd": weighted_correlation,
                          "total_kd_loss": kd["total_kd_loss"],
                          "output_kd_to_task_ratio": weighted_output / task_loss.clamp_min(epsilon),
                          "feature_kd_to_task_ratio": weighted_feature / task_loss.clamp_min(epsilon),
                          "correlation_kd_to_task_ratio": weighted_correlation / task_loss.clamp_min(epsilon),
                          "total_loss": total_loss}
                for key, value in values.items():
                    totals[key] += float(value.detach())
                batches += 1
            student.eval()
            views = evaluate(student, valid_data, args.batch_size, device,
                             args.seed + 9001, None, views=("clean", "interval"))
            val_score = sum(view["mae"] / 3.0 + (1.0 - view["macro_f1"])
                            for view in views.values()) / len(views)
            row = {"epoch": epoch,
                   **{key: totals[key] / max(1, batches) for key in totals},
                   "val_score": val_score}
            writer.writerow(row)
            handle.flush()
            print(" ".join(f"{key}={value:.5f}" for key, value in row.items()
                           if key not in {"epoch", "val_score"}) + f" val_score={val_score:.5f}")
            if val_score < best_score:
                best_score, best_epoch, stale = val_score, epoch, 0
                torch.save({"model": student.state_dict(), "epoch": epoch, "seed": args.seed,
                            "model_config": teacher_config, "teacher_checkpoint": str(args.teacher_checkpoint),
                            "val_views": views, "val_score": val_score, "distillation": vars(kd_config)},
                           args.output_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                break
    run = vars(args).copy()
    run.update({"best_epoch": best_epoch, "elapsed_seconds": round(time.time() - started, 2),
                "device_used": str(device), "output_dir": str(args.output_dir)})
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
