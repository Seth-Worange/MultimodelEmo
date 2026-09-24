"""两阶段训练 CICA 启发的置信度感知多模态情感模型。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from model.cica_net import CICAAffectiveModel, MODALITIES
from utils.augmentation import INTERVAL_RATIOS, mask_batch
from utils.config import parse_config_args
from utils.data import load_main
from utils.text import encode_text, load_text_encoder
from scripts.train import (
    LOCAL_RATE_RANGE, VIEWS, as_tensors, build_view, evaluate, make_batch,
    metrics, model_inputs, move_inputs, seed_everything, selection_score, evaluation_seed,
)

CAP_PREFIXES = (
    "embedding", "text_input", "text_gru", "audio_input", "audio_gru",
    "vision_input", "vision_gru", "text_transformer", "audio_transformer",
    "vision_transformer", "modal_heads", "confidence_calibration",
)
FUSION_PREFIXES = (
    "gate", "availability_embedding", "fusion_gru", "pool_score", "classifier", "regressor",
)


def _curriculum(epoch: int, start: int, ramp: int) -> float:
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start + 1) / max(1, ramp))


def _masked_batch(clean_cpu, args, epoch: int, offset: int):
    progress = _curriculum(epoch, args.text_missing_start_epoch,
                           args.text_missing_ramp_epochs)
    masked = mask_batch(
        clean_cpu,
        seed=args.seed + epoch * 100_003 + offset,
        probability=args.corruption_probability,
        pattern=args.corruption_mode,
        ratios=tuple(args.interval_ratios),
        interval_modalities=tuple(args.interval_modalities) if args.interval_modalities else None,
        overlap_probability=args.overlap_probability,
        whole_probability=args.whole_probability,
        local_rate_range=(args.local_rate_min, args.local_rate_max),
        text_whole_probability=args.text_whole_probability * progress,
        text_local_probability=args.text_local_probability * progress,
        text_local_rate_range=(args.text_local_rate_min, args.text_local_rate_max),
    )
    # 避免合成增强后一个样本三模态同时为空。
    empty = ~(masked["text_mask"].any(1) | masked["audio_mask"].any(1)
              | masked["vision_mask"].any(1))
    if empty.any():
        masked["tokens"][empty] = clean_cpu["tokens"][empty]
        masked["text_mask"][empty] = clean_cpu["text_mask"][empty]
    return masked


def _prepare_batch(cpu_batch, device, text_encoder, clean_cpu=None):
    batch = move_inputs(cpu_batch, device)
    if (text_encoder is not None and clean_cpu is not None
            and not torch.equal(cpu_batch["text_mask"], clean_cpu["text_mask"])):
        batch["teacher"] = encode_text(batch, text_encoder)
    return batch


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _branch_task_loss(model, output, batch, class_weights, args, reference_batch=None):
    losses = []
    reference = reference_batch or batch
    text_missing_rate = (1.0 - batch["text_mask"].sum(dim=1).float()
                         / reference["text_mask"].sum(dim=1).clamp_min(1).float()).clamp(0, 1)
    for index, name in enumerate(MODALITIES):
        valid = batch[f"{name}_mask"].any(dim=1)
        if not valid.any():
            continue
        labels = batch["classes"]
        sentiment = batch["sentiment"]
        cls = F.cross_entropy(output["modal_logits"][:, index], labels,
                              weight=class_weights, reduction="none")
        reg = F.smooth_l1_loss(output["modal_sentiment"][:, index], sentiment,
                               reduction="none")
        sample_weight = torch.ones_like(reg)
        if name in {"audio", "vision"}:
            sample_weight = 1.0 + (args.nontext_loss_boost - 1.0) * text_missing_rate
        sample_weight = sample_weight * valid.to(reg.dtype)
        task = _weighted_mean(cls + reg, sample_weight)
        target = torch.tanh((sentiment - output["modal_sentiment"][:, index].detach()).abs())
        uncertainty = F.mse_loss(output["uncertainty"][:, index], target, reduction="none")
        uncertainty = _weighted_mean(uncertainty, sample_weight)
        confidence = model.confidence_loss(name, output["confidence"][:, index][valid])
        losses.append(task + args.lambda_ca * confidence + args.lambda_uncert * uncertainty)
    return torch.stack(losses).mean() if losses else output["modal_logits"].sum() * 0.0


def _set_trainable_phase(model, phase: str):
    if phase == "cap":
        prefixes = CAP_PREFIXES
    elif phase == "fusion":
        prefixes = FUSION_PREFIXES
    else:
        raise ValueError(f"Unknown CICA training phase: {phase}")
    known_prefixes = CAP_PREFIXES + FUSION_PREFIXES
    unknown = [name for name, _ in model.named_parameters()
               if not name.startswith(known_prefixes)]
    if unknown:
        raise RuntimeError(f"Parameters are not assigned to a CICA phase: {unknown}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
        parameter.grad = None
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(not name.startswith(prefixes) for name in trainable):
        raise RuntimeError(f"Unexpected trainable parameters in {phase}: {trainable}")
    return trainable


def _fusion_train_mode(model):
    model.train()
    for module_name in ("embedding", "text_input", "text_gru", "audio_input", "audio_gru",
                        "vision_input", "vision_gru", "text_transformer", "audio_transformer",
                        "vision_transformer", "modal_heads", "confidence_calibration", "modal_dropout"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()


@torch.no_grad()
def _evaluate_branches(model, data, batch_size, device, text_encoder):
    model.eval()
    gathered = {name: {"logits": [], "sentiment": [], "classes": [], "target": [],
                       "confidence": [], "uncertainty": [], "uncertainty_target": []}
                for name in MODALITIES}
    for start in range(0, len(data["tokens"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        cpu_batch = make_batch(data, indices)
        batch = _prepare_batch(cpu_batch, device, text_encoder)
        output = model.forward_cap(*model_inputs(batch))
        for index, name in enumerate(MODALITIES):
            valid = batch[f"{name}_mask"].any(dim=1)
            if valid.any():
                gathered[name]["logits"].append(output["modal_logits"][valid, index].cpu().numpy())
                gathered[name]["sentiment"].append(output["modal_sentiment"][valid, index].cpu().numpy())
                gathered[name]["classes"].append(batch["classes"][valid].cpu().numpy())
                gathered[name]["target"].append(batch["sentiment"][valid].cpu().numpy())
                uncertainty_target = torch.tanh(
                    (batch["sentiment"] - output["modal_sentiment"][:, index]).abs())
                gathered[name]["confidence"].append(output["confidence"][valid, index].cpu().numpy())
                gathered[name]["uncertainty"].append(output["uncertainty"][valid, index].cpu().numpy())
                gathered[name]["uncertainty_target"].append(uncertainty_target[valid].cpu().numpy())
    result = {}
    for name, values in gathered.items():
        if not values["classes"]:
            result[name] = {"accuracy": 0.0, "macro_f1": 0.0, "mae": 3.0, "pearson": 0.0,
                            "mean_confidence": 0.0, "mean_uncertainty": 1.0,
                            "uncertainty_target_mae": 1.0}
        else:
            result[name] = metrics(
                np.concatenate(values["classes"]), np.concatenate(values["target"]),
                np.concatenate(values["logits"]), np.concatenate(values["sentiment"]))
            confidence = np.concatenate(values["confidence"])
            uncertainty = np.concatenate(values["uncertainty"])
            uncertainty_target = np.concatenate(values["uncertainty_target"])
            result[name].update({
                "mean_confidence": float(confidence.mean()),
                "mean_uncertainty": float(uncertainty.mean()),
                "uncertainty_target_mae": float(np.mean(np.abs(uncertainty - uncertainty_target))),
            })
    return result


def _cap_score(branch_metrics):
    scores = [0.5 * values["mae"] / 3 + 0.5 * (1 - values["macro_f1"])
              for values in branch_metrics.values()]
    return float(np.mean(scores))


def _mcp_loss(output, batch, temperature: float):
    losses = []
    for index, name in enumerate(MODALITIES):
        valid = batch[f"{name}_mask"].any(dim=1)
        if int(valid.sum()) < 2:
            continue
        fused = F.normalize(output["pooled"][valid], dim=-1)
        unimodal = F.normalize(output["modal_pooled"][valid, index], dim=-1)
        similarity = fused @ unimodal.T / temperature
        target = torch.arange(similarity.shape[0], device=similarity.device)
        losses.append(0.5 * (F.cross_entropy(similarity, target)
                             + F.cross_entropy(similarity.T, target)))
    return torch.stack(losses).mean() if losses else output["pooled"].sum() * 0.0


def _fusion_task_loss(output, batch, class_weights, boost, reference_batch=None):
    cls = F.cross_entropy(output["logits"], batch["classes"],
                          weight=class_weights, reduction="none")
    reg = F.smooth_l1_loss(output["sentiment"], batch["sentiment"], reduction="none")
    reference = reference_batch or batch
    missing_rate = (1.0 - batch["text_mask"].sum(dim=1).float()
                    / reference["text_mask"].sum(dim=1).clamp_min(1).float()).clamp(0, 1)
    weights = 1.0 + (boost - 1.0) * missing_rate
    return _weighted_mean(cls + reg, weights)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "runs" / "cica")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cap-epochs", type=int, default=25)
    parser.add_argument("--cap-patience", type=int, default=6)
    parser.add_argument("--cap-lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--evaluation-protocol", choices=("legacy_batch", "sample_v2"), default="legacy_batch")
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder-type", choices=("bigru", "transformer"), default="bigru")
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--normalize-inputs", action="store_true")
    parser.add_argument("--pack-aligned-grus", action="store_true")
    parser.add_argument("--text-mode", choices=("tokens", "bert"), default="bert")
    parser.add_argument("--audio-dynamics", action="store_true")
    parser.add_argument("--regression-mode", choices=("signed", "soft", "hard"), default="soft")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--corruption-probability", type=float, default=0.8)
    parser.add_argument("--corruption-mode", choices=("auto", "whole", "local", "interval", "mixed"), default="auto")
    parser.add_argument("--whole-probability", type=float, default=0.3)
    parser.add_argument("--overlap-probability", type=float, default=0.5)
    parser.add_argument("--interval-ratios", type=float, nargs="+", default=list(INTERVAL_RATIOS))
    parser.add_argument("--interval-modalities", nargs="+", default=["audio", "vision"], choices=list(MODALITIES))
    parser.add_argument("--local-rate-min", type=float, default=LOCAL_RATE_RANGE[0])
    parser.add_argument("--local-rate-max", type=float, default=LOCAL_RATE_RANGE[1])
    parser.add_argument("--lambda-ca", type=float, default=0.05)
    parser.add_argument("--ca-temperature", type=float, default=0.05)
    parser.add_argument("--lambda-uncert", type=float, default=0.5)
    parser.add_argument("--mcp-weight", type=float, default=0.1)
    parser.add_argument("--mcp-temperature", type=float, default=0.1)
    parser.add_argument("--nontext-loss-boost", type=float, default=1.5)
    parser.add_argument("--text-whole-probability", type=float, default=0.12)
    parser.add_argument("--text-local-probability", type=float, default=0.25)
    parser.add_argument("--text-local-rate-min", type=float, default=0.1)
    parser.add_argument("--text-local-rate-max", type=float, default=0.4)
    parser.add_argument("--text-missing-start-epoch", type=int, default=2)
    parser.add_argument("--text-missing-ramp-epochs", type=int, default=8)
    return parser


def main():
    args = parse_config_args(_parser(), "train")
    if (args.cap_epochs < 1 or args.epochs < 1 or args.batch_size < 1
            or args.cap_patience < 1 or args.patience < 1 or args.cap_lr <= 0 or args.lr <= 0
            or args.embedding_dim < 1 or args.hidden_dim < 1
            or args.transformer_layers < 1 or args.transformer_heads < 1
            or (args.hidden_dim * 2) % args.transformer_heads != 0
            or not 0 <= args.dropout < 1 or args.class_weight_power < 0
            or not 0 <= args.corruption_probability <= 1
            or not 0 <= args.whole_probability <= 1
            or not 0 <= args.overlap_probability <= 1
            or not 0 <= args.local_rate_min <= args.local_rate_max <= 1
            or any(not 0 < ratio <= 1 for ratio in args.interval_ratios)
            or args.lambda_ca < 0 or args.ca_temperature <= 0
            or args.lambda_uncert < 0 or args.mcp_weight < 0
            or args.mcp_temperature <= 0 or args.nontext_loss_boost < 1
            or args.text_missing_start_epoch < 1 or args.text_missing_ramp_epochs < 1
            or not 0 <= args.text_whole_probability <= 1
            or not 0 <= args.text_local_probability <= 1
            or args.text_whole_probability + args.text_local_probability > 1
            or not 0 <= args.text_local_rate_min <= args.text_local_rate_max <= 1):
        raise ValueError("训练轮数、学习率、损失权重或文本缺失配置不合法")

    seed_everything(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    train_data = as_tensors(load_main(args.data_root, "train"))
    valid_data = as_tensors(load_main(args.data_root, "valid"))
    model = CICAAffectiveModel(
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
        text_mode=args.text_mode, audio_dynamics=args.audio_dynamics,
        regression_mode=args.regression_mode, dropout=args.dropout,
        encoder_type=args.encoder_type, transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        ca_temperature=args.ca_temperature,
        normalize_inputs=args.normalize_inputs,
        pack_aligned_grus=args.pack_aligned_grus).to(device)
    if args.normalize_inputs:
        model.fit_input_stats(train_data)
        print("fitted audio/vision normalization from training split only")
    text_encoder = load_text_encoder(device) if args.text_mode == "bert" else None
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** args.class_weight_power,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_config = {"architecture": "cica", "embedding_dim": args.embedding_dim,
                    "hidden_dim": args.hidden_dim, "text_mode": args.text_mode,
                    "audio_dynamics": args.audio_dynamics,
                    "regression_mode": args.regression_mode, "dropout": args.dropout,
                    "encoder_type": args.encoder_type,
                    "transformer_layers": args.transformer_layers,
                    "transformer_heads": args.transformer_heads,
                    "ca_temperature": args.ca_temperature,
                    "normalize_inputs": args.normalize_inputs,
                    "pack_aligned_grus": args.pack_aligned_grus}

    cap_path = args.output_dir / "cap_metrics.csv"
    cap_best, cap_epoch, stale = float("inf"), 0, 0
    cap_trainable_names = _set_trainable_phase(model, "cap")
    print(f"CAP trainable tensors={len(cap_trainable_names)}; fusion tensors frozen")
    cap_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.cap_lr, weight_decay=1e-4)
    with cap_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["epoch", "train_loss", "val_score"] + [
            f"{name}_{key}" for name in MODALITIES
            for key in ("accuracy", "macro_f1", "mae", "pearson", "mean_confidence",
                        "mean_uncertainty", "uncertainty_target_mae")]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.cap_epochs + 1):
            model.train()
            order = torch.randperm(len(train_data["tokens"]))
            total, batches = 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                masked_cpu = _masked_batch(clean_cpu, args, epoch, offset)
                clean = _prepare_batch(clean_cpu, device, text_encoder)
                masked = _prepare_batch(masked_cpu, device, text_encoder, clean_cpu)
                cap_optimizer.zero_grad(set_to_none=True)
                clean_out = model.forward_cap(*model_inputs(clean))
                masked_out = model.forward_cap(*model_inputs(masked))
                loss = (0.35 * _branch_task_loss(model, clean_out, clean, class_weights, args)
                        + 0.65 * _branch_task_loss(
                            model, masked_out, masked, class_weights, args, clean))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                cap_optimizer.step()
                total += float(loss.detach())
                batches += 1
            branch_metrics = _evaluate_branches(model, valid_data, args.batch_size,
                                                device, text_encoder)
            score = _cap_score(branch_metrics)
            row = {"epoch": epoch, "train_loss": total / max(1, batches), "val_score": score}
            for name, values in branch_metrics.items():
                row.update({f"{name}_{key}": value for key, value in values.items()})
            writer.writerow(row)
            file.flush()
            print(f"CAP epoch={epoch:03d} loss={row['train_loss']:.4f} score={score:.4f}")
            if score < cap_best:
                cap_best, cap_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed,
                            "model_config": model_config, "val_branches": branch_metrics,
                            "val_score": score}, args.output_dir / "cap_best.pt")
            else:
                stale += 1
            if stale >= args.cap_patience:
                print(f"CAP early stopping at epoch {epoch}; best epoch={cap_epoch}")
                break

    cap_checkpoint = torch.load(args.output_dir / "cap_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(cap_checkpoint["model"])
    fusion_trainable_names = _set_trainable_phase(model, "fusion")
    print(f"CIF trainable tensors={len(fusion_trainable_names)}; CAP tensors frozen")
    fusion_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    metrics_path = args.output_dir / "metrics.csv"
    best_score, best_epoch, stale = float("inf"), 0, 0
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fields = ["epoch", "train_loss", "mcp_loss", "val_score"] + [
            f"{view}_{key}" for view in VIEWS
            for key in ("accuracy", "macro_f1", "mae", "pearson")]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            _fusion_train_mode(model)
            order = torch.randperm(len(train_data["tokens"]))
            total, mcp_total, batches = 0.0, 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                masked_cpu = _masked_batch(clean_cpu, args, epoch, offset)
                clean = _prepare_batch(clean_cpu, device, text_encoder)
                masked = _prepare_batch(masked_cpu, device, text_encoder, clean_cpu)
                fusion_optimizer.zero_grad(set_to_none=True)
                clean_out = model.forward_fusion(*model_inputs(clean))
                masked_out = model.forward_fusion(*model_inputs(masked))
                clean_task = _fusion_task_loss(
                    clean_out, clean, class_weights, args.nontext_loss_boost, clean)
                masked_task = _fusion_task_loss(
                    masked_out, masked, class_weights, args.nontext_loss_boost, clean)
                teacher_prob = torch.softmax(clean_out["logits"].detach() / 2, dim=-1)
                class_consistency = F.kl_div(
                    F.log_softmax(masked_out["logits"] / 2, dim=-1), teacher_prob,
                    reduction="batchmean") * 4
                regression_consistency = F.smooth_l1_loss(
                    masked_out["sentiment"], clean_out["sentiment"].detach())
                consistency = class_consistency + regression_consistency
                mcp = 0.5 * (_mcp_loss(clean_out, clean, args.mcp_temperature)
                             + _mcp_loss(masked_out, masked, args.mcp_temperature))
                loss = 0.35 * clean_task + 0.65 * masked_task + 0.05 * consistency + args.mcp_weight * mcp
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
                fusion_optimizer.step()
                total += float(loss.detach())
                mcp_total += float(mcp.detach())
                batches += 1

            view_metrics = evaluate(
                model, valid_data, args.batch_size, device, evaluation_seed(args), text_encoder,
                evaluation_protocol=args.evaluation_protocol,
                local_rate_range=(args.local_rate_min, args.local_rate_max))
            score = selection_score(view_metrics)
            row = {"epoch": epoch, "train_loss": total / max(1, batches),
                   "mcp_loss": mcp_total / max(1, batches), "val_score": score}
            for view in VIEWS:
                row.update({f"{view}_{key}": value for key, value in view_metrics[view].items()})
            writer.writerow(row)
            file.flush()
            print(f"CIF epoch={epoch:03d} loss={row['train_loss']:.4f} "
                  f"MCP={row['mcp_loss']:.4f} score={score:.4f} "
                  f"clean_f1={view_metrics['clean']['macro_f1']:.4f} "
                  f"local_f1={view_metrics['local']['macro_f1']:.4f} "
                  f"clean_mae={view_metrics['clean']['mae']:.4f}")
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "seed": args.seed,
                            "model_config": model_config, "val_views": view_metrics,
                            "val_score": score, "cap_epoch": cap_epoch,
                            "evaluation_protocol": args.evaluation_protocol,
                            "evaluation_seed": evaluation_seed(args),
                            "cap_val_score": cap_best}, args.output_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                print(f"CIF early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    run = vars(args).copy()
    run.update({"data_root": str(args.data_root) if args.data_root else "default",
                "config": str(args.config) if args.config else None,
                "output_dir": str(args.output_dir), "architecture": "cica",
                "device_used": str(device), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "cap_best_epoch": cap_epoch,
                "best_epoch": best_epoch, "elapsed_seconds": round(time.time() - started, 2),
                "cap_trainable_parameters": cap_trainable_names,
                "fusion_trainable_parameters": fusion_trainable_names})
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"saved {args.output_dir / 'best.pt'}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
