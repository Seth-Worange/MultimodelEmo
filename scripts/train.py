"""在 aligned_50.pkl 上训练并验证缺失鲁棒模型。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from utils.augmentation import INTERVAL_RATIOS, drop_modalities, mask_batch
from utils.config import parse_config_args
from utils.data import load_main
from utils.normalization import (apply_input_normalization, fit_input_normalization,
                                 normalization_summary)
from utils.text import encode_text, load_text_encoder
from model import AffectiveModel
from model.fuse_net import FactorizedAffectiveModel, build_fuse_regularization

ARCHITECTURES = ("baseline", "fuse")


def build_model(args) -> torch.nn.Module:
    """按配置构造情感模型；fuse 为 FUSE-Net 风格的三因子分解架构。"""
    if args.architecture == "fuse":
        return FactorizedAffectiveModel(
            text_mode=args.text_mode, audio_dynamics=args.audio_dynamics,
            regression_mode=args.regression_mode, dropout=args.dropout, tau=args.tau)
    return AffectiveModel(fusion=args.fusion, text_mode=args.text_mode,
                          audio_dynamics=args.audio_dynamics,
                          regression_mode=args.regression_mode, dropout=args.dropout)


def model_config(args) -> dict:
    """保存到检查点、供 load_model 还原结构的配置。"""
    common = {"architecture": args.architecture, "text_mode": args.text_mode,
              "audio_dynamics": args.audio_dynamics, "regression_mode": args.regression_mode,
              "dropout": args.dropout}
    if args.architecture == "fuse":
        return {**common, "tau": args.tau}
    return {**common, "fusion": args.fusion}


REG_FIELDS = ("contrast", "info", "dual", "mrc", "kl", "cross")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """数值字段转张量；样本标识等非数值字段原样保留。"""
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            out[key] = value
    return out


def make_batch(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in data.items():
        out[key] = value.index_select(0, indices) if isinstance(value, torch.Tensor) \
            else [value[i] for i in indices.tolist()]
    return out


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def model_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (batch["tokens"][:, 0], batch["lengths"], batch["audio"], batch["vision"],
            batch["text_mask"], batch["audio_mask"], batch["vision_mask"], batch.get("teacher"))


def supervised_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                    class_weights: torch.Tensor | None = None) -> torch.Tensor:
    classification, regression = supervised_loss_components(output, batch, class_weights)
    return classification + regression


def supervised_loss_components(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                               class_weights: torch.Tensor | None = None
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the existing classification and regression terms separately."""
    classification = F.cross_entropy(output["logits"], batch["classes"], weight=class_weights)
    regression = F.smooth_l1_loss(output["sentiment"], batch["sentiment"])
    return classification, regression


def distillation_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    predicted = F.normalize(output["teacher_pred"], dim=-1)
    target = F.normalize(batch["teacher"], dim=-1)
    valid = batch["text_mask"].to(predicted.dtype)
    cosine_error = 1 - (predicted * target).sum(dim=-1)
    return (cosine_error * valid).sum() / valid.sum().clamp_min(1)


def consistency_loss(clean: dict[str, torch.Tensor], masked: dict[str, torch.Tensor]) -> torch.Tensor:
    teacher_prob = torch.softmax(clean["logits"].detach() / 2, dim=-1)
    student_log_prob = torch.log_softmax(masked["logits"] / 2, dim=-1)
    classification = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * 4
    regression = F.smooth_l1_loss(masked["sentiment"], clean["sentiment"].detach())
    return classification + regression


def metrics(classes: np.ndarray, sentiment: np.ndarray, logits: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    guess = logits.argmax(axis=1)
    f1s = []
    for label in range(3):
        tp = np.sum((guess == label) & (classes == label))
        fp = np.sum((guess == label) & (classes != label))
        fn = np.sum((guess != label) & (classes == label))
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    corr = np.corrcoef(sentiment, predicted)[0, 1] if len(sentiment) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "accuracy": float(np.mean(guess == classes)),
        "macro_f1": float(np.mean(f1s)),
        "negative_f1": float(f1s[0]),
        "neutral_f1": float(f1s[1]),
        "positive_f1": float(f1s[2]),
        "mae": float(np.mean(np.abs(sentiment - predicted))),
        "pearson": float(corr),
    }


VIEWS = ("clean", "local", "whole", "interval")
LOCAL_RATE_RANGE = (0.05, 0.45)


def build_view(cpu_batch: dict[str, torch.Tensor], view: str, seed: int,
               local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
               ratios: tuple[float, ...] = INTERVAL_RATIOS) -> dict[str, torch.Tensor]:
    """构造一个验证视图。

    - ``clean``：完整输入。
    - ``local``：语音与视觉在同词位上出现多段短游程缺失（附件3的实测形态）。
    - ``whole``：整段丢弃语音与视觉（极端压力测试）。
    - ``interval``：多尺度连续缺失区间，比例在候选集合中抽取。
    """
    if view == "clean":
        return cpu_batch
    if view == "whole":
        return mask_batch(cpu_batch, seed=seed, probability=1.0,
                          missing_modalities=("audio", "vision"))
    if view == "local":
        rate = random.Random(seed).uniform(*local_rate_range)
        return mask_batch(cpu_batch, seed=seed, probability=1.0, local_rate=rate)
    if view == "interval":
        return mask_batch(cpu_batch, seed=seed, probability=1.0, pattern="interval", ratios=ratios)
    raise ValueError(f"Unknown view: {view}")


def selection_score(view_metrics: dict[str, dict[str, float]]) -> float:
    """模型选择分数，越低越好：三个视图的 MAE 与 macro-F1 等权平均。"""
    mae = float(np.mean([m["mae"] for m in view_metrics.values()])) / 3.0
    f1 = 1.0 - float(np.mean([m["macro_f1"] for m in view_metrics.values()]))
    return 0.5 * mae + 0.5 * f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    seed: int,
    text_encoder=None,
    views: tuple[str, ...] = VIEWS,
    local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
    drop: tuple[str, ...] = (),
) -> dict[str, dict[str, float]]:
    model.eval()
    gathered: dict[str, list[np.ndarray]] = {f"{view}_{key}": [] for view in views
                                             for key in ("logits", "sentiment")}
    rng = random.Random(seed)
    for start in range(0, len(data["tokens"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        clean_cpu = make_batch(data, ix)
        if drop:
            clean_cpu = drop_modalities(clean_cpu, drop)
        # 先为所有视图取样，保证视图之间互不影响且逐轮可复现。
        seeds = {view: rng.randrange(2**31) for view in views if view != "clean"}
        for view in views:
            cpu_batch = build_view(clean_cpu, view, seeds.get(view, 0), local_rate_range)
            batch = move_inputs(cpu_batch, device)
            if (view != "clean" and text_encoder is not None
                    and not torch.equal(cpu_batch["text_mask"], clean_cpu["text_mask"])):
                batch["teacher"] = encode_text(batch, text_encoder)
            elif (view != "clean" and "teacher" in batch
                  and not torch.equal(cpu_batch["text_mask"], clean_cpu["text_mask"])):
                batch["teacher"] = batch["teacher"].clone()
                batch["teacher"][~batch["text_mask"]] = 0.0
            output = model(*model_inputs(batch))
            gathered[f"{view}_logits"].append(output["logits"].cpu().numpy())
            gathered[f"{view}_sentiment"].append(output["sentiment"].cpu().numpy())
    classes = data["classes"].numpy()
    sentiment = data["sentiment"].numpy()
    return {view: metrics(classes, sentiment, np.concatenate(gathered[f"{view}_logits"]),
                          np.concatenate(gathered[f"{view}_sentiment"]))
            for view in views}


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    text_encoder=None,
    views: tuple[str, ...] = VIEWS,
    local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
    seed: int = 0,
    drop: tuple[str, ...] = (),
) -> dict[str, dict[str, np.ndarray]]:
    """逐样本预测，用于错误归因、混淆矩阵与可视化。"""
    model.eval()
    rng = random.Random(seed)
    gathered = {view: {"logits": [], "sentiment": []} for view in views}
    for start in range(0, len(data["tokens"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        clean_cpu = make_batch(data, ix)
        if drop:
            clean_cpu = drop_modalities(clean_cpu, drop)
        seeds = {view: rng.randrange(2**31) for view in views if view != "clean"}
        for view in views:
            cpu_batch = build_view(clean_cpu, view, seeds.get(view, 0), local_rate_range)
            batch = move_inputs(cpu_batch, device)
            if (view != "clean" and text_encoder is not None
                    and not torch.equal(cpu_batch["text_mask"], clean_cpu["text_mask"])):
                batch["teacher"] = encode_text(batch, text_encoder)
            output = model(*model_inputs(batch))
            gathered[view]["logits"].append(output["logits"].cpu().numpy())
            gathered[view]["sentiment"].append(output["sentiment"].cpu().numpy())
    return {view: {name: np.concatenate(parts) for name, parts in values.items()}
            for view, values in gathered.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "runs" / "main")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize-inputs", action="store_true",
                        help="fit audio/vision normalization on train valid positions only")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--corruption-probability", type=float, default=0.75)
    parser.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    parser.add_argument("--text-mode", choices=("tokens", "bert"), default="bert",
                        help="使用可训练词嵌入或题目提供的上下文 BERT 特征")
    parser.add_argument("--audio-dynamics", action="store_true",
                        help="加入相邻有效词位的声学变化特征")
    parser.add_argument("--regression-mode", choices=("signed", "soft", "hard"), default="soft",
                        help="signed 直接回归强度；soft 将强度与类别概率关联")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="类别权重指数，设为 0 可关闭")
    parser.add_argument("--whole-probability", type=float, default=0.3,
                        help="auto 模式下整段丢弃音视频（而非局部/区间缺失）的概率")
    parser.add_argument("--corruption-mode", choices=("auto", "whole", "local", "interval", "mixed"),
                        default="auto",
                        help="缺失形态：auto 混合三者；whole 整段；local 零散短游程；interval 连续区间")
    parser.add_argument("--interval-ratios", type=float, nargs="+", default=list(INTERVAL_RATIOS),
                        help="连续缺失区间占有效序列长度的候选比例")
    parser.add_argument("--overlap-probability", type=float, default=0.5,
                        help="多模态缺失区间部分重叠的概率")
    parser.add_argument("--interval-modalities", nargs="+", default=None,
                        choices=["text", "audio", "vision"],
                        help="限定连续区间族作用的模态；缺省按权重抽样（含文本）")
    parser.add_argument("--local-rate-min", type=float, default=LOCAL_RATE_RANGE[0],
                        help="局部缺失率下界，默认取自附件3实测分布")
    parser.add_argument("--local-rate-max", type=float, default=LOCAL_RATE_RANGE[1],
                        help="局部缺失率上界，默认取自附件3实测分布")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="baseline",
                        help="baseline 为门控融合；fuse 为 FUSE-Net 风格三因子分解")
    parser.add_argument("--drop-modalities", nargs="+", default=None,
                        choices=["text", "audio", "vision"],
                        help="永久遮蔽指定模态，用于单模态基线")
    parser.add_argument("--tau", type=float, default=0.1, help="对比分离的温标（仅 fuse）")
    parser.add_argument("--kl-beta", type=float, default=1e-2, help="变分信息瓶颈 KL 权重（仅 fuse）")
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--info-weight", type=float, default=0.1)
    parser.add_argument("--dual-weight", type=float, default=0.05)
    parser.add_argument("--mrc-weight", type=float, default=0.1)
    parser.add_argument("--cross-weight", type=float, default=0.1,
                        help="用可用模态重建缺失模态的交叉重建权重（仅 fuse）")
    args = parse_config_args(parser, "train")
    if (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0
            or not 0 <= args.dropout < 1 or args.class_weight_power < 0
            or not 0 <= args.whole_probability <= 1
            or not 0 <= args.overlap_probability <= 1
            or any(not 0 < ratio <= 1 for ratio in args.interval_ratios)
            or not 0 <= args.local_rate_min <= args.local_rate_max <= 1):
        parser.error("epochs, batch-size, patience and lr must be positive; dropout in [0, 1); "
                     "whole/overlap probability in [0, 1]; interval ratios in (0, 1]; "
                     "local rate range must satisfy 0 <= min <= max <= 1")

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    print("loading aligned_50.pkl train/valid; this file is about 1 GB")
    train_raw = load_main(args.data_root, "train")
    valid_raw = load_main(args.data_root, "valid")
    normalization = fit_input_normalization(train_raw) if args.normalize_inputs else None
    if normalization is not None:
        train_raw = apply_input_normalization(train_raw, normalization)
        valid_raw = apply_input_normalization(valid_raw, normalization)
    train_data = as_tensors(train_raw)
    valid_data = as_tensors(valid_raw)
    model = build_model(args).to(device)
    drop = tuple(args.drop_modalities or ())
    if drop:
        print(f"permanently dropping modalities: {', '.join(drop)}")
    text_encoder = load_text_encoder(device) if args.text_mode == "bert" else None
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** args.class_weight_power,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    best_score, best_epoch, stale = float("inf"), 0, 0
    started = time.time()

    reg_fields = [f"reg_{name}" for name in REG_FIELDS] if args.architecture == "fuse" else []
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"] + reg_fields +
                                [f"{view}_{m}" for view in VIEWS
                                 for m in ("accuracy", "macro_f1", "negative_f1", "neutral_f1",
                                           "positive_f1", "mae", "pearson")])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_data["tokens"]))
            total_loss, batches = 0.0, 0
            reg_totals = {name: 0.0 for name in REG_FIELDS}
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                if drop:
                    clean_cpu = drop_modalities(clean_cpu, drop)
                masked_cpu = mask_batch(
                    clean_cpu,
                    seed=args.seed + epoch * 100_003 + offset,
                    probability=args.corruption_probability,
                    pattern=args.corruption_mode,
                    ratios=tuple(args.interval_ratios),
                    interval_modalities=tuple(args.interval_modalities) if args.interval_modalities else None,
                    overlap_probability=args.overlap_probability,
                    whole_probability=args.whole_probability,
                    local_rate_range=(args.local_rate_min, args.local_rate_max),
                )
                clean = move_inputs(clean_cpu, device)
                masked = move_inputs(masked_cpu, device)
                if text_encoder is not None and not torch.equal(masked_cpu["text_mask"], clean_cpu["text_mask"]):
                    masked["teacher"] = encode_text(masked, text_encoder)
                optimizer.zero_grad(set_to_none=True)
                clean_out = model(*model_inputs(clean))
                masked_out = model(*model_inputs(masked))
                loss = (0.35 * supervised_loss(clean_out, clean, class_weights)
                        + 0.65 * supervised_loss(masked_out, masked, class_weights)
                        + 0.05 * consistency_loss(clean_out, masked_out))
                if args.text_mode == "tokens":
                    loss += args.distill_weight * distillation_loss(clean_out, clean)
                if args.architecture == "fuse":
                    reg = build_fuse_regularization(
                        model, masked_out, clean_out, masked["classes"], class_weights,
                        tau=args.tau, kl_beta=args.kl_beta, contrast_weight=args.contrast_weight,
                        info_weight=args.info_weight, dual_weight=args.dual_weight,
                        mrc_weight=args.mrc_weight, cross_weight=args.cross_weight)
                    loss = loss + reg["total"]
                    for name in REG_FIELDS:
                        reg_totals[name] += float(reg[name].detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1

            view_metrics = evaluate(
                model, valid_data, args.batch_size, device, args.seed + 9001, text_encoder,
                local_rate_range=(args.local_rate_min, args.local_rate_max), drop=drop)
            score = selection_score(view_metrics)
            row = {"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_score": score}
            reg_summary = ""
            if reg_fields:
                for name in REG_FIELDS:
                    row[f"reg_{name}"] = round(reg_totals[name] / max(1, batches), 5)
                reg_summary = " reg=[" + " ".join(f"{name}:{row[f'reg_{name}']:.3f}"
                                                  for name in REG_FIELDS) + "]"
            for view in VIEWS:
                row.update({f"{view}_{k}": v for k, v in view_metrics[view].items()})
            writer.writerow(row)
            f.flush()
            print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} score={score:.4f} "
                  f"clean_f1={view_metrics['clean']['macro_f1']:.4f} "
                  f"local_f1={view_metrics['local']['macro_f1']:.4f} "
                  f"whole_f1={view_metrics['whole']['macro_f1']:.4f} "
                  f"interval_f1={view_metrics['interval']['macro_f1']:.4f} "
                  f"clean_mae={view_metrics['clean']['mae']:.4f} "
                  f"local_mae={view_metrics['local']['mae']:.4f} "
                  f"whole_mae={view_metrics['whole']['mae']:.4f}"
                  f"{reg_summary}")

            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {"embedding_dim": 96, "hidden_dim": 64, **model_config(args)},
                    "val_views": view_metrics,
                    "val_score": score,
                    "input_normalization": normalization_summary(normalization),
                }, args.output_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    run = vars(args).copy()
    run["data_root"] = str(args.data_root) if args.data_root else "default"
    run["config"] = str(args.config) if args.config else None
    run["output_dir"] = str(args.output_dir)
    run.update({"device_used": str(device), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "best_epoch": best_epoch,
                "input_normalization": normalization_summary(normalization),
                "elapsed_seconds": round(time.time() - started, 2)})
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"saved {args.output_dir / 'best.pt'}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
