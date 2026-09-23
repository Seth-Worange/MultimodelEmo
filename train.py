"""Train and validate the local-missingness model on aligned_50.pkl."""

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

from augmentation import mask_batch
from data import load_main
from model import AffectiveModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in data.items()}


def make_batch(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in data.items()}


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def model_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (batch["tokens"][:, 0], batch["lengths"], batch["audio"], batch["vision"],
            batch["text_mask"], batch["audio_mask"], batch["vision_mask"], batch.get("teacher"))


def supervised_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                    class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(output["logits"], batch["classes"], weight=class_weights) + F.smooth_l1_loss(
        output["sentiment"], batch["sentiment"]
    )


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
        "mae": float(np.mean(np.abs(sentiment - predicted))),
        "pearson": float(corr),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    gathered: dict[str, list[np.ndarray]] = {f"{view}_{key}": [] for view in ("clean", "masked")
                                             for key in ("logits", "sentiment")}
    rng = random.Random(seed)
    for start in range(0, len(data["tokens"]), batch_size):
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        clean_cpu = make_batch(data, ix)
        masked_cpu = mask_batch(clean_cpu, seed=rng.randrange(2**31), probability=1.0)
        for view, cpu_batch in (("clean", clean_cpu), ("masked", masked_cpu)):
            batch = move_inputs(cpu_batch, device)
            output = model(*model_inputs(batch))
            gathered[f"{view}_logits"].append(output["logits"].cpu().numpy())
            gathered[f"{view}_sentiment"].append(output["sentiment"].cpu().numpy())
    classes = data["classes"].numpy()
    sentiment = data["sentiment"].numpy()
    clean = metrics(classes, sentiment, np.concatenate(gathered["clean_logits"]),
                    np.concatenate(gathered["clean_sentiment"]))
    masked = metrics(classes, sentiment, np.concatenate(gathered["masked_logits"]),
                     np.concatenate(gathered["masked_sentiment"]))
    return clean, masked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "runs" / "main")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--corruption-probability", type=float, default=0.75)
    parser.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    parser.add_argument("--text-mode", choices=("tokens", "bert"), default="bert",
                        help="use trainable token embeddings or the supplied contextual BERT vectors")
    parser.add_argument("--audio-dynamics", action="store_true",
                        help="include mask-safe adjacent changes in aligned acoustic features")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="inverse-frequency class weight exponent; 0 disables weighting")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.class_weight_power < 0:
        parser.error("epochs, batch-size, and patience must be positive; class-weight-power non-negative")

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    print("loading aligned_50.pkl train/valid; this file is about 1 GB")
    train_data = as_tensors(load_main(args.data_root, "train"))
    valid_data = as_tensors(load_main(args.data_root, "valid"))
    model = AffectiveModel(fusion=args.fusion, text_mode=args.text_mode,
                           audio_dynamics=args.audio_dynamics).to(device)
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** args.class_weight_power,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    best_score, best_epoch, stale = float("inf"), 0, 0
    started = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"] +
                                [f"clean_{m}" for m in ("accuracy", "macro_f1", "mae", "pearson")] +
                                [f"masked_{m}" for m in ("accuracy", "macro_f1", "mae", "pearson")])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_data["tokens"]))
            total_loss, batches = 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                masked_cpu = mask_batch(
                    clean_cpu,
                    seed=args.seed + epoch * 100_003 + offset,
                    probability=args.corruption_probability,
                )
                clean = move_inputs(clean_cpu, device)
                masked = move_inputs(masked_cpu, device)
                optimizer.zero_grad(set_to_none=True)
                clean_out = model(*model_inputs(clean))
                masked_out = model(*model_inputs(masked))
                loss = (0.35 * supervised_loss(clean_out, clean, class_weights)
                        + 0.65 * supervised_loss(masked_out, masked, class_weights)
                        + 0.05 * consistency_loss(clean_out, masked_out))
                if args.text_mode == "tokens":
                    loss += args.distill_weight * distillation_loss(clean_out, clean)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.detach())
                batches += 1

            clean_metrics, masked_metrics = evaluate(model, valid_data, args.batch_size, device, args.seed + 9001)
            score = 0.25 * (clean_metrics["mae"] + masked_metrics["mae"]) / 3.0 + \
                    0.25 * (2 - clean_metrics["macro_f1"] - masked_metrics["macro_f1"])
            row = {"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_score": score}
            row.update({f"clean_{k}": v for k, v in clean_metrics.items()})
            row.update({f"masked_{k}": v for k, v in masked_metrics.items()})
            writer.writerow(row)
            f.flush()
            print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} score={score:.4f} "
                  f"clean_f1={clean_metrics['macro_f1']:.4f} masked_f1={masked_metrics['macro_f1']:.4f} "
                  f"clean_mae={clean_metrics['mae']:.4f} masked_mae={masked_metrics['mae']:.4f}")

            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {"embedding_dim": 96, "hidden_dim": 64, "fusion": args.fusion,
                                     "text_mode": args.text_mode, "audio_dynamics": args.audio_dynamics},
                    "val_clean": clean_metrics,
                    "val_masked": masked_metrics,
                    "val_score": score,
                }, args.output_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    run = vars(args).copy()
    run["data_root"] = str(args.data_root) if args.data_root else "default"
    run["output_dir"] = str(args.output_dir)
    run.update({"device_used": str(device), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "best_epoch": best_epoch,
                "elapsed_seconds": round(time.time() - started, 2)})
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"saved {args.output_dir / 'best.pt'}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
