"""Train an independent full-modality Teacher on the clean training split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import AffectiveModel
from utils.config import parse_config_args
from utils.data import load_main
from scripts.train import (
    as_tensors,
    make_batch,
    metrics,
    model_inputs,
    move_inputs,
    seed_everything,
    supervised_loss,
)


def evaluate_clean(model: torch.nn.Module, data: dict[str, torch.Tensor],
                   batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    logits, sentiment = [], []
    with torch.no_grad():
        for start in range(0, len(data["tokens"]), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(data["tokens"])))
            batch = move_inputs(make_batch(data, indices), device)
            output = model(*model_inputs(batch))
            logits.append(output["logits"].cpu().numpy())
            sentiment.append(output["sentiment"].cpu().numpy())
    return metrics(data["classes"].numpy(), data["sentiment"].numpy(),
                   np.concatenate(logits), np.concatenate(sentiment))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "teacher")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    parser.add_argument("--text-mode", choices=("tokens", "bert"), default="bert")
    parser.add_argument("--audio-dynamics", action="store_true")
    parser.add_argument("--regression-mode", choices=("signed", "soft", "hard"), default="soft")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    args = parse_config_args(parser, "teacher")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        parser.error("epochs, batch-size, patience and lr must be positive")
    if not 0 <= args.dropout < 1 or args.class_weight_power < 0:
        parser.error("dropout must be in [0, 1) and class-weight-power non-negative")

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this environment")

    train_data = as_tensors(load_main(args.data_root, "train", need_teacher=args.text_mode == "bert"))
    valid_data = as_tensors(load_main(args.data_root, "valid", need_teacher=args.text_mode == "bert"))
    model = AffectiveModel(fusion=args.fusion, text_mode=args.text_mode,
                           audio_dynamics=args.audio_dynamics,
                           regression_mode=args.regression_mode,
                           dropout=args.dropout).to(device)
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** args.class_weight_power,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = float("inf"), 0, 0
    started = time.time()
    metrics_path = args.output_dir / "metrics.csv"

    import csv
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_score",
                                                     "val_accuracy", "val_macro_f1",
                                                     "val_mae", "val_pearson"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_data["tokens"]))
            total, batches = 0.0, 0
            for offset in range(0, len(order), args.batch_size):
                indices = order[offset:offset + args.batch_size]
                batch = move_inputs(make_batch(train_data, indices), device)
                optimizer.zero_grad(set_to_none=True)
                loss = supervised_loss(model(*model_inputs(batch)), batch, class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                batches += 1
            validation = evaluate_clean(model, valid_data, args.batch_size, device)
            score = validation["mae"] / 3.0 + (1.0 - validation["macro_f1"])
            row = {"epoch": epoch, "train_loss": total / max(1, batches), "val_score": score,
                   **{f"val_{key}": value for key, value in validation.items()}}
            writer.writerow(row)
            handle.flush()
            print(f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
                  f"val_f1={validation['macro_f1']:.4f} val_mae={validation['mae']:.4f}")
            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {"embedding_dim": 96, "hidden_dim": 64,
                                     "fusion": args.fusion, "text_mode": args.text_mode,
                                     "audio_dynamics": args.audio_dynamics,
                                     "regression_mode": args.regression_mode,
                                     "dropout": args.dropout},
                    "val_clean": validation,
                    "val_score": score,
                    "teacher": True,
                }, args.output_dir / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                break

    run = vars(args).copy()
    run.update({"device_used": str(device), "best_epoch": best_epoch,
                "elapsed_seconds": round(time.time() - started, 2)})
    run["data_root"] = str(args.data_root) if args.data_root else "default"
    run["output_dir"] = str(args.output_dir)
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
