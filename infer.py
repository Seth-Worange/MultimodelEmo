"""Generate attachment 3 predictions or attachment 4 predictions with evidence."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from data import prepare_sample, read_pickle, resolve_data_root
from model import AffectiveModel

LABELS = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")


class ModelEnsemble(torch.nn.Module):
    def __init__(self, models: list[AffectiveModel]):
        super().__init__()
        if any(model.text_mode != models[0].text_mode for model in models):
            raise ValueError("Ensemble checkpoints must use the same text mode")
        self.text_mode = models[0].text_mode
        self.models = torch.nn.ModuleList(models)

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [model(*inputs) for model in self.models]
        logits = torch.stack([output["logits"] for output in outputs]).mean(dim=0)
        magnitude = torch.stack([output["magnitude"] for output in outputs]).mean(dim=0)
        sentiment = magnitude * (logits.argmax(dim=-1) - 1).to(magnitude.dtype)
        return {"logits": logits, "magnitude": magnitude, "sentiment": sentiment}


def load_model(path: Path | list[Path], device: torch.device) -> torch.nn.Module:
    paths = path if isinstance(path, list) else [path]
    if not paths:
        raise ValueError("At least one checkpoint is required")
    models = []
    for checkpoint_path in paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = AffectiveModel(**checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model"])
        models.append(model)
    ensemble = models[0] if len(models) == 1 else ModelEnsemble(models)
    return ensemble.to(device).eval()


def load_text_encoder(device: torch.device):
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "cache" / "huggingface"))
    os.environ.setdefault("USE_TF", "0")
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise RuntimeError("Install transformers to encode text for bert-mode checkpoints") from error
    return AutoModel.from_pretrained("google-bert/bert-base-uncased", local_files_only=True).to(device).eval()


def tensor_item(item: dict, device: torch.device) -> dict[str, torch.Tensor]:
    result = {}
    for key in ("tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask"):
        dtype = torch.long if key == "tokens" else (torch.bool if key.endswith("mask") else torch.float32)
        result[key] = torch.as_tensor(item[key], dtype=dtype).unsqueeze(0).to(device)
    if "teacher" in item:
        result["teacher"] = torch.as_tensor(item["teacher"], dtype=torch.float32).unsqueeze(0).to(device)
    result["lengths"] = torch.as_tensor([int(item["lengths"])], dtype=torch.long, device=device)
    return result


def run_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], masks: dict[str, torch.Tensor] | None = None):
    use_masks = masks or {name: batch[f"{name}_mask"] for name in MODALITIES}
    return model(batch["tokens"][:, 0], batch["lengths"], batch["audio"], batch["vision"],
                 use_masks["text"], use_masks["audio"], use_masks["vision"], batch.get("teacher"))


def margin(logits: torch.Tensor, label: int) -> float:
    other = torch.cat((logits[:, :label], logits[:, label + 1:]), dim=1).max(dim=1).values
    return float((logits[:, label] - other).item())


@torch.inference_mode()
def explain_one(model: torch.nn.Module, batch: dict[str, torch.Tensor], window: int = 5, stride: int = 2):
    base_masks = {name: batch[f"{name}_mask"].clone() for name in MODALITIES}
    full = run_model(model, batch, base_masks)
    predicted = int(full["logits"].argmax(dim=1).item())
    base_margin = margin(full["logits"], predicted)
    values: dict[tuple[str, ...], tuple[float, float]] = {}

    # Eight forward passes give exact Shapley values for three modalities.
    for count in range(4):
        for subset in itertools.combinations(MODALITIES, count):
            masks = {name: base_masks[name] if name in subset else torch.zeros_like(base_masks[name])
                     for name in MODALITIES}
            output = run_model(model, batch, masks)
            values[subset] = (margin(output["logits"], predicted), float(output["sentiment"].item()))

    phi_class, phi_regression = {}, {}
    for modality in MODALITIES:
        others = [name for name in MODALITIES if name != modality]
        phi_class[modality] = phi_regression[modality] = 0.0
        for count in range(3):
            for subset in itertools.combinations(others, count):
                weight = math.factorial(count) * math.factorial(2 - count) / math.factorial(3)
                plus = tuple(sorted((*subset, modality), key=MODALITIES.index))
                before = values[tuple(subset)]
                after = values[plus]
                phi_class[modality] += weight * (after[0] - before[0])
                phi_regression[modality] += weight * (after[1] - before[1])

    full_value, empty_value = values[MODALITIES], values[()]
    if not (math.isclose(sum(phi_class.values()), full_value[0] - empty_value[0], abs_tol=1e-5)
            and math.isclose(sum(phi_regression.values()), full_value[1] - empty_value[1], abs_tol=1e-5)):
        raise RuntimeError("Shapley contributions do not sum to the full-minus-empty prediction")

    positive = {key: max(0.0, value) for key, value in phi_class.items()}
    total_positive = sum(positive.values())
    class_share = {key: (positive[key] / total_positive if total_positive else 0.0) for key in MODALITIES}
    main_modality = max(class_share, key=class_share.get) if total_positive else "none"

    evidence = []
    for modality in MODALITIES:
        original_mask = base_masks[modality]
        for start in range(0, original_mask.shape[1], stride):
            end = min(original_mask.shape[1], start + window)
            local = original_mask[:, start:end]
            if not local.any():
                continue
            masks = {name: base_masks[name].clone() for name in MODALITIES}
            drops = masks[modality][:, start:end].clone()
            masks[modality][:, start:end] = False
            ablated = {key: value for key, value in batch.items()}
            if modality == "text":
                ablated["tokens"] = batch["tokens"].clone()
                ablated["tokens"][:, 0, start:end] = 0
            else:
                ablated[modality] = batch[modality].clone()
                ablated[modality][:, start:end] = 0
            output = run_model(model, ablated, masks)
            evidence.append({
                "modality": modality,
                "position_start": start,
                "position_end_exclusive": end,
                "valid_positions": int(drops.sum().item()),
                "class_margin_drop": base_margin - margin(output["logits"], predicted),
                "sentiment_change": float(full["sentiment"].item() - output["sentiment"].item()),
            })
    evidence.sort(key=lambda row: (row["modality"], row["position_start"]))
    return full, predicted, phi_class, phi_regression, class_share, main_modality, evidence


def find_files(root: Path, part: str, override: Path | None) -> list[Path]:
    if override:
        files = sorted(override.glob("*.pkl"))
    elif part == "q2":
        folder = next(root.glob("附件3-模态缺失特征样本/对齐版本"), None)
        files = sorted(folder.glob("*.pkl")) if folder else []
    else:
        folders = list(root.glob("附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"))
        files = sorted(folders[0].glob("*.pkl")) if folders else []
    if not files:
        raise FileNotFoundError(f"No {part} test .pkl files found")
    return files


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("q2", "q3"), required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True,
                        help="repeat to average multiple trained checkpoints")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs" / "predictions")
    parser.add_argument("--alignment-file", type=Path, default=None,
                        help="JSON from align_q3.py; required for video-time evidence")
    parser.add_argument("--max-samples", type=int, default=0, help="0 processes every matching file")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = resolve_data_root(args.data_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device)
    text_encoder = None
    files = find_files(root, args.part, args.input_dir)
    if args.max_samples < 0:
        parser.error("max-samples must be non-negative")
    if args.max_samples:
        files = files[:args.max_samples]
    alignment = json.loads(args.alignment_file.read_text(encoding="utf-8")) if args.alignment_file else {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions, explanations = [], []

    for path in files:
        raw = read_pickle(path)
        item = prepare_sample(raw)
        sample_id = item["id"] or path.stem
        batch = tensor_item(item, device)
        if model.text_mode == "bert" and "teacher" not in batch:
            if text_encoder is None:
                text_encoder = load_text_encoder(device)
            tokens = batch["tokens"]
            with torch.inference_mode():
                batch["teacher"] = text_encoder(input_ids=tokens[:, 0], attention_mask=tokens[:, 1],
                                                token_type_ids=tokens[:, 2]).last_hidden_state
        with torch.inference_mode():
            output = run_model(model, batch)
        probabilities = torch.softmax(output["logits"], dim=1)[0].cpu().numpy()
        predicted = int(probabilities.argmax())
        row = {
            "id": sample_id,
            "polarity": LABELS[predicted],
            "class_id": predicted,
            "sentiment_strength": float(output["sentiment"].item()),
            "p_negative": float(probabilities[0]),
            "p_neutral": float(probabilities[1]),
            "p_positive": float(probabilities[2]),
            "raw_text": item["raw_text"],
        }
        print(f"{args.part}: {sample_id} -> {row['polarity']}, {row['sentiment_strength']:.3f}")

        if args.part == "q3":
            full, predicted, phi_cls, phi_reg, share, main_modality, windows = explain_one(model, batch)
            row.update({f"shapley_class_{m}": phi_cls[m] for m in MODALITIES})
            row.update({f"class_share_{m}": share[m] for m in MODALITIES})
            row.update({f"shapley_strength_{m}": phi_reg[m] for m in MODALITIES})
            row["main_modality"] = main_modality
            item_alignment = alignment.get(sample_id, {})
            positions = item_alignment.get("positions", [])
            row["evidence_time_status"] = item_alignment.get("status", "position_only")
            for rank, window_row in enumerate(sorted(windows, key=lambda x: x["class_margin_drop"], reverse=True), 1):
                mapped = [positions[i] for i in range(window_row["position_start"],
                                                       min(window_row["position_end_exclusive"], len(positions)))
                          if positions[i] is not None] if positions else []
                timed = item_alignment.get("token_match_fraction", 0.0) >= 0.9 and bool(mapped)
                evidence = {"id": sample_id, "rank_global": rank, **window_row,
                            "evidence_words": " ".join(dict.fromkeys(p["word"] for p in mapped)),
                            "start_seconds": min((p["start"] for p in mapped), default=""),
                            "end_seconds": max((p["end"] for p in mapped), default=""),
                            "alignment_status": "mapped" if timed else item_alignment.get("status", "position_only")}
                explanations.append(evidence)
        predictions.append(row)

    write_csv(args.output_dir / f"{args.part}_predictions.csv", predictions)
    if args.part == "q3":
        write_csv(args.output_dir / "q3_evidence_windows.csv", explanations)
    print(f"wrote {len(predictions)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
