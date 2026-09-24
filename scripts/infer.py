"""生成附件3预测或附件4预测与解释。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from utils.data import prepare_sample, read_pickle, resolve_data_root
from utils.config import parse_config_args
from utils.alignment import alignment_coverage, mapped_word_text
from utils.text import (DEFAULT_BERT, DEFAULT_BERT_REVISION, encode_text,
                        load_text_encoder, load_text_tokenizer,
                        mask_full_text_attention, prepare_full_text_inputs)
from model import AffectiveModel
from model.fuse_net import FactorizedAffectiveModel
from model.cica_net import CICAAffectiveModel
from model.input_processing import fill_legacy_input_buffers

LABELS = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")


class ModelEnsemble(torch.nn.Module):
    def __init__(self, models: list[AffectiveModel]):
        super().__init__()
        if any(model.text_mode != models[0].text_mode for model in models):
            raise ValueError("Ensemble checkpoints must use the same text mode")
        if any(model.regression_mode != models[0].regression_mode for model in models):
            raise ValueError("Ensemble checkpoints must use the same regression mode")
        for name in ("bert_finetune", "bert_model_name", "bert_model_revision", "bert_max_length"):
            if any(getattr(model, name, None) != getattr(models[0], name, None) for model in models):
                raise ValueError("Ensemble checkpoints must use the same BERT configuration")
        self.text_mode = models[0].text_mode
        self.regression_mode = models[0].regression_mode
        self.bert_finetune = getattr(models[0], "bert_finetune", False)
        self.bert_model_name = getattr(models[0], "bert_model_name", DEFAULT_BERT)
        self.bert_model_revision = getattr(models[0], "bert_model_revision", DEFAULT_BERT_REVISION)
        self.bert_max_length = getattr(models[0], "bert_max_length", 512)
        self.models = torch.nn.ModuleList(models)

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = [model(*inputs) for model in self.models]
        logits = torch.stack([output["logits"] for output in outputs]).mean(dim=0)
        magnitude = torch.stack([output["magnitude"] for output in outputs]).mean(dim=0)
        if self.regression_mode == "signed":
            sentiment = torch.stack([output["sentiment"] for output in outputs]).mean(dim=0)
            magnitude = sentiment.abs()
        elif self.regression_mode == "soft":
            probabilities = logits.softmax(dim=-1)
            sentiment = magnitude * (probabilities[:, 2] - probabilities[:, 0])
        else:
            sentiment = magnitude * (logits.argmax(dim=-1) - 1).to(magnitude.dtype)
        result = {"logits": logits, "magnitude": magnitude, "sentiment": sentiment}
        for key in ("confidence", "uncertainty", "reliability", "modality_weights", "time_weights"):
            if all(key in output for output in outputs):
                result[key] = torch.stack([output[key] for output in outputs]).mean(dim=0)
        return result


def build_from_config(config: dict | None) -> torch.nn.Module:
    """按检查点里记录的结构配置还原模型（缺省为门控基线）。"""
    config = dict(config or {})
    architecture = config.pop("architecture", "baseline")
    if architecture == "fuse":
        return FactorizedAffectiveModel(**config)
    if architecture == "cica":
        return CICAAffectiveModel(**config)
    config.pop("tau", None)
    return AffectiveModel(**config)


def load_checkpoint_state(model: torch.nn.Module, state: dict[str, torch.Tensor],
                          compact_bert: bool) -> None:
    """还原完整或仅含可训练BERT层的检查点。"""
    state = fill_legacy_input_buffers(model, state)
    if not compact_bert:
        model.load_state_dict(state)
        return
    loaded = model.load_state_dict(state, strict=False)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    prefix = "bert_text_encoder.backbone."
    expected_missing = {name for name in model.state_dict()
                        if name.startswith(prefix) and name not in trainable}
    if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
        raise RuntimeError("Compact BERT checkpoint does not match its base model configuration")


class NeutralZeroModel(torch.nn.Module):
    """可选决策规则：预测类别为中性时，输出强度为零。"""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.text_mode = model.text_mode
        self.regression_mode = model.regression_mode
        self.bert_finetune = getattr(model, "bert_finetune", False)
        self.bert_model_name = getattr(model, "bert_model_name", DEFAULT_BERT)
        self.bert_model_revision = getattr(model, "bert_model_revision", DEFAULT_BERT_REVISION)
        self.bert_max_length = getattr(model, "bert_max_length", 512)

    def forward(self, *inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        output = dict(self.model(*inputs))
        raw = output["sentiment"]
        output["raw_sentiment"] = raw
        output["sentiment"] = torch.where(output["logits"].argmax(-1) == 1, 0.0, raw)
        return output


def load_model(path: Path | list[Path], device: torch.device,
               neutral_zero: bool = False) -> torch.nn.Module:
    paths = path if isinstance(path, list) else [path]
    if not paths:
        raise ValueError("At least one checkpoint is required")
    models = []
    for checkpoint_path in paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_from_config(checkpoint.get("model_config", {}))
        state = checkpoint["model"]
        if getattr(model, "fusion", None) == "gate" and "availability_embedding.weight" not in state:
            # 旧权重用零嵌入保持原有预测行为。
            state = {**state, "availability_embedding.weight": model.availability_embedding.weight}
        load_checkpoint_state(model, state, checkpoint.get("compact_bert_state", False))
        models.append(model)
    ensemble = models[0] if len(models) == 1 else ModelEnsemble(models)
    if neutral_zero:
        ensemble = NeutralZeroModel(ensemble)
    return ensemble.to(device).eval()


def checkpoint_metadata(paths: list[Path]) -> list[dict]:
    """记录实际权重与结构，避免结果文件对应错实验。"""
    return [{"path": str(path.resolve()),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "model_config": torch.load(path, map_location="cpu", weights_only=True).get("model_config", {})}
            for path in paths]


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
    inputs = (batch["tokens"][:, 0], batch["lengths"], batch["audio"], batch["vision"],
              use_masks["text"], use_masks["audio"], use_masks["vision"], batch.get("teacher"))
    if getattr(model, "bert_finetune", False):
        attention = mask_full_text_attention(batch["bert_attention_mask"], use_masks["text"])
        inputs += (batch["bert_input_ids"], attention, batch.get("bert_token_type_ids"))
    return model(*inputs)


def margin(logits: torch.Tensor, label: int) -> float:
    other = torch.cat((logits[:, :label], logits[:, label + 1:]), dim=1).max(dim=1).values
    return float((logits[:, label] - other).item())


@torch.inference_mode()
def explain_one(model: torch.nn.Module, batch: dict[str, torch.Tensor], window: int = 5,
                stride: int = 2, text_encoder=None):
    base_masks = {name: batch[f"{name}_mask"].clone() for name in MODALITIES}
    full = run_model(model, batch, base_masks)
    predicted = int(full["logits"].argmax(dim=1).item())
    base_margin = margin(full["logits"], predicted)
    values: dict[tuple[str, ...], tuple[float, float]] = {}

    # 三模态共8个子集，计算精确 Shapley 值。
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
                if model.text_mode == "bert" and not getattr(model, "bert_finetune", False):
                    if text_encoder is None:
                        raise ValueError("Text occlusion requires the frozen BERT encoder")
                    ablated["teacher"] = encode_text(ablated, text_encoder, masks["text"])
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
    parser.add_argument("--neutral-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--part", choices=("q2", "q3"))
    parser.add_argument("--checkpoint", type=Path, action="append",
                        help="可重复指定多个检查点并集成")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "predictions")
    parser.add_argument("--alignment-file", type=Path, default=None,
                        help="align_q3.py 生成的词时间映射")
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示处理全部样本")
    parser.add_argument("--device", default="auto")
    args = parse_config_args(parser, "infer")
    if args.part is None or not args.checkpoint:
        parser.error("provide --part and --checkpoint or set them in the config")
    root = resolve_data_root(args.data_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = load_model(args.checkpoint, device, neutral_zero=args.neutral_zero)
    bert_finetune = bool(getattr(model, "bert_finetune", False))
    bert_model_name = getattr(model, "bert_model_name", DEFAULT_BERT)
    bert_revision = getattr(model, "bert_model_revision", DEFAULT_BERT_REVISION)
    bert_max_length = getattr(model, "bert_max_length", 512)
    tokenizer = (load_text_tokenizer(bert_model_name, bert_revision)
                 if bert_finetune else None)
    text_encoder = (load_text_encoder(device, bert_model_name, bert_revision)
                    if model.text_mode == "bert" and args.part == "q3" and not bert_finetune else None)
    files = find_files(root, args.part, args.input_dir)
    if args.max_samples < 0:
        parser.error("max-samples must be non-negative")
    if args.max_samples:
        files = files[:args.max_samples]
    alignment = {}
    if args.alignment_file:
        if not args.alignment_file.is_file():
            raise FileNotFoundError(
                f"alignment file not found: {args.alignment_file}; run `python -m scripts.align_q3` first")
        alignment = json.loads(args.alignment_file.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"part": args.part, "neutral_zero": args.neutral_zero,
                "checkpoints": checkpoint_metadata(args.checkpoint)}
    (args.output_dir / f"{args.part}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions, explanations = [], []

    for path in files:
        raw = read_pickle(path)
        item = prepare_sample(raw)
        sample_id = item["id"] or path.stem
        batch = tensor_item(item, device)
        if bert_finetune:
            prepared = prepare_full_text_inputs(
                {"tokens": batch["tokens"].cpu().numpy(), "raw_text": [item["raw_text"]],
                 "ids": [sample_id]}, tokenizer, bert_max_length)
            for key in ("bert_input_ids", "bert_attention_mask", "bert_token_type_ids"):
                batch[key] = torch.as_tensor(prepared[key], dtype=torch.long, device=device)
        elif model.text_mode == "bert" and "teacher" not in batch:
            if text_encoder is None:
                text_encoder = load_text_encoder(device, bert_model_name, bert_revision)
            tokens = batch["tokens"]
            batch["teacher"] = encode_text(batch, text_encoder)
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
            "available_modalities": ",".join(m for m in MODALITIES if item[f"{m}_mask"].any()),
            "missing_modalities": ",".join(m for m in MODALITIES if not item[f"{m}_mask"].any()),
        }
        if "modality_weights" in output:
            importance = (output["modality_weights"]
                          * output["time_weights"].unsqueeze(-1)).sum(dim=1)[0].cpu().numpy()
            row.update({f"fusion_weight_{name}": float(importance[index])
                        for index, name in enumerate(MODALITIES)})
        for key in ("confidence", "uncertainty", "reliability"):
            if key in output:
                row.update({f"{key}_{name}": float(output[key][0, index].item())
                            for index, name in enumerate(MODALITIES)})
        print(f"{args.part}: {sample_id} -> {row['polarity']}, {row['sentiment_strength']:.3f}")

        if args.part == "q3":
            full, predicted, phi_cls, phi_reg, share, main_modality, windows = explain_one(
                model, batch, text_encoder=text_encoder)
            row.update({f"shapley_class_{m}": phi_cls[m] for m in MODALITIES})
            row.update({f"class_share_{m}": share[m] for m in MODALITIES})
            row.update({f"shapley_strength_{m}": phi_reg[m] for m in MODALITIES})
            row["main_modality"] = main_modality
            item_alignment = alignment.get(sample_id, {})
            row.update(alignment_coverage(item_alignment))
            positions = item_alignment.get("positions", [])
            row["evidence_time_status"] = item_alignment.get("status", "position_only")
            for rank, window_row in enumerate(sorted(windows, key=lambda x: x["class_margin_drop"], reverse=True), 1):
                mapped = [positions[i] for i in range(window_row["position_start"],
                                                       min(window_row["position_end_exclusive"], len(positions)))
                          if positions[i] is not None] if positions else []
                timed = item_alignment.get("token_match_fraction", 0.0) >= 0.9 and bool(mapped)
                evidence = {"id": sample_id, "rank_global": rank, **window_row,
                            "evidence_words": mapped_word_text(mapped) if timed else "",
                            "evidence_scope": row["evidence_scope"],
                            "start_seconds": min(p["start"] for p in mapped) if timed else "",
                            "end_seconds": max(p["end"] for p in mapped) if timed else "",
                            "alignment_status": "mapped" if timed else item_alignment.get("status", "position_only")}
                explanations.append(evidence)
        predictions.append(row)

    write_csv(args.output_dir / f"{args.part}_predictions.csv", predictions)
    if args.part == "q3":
        write_csv(args.output_dir / "q3_evidence_windows.csv", explanations)
    print(f"wrote {len(predictions)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
