"""按语义上下文和条件充分性筛选附件4局部证据。"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from scripts.infer import MODALITIES, find_files, load_model, margin, run_model, tensor_item, write_csv
from utils.config import parse_config_args
from utils.data import prepare_sample, read_pickle, resolve_data_root
from utils.q3_evidence import context_positions, mapped_event
from utils.text import encode_text, load_text_encoder


def score_mask(model, batch: dict, encoder, name: str, keep: torch.Tensor,
               mode: str = "early") -> dict:
    """其余模态保持原状；文本比较重编码和完整上下文特征后遮蔽。"""
    masks = {key: batch[f"{key}_mask"].clone() for key in MODALITIES}
    masks[name] &= keep
    changed = dict(batch)
    if name == "text":
        if mode == "early":
            changed["tokens"] = batch["tokens"].clone()
            token_ids = changed["tokens"][:, 0]
            special = (token_ids == 101) | (token_ids == 102)
            token_ids[~masks[name] & ~special] = 0
            changed["teacher"] = encode_text(changed, encoder, masks[name])
        else:
            changed["teacher"] = batch["teacher"].clone()
            changed["teacher"][:, ~masks[name][0]] = 0
    else:
        changed[name] = batch[name].clone()
        changed[name][:, ~masks[name][0]] = 0
    return run_model(model, changed, masks)


def window_masks(batch: dict, row: dict, mapping: dict) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    name = row["modality"]
    start, end = int(row["position_start"]), int(row["position_end_exclusive"])
    original = batch[f"{name}_mask"]
    core = torch.zeros_like(original)
    core[:, start:end] = original[:, start:end]
    context = torch.zeros_like(original)
    if name == "text":
        context_ids = context_positions(mapping, start, end)
    else:
        width = end - start
        context_ids = list(range(max(0, start - width), start))
        context_ids += list(range(end, min(original.shape[1], end + width)))
    context_ids = [i for i in context_ids if i < original.shape[1] and not core[0, i]]
    context[:, context_ids] = original[:, context_ids]
    return core, context, context_ids


def evaluate_window(model, batch: dict, encoder, predicted: int,
                    full_margin: float, row: dict, mapping: dict,
                    verify_text: bool = False) -> dict:
    name = row["modality"]
    core, context, context_ids = window_masks(batch, row, mapping)
    original = batch[f"{name}_mask"]
    deleted = score_mask(model, batch, encoder, name, original & ~core)
    with_core = score_mask(model, batch, encoder, name, context | core)
    without_core = score_mask(model, batch, encoder, name, context)
    deletion = full_margin - margin(deleted["logits"], predicted)
    gain = margin(with_core["logits"], predicted) - margin(without_core["logits"], predicted)
    result = {"class_margin_drop": deletion, "conditional_gain": gain,
              "context_positions": context_ids, "context_start": min(context_ids, default=""),
              "context_end_exclusive": max(context_ids, default=-1) + 1 if context_ids else ""}
    if name == "text" and verify_text:
        late_deleted = score_mask(model, batch, encoder, name, original & ~core, "late")
        late_with = score_mask(model, batch, encoder, name, context | core, "late")
        late_without = score_mask(model, batch, encoder, name, context, "late")
        late_drop = full_margin - margin(late_deleted["logits"], predicted)
        late_gain = margin(late_with["logits"], predicted) - margin(late_without["logits"], predicted)
        result.update({"late_deletion": late_drop, "late_conditional_gain": late_gain,
                       "text_mask_agreement": deletion > 0 and gain > 0
                       and late_drop > 0 and late_gain > 0})
    else:
        result.update({"late_deletion": "", "late_conditional_gain": "",
                       "text_mask_agreement": "not_checked" if name == "text" else "not_applicable"})
    return result


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def members(model) -> list:
    base = getattr(model, "model", model)
    return list(getattr(base, "models", [base]))


def member_main_modality(model, batch: dict, predicted: int) -> str:
    """对单个检查点用同一个三模态八子集规则求主支持模态。"""
    values = {}
    for count in range(4):
        for subset in itertools.combinations(MODALITIES, count):
            masks = {name: batch[f"{name}_mask"] if name in subset
                     else torch.zeros_like(batch[f"{name}_mask"]) for name in MODALITIES}
            values[subset] = margin(run_model(model, batch, masks)["logits"], predicted)
    contribution = {}
    for name in MODALITIES:
        others = [other for other in MODALITIES if other != name]
        contribution[name] = sum(
            math.factorial(len(subset)) * math.factorial(2 - len(subset)) / 6
            * (values[tuple(sorted((*subset, name), key=MODALITIES.index))] - values[subset])
            for count in range(3) for subset in itertools.combinations(others, count))
    return max(MODALITIES, key=contribution.get)


def overlap(first: dict, second: dict) -> float:
    a, b = int(first["position_start"]), int(first["position_end_exclusive"])
    c, d = int(second["position_start"]), int(second["position_end_exclusive"])
    return max(0, min(b, d) - max(a, c)) / max(1, min(b - a, d - c))


def checkpoint_stability(model, batch: dict, encoder, winner: dict,
                         candidates: list[dict], reference_class: int) -> dict:
    pool = members(model)
    if len(pool) == 1:
        return {"s_pred": 1.0, "s_mod": 1.0, "s_evidence": 1.0, "n_checkpoints": 1}
    pred_ok = mod_ok = evidence_ok = 0
    for member in pool:
        output = run_model(member, batch)
        own_class = int(output["logits"].argmax(-1).item())
        pred_ok += own_class == reference_class
        own_modality = member_main_modality(member, batch, own_class)
        mod_ok += own_modality == winner["modality"]
        own_margin = margin(output["logits"], own_class)
        same = [row for row in candidates if row["modality"] == own_modality]
        if same:
            best = max(same, key=lambda row: own_margin - margin(score_mask(
                member, batch, encoder, own_modality,
                batch[f"{own_modality}_mask"] & ~window_masks(batch, row, {})[0])["logits"], own_class))
            evidence_ok += own_modality == winner["modality"] and overlap(best, winner) >= 0.5
    n = len(pool)
    return {"s_pred": pred_ok / n, "s_mod": mod_ok / n,
            "s_evidence": evidence_ok / n, "n_checkpoints": n}


def randomized_head(model):
    """仅随机化预测头，保留输入编码以检查解释对预测参数的依赖。"""
    randomized = copy.deepcopy(model)
    changed = 0
    cuda_devices = sorted({parameter.device.index for parameter in randomized.parameters()
                           if parameter.is_cuda})
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(2026)
        if cuda_devices:
            torch.cuda.manual_seed_all(2026)
        for name, module in randomized.named_modules():
            if name.rsplit(".", 1)[-1] in {"classifier", "neutral_classifier", "polarity_classifier"} \
                    and isinstance(module, torch.nn.Linear):
                module.reset_parameters()
                changed += 1
    if not changed:
        raise ValueError("未找到可随机化的预测输出层")
    for module in randomized.modules():
        if isinstance(module, torch.nn.GRU):
            module.flatten_parameters()
    return randomized.eval()


def conditional_gain_only(model, batch: dict, encoder, predicted: int,
                          row: dict, mapping: dict) -> float:
    core, context, _ = window_masks(batch, row, mapping)
    name = row["modality"]
    with_core = score_mask(model, batch, encoder, name, context | core)
    without_core = score_mask(model, batch, encoder, name, context)
    return margin(with_core["logits"], predicted) - margin(without_core["logits"], predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--evidence-csv", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--alignment-file", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--neutral-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    args = parse_config_args(parser, "refine")
    if not args.checkpoint or not args.evidence_csv or not args.output_dir:
        parser.error("checkpoint, evidence-csv and output-dir are required")
    predictions_path = args.predictions_csv or args.evidence_csv.with_name("q3_predictions.csv")
    alignment_path = args.alignment_file or Path("outputs/runs/main/q3_alignment.json")
    predictions = {row["id"]: row for row in read_rows(predictions_path)}
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    windows = defaultdict(list)
    for row in read_rows(args.evidence_csv):
        windows[row["id"]].append(row)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, neutral_zero=args.neutral_zero)
    if model.text_mode != "bert" or getattr(model, "bert_finetune", False):
        raise ValueError("Early/Late 文本复核当前要求冻结 BERT 特征接口")
    encoder = load_text_encoder(device, model.bert_model_name, model.bert_model_revision)
    randomized = randomized_head(model)
    selected = []
    files = find_files(resolve_data_root(args.data_root), "q3", None)
    for path in tqdm(files[:args.max_samples] if args.max_samples else files,
                     desc="Q3证据复核", unit="样本"):
        item = prepare_sample(read_pickle(path))
        sample_id = item["id"] or path.stem
        if sample_id not in predictions or sample_id not in alignment:
            raise ValueError(f"缺少样本 {sample_id} 的预测或时间映射")
        pred, mapping = predictions[sample_id], alignment[sample_id]
        candidates = windows[sample_id]
        if not candidates:
            continue
        # 负贡献保留在预测表中，但不作为支持当前预测的证据窗口。
        supporters = [name for name in MODALITIES if float(pred[f"shapley_class_{name}"]) > 0]
        supporters.sort(key=lambda name: float(pred[f"shapley_class_{name}"]), reverse=True)
        name = next((name for name in supporters if any(r["modality"] == name for r in candidates)), None)
        if name is None:
            continue
        ranked = sorted((r for r in candidates if r["modality"] == name),
                        key=lambda r: float(r["class_margin_drop"]), reverse=True)
        batch = tensor_item(item, device)
        with torch.inference_mode():
            batch["teacher"] = encode_text(batch, encoder)
            full = run_model(model, batch)
            predicted = int(full["logits"].argmax(-1).item())
            if predicted != int(pred["class_id"]):
                raise ValueError(f"{sample_id}: 当前预测器与候选生成时类别不同")
            full_margin = margin(full["logits"], predicted)
            scored = []
            for row in ranked[:args.top_k]:
                values = evaluate_window(model, batch, encoder, predicted, full_margin, row, mapping)
                joint = min(max(values["class_margin_drop"], 0), max(values["conditional_gain"], 0))
                scored.append({**row, **{k: v for k, v in values.items() if k != "context_positions"},
                               "joint_evidence_score": joint, "_context_ids": values["context_positions"]})
        winner = max(scored, key=lambda r: r["joint_evidence_score"])
        if name == "text":
            with torch.inference_mode():
                audit = evaluate_window(model, batch, encoder, predicted, full_margin,
                                        winner, mapping, verify_text=True)
            winner.update({key: audit[key] for key in (
                "late_deletion", "late_conditional_gain", "text_mask_agreement")})
        start, end = int(winner["position_start"]), int(winner["position_end_exclusive"])
        context_ids = winner.pop("_context_ids")
        event_ids = [i for i in sorted(set(context_ids + list(range(start, end))))
                     if bool(batch[f"{name}_mask"][0, i])]
        alternatives = [row for row in ranked if overlap(row, winner) == 0]
        random.Random(int(sample_id.encode().hex()[:8], 16)).shuffle(alternatives)
        alternatives = alternatives[:19]
        with torch.inference_mode():
            random_gains = [conditional_gain_only(model, batch, encoder, predicted, row, mapping)
                            for row in alternatives]
            stability = checkpoint_stability(model, batch, encoder, winner, ranked[:args.top_k], predicted)
            random_full = margin(run_model(randomized, batch)["logits"], predicted)
            core, _, _ = window_masks(batch, winner, mapping)
            random_deleted = score_mask(randomized, batch, encoder, name,
                                        batch[f"{name}_mask"] & ~core)
            random_drop = random_full - margin(random_deleted["logits"], predicted)
            event_drops = {}
            for modality in MODALITIES:
                event_mask = torch.zeros_like(batch[f"{modality}_mask"])
                event_mask[:, event_ids] = batch[f"{modality}_mask"][:, event_ids]
                removed = score_mask(model, batch, encoder, modality,
                                     batch[f"{modality}_mask"] & ~event_mask)
                event_drops[f"event_drop_{modality}"] = full_margin - margin(
                    removed["logits"], predicted)
        empirical_p = ((1 + sum(value >= winner["conditional_gain"] for value in random_gains))
                       / (len(random_gains) + 1))
        random_pass = len(random_gains) >= 19 and empirical_p <= 0.05
        parameter_pass = abs(winner["class_margin_drop"]) > abs(random_drop)
        core_event = mapped_event(mapping, [i for i in range(start, end)
                                            if bool(batch[f"{name}_mask"][0, i])])
        full_event = mapped_event(mapping, event_ids)
        if name == "text":
            entries = mapping.get("positions", [])
            words = {entries[i]["word_index"]: entries[i]["word"] for i in context_ids
                     if i < len(entries) and entries[i] is not None}
            context_text = " ".join(words[i] for i in sorted(words))
            core_text = winner["evidence_words"]
        else:
            core_text = "语音声学片段" if name == "audio" else "视觉表情片段"
            context_text = f"同期转写：{winner['evidence_words']}" if winner["evidence_words"] else "同期转写未映射"
        selected.append({**winner, "core_start_seconds": core_event["start_seconds"],
                         "core_end_seconds": core_event["end_seconds"],
                         "start_seconds": full_event["start_seconds"],
                         "end_seconds": full_event["end_seconds"],
                         "mapping_status": full_event["mapping_status"],
                         "core_evidence": core_text, "semantic_context": context_text,
                         "random_window_p": empirical_p, "random_window_count": len(random_gains),
                         "randomized_head_drop": random_drop,
                         **event_drops,
                         "random_control_pass": random_pass,
                         "parameter_control_pass": parameter_pass,
                         **stability,
                         "selection_status": "confirmed" if winner["joint_evidence_score"] > 0
                         and winner["text_mask_agreement"] is not False
                         and random_pass and parameter_pass
                         and full_event["mapping_status"] != "unmapped" else "unconfirmed"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "q3_selected_evidence.csv", selected)
    summary = {"n_samples": len(selected), "confirmed": sum(
        row["selection_status"] == "confirmed" for row in selected),
        "method": "signed Shapley + deletion + conditional gain + text Early/Late",
        "random_control": "same-modality nonoverlapping windows; empirical rank threshold 0.05",
        "parameter_control": "one randomized prediction-head replicate; descriptive only"}
    (args.output_dir / "q3_selected_evidence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
