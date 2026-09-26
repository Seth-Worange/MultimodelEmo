"""在分层验证样本上复核局部证据，区别预测正确性与解释忠实性。"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from scripts.infer import explain_one, load_model, margin, run_model, write_csv
from scripts.refine_q3_evidence import score_mask, randomized_head
from scripts.train import as_tensors, make_batch, move_inputs
from utils.data import load_main
from utils.text import load_text_encoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=24)
    args = parser.parse_args()
    torch.manual_seed(2026)
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    randomized = randomized_head(model)
    encoder = load_text_encoder(device)
    data = as_tensors(load_main(Path("data"), "valid", need_teacher=True))
    rng = np.random.default_rng(2026)
    indices = np.concatenate([rng.choice(np.where(data["classes"].numpy() == label)[0],
                  min(args.per_class, int((data["classes"] == label).sum())), replace=False)
                  for label in range(3)])
    rows = []
    for index in tqdm(indices, desc="Validation explanation audit"):
        batch = move_inputs(make_batch(data, torch.tensor([int(index)])), device)
        full, predicted, phi, _, _, _, candidates = explain_one(
            model, batch, window=5, stride=5, text_encoder=encoder)
        winner = max(candidates, key=lambda row: row["class_margin_drop"])
        name = winner["modality"]
        start, end = winner["position_start"], winner["position_end_exclusive"]
        original = batch[f"{name}_mask"]
        core = torch.zeros_like(original)
        core[:, start:end] = original[:, start:end]
        context = torch.zeros_like(original)
        context[:, max(0, start - 5):min(50, end + 5)] = original[:, max(0, start - 5):min(50, end + 5)]
        context &= ~core
        gain = margin(score_mask(model, batch, encoder, name, context | core)["logits"], predicted)
        gain -= margin(score_mask(model, batch, encoder, name, context)["logits"], predicted)
        late_drop = margin(full["logits"], predicted) - margin(
            score_mask(model, batch, encoder, name, original & ~core, "late")["logits"], predicted)
        alternatives = [row for row in candidates if row["modality"] == name
                        and (row["position_end_exclusive"] <= start or row["position_start"] >= end)
                        and row["valid_positions"] == winner["valid_positions"]]
        other = random.Random(2026 + int(index)).choice(alternatives) if alternatives else None
        random_drop = other["class_margin_drop"] if other else None
        random_full = margin(run_model(randomized, batch)["logits"], predicted)
        random_deleted = margin(score_mask(randomized, batch, encoder, name, original & ~core)["logits"], predicted)
        rows.append({"id": str(batch["ids"][0]), "label": int(data["classes"][index]),
            "prediction": predicted, "correct": predicted == int(data["classes"][index]),
            "modality": name, "start": start, "end": end,
            "deletion": winner["class_margin_drop"], "conditional_gain": gain,
            "late_deletion": late_drop, "early_late_agreement": winner["class_margin_drop"] * late_drop > 0,
            "random_window_deletion": random_drop, "randomized_head_deletion": random_full - random_deleted,
            **{f"signed_shapley_{key}": value for key, value in phi.items()}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "validation_explanations.csv", rows)
    paired = [row for row in rows if row["random_window_deletion"] is not None]
    summary = {"n": len(rows), "seed": 2026, "sampling": "up to 24 per true class without replacement",
        "positive_deletion_fraction": float(np.mean([row["deletion"] > 0 for row in rows])),
        "positive_conditional_fraction": float(np.mean([row["conditional_gain"] > 0 for row in rows])),
        "early_late_agreement": float(np.mean([row["early_late_agreement"] for row in rows if row["modality"] == "text"])),
        "random_paired_n": len(paired),
        "mean_selected_minus_random_deletion": float(np.mean([row["deletion"] - row["random_window_deletion"] for row in paired])),
        "limitations": ["候选按删除分数选择，优于随机存在选择优势，不是独立显著性证明",
                         "上下文使用邻近五位，附件4另做语义闭包复核",
                         "Shapley基于固定上下文特征子集；文本局部删除另做重编码",
                         "参数随机化只有一次，属于描述性检查；无人工证据标注"]}
    for correct in (True, False):
        group = [row for row in rows if row["correct"] == correct]
        summary["correct" if correct else "incorrect"] = {"n": len(group),
            "mean_deletion": float(np.mean([row["deletion"] for row in group])),
            "mean_conditional_gain": float(np.mean([row["conditional_gain"] for row in group]))}
    (args.output_dir / "explanation_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
