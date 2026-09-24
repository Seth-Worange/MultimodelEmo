"""用训练/验证数据诊断性能瓶颈，不读取测试标签进行选型。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.infer import load_model
from scripts.train import as_tensors, build_view, make_batch, metrics, model_inputs, move_inputs
from utils.augmentation import drop_modalities
from utils.data import prepare_split, read_pickle


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-10 or b.std() < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(classes, target, logits, predicted):
    guess = logits.argmax(1)
    result = metrics(classes, target, logits, predicted)
    result["confusion"] = confusion_matrix(classes, guess, labels=[0, 1, 2]).tolist()
    result["per_class"] = classification_report(
        classes, guess, labels=[0, 1, 2], output_dict=True, zero_division=0)
    result["target_std"] = float(target.std())
    result["prediction_std"] = float(predicted.std())
    result["strength_groups"] = {}
    for name, select in {
        "neutral": target == 0,
        "weak_nonzero": (np.abs(target) > 0) & (np.abs(target) <= 0.5),
        "medium": (np.abs(target) > 0.5) & (np.abs(target) < 2),
        "strong": np.abs(target) >= 2,
    }.items():
        if select.any():
            result["strength_groups"][name] = {
                "n": int(select.sum()),
                "mae": float(np.abs(target[select] - predicted[select]).mean()),
                "accuracy": float((classes[select] == guess[select]).mean()),
                "mean_target_abs": float(np.abs(target[select]).mean()),
                "mean_prediction_abs": float(np.abs(predicted[select]).mean()),
            }
    return result


def pooled(values, mask, with_std=False):
    weight = mask[..., None]
    count = weight.sum(1).clip(1)
    mean = (values * weight).sum(1) / count
    if not with_std:
        return mean
    variance = (((values - mean[:, None]) ** 2) * weight).sum(1) / count
    return np.concatenate((mean, np.sqrt(variance.clip(0))), axis=1)


def profile(raw, data):
    y, cls = data["sentiment"], data["classes"]
    out = {
        "n": len(cls),
        "class_counts": np.bincount(cls, minlength=3).tolist(),
        "sign_label_disagreement": int((cls != (np.sign(y).astype(int) + 1)).sum()),
        "neutral_nonzero_targets": int(((cls == 1) & (y != 0)).sum()),
        "length_quantiles": np.quantile(data["text_mask"].sum(1), [0, .25, .5, .75, 1]).tolist(),
        "id_examples": data["ids"][:3],
        "feature_scale": {},
    }
    for name in ("audio", "vision"):
        values = data[name][data[name + "_mask"]]
        std = values.std(0)
        mean = values.mean(0)
        out["feature_scale"][name] = {
            "std_quantiles": np.quantile(std, [0, .25, .5, .75, 1]).tolist(),
            "mean_abs_max": float(np.abs(mean).max()),
            "max_abs": float(np.abs(values).max()),
            "constant_dimensions": np.flatnonzero(std < 1e-6).tolist(),
            "std_ratio_nonconstant": float(std.max() / std[std > 1e-6].min()),
            "nonfinite_raw": int((~np.isfinite(raw[name])).sum()),
        }
    return out


@torch.inference_mode()
def predict(model, data, device, drop=(), cap=False):
    gathered = {}
    for start in range(0, len(data["classes"]), 64):
        batch = make_batch(data, torch.arange(start, min(start + 64, len(data["classes"]))))
        if drop:
            batch = drop_modalities(batch, drop)
        batch = move_inputs(batch, device)
        output = model.forward_cap(*model_inputs(batch)) if cap else model(*model_inputs(batch))
        keys = ("modal_logits", "modal_sentiment", "confidence", "uncertainty") if cap else (
            "logits", "sentiment")
        for key in keys:
            gathered.setdefault(key, []).append(output[key].cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in gathered.items()}


def run_probes(train, valid):
    result = {}
    features = {}
    for name, key, with_std in (
        ("text", "teacher", False), ("audio", "audio", True), ("vision", "vision", True)):
        features[name] = [pooled(d[key], d[name + "_mask"], with_std) for d in (train, valid)]
    features["all"] = [np.concatenate([features[m][i] for m in ("text", "audio", "vision")], 1)
                       for i in range(2)]
    # 固定参数的诊断探针；不搜索验证集超参数。
    for name, (x_train, x_valid) in features.items():
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500))
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        classifier.fit(x_train, train["classes"])
        regressor.fit(x_train, train["sentiment"])
        result[name] = summarize(
            valid["classes"], valid["sentiment"], classifier.predict_proba(x_valid),
            np.clip(regressor.predict(x_valid), -3, 3))
        print("probe", name, {k: round(result[name][k], 4) for k in
                             ("accuracy", "macro_f1", "mae", "pearson")}, flush=True)
    return result


def cap_history(folder):
    import csv
    rows = list(csv.DictReader((folder / "cap_metrics.csv").open(encoding="utf-8")))
    manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    selected = next(row for row in rows if int(row["epoch"]) == manifest["cap_best_epoch"])
    result = {"selected_epoch": int(selected["epoch"]), "modalities": {}}
    for modality in ("text", "audio", "vision"):
        def score(row):
            return float(row[modality + "_mae"]) / 6 + (1 - float(row[modality + "_macro_f1"])) / 2
        best = min(rows, key=score)
        result["modalities"][modality] = {
            "selected_mae": float(selected[modality + "_mae"]),
            "selected_f1": float(selected[modality + "_macro_f1"]),
            "individual_best_epoch": int(best["epoch"]),
            "individual_best_mae": float(best[modality + "_mae"]),
            "individual_best_f1": float(best[modality + "_macro_f1"]),
        }
    return result


def evaluation_mask_audit(data):
    def generate(batch_size, seed):
        rng = random.Random(seed)
        collected = {"local": [], "interval": []}
        for start in range(0, len(data["classes"]), batch_size):
            batch = make_batch(data, torch.arange(start, min(start + batch_size, len(data["classes"]))))
            seeds = {view: rng.randrange(2**31) for view in ("local", "whole", "interval")}
            for view in collected:
                transformed = build_view(batch, view, seeds[view])
                masks = torch.stack([transformed[m + "_mask"] for m in ("text", "audio", "vision")], -1)
                collected[view].append(masks)
        return {view: torch.cat(parts) for view, parts in collected.items()}

    a, b, c = generate(32, 2026), generate(64, 2026), generate(32, 11027)
    return {view: {
        "sample_change_same_seed_batch32_vs64": float((a[view] != b[view]).flatten(1).any(1).float().mean()),
        "sample_change_eval_vs_training_seed": float((a[view] != c[view]).flatten(1).any(1).float().mean()),
        "text_missing_samples": int((a[view][:, :, 0] != data["text_mask"]).any(1).sum()),
    } for view in a}


@torch.inference_mode()
def padding_audit(model, data, device):
    model = model.models[0] if hasattr(model, "models") else model
    if not hasattr(model, "audio_gru"):
        return {}
    rows = torch.where((data["lengths"] >= 5) & (data["lengths"] <= 35))[0][:16]
    differences = []
    for index in rows:
        batch = move_inputs(make_batch(data, index.reshape(1)), device)
        length = int(batch["lengths"][0])
        values = model.audio_input(model._audio_features(batch["audio"], batch["audio_mask"]))
        full, _ = model.audio_gru(values)
        trimmed, _ = model.audio_gru(values[:, :length])
        valid = batch["audio_mask"][:, :length]
        differences.append(float((full[:, :length][valid] - trimmed[valid]).abs().mean()))
    return {"samples": len(differences), "audio_hidden_mean_abs_delta_remove_trailing_padding":
            float(np.mean(differences))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/bottleneck_audit_20260924"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = read_pickle("data/附件2-数据集特征文件/aligned_50.pkl")
    train, valid = (prepare_split(raw[split]) for split in ("train", "valid"))
    report = {"protocol": {"splits_used": ["train", "valid"], "device": str(device),
                           "torch": torch.__version__, "probe_C": 0.1, "probe_ridge_alpha": 100.0}}
    report["data"] = {s: profile(raw[s], d) for s, d in (("train", train), ("valid", valid))}
    train_ids, valid_ids = set(train["ids"]), set(valid["ids"])
    report["data"]["overlap_ids"] = len(train_ids & valid_ids)
    train_videos = {s.rsplit("$_$", 1)[0] for s in train_ids}
    valid_videos = {s.rsplit("$_$", 1)[0] for s in valid_ids}
    report["data"]["source_video_overlap"] = {
        "train_videos": len(train_videos), "valid_videos": len(valid_videos),
        "overlap": len(train_videos & valid_videos)}
    report["data"]["duplicate_raw_text_across_splits"] = len(
        set(map(str, raw["train"]["raw_text"])) & set(map(str, raw["valid"]["raw_text"])))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-uncased", cache_dir="cache/huggingface/hub", local_files_only=True)
    for split in ("train", "valid"):
        counts = np.array([len(tokenizer.encode(str(s), truncation=False))
                           for s in raw[split]["raw_text"]])
        report["data"][split]["full_bert_tokens_over_50"] = int((counts > 50).sum())
        report["data"][split]["full_bert_tokens_max"] = int(counts.max())
    del raw
    unaligned = read_pickle("data/附件2-数据集特征文件/unaligned_50.pkl")["train"]
    report["data"]["unaligned_train"] = {
        "shapes": {k: list(v.shape) for k, v in unaligned.items() if hasattr(v, "shape")},
        "length_fields": {k: np.quantile(unaligned[k], [0, .5, 1]).tolist()
                          for k in ("audio_lengths", "vision_lengths")},
        "keys": list(unaligned),
    }
    del unaligned
    with threadpool_limits(limits=4):
        report["probes"] = run_probes(train, valid)
    report["naive"] = {"median_train_regression_mae": float(np.abs(
        valid["sentiment"] - np.median(train["sentiment"])).mean()),
        "majority_accuracy": float((valid["classes"] == np.bincount(train["classes"]).argmax()).mean())}
    tensor_train, tensor_valid = as_tensors(train), as_tensors(valid)
    report["evaluation_masks"] = evaluation_mask_audit(tensor_valid)
    mapping_path = Path("outputs/runs/main/q3_alignment.json")
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        report["q3_mapping"] = {key: {
            "token_match": value["token_match_fraction"],
            "mapped_words": len({p["word_index"] for p in value["positions"] if p}),
            "total_aligned_words": len(value["words"]),
            "latest_mapped_time": max(p["end"] for p in value["positions"] if p),
            "latest_word_time": max(w["end"] for w in value["words"]),
        } for key, value in mapping.items()}
    cases = {
        "baseline_transformer": ["baseline_transformer_s2026"],
        "baseline_bigru_ensemble": ["availability_s2026", "availability_s2027"],
        "fuse_ensemble": ["fuse_s2026", "fuse_s2027"],
        "cica_bigru": ["cica_softca_bigru_s20260924"],
        "cica_transformer": ["cica_softca_transformer_s20260924"],
        "cica_old": ["cica_s2027"],
        "text_only": ["text_only"],
    }
    report["checkpoints"] = {}
    for name, runs in cases.items():
        paths = [Path("outputs/runs") / run / "best.pt" for run in runs]
        model = load_model(paths, device)
        drop = ("audio", "vision") if name == "text_only" else ()
        outputs = predict(model, tensor_valid, device, drop)
        entry = {"checkpoints": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                  for p in paths],
                 "valid_clean": summarize(valid["classes"], valid["sentiment"],
                                           outputs["logits"], outputs["sentiment"])}
        np.savez_compressed(args.output / (name + "_valid.npz"), **outputs,
                            classes=valid["classes"], target=valid["sentiment"])
        neutral_zero = outputs["sentiment"].copy()
        neutral_zero[outputs["logits"].argmax(1) == 1] = 0
        entry["neutral_zero_postprocess_mae"] = float(np.abs(neutral_zero - valid["sentiment"]).mean())
        entry["padding_audit"] = padding_audit(model, tensor_valid, device)
        if name in ("fuse_ensemble", "cica_bigru", "baseline_transformer"):
            training = predict(model, tensor_train, device)
            entry["train_clean"] = summarize(train["classes"], train["sentiment"],
                                             training["logits"], training["sentiment"])
            # 只使用声画整体消融，避免缓存文本造成遮蔽语义不一致。
            no_av = predict(model, tensor_valid, device, ("audio", "vision"))
            entry["valid_without_av"] = summarize(valid["classes"], valid["sentiment"],
                                                   no_av["logits"], no_av["sentiment"])
        if name.startswith("cica"):
            branch = predict(model, tensor_valid, device, cap=True)
            entry["cap_history"] = cap_history(paths[0].parent)
            entry["branches"] = {}
            for i, modality in enumerate(("text", "audio", "vision")):
                keep = valid[modality + "_mask"].any(1)
                error = np.abs(branch["modal_sentiment"][keep, i] - valid["sentiment"][keep])
                confidence = branch["confidence"][keep, i]
                uncertainty = branch["uncertainty"][keep, i]
                correct = branch["modal_logits"][keep, i].argmax(1) == valid["classes"][keep]
                entry["branches"][modality] = {
                    "accuracy": float(correct.mean()), "confidence_mean": float(confidence.mean()),
                    "confidence_std": float(confidence.std()),
                    "confidence_correctness_corr": correlation(confidence, correct),
                    "uncertainty_error_corr": correlation(uncertainty, error),
                    "uncertainty_target_mae": float(np.abs(uncertainty - np.tanh(error)).mean()),
                    "oracle_constant_target_mae": float(np.abs(np.tanh(error)
                                                              - np.median(np.tanh(error))).mean()),
                }
        report["checkpoints"][name] = entry
        print("checkpoint", name, {k: round(entry["valid_clean"][k], 4) for k in
                                   ("accuracy", "macro_f1", "mae", "pearson")}, flush=True)
        (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.output / "audit.json", flush=True)


if __name__ == "__main__":
    main()
