"""审计附件2的标签、模态覆盖和简单线性可分性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.data import prepare_split, read_pickle, resolve_data_root


def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """汇总有效词位的均值和波动，保留缺失比例。"""
    weight = mask[..., None].astype(np.float32)
    count = weight.sum(axis=1).clip(1)
    mean = (values * weight).sum(axis=1) / count
    variance = ((values - mean[:, None]) ** 2 * weight).sum(axis=1) / count
    coverage = mask.mean(axis=1, keepdims=True).astype(np.float32)
    return np.concatenate((mean, np.sqrt(variance), coverage), axis=1)


def features(data: dict) -> dict[str, np.ndarray]:
    return {
        "text": masked_mean_std(data["teacher"], data["text_mask"]),
        "audio": masked_mean_std(data["audio"], data["audio_mask"]),
        "vision": masked_mean_std(data["vision"], data["vision_mask"]),
    }


def split_stats(data: dict) -> dict:
    labels = data["classes"]
    strength = np.abs(data["sentiment"])
    stats = {
        "n": int(len(labels)),
        "classes_negative_neutral_positive": np.bincount(labels, minlength=3).tolist(),
        "sign_label_disagreements": int(np.count_nonzero(np.sign(data["sentiment"]) != labels - 1)),
        "aligned_slots_filled_50": int(np.count_nonzero(data["lengths"] >= 50)),
        "strength_counts": {
            "zero": int(np.count_nonzero(strength == 0)),
            "weak_0_to_0_5": int(np.count_nonzero((strength > 0) & (strength <= 0.5))),
            "medium_0_5_to_1": int(np.count_nonzero((strength > 0.5) & (strength <= 1))),
            "strong_over_1": int(np.count_nonzero(strength > 1)),
        },
    }
    for name in ("text", "audio", "vision"):
        mask = data[f"{name}_mask"]
        stats[f"{name}_coverage"] = {
            "mean_valid_ratio": float(mask.mean()),
            "mean_ratio_within_text_content": float((mask & data["text_mask"]).sum()
                                                     / data["text_mask"].sum()),
            "whole_missing_samples": int(np.count_nonzero(~mask.any(axis=1))),
        }
    return stats


def probe(train: dict, valid: dict) -> dict:
    train_features, valid_features = features(train), features(valid)
    results = {}
    for names in (("text",), ("audio",), ("vision",),
                  ("audio", "vision"), ("text", "audio", "vision")):
        key = "+".join(names)
        x_train = np.concatenate([train_features[name] for name in names], axis=1)
        x_valid = np.concatenate([valid_features[name] for name in names], axis=1)
        components = min(64, x_train.shape[1] - 1)
        model = make_pipeline(StandardScaler(), PCA(n_components=components,
                                                    svd_solver="randomized", random_state=2026),
                              LogisticRegression(C=1.0, max_iter=300))
        model.fit(x_train, train["classes"])
        predicted = model.predict(x_valid)
        confusion = confusion_matrix(valid["classes"], predicted, labels=[0, 1, 2])
        results[key] = {
            "n_features": int(x_train.shape[1]),
            "pca_components": components,
            "train_accuracy": float(accuracy_score(train["classes"], model.predict(x_train))),
            "accuracy": float(accuracy_score(valid["classes"], predicted)),
            "macro_f1": float(f1_score(valid["classes"], predicted, average="macro")),
            "neutral_recall": float(confusion[1, 1] / max(1, confusion[1].sum())),
            "confusion_rows_true": confusion.tolist(),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/q2_data_audit.json"))
    args = parser.parse_args()
    source = resolve_data_root(args.data_root) / "附件2-数据集特征文件" / "aligned_50.pkl"
    raw = read_pickle(source)
    train = prepare_split(raw["train"])
    valid = prepare_split(raw["valid"])
    train_groups = {sample.rsplit("$_$", 1)[0] for sample in train["ids"]}
    valid_groups = {sample.rsplit("$_$", 1)[0] for sample in valid["ids"]}
    report = {
        "source": str(source),
        "train": split_stats(train),
        "valid": split_stats(valid),
        "video_group_overlap": len(train_groups & valid_groups),
        "linear_probes": probe(train, valid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
