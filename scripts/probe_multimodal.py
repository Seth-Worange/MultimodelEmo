"""冻结BERT与声画统计的多模态核模型探针。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from scripts.train import metrics
from utils.config import parse_config_args
from utils.data import prepare_split, read_pickle, resolve_data_root


def statistics(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    weight = mask[..., None].astype(np.float32)
    count = weight.sum(1).clip(1)
    mean = (values * weight).sum(1) / count
    variance = ((values - mean[:, None]) ** 2 * weight).sum(1) / count
    availability = mask.mean(1, keepdims=True).astype(np.float32)
    return np.concatenate((mean, np.sqrt(variance + 1e-8), availability), axis=1)


def features(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text_mask = data["text_mask"]
    body = statistics(data["teacher"], text_mask)[:, :768]
    text = np.concatenate((data["teacher"][:, 0], body), axis=1)
    audio = statistics(data["audio"], data["audio_mask"])
    vision = statistics(data["vision"], data["vision_mask"])
    return text, audio, vision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--av-weights", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--classification-c", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--regression-c", type=float, default=10.0)
    args = parse_config_args(parser, "probe")
    if args.output_dir is None:
        parser.error("output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = resolve_data_root(args.data_root) / "附件2-数据集特征文件" / "aligned_50.pkl"
    raw = read_pickle(path)
    train, valid = (prepare_split(raw[split]) for split in ("train", "valid"))
    train_blocks, valid_blocks = features(train), features(valid)
    scalers = [StandardScaler().fit(block) for block in train_blocks]
    train_scaled = [scaler.transform(block) for scaler, block in zip(scalers, train_blocks)]
    valid_scaled = [scaler.transform(block) for scaler, block in zip(scalers, valid_blocks)]
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=2026)
    text_train = pca.fit_transform(train_scaled[0])
    text_valid = pca.transform(valid_scaled[0])
    train_scaled[0], valid_scaled[0] = text_train, text_valid
    train_scaled = [block / np.sqrt(block.shape[1]) for block in train_scaled]
    valid_scaled = [block / np.sqrt(block.shape[1]) for block in valid_scaled]
    results = []
    for av_weight in args.av_weights:
        x = np.concatenate((train_scaled[0], av_weight * train_scaled[1],
                            av_weight * train_scaled[2]), axis=1)
        xv = np.concatenate((valid_scaled[0], av_weight * valid_scaled[1],
                             av_weight * valid_scaled[2]), axis=1)
        reg = SVR(C=args.regression_c, epsilon=0.1)
        reg.fit(x, train["sentiment"])
        predicted = reg.predict(xv).clip(-3, 3)
        for c in args.classification_c:
            clf = SVC(C=c, class_weight="balanced")
            clf.fit(x, train["classes"])
            guess = clf.predict(xv)
            result = {"av_weight": av_weight, "classification_c": c,
                      **metrics(valid["classes"], valid["sentiment"],
                                np.eye(3)[guess], predicted)}
            results.append(result)
            print(json.dumps(result), flush=True)
            if result["accuracy"] >= 0.65 or result["macro_f1"] >= 0.64:
                joblib.dump({"scalers": scalers, "pca": pca, "classifier": clf,
                             "regressor": reg, "av_weight": av_weight},
                            args.output_dir / f"probe_av{av_weight:g}_c{c:g}.joblib")
    (args.output_dir / "report.json").write_text(json.dumps({
        "split": "valid", "n_train": len(train["classes"]), "n_valid": len(valid["classes"]),
        "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
