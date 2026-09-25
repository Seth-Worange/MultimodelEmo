"""冻结文本特征探针，仅用于诊断表示质量与复杂模型瓶颈。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from utils.config import parse_config_args
from utils.data import resolve_data_root, read_pickle, prepare_split
from scripts.train import metrics


def pooled_text(data):
    mask = data["text_mask"][..., None]
    mean = (data["teacher"] * mask).sum(1) / mask.sum(1).clip(1)
    return np.concatenate([data["teacher"][:, 0], mean], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--classification-c", type=float, nargs="+", default=[1., 10.])
    parser.add_argument("--regression-c", type=float, default=10.)
    args = parse_config_args(parser, "probe")
    if args.output_dir is None:
        parser.error("output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_pickle(resolve_data_root(args.data_root) / "附件2-数据集特征文件" / "aligned_50.pkl")
    train, valid = (prepare_split(raw[s]) for s in ("train", "valid"))
    x, xv = pooled_text(train), pooled_text(valid)
    del raw
    # 标准化与拟合仅使用训练集；验证集仅报告指标。
    reg = make_pipeline(StandardScaler(), SVR(C=args.regression_c, epsilon=0.1))
    reg.fit(x, train["sentiment"])
    predicted = reg.predict(xv).clip(-3, 3)
    results = []
    for c in args.classification_c:
        clf = make_pipeline(StandardScaler(), SVC(C=c, class_weight="balanced"))
        clf.fit(x, train["classes"])
        # SVC决策分数的argmax未必等于其投票结果，使用真实predict结果。
        guess = clf.predict(xv)
        logits = np.eye(3)[guess]
        result = {"classification_c": c, **metrics(valid["classes"], valid["sentiment"], logits, predicted)}
        results.append(result)
        joblib.dump({"classifier": clf, "regressor": reg, "pooling": "cls+masked_mean"}, args.output_dir / f"probe_c{c:g}.joblib")
        np.savez_compressed(args.output_dir / f"valid_c{c:g}.npz", classes=valid["classes"], target=valid["sentiment"], guess=guess, sentiment=predicted)
        print(json.dumps(result), flush=True)
    report = {"split": "valid", "kind": "text_only_diagnostic", "train_size": len(x), "valid_size": len(xv), "regression_c": args.regression_c, "results": results}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
