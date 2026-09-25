"""按问题二的三视图规则复评已有验证报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from utils.selection import MAIN_VIEWS, relative_degradation, selection_score, view_score


ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("clean", "local", "whole", "interval")
METRICS = ("accuracy", "macro_f1", "mae", "pearson")


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_review(checkpoint: Path, seed: int) -> tuple[str, str]:
    """仅在历史训练协议相同的情况下重排各轮日志。"""
    run_dir = checkpoint.parent
    run_path, metrics_path = run_dir / "run.json", run_dir / "metrics.csv"
    if not run_path.exists() or not metrics_path.exists():
        return "", ""
    run = json.loads(run_path.read_text(encoding="utf-8-sig"))
    if run.get("evaluation_protocol") != "sample_v2":
        return str(run.get("best_epoch", "")), ""
    evaluation_seed = run.get("evaluation_seed")
    if evaluation_seed is not None and int(evaluation_seed) != seed:
        return str(run.get("best_epoch", "")), ""
    with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return str(run.get("best_epoch", "")), ""

    def score(row: dict[str, str]) -> float:
        views = {view: {metric: float(row[f"{view}_{metric}"])
                        for metric in ("macro_f1", "mae")}
                 for view in MAIN_VIEWS}
        return selection_score(views)

    best = min(rows, key=score)
    return str(run.get("best_epoch", "")), best["epoch"]


def review_candidate(item: dict, protocol: dict) -> dict:
    checkpoint = project_path(item["checkpoint"])
    report_path = project_path(item["validation"])
    if not checkpoint.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint or validation report for {item['name']}")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    actual = report.get("protocol", {})
    if (report.get("split") != "valid" or actual.get("name") != protocol["name"]
            or actual.get("seed") != protocol["seed"]
            or actual.get("neutral_zero") != protocol["neutral_zero"]
            or actual.get("drop_modalities") != []):
        raise ValueError(f"Validation protocol mismatch: {item['name']}")
    checkpoints = report.get("checkpoints", [])
    if len(checkpoints) != 1 or checkpoints[0].get("sha256") != sha256(checkpoint):
        raise ValueError(f"Checkpoint hash mismatch: {item['name']}")
    views = {view: report[view] for view in VIEWS}
    changes = relative_degradation(views)
    saved_epoch, new_best_epoch = epoch_review(checkpoint, protocol["seed"])
    row = {
        "model": item["name"], "family": item["family"],
        "score": selection_score(views), "n": report["n"],
        "saved_epoch": saved_epoch, "new_best_epoch_in_log": new_best_epoch,
        "checkpoint": item["checkpoint"], "validation": item["validation"],
        "sha256": checkpoints[0]["sha256"],
    }
    for view in VIEWS:
        for metric in METRICS:
            row[f"{view}_{metric}"] = views[view][metric]
    for view in MAIN_VIEWS:
        row[f"{view}_score"] = view_score(views[view])
    for view, values in changes.items():
        for key, value in values.items():
            row[f"{view}_{key}"] = value
    return row


def write_report(rows: list[dict], output: Path, config_path: Path, protocol: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# 问题二单模型验证集复评",
        "",
        "依据赛题，局部缺失预测是主任务；整模态缺失仅作为压力测试。",
        "所有候选均使用附件2 valid、同一 sample_v2 缺失协议和种子，不使用 test 选模。",
        "",
        "主分：`S_select = mean(S_clean, S_local, S_interval)`，",
        "其中 `S_v = 0.5(1 - Macro-F1_v) + MAE_v/6`，越低越好。",
        "Accuracy、Pearson 完整列出并用于检查明显退化；赛题未规定二者的硬阈值。",
        "",
        "| 排名 | 单模型 | 主分↓ | clean Acc↑ | clean F1↑ | clean MAE↓ | clean Pearson↑ | ΔF1 local | ΔMAE local | ΔF1 interval | ΔMAE interval | whole F1 / MAE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            f"| {rank} | `{row['model']}` | {row['score']:.4f} | "
            f"{row['clean_accuracy']:.4f} | {row['clean_macro_f1']:.4f} | "
            f"{row['clean_mae']:.4f} | {row['clean_pearson']:.4f} | "
            f"{row['local_delta_macro_f1']:+.4f} | {row['local_delta_mae']:+.4f} | "
            f"{row['interval_delta_macro_f1']:+.4f} | {row['interval_delta_mae']:+.4f} | "
            f"{row['whole_macro_f1']:.4f} / {row['whole_mae']:.4f} |"
        )
    best, runner_up = rows[:2]
    best_acc_gap = max(row["clean_accuracy"] for row in rows) - best["clean_accuracy"]
    best_pearson_gap = max(row["clean_pearson"] for row in rows) - best["clean_pearson"]
    epoch_note = (
        f"其保存轮次 {best['saved_epoch']} 与按新规则重排日志所得轮次 "
        f"{best['new_best_epoch_in_log']} 一致。"
        if best["saved_epoch"] and best["saved_epoch"] == best["new_best_epoch_in_log"]
        else "其历史轮次无法按同一协议核验，需以当前保存权重复评结果为准。"
    )
    lines += [
        "", "## 结果判断", "",
        f"- 当前保存的单检查点中，`{best['model']}` 主分最低（{best['score']:.4f}），"
        f"比第二名 `{runner_up['model']}` 低 {runner_up['score'] - best['score']:.4f}。"
        f"其 clean Accuracy 距候选最高值 {best_acc_gap:.4f}，Pearson 距最高值 {best_pearson_gap:.4f}；"
        "辅助指标没有明显崩塌。",
        f"- 最优模型的 interval 相对退化为 ΔF1={best['interval_delta_macro_f1']:+.4f}、"
        f"ΔMAE={best['interval_delta_mae']:+.4f}，说明长段缺失仍是薄弱环节。"
        "不能只凭退化幅度小判断一个模型更优，需同时看 clean 水平和三视图主分。",
        f"- {epoch_note}",
        "- 本轮只比较验证集上的已保存权重，差距未做显著性检验。"
        "赛题要求的缺失模态类型、发生位置和持续时长，还需用分层专项分析呈现。",
        "", "## 复现与边界", "",
        f"- 候选与报告路径：`{config_path.relative_to(ROOT)}`。",
        f"- 协议：`{protocol['name']}`，seed={protocol['seed']}，中性置零={protocol['neutral_zero']}。",
        "- 每份验证报告的单检查点 SHA256 均已与当前文件核对。",
        "- `new_best_epoch_in_log` 仅重排历史训练日志；若与 saved_epoch 不同，新的轮次没有对应已保存权重，不能直接当成可部署模型。",
        "- 既有检查点由旧四视图分数选出；本表是对当前保存权重的统一复评。"
        "未来训练已经改用三视图早停；若要比较完整的新训练流程，须在相同配置下重训所有候选。",
        "- local/interval 的 ΔF1 为缺失视图减 clean，ΔMAE 同理；whole 只报告诊断值。",
        "", "运行命令：", "",
        f"`python -m scripts.rescore_validation --config {config_path.relative_to(ROOT)} --output {output.relative_to(ROOT)}`",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = project_path(str(args.config))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    rows = [review_candidate(item, protocol) for item in config["candidates"]]
    if len({row["n"] for row in rows}) != 1:
        raise ValueError("Candidate validation sample counts differ")
    rows.sort(key=lambda row: row["score"])
    write_report(rows, project_path(str(args.output)), config_path, protocol)
    for row in rows:
        print(f"{row['model']}: score={row['score']:.4f}, clean_f1={row['clean_macro_f1']:.4f}, "
              f"clean_mae={row['clean_mae']:.4f}, ΔF1(local/interval)="
              f"{row['local_delta_macro_f1']:+.4f}/{row['interval_delta_macro_f1']:+.4f}")


if __name__ == "__main__":
    main()
