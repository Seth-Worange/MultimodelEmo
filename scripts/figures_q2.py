"""整理问题二全量鲁棒性实验结果，产出论文表格、趋势量与绘图。

输入为 ``scripts.robustness`` 产出的 ``robustness.csv``、``scripts.evaluate`` 产出的
``validation_metrics.json`` 与 ``per_sample_valid.csv``、各消融臂 ``metrics.csv``。
输出（默认 ``outputs/diagnostics/q2_paper/``）：
- ``q2_basic_performance.csv``：验证集四视图基础性能（题面问题2(4)）。
- ``q2_table_missing_type.csv``：缺失模态类型对照（whole 与 interval@参考率）。
- ``q2_table_missing_rate.csv``：缺失率扫描曲线（local/interval，音视频）。
- ``q2_table_missing_location.csv``：缺失位置对照。
- ``q2_table_ablation.csv``：消融臂 clean 与缺失情形性能（题面问题2(2)）。
- ``q2_trend_slopes.csv``：每 10% 缺失率的 F1/MAE 退化斜率。
- ``q2_error_slices.json``：验证集按类错误切片（错误归因，题面问题2(4)）。
- ``q2_fig_*.png``：缺失率曲线、模态类型、缺失位置、消融对比、热力图。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MAINLINE = "q2_fuse_lowaux_s2026"
SEED_CHECK = "q2_fuse_lowaux_s2027"
AB_PREFIX = "ab_"
REF_RATE = 0.4          # 模态类型对照的参考缺失率
VIEW_KEYS = ("accuracy", "macro_f1", "mae", "pearson")
CLASS_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def configure_font() -> bool:
    """选用可用的中文字体；找不到时返回 False，图形改用英文标注。"""
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def f(row: dict, key: str) -> float:
    return float(row[key])


def best_epoch_row(metrics_csv: Path) -> dict:
    rows = read_csv(metrics_csv)
    return min(rows, key=lambda r: float(r["val_score"]))


def robustness_rows(runs_dir: Path, run_name: str) -> list[dict]:
    path = runs_dir / run_name / "robustness.csv"
    return read_csv(path) if path.exists() else []


def pick(rows: list[dict], pattern: str, modalities: str, rate: float | None,
         location: str | None) -> dict | None:
    for row in rows:
        if row["pattern"] != pattern or row["missing_modalities"] != modalities:
            continue
        if rate is not None and abs(float(row["missing_rate"]) - rate) > 1e-9:
            continue
        if location is not None and row["location"] != location:
            continue
        return row
    return None


def slope(rows: list[dict], pattern: str, modalities: str, location: str, key: str) -> float | None:
    """按缺失率做线性拟合，返回每 10% 缺失率的性能变化。"""
    points = [(float(r["missing_rate"]), f(r, key)) for r in rows
              if r["pattern"] == pattern and r["missing_modalities"] == modalities
              and r["location"] == location]
    if len(points) < 2:
        return None
    x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    return float(np.polyfit(x, y, 1)[0] * 0.1)


def build_tables(runs_dir: Path, output_dir: Path, main: list[dict], arms: dict[str, dict]) -> None:
    # ---- 基础性能（四视图） ----
    metrics_json = runs_dir / MAINLINE / "validation_metrics.json"
    if metrics_json.exists():
        payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        rows = []
        for view, item in payload.items():
            if isinstance(item, dict) and "macro_f1" in item:
                rows.append({"view": view, **{k: f"{item[k]:.4f}" for k in VIEW_KEYS}})
        write_csv(output_dir / "q2_basic_performance.csv", rows, ("view",) + VIEW_KEYS)

    # ---- 缺失模态类型：whole 全部 + interval@参考率全部组合 ----
    rows = []
    for row in main:
        if row["pattern"] == "whole" or (
                row["pattern"] == "interval" and row["location"] == "random"
                and abs(float(row["missing_rate"]) - REF_RATE) < 1e-9):
            rows.append({"pattern": row["pattern"], "missing_modalities": row["missing_modalities"],
                         "missing_rate": row["missing_rate"], **{k: f"{f(row, k):.4f}" for k in VIEW_KEYS}})
    write_csv(output_dir / "q2_table_missing_type.csv", rows,
              ("pattern", "missing_modalities", "missing_rate") + VIEW_KEYS)

    # ---- 缺失率曲线 ----
    rows = []
    for row in main:
        curve = ((row["pattern"] == "local" and row["missing_modalities"] == "audio+vision"
                  and row["location"] == "random")
                 or (row["pattern"] == "interval" and row["location"] == "random"))
        if curve:
            rows.append({"pattern": row["pattern"], "missing_modalities": row["missing_modalities"],
                         "missing_rate": row["missing_rate"], **{k: f"{f(row, k):.4f}" for k in VIEW_KEYS}})
    rows.sort(key=lambda r: (r["pattern"], r["missing_modalities"], float(r["missing_rate"])))
    write_csv(output_dir / "q2_table_missing_rate.csv", rows,
              ("pattern", "missing_modalities", "missing_rate") + VIEW_KEYS)

    # ---- 缺失位置 ----
    rows = []
    for row in main:
        if row["pattern"] in ("local", "interval") and row["missing_modalities"] == "audio+vision":
            rows.append({"pattern": row["pattern"], "location": row["location"],
                         "missing_rate": row["missing_rate"], **{k: f"{f(row, k):.4f}" for k in VIEW_KEYS}})
    rows.sort(key=lambda r: (r["pattern"], float(r["missing_rate"]), r["location"]))
    write_csv(output_dir / "q2_table_missing_location.csv", rows,
              ("pattern", "location", "missing_rate") + VIEW_KEYS)

    # ---- 趋势斜率 ----
    rows = []
    for pattern, modalities, location, label in (
            ("local", "audio+vision", "random", "local_audio+vision"),
            ("interval", "audio+vision", "random", "interval_audio+vision"),
            ("interval", "text", "random", "interval_text"),
            ("interval", "audio", "random", "interval_audio"),
            ("interval", "vision", "random", "interval_vision"),
            ("interval", "text+audio+vision", "random", "interval_all")):
        entry = {"series": label}
        for key in ("macro_f1", "mae"):
            value = slope(main, pattern, modalities, location, key)
            entry[f"d_{key}_per_10pct"] = f"{value:.4f}" if value is not None else ""
        rows.append(entry)
    write_csv(output_dir / "q2_trend_slopes.csv", rows,
              ("series", "d_macro_f1_per_10pct", "d_mae_per_10pct"))

    # ---- 消融表：clean 与代表性缺失情形 ----
    rows = []
    for name, info in arms.items():
        best = info["best"]
        row = {"arm": name, "best_epoch": best.get("epoch", ""),
               "clean_accuracy": f"{float(best['clean_accuracy']):.4f}",
               "clean_macro_f1": f"{float(best['clean_macro_f1']):.4f}",
               "clean_mae": f"{float(best['clean_mae']):.4f}"}
        for tag, pattern, modalities, rate, location in (
                ("interval0.4", "interval", "audio+vision", 0.4, "random"),
                ("interval0.6", "interval", "audio+vision", 0.6, "random"),
                ("local0.4", "local", "audio+vision", 0.4, "random"),
                ("whole_av", "whole", "audio+vision", 1.0, None)):
            hit = pick(info["robust"], pattern, modalities, rate, location)
            row[f"{tag}_macro_f1"] = f"{f(hit, 'macro_f1'):.4f}" if hit else ""
            row[f"{tag}_mae"] = f"{f(hit, 'mae'):.4f}" if hit else ""
        rows.append(row)
    fields = ["arm", "best_epoch", "clean_accuracy", "clean_macro_f1", "clean_mae"]
    for tag in ("interval0.4", "interval0.6", "local0.4", "whole_av"):
        fields += [f"{tag}_macro_f1", f"{tag}_mae"]
    rows.sort(key=lambda r: -float(r["clean_macro_f1"] or 0))
    write_csv(output_dir / "q2_table_ablation.csv", rows, fields)

    # ---- 错误切片 ----
    per_sample = runs_dir / MAINLINE / "per_sample_valid.csv"
    if per_sample.exists():
        slices = {}
        data = read_csv(per_sample)
        for view in ("clean", "interval"):
            entry = {}
            for cls, name in CLASS_NAMES.items():
                sub = [r for r in data if int(r["label_class"]) == cls]
                if not sub:
                    continue
                entry[name] = {
                    "n": len(sub),
                    "recall": float(np.mean([int(r[f"{view}_correct"]) for r in sub])),
                    "strength_mae": float(np.mean([
                        abs(float(r[f"{view}_strength"]) - float(r["label_strength"])) for r in sub])),
                    "true_strength_mean": float(np.mean([float(r["label_strength"]) for r in sub])),
                    "pred_strength_mean": float(np.mean([float(r[f"{view}_strength"]) for r in sub])),
                }
            slices[view] = entry
        (output_dir / "q2_error_slices.json").write_text(
            json.dumps(slices, ensure_ascii=False, indent=1), encoding="utf-8")


def plot_rate_curves(output_dir: Path, main: list[dict], chinese: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("interval", "audio+vision", "interval " + ("音视频" if chinese else "audio+vision")),
              ("interval", "text", "interval " + ("文本" if chinese else "text")),
              ("interval", "audio", "interval " + ("语音" if chinese else "audio")),
              ("interval", "vision", "interval " + ("视觉" if chinese else "vision")),
              ("interval", "text+audio+vision", "interval " + ("三模态" if chinese else "all")),
              ("local", "audio+vision", "local " + ("音视频短游程" if chinese else "audio+vision runs"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for pattern, modalities, label in series:
        points = sorted((float(r["missing_rate"]), f(r, "macro_f1"), f(r, "mae"))
                        for r in main if r["pattern"] == pattern
                        and r["missing_modalities"] == modalities and r["location"] == "random")
        if not points:
            continue
        x = [p[0] for p in points]
        axes[0].plot(x, [p[1] for p in points], marker="o", label=label)
        axes[1].plot(x, [p[2] for p in points], marker="o", label=label)
    base = pick(main, "none", "audio+vision", 0.0, None) or pick(main, "none", "audio+vision", 0.0, "none")
    if base:
        for ax in axes:
            ax.axhline(f(base, "macro_f1" if ax is axes[0] else "mae"), color="gray",
                       linestyle="--", linewidth=1)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("MAE")
    for ax in axes:
        ax.set_xlabel("缺失率" if chinese else "missing rate")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "q2_fig_rate_curve.png", dpi=300)
    plt.close(fig)


def plot_missing_type(output_dir: Path, main: list[dict], chinese: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, f1, mae = [], [], []
    for row in main:
        whole = row["pattern"] == "whole"
        mid = (row["pattern"] == "interval" and row["location"] == "random"
               and abs(float(row["missing_rate"]) - REF_RATE) < 1e-9)
        if not (whole or mid):
            continue
        tag = ("整段-" if whole else f"区间@{REF_RATE}-") if chinese else ("whole-" if whole else f"interval@{REF_RATE}-")
        labels.append(tag + row["missing_modalities"])
        f1.append(f(row, "macro_f1"))
        mae.append(f(row, "mae"))
    base = pick(main, "none", "audio+vision", 0.0, None) or pick(main, "none", "audio+vision", 0.0, "none")
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].bar(x, f1, color="#4C72B0")
    axes[1].bar(x, mae, color="#C44E52")
    if base:
        axes[0].axhline(f(base, "macro_f1"), color="gray", linestyle="--", linewidth=1)
        axes[1].axhline(f(base, "mae"), color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "q2_fig_missing_type.png", dpi=300)
    plt.close(fig)


def plot_location(output_dir: Path, main: list[dict], chinese: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    locations = ["start", "middle", "end", "random"]
    rates = sorted({float(r["missing_rate"]) for r in main
                    if r["pattern"] == "interval" and r["missing_modalities"] == "audio+vision"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.18
    for i, rate in enumerate(rates):
        f1, mae = [], []
        for loc in locations:
            hit = pick(main, "interval", "audio+vision", rate, loc)
            f1.append(f(hit, "macro_f1") if hit else np.nan)
            mae.append(f(hit, "mae") if hit else np.nan)
        x = np.arange(len(locations)) + (i - len(rates) / 2) * width + width / 2
        axes[0].bar(x, f1, width=width, label=f"rate={rate:g}")
        axes[1].bar(x, mae, width=width, label=f"rate={rate:g}")
    for ax, ylabel in zip(axes, ("Macro-F1", "MAE")):
        ax.set_xticks(np.arange(len(locations)), locations)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("缺失位置" if chinese else "missing location")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "q2_fig_location.png", dpi=300)
    plt.close(fig)


def plot_ablation(output_dir: Path, arms: dict[str, dict], chinese: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(arms, key=lambda n: -float(arms[n]["best"]["clean_macro_f1"]))
    clean = [float(arms[n]["best"]["clean_macro_f1"]) for n in names]
    miss = []
    for name in names:
        hit = pick(arms[name]["robust"], "interval", "audio+vision", 0.4, "random")
        miss.append(f(hit, "macro_f1") if hit else np.nan)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(y + 0.18, clean, height=0.34, label="clean" + ("" if not chinese else "（完整输入）"), color="#4C72B0")
    ax.barh(y - 0.18, miss, height=0.34, label=("interval 0.4" + ("（区间缺失）" if chinese else "")), color="#C44E52")
    ax.set_yticks(y, names)
    ax.set_xlabel("Macro-F1")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "q2_fig_ablation.png", dpi=300)
    plt.close(fig)


def plot_heatmap(output_dir: Path, main: list[dict], chinese: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("audio+vision", "interval"), ("text", "interval"), ("audio", "interval"),
              ("vision", "interval"), ("text+audio+vision", "interval"), ("audio+vision", "local")]
    rates = sorted({float(r["missing_rate"]) for r in main})
    grid = np.full((len(series), len(rates)), np.nan)
    for i, (modalities, pattern) in enumerate(series):
        for j, rate in enumerate(rates):
            hit = pick(main, pattern, modalities, rate, "random")
            if hit:
                grid[i, j] = f(hit, "macro_f1")
    labels = [f"{pattern}:{modalities}" for modalities, pattern in series]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    image = ax.imshow(grid, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(rates)), [f"{r:g}" for r in rates])
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    ax.set_xlabel("缺失率" if chinese else "missing rate")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(output_dir / "q2_fig_heatmap.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "runs")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mainline", default=MAINLINE)
    args = parser.parse_args()
    output_dir = args.output_dir or args.runs_dir.parent / "diagnostics" / "q2_paper"
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = robustness_rows(args.runs_dir, args.mainline)
    if not main_rows:
        raise SystemExit(f"missing {args.mainline}/robustness.csv — run scripts.robustness first")

    arms: dict[str, dict] = {}
    metrics_csv = args.runs_dir / args.mainline / "metrics.csv"
    if metrics_csv.exists():
        arms[args.mainline] = {"best": best_epoch_row(metrics_csv), "robust": main_rows}
    seed_rows = robustness_rows(args.runs_dir, SEED_CHECK)
    if seed_rows and (args.runs_dir / SEED_CHECK / "metrics.csv").exists():
        arms[SEED_CHECK] = {"best": best_epoch_row(args.runs_dir / SEED_CHECK / "metrics.csv"),
                            "robust": seed_rows}
    for run_dir in sorted(args.runs_dir.glob(f"{AB_PREFIX}*_s2026")):
        m = run_dir / "metrics.csv"
        r = run_dir / "robustness.csv"
        if m.exists() and r.exists():
            arms[run_dir.name] = {"best": best_epoch_row(m), "robust": read_csv(r)}

    chinese = configure_font()
    if not chinese:
        print("warning: no CJK font found, falling back to English labels")

    build_tables(args.runs_dir, output_dir, main_rows, arms)
    plot_rate_curves(output_dir, main_rows, chinese)
    plot_missing_type(output_dir, main_rows, chinese)
    plot_location(output_dir, main_rows, chinese)
    plot_ablation(output_dir, arms, chinese)
    plot_heatmap(output_dir, main_rows, chinese)

    # 全量情形表随附（论文附录/补充材料）
    write_csv(output_dir / "q2_robustness_full.csv", main_rows, list(main_rows[0]))

    # 供论文文字直接引用的关键趋势
    base = pick(main_rows, "none", "audio+vision", 0.0, None)
    if base:
        print(f"clean baseline: F1={f(base, 'macro_f1'):.4f} MAE={f(base, 'mae'):.4f}")
    for name in sorted(arms):
        best = arms[name]["best"]
        print(f"{name:>28} best_epoch={best.get('epoch','?'):>2} "
              f"clean F1={float(best['clean_macro_f1']):.4f} MAE={float(best['clean_mae']):.4f}")
    print(f"wrote tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
