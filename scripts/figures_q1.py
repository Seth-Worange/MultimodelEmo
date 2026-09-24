"""生成问题1要求的全量结果汇总表与典型样本三模态对应图。

输入为 ``scripts/features_q1.py`` 产出的 ``manifest.csv`` 与逐样本 ``.npz``。
输出：
- ``q1_summary.csv``：100条样本的模态、时长、维度、对齐粒度与覆盖率汇总（题面问题1(2)）。
- ``q1_coverage.csv``：按模态统计的覆盖率与异常样本数。
- ``q1_typical_<id>.png``：典型样本的文本词区间、语音能量、视觉覆盖在同一时间轴上的对应关系（题面问题1(3)）。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SUMMARY_FIELDS = ("id", "video_id", "clip_id", "seconds", "words", "alignment_unit",
                  "text_dim", "audio_dim", "prosody_dim", "vision_dim", "mean_alignment_score",
                  "alignment_text_similarity", "face_coverage", "pose_coverage", "vision_coverage",
                  "nearest_frame_fraction", "status", "error")


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


def read_summary(manifest: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["alignment_unit"] = "word"  # 词区间为最小对齐单元，与附件2的50词位同构。
    return rows


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})


def pick_typical(rows: list[dict]) -> dict:
    """选一条词数接近中位、三模态覆盖率都高的样本作为典型样本。"""
    usable = [row for row in rows if row["status"] == "ok"]
    if not usable:
        raise ValueError("manifest has no successful sample")
    words = np.array([int(row["words"]) for row in usable])
    median = float(np.median(words))
    def score(row: dict) -> float:
        return (abs(int(row["words"]) - median) / max(median, 1.0)
                + (1 - float(row["vision_coverage"])) + (1 - float(row["face_coverage"])))
    return min(usable, key=score)


def plot_typical(row: dict, features_dir: Path, output: Path, chinese: bool) -> None:
    import matplotlib.pyplot as plt

    payload = np.load(features_dir / f"{row['video_id']}_{row['clip_id']}.npz", allow_pickle=False)
    words = [str(value) for value in payload["words"]]
    times = payload["times"]
    audio = payload["audio"]
    prosody = payload["prosody"]
    audio_mask = payload["audio_mask"].astype(bool)
    face_mask = payload["face_mask"].astype(bool)
    pose_mask = payload["pose_mask"].astype(bool)
    vision_mask = payload["vision_mask"].astype(bool)
    nearest = payload["vision_nearest_mask"].astype(bool)

    text = (lambda zh, en: zh if chinese else en)
    # 样本编号含 `$`，必须转义，否则 matplotlib 会按 mathtext 解析。
    ident = str(row["id"]).replace("$", r"\$")
    figure, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True,
                                gridspec_kw={"height_ratios": [1.0, 1.2, 0.8]})
    span = float(times[:, 1].max())

    energy = audio[:, 40]
    axes[0].bar((times[:, 0] + times[:, 1]) / 2, energy, width=np.maximum(times[:, 1] - times[:, 0], 1e-3) * 0.9,
                color=np.where(audio_mask, "#2b6cb0", "#cbd5e0"))
    axes[0].set_ylabel(text("语音能量 log-RMS", "audio log-RMS"))
    axes[0].set_title(text(
        f"典型样本 {ident}：词数 {row['words']}，时长 {float(row['seconds']):.2f}s，"
        f"平均对齐分数 {row['mean_alignment_score']}，灰色柱=语音缺失词位",
        f"typical sample {ident}: {row['words']} words, {float(row['seconds']):.2f}s, "
        f"mean alignment {row['mean_alignment_score']}, grey bar = missing speech"))

    for index, word in enumerate(words):
        axes[1].barh(0, times[index, 1] - times[index, 0], left=times[index, 0], height=0.5,
                     color="#805ad5" if audio_mask[index] else "#e2e8f0", edgecolor="white")
        axes[1].text(times[index, 0], 0.42, word, rotation=45, ha="left", va="bottom", fontsize=7)
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_yticks([])
    axes[1].set_ylabel(text("文本词区间", "word spans"))

    for index in range(len(words)):
        marker = "o" if vision_mask[index] else "x"
        axes[2].plot((times[index, 0] + times[index, 1]) / 2, 1.0, marker,
                     color="#2f855a" if face_mask[index] else "#718096", markersize=5)
        if pose_mask[index]:
            axes[2].plot((times[index, 0] + times[index, 1]) / 2, 0.7, "s", color="#dd6b20", markersize=4)
        if nearest[index]:
            axes[2].plot((times[index, 0] + times[index, 1]) / 2, 1.3, "^", color="#d53f8c", markersize=4)
    axes[2].set_ylim(0.4, 1.6)
    axes[2].set_yticks([0.7, 1.0, 1.3])
    axes[2].set_yticklabels([text("姿态", "pose"), text("人脸", "face"), text("借用最近帧", "nearest frame")])
    axes[2].set_xlabel(text("时间（秒）", "time (s)"))
    axes[2].set_xlim(0, span * 1.02)

    rate = float(payload["utterance_speech_rate_wps"])
    figure.suptitle(text(
        f"词级三模态对齐（{len(words)} 词，语速 {rate:.2f} 词/秒，"
        f"视觉覆盖 {float(row['vision_coverage']):.2f}）",
        f"word-level trimodal alignment ({len(words)} words, {rate:.2f} w/s, "
        f"vision coverage {float(row['vision_coverage']):.2f})"), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs" / "features_q1_face_pose")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="默认与 --features-dir 相同")
    parser.add_argument("--sample", default=None,
                        help="典型样本 id（video_id$_$clip_id），默认自动挑选")
    parser.add_argument("--max-samples", type=int, default=1,
                        help="为前 N 条样本各出一张图；0 表示只为自动挑选的样本出一张")
    args = parser.parse_args()
    features_dir = args.features_dir
    output_dir = args.output_dir or features_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = features_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows = read_summary(manifest)
    write_csv(output_dir / "q1_summary.csv", rows, SUMMARY_FIELDS)

    coverage = []
    for name, total in (("text", len(rows)), ("audio", len(rows)),
                        ("prosody", len(rows)), ("face", len(rows)),
                        ("pose", len(rows)), ("vision", len(rows))):
        field = {"text": "words", "audio": "audio_dim", "prosody": "prosody_dim",
                 "face": "face_coverage", "pose": "pose_coverage", "vision": "vision_coverage"}[name]
        values = [float(row[field]) for row in rows if row[field] not in ("", None)] \
            if "_coverage" in field else []
        zero = sum(1 for row in rows if row[field] not in ("", None) and float(row[field]) == 0) \
            if "_coverage" in field else 0
        coverage.append({"modality": name, "n": total,
                         "mean_coverage": round(float(np.mean(values)), 4) if values else "",
                         "samples_with_zero_coverage": zero if values else "",
                         "note": "词区间为对齐单元" if name == "text" else ""})
    write_csv(output_dir / "q1_coverage.csv", coverage, ("modality", "n", "mean_coverage",
                                                         "samples_with_zero_coverage", "note"))

    chinese = configure_font()
    if not chinese:
        print("warning: no CJK font found, falling back to English labels")
    if args.sample:
        chosen = [row for row in rows if row["id"] == args.sample]
        if not chosen:
            raise SystemExit(f"sample not found in manifest: {args.sample}")
        targets = chosen
    elif args.max_samples and args.max_samples > 0:
        targets = [row for row in rows if row["status"] == "ok"][:args.max_samples]
    else:
        targets = [pick_typical(rows)]
    for row in targets:
        if row["status"] != "ok":
            continue
        output = output_dir / f"q1_typical_{row['video_id']}_{row['clip_id']}.png"
        plot_typical(row, features_dir, output, chinese)
        print(f"wrote {output}")
    print(f"wrote {output_dir / 'q1_summary.csv'} ({len(rows)} rows) and q1_coverage.csv")


if __name__ == "__main__":
    main()
