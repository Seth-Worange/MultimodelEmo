"""从实际输出生成论文用表格、诊断图和实验记录。"""

import csv
import json
import hashlib
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


ROOT = Path("outputs/final_experiments")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def table(rows, fields):
    return "|" + "|".join(fields) + "|\n|" + "|".join(["---"] * len(fields)) + "|\n" + "\n".join(
        "|" + "|".join(f"{float(row[field]):.4f}" if field != "name" else row[field]
                         for field in fields) + "|" for row in rows)


def save_plot(name):
    plt.tight_layout()
    plt.savefig(ROOT / "figures" / f"{name}.png", dpi=300)
    plt.savefig(ROOT / "figures" / f"{name}.pdf")
    plt.close()


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    rows = read_csv(ROOT / "model_comparison.csv")
    families = [next(row for row in rows if row["name"] == name) for name in (
        "q2_fuse_lowaux_s2026", "gate_norm_s2026", "cica_softca_bigru_s20260924")]
    fields = ["name", "selection_score", "clean_accuracy", "clean_macro_f1", "clean_mae", "clean_pearson"]
    ablations = [row for row in rows if row["name"].startswith("ab_")]
    extra = ROOT / "q2/ab_noavailability_s2026/validation.json"
    if extra.is_file():
        value = json.loads(extra.read_text(encoding="utf-8"))
        ablations.append({"name": "ab_noavailability_s2026", "selection_score": value["selection"]["score"],
            **{f"clean_{key}": item for key, item in value["clean"].items()}})
    for name, data in (("family_comparison", families), ("ablation_comparison", [families[0], *ablations])):
        with (ROOT / "q2" / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(data)
    mainline = families[0]
    display_names = {"q2_fuse_lowaux_s2026": "FUSE", "gate_norm_s2026": "Gated fusion",
        "cica_softca_bigru_s20260924": "CICA", "ab_nonorm_s2026": "Without input normalization",
        "ab_nofactor_s2026": "Without factorization", "ab_nodynamics_s2026": "Without audio differences",
        "ab_noaux_s2026": "Without auxiliary losses", "ab_noaug_s2026": "Without missingness augmentation",
        "ab_aux1_s2026": "Auxiliary scale = 1.0", "ab_noavailability_s2026": "Without availability embedding"}
    for title, data in (("model_comparison", families), ("ablation", [mainline, *ablations])):
        plt.figure(figsize=(7, max(2.8, .42 * len(data))))
        y = np.arange(len(data))
        plt.scatter([float(row["selection_score"]) for row in data], y, c="#0072B2", s=45)
        plt.yticks(y, [display_names[row["name"]] for row in data])
        plt.xlabel("Validation selection score (lower is better)")
        plt.gca().invert_yaxis()
        save_plot(title)
    robustness = read_csv(ROOT / "q2/robustness.csv")
    if len(robustness) != 165:
        raise ValueError("鲁棒性网格尚未完成")
    for location in ("start", "middle", "end", "random"):
        subset = [row for row in robustness if row["pattern"] == "interval" and row["location"] == location]
        names = list(dict.fromkeys(row["missing_modalities"] for row in subset))
        for metric in ("delta_macro_f1", "delta_mae"):
            values = np.array([[float(next(row for row in subset if row["missing_modalities"] == name
                and float(row["missing_rate"]) == rate)[metric]) for rate in (.1, .2, .4, .6, .8)] for name in names])
            plt.figure(figsize=(6.5, 4.2))
            lim = max(abs(values.min()), abs(values.max()), 1e-5)
            plt.imshow(values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            plt.yticks(range(len(names)), [" + ".join({"text": "Text", "audio": "Audio", "vision": "Vision"}[key] for key in name.split("+")) for name in names])
            plt.xticks(range(5), ["10%", "20%", "40%", "60%", "80%"])
            plt.xlabel("Requested missing fraction")
            plt.title(f"{location.capitalize()} interval missingness")
            plt.colorbar(label="Macro-F1 change from clean" if metric == "delta_macro_f1" else "MAE change from clean")
            for i in range(len(names)):
                for j in range(5):
                    plt.text(j, i, f"{values[i,j]:+.3f}", ha="center", va="center", fontsize=8,
                             color="white" if abs(values[i,j]) > .6 * lim else "black")
            save_plot(f"{location}_{metric}")
    predictions = read_csv(ROOT / "q2/q2_fuse_lowaux_s2026/predictions.csv")
    labels = np.array([int(row["label_class"]) for row in predictions])
    truth = np.array([float(row["label_strength"]) for row in predictions])
    diagnosis = {}
    for view in ("clean", "local", "interval"):
        pred = np.array([int(row[f"{view}_class"]) for row in predictions])
        strengths = np.array([float(row[f"{view}_strength"]) for row in predictions])
        matrix = confusion_matrix(labels, pred, labels=[0, 1, 2])
        diagnosis[view] = {"confusion_matrix": matrix.tolist(),
            "per_class_f1": f1_score(labels, pred, labels=[0, 1, 2], average=None).tolist(),
            "strong_mae": float(np.abs(strengths[abs(truth) > 1] - truth[abs(truth) > 1]).mean()),
            "strong_mean_pred_abs": float(abs(strengths[abs(truth) > 1]).mean()),
            "strong_mean_true_abs": float(abs(truth[abs(truth) > 1]).mean())}
        plt.figure(figsize=(4.5, 4))
        plt.imshow(matrix, cmap="Blues")
        plt.xticks(range(3), ["Negative", "Neutral", "Positive"])
        plt.yticks(range(3), ["Negative", "Neutral", "Positive"])
        plt.xlabel("Predicted class"); plt.ylabel("True class")
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
        save_plot(f"confusion_{view}")
    (ROOT / "q2/error_analysis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    selection = json.loads((ROOT / "q3/selection.json").read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "q3/validation_explanation/explanation_audit.json").read_text(encoding="utf-8"))
    report = "# 问题二、三最终实验记录\n\n"
    report += "## 数据与评价协议\n\n所有学习仅用附件2训练标签；选模与诊断使用附件2验证集728条，不使用test选模。附件3、4只做无标签推理。沿用aligned版本和冻结BERT。CUDA环境，扰动种子2026；固定每个样本的缺失掩码，完整报告Accuracy、Macro-F1、MAE与Pearson。无中性强度归零后处理。\n\n"
    report += "## Q2主线架构\n\n选用q2_fuse_lowaux_s2026。文本768维，语音74维及相邻差分，视觉35维，分别投影到128维并经三路BiGRU。声画用训练集统计量标准化，零缺失位保留掩码。每模态投影到shared/private/noise三个子空间；样本调制、因子系数和分支注意力在可用性约束下动态聚合，noise分支另用门控抑制。拼接三个聚合因子，经投影、可用模式嵌入、融合BiGRU和时间注意力池化，输出三分类概率与强度。子空间名称是设计意图，不能据名称证明获得了纯粹的语义分解。\n\n"
    report += "训练目标为0.35完整输入监督+0.65增强输入监督+0.05一致性+0.1辅助正则；监督为加权交叉熵与SmoothL1。辅助正则包含对比、信息监督、对偶、变分重建及交叉重建，内部权重见配置。AdamW学习率0.001、batch32。软强度为3*sigmoid(幅度)*(正向概率-负向概率)。\n\n"
    report += "## 三类方法比较\n\n" + table(families, fields) + "\n\nFUSE在主分和完整输入F1/MAE方面均优于门控与CICA。这里只比较各方法已有检查点在统一协议下的结果，训练范式不同，不称为单变量消融。门控和CICA各候选完整记录保留于CSV。\n\n"
    report += "## 模块消融\n\n" + table([mainline, *ablations], fields) + "\n\n既有六组消融重新评估，新增无可用模式嵌入重新训练。nofactor同时移除了因子相关辅助路径，因此是结构组合消融。noaux与主线差距很小，不能以单种子宣称显著收益；所有负结果保留。\n\n"
    report += "## 缺失规律实验\n\n165组：完整输入1组、整模态压力4组、声画短游程20组、连续区间140组。连续区间完整交叉七种非空模态组合、四种位置和五档比例。多模态位置实验关闭随机重叠锚点。每组报告F1/MAE及相对clean的差值，并记录各模态实际缺失率。local的局部区域限制可能无法实现高请求比例，不能将其标称80%当作实际80%；主要比例规律依据interval网格。位置与比例只在有限验证样本上描述，不能强求曲线严格单调。\n\n"
    report += "## Q3主线与解释验证\n\n完整输入单视图选模，最终选定：" + selection["name"] + "。候选包括三类预测器及完整输入适配、时序交互和残差骨干；最终选择只依据有标签验证集。\n\n"
    report += "验证解释抽取每类至多24条，共" + str(audit["n"]) + "条，固定种子，分别报告预测正确和错误样本的证据删除与条件增益。保留signed Shapley、文本Early/Late、等有效长度随机同模态窗口、随机输出头诊断。该分层样本不是全验证集的自然分布，不能用其准确率替代728条基础评价。窗口按删除量筛选，正删除及优于随机存在选择优势，不构成独立显著性证据。\n\n"
    report += "完整输入候选结果：\n\n" + table([row for row in rows if row["task"] == "q3"], fields) + "\n\n"
    report += "解释复核摘要：\n```json\n" + json.dumps(audit, ensure_ascii=False, indent=2) + "\n```\n\n"
    report += (f"验证解释显示，预测错误的{audit['incorrect']['n']}条也有平均"
               f"{audit['incorrect']['mean_deletion']:.4f}的正删除收益和"
               f"{audit['incorrect']['mean_conditional_gain']:.4f}的条件增益。"
               "忠实解释可以解释错误预测，不能把强证据敏感性等同于真实情绪理解。"
               f"文本Early/Late方向一致率为{audit['early_late_agreement']:.4f}。\n\n")

    report += "附件4进一步运行语义闭包、条件充分性、统一时间事件及partial mapping检查；原视频的不可映射词位不补造时间。解释卡未确认状态表示证据检验尚不足，不能推断预测必错。没有人工证据标注，不能报告解释准确率。\n\n"
    special = json.loads((ROOT / "q3/predictions/q3_selected_evidence_summary.json").read_text(encoding="utf-8"))
    report += (f"附件4共{special['n_samples']}条解释记录，严格随机窗口规则确认{special['confirmed']}条，"
               "其余不自动确认为最终支持证据。短视频可用非重叠随机窗口不足19个时，经验检验无法达到0.05阈值；这不是证据已经被证伪。\n\n")
    report += "## 复现与支撑材料\n\n配置：config/final_experiments.yaml。依次运行scripts.final_experiments、scripts.final_q3_experiments、scripts.summarize_final_experiments。manifest和各日志保留命令、分支、源码提交及检查点SHA256；新增消融有run.json和逐轮metrics.csv。图表来自真实CSV，单种子不绘制伪造误差条，单检查点跨检查点稳定性为空而非1。未执行push、merge或PR。\n"
    (ROOT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    Path("analysis/q2_q3_final_experiments.md").write_text(report, encoding="utf-8")
    source_hashes = {}
    with ZipFile(ROOT / "source_snapshot.zip", "w") as archive:
        for directory in ("scripts", "model", "utils", "config", "tests"):
            for path in sorted(Path(directory).rglob("*")):
                if path.suffix in (".py", ".yaml"):
                    source_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
                    archive.write(path, path.as_posix())
    (ROOT / "source_manifest.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    print("Saved final report and figures")


if __name__ == "__main__":
    main()
