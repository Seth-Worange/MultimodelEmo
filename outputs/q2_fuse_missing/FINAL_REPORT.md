# Q2 冻结模型最终验证报告

## 协议与结论

- 最终结构：**B0 FUSE**；输入归一化关闭，结构化缺失训练关闭；学习率 **1e-3**。
- 正式种子：2026、2027、2028；每个 seed 最多 50 epoch，patience=8；按 clean/local/whole/interval 验证 selection score 选 checkpoint。
- 缺失鲁棒套件：每个最终 checkpoint 在同一 728 条验证样本上运行 50 个缺失场景；固定比较行 R1–R5 见下表。
- 本报告所有新计算仅使用 validation。没有读取 test 标签，也没有用 test 选择模型。
- 训练环境：PyTorch 2.8.0+cu126，CUDA 12.6，NVIDIA GeForce RTX 4060 Laptop GPU。

**筛选结论保持不变：B1（输入归一化）与 B2（结构化缺失增强）均为负结果，未进入最终模型。**学习率筛选后固定为 1e-3。最终 checkpoint 已在当前 CUDA 环境下用冻结配置重建；由于原交接副本未携带 Final 三个 checkpoint，本报告将重建后的同环境三 seed 作为正式汇总。与交接记录的 seed=2026/2027 单次指标存在环境/运行差异，见下方 historical 对比，不混用两套 checkpoint 指标。

## Discovery 与学习率筛选

下表保留此前 discovery 记录，不与本次三 seed 最终重训混为一谈。Robust-F1/MAE 是固定 R1–R5 的均值。

| 候选 | Selection score | Clean Macro-F1 | Clean MAE | Robust-F1 | Robust-MAE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.2941 | 0.6303 | 0.6112 | 0.6032 | 0.6155 |
| B1 | 0.3124 | 0.6052 | 0.6465 | 0.5690 | 0.6731 |
| B2 | 0.3109 | 0.6035 | 0.6362 | 0.5714 | 0.6549 |

| 学习率 | Selection score |
|---:|---:|
| 3e-4 | 0.2960 |
| 5e-4 | 0.2954 |
| 1e-3 | 0.2941 |
| 2e-3 | 0.3045 |

B1、B2 的结果低于 B0：B1 将 Clean Macro-F1 从 0.6303 降至 0.6052，Robust-F1 从 0.6032 降至 0.5690；B2 的 Clean Macro-F1 为 0.6035、Robust-F1 为 0.5714。因此不保留这两项改动。

## 最终三 seed 验证结果

| Seed | Best epoch | Selection score | Clean Macro-F1 | Clean MAE | Clean Neutral F1 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 2 | 0.2983 | 0.6085 | 0.5779 | 0.5076 |
| 2027 | 4 | 0.3053 | 0.6005 | 0.5753 | 0.4360 |
| 2028 | 5 | 0.3141 | 0.5849 | 0.5869 | 0.4795 |

| 验证场景 | Macro-F1（mean ± SD） | MAE（mean ± SD） | Neutral F1（mean ± SD） |
|---|---:|---:|---:|
| clean | 0.5980 ± 0.0120 | 0.5800 ± 0.0061 | 0.4744 ± 0.0361 |
| R1 | 0.5909 ± 0.0172 | 0.5814 ± 0.0082 | 0.4821 ± 0.0225 |
| R2 | 0.5112 ± 0.0398 | 0.6789 ± 0.0333 | 0.3441 ± 0.1013 |
| R3 | 0.5924 ± 0.0076 | 0.5818 ± 0.0057 | 0.4646 ± 0.0418 |
| R4 | 0.5570 ± 0.0063 | 0.6226 ± 0.0074 | 0.3812 ± 0.0660 |
| R5 | 0.5913 ± 0.0209 | 0.5822 ± 0.0085 | 0.4926 ± 0.0121 |
| R1-R5_mean | 0.5685 ± 0.0112 | 0.6094 ± 0.0106 | 0.4329 ± 0.0393 |

R1–R5 定义为：R1 音频+视觉 40% 中段缺失；R2 文本 40% 中段缺失；R3 视觉 40% 中段缺失；R4 文本+视觉 40% 中段缺失；R5 音频+视觉 60% 中段缺失。R1–R5_mean 是五行等权平均。标准差使用样本标准差（ddof=1），仅描述三个训练 seed 的离散程度，不解释为置信区间。

## 与 historical FUSE 的配对比较

历史参照为同工程中先前的 FUSE seed=2026/2027 checkpoint，采用相同验证样本、缺失掩码与 R1–R5 定义重新评估。每个 seed 与同 seed 历史 checkpoint 配对，差值定义为“最终 B0 − historical FUSE”；Macro-F1 正值代表提升，MAE 负值代表改善。

| 场景 | 指标 | 配对差值均值 ± SD（n=2 seed pairs） |
|---|---|---:|
| clean | macro_f1 | -0.0078 ± 0.0055 |
| clean | mae | -0.0159 ± 0.0265 |
| R1-R5_mean | macro_f1 | -0.0074 ± 0.0130 |
| R1-R5_mean | mae | -0.0096 ± 0.0085 |

该 paired delta 是两个同 seed 模型对的描述性差值，不做显著性检验。历史 checkpoint 的训练配置与最终 B0 不完全相同：历史 run 使用 corruption probability=0.8、auto missing mode，最终 B0 使用 0.0、whole mode。因此该对比反映历史整体训练方案与冻结方案的差别，不能单独归因于某一结构改动。完整逐 seed 差值见 `paired_delta.csv`；历史模型的完整 50-case 结果保存在 Final 输出目录。

## 中性类别表现

| 场景 | Neutral F1（mean ± SD，3 seeds） |
|---|---:|
| Clean | 0.4744 ± 0.0361 |
| Text interval 40% (R2) | 0.3441 ± 0.1013 |

完整的 Negative/Neutral/Positive F1 按 seed 与 R1–R5 场景列于 `neutral_f1_analysis.csv`。文本中段缺失会显著考验模型对非文本模态的依赖能力；中性类别的 F1 应与总体 Macro-F1 一起解读，避免被总体准确率掩盖。

## 回归幅度收缩

按真实情感强度绝对值分箱，在 clean validation 输入上比较真实与预测的绝对强度。`prediction − target < 0` 表示预测幅度被压小；ratio 小于 1 也表示低估。表中预测值为三 seed 均值 ± 样本标准差。

| 真实强度区间 | n / seed | 平均真实幅度 | 平均预测幅度 | 预测−真实 | 幅度比 |
|---|---:|---:|---:|---:|---:|
| near-neutral (0–0.5) | 331 | 0.148 | 0.351 ± 0.028 | +0.204 | 2.382 |
| weak (0.5–1) | 118 | 0.667 | 0.496 ± 0.033 | -0.171 | 0.744 |
| moderate (1–2) | 205 | 1.295 | 0.728 ± 0.036 | -0.567 | 0.562 |
| strong (2–3) | 74 | 2.255 | 1.017 ± 0.055 | -1.238 | 0.451 |

按 seed 的原始数值见 `regression_magnitude_shrinkage.csv`，图见 `regression_magnitude_shrinkage.pdf`。混淆矩阵汇总三个 seed 在 clean validation 上的 2,184 次样本预测，按真实类别行归一化，原始计数见 `confusion_matrix.csv`。

![三 seed clean validation 混淆矩阵](confusion_matrix_final.png)

![Clean validation 情感幅度回归收缩](regression_magnitude_shrinkage.png)

## 可复现性与限制

- 冻结配置：`config/q2_fuse_b0.yaml`，命令：`python -m scripts.finalize_q2_fuse --data-root data --device cuda`；三 seed 训练命令在 `HANDOFF_FOR_TEAMMATE.md` 中。
- 指标文件、50-case robustness CSV、混淆矩阵、Neutral F1 和幅度分箱 CSV 均由验证集计算生成。
- seed=2028 的最终 checkpoint best epoch 与训练日志一致；所有模型均按 selection score 保存 best checkpoint。
- 文本缺失压力场景下性能下降，说明模型仍有文本依赖；Neutral F1 与强度幅度收缩均需结合分箱结果报告。
- 推送（push）：**NO**；合并（merge）：**NO**；PR：**NO**。
