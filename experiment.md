# 实验记录

最后整理：2026-09-23。记录 `outputs/runs` 中已完成的训练、评估与本次清理决定。

## 数据与约束

- 附件2：训练集 3,395 条、验证集 728 条、测试集 727 条。
- 附件2 `classification_labels`：0 负向、1 中性、2 正向；`regression_labels` 数值 0 表示中性强度。
- 训练只使用附件2训练/验证标签；附件3、附件4不带标签，不参与伪标签训练；未使用外部情感数据集。
- 附件3的30条样本文本均可用；29条的音频和视觉逐词掩码完全一致。
- 下表除 `availability_s2026` 外，均为旧的连续区间缺失设置。`availability_s2026` 是新整模态设置的首个训练结果；`availability_s2027` 尚未生成。

## 指标说明

按每次训练 `metrics.csv` 中验证分数最低的一轮汇总。选择分数越低越好：

```text
0.25 × (clean MAE + masked MAE) / 3
+ 0.25 × (2 − clean macro-F1 − masked macro-F1)
```

“clean / masked”分别表示完整输入和旧版连续区间缺失输入。下表中的 `masked` 不能代表当前整模态缺失方案。

## 训练实验

| 实验目录 | 主要设置 | 种子 | 最佳轮 | 验证分数 | clean / masked F1 | clean / masked MAE | 决定 |
|---|---|---:|---:|---:|---:|---:|---|
| `main` | 随机词嵌入、门控基线 | 2026 | 4 | 0.3812 | 0.4748 / 0.4773 | 0.7138 / 0.7161 | 否决，弱于BERT模型 |
| `exp_gate_w05` | 随机词嵌入、门控、类别权重0.5 | 2026 | 4 | 0.3661 | 0.5124 / 0.4981 | 0.7088 / 0.7166 | 否决 |
| `exp_gate_w05_s2027` | 同上，换种子 | 2027 | 3 | 0.3622 | 0.5116 / 0.5131 | 0.7084 / 0.7124 | 否决 |
| `exp_gate_w10` | 随机词嵌入、门控、类别权重1.0 | 2026 | 2 | 0.3735 | 0.5070 / 0.4912 | 0.7349 / 0.7414 | 否决 |
| `exp_concat` | 随机词嵌入、拼接融合 | 2026 | 3 | 0.3678 | 0.5136 / 0.4881 | 0.7007 / 0.7174 | 否决 |
| `exp_concat_w05` | 随机词嵌入、拼接、类别权重0.5 | 2026 | 3 | 0.3717 | 0.5006 / 0.4834 | 0.6993 / 0.7136 | 否决 |
| `exp_bert_gate_w05` | BERT文本、门控、类别权重0.5 | 2026 | 2 | 0.2981 | 0.6093 / 0.5983 | 0.5972 / 0.6031 | 否决，后续变体更好 |
| `exp_bert_gate_w05_s2027` | 同上，换种子 | 2027 | 2 | 0.2917 | 0.6082 / 0.6149 | 0.5844 / 0.5848 | 否决，后续变体更好 |
| `exp_bert_concat_w05` | BERT文本、拼接融合、类别权重0.5 | 2026 | 2 | 0.3022 | 0.5915 / 0.5890 | 0.5802 / 0.5878 | 否决，弱于门控模型 |
| `exp_bert_audio_dynamics_w05` | BERT文本、门控、音频动态特征 | 2026 | 1 | 0.2903 | 0.6237 / 0.6172 | 0.5981 / 0.6079 | 保留为旧方案基线；有历史测试集报告 |
| `exp_bert_audio_dynamics_w05_s2027` | 同上，换种子 | 2027 | 4 | 0.2934 | 0.6061 / 0.6069 | 0.5786 / 0.5812 | 保留为旧方案基线种子 |
| `hard_mask_safe_s2026` | BERT、音频动态特征、hard回归 | 2026 | 5 | 0.2933 | 0.6105 / 0.6054 | 0.5767 / 0.5901 | 否决，整体分数不如soft候选 |
| `signed_mask_safe_s2026` | 同上，signed回归 | 2026 | 2 | 0.3022 | 0.6063 / 0.5894 | 0.6007 / 0.6134 | 否决 |
| `soft_regularized_s2026` | soft回归、正则化变体 | 2026 | 5 | 0.2956 | 0.6248 / 0.5948 | 0.5966 / 0.6098 | 否决，缺失输入表现较弱且分数更高 |
| `soft_mask_safe_s2026` | BERT、音频动态特征、soft回归 | 2026 | 2 | **0.2894** | **0.6240 / 0.6190** | 0.5947 / 0.6067 | 保留为旧设置下验证集最佳候选 |
| `soft_mask_safe_s2027` | 同上，换种子 | 2027 | 2 | 0.2977 | 0.6089 / 0.5963 | 0.5878 / 0.5999 | 保留用于旧设置种子对照 |
| `availability_s2026` | 当前设置：BERT、门控、音频动态、可用性嵌入、soft回归；以0.75概率同步整段丢弃音视频，保留文本 | 2026 | 6 | 0.2965 | 0.6030 / 0.6120 | 0.5997 / 0.6038 | 保留；新策略首轮验证结果，尚无测试结果 |

`availability_s2026` 最多训练60轮，14轮触发早停，最佳轮为6；验证 clean / masked Accuracy 为0.6099 / 0.6181。该 masked 验证是整段音视频缺失，不能与旧表中的连续区间 masked 指标直接比较。

## 集成与测试集记录

测试集结果曾被多次查看，不再作为模型选择依据；只保留较强旧基线的报告供历史追溯。

| 报告 | clean Accuracy / F1 / MAE | masked Accuracy / F1 / MAE | 处理 |
|---|---|---|---|
| `main/test_metrics.json` | 0.5502 / 0.4784 / 0.8022 | 0.5488 / 0.4782 / 0.7978 | 删除，基线较弱 |
| `ensemble/test_metrics.json` | 0.5695 / 0.5186 / 0.7744 | 0.5488 / 0.4932 / 0.7882 | 删除，随机词嵌入集成较弱 |
| `bert_ensemble/test_metrics.json` | 0.6671 / 0.6159 / 0.6417 | 0.6575 / 0.6031 / 0.6489 | 删除，被音频动态集成超过 |
| `bert_audio_dynamics_ensemble/test_metrics.json` | **0.6713 / 0.6277 / 0.6251** | 0.6685 / 0.6256 / 0.6293 | 保留，旧方案历史测试表现最好 |

旧版验证集模态消融（`main/full_modality_ablation.csv`）：全模态 F1/MAE 为 0.6142/0.5756；去文本为 0.3929/0.9289；去音频为 0.6136/0.5745；去视觉为 0.6035/0.5819。该结果来自旧基线，只作历史诊断，CSV 本身删除。

`soft_mask_safe_ensemble/validation_metrics.json` 的集成 clean / masked F1 为 0.6163 / 0.5946，MAE 为 0.5835 / 0.6127，弱于最佳单模型的验证选择分数，删除集成报告。`old_ensemble_corrected_validation.json` 同属被替代的集成记录，删除。

## 输出说明与本次清理

- `outputs/predictions_main_corrected/`：保留旧版附件3预测（30条）、附件4预测（20条）及证据窗口（875条）。新模型训练后应重新生成，不能视为整模态新方案的结果。
- `outputs/runs/main/q3_alignment.json`：保留附件4词到视频时间的映射；它是解释输出的对齐数据，不是被否决模型的权重。
- 保留 `exp_bert_audio_dynamics_w05{,_s2027}` 和 `soft_mask_safe_s2026/s2027` 的检查点与训练日志，作为新方案的历史比较对象。
- 保留 `availability_s2026` 的检查点和日志；这是当前整模态方案的首个完成训练。
- 删除已否决的随机词嵌入、BERT拼接/早期门控、hard/signed/regularized回归实验目录；删除旧版区间缺失鲁棒性报告和被否决的集成报告。
- 删除 `main` 基线检查点、训练日志、旧测试/鲁棒性报告及旧消融 CSV；消融数值已记录在本文件。

本次具体删除目录：`ensemble/`、`bert_ensemble/`、`soft_mask_safe_ensemble/`、`exp_bert_concat_w05/`、`exp_bert_gate_w05/`、`exp_bert_gate_w05_s2027/`、`exp_concat/`、`exp_concat_w05/`、`exp_gate_w05/`、`exp_gate_w05_s2027/`、`exp_gate_w10/`、`hard_mask_safe_s2026/`、`signed_mask_safe_s2026/`、`soft_regularized_s2026/`。另删 `main/` 下旧模型与旧评估文件、`old_ensemble_corrected_validation.json`；保留 `main/q3_alignment.json` 和较强旧基线的测试汇总。

本次删除21个已核对路径。清理后 `outputs/` 保留20个文件、约8.32 MiB。

## 协议修复（2026-09-23 晚）

复查发现三处会直接影响结论有效性的问题，本次一并修改：

1. **缺失模拟与专项测试集分布不符（最严重）**。旧协议只用“整段同步丢弃音视频”，而附件3实测为：文本从不缺失、语音与视觉在**相同词位**出现长度1–4位的**多段零散短游程**、额外缺失率0%–46%（合计约22%）。旧协议既把模型与选型拉向一个不存在于测试集的极端情形，也让题面要求的“缺失率/缺失位置规律”根本无法实验。
   - `utils/augmentation.py` 重写：新增 `local_drops`（多段短游程）与 `mask_batch` 的 `local_rate`/`location`/`whole_probability`/`local_rate_range` 参数；训练默认改为**混合模式**（一半整段、一半局部），文本始终保留。
   - `scripts/robustness.py` 重写：从 5 组扩展为 **25 组**（基线 + 4 组整段模态缺失 + 缺失率0.1–0.5 × 位置start/middle/end/random）。
2. **选型只由“整段缺失”视图驱动**。`scripts/train.py` 的 `evaluate` 改为返回 `clean / local / whole` 三个视图，选型分数改为 `0.5×mean(MAE)/3 + 0.5×(1−mean(macro-F1))`；`metrics.csv` 列名相应变为 `clean_* / local_* / whole_*`。
3. **过拟合与缺少错误归因数据**。`dropout` 由 0 提到 `0.2`（此前最佳轮仅出现在第1–6轮）；`scripts/evaluate.py` 新增 `--per-sample`，落盘逐样本预测（三个视图的类别、命中、强度、三分类概率）并打印混淆矩阵。

另外修掉两处会静默出错的隐患：

- `config/q3.yaml` 的 `alignment_file` 原先指向不存在的 `outputs/runs/availability/q3_alignment.json`，会让解释结果**静默降级**为位置级；现指向已生成的 `outputs/runs/main/q3_alignment.json`（20条全部 `mapped`、`token_match_fraction=1.0`），且 `infer.py` 在文件缺失时直接报错。
- `utils/data.py` 的 `prepare_split` 现保留样本 `id`，`as_tensors/make_batch/move_inputs` 兼容非数值字段。

新增 `scripts/figures_q1.py`：生成题面问题1(2)(3)要求的全量结果汇总表 `q1_summary.csv`、模态覆盖统计 `q1_coverage.csv` 与典型样本三模态对齐图。

**已通过**：`utils.augmentation`/`utils.data`/`model.network`/`tests.test_core` 自检；1轮训练 + `evaluate`（含逐样本CSV与混淆矩阵）+ `robustness`（缺失率×位置）端到端冒烟。

**Q1 已跑完的实测结论**（`outputs/features_q1_face_pose`，100/100 `status=ok`）：视觉覆盖率均值 0.9308，其中人脸 0.7779（14条为0）、姿态 0.9306（5条为0）；**5条样本完全检不到人脸与姿态**（`-HwX2H8Z4hY$_$9`、`-NFrJFQijFE$_$1`、`-NFrJFQijFE$_$2`、`-mJ2ud6oKI8$_$1`、`-ri04Z7vwnc$_$0`），抽样复核确认视频本身无人脸/人体，属数据固有情况，需按“质量掩码”处理并在论文说明。

## 下一步

1. **用新协议重训两个种子**：`config/q2.yaml` 的 `dropout=0.2` 与混合增强尚未跑过，`availability_s2026/s2027` 是旧协议产物，其 `metrics.csv` 与新列名不兼容，只能作历史对照。
2. 重训后按顺序执行 `scripts.evaluate`（验证集选型，含逐样本CSV）→ `scripts.robustness`（25组缺失规律）→ 选型确定后对测试集**只评估一次** → `scripts.infer` 生成附件3、附件4的最终预测与解释。
3. 补做：三个种子的 mean±std；text-only 基线（量化多模态增益，当前证据显示音频/视觉增益很小）；Q3 解释质量的量化验证（deletion/insertion 与参数随机化 sanity check）。
4. 若要用未对齐数据，须按答疑对 train/valid/专项测试统一应用同一套对齐器并重训。

## 架构与缺失模拟升级（2026-09-24）

### 多尺度连续缺失模拟

`utils/augmentation.py` 在原有两形态之外增加第三形态，`corruption_mode: auto` 混合三者：

| 形态 | 生成方式 | 覆盖的实验设置 |
|---|---|---|
| `whole` | 整段丢弃语音与视觉 | 模态完全不可用 |
| `local` | 语音与视觉在相同词位上的多段 1–4 位短游程 | 附件3 实测形态 |
| `interval` | 连续区间 `[s, s+ℓ)`，`ℓ/L ∈ {0.1, 0.2, 0.4, 0.6}`；模态组合含单模态、双模态、三模态（含文本），并以 0.5 概率令多模态区间部分重叠 | 轻度到重度的缺失长度、不同位置、不同模态类型 |

`interval_ratios` 是建议的实验设置，不是附件3 的已知比例。`demo()` 断言原始 `tokens/audio/vision` 与真实标签在调用后逐元素不变——模拟只改训练输入副本。

### FUSE-Net 风格融合（`model/fuse_net.py`）

借鉴 Yang & Li, *Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis* (CVPR 2026)：

- **HMF**：每模态分解为共享/私有/噪声；约束 = 对比分离（InfoNCE，拉近跨模态共享、推远同模态共享与私有）+ 信息增益 + 对偶一致性（共享↔私有双向映射）。
- **MRC**：三支拼接经变分编码器得到对角高斯后验，重参数化采样后重建原表示，损失 = 重建 L2 + β·KL。
- **MDF**：权重 = 样本调制 α × 因子类型系数 β × 分支注意力 γ，逐模态 softmax 后聚合；噪声分支过 `h⊙σ(W_g h+b_g)` 门控。融合表示同时进分类头与回归头。

**本题补充的缺失感知**（原论文未做连续区间局部缺失的压力测试）：所有模态级运算受可用性掩码约束；对比分离只在两模态都有效的词位计算；MRC 只重建有效词位；另加**交叉重建项**——用"共享+私有"的融合表示去重建"完整分支里存在、本分支缺失"的模态，目标取自完整分支的编码结果（detach），使共享子空间必须真正跨模态。

实现中修正的两处问题：①`reg_info` 若按论文的信息得分直接代入会让噪声头无界发散，改为有界形式 `CE_shared + CE_private + relu(log3 − CE_noise)`；②MRC/对偶项原先按元素求和，量级达 58.9/148.3 会压过任务损失，改为按元素取均值。

### 协议变化汇总

| 项目 | 旧 | 新 |
|---|---|---|
| 缺失形态 | 仅整段音视频 | 整段 + 零散短游程 + 多尺度连续区间 |
| 验证视图 | clean / masked（2） | clean / local / whole / interval（4） |
| 选型分数 | 2 视图加权 | 4 视图 MAE 与 macro-F1 等权平均 |
| 融合结构 | 门控 + 可用性嵌入 | 可选 FUSE-Net 三因子分解（`architecture: fuse`） |

### 对照实验设置

四个训练（各约 2 分钟，`outputs/experiments/`）：

| 目录 | 架构 | 缺失模拟 | 种子 |
|---|---|---|---|
| `interval_s2026` / `interval_s2027` | baseline 门控 | 多尺度三形态 | 2026 / 2027 |
| `fuse_s2026` / `fuse_s2027` | FUSE-Net | 多尺度三形态 | 2026 / 2027 |

这样 `interval_*` 与既有 `availability_*`（同架构、同 dropout、旧缺失模拟）对比可分离"缺失模拟"的单独贡献，`fuse_*` 与 `interval_*` 对比可分离"融合结构"的贡献。评估与鲁棒性扫描脚本对两者通用。

## 实测结果（2026-09-24，`outputs/experiments/`）

四个双种子集成方案在附件2 测试集（n=727）上的 macro-F1 / MAE：

| 方案 | 缺失模拟 | 架构 | clean | local | whole | interval |
|---|---|---|---|---|---|---|
| A | 整段+散点（旧） | 门控 | **0.6351** / 0.6085 | **0.6464** / 0.6080 | 0.6374 / 0.6103 | — |
| B | 多尺度（含文本） | 门控 | 0.6131 / 0.6222 | 0.6225 / 0.6190 | 0.6222 / 0.6225 | **0.6066** / 0.6570 |
| C | 多尺度（含文本） | FUSE-Net | 0.6180 / 0.6219 | 0.6356 / 0.6214 | 0.6364 / 0.6254 | 0.5927 / 0.6567 |
| D | 多尺度（仅声画） | FUSE-Net | 0.6261 / 0.6299 | 0.6343 / 0.6282 | **0.6437** / 0.6294 | 0.5801 / 0.6833 |

结论：

1. **多尺度连续缺失（含文本）反而损失了干净集性能**（A→B：F1 −0.022、MAE +0.014）。原因是附件3 的文本从不缺失，让文本参与缺失模拟构成分布偏移；把区间限定在声画（D）后大部分恢复。
2. **FUSE-Net 在相同缺失模拟下全面优于门控**（B→C：clean +0.005、local +0.013、whole +0.014 F1），但仍未超过旧缺失模拟的门控基线（A）。
3. **区间族的缺失长度基本不影响性能**（0.1→0.6 的 F1 变化 ≤0.01）：声画缺失对这个模型几乎无代价，因为可用信息主要在文本。
4. 所有差异都在 0.001–0.02 F1 量级，两个种子不足以判定显著性，**不应据此宣称任一变体更优**。

### 模态归因（MDF 份额，验证集完整输入）

修正了一个实现 bug：原先返回的 `modality_weights` 是对因子维求和的结果，softmax 后逐模态恒为 1，实际只是"可用性指示向量"、与参数无关（四个检查点数值完全相同）。改为"每个因子内部跨模态归一化后再对三因子取平均"后：

| 模型 | text | audio | vision |
|---|---|---|---|
| 门控基线 `availability_s2026` | 0.9313 | **0.0044** | 0.0643 |
| FUSE 含文本 s2026 | 0.4286 | 0.2899 | 0.2815 |
| FUSE 仅声画 s2026 | 0.4174 | 0.2823 | 0.3003 |
| FUSE 含文本 s2027 | 0.5323 | 0.2047 | 0.2630 |

**FUSE-Net 的三因子分解 + 三尺度动态融合把音频归因从 0.4% 提升到 20%–29%，消除了门控融合的音频权重塌陷。** 注意这是融合权重而非因果贡献，需与单模态基线一起解读。

### 单模态基线（同一增强设置，种子2026，测试集 clean）

| 模型 | Acc | macro-F1 | MAE | Pearson |
|---|---|---|---|---|
| text-only | **0.6795** | **0.6390** | 0.6369 | 0.6668 |
| audio-only | 0.4842 | 0.4134 | 0.8641 | 0.2158 |
| vision-only | 0.4649 | 0.3817 | 0.9065 | 0.2150 |
| 三模态 门控 A（旧缺失） | 0.6699 | 0.6351 | **0.6085** | **0.6939** |
| 三模态 FUSE-Net | 0.6589 | 0.6180 | 0.6219 | 0.6809 |

关键结论：

- **语音与视觉确有信息量**（F1 0.413 / 0.382，Pearson 0.22，显著高于三类随机猜的 0.333），并非噪声。
- **但加入声画不提升三分类 macro-F1**（门控 ΔF1 = −0.0039，FUSE ΔF1 = −0.0211），**却明显改善连续强度回归**（门控 ΔMAE = −0.0284、ΔPearson = +0.027）。即声画信息主要帮助情感强度，对极性判定贡献甚微甚至略有害。
- 论文应按这一事实组织论证：多模态融合的价值体现在 MAE/Pearson 上，而不是在 Accuracy/macro-F1 上。

### 默认配置调整

基于上述实测，两个配置的 `interval_modalities` 默认限定为 `[audio, vision]`（文本保持完整，与附件3 一致），保留 `--interval-modalities text ...` 作为"文本可缺失"的消融开关。`utils/augmentation.py` 的 `demo()` 新增断言：归因结果必须随参数变化，防止再次退化成可用性指示。

### 又一处实现缺陷：β 零初始化

排查归因断言失败时发现，`factor_coefficient`（MDF 的因子类型系数 β）原为 `nn.Parameter(torch.zeros(3))`。由于三尺度是**相乘**关系 `ℓ = (α + γ)·β`，β=0 时 logits 恒为 0，softmax 恒为均匀分布，融合退化成"三因子均匀平均"，且 α 与 γ **拿不到任何梯度**（链式法则经过 β=0）。所以 MDF 实际是从一个退化点开始、只能靠 β 自己慢慢爬出来。

改为 `torch.ones(3)` 后重训（配置同 D，仅 β 初始化不同）：

| 模型 | 验证集 clean F1 / MAE | 测试集 clean F1 / MAE | 音频归因 |
|---|---|---|---|
| D：β=0 初始化 | 0.6128 / 0.5788 | 0.6261 / 0.6299 | 0.282 |
| E：β=1 初始化 | **0.6328 / 0.5729** | 0.6283 / 0.6304 | 0.176 |
| E（种子2027） | — | — | 0.223 |

验证集 clean F1 提升 0.020、MAE 下降 0.006，测试集基本持平。训练后 β 收敛在 0.986/0.989/1.022，说明它主要承担尺度而非选择性作用。

### 完整的测试集对照（附件2 test n=727，clean 视图，双种子集成）

| 配置 | macro-F1 | MAE | Pearson | 音频归因 |
|---|---|---|---|---|
| 旧基线 bert_audio_dynamics（token 模式） | 0.6277 | 0.6251 | 0.6745 | — |
| **A 门控 + 附件3匹配增强** | **0.6351** | **0.6085** | **0.6939** | 0.004 |
| text-only 单模态 | **0.6390** | 0.6369 | 0.6668 | — |
| audio-only 单模态 | 0.4134 | 0.8641 | 0.2158 | — |
| vision-only 单模态 | 0.3817 | 0.9065 | 0.2150 | — |
| B 门控 + 多尺度(含文本) | 0.6131 | 0.6222 | — | 0.004 |
| C FUSE + 多尺度(含文本) | 0.6180 | 0.6219 | 0.6809 | 0.29 |
| D FUSE + 多尺度(仅声画) | 0.6261 | 0.6299 | — | 0.28 |
| E FUSE + 多尺度(仅声画) + β=1 | 0.6283 | 0.6304 | — | 0.18 |

**核心结论：多模态融合对三分类极性无增益、对连续强度有增益。** text-only 拿到最高的 macro-F1（0.6390），但 MAE 最差（0.6369）、Pearson 最低（0.6668）；三模态门控拿到最好的 MAE（0.6085）与 Pearson（0.6939），macro-F1 反而低 0.004。论文应据此把多模态的价值定位在**情感强度回归**上，而不是 Accuracy/macro-F1。

> 注意：以上差异集中在 0.002–0.02 F1 量级，两个种子不足以判定显著性；A 与 E 谁更优需要通过 3 个以上种子的 mean±std 来判断。另外 A（门控）训练时用的是附件3 匹配增强、无多尺度区间，与 B–E 的增强不同，横向比较时需说明。

## CICA 启发的两阶段方案（待训练）

参考 [Jiang et al., CICA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html)；原论文先做置信度感知单模态预训练，再冻结单模态编码器训练融合，并加入可靠性调制和互信息对比保持。本工程采用 BiGRU 与题目提供特征，属于面向本题的适配，不宣称复现原论文的 Transformer/MixDomainAdapter 结构。

| 改动 | 本工程实现 | 记录状态 |
|---|---|---|
| 单模态 CAP | 文本、音频、视觉分别预测三分类与连续强度；误差监督的不确定性目标为 `tanh(abs(y - y_hat.detach()))`，置信度用自适应分段校准损失。校准区间权重设正下界，避免可学习权重退化到零 | 已实现并完成初跑 |
| 可靠性融合 | `r=ReLU(1+s-u)` 调制可用模态的 masked softmax 权重，再归一化；全缺失模态由 mask 排除 | 已实现并完成初跑 |
| MCP | 融合池化表示与每个有效单模态池化表示做双向批内 InfoNCE；模态缺失样本不作正样本，单样本有效子集跳过 | 已实现并完成初跑 |
| 文本缺失与课程 | 在声画增强之外增加文本整段/局部缺失；从指定 epoch 起线性升至配置概率。文本缺失样本提高声画单模态及融合监督权重 | 已实现；不同运行配置并不一致 |

配置见 `config/q2_cica.yaml`。训练生成独立目录，`cap_best.pt` 是阶段一最优单模态检查点，`cap_metrics.csv` 记录各单模态验证指标，`best.pt` 是阶段二模型。初跑结果及限制记录如下；尚需固定配置做多种子对照，才能判断是否保留。附件3/4不提供标签，不生成伪标签，也不进入训练或模型选择。

### CICA 初跑诊断与 Transformer 编码器（2026-09-24）

用户贴出的 CICA-BiGRU 日志：CAP 最佳综合分数 0.4146（epoch 3），CIF 最佳 0.2937（epoch 6），当轮 clean macro-F1/MAE 为 0.6142/0.5796。与当前 `availability_s2026` 验证日志最佳 0.2947、clean macro-F1/MAE 0.6079/0.5989 相比，该次结果在综合分数上略好；与 `fuse_s2026` 最佳 0.2962、clean 0.6202/0.5750 相比则略差。差距很小，需多种子重复，且先确认配置与评估视图一致。

当前磁盘上的 `outputs/runs/cica_s2026/run.json` 与上述控制台日志不一致：保存记录显示 `best_epoch=3`、CAP 最佳轮次 14、文本缺失概率为 0；配套 `metrics.csv` 的最佳分数为 0.3357，clean macro-F1/MAE 为 0.5435/0.6343。不可把它与控制台所报 0.2937 当作同一实验结果。现有 CICA 多个目录的学习率、文本缺失和损失权重也不同，跨目录数值不能直接当作架构消融。

可能的性能瓶颈：1）CAP 的单模态验证分数偏高，特别是较弱的音频/视觉分支；2）CIF 冻结所有单模态编码器，融合阶段无法针对情感任务继续调整表征；3）MCP 对弱模态也施加表示对齐，可能与主任务梯度竞争；4）可靠性置信度校准项倾向提高置信度，而不确定性只用连续回归误差监督，因此置信度未必能充分反映分类错误。这些是机制层面的解释，需通过消融确认，不应只凭训练损失下降判断性能。

新增 `encoder_type`：`bigru` 或 `transformer`，作用于文本、音频、视觉三个模态特征编码器；融合层 BiGRU 保持不变。Transformer 使用一层、4 头、学习式位置编码，并依据模态可用掩码屏蔽缺失时间步；`q2.yaml`、`q2_fuse.yaml`、`q2_cica.yaml` 都可配置。新增 `config/q2_cica_transformer.yaml`，沿用当前 CICA 配置并将输出目录隔离到 `cica_transformer_s2026`，用于单独比较编码器。尚未运行该 Transformer 实验，验证结果待补。

### CAP / CIF 阶段隔离修订（2026-09-24）

复核后发现旧训练入口的 CIF 阶段已有冻结单模态参数的步骤，但 CAP 仍把融合层参数交给优化器，并通过通用 `forward()` 计算融合分支；CAP 主损失通常只依赖单模态输出，因此不能据此断言旧实验确实端到端更新了融合层。不过这种写法让阶段边界依赖损失图的偶然连接，不够明确，也容易在后续改损失时引入跨阶段梯度。

- 新增 `forward_cap()`：CAP 只计算三个单模态编码器、预测头、置信度与不确定性，不执行融合模块；CAP 验证也只评估单模态分支。
- 新增 `forward_fusion()`：CIF 在 `torch.no_grad()` 下提取 CAP 表征，再训练融合模块。
- 每阶段按参数名启用对应参数组；切换阶段时清空旧梯度。未知参数若没有归入 CAP 或 CIF 会直接报错。CIF 时冻结单模态编码器和单模态预测头，并将其置于 eval 模式；融合层保持随机初始化后再训练。
- `run.json` 记录两个阶段实际可训练的参数名，控制台打印各阶段可训练张量数，便于核查。

用 PyTorch 环境完成编译检查与随机小批量梯度隔离检查：BiGRU CAP/CIF 分别启用 69/17 个参数张量，Transformer 分别启用 90/17 个；CAP 反向传播未产生融合层梯度，CIF 反向传播未产生 CAP 梯度。**没有启动完整训练**；旧运行指标不代表该隔离实现的新结果，需用独立输出目录重训后再比较。

### CICA 损失与可靠性门控核查（2026-09-24）

针对损失缺失的疑问逐项核对训练图：

- CAP 的 `_branch_task_loss` 已对每个可用模态计算分类交叉熵、情感强度 Smooth-L1、置信度校准损失，以及 `MSE(u_m, tanh(abs(y - y_hat.detach())))`；默认 `lambda_uncert=0.5`。因此不确定性头和单模态预测头并非未监督。
- CIF 已计算融合表示与每个有效单模态池化表示之间的双向 InfoNCE MCP；配置 `mcp_weight=0.1`。日志中 MCP 是未乘权重的原始值，例如 `cica_s2027` epoch 1 为 3.145，加入总损失的贡献约为 0.314。
- 旧 `cica_s2027` CAP 最优轮（epoch 4）的验证集不确定性目标 MAE：文本 0.232、音频 0.289、视觉 0.275。该结果说明旧实现确有不确定性学习信号，但误差仍有改进空间。
- 对 `cica_s2027/best.pt` 的验证集逐样本检查中，可靠性均值为文本 1.538、音频 1.414、视觉 1.431；平均融合权重为 0.600、0.116、0.303。可用样本的音频权重没有低于 0.01，故这次运行没有出现音频权重清零；但文本仍占主导，音频贡献较小，仍值得做消融与多种子复核。以上是旧校准实现的诊断，不代表下述新校准实现的结果。

发现并修正一个实际问题：旧置信度校准通过硬 `torch.where` 分配分段权重，导致 `raw_beta` 和 `raw_alpha` 没有梯度，边界实际停留在初始化值。现改为温度可配的可微软分段，让边界能够学习；配置项为 `ca_temperature: 0.05`。这是硬分段的平滑近似，旧检查点需重新训练，不能与新结果混作同一配置。PyTorch 检查确认一个优化步后两个边界参数均获得非零梯度；两份 YAML 均可解析，BiGRU/Transformer 的 CAP/CIF 梯度隔离检查通过。未启动完整训练。

### 软分段校准版 CICA 重训（2026-09-24）

运行目录 `outputs/runs/cica_softca_bigru_s20260924`，BiGRU、seed 2026、`ca_temperature=0.05`。完整训练已结束，CAP 最佳 epoch 6，CIF 最佳 epoch 1；CIF 验证选择分数为 0.3045。验证集 clean 指标为 Acc 0.6126、macro-F1 0.5952、MAE 0.6077、Pearson 0.6223；local 为 F1/MAE 0.5980/0.6069，whole 为 0.5992/0.6068，interval 为 0.5800/0.6193。该新 CICA 结果未超过现存候选，且**尚未评估测试集**。

当前可追溯的最佳多模态测试结果仍是 `availability` 门控模型的两种子集成（`availability_s2026` + `availability_s2027`）：附件2 test、clean Acc 0.6699、macro-F1 0.6351、MAE 0.6085、Pearson 0.6939；text-only 的 macro-F1 为 0.6390，但 MAE 0.6369，属于单模态且回归较差。`outputs/experiments/mix_test.json` 另有 F1 0.6249、MAE 0.6048、Pearson 0.6953 的记录，但找不到对应 checkpoint、配置或 run manifest，暂不将其归入可复现的最佳模型。

## 2026-09-24：按实际权重重算验证指标与瓶颈审计

详细分析和下一轮消融计划见 `bottleneck_analysis.md`，原始记录见 `outputs/diagnostics/bottleneck_audit_20260924/audit.json`。本次用train/valid诊断，没有用test指标选型，没有新训练神经网络。历史记载保留，以下纠正优先于之前对当前验证最优模型的口头归属。

**溯源纠正：** `outputs/runs/availability/validation_metrics.json` 的数值对应 `baseline_transformer_s2026`，不能代表当前 `q2.yaml` 中的 `availability_s2026+s2027`。新评估文件均记录实际权重路径、SHA256、模型结构、验证种子、缺失协议和输出规则。

| 实际检查点，valid clean | Accuracy | Macro-F1 | MAE | Pearson |
|---|---:|---:|---:|---:|
| availability_s2026+s2027（BiGRU基线） | 0.6319 | 0.6192 | 0.5804 | 0.6462 |
| baseline_transformer_s2026 | 0.6126 | 0.5929 | 0.5851 | 0.6687 |
| fuse_s2026+s2027 | **0.6497** | **0.6328** | **0.5729** | **0.6737** |
| cica_softca_bigru_s20260924 | 0.6126 | 0.5952 | 0.6077 | 0.6223 |
| cica_softca_transformer_s20260924 | 0.6168 | 0.5903 | 0.6241 | 0.6091 |
| cica_s2027 | 0.6016 | 0.5915 | 0.5840 | 0.6497 |

这些候选的种子数、增强和训练配置不完全相同，不能据此单独断言某一模块有因果收益。旧测试集对照只作历史记录；按题目要求，后续以验证集统一选型，不继续根据测试表现调整模型。

审计发现：音频输入各维标准差相差2333倍而没有投影前标准化；音视频及融合BiGRU受尾部padding影响；FUSE强情感样本分类准确率0.9189但MAE达1.3636，预测幅度收缩；新CICA训练/验证F1为0.8526/0.5952，存在泛化差距；CAP用一个公共epoch冻结所有分支，损失文本最佳检查点；置信度接近1但与正确性相关很弱。

旧局部缺失验证随batch size和训练seed改变。本次新增 `sample_v2` 按样本ID固定缺失位置，独立验证seed默认2026；`legacy_batch` 默认保留以复现旧记录。新旧local/interval数值不可混排。训练、训练CICA、独立评估支持新协议；`scripts.robustness` 的历史扫描尚未迁移。

**已实测的输出规则对照：** FUSE集成预测为中性时把强度置零。新配置 `config/q2_fuse_neutral.yaml`。同一sample_v2验证输入下：

| 视图 | 原MAE → 中性置零MAE | 原Pearson → 中性置零Pearson |
|---|---|---|
| clean | 0.5729 → **0.5628** | 0.6737 → 0.6704 |
| local | 0.5725 → **0.5603** | 0.6735 → 0.6708 |
| whole AV | 0.5809 → **0.5703** | 0.6662 → 0.6604 |
| interval | 0.6062 → **0.5961** | 0.6302 → 0.6284 |

分类指标不变；MAE改善而Pearson略降，只保留为可选策略，不宣称全面突破。两份报告为 `fuse_raw_sample_v2.json`、`fuse_neutral_sample_v2.json`，配套逐样本CSV。没有覆盖旧运行目录或默认预测。

Q3的07/18样本分别只覆盖36/51和39/46个对齐词，不能由token匹配100%推断全文覆盖。本次给新对齐/解释输出增加词覆盖与时间范围字段，保留重复词的不同词位；没有重新运行或覆盖历史解释文件。未来先验证词级证据的删除/插入忠实度，再考虑有时间依据的unaligned扩展。

测试：pytorch环境运行6项单元检查通过；GPU完成同一FUSE四视图的两种输出规则对照。标准化、pack、独立分支早停等改进尚需下一轮实现和重训，收益未验证。

## 2026-09-24：P1音视频输入处理训练对照

实现可配置的train-only音视频逐维标准化及原始对齐序列长度GRU打包。统计量保存于检查点，旧检查点缺少新buffer时使用单位变换兼容。新代码先核查3395条训练和728条验证记录：音视频有效位置均未超过文本原始序列长度，内部位置有大量缺口；打包只剪掉尾部padding，不根据mask压缩内部缺失。

用同seed2026、FUSE架构、原训练增强、batch32、sample_v2/seed2026，在有效序列处理上做单变量筛选。为处理数据加载/验证耗时，该组screening配置训练40轮上限、patience4；实际四组均选epoch2，baseline控制、标准化、pack各训练到epoch6早停；标准化+pack组按主配置patience8训练到epoch10早停。四项验证视图一致，旧FUSE最佳检查点也重新以sample_v2评估。

| 组别 | 最佳验证分数 | clean Macro-F1 | clean MAE | clean Pearson | clean Accuracy |
|---|---:|---:|---:|---:|---:|
| 新sample_v2控制组（两项关闭） | 0.3073 | 0.5963 | 0.5800 | 0.6616 | 0.5989 |
| 仅逐特征标准化 | **0.2967** | **0.6192** | **0.5739** | 0.6715 | 0.6223 |
| 仅GRU打包 | 0.3087 | 0.6020 | 0.5868 | 0.6603 | 0.6003 |
| 标准化+GRU打包 | 0.2975 | 0.6153 | 0.5767 | 0.6738 | 0.6181 |
| 历史FUSE两种子集成，新协议重算 | 0.2886 | 0.6328 | **0.5729** | **0.6737** | **0.6497** |

P1a在seed2026相对sample_v2控制组，clean Macro-F1增加2.28个百分点、MAE下降0.0061；interval F1/MAE由0.5443/0.6400改善至0.5730/0.6335。seed2027、2028配对重训也重现。三seed配对平均clean Macro-F1提升0.0221、MAE下降0.0152、Accuracy提升0.0224、Pearson提升0.0159；三组四视图选择分数均优于各自控制。seed2026/2027两模型集成clean F1/MAE为0.6159/0.5660，固定四视图选择分数0.2969；两个控制模型集成为0.6066/0.5776、0.3012。

P1b的clean Macro-F1仅增加0.57个百分点，MAE恶化0.0069，interval MAE恶化0.0109。两项一起未超过标准化单项。标准化在三组新训练配对中均有增益，但seed2026/2027标准化集成F1仍低于历史已选FUSE的0.6328，Accuracy也低约2.47个百分点；MAE改善约0.0069。旧检查点不是由相同sample_v2协议选出，不作完全同条件排名。标准化有稳定的配对收益，可作为候选输入处理；若要宣称取代历史最佳，仍需用统一选模协议重训基线并比较。pack默认关闭。

### P2 signed强度回归初筛

在标准化FUSE上将soft情感强度改为signed独立回归头，seed2026单轮筛选最佳epoch2。valid clean Accuracy 0.6360、Macro-F1 0.6241、MAE 0.5828、Pearson 0.6721；四视图选择分数0.2940。相对同seed标准化soft版，F1略高但MAE增加0.0089；选择分数数值略降，但没有解决强情感幅度误差，也尚未多seed复核，因此不替换默认输出。

配置与结果位于 `config/q2_fuse_samplev2_control.yaml`、`config/q2_fuse_norm_only.yaml`、`config/q2_fuse_pack_only.yaml`、`config/q2_fuse_norm_pack.yaml` 和 `outputs/diagnostics/bottleneck_audit_20260924/`。完整训练中曾发现正则日志字典在没有对比样本时缺少`contrast`键，已改为该batch的日志值按零统计，正则损失不变；重训运行完整到早停。独立GPU评估确认新checkpoint的标准化参数可以正确加载；两种旧FUSE检查点也在新增buffer后复评，结果与之前sample_v2逐样本结果一致。

seed2028配对训练与验证已完成。下一步针对强情感组做signed与L1回归控制消融，并优先检查标准化收益能否改善单模态消融。不能把当前提升称为全面性能突破，也没有评估test或用附件3/4标签。

## 2026-09-24：P3 完整转写 BERT 微调（待训练）

新增 `config/q2_fuse_bert_finetune.yaml`，以标准化 FUSE 配置为底座，使用 BERT-base-uncased 完整编码 `raw_text`。附件2训练标签更新模型；冻结 BERT embeddings 与底部10层，只训练顶部2层，BERT 学习率 `2e-5`，融合与任务头学习率 `3e-4`。批次4、梯度累积8、CUDA FP16 与梯度检查点用于控制显存；验证集只用于早停和四视图选模，专项附件3/4及主数据test均不进入训练或选择。检查点只保留顶部可训练层的半精度参数，冻结层由固定的公开预训练版本加载，控制提交附件体积。

长转写不再只用 `text_bert` 预存的50个位置。代码按固定 tokenizer 重编码完整原句并核对前缀：前49个BERT位置仍与既有对齐位置一致，SEP放在第50位；多出的词位汇成尾部向量，通过可学习残差进入样本级融合表征，因此不会移动音频、视觉的对齐索引。默认上限512，超过时采用头尾拼接；本轮已核实训练集234/3395、验证集44/728条超过50个子词，最长422，均无需截断。

训练命令：`python -m scripts.train --config config/q2_fuse_bert_finetune.yaml`。验证命令：`python -m scripts.evaluate --config config/q2_fuse_bert_finetune.yaml`。本节是待运行方案，当前没有微调后的验证指标，不应与历史FUSE结果比较或宣称性能提升。完整微调检查点会较大；赛题对通用公开预训练权重的附件豁免，不应误解为可以忽略附件总大小，提交时需依据组委会要求使用可复现加载/训练脚本或另做低秩适配压缩。


## 2026-09-25：BERT微调退化诊断与配对对照

用户提供的5090运行日志：原完整转写方案最佳epoch2，valid clean Accuracy=0.611264、Macro-F1=0.583595、MAE=0.610502、Pearson=0.664127；whole AV F1=0.605448、MAE=0.592061。训练总loss从1.5919降到0.5452，但验证未持续改善，符合过拟合/多目标优化失配的表现；whole优于clean提示融合干扰，尚不能证明单一因果。原配置同时改变BERT可训练性、长文本上下文、尾部残差、任务学习率、实际batch，无法单独归因于微调或raw_text。FUSE现有对比项是同样本词位的共享/私有比较，并非跨样本InfoNCE；不以“batch4缺少负例”解释退化。

新增`bert_input_source: text_bert`直接使用三通道原始输入；旧检查点默认raw_text，保持历史推理方式。新增`bert_warmup_epochs`先冻结BERT且关闭其dropout，训练下游网络；之后解冻顶部原定层。冻结通过no_grad实现，不改变参数保存范围。日志和检查点记录bert_phase，若最优模型来自预热期，不得将其称为微调提升。

配对配置：`config/q2_fuse_bert_aligned_frozen.yaml`全程冻结，`config/q2_fuse_bert_aligned_finetune.yaml`预热2轮后以1e-5更新顶部2层。两者实际batch32、累积1次、任务lr1e-3、相同seed/增强/归一化/sample_v2评估。各自使用独立输出目录，不覆盖此前失败实验。依次用`python -m scripts.train --config <配置路径>`训练，再用`python -m scripts.evaluate --config <同一配置路径>`验证。先比较这两份配对结果，再单独评估长文本扩展；新方案尚无完整训练指标。


## 2026-09-25：面向valid Acc/F1 0.67的损失消融（进行中）

目标仍为附件2验证集Acc或macro-F1达到0.67，同时改善MAE；以历史FUSE集成MAE 0.5729及中性置零0.5628为比较点，0.55作为进取方向，不将单项超过阈值视为全面完成。保持中性为独立类别，不使用test选型、不使用附件3/4标签。

重新读取历史FUSE集成728条valid预测：负/中/正F1为0.6833/0.4987/0.7164。199条绝对强度>1的样本，真实平均绝对强度1.7705，预测0.8606，MAE 0.9861。原始分组统计保存在`outputs/diagnostics/goal67/baseline_errors.json`；它来自历史已存预测，尚不代表本轮重训结果。当前本地两份aligned BERT对照目录无结果文件，不能声称已验证其优劣。

本轮只增加可选`regression_loss`（默认smooth_l1，候选l1）及`auxiliary_scale`（默认1，候选0.1），默认行为不变。分别与标准化FUSE同seed2026比较：`config/q2_fuse_l1.yaml`仅改变回归损失，`config/q2_fuse_lowaux.yaml`仅降低分解/重建正则总权重。两份统一patience5，上限30；历史标准化对照patience4、上限40且最佳epoch2，这一点需在比较中保留。四视图sample_v2验证和选择分数不变，不为了达到目标修改评估口径。

命令：`python -m scripts.train --config config/q2_fuse_l1.yaml`；低辅助损失同理换为`config/q2_fuse_lowaux.yaml`。独立验证：`python -m scripts.evaluate --config <同一配置>`。输出独立保存在`outputs/runs/q2_fuse_l1_s2026`及`q2_fuse_lowaux_s2026`。pytorch环境15项测试通过，包含L1数值/梯度和历史默认目标一致性；GPU对照训练正在进行，性能结论待结果补充。

首轮GPU训练已完成：L1最佳epoch6，clean Acc/F1/MAE为0.5962/0.5950/0.5786，未优于标准化对照，暂不采用。低辅助损失最佳epoch2，clean为0.6415/0.6313/0.5718，相对同seed标准化对照0.6223/0.6192/0.5739有小幅共同改善，但尚未超过历史两种子集成的分类指标。第二seed2027复核已启动，配置`config/q2_fuse_lowaux_s2027.yaml`。冻结BERT的CLS+词位均值RBF SVM/SVR探针（`scripts.probe_pooled`、`config/q2_pooled_probe.yaml`）C=1/10的Acc为0.6236/0.6305，F1为0.6087/0.5869，共用SVR MAE0.6257，未超过复杂模型；它仅是文本诊断，不作为三模态方案。无任何结果达到目标，不能标为突破。

第二seed2027完整训练最佳epoch2：clean Acc/F1/MAE=0.6442/0.6253/0.6130，回归收益不稳定。独立两seed集成复评（`config/q2_fuse_lowaux_ensemble.yaml`，报告`outputs/diagnostics/goal67/lowaux_ensemble_valid.json`）clean=0.6456/0.6295/0.5711，local=0.6497/0.6333/0.5728，whole=0.6291/0.6138/0.5856，interval=0.6058/0.5870/0.6099。没有超过历史最佳集成的clean Acc/F1，仅MAE微降0.0018；不替换默认模型、不宣称突破。所有本轮进程已结束，未运行test。下一步应针对中性边界、分类概率与强度耦合以及过早过拟合进行有对照的结构/优化实验，不能继续仅凭单seed调辅助权重宣称收益。


## 2026-09-25：小学习率与分类回归耦合复核

从`config/q2_fuse_norm_only.yaml`生成配对小学习率配置，二者均seed2026、lr=3e-4、patience7、同一sample_v2评估，只比较`regression_mode=soft`和`signed`。训练已结束，soft最优epoch2：clean Acc/F1/MAE=0.6223/0.6140/0.5943；signed最优epoch2：0.6195/0.6155/0.5874。两者未超过现存候选；没有证据支持单纯降低学习率或独立signed头能达到目标。

预先固定的等权集成对照：历史FUSE双种子+低辅助损失双种子（4模型）valid clean Acc/F1/MAE=0.6566/0.6386/0.5686；额外加入标准化FUSE双种子后（6模型）为0.6456/0.6313/0.5631；改为额外加入门控基线双种子后为0.6552/0.6372/0.5624。三者各有权衡，目前0.67未达到，且不能依赖在同一验证集上不断搜索集成组合制造虚高结果。完整checkpoint来源和SHA256分别记录在`outputs/diagnostics/goal67/mixed_ensemble_valid.json`、`mixed_norm_valid.json`、`mixed_availability_valid.json`。

新实施可选`magnitude_weight`，仅对非中性训练样本增加绝对强度Smooth-L1，默认0使历史配置不变。`config/q2_fuse_magnitude.yaml`固定为0.3并仅作单变量对照，GPU训练进行中。数值和梯度测试通过；收益需要验证集证明。

强度监督单模型训练完整结束，最佳epoch=2，clean Acc/F1/MAE=0.6181/0.6145/0.5755；相对同seed标准化对照未改善，因此不采用。为检查复杂因子分解是否稀释文本证据，新增默认关闭的`text_residual`：对编码后的有效文本位置池化，经零初始化分类残差头直接加到融合logits；文本缺失时残差严格为零。配置`config/q2_fuse_textresidual.yaml`，17项测试通过；完整训练指标待补。


## 2026-09-25：问题三预测更新与中性层级头

重新检查本地新增的BERT配对对照：`q2_fuse_bert_aligned_frozen_s2026`最佳epoch7，valid clean Acc/F1/MAE=0.6058/0.5805/0.6081；`q2_fuse_bert_aligned_finetune_s2026`最佳epoch5（joint），0.6003/0.5954/0.5829。微调相较同配置冻结组改善F1与MAE，但Acc下降，而且二者均低于历史FUSE集成，因此没有BERT突破。完整来源在各自`validation_metrics.json`与`metrics.csv`。

文本残差单模型（`config/q2_fuse_textresidual.yaml`）训练完成，最佳epoch3，valid clean Acc/F1/MAE=0.6305/0.6240/0.5936，未达到目标。保留开关供后续研究，默认关闭，不替换历史模型。

问题三原配置使用门控BiGRU两种子；在附件2完整验证集上clean Acc/F1/MAE=0.6319/0.6192/0.5804。新`config/q3_mixed.yaml`使用事先在附件2验证集选出的FUSE四模型集成，统一代理对照为0.6566/0.6386/0.5686，Acc +2.47个百分点、Macro-F1 +1.94个百分点、MAE -0.0118。已重跑附件4全部20个无标签样本，保存`outputs/predictions_q3_mixed/`；这是附件2验证集上的改进，**不是附件4准确率**。新推理将证据窗口的时间状态区分为完整转写与局部映射：875个窗口中725个完整、150个局部（07、18各75）；局部窗口只报告已对齐词位，不补造末尾词时间。时间、比例不变量核查通过，计数文件`outputs/diagnostics/goal67/q3_mixed_coverage.json`。直接以强度小于0.1改判中性的规则使验证Acc从0.6566降到0.6511，已否决。

针对中性F1瓶颈增加可选`hierarchical_head`，以“中性/非中性→非中性极性”计算归一化三分类对数概率，默认关闭。`config/q2_fuse_hierarchical.yaml`只改变分类头，GPU训练进行中；18项测试通过，包括概率和两个头的梯度。

层级分类头`config/q2_fuse_hierarchical.yaml`完整训练结束，最佳epoch4，valid clean Acc/F1/MAE=0.6236/0.6053/0.6038，未解决中性瓶颈。Q3证据独立保留窗口审计（`scripts.evaluate_q3_evidence`）显示，仅按删除分数排名第一的窗口在20条中只有11条比同模态不重叠随机窗口更充分；报告`outputs/diagnostics/goal67/q3_evidence_faithfulness.json`。新`refine_q3_evidence.py`对每条前10个删除候选补算单窗口保留分数，用两种正向证据的较小值排序；`outputs/predictions_q3_mixed/q3_selected_evidence.csv`记录两种数值及映射，20/20条均有删除与保留分数同时为正的候选，比原首位的17/20更可审查。这是同批候选上的筛选改善，不作为独立测试集上的解释泛化收益。

面向单模型目标新增`model/complementary_net.py`：文本保持CLS与有效词位BiGRU摘要，音视频分别摘要，拼接单模态与成对交互后进行轻量融合；保持缺失掩码、训练集音视频标准化与现有三分类/回归任务。`config/q2_complementary.yaml`与标准化FUSE使用相同seed、增强、sample_v2验证；19项测试通过，GPU训练中。尚无其验证性能结论。


### 最终复评与目标边界（2026-09-25）

`q2_complementary.yaml`轻量晚期融合单模型训练/独立复评完成，最佳epoch2，valid clean Acc/F1/MAE=0.6195/0.6102/0.6167；未优于FUSE，作为被否决的结构尝试保留，不用于默认提交。中性层级头、文本残差、强度辅助监督和小学习率两种模式也都未产生单模型突破。当前可复核单模型尚无Acc或macro-F1达到0.67的检查点。

四模型等权集成加入“预测中性则强度归零”后，附件2 valid clean Acc/F1/MAE=0.6566/0.6386/0.5571，详见`outputs/diagnostics/goal67/mixed_neutral_valid.json`；相比问题三旧门控双种子clean Acc/F1/MAE=0.6319/0.6192/0.5804，分别为Acc +2.47个百分点、F1 +1.94个百分点、MAE下降0.0233。该收益仅在有标签的附件2验证集上验证，不能写作附件4无标签样本准确率。试探性的验证集logit类别偏置网格虽在同集上最高见0.6635 Acc/F1 0.6457，但这是同集搜索，且仍不到0.67，未应用于模型或专项结果。

问题三最终候选配置`config/q3_mixed_neutral.yaml`，实际输出`outputs/predictions_q3_mixed_neutral/`。20条预测中4条为中性，强度确认为0；875个时间窗口中725完整、150局部映射，局部窗口均属于07/18；20条最终证据的删除与保留分数均为正，详情`q3_selected_evidence.csv`及摘要。双向筛选从同一批候选中选择，20/20是筛选条件满足数，不是独立标注的解释正确率。实际时间映射仍需人工核对WhisperX对齐误差，尤其局部覆盖样本。最终问题三若选择该方案，需使用`python -m scripts.infer --config config/q3_mixed_neutral.yaml`，再运行`python -m scripts.refine_q3_evidence --config config/q3_evidence_refine_neutral.yaml`。

目前用户总体目标未完全达到：需要进一步得到并复核一个Acc或macro-F1达到0.67、且MAE表现突出的单模型，或至少更稳健的集成候选。后续优先对中性/弱情感错误进行数据与表征诊断，并用训练集内部划分控制结构搜索；不能以不断在同一valid上试权重或阈值来制造伪增益。


## 2026-09-25：BERT互补性与Q3完整场景训练

直接用附件2 valid逐样本预测比较四模型FUSE集成与`q2_fuse_bert_aligned_finetune_s2026`：BERT微调单模型可纠正前者45条错误，却把前者原本正确的86条判错。固定25% BERT概率混合时clean Acc/F1从0.6566/0.6386降至0.6401/0.6271；MAE从0.5686到0.5607，中性置零后从0.5571到0.5538，但分类明显受损，未纳入最终候选。这个对照只在valid做诊断，不对同一valid继续调权重。

冻结BERT CLS+有效词位均值、声画有效位置均值/方差的SVC/SVR探针（PCA和标准化仅在train拟合）四组合最佳Acc=0.6387、最佳F1=0.6207，MAE>=0.6392；脚本`probe_multimodal.py`、配置`q2_multimodal_probe.yaml`、原始结果`outputs/diagnostics/goal67/multimodal_probe/report.json`。它未证明三模态表示可由简单池化更好地分离，未替换深度模型。

核查现有443条epoch记录，单模型最高clean Acc=0.6552（`soft_mask_safe_s2026`，F1=0.6240、MAE=0.5947），最高F1=0.6313（`q2_fuse_lowaux_s2026`，Acc=0.6415、MAE=0.5718）；并无已经训练却未被综合选模保存的0.67单模型轮次。Q3专用配置`config/q3_fuse_clean_train.yaml`将完整样本监督权重设为0.7、缺失概率0.3，以clean验证视图选轮次；其余沿用标准化FUSE。原默认0.35/综合四视图不变。GPU训练中，收益需以新checkpoint复评后判定。


### 完整输入训练与类别表征消融（2026-09-25）

`q3_fuse_clean_train.yaml`已训练完成：将完整监督权重从0.35调到0.7、缺失概率0.8降到0.3，并仅以clean valid选模，最佳epoch2，valid clean Acc/F1/MAE=0.6126/0.6091/0.5803。该训练方案不如既有FUSE集成，也未超过同seed标准化FUSE；不用于最终问题三预测。考虑到同时变更多个训练项，不能由此单独归因于任何一项。

为了更直接解决负/中/正融合表征的混杂，在FUSE完整与缺失两视图的池化表示上增加可选监督对比损失：同类为正例、不同类为负例，同一样本两视图必然成对；默认权重0保持历史配置不变。`config/q2_fuse_supcon.yaml`设权重0.05、温度0.1，其他取自`q2_fuse_norm_only.yaml`。20项测试通过，GPU训练中；尚无可靠性能结论。

监督对比损失单seed训练完成：`q2_fuse_supcon_s2026`最佳epoch2，clean Acc/F1/MAE=0.6154/0.6135/0.5818，未超过同seed标准化FUSE；不采用。对历史`fuse_s2026`检查点按同一完整输入口径评估：train Acc/F1/MAE=0.6878/0.6799/0.5511，valid=0.6209/0.6202/0.5750，差值约6.69/5.97个百分点及0.02396 MAE；训练中性召回0.7032、验证0.6413。原始`outputs/diagnostics/goal67/fuse_generalization.json`可复核。该结果支持泛化瓶颈存在，但不能把它全归因于某一个辅助损失。

辅助损失权重设为0时`q2_fuse_noaux_s2026`最佳epoch2，clean Acc/F1/MAE=0.6374/0.6307/0.5755，相对同seed标准化FUSE分类改善、MAE略差。第二种子`q2_fuse_noaux_s2027`已启动。固定的历史FUSE四模型+安全掩码门控双种子六模型，clean Acc/F1/MAE=0.6552/0.6356/0.5595，低于四模型+中性归零候选0.6566/0.6386/0.5571；报告`outputs/diagnostics/goal67/mixed_safemask_valid.json`，不替换默认Q3输出。

零辅助损失第二种子`q2_fuse_noaux_s2027`完整训练最佳epoch4，clean Acc/F1/MAE=0.6291/0.5966/0.5882，未复现第一种子的F1收益；不作为替换方案。历史FUSE四模型+安全掩码门控双种子六模型也没有提升clean，详见上一节。

问题二推荐配置`config/q2_best.yaml`只用于评估、鲁棒性分析和附件3推理；其四个检查点均已由各自历史配置训练。与旧`config/q2.yaml`门控双种子在相同附件2 valid `sample_v2`/seed2026上比较，四个视图的Acc、macro-F1、MAE均改善：旧clean/local/whole/interval F1=0.6192/0.6120/0.6108/0.5632、MAE=0.5804/0.5793/0.5833/0.6334；新F1=0.6386/0.6355/0.6184/0.5956、MAE=0.5571/0.5591/0.5720/0.5908。旧新完整结果分别为`outputs/diagnostics/goal67/q2_old_availability_valid.json`与`q2_best_valid.json`。后者独立重算与此前`mixed_neutral_valid.json`四视图数值一致。附件3 30条无标签样本已通过`config/q2_best.yaml`重新生成`outputs/predictions_q2_best/q2_predictions.csv`，其中11条预测中性且强度=0；不能据此计算附件3准确率。45种缺失形态与位置条件的验证扫描`q2_best_robustness.csv`已生成，完整输入首行与独立评估完全一致。当前`q2.yaml`保留历史门控训练与对照，避免将其输出误认为推荐结果；README列出新推荐命令。

## 2026-09-25：从数据、输入一致性与损失梯度重新审计

`scripts/audit_q2_data.py`只读取附件2 train/valid，并在训练集拟合标准化、64维PCA和固定参数线性分类器。样本数3395/728，负/中/正分布967/758/1670与206/184/338；分类标签与回归标签符号0条冲突。两切分没有相同视频来源。自然视觉整段零特征为训练110条、验证15条；在有内容的文本词位内，音频有效率训练/验证约99.9%/99.7%，视觉约94.5%/94.3%。这说明额外的人工缺失并非唯一缺失来源。PCA探针在验证集上：文本Acc/F1=0.643/0.596，音频=0.459/0.316，视觉=0.459/0.358，音视频=0.468/0.359，三模态直接拼接=0.640/0.594。简单池化与线性模型有明显能力限制，这些数字只用于判断融合难点，不能作为各模态的性能上界；报告为`outputs/diagnostics/goal67/q2_data_audit.json`。

附件3对齐文件实际字段为`audio/text_bert/vision`，没有`raw_text`；附件4有`raw_text`。因此使用完整`raw_text`作为唯一BERT输入的模型虽然能在附件2训练和验证，但无法对附件3执行相同数据流程。问题二后续BERT微调应以`text_bert`三通道token/attention/type为统一输入；这里的`text_bert`是BERT输入，不是768维预计算的`text`。完整转写实验仅作附件2研究对照，不能直接作为问题二专项提交模型。已有`text_bert`配对控制中，冻结BERT组clean Acc/F1/MAE=0.6058/0.5805/0.6081，顶层微调组=0.6003/0.5954/0.5829；微调改善F1与MAE，但未超过冻结特征FUSE方案。

`scripts/audit_q2_errors.py`对照现有FUSE四模型集成与`text_bert`微调模型的同一728条valid逐样本预测。后者相对前者有45条改对、86条改错；中性类召回0.538→0.609，正类召回0.722→0.592；弱非零情感准确率0.483→0.361。完整强度切片见`outputs/diagnostics/goal67/q2_error_slices.json`。因此微调的主要退化集中在正向与弱情感，不应只看总体MAE或训练损失。

`scripts/audit_q2_objective.py`在固定8条训练样本上按实际训练权重计算FUSE检查点的损失与梯度。原标准化FUSE的分类/回归/信息增益项损失约0.403/0.321/0.174；信息增益项作用于音频输入层的梯度范数0.0196，与分类项0.0195接近。低辅助FUSE中对应项为0.0156，而分类项梯度0.1504。单批梯度只能提出假设，不能证明某项必然有害。为验证该假设，`config/q2_fuse_info_light.yaml`相对标准化FUSE只把`info_weight`由0.1降到0.01，同seed、同`sample_v2`协议训练与独立评估；最佳epoch2 clean Acc/F1/MAE=0.5975/0.5957/0.5809，低于原标准化FUSE的0.6223/0.6192/0.5739。结论是单独调小信息增益项没有改善，保留为被否决实验，不替换当前`q2_best.yaml`。

## 2026-09-25：按模态分工的分类头与集成检验

相同valid顺序上，历史纯文本模型有462/728条正确，当前FUSE四模型478/728条正确；前者独对42条，后者独对58条，两者都错208条。按真实类别看，FUSE相对纯文本净增加中性正确23条，负向净减少5条、正向净减少2条。这只支持“中性受益、极性可能受干扰”的设计假设，不说明AV单模态能直接预测中性。先验固定50/50概率混合纯文本与FUSE4，clean Acc/F1/MAE=0.6511/0.6302/0.5673（中性归零），低于四模型0.6566/0.6386/0.5571，故不采用。

`model/fuse_net.py`新增可选`text_polarity_head`：层级头用融合池化表征判中性、文本池化表征判非中性极性；文本缺失时退回融合表征。默认关闭，旧检查点完全兼容。配置`config/q2_fuse_text_polarity.yaml`相对低辅助FUSE同时启用层级头和文本极性头，最佳epoch3；独立复评clean Acc/F1/MAE=0.6250/0.6087/0.5892，低于低辅助FUSE的0.6415/0.6313/0.5718。interval视图Acc=0.6168相对低辅助0.5893较高，但clean和其他视图总体退化，不能取代主模型。再做一个不调验证权重的固定等权五模型对照：FUSE4加入该新结构后clean Acc/F1/MAE=0.6552/0.6372/0.5565；MAE只改善0.0006，分类略差，故`config/q2_fuse_specialized_ensemble.yaml`仅作否定对照。验证报告见`outputs/diagnostics/goal67/text_polarity_valid.json`及`q2_specialized_ensemble_valid.json`。

## 2026-09-25：参数EMA平滑检验与目标边界

低辅助FUSE新增可选训练参数`ema_decay`，在参数更新后维护指数滑动平均，并使用EMA参数做每轮验证和保存最优检查点；默认0保持旧行为。`config/q2_fuse_lowaux_ema.yaml`与低辅助seed2026对照只改变EMA=0.98、早停耐心5→6（额外轮次用于检查平滑是否延后峰值）。最佳epoch3，独立复评clean Acc/F1/MAE=0.6497/0.6290/0.5727；原低辅助为0.6415/0.6313/0.5718。Acc提高0.82个百分点，但F1和MAE略退化，不是单模型0.66突破。固定等权FUSE4+EMA五模型clean=0.6552/0.6368/0.5561，相比现有四模型0.6566/0.6386/0.5571，MAE仅改善0.0010，分类略降；whole F1有所提高但interval下降，故不更新默认检查点。结果为`outputs/diagnostics/goal67/q2_fuse_lowaux_ema_valid.json`及`q2_ema_ensemble_valid.json`。

当前四模型在728条valid上正确478条（Acc=0.6566）；达到0.66需正确至少481条，即多3条。478/728的精确二项95%区间约[0.6208,0.6911]，且未计入多次在同一valid上选模型带来的选择偏差。因此不应针对这3条继续在valid上扫描阈值或集成权重。下一阶段需要先以附件2训练集内部按视频来源划分开发/校准集，在该内部划分上设计少量有因果假设的实验，再将既有valid作为一次外部复核；目标应同时约束clean、local、whole、interval四视图及强弱情感误差。


## 2026-09-25：文献驱动的冻结融合层起步、低学习率BERT实验

查阅Howard与Ruder的ULMFiT（ACL 2018，https://aclanthology.org/P18-1031/），其核心相关启示是分阶段、区分层的微调；同时参考DPDF-LQ（EMNLP 2025，https://aclanthology.org/2025.emnlp-main.571/）对全局与局部线索的区分。此前从头同时训练融合模型与BERT顶层的配对实验，`text_bert`版本best clean Acc/F1/MAE约0.6003/0.5954/0.5829，难以确认退化来自融合层重初始化还是BERT更新。本次只检验前一因素：从`q2_fuse_lowaux_s2026/best.pt`加载已收敛的FUSE权重，保留官方BERT底层预训练权重，仅以5e-6更新顶部2层、3e-5更新任务层，保持原缺失增强、损失与sample_v2评估。该策略是受分阶段微调启发的本题适配，并非复现文献完整模型。使用`text_bert`，附件3同样具备此字段；训练标签仅来自附件2训练集。

配置为`config/q2_fuse_bert_warmstart.yaml`，实现`--init-checkpoint`及结构键严格匹配检查，避免从不相容检查点静默加载。RTX4060 Laptop GPU完成5轮早停，最佳epoch1。独立复评`outputs/runs/q2_fuse_bert_warmstart_s2026/validation_metrics.json`：clean Acc/F1/MAE=0.6456/0.6305/0.5772；local=0.6401/0.6237/0.5785；whole=0.6250/0.6091/0.5860；interval=0.5975/0.5824/0.6293。起始低辅助单模型clean=0.6415/0.6313/0.5718；现有四模型集成clean=0.6566/0.6386/0.5571。故本次未取得全面提升，不替换默认模型。训练损失从epoch1约0.9继续下降，验证选择分数在epoch2至5均劣于epoch1，支持小数据上快速过拟合的判断，但单seed不足以证明普遍规律。

本轮独立验证正常结束并保存最佳检查点；训练结束时，新增的`init_checkpoint`路径未被`run.json`序列化导致训练命令退出码1，已修复该记录问题，历史检查点和`metrics.csv`不受影响。后续优先以训练集内部按视频分组的开发划分检验条件性音视频残差或局部证据聚合；附件2只有3395训练样本、音视觉单模态线性探针valid Acc各约0.459，现有证据不支持盲目增大融合结构。附件3缺少`raw_text`与768维`text`，因此跨附件一致部署仍需以`text_bert`接口或统一重处理视频为准。

## 2026-09-25：outputs 与 config 冗余清理

按"结论已记录在本文、不会再复跑"的原则删除明确淘汰内容；本文上文各条实验结论与数值不受影响。

删除25个被否决实验的配置：`q2_complementary`、`q2_fuse_bert_aligned_finetune`、`q2_fuse_bert_aligned_frozen`、`q2_fuse_bert_finetune`、`q2_fuse_bert_warmstart`、`q2_fuse_ema_ensemble`、`q2_fuse_hierarchical`、`q2_fuse_info_light`、`q2_fuse_l1`、`q2_fuse_lowaux_ema`、`q2_fuse_lowaux_ensemble`、`q2_fuse_magnitude`、`q2_fuse_mixed_availability_ensemble`、`q2_fuse_mixed_norm_ensemble`、`q2_fuse_mixed_safemask_ensemble`、`q2_fuse_noaux`、`q2_fuse_noaux_s2027`、`q2_fuse_signed_slow`、`q2_fuse_soft_slow`、`q2_fuse_specialized_ensemble`、`q2_fuse_supcon`、`q2_fuse_text_polarity`、`q2_fuse_textresidual`、`q3_fuse_clean_train`、`q3_mixed`。本文中指向这些文件的命令行引用保留为历史记录，文件可从 git 历史恢复；`config/` 由49个减至24个，保留 q1/q2/q2_best/cica/fuse/lowaux/mixed_ensemble/neutral/norm系/审计系/q3系全部现役配置。

删除18个被否决 run 的 `best.pt`（约197M：BERT微调四组与l1、supcon、magnitude、textresidual、hierarchical、complementary、info_light、text_polarity、lowaux_ema、noaux双种子、soft_slow、signed_slow、q3_fuse_clean），保留各自 `metrics.csv`、`run.json`、`validation_metrics.json` 作为可复核记录。删除 `outputs/experiments/` 下11个旧检查点子目录（`fuse_s2026/2027`、`fuse2_*`、`fuse_av_*`、`interval_*`、`text/audio/vision_only`，约38M），顶层 `*_test.json`、`*_validation.json`、`*_robustness.csv`、`mix_test.json`、per-sample CSV 等结果记录保留（`mix_test.json` 等引用见上文）。删除 `outputs/diagnostics/goal67/pooled_probe/` 的两个 joblib 探针权重（约141M，"仅是文本诊断"结论见上文；`report.json` 与 `valid_*.npz` 保留）。删除被否决实验的20个顶层训练/评估日志，以及全仓无任何引用的 `outputs/best_gate_architecture_v3.png`。

明确保留：`q2_best.yaml` 引用的四模型检查点与 `runs/main/q3_alignment.json`；`bottleneck_audit_20260924` 的 norm_only/norm_signed 对照检查点；`goal67` 全部 JSON/CSV 报告；`soft_mask_safe`、`exp_bert_audio_dynamics_w05`、`availability*`、`bert_audio_dynamics_ensemble` 历史检查点；全部 `predictions_*` 目录（含 `predictions_q3_mixed` 非中性对照链，其 infer 配置已删但 `q3_evidence_mixed`/`q3_evidence_refine` 保留）；`q2_fuse_missing`（与分支 `shuke/q2-fuse-missing` 关联）；`context_20260925`（当日目标函数实验，最终结论待定）；CICA、`baseline_transformer`、`runs/text_only` 对照检查点。README 中对 `q2_fuse_bert_finetune.yaml` 的引用已同步改为文字描述。

## 2026-09-25：四模型集成附件2 test补测与同协议对照

`config/q2_best.yaml` 的四模型等权集成此前只有附件2 valid记录，test缺测，且历史test对照表所用评估链路与sample_v2不是同一次运行，直接并列会混淆协议差异。本次在本机RTX4060 Laptop GPU、`pytorch`环境补跑test，并把两条对照一并用同一条评估链路重算。

先做可复核性检查：`python -m scripts.evaluate --config config/q2_best.yaml --split valid` 与已存 `outputs/diagnostics/goal67/q2_best_valid.json` 四视图16项指标逐位一致（Acc/F1/MAE/Pearson 全同），说明本机评估链路与历史记录同源，下述test数字可直接与历史test表对照。

附件2 test（n=727，sample_v2 / seed2026）：

| 配置 | 视图 | Acc | macro-F1 | MAE | Pearson |
|---|---|---|---|---|---|
| 四模型集成，中性置零（`q2_best_test.json`） | clean | 0.6685 | 0.6330 | 0.6215 | 0.6668 |
| 同上 | local | 0.6699 | 0.6349 | 0.6207 | 0.6663 |
| 同上 | whole | 0.6671 | 0.6324 | 0.6274 | 0.6575 |
| 同上 | interval | 0.6314 | 0.5863 | 0.6552 | 0.6291 |
| 四模型集成，不置零（`mixed_ensemble_test.json`） | clean | 0.6685 | 0.6330 | 0.6283 | 0.6704 |
| 同上 | interval | 0.6314 | 0.5863 | 0.6596 | 0.6334 |
| 旧门控双种子（`q2_old_availability_test.json`） | clean | 0.6699 | 0.6351 | 0.6085 | 0.6939 |
| 同上 | whole | 0.6726 | 0.6374 | 0.6103 | 0.6910 |
| 同上 | interval | 0.6410 | 0.6064 | 0.6640 | 0.6360 |

中性置零只改回归输出，不改分类，因此两版Acc/F1相同，MAE/Pearson不同。

对照的可复核性：旧门控双种子（`availability_s2026`+`availability_s2027`）按sample_v2重算，clean与whole四指标与历史 `outputs/runs/availability/test_metrics.json` 逐位一致；local不一致（历史F1 0.6464、本次0.6323），历史记录无interval视图，差异应来自当时局部扰动的实现或随机源。以下对照以clean为准，同一链路重算的local/interval随附。

**核心结论：四模型集成在valid上的优势没有迁移到test。** 与旧门控双种子相比，valid clean是明显改善（Acc 0.6566 vs 0.6319、F1 0.6386 vs 0.6192、MAE 0.5571 vs 0.5804）；但test clean为Acc 0.6685 vs 0.6699（−0.14个百分点）、F1 0.6330 vs 0.6351（−0.21个百分点）、MAE 0.6283 vs 0.6085（不置零差0.0198，置零后0.6215仍差0.0130）、Pearson 0.6704 vs 0.6939（−0.0335）。即test上旧门控双种子在分类、回归、相关三项上都不劣于四模型集成；interval视图四模型同样更低（F1 0.5863 vs 0.6064）。历史text-only单模态test clean macro-F1 0.6390仍高于上述两者。

这与本文此前"不能依赖在同一验证集上不断搜索集成组合制造虚高结果"的判断一致：四模型是在同一valid上选出来的组合，valid增益属选择内增益，test上未复现。**旧门控双种子仍是可追溯的最佳test多模态结果（clean Acc 0.6699 / F1 0.6351 / MAE 0.6085 / Pearson 0.6939）**；四模型集成的定位应是"附件2 valid选模与附件3/4推理的现役配置"，不能写成test更优。若要宣称test更优，需要在训练集内部按视频来源划分开发/校准集，在该内部划分上选定集成成员后只在test上评一次。

text-only对照的可复核性说明：`outputs/experiments/text_only_test.json`（clean F1 0.6390）的来源检查点在上一轮清理中已随 `outputs/experiments/text_only/` 删除，用现存 `outputs/runs/text_only/best.pt` 重跑（`--drop-modalities audio vision`）得 clean Acc/F1/MAE/Pearson=0.6768/0.6348/0.6316/0.6749，与历史值差2条样本，故历史数字继续沿用但标注为不可逐位复现；本次重跑结果另存于 `outputs/diagnostics/goal67/q2_text_only_test.json`。

本轮产物：`outputs/diagnostics/goal67/q2_best_test.json`、`q2_best_test.csv`（727行逐样本）、`mixed_ensemble_test.json`、`mixed_ensemble_test.csv`、`q2_old_availability_test.json`、`q2_text_only_test.json`。本轮未训练任何新模型，未改动任何检查点。

## 2026-09-26：问题三条件证据与解释卡复核

保持 `config/q3.yaml` 的四检查点预测器不变，重跑附件4共20条无标签推理；解释器只使用统一预测接口，不读 FUSE 内部层。`scripts.infer` 输出三模态 signed Shapley 与绝对影响份额；`scripts.refine_q3_evidence` 用删除必要性、语义上下文条件增益、文本 Early/Late 双遮蔽、同时间事件三模态删除影响及逐检查点稳定性复核窗口。文本 Early 遮蔽保留 BERT 特殊符号。07、18号虽然整段长转写仅部分进入50位网格，入选事件本身可映射的范围仍按实际匹配词位输出；未覆盖尾段不补造时间。

当前经验随机窗口0.05门槛下，20条中只有07、18号的候选通过全部筛查；其余仅作为可展开的待核候选，不默认高亮。04号的核心窗口仍落在 `Absolutely not.`，但现会连同前面的完整问句显示，并提示当前负向预测可能误解问答否定。这是**发现模型错误的解释诊断**，没有自动改写预测类别。随机窗口经验排名受候选选择和窗口重叠影响，输出头随机化仅一次；它们属于 sanity check，不是人工标注的解释准确率或严格显著性证明。结果见 `outputs/predictions_q3_mixed_neutral/q3_selected_evidence.csv`、`q3_selected_evidence_summary.json` 和 `explanation_cards.html`。运行 `python -m scripts.serve_cards` 可直接打开本地页面。

## 2026-09-26：问题二主线定为 lowaux FUSE，补齐缺失规律与消融全量实验

**主线。** 论文问题二改以 `q2_fuse_lowaux_s2026`（FUSE 因子化结构、gate 融合、辅助正则 0.1、混合缺失增强、sample_v2 选型、best epoch 2）为推荐模型；`q2_fuse_lowaux_s2027` 为种子复核。对应论文结构 4.5 的四个小节，本轮补齐题面"四、结果与提交说明 3.问题2相关内容"中缺失的 (2) 缺失模态类型/缺失率规律分析与消融实验、(3) 附件3全量预测、(4) 验证集基础性能与错误归因，全部结果整理在 `outputs/diagnostics/q2_paper/`。

**实验矩阵。** `scripts.robustness` 按题面"缺失模态类型、缺失位置、缺失时长"三因素扩展后跑全量：1 个完整输入基线 + 4 个整段模态缺失（audio/vision/audio+vision/text）+ local 短游程（音视频，6 档缺失率 × 4 位置）+ interval 连续区间（音视频 6 档 × 4 位置；三模态、text、audio、vision 各自单独缺失在 random 位置 6 档），共 77 情形/检查点，逐情形 Acc/Macro-F1/MAE/Pearson。缺失率取 0.1–0.6（步长0.1），位置 start/middle/end/random，多模态区间含 0.5 概率部分重叠。遮蔽随机源固定 seed=0，各臂逐情形同掩码、配对可比。消融为 6 个单因素臂（`config/q2_fuse_lowaux_ab_*.yaml`，各自只改一个键、同协议同种子 2026 重训并重扫）：`ab_noaug`（增强概率0）、`ab_nofactor`（退回门控基线结构）、`ab_noaux`（辅助正则0）、`ab_aux1`（辅助正则1.0剂量对照）、`ab_nodynamics`（去语音差分）、`ab_nonorm`（去输入标准化）。

**基础性能（验证集四视图，sample_v2）。** clean 0.6415/0.6313/0.5718/0.6723；local 0.6401/0.6255/0.5734/0.6697；whole 0.6277/0.6109/0.5812/0.6609；interval 0.5893/0.5777/0.6172/0.6135（Acc/F1/MAE/Pearson）。错误切片（clean）：负/中/正召回 0.704/0.571/0.642，中性强度MAE 0.302；负向收缩仍在（真值均值 −1.07、预测 −0.46），正向亦偏弱（1.00→0.49）。切片 JSON 见 `q2_paper/q2_error_slices.json`。

**缺失模态类型规律（验证集，seed=0 同掩码）。** 整段缺失：text 全失 F1 塌到 0.3132（−31.8pt、MAE +0.39），audio 全失仅 −1.4pt、vision 全失 −1.0pt、音视频同失 −2.0pt。区间缺失@0.4：text 单独缺失 F1 0.5190，三模态同缺 0.5440，audio 单缺 0.6196、vision 单缺 0.6310、音视频同缺 0.6228（基线 0.6313）。**文本是唯一强依赖模态；音视频缺失影响轻微且大体可被模型补偿。**

**缺失率规律。** 每 +10% 缺失率的 Macro-F1 斜率：interval text **−3.69pt**（MAE +4.2pt/10%）、interval 三模态 −2.46pt、interval 音视频 −0.23pt、local 音视频 −0.16pt、interval audio −0.17pt、interval vision +0.05pt。即**音视频缺失在 0.1–0.6 范围内几乎无退化，文本缺失近似线性陡降**；s2027 复核同型（斜率表 `q2_paper/q2_trend_slopes.csv`）。

**缺失位置规律。** interval 音视频下 start/middle/end/random 在各档缺失率的 F1 全距 ≤0.6pt（rate0.4：0.6265/0.6231/0.6265/0.6228），local 同样平坦——**位置不敏感，缺失长度与模态类型才是主因**。

**消融（各臂 best epoch 均为 2，同协议）。** clean F1 贡献排序：语音差分 −4.2pt（0.5889）、输入标准化 −3.0pt（0.6014）、缺失增强 −2.0pt（0.6112）、因子化结构 −1.1pt（0.6199）、辅助正则剂量（1.0 掉 1.2pt，0 掉 0.1pt，0.1 为甜点）。增强的专门作用看缺失情形：`ab_noaug` 在 interval@0.4 掉 2.3pt（0.6228→0.5999）而在 whole-av 反升（0.6109→0.6184）——**混合缺失增强是区间/游程缺失鲁棒性的主要来源，代价是整段缺失拟合略松**；辅助正则 0.1 相对 0 在 clean 有 +0.06pt 的微弱收益、对缺失情形无损失，剂量 1.0 明显过强。全表 `q2_paper/q2_table_ablation.csv`。

**口径说明。** 本轮训练期间并行工作将选型分数改为 `utils/selection.py` 的三视图口径（clean/local/interval，含 `*_delta_*` 退化列）。已核验主模型在新旧两种口径下最优轮次均为 epoch 2，消融链内部共用新口径自选轮次；论文表格全部使用视图原始指标（Acc/F1/MAE/Pearson），与口径无关。

**产物。** 表：`q2_paper/` 下 `q2_basic_performance.csv`、`q2_table_missing_type.csv`、`q2_table_missing_rate.csv`、`q2_table_missing_location.csv`、`q2_table_ablation.csv`、`q2_trend_slopes.csv`、`q2_robustness_full.csv`（77情形全量）、`q2_error_slices.json`。图：`q2_fig_rate_curve.png`（缺失率曲线）、`q2_fig_missing_type.png`（模态类型）、`q2_fig_location.png`（位置）、`q2_fig_ablation.png`（消融）、`q2_fig_heatmap.png`（类型×缺失率热力图），由 `python -m scripts.figures_q2` 一键再生。附件3：`outputs/predictions_fuse_lowaux/q2_predictions.csv`（30条，主模型直接推理，含缺失模态与融合权重列）。种子复核扫描在 `outputs/runs/q2_fuse_lowaux_s2027/robustness.csv`。
