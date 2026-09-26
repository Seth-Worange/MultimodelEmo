# E题代码

包含问题1词级特征提取、问题2鲁棒模型训练与缺失实验、问题2/3专项推理和问题3局部证据解释。代码默认使用题目提供的 `aligned_50.pkl`，不读取附件2测试标签进行选型。

以下命令均在本目录执行，入口统一为 `python -m scripts.<模块名>`。

## 环境

本机环境为 `C:\Anaconda3\envs\pytorch`、PyTorch 2.8.0+cu126、RTX 4060 Laptop GPU：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

依赖清单分为两份：`requirements.txt` 是训练、评估和表格处理的基础依赖；`requirements-q1.txt` 只列第一问音视频特征提取额外需要的 Transformers、WhisperX、MediaPipe 和 OpenCV。这样只跑问题2/3时不必安装体积较大的音视频工具。请在当前 PyTorch 环境中安装，尤其不要让 pip 替换已配置的 CUDA 版 PyTorch：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m pip install -r requirements.txt
```

问题1和附件4时间戳定位需要 WhisperX、Transformers、MediaPipe、OpenCV 以及 FFmpeg。安装可选依赖时要保留当前 CUDA 版 PyTorch；首次运行会下载英文BERT和WhisperX对齐模型，或从本地缓存加载。问题1需准备 MediaPipe [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) 和 [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) 模型。姿态模型可用官方[下载地址](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)，保存为 `task\pose_landmarker_full.task`；提取结果会记录权重 SHA-256，便于复现。

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m pip install -r requirements-q1.txt
```

默认数据路径为 `data`，人脸和姿态模型分别位于 `task\face_landmarker.task`、`task\pose_landmarker_full.task`。参数集中放在 `config\q1.yaml`、`config\q2.yaml`、`config\q2_fuse.yaml`、`config\q2_cica.yaml`、`config\q2_cica_transformer.yaml`、`config\q3.yaml`；命令行参数可覆盖配置值。BERT、语音对齐权重和 NLTK 数据默认缓存在 `cache`；只对题目提供的可信 pickle 文件使用 `pickle` 加载器。

## 问题1：处理附件1的全部100条视频

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.features_q1 --config config\q1.yaml
```

每条视频生成一个压缩 `.npz`，另有 `manifest.csv` 和 `environment.json`。词时间由 WhisperX 对给定转写做强制对齐；音频为40维 log-Mel 加能量、过零率和基频，共43维帧特征。视觉同时提取面部表情与上半身姿态：16个人脸点相对鼻尖归一化（鼻尖只作原点，不重复输出恒零坐标）；另取双肩、双肘、双腕和双髋共8个姿态点，记录肩宽归一化的三维坐标与可见度。每帧共66维，另有面部/姿态检测位；词区间池化均值与标准差后为132维。最近有效人脸或姿态帧距词边界不超过0.1秒时才补齐，并在 `vision_nearest_mask` 中标记。`face_mask`、`pose_mask` 分别记录词位覆盖，`vision_mask` 记录任一视觉特征有效。另输出12维 `prosody`：词时长、停顿、局部语速、相对音高均值/范围/斜率、浊音比例、能量均值/范围/斜率和过零率。相对音高以该段语音的浊音基频中位数为参照。文本为768维BERT词向量。

`manifest.csv` 记录面部、姿态和总体视觉覆盖率、平均对齐分数及失败信息。`review_candidates.csv` 用平均对齐分数低于0.3作人工复核筛查；该阈值是启发式，不能替代听看视频。`--max-samples 1` 可先跑通单条样本。

## 问题2：训练和鲁棒性评估

当前推荐推理与验证配置是 `config\q2_best.yaml`，使用四个已训练FUSE检查点；该配置只负责评估、鲁棒性分析和专项推理，单模型训练配置分别保留在实验记录中。附件2 valid 四视图均优于旧门控双模型。

附件3只提供 `text_bert`，没有 `raw_text`；需要对附件3推理的BERT微调方案应在训练、验证和专项测试统一使用 `bert_input_source: text_bert`。这里的 `text_bert` 是token id、attention mask、token type三通道，经可训练BERT得到768维表征。以完整转写作输入的BERT微调实验（已否决，记录见 [experiment.md](experiment.md)）只能作为附件2研究对照，不可直接用其检查点处理附件3。数据、损失与错误切片审计见 [experiment.md](experiment.md)。

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2_best.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.robustness --config config\q2_best.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.infer --config config\q2_best.yaml
```

以下为历史架构的训练及对照命令。`q2.yaml` 的训练默认写入 `availability_retrain_s2026`，避免覆盖已选检查点；同一文件的评估节仍读取原历史检查点。若评估新训练权重，需用 `--checkpoint` 指定新路径：

```powershell
# 门控基线架构
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2.yaml --seed 2027 --output-dir outputs\runs\interval_s2027
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.robustness --config config\q2.yaml
# FUSE-Net 风格三因子分解架构
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2_fuse.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2_fuse.yaml --seed 2027 --output-dir outputs\runs\fuse_s2027
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2_fuse.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.robustness --config config\q2_fuse.yaml
# CICA 启发的两阶段方案：先单模态 CAP，再冻结编码器训练置信度融合
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train_cica --config config\q2_cica.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train_cica --config config\q2_cica.yaml --seed 2027 --output-dir outputs\runs\cica_s2027
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2_cica.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.robustness --config config\q2_cica.yaml
# CICA Transformer 对照：仅替换三种模态的 BiGRU 编码器
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train_cica --config config\q2_cica_transformer.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2_cica_transformer.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.robustness --config config\q2_cica_transformer.yaml
```

`encoder_type` 在门控基线、FUSE 和 CICA 三种模型配置中都可选 `bigru` 或 `transformer`。Transformer 使用一层、4头和可学习时间位置编码作为小样本起点；融合层 BiGRU 保持不变。也可只通过命令行覆盖，例如 `--encoder-type transformer`。基线与 FUSE 对照命令如下，验证时直接传入对应检查点：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2.yaml --encoder-type transformer --output-dir outputs\runs\baseline_transformer_s2026
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --checkpoint outputs\runs\baseline_transformer_s2026\best.pt --split valid --data-root data --device cuda --output outputs\runs\baseline_transformer_s2026\validation_metrics.json
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config\q2_fuse.yaml --encoder-type transformer --output-dir outputs\runs\fuse_transformer_s2026
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --checkpoint outputs\runs\fuse_transformer_s2026\best.pt --split valid --data-root data --device cuda --output outputs\runs\fuse_transformer_s2026\validation_metrics.json
```

`train_cica.py` 是独立实验入口，不改写门控基线与 FUSE 配置。阶段一将每个模态的分类、连续强度、置信度和不确定性头与对应序列编码器一起训练，阶段二加载 `cap_best.pt` 并冻结这些单模态分支，仅训练可靠性调制门控和融合预测层。训练会额外保存 `cap_metrics.csv`；最终 `best.pt` 可交给现有 `evaluate.py`、`robustness.py` 和 `infer.py` 使用。此方案借鉴 [CICA（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence-Aware_Pretraining_with_Confidence-Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html)，是针对本题三分类与连续情感分数任务的适配实现，不是对论文网络的逐层复现；附件3/4仍只用于推理，不参加训练、伪标签或选型。

模型用768维上下文BERT词特征、音频和视觉序列作为输入。附件2已有的 `text` 与本地 `google-bert/bert-base-uncased` 从 `text_bert` 重新编码的结果一致（已逐元素核对，cosine=1.0、MAE=0.0）；附件3没有 `text` 时，推理代码直接用 `text_bert` 的 token id 通过该冻结BERT生成特征，不读取标签，也不要求 `text` 字段。实际检查的附件3中30条文本都可用，29条的音频与视觉逐词掩码完全相同。BERT权重需在 `cache\huggingface` 可用。

### 缺失模拟：整段 / 零散短游程 / 多尺度连续区间

`utils/augmentation.py` 提供三种形态，`corruption_mode: auto` 时按 `whole_probability` 抽整段、其余概率在另两者间均分：

| 形态 | 做法 | 用途 |
|---|---|---|
| `whole` | 整段同步丢弃语音与视觉 | 模态完全不可用的极端压力测试 |
| `local` | 语音与视觉在**相同词位**上出现长度1–4位的**多段零散短游程** | 附件3 的实测形态（文本从不缺失、额外缺失率0%–46%、合计约22%） |
| `interval` | 有效序列上生成连续缺失区间 `[s, s+ℓ)`，`ℓ/L` 抽取自 `interval_ratios`（默认 **0.1/0.2/0.4/0.6**）；模态组合覆盖**单模态、双模态、三模态**（含文本），并以 `overlap_probability` 让多模态区间**部分重叠** | 让模型同时见到轻度与严重缺失，而不是只学一种固定模式 |

`interval_ratios` 是**建议的实验设置**，不是附件3 的已知缺失比例。所有模拟只改动训练输入副本：`demo()` 里断言原始 `tokens/audio/vision` 与真实标签在调用后逐元素不变。

### 验证视图与选型分数

`scripts/train.py` 每轮在四个视图上评估——`clean`（完整）、`local`（零散短游程）、`whole`（整段丢弃音视频）、`interval`（连续区间）。选型分数越低越好：

```text
0.5 × mean(四个视图的 MAE) / 3 + 0.5 × (1 − mean(四个视图的 macro-F1))
```

`metrics.csv` 因此含 `clean_* / local_* / whole_* / interval_*` 四组指标；`fuse` 架构另含 `reg_*` 六项正则分量。

### 两种融合架构

- `baseline`（`model/network.py`）：逐位置门控融合 + 可用性嵌入。
- `fuse`（`model/fuse_net.py`）：借鉴 Yang & Li, *Factorize, Reconstruct, Enhance* (CVPR 2026) 的 FUSE-Net——
  **HMF** 把每个模态分解为共享/私有/噪声三个子空间（对比分离 + 信息增益 + 对偶一致性）；
  **MRC** 对三支拼接做变分信息瓶颈重建（重建 + KL）；
  **MDF** 用样本调制 α × 因子类型系数 β × 分支注意力 γ 三尺度相乘，逐模态 softmax 后聚合，噪声分支加门控。
  融合表示同时送入分类头与回归头。
  **本题补充的缺失感知**：所有模态级运算受可用性掩码约束，对比分离只在两模态都有效的词位上计算，MRC 只重建有效词位，另加"用可用模态重建缺失模态"的**交叉重建项**（`--cross-weight`），使共享子空间必须真正跨模态。原论文未针对连续区间局部缺失做压力测试，这一项正是为该场景补的。

`reg_info` 采用有界形式：共享/私有分支取交叉熵（越小越好），噪声分支取 `relu(log3 − CE)`，即"一旦不比随机猜好就不再产生梯度"，避免噪声头发散。

附件2实际 `classification_labels` 编码为0负向、1中性、2正向；`regression_labels` 的数值0表示中性强度。训练只使用附件2的训练/验证标签。附件3、附件4推理仅生成预测，不读取或生成伪标签；代码不引入外部情感数据集。旧检查点仍可加载（`load_model` 按 `model_config.architecture` 分发，缺省为 `baseline`）。

第二个种子由命令行覆盖 `seed` 和 `output-dir`。训练会保存 `best.pt`、逐轮 `metrics.csv` 和配置/环境记录 `run.json`。命令行参数优先于 YAML，可用 `--epochs`、`--batch-size`、`--lr`、`--architecture`、`--corruption-mode` 等直接改参。

### 鲁棒性扫描与错误归因

`scripts/robustness.py` 在验证集上输出约 45 组对照，带 `pattern` 列：

- `whole`：完整输入基线 + 4 组整段模态缺失（音视频成对／音频／视觉／文本）。
- `local`：缺失率 × 位置（`--rates` 默认 0.1/0.2/0.4/0.6，`--locations` 默认 start/middle/end/random）。
- `interval`：同样的缺失率 × 位置扫描，另加三模态与单模态文本组合。

输出 Accuracy、macro-F1、MAE 与 Pearson，直接对应题面「缺失模态类型、缺失率、缺失位置对预测性能影响的规律分析」。

`scripts/evaluate.py` 的 `--per-sample <csv>` 落盘每条样本的真实类别/强度与四个视图的预测类别、命中与否、预测强度和三分类概率，并打印各视图混淆矩阵，用于题面要求的错误归因。**逐样本 CSV 必须按划分显式指定**，否则 `--split test` 会覆盖验证集的结果。

历史连续区间缺失设置下，旧版双模型验证集 clean / masked macro-F1 为0.6232 / 0.5909，MAE为0.5712 / 0.6098；新版 soft 双模型对应为0.6163 / 0.5946、0.5835 / 0.6127。它们只作旧设置记录，不能与新的整模态缺失验证直接比较。训练集为3,395条，验证/测试各约728条；Self-MM/MIRD论文报告的Acc-2约85–86%是二分类指标，不能和此处三分类Accuracy直接比较。

**协议已变更，`availability_s2026/s2027` 需重训**：这两个检查点是用"仅整段丢弃音视频 + dropout=0"的旧协议训练的，其 `metrics.csv` 只有 `clean_* / masked_*` 两视图。当前协议改为混合增强（含局部短游程）、`dropout=0.2`、三视图选型分数，因此旧指标既不能代表新设置，也不能与新 `metrics.csv` 逐列比较。详见 [experiment.md](experiment.md)。

模型选定后，对独立测试集只评估一次：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config\q2.yaml --split test --output outputs\runs\availability\test_metrics.json
```

旧版测试集 n=727：三分类 clean accuracy 0.6713、macro-F1 0.6277、MAE 0.6251；旧缺失模拟的 masked 结果不用于修正后的方法对比。新模型仅在完成验证集选型后评估测试集。

验证集模态消融（clean）macro-F1／MAE：全模态0.6142／0.5756，去文本0.3929／0.9289，去音频0.6136／0.5745，去视觉0.6035／0.5819，说明现有数据中文本贡献最大、音频增益很小。相同种子与训练设置下，BERT拼接融合最佳验证选择分数0.3022，门控融合0.2981（越低越好），暂不支持单靠更换融合层突破性能。

## 问题2/3：专项预测

附件3预测：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.infer --config config\q2_best.yaml
# CICA 方案训练完成后，可通过对应配置预测并输出置信度/不确定性/融合权重
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.infer --config config\q2_cica.yaml
```

附件4预测及解释（先生成词到时间映射）：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.align_q3 --config config\q3.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.infer --config config\q3.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.refine_q3_evidence --config config\q3_evidence_refine_neutral.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.build_q3_cards
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.serve_cards
```

可用 `--max-samples 1` 先验证接口。

问题2当前推荐结果位于 `outputs\predictions_q2_best`（旧门控结果仍在 `outputs\predictions_availability`），问题3默认结果位于 `outputs\predictions_q3_mixed_neutral`。问题3将已选预测器当作统一输入输出接口，按三模态8子集计算 signed Shapley，并另报绝对影响份额。证据先按删除必要性筛选，再以语义上下文条件增益复核；文本还比较重编码遮蔽和固定上下文特征遮蔽。报告同时间事件的三模态删除影响、多检查点稳定性、同模态随机窗口经验排名与输出头随机化对照。随机窗口少于19个时无法达到本流程0.05经验门槛，窗口仅保留为待核候选，网页默认不高亮。经验排名受候选筛选与窗口相关性影响，不应解释为独立标注下的解释正确率或严格显著性检验。`serve_cards` 会自动打开浏览器；已有结果更新后可加 `--rebuild` 重建页面。

附件4特征文件没有时间戳，词位时间由 `align_q3.py` 从视频强制对齐并核对。07、18号等长转写的未覆盖词位仍标记为部分映射，不补造时间。若 `alignment_file` 缺失，推理与精炼会报错。

新增脚本 `scripts\figures_q1.py` 生成问题1的典型样本对齐图与覆盖率汇总表，见下节。

## 目录与数据边界

### 2026-09-24 瓶颈诊断与固定验证

当前候选的可追溯验证比较和下一轮实验计划见 [bottleneck_analysis.md](bottleneck_analysis.md)；历史修订见 [experiment.md](experiment.md)。注意旧 `availability/validation_metrics.json` 对应Transformer单模型，不能代表当前BiGRU集成。

新增 `sample_v2` 按样本ID固定缺失位置；训练入口 `scripts.train`、`scripts.train_cica` 可加 `--evaluation-protocol sample_v2 --evaluation-seed 2026`，并使用新的输出目录。旧默认 `legacy_batch` 保留用于复现，新旧缺失视图的数值不可混用；完整输入指标可直接对照。

已有FUSE权重的可选中性强度置零评估：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config/q2_fuse_neutral.yaml
```

验证集clean MAE由0.5729降至0.5628，分类指标不变，Pearson由0.6737略降至0.6704。该配置只增加显式输出规则，不重训。结果与逐样本预测写入 `outputs/diagnostics/bottleneck_audit_20260924/`；不覆盖历史评估。

新Q3输出的 `evidence_scope` 和 `aligned_word_coverage` 说明模型可解释的词覆盖范围。`token_match_fraction=1` 不能代表完整视频覆盖或声学对齐准确，已有映射在下一次推理时也会生成覆盖字段。当前默认方案在附件2验证集的clean Acc/F1/MAE为0.6566/0.6386/0.5571；附件4无标签，不能据此写出附件4的准确率。

### P1 音视频标准化与GRU打包对照

逐特征标准化使用训练集有效位置拟合，统计量写入checkpoint；打包使用原始文本attention长度。仅标准化在seed2026、2027、2028三组配对训练中均优于sample_v2控制；GRU打包单项退化。标准化仍未超过历史FUSE分类最优，因此保留为候选配置，不替换原默认值。训练配置、checkpoint和逐视图指标位于 `outputs/diagnostics/bottleneck_audit_20260924/`。

```powershell
# 继续做配对复核时，两组使用相同的新seed和独立输出目录。
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config/q2_fuse_samplev2_control.yaml --seed 2029 --output-dir outputs/runs/fuse_control_s2029
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config/q2_fuse_norm_only.yaml --seed 2029 --output-dir outputs/runs/fuse_norm_only_s2029
```

- `config/`：问题1、2、3的 YAML 配置。
- `model/`：情感模型网络。
- `utils/`：数据读取、验证与整模态缺失增强。
- `scripts/`：特征提取、训练、评估、预测和证据定位入口。
- `experiment.md`：实验设置、指标、历史结果与清理记录。

根据赛题答疑，公开通用特征提取权重可用固定版本、下载链接或复现脚本说明；人脸和姿态权重只用于冻结特征提取，并记录权重哈希，不参与情感模型训练。自行使用外部数据预训练的特征提取器须说明训练数据、冻结节点和与后续情感模型训练的隔离。融合模块及预测头均随机初始化。附件1中短于题面描述范围的视频正常处理。

允许用附件2训练集的成对对齐/非对齐特征学习对齐器；对齐器拟合完成后，须对非对齐训练集、验证集和专项测试集应用完全相同的流程，不用验证或测试数据反向调整对齐规则。当前代码仍使用题目提供的对齐版，尚未训练额外对齐器。

附件4第13条的对齐版视觉特征为全零，推理时会通过掩码将视觉标为缺失，只用文本和音频，输出的 `missing_modalities` 会注明 `vision`；视觉 Shapley 贡献为0。未对齐版虽有17个非零视觉位置，但它与50词位对齐特征并非同一表示，不能只替换第13条。若切换未对齐数据，须按答疑要求对训练、验证和专项测试统一应用同一套对齐流程并重新训练。

提交时排除 `cache` 中的模型下载文件；运行 `Get-ChildItem outputs\features_q1_face_pose,outputs\runs\fuse_s2026,outputs\runs\fuse_s2027,outputs\runs\q2_fuse_lowaux_s2026,outputs\runs\q2_fuse_lowaux_s2027,outputs\predictions_q2_best,outputs\predictions_q3_mixed_neutral -Recurse | Measure-Object -Property Length -Sum` 并确认附件总大小满足题目50 MB上限。
