# E题代码

包含问题1词级特征提取、问题2鲁棒模型训练与缺失实验、问题2/3专项推理和问题3局部证据解释。代码默认使用题目提供的 `aligned_50.pkl`，不读取附件2测试标签进行选型。

## 环境

本机环境为 `C:\Anaconda3\envs\pytorch`、PyTorch 2.8.0+cu126、RTX 4060 Laptop GPU：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

核心训练仅需 NumPy 与 PyTorch：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m pip install -r code\requirements.txt
```

问题1和附件4时间戳定位需要 WhisperX、Transformers、MediaPipe、OpenCV 以及 FFmpeg。安装可选依赖时要保留当前 CUDA 版 PyTorch；首次运行会下载英文BERT和WhisperX对齐模型，或从本地缓存加载。问题1还需从[官方模型页](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)下载 Face Landmarker `.task` 模型。

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m pip install -r code\requirements-q1.txt
```

默认数据路径为 `code\data`，人脸模型默认为 `code\task\face_landmarker.task`。也可向脚本传 `--data-root` 或 `--face-model`。BERT、语音对齐权重和 NLTK 数据默认缓存在 `code\cache`；只对题目提供的可信 pickle 文件使用 `pickle` 加载器。

## 问题1：处理附件1的全部100条视频

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\features_q1.py --device cuda --output-dir code\outputs\features_q1_prosody
```

每条视频生成一个压缩 `.npz`，另有 `manifest.csv` 和 `environment.json`。词时间由 WhisperX 对给定转写做强制对齐；音频基础特征是40维log-Mel加能量、过零率和基频，共43维帧特征；视觉是17个MediaPipe人脸锚点的归一化坐标加检测位，共35维帧特征。按词区间池化声画均值与标准差。另输出12维 `prosody`：词时长、停顿、局部语速、相对音高均值/范围/斜率、浊音比例、能量均值/范围/斜率和过零率。相对音高以该段语音的浊音基频中位数为参照，降低说话人音高差异的影响。无有效帧的位置由掩码表示。文本为768维BERT词向量。

每行 `manifest.csv` 记录视频时长、词数、维度、平均对齐分数、转写文本保留率和失败信息。`review_candidates.csv` 用平均对齐分数低于0.3作人工复核筛查；该阈值是启发式，不能替代听看视频。转写文本保留率不验证时间边界准确性。`--max-samples 1` 可先跑通单条样本。

## 问题2：训练和鲁棒性评估

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\train.py --device cuda --epochs 40 --patience 8 --seed 2026 --output-dir code\outputs\runs\exp_bert_audio_dynamics_w05 --fusion gate --text-mode bert --class-weight-power 0.5 --audio-dynamics
& 'C:\Anaconda3\envs\pytorch\python.exe' code\train.py --device cuda --epochs 40 --patience 8 --seed 2027 --output-dir code\outputs\runs\exp_bert_audio_dynamics_w05_s2027 --fusion gate --text-mode bert --class-weight-power 0.5 --audio-dynamics
& 'C:\Anaconda3\envs\pytorch\python.exe' code\robustness.py --device cuda --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05\best.pt --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05_s2027\best.pt --output code\outputs\runs\bert_audio_dynamics_ensemble\robustness.csv
```

模型用768维上下文BERT词特征、音频和视觉序列作为输入。附件2已有的 `text` 与本地 `google-bert/bert-base-uncased` 从 `text_bert` 重新编码的结果一致；附件3没有 `text` 时，推理代码用该本地BERT模型生成特征。BERT权重需在 `code\cache\huggingface` 可用。另保留 `--text-mode tokens` 作为随机词嵌入蒸馏消融。训练时按模态生成连续缺失窗口；损失包括三分类、强度Huber和缺失前后一致性。模型选择综合验证集完整输入和固定随机缺失输入上的macro-F1与MAE。标签编码为0负向、1中性、2正向；强度由分类确定正负方向，回归头预测非负幅度，确保分类与强度符号一致。

训练输出包括 `best.pt`、逐轮 `metrics.csv` 和记录随机种子、软硬件与超参数的 `run.json`。鲁棒性脚本在验证集上分别改变缺失模态、缺失率和区间位置，输出Accuracy、macro-F1、MAE与Pearson到CSV。`--audio-dynamics` 将相邻有效词位的声学特征差分拼入音频分支；单模型结果随随机种子波动，但两模型集成在验证和测试上均优于静态音频基线。

当前候选为两个不同随机种子的BERT门控模型集成，并使用音频相邻差分。模型融合 logits 与非负强度幅度后再确定类别和强度。验证集 clean / masked macro-F1 为0.6232 / 0.6202，MAE为0.5712 / 0.5847。训练集为3,395条，验证/测试各约728条，是标准MOSEI训练测试规模的子集；Self-MM/MIRD论文报告的Acc-2约85–86%是二分类指标，MIRD还使用更多标注与无标签数据，不能和此处三分类Accuracy直接比较。

模型选定后，对独立测试集只评估一次：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\evaluate.py --device cuda --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05\best.pt --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05_s2027\best.pt --output code\outputs\runs\bert_audio_dynamics_ensemble\test_metrics.json
```

测试集 n=727：三分类 clean accuracy 0.6713、macro-F1 0.6277、MAE 0.6251；固定连续缺失评估 accuracy 0.6685、macro-F1 0.6256、MAE 0.6293。相对静态音频两模型基线，clean / masked macro-F1 提升0.0118 / 0.0226，MAE下降0.0166 / 0.0196。模型选择只使用训练集和验证集。

验证集模态消融（clean）macro-F1／MAE：全模态0.6142／0.5756，去文本0.3929／0.9289，去音频0.6136／0.5745，去视觉0.6035／0.5819，说明现有数据中文本贡献最大、音频增益很小。相同种子与训练设置下，BERT拼接融合最佳验证选择分数0.3022，门控融合0.2981（越低越好），暂不支持单靠更换融合层突破性能。

## 问题2/3：专项预测

附件3预测：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\infer.py --part q2 --device cuda --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05\best.pt --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05_s2027\best.pt --output-dir code\outputs\predictions_audio_dynamics_ensemble
```

附件4预测及解释：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\infer.py --part q3 --device cuda --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05\best.pt --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05_s2027\best.pt --output-dir code\outputs\predictions_audio_dynamics_ensemble
```

可用 `--max-samples 1` 先验证接口。

输出位于 `code\outputs\predictions_audio_dynamics_ensemble`。问题3使用三模态8个子集（含空模态基线）做精确Shapley贡献，并逐模态遮蔽连续5位置窗口，输出预测分数变化。附件4特征文件本身不含词时间戳。对原视频转写强制对齐后，可导出可回看的秒数：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' code\align_q3.py --device cuda
& 'C:\Anaconda3\envs\pytorch\python.exe' code\infer.py --part q3 --device cuda --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05\best.pt --checkpoint code\outputs\runs\exp_bert_audio_dynamics_w05_s2027\best.pt --output-dir code\outputs\predictions_audio_dynamics_ensemble --alignment-file code\outputs\runs\main\q3_alignment.json
```

`align_q3.py` 会逐位置核对重建的BERT词元与附件4中的词元；不匹配率不足时标记 `review_required`，不会输出貌似精确的时间。人工核查后再将映射用于答卷。

## 文件说明

- `data.py`：附件路径解析、NumPy兼容读取、形状／词元／标签校验。
- `augmentation.py`：连续缺失窗口生成。
- `model.py`：上下文BERT/词元文本分支、多模态时序融合和分类回归双头。
- `train.py`：CUDA训练、验证、早停和可复现记录。
- `evaluate.py`：选定模型在独立测试集上的一次性评估。
- `robustness.py`：验证集缺失类型／比例／位置实验。
- `infer.py`：单模型/多模型集成预测、Shapley模态贡献、片段遮蔽解释。
- `features_q1.py`：附件1的转写强制对齐和词级三模态特征。
- `align_q3.py`：附件4解释位置到原视频时间的映射。

输出文件未包含队伍身份信息。提交时排除 `code\cache` 中的模型下载文件；运行 `Get-ChildItem code\outputs\features_q1_prosody,code\outputs\runs\main,code\outputs\predictions_audio_dynamics_ensemble -Recurse | Measure-Object -Property Length -Sum` 并确认附件总大小满足题目50 MB上限。
