# 问题三独立预测骨干：首轮实验

## 研究边界

问题三需要极性、强度、模态作用和视频证据。附件4没有标签；独立骨干只能使用附件2训练集拟合，并使用附件2验证集选模。附件4的“可解释专项”不等于所有样本的三模态特征都完整：例如样本13的视觉特征整段为零。所有候选沿用50位对齐输入和同一自然缺失掩码，不用附件4预测结果调整参数。

## 结构选择

MulT 的跨模态注意力针对异步模态与长距离交互；本题提供50位对齐网格，因此没有必要照搬其无对齐全连接结构。这里采用文本时序作查询，在整段语音和视觉编码中检索上下文，再沿时间轴汇总。MISA 的模态共享／私有表示说明单纯拼接可能丢失模态差异，但已有 FUSE 模型包含相关分解，因此不重复叠加同类正则。上述论文只提供设计启发，网络融合与输出层均由附件2标签自主训练。

参考：[MulT 原论文](https://aclanthology.org/P19-1656/)；[MISA 原论文](https://arxiv.org/abs/2005.03545)。

## 三个可复现候选

1. `q3_temporal`：从头训练三路 BiGRU、声画跨时间注意力、全局时序编码和分类／强度头。只用完整输入监督，按完整输入验证分数选轮次。配置 `config/q3_backbone.yaml`。
2. `q3_fuse_adapt`：从用户选定的 `q2_fuse_lowaux_s2026` 完整权重初始化，取消合成缺失与分解辅助损失，以低学习率继续训练。配置 `config/q3_fuse_adapt.yaml`。
3. `q3_residual`：冻结相同的问题二主线权重，只训练跨时段交互残差；残差零初始化，训练前输出与源模型逐元素一致。配置 `config/q3_residual.yaml`。

评价均在附件2原验证集728条样本上，使用同一检查点还原、`sample_v2`协议、`neutral_zero=true`。完整输入是问题三的主要选模视图；局部、整段和连续缺失仅作诊断。分数定义为 `0.5*(1-MacroF1)+MAE/6`，越低越好。

| 预测器 | 最佳轮 | Accuracy | Macro-F1 | MAE | Pearson | 完整输入分数 |
|---|---:|---:|---:|---:|---:|---:|
| Q2主线原模型 | 2 | 0.6415 | **0.6313** | **0.5640** | **0.6692** | **0.2784** |
| 完整输入低学习率适配 | 1 | **0.6442** | 0.6279 | 0.5682 | 0.6667 | 0.2807 |
| 冻结骨干时序残差 | 1 | 0.6332 | 0.6239 | 0.5702 | 0.6634 | 0.2831 |
| 从头训练时序交互 | 4 | 0.6058 | 0.5969 | 0.6092 | 0.6210 | 0.3031 |

结论：三个独立训练候选都没有改善完整输入主分；低学习率适配仅有 Accuracy 小幅增加，同时 Macro-F1 与 MAE 下降，因此不应据单一 Accuracy 晋级。时序残差在局部和连续缺失下的 F1 略有改善，但问题三主要关心完整输入，此收益不足以替代主线模型。当前保留独立骨干代码与检查点用于后续消融，正式预测仍建议使用 Q2 单模型主线。不能把附件4无标签样本的解释视觉效果当成准确率证据。

配对逐样本比较显示，残差模型使28条完整输入验证样本改变类别，其中10条由错变对、16条由对变错。它对强情绪切片（$|y|>1$，199条）的MAE从1.044降至0.917，但整体分类和整体MAE仍退化。这是值得后续分析的强度收益，不足以单独宣布问题三预测器性能提升。

## 复现

在项目根目录使用 `pytorch` 环境运行：

```powershell
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config/q3_backbone.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config/q3_fuse_adapt.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.train --config config/q3_residual.yaml
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.evaluate --config config/q3_residual.yaml --split valid
& 'C:\Anaconda3\envs\pytorch\python.exe' -m scripts.infer --config config/q3_residual.yaml
```

各配置记录训练种子2026、优化设置、源检查点和输出路径。验证 JSON 与逐样本 CSV 位于 `outputs/diagnostics/q3_*`；附件4候选推理结果在各自 `outputs/predictions_q3_*`。问题二正式检查点没有被覆盖。
