# Q2 最终实验交接记录

## 当前状态

- 分支：`shuke/q2-fuse-missing`
- 代码基线：`08d63ba docs(q2): add experiment handoff and fixed robustness suite`
- 本次完成：CUDA 下以冻结 B0 FUSE 配置重训 seed 2026、2027、2028；对三个最终 checkpoint 和两个历史 FUSE checkpoint 运行 50-case validation robustness；生成三 seed 汇总、诊断图与最终报告。
- 环境：`C:\Anaconda3\envs\pytorch\python.exe`，PyTorch 2.8.0+cu126，RTX 4060 Laptop GPU。未使用 Pytorch39。
- 未执行 push、merge、PR。

## 冻结方案和结果

最终方案为 **B0 FUSE**，学习率 `1e-3`，输入归一化关闭，结构化缺失训练关闭。按 validation selection score 保存最佳 checkpoint；测试集未用于模型选择或本次新分析。

| seed | best epoch | selection score | clean Macro-F1 | clean MAE | clean Neutral F1 |
|---:|---:|---:|---:|---:|---:|
| 2026 | 2 | 0.2983 | 0.6085 | 0.5779 | 0.5076 |
| 2027 | 4 | 0.3053 | 0.6005 | 0.5753 | 0.4360 |
| 2028 | 5 | 0.3141 | 0.5849 | 0.5869 | 0.4795 |
| mean ± SD | — | — | 0.5980 ± 0.0120 | 0.5800 ± 0.0061 | 0.4744 ± 0.0361 |

固定 R1–R5 鲁棒性平均 Macro-F1 为 `0.5685 ± 0.0112`，MAE 为 `0.6094 ± 0.0106`。B1（归一化）与 B2（结构化缺失）是 discovery 阶段的负结果；候选对比见 `FINAL_REPORT.md`。

## 主要结果文件

- `FINAL_REPORT.md`：完整实验协议、指标、历史比较、Neutral F1 与回归幅度分析。
- `final_summary.csv`、`final_multiseed_summary.csv`：逐 seed 与 mean/SD 汇总。
- `paired_delta.csv`：与历史 FUSE 的同 seed 描述性配对差值（n=2）。
- `neutral_f1_analysis.csv`、`regression_magnitude_shrinkage.csv`、`confusion_matrix.csv`：诊断明细。
- `confusion_matrix_final.{png,pdf,svg,tiff}`、`regression_magnitude_shrinkage.{png,pdf,svg,tiff}`：图表导出。
- `final_run_manifest.json`：checkpoint 哈希、设备与运行元数据。
- `Final/s2026/`、`Final/s2027/`、`Final/s2028/`：三 seed checkpoint、训练指标和运行配置。
- `Final/*_robustness.csv`：三个最终模型和两个历史模型各自的 50-case 验证结果。

## 复现命令

在 `MultimodelEmo` 工程目录运行：

```powershell
$py = 'C:\Anaconda3\envs\pytorch\python.exe'
& $py -m scripts.train --config config\q2_fuse_b0.yaml --device cuda --seed 2026 --lr 0.001 --output-dir outputs\q2_fuse_missing\Final\s2026
& $py -m scripts.train --config config\q2_fuse_b0.yaml --device cuda --seed 2027 --lr 0.001 --output-dir outputs\q2_fuse_missing\Final\s2027
& $py -m scripts.train --config config\q2_fuse_b0.yaml --device cuda --seed 2028 --lr 0.001 --output-dir outputs\q2_fuse_missing\Final\s2028
```

训练后对每个 seed 执行 `scripts.robustness`，然后运行：

```powershell
$env:NATURE_FIGURE_AUDIT_SCRIPTS = 'C:\Users\Orange\.codex\skills\nature-figure\scripts'
& $py -m scripts.finalize_q2_fuse --data-root data --device cuda
```

## 解读边界

历史 FUSE checkpoint 使用 corruption probability `0.8`、missing mode `auto`；最终 B0 使用 `0.0`、`whole`。历史比较使用一致的验证样本和缺失场景，但训练方案不同，因此配对差值只作描述性方案对比，不能解释为单一改动的因果效应。三 seed 标准差是 seed 间离散程度，不是置信区间。

最终执行状态：**push: NO；merge: NO；PR: NO。**
