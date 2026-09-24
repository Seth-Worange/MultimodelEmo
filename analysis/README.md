# `analysis/` — 论文数据分析与绘图模块

本目录是**论文里每一个数字、每一张图、每一张表的唯一来源**。论文正文不手抄任何数值：
`thesis/paper.tex` 通过 `\input{tables/...}` 引入表格，通过 `\includegraphics{figures/...}`
引入插图，而 `thesis/figures/` 与 `thesis/tables/` 全部由本目录的脚本从落盘的实验结果自动生成。
因此"论文结论 ↔ 实验产物 ↔ 源代码"三者可逐项溯源。

## 目录结构

| 文件 | 职责 |
| --- | --- |
| `style.py` | 出版级绘图样式：中文字体、Okabe–Ito 色盲安全配色、字号/线宽、`save()` 三格式导出（PDF 矢量 + PNG 预览 + 灰度预览） |
| `loader.py` | 统一的数据装载层。封装 `outputs/`、`data/`、`outputs/experiments/` 的读取与字段解析，对上提供语义化接口（`robustness()`、`deep_experiment()`、`q3_evidence()`、`diagnostics_facts()` …） |
| `measure_diagnostics.py` | **数据质量诊断**：直接读原始附件，实测四类数据缺陷（容器碎片化、网格稀疏与截断、零值语义、额外缺失分布），落盘为 `results/data_diagnostics.csv`、`diagnostics_facts.json`、`grid_evidence.csv` 等 |
| `make_figures.py` | 生成论文全部 13 张插图 |
| `make_tables.py` | 把实验结果渲染为 LaTeX 片段（booktabs 三线表 / `longtable`），共 18 张表 |
| `measure_extra.py` | 计算需要前向推理才能得到的「额外测量量」（模态归因份额），落盘到 `results/` 后再供绘图/建表引用 |
| `results/` | 上述额外测量量与诊断结果的 CSV / JSON 产物 |

## 运行方式

在**工程根目录**（`MultimodelEmo/`）执行：

```bash
# 数据质量诊断（读原始附件，产出 results/ 下的诊断清单与关键数字）
python -m analysis.measure_diagnostics

# 生成全部插图（同时写出 thesis/figures/*.pdf|*.png|*_gray.png）
python -m analysis.make_figures

# 只重绘其中几张
python -m analysis.make_figures --only data_profile training robustness

# 生成全部表格片段（写入 thesis/tables/*.tex）
python -m analysis.make_tables

# 重新计算模态归因（需要 GPU/CPU 前向推理，依赖 outputs/runs/ 下的检查点）
python -m analysis.measure_extra
```

Windows + Anaconda 示例：

```powershell
cd MultimodelEmo
$env:PYTHONPATH = (Get-Location).Path
C:\Anaconda3\envs\pytorch\python.exe -m analysis.measure_diagnostics
C:\Anaconda3\envs\pytorch\python.exe -m analysis.make_figures
C:\Anaconda3\envs\pytorch\python.exe -m analysis.make_tables
```

## 绘图规范（`style.py` 中固化，避免逐图重复设置）

1. **尺寸即最终尺寸**：`figsize` 直接取论文版面实际宽度（正文宽 165 mm → `TEXT_WIDTH_IN = 6.42`，
   半栏 `HALF_WIDTH_IN = 3.10`），LaTeX 端 `\includegraphics` 不再缩放，保证图中字号是真实字号。
2. **字号 7–9 pt**，最小不低于 6 pt；线宽 0.8–1.4 pt；去顶/右边框，刻度朝内。
3. **色盲安全 + 冗余编码**：分类信息同时用颜色与线型/marker/填充纹理编码，灰度打印仍可区分；
   每张图另存 `*_gray.png` 用于自检。
4. **中文字体**：优先 Microsoft YaHei / SimHei，并设 `axes.unicode_minus = False` 修复负号方块。
5. **一图一论点**：图内不重复图注信息；面板用 `panel_labels()` 统一标注 (a)(b)(c)；
   数值标注用 `annotate()` 统一格式。

## 插图与论文论点的对应

| 图 | 对应论点 |
| --- | --- |
| `fig_data_profile` | 数据画像：50 位网格的有效内容位仅 $m-2$ 个、强度近似单峰、类别不平衡 |
| `fig_data_defects` | 四个附件的缺陷画像：容器碎片化、稀疏与截断、额外缺失分布、通道失效 |
| `fig_grid_layout` | 索引层结构反演：子词复制展开规则（两条样本实证） |
| `fig_grid_evidence` | 结构反演的三条定量证据：零值语义、复制规则、逐位置有效频率 |
| `fig_q1_alignment` | 词级强制对齐 + 词区间池化的三联验证（文本区间/能量/基频/视觉覆盖） |
| `fig_q1_coverage` | 视觉覆盖率分档、最近帧补位比例与强制对齐置信度分布 |
| `fig_training` | 训练曲线与四视图选型分数（早停位置） |
| `fig_robustness` | 整段缺失与连续区间缺失下的性能变化规律 |
| `fig_modality_ablation` | 相对纯文本的性能变化：声画增益落在强度回归而非极性分类 |
| `fig_attribution` | 门控基线 vs 三因子分解的模态归因对照 |
| `fig_q3_evidence` | 问题三：词位时间定位与局部证据窗口 |
| `fig_confusion` | 测试集混淆矩阵（门控基线 vs FUSE-Net） |
| `fig_fuse_regularization` | 三因子分解正则项训练动态 |

## 制图避坑清单（本目录在迭代中固化下来的规则）

1. 柱形图不得从非零基线截断：需要比较小差异时改用「相对基准的变化量」或点图。
2. 结论标注统一用 `style.annotate()`，并把坐标轴的 `ylim`/`xlim` 预留出标注所需的空白，
   不允许标注压在数据、图例或坐标轴上。
3. 面板标题不要长过坐标区宽度，否则会与 `panel_labels()` 打的 (a)(b)(c) 相撞；
   标题过长时缩短标题并把细节移入图注。
4. 参考线若与左上角标注冲突，改用半高线（`vlines`）而不是贯穿全图的 `axvline`。
5. 热力图用顺序色标（`Blues`），并在格内写出数值，不再额外挂冗余 colorbar 挤窄坐标区。
6. 单元格文本必须转义 `%` 与 `&`；`make_tables.render()` 与 `longtable` 生成器都会在出口
   fail loudly，避免生成"注释掉行尾、触发 Extra alignment tab"的隐性坏表。
7. 行数很多的表用 `longtable`，否则整表被推到下一页并在前一页留下大面积空白。

## 复现约定

- 全部实验由 `scripts/` 下的入口产出，落盘于 `outputs/runs/<run>/` 与 `outputs/experiments/`；
  本目录只做**读取与呈现**，不训练、不改动原始张量。
- 随机种子、环境版本与权重哈希记录在 `MultimodelEmo/experiment.md` 与 `outputs/` 的元数据中。
- 若 `outputs/` 下缺少某个 run，对应图表会显式报错而不是静默降级。
