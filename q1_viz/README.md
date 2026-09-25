# q1_viz · 问题1 数据处理成果展示页

轻度苹果风（Apple-inspired）+ 柔和活力色彩的结果展示页：主视频播放器（5 条典型样本可切换）+
便当盒（Bento Grid）数据卡 + 全量 100 条逐样本检视（关键点/对齐/声学/异常说明）+ 全量样本表。
**展示目标是"如何解决复杂场景多模态数据质量问题"，不是模型预测结果。**

## 部署（Cloudflare Pages）

将**本目录整体**作为静态站点部署（无需构建步骤）：

```
q1_viz/
├── index.html              # 页面入口（CSS/JS/聚合数据内联）
├── assets/
│   ├── data/*.json         # 100 条逐样本数据（页面按需懒加载）
│   ├── videos/*.mp4        # 100 条样本视频（单文件 ≤ 5 MB，均低于 20 MiB 上限）
│   └── thumbs/*.jpg        # 样本缩略图
├── site_data.json          # 与 index.html 内联的聚合数据同源
└── tools/                  # 数据生成与调色板校验脚本（部署不需要）
```

- Cloudflare Pages → Create application → **Upload assets** → 拖入 `q1_viz` 目录内容。
- 本地预览：`python -m http.server 8000` 后访问 `http://localhost:8000/`。
- 单样本深链：`index.html#sample=<sample_id>`（直达逐样本检视）。

## 页面结构

1. **典型样本**：5 条典型样本切换播放（默认 `-THoVjtIkeU__2`，人工边界误差最小样本），
   视频叠加 OpenFace 68 点关键点拓扑，词级时间戳可点击跳转。
2. **数据质量与安全路由**：人工 QA 分布、安全路由 2×2、Correspondence QA、时序对齐精度、置信度、视觉 QA。
3. **三模态特征**：声学时频图 + 波形、OpenFace68 四块、AU 词级均值。
4. **逐样本检视**：100 条全部可达（全量表点击 / 左右按钮 / 键盘方向键 / 触摸滑动切换）；
   被路由拦截样本的说明/警告卡片位于**视频左侧**，并保留各模态线索徽章。
5. **全量样本表**：标注、强度、时长、质量状态、路由、对齐词数、特征包状态，任意行点击跳转检视。

## 数据来源（不硬编码实验结果）

全部数值由 `tools/build_site_data.py` 从实验产物生成：

```bash
D:\anaconda\envs\q1-v2\python.exe q1_viz/tools/build_site_data.py
```

| 展示区块 | 来源 |
|---|---|
| KPI / 质量分布 / 安全路由 | `outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json` |
| 时序对齐精度 / 置信度 | `outputs/q1_v2_round5/manual_boundary/*`、`alignment_confidence/*` |
| 逐样本关键点（像素坐标） | `outputs/q1_v2_full100/samples/*/visual/openface68/openface_raw/openface68.csv` |
| 逐样本词级对齐（MFA/人工/ASR） | `outputs/q1_v2_full100/samples/*/word_alignment.csv` |
| 逐样本声学（波形/F0/RMS/ZCR/质心） | `outputs/q1_v2_full100/samples/*/audio_features.npz` 等 |
| 100 条全量表 | `label-100.xlsx`（官方标注，仅作元信息展示）+ 自动 QA + `feature_summary.csv` |

人工核验文件（`人工核验100条.xlsx`、`check5.xlsx`）只作为评估证据展示，不进入模型与路由。

## 特殊情况展示规则

每条样本保留可用模态的全部线索，说明卡位于视频左侧：

- **文本-音频疑似不一致**（PARTIAL_MATCH）：保守拦截词级对齐（掩码 0、时间 NaN），
  保留文本特征、ASR 词序线索与人脸关键点。
- **无语音 / 无音频 / 无脸**：保留其余模态线索（文本、场景/声学底噪、人脸帧记录），对齐不运行。
- **非英语语音**：保留文本特征与人脸线索，说明转写与语音不对应。
- **静态人物图 / 低视觉覆盖**：标注说话者身份存疑或视觉缺失策略（掩码 0，不近邻填补）。

## 设计说明

- 浅色 `#F5F5F7` 背景 + 柔和弥散渐变光晕；磨砂玻璃导航；大字号与留白；便当盒卡片网格。
- 图表标注规范：细条形、选择性直标、图例、悬停提示、表格孪生；调色板经
  `tools/validate_palette.py`（dataviz 校验器的忠实 Python 移植）六项计算校验通过。
