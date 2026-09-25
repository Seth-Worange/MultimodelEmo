# q1_viz · 问题1 数据处理成果展示页

轻度苹果风（Apple-inspired）+ 柔和活力色彩的结果展示页：主视频播放器 + 便当盒（Bento Grid）数据卡 +
逐样本关键点/对齐/声学检视 + 100 条全量样本表。**展示目标是"如何解决复杂场景多模态数据质量问题"，不是模型预测结果。**

## 部署（Cloudflare Pages）

将**本目录整体**作为静态站点部署即可（无需构建步骤）：

```
q1_viz/
├── index.html          # 唯一页面（CSS/JS/展示数据全部内联）
├── assets/
│   ├── videos/*.mp4    # 10 条 smoke 样本视频（供播放器播放）
│   └── thumbs/*.jpg    # 样本缩略图
├── site_data.json      # 与 index.html 内联数据同源（可单独分发）
└── tools/              # 数据生成与调色板校验脚本（部署不需要）
```

- Cloudflare Pages → Create application → **Upload assets** → 拖入 `q1_viz` 目录（或该目录内容）。
- 仅上传 `index.html` 时页面与全部数据仍可完整展示，视频区会提示需要 `assets/videos/`。
- 本地预览：`python -m http.server 8000` 后访问 `http://localhost:8000/`。
- 单样本深链：`index.html#sample=-s9qJ7ATP7w__6`（自动定位检视器并选中该样本）。

## 数据来源（不硬编码实验结果）

全部数值由 `tools/build_site_data.py` 从实验产物生成：

```bash
D:\anaconda\envs\q1-v2\python.exe q1_viz/tools/build_site_data.py
```

| 展示区块 | 来源 |
|---|---|
| KPI / 质量分布 / 安全路由 | `outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json` |
| 时序对齐精度 / 置信度 | `outputs/q1_v2_round5/manual_boundary/*`、`alignment_confidence/*` |
| 逐样本关键点（像素坐标） | `outputs/q1_v2_round5_1/smoke_test_details/*/visual/openface68/openface_raw/openface68.csv` |
| 逐样本词级对齐（MFA/人工/ASR） | `smoke_test_details/*/word_alignment.csv` + round5 `manual_boundary/*` |
| 逐样本声学（波形/F0/RMS/ZCR/质心） | `smoke_test_details/*/audio_features.npz` 与 `audio_mfa_16k_mono.wav` |
| 100 条全量表 | `label-100.xlsx`（官方标注，仅作元信息展示）+ `outputs/q1_v2_round4/correspondence_audit.csv` |

人工核验文件（`人工核验100条.xlsx`、`check5.xlsx`）只作为评估证据展示，不进入模型与路由。
全量特征提取 = **NOT_RUN**（页面如实标注）。

## 特殊情况展示规则

每条样本保留可用模态的全部线索，并在说明卡标注特殊情况：

- **文本-音频疑似不一致**（`-s9qJ7ATP7w__6`、`-AUZQgSxyPQ__2`）：保守拦截词级对齐（掩码 0、时间 NaN），
  保留文本特征、ASR 词序线索与人脸关键点（24/25 帧有效）。
- **无语音 + 无脸**（`-NFrJFQijFE__1`）：保留文本特征与场景/声学底噪线索，对齐不运行。
- **非英语**（`-yRb-Jum7EQ__1`）：保留 49 词文本特征与 292/293 帧人脸线索。
- **静态人物图**（`-UuX1xuaiiE__1`）：人脸持续检出但说话者身份未确认；文本-音频真实 MFA 对齐仍可用。
- **低视觉覆盖**（`-HwX2H8Z4hY__5`）：视觉词级特征按缺失策略置 0，不做近邻填补。

## 设计说明

- 浅色 `#F5F5F7` 背景 + 柔和弥散渐变光晕；磨砂玻璃导航；大字号与留白；便当盒卡片网格。
- 图表标注规范：细条形、选择性直标、图例、悬停提示、表格孪生；调色板经
  `tools/validate_palette.py`（dataviz 校验器的忠实 Python 移植）六项计算校验通过。
