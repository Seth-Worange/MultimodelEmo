# q1_viz · 问题1 数据处理成果展示页

科研展示风格的静态页面，展示「复杂场景下多模态情感识别 · 问题1」的数据质量诊断、
文本-音频一致性检测、时序对齐与 OpenFace68 视觉特征构建成果。
**目的不是展示模型预测结果，而是展示如何解决复杂场景多模态数据质量问题。**

## 目录结构

```
q1_viz/
├── index.html                 # 页面入口（模块 1–5）
├── package.json               # npm install / npm run dev（Vite 开发服务器）
├── src/
│   ├── style.css              # 设计令牌（浅色/深色）、版式、组件
│   ├── charts.js              # 手写 SVG 图表（条形/分组条形/热表/时间轴/68点示意）
│   └── main.js                # 数据装配与模块渲染
├── public/
│   ├── frontend_data.json     # 由工具生成的全部展示数据（禁止手填数值）
│   └── thumbs/                # 典型样本真实视频帧
├── tools/
│   ├── build_frontend_data.py # 从 outputs/ 实验产物聚合 frontend_data.json
│   └── validate_palette.py    # 图表调色板六项检查（dataviz 校验器的 Python 移植）
└── README.md
```

## 运行

```bash
cd q1_viz
npm install
npm run dev        # http://localhost:5173
```

无 Node.js 时可用任意静态服务器（数据文件同目录回退已内置）：

```bash
python -m http.server 8000   # 在 q1_viz/ 目录下
```

## 数据生成（不硬编码实验结果）

页面所有数值来自实验产物，由以下命令生成/刷新：

```bash
D:\anaconda\envs\q1-v2\python.exe q1_viz/tools/build_frontend_data.py
```

数据来源（只读）：

| 区块 | 来源文件 |
|---|---|
| 数据质量总览 | `outputs/q1_v2_round5/manual_qa/manual_qa_metrics.json` |
| 混淆矩阵 | `outputs/q1_v2_round5/manual_qa/manual_qa_confusion_matrix.csv` |
| Correspondence QA | 同上 + `manual_auto_disagreements.csv` |
| Extra speech | `outputs/q1_v2_round5/qa/extra_speech_metrics.json` |
| 人工逐词边界 | `outputs/q1_v2_round5/manual_boundary/*` |
| 对齐置信度 | `outputs/q1_v2_round5/alignment_confidence/*` |
| 视觉后端对比 | `outputs/q1_v2_round5/visual_backend/visual_backend_comparison.csv` |
| 特征包与样本浏览器 | `outputs/q1_v2_round5_1/smoke_test_summary.csv` 与 `smoke_test_details/*` |

人工核验文件（`人工核验100条.xlsx`、`check5.xlsx`）仅作为评估证据展示，不进入任何模型或路由。

## 设计说明

- 图表遵循可计算的设计规范：条形 ≤24px、数据端 4px 圆角、2px 表面间隙、实线发丝网格、
  ≥2 系列必配图例、数值标签选择性直标、文字使用墨色令牌、悬停提示 + 表格孪生视图。
- 调色板为参考实例（slots 1–3），经 `tools/validate_palette.py` 验证：
  全对 CVD ΔE 9.2（浅色）/ 9.4（深色），正常视觉 ΔE 24.0 / 20.9，全部通过。
- 深色模式同时支持系统偏好与 `prefers-color-scheme`。
- 无动画、无商业仪表盘装饰，适配论文/答辩投屏。
