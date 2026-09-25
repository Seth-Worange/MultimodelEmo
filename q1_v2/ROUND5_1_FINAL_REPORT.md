# Q1 Round 5.1 Final Integration Freeze 报告

执行分支：`feature/q1-improvement`。本轮目标是把已验证模块整合为最终 Q1 feature pipeline 并冻结 schema，
完成 10 条真实 smoke test，新增 Q1 成果可视化页面。本轮**没有**修改官方 MP4、官方标签、
人工 Excel（`人工核验100条.xlsx`、`check5.xlsx`）、Q2/Q3 路径或旧版 `features_q1.py` 的行为；
人工标签只用于事后评估。`full_100_sample_feature_extraction = NOT_RUN`（未经批准不启动）。

## 1. 当前 pipeline

冻结入口：`python -m q1_v2.final_pipeline`（`--smoke10` 或 `--sample-id`，上限 10 条，无 `--all`）。

逐样本链路（`process_one`，全部真实执行）：

1. **Stage-1 QA 校验导入**（`_verified_qa`）：SHA256 + 官方文本核对 round4 correspondence QA，路由阈值不变；
2. **媒体抽取**：PyAV 原始 PTS → 16 kHz 单声道 WAV + `media_metadata.json`；
3. **对齐**（4 条策略，互斥）：
   `REUSED_VERIFIED_REAL_MFA`（校验源 SHA 后复用 round4 真实 MFA 并重映时间轴）、
   `NEW_LOCAL_MFA`（匹配区间局部裁剪 + MFA 精对齐）、
   `PROHIBITED_BY_ROUTER`（非 HIGH_CONFIDENCE_MATCH，禁止对齐，NaN 时间 + 掩码 0）、
   `MFA_FAILED_NO_PSEUDO_ALIGNMENT`（失败保留全部原词，不伪造对齐）；
4. **文本/音频特征**：round4 原生特征经校验复用，否则现场 BERT（768 维词级）+ 69 维帧级声学（词级 mean/std = 138 维）；
5. **视觉特征**：经 `VisualBackend` 抽象（`make_visual_backend`）走 OpenFace68 后端，四块独立保存；
6. **对齐置信度**：MFA–ASR 边界分歧分级（LOW/MEDIUM/HIGH/UNAVAILABLE），语义是不确定性不是精度；
7. **打包 + 校验**：`feature_package.npz`（`allow_pickle=False` 可重读）+ `feature_package_metadata.json`，
   `validate_package` 检查样本身份、词轴/帧轴长度、块维度、时间单调性、quality 块完整性。

VisualBackend 抽象（任务 A）已闭环：`VisualBackend(ABC)` → `OpenFace68Backend` / `MediaPipe478Backend`，
工厂 `make_visual_backend`，**最终 pipeline 只经抽象调用**（`final_pipeline.py` 不直接 import/调用 OpenFace、MediaPipe）。
OpenFace68 是默认后端（`Q1Config.visual_backend="openface68"`）；MediaPipe478 仅为基线/历史兼容桥
（`extract_native` 保留 Round-4 融合 schema，不改写历史输出）。历史遗留的直接调用仅存在于
QA 场景探针（correspondence_qa/visual_scene_qa，MediaPipe-only 语义）与旧版 CLI 选择器（`extract_with_backend`），
不影响最终特征路径；round3/round4/run.py 的 legacy 路径仍钉在 MediaPipe 融合 schema 以保持历史可复现。

## 2. 第五轮（Round 5）完成情况回顾

Round 5 已验证并保留（见 `ROUND5_REPORT.md`）：自动/人工 100 条 QA（一致率 90%）、
保守路由（precision 100%、recall 92.31%、**0 危险假阳性**）、VAD extra-speech 新定义、
`check5.xlsx` 人工逐词边界（5 条 73 词）对真实 MFA 校验（微平均边界误差 100.51 ms、IoU 0.653）、
真实 OpenFace 2.2.0 后端（官方 FeatureExtraction.exe + CE-CLM 模型，哈希留档）、
MediaPipe478 基线对照（6 条）、76 个合成测试。Round 5 停在“等待复核授权，不启动 full100”。

### 本轮开始时的实际断点（5.1 中断现场）

5.1 的集成代码（`final_pipeline.py`/`final_schema.py`）已写好但**未提交**；smoke test 已跑过一轮但
**5/10 失败**（产物保留于 `outputs/q1_v2_round5_1/interrupted_run_20260925/`）：
5 条无 round4 特征可复用的样本全部死在 `text_audio` 阶段（BERT 默认缓存
`C:\Users\Lenovo\.cache\huggingface` PermissionError）；其中 2 条 MATCHED 样本此前还叠加
`MFAUnavailableError: english_us_arpa`。诊断结论：

1. **BERT**：权重已完整存在于 `outputs/q1_v2_smoke/cache/huggingface/`（旧 run.py 的 HF_HOME 约定），
   `final_pipeline` 未设 HF_HOME 才回退到不可写默认路径。**模型无需重新下载**。
   （独立复核确认：指向该缓存并离线模式可正常加载，隐藏维 768。）
2. **MFA**：声学模型文件齐全（`D:\q1_v2_mfa_root`）。“acoustic model unavailable” 是
   **MFA 启动失败引起的误导性前置检查结果**：此前只把 `Scripts` 加入 PATH 而未完成 conda 激活，
   `Library\bin` 不在 PATH，`mfa.exe` 进口 `soundfile` 时 `libsndfile.dll` 解析失败（0x7e）。
   `conda activate q1-v2` 后 `mfa version` = 3.4.2、`mfa model list acoustic` = `english_us_arpa`。
   （此前“DLL 文件名不匹配”的初判不准确，已按复核结论更正：文件在 `Library\bin\sndfile.dll`，
   是 PATH 问题。）

## 3. 本轮补充完成

### 3.1 环境自持修复（代码级，不依赖操作者 shell 状态）

- `final_pipeline.main()` 现在把 MFA 所属 conda 环境的 `Library\bin` 与环境根目录并入 `PATH`
  （等价传播 `conda activate` 的关键部分），并 `setdefault("HF_HOME", output_dir/cache/huggingface)`
  与 run.py 约定一致。任意外部进程启动 pipeline 不再要求先激活环境。
- 本次运行以 `HF_HUB_OFFLINE=1` + 预置缓存执行（缓存已复制至 `outputs/q1_v2_round5_1/cache/`）。

### 3.2 Schema 与包格式补齐

- **quality 块**（任务九要求）：NPZ 新增标量 `speech_status` / `language_status` / `face_status`
  （与既有 `quality_status` 组成 correspondence/speech/language/face 四状态），metadata 新增嵌套
  `quality{correspondence_status,speech_status,language_status,face_status}`；全部来自自动 QA 证据。
- `validate_package` 强制校验四状态标量与 metadata quality 块；`feature_schema.json` 同步冻结。
- `MFA_FAILED_NO_PSEUDO_ALIGNMENT` 路径补写 `alignment_metadata.json`（与其余两条路径的溯源结构一致）。

### 3.3 Smoke 选样修正

原选样缺 **NON_ENGLISH** 覆盖。现选样规则（全部自动证据，不用人工标签）：
6 条赛题点名样本 + NO_SPEECH/无脸（`-NFrJFQijFE__1`）+ 静态人物图（`-UuX1xuaiiE__1`）+
NON_ENGLISH（自动 NON_ENGLISH_SPEECH 字典序首条 `-yRb-Jum7EQ__1`）+
普通 MATCHED（seed=2026 从自动 HIGH_CONFIDENCE_MATCH ∩ NORMAL_FACE_VIDEO 池抽取 `-HwX2H8Z4hY__5`）。
“无脸/静态视觉”类目由 `-NFrJFQijFE__1`（无脸）与 `-UuX1xuaiiE__1`（静态图）共同覆盖；
自动标志 `static_or_near_static_visual=True` 仅出现在 `-a55Q6RWvTA__3`（未入选，语义为运动静止，
与人工 `static_or_nonhuman` 不同口径）。

### 3.4 其他修正

- `features_q1.py` 回退机器特定绝对路径（恢复仓库相对默认，冻结合规）；
- `landmark_definition.py` 过时注释（“OpenFace 未实现”）更新；
- 新增合成测试：quality 块映射、`_package`→`validate_package` 往返、`select_smoke_ids` NON_ENGLISH 覆盖。
  **92/92 通过**（base python `D:\anaconda\python.exe -m pytest q1_v2\tests -q`，无重依赖）。

## 4. OpenFace68 说明（冻结）

- 后端：官方 OpenFace Windows 2.2.0 `FeatureExtraction.exe` + CE-CLM 模型（下载来源与逐文件 SHA256 记录于
  `outputs/q1_v2_round5/visual_backend/openface_download/.../model_download_report.json`；
  可执行文件 SHA256 `a29ba49c…96ae`）。**不是** MediaPipe 478 挑 68 点。
- 帧级：原生 `x_0..x_67,y_0..y_67`（几何 136 维；眼中心 36–41/42–47 中点平移 + 眼距欧氏尺度归一化，
  不做 roll、不编造 z）、AU 35 维（`_r` 强度 + `_c` 出现）、头姿 6 维、视线 8 维。
- 词级：区间内有效帧 mean + population std → 272/70/12/16（合计 370）。
  **保持 block 结构，不拼接不可解释向量**。
- 帧有效性：success、confidence≥0.5、几何有限、眼距>1px、PyAV PTS↔OpenFace 时间差≤60ms。
- 已知边界：OpenFace 只证明“有被跟踪的脸”，不证明“是说话者的脸”；多脸/身份问题保留为后续人工/说话者跟踪议题。

## 5. Feature schema（冻结版本 `q1_v2.5.1-openface-blocks-v1`）

- `outputs/q1_v2_round5_1/feature_schema.json` + `visual_feature_schema.json`。
- 词轴 = 官方原始词序全保留（不截断）；缺失策略 = NaN 特征 + 掩码 0，无合成时间、无最近帧填补。
- 维度：text 768 / audio 帧 69、词 138 / visual 块 136+35+6+8（词级 ×2）。
- 掩码：`text_available` / `audio_available` / `visual_available` / `alignment_available`。
- 包读取：`np.load(path, allow_pickle=False)`；逐词音频帧索引与视频原始帧索引保留在 metadata trace。

## 6. Smoke test 结果（10/10 完成，0 失败，0 MFA 失败）

执行命令见 §9。结果（`outputs/q1_v2_round5_1/smoke_test_summary.csv`）：

| 样本 | 质量状态 | 路由 | MFA 策略 | 词数 原/对齐 | 词级有效 T/A/V | 视觉覆盖 | 备注 |
|---|---|---|---|---|---|---|---|
| `-3g5yACwYnA__13` | MATCHED | HIGH_CONFIDENCE_MATCH | REUSED_VERIFIED_REAL_MFA | 15/15 | 15/15/15 | 100% | 人工边界样本 |
| `-3g5yACwYnA__2` | MATCHED | HIGH_CONFIDENCE_MATCH | REUSED_VERIFIED_REAL_MFA | 14/14 | 14/14/12 | 85.7% | OOV 样本 |
| `-3g5yACwYnA__3` | MATCHED | HIGH_CONFIDENCE_MATCH | REUSED_VERIFIED_REAL_MFA | 29/29 | 29/29/29 | 100% | 人工边界样本 |
| `-THoVjtIkeU__2` | MATCHED | HIGH_CONFIDENCE_MATCH | REUSED_VERIFIED_REAL_MFA | 10/10 | 10/10/9 | 90% | 人工边界样本 |
| `-s9qJ7ATP7w__6` | PARTIAL_MATCH | REVIEW_REQUIRED | PROHIBITED_BY_ROUTER | 5/0 | 5/0/0 | 0% | 保守拦截，无伪造对齐 |
| `-AUZQgSxyPQ__2` | PARTIAL_MATCH | REVIEW_REQUIRED | PROHIBITED_BY_ROUTER | 41/0 | 41/0/0 | 0% | 保守拦截 |
| `-NFrJFQijFE__1` | NO_SPEECH | INVALID_CORRESPONDENCE | PROHIBITED_BY_ROUTER | 16/0 | 16/0/0 | 0% | NO_SPEECH + 无脸 |
| `-UuX1xuaiiE__1` | MATCHED | HIGH_CONFIDENCE_MATCH | **NEW_LOCAL_MFA** | 27/23 | 27/23/22 | 81.5% | 静态人物图；本轮新跑真实 MFA |
| `-yRb-Jum7EQ__1` | NON_ENGLISH_SPEECH | INVALID_CORRESPONDENCE | PROHIBITED_BY_ROUTER | 49/0 | 49/0/0 | 0% | NON_ENGLISH（希伯来语） |
| `-HwX2H8Z4hY__5` | MATCHED | HIGH_CONFIDENCE_MATCH | **NEW_LOCAL_MFA** | 10/9 | 10/9/3 | 30% | 本轮新跑真实 MFA |

要点：

1. **NEW_LOCAL_MFA 路径首次真实走通**（此前从未执行）：`-UuX1xuaiiE__1`、`-HwX2H8Z4hY__5`
   本轮实际执行 MFA（`mfa_executed_this_run=True`，`mfa_real_output_available=True`）。
2. 4 条受限路由样本全部保持 0 伪造对齐（对齐掩码全 0、时间 NaN），文本特征完整保留。
3. 全部 10 包通过 `validate_package`（含新 quality 块），`allow_pickle=False` 可重读；
   visual_backend 全部为 `openface68`。
4. 对齐置信度汇总：LOW 24 / MEDIUM 38 / HIGH 36 / UNAVAILABLE 118（UNAVAILABLE 主要来自 4 条无对齐样本的词）。
5. 中断轮产物完整保留于 `outputs/q1_v2_round5_1/interrupted_run_20260925/`（含 5 条失败样本的 `_FAILED.json`）。

## 7. 可视化页面（`q1_viz/`）

- 结构：`src/`（style.css、charts.js 手写 SVG、main.js）、`public/`（`frontend_data.json` + 真实视频帧）、
  `tools/build_frontend_data.py`（数据生成）、`tools/validate_palette.py`（调色板校验 Python 移植）、
  `package.json`（`npm install` / `npm run dev`，Vite 开发服务器；亦可用 `python -m http.server`）。
- 页面模块：处理流程 → 数据质量总览（图1）→ Correspondence QA（图2 安全路由 / 图3 混淆矩阵 + extra-speech 表）
  → 时序对齐（图4 MFA vs 人工词边界 + 音频包络 + 真实视频帧）→ OpenFace68（图5 68 点拓扑 + 特征块 / 图6 后端对照）
  → 样本浏览器（10 样本质量标志、掩码维度、词级对齐）。
- **数据不硬编码**：全部数值由 `build_frontend_data.py` 从 `outputs/` 实验产物生成，
  页面 footer 列出来源文件。人工标签仅作为评估证据展示。
- 图表规范：≤24px 细条形、数据端 4px 圆角、2px 表面间隙、实线发丝网格、≥2 系列配图例、
  选择性直标、悬停提示 + 表格孪生；调色板（参考实例 slots 1–3）经六项计算校验通过
  （全对 CVD ΔE 9.2 浅色 / 9.4 深色，正常视觉 24.0 / 20.9，与基准文档一致）。
- 局限：本机无 Node.js 与浏览器自动化，`npm run dev` 未在本机实测（Vite 标准配置）；
  已用 `python -m http.server` 验证全部资源 HTTP 200 与 JSON 完整性；投屏前建议本机打开目检一次。

## 8. full100 是否执行

**`full_100_sample_feature_extraction = NOT_RUN`。** 本轮严格限于 10 条 smoke 样本（CLI 上限 10，无 `--all`）。
是否进入 full100 待批准（见 §10 建议）。

## 9. 复现命令

```powershell
# 环境（或依赖 pipeline 自身的 PATH 传播；建议与 5.1 相同）
conda activate q1-v2
$env:MFA_ROOT_DIR = 'D:\q1_v2_mfa_root'
$env:HF_HOME = (Resolve-Path 'outputs\q1_v2_round5_1\cache\huggingface').Path   # 或 outputs\q1_v2_smoke\cache\huggingface
$env:HF_HUB_OFFLINE = '1'

# 合成测试（base python）
& 'D:\anaconda\python.exe' -m pytest q1_v2\tests -q -p no:cacheprovider

# 10 条 smoke（幂等续跑：已成功的样本经校验后直接复用）
& 'D:\anaconda\envs\q1-v2\python.exe' -m q1_v2.final_pipeline `
  --data-root '..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条' `
  --qa-root 'outputs\q1_v2_round4' --round5-qa-root 'outputs\q1_v2_round5' `
  --output-dir 'outputs\q1_v2_round5_1' `
  --mfa-dictionary 'outputs\q1_v2_round3\dictionary\english_us_arpa_plus_two.dict' `
  --mfa-executable 'D:\anaconda\envs\q1-v2\Scripts\mfa.exe' `
  --mfa-root-dir 'D:\q1_v2_mfa_root' --mfa-work-dir 'D:\q1_v2_mfa_work' `
  --openface-executable 'outputs\q1_v2_round5\visual_backend\openface_download\extracted\OpenFace_2.2.0_win_x64\FeatureExtraction.exe' `
  --face-model 'outputs\q1_v2_round2\models\face_landmarker.task' --smoke10

# 可视化数据
& 'D:\anaconda\envs\q1-v2\python.exe' q1_viz\tools\build_frontend_data.py
```

## 10. 未完成与剩余风险

1. **full100 未运行**（设计如此）。若批准，预计可直接用同一入口把上限放宽执行；
   建议批准前确认附件存储预算（10 条 sample 输出约 26 MB，100 条外推约 250 MB+，
   提交附件需裁剪至特征包本体）。
2. OpenFace 静态图高检出 ≠ 说话者身份；多脸样本的说话者归属仍未解决（保留人工/后续跟踪）。
3. extra-speech VAD 规则语义正确但方向准确率未超 R4（同口径 71 条）；5 FPS 场景 QA 未胜出——
   不在本 100 条上继续调参，避免把评估集当训练集。
4. 配置仍有一处历史分叉：stage-1 `correspondence_qa.QAConfig.visual_probe_fps=2.0` 与
   `Q1Config.visual_probe_fps=5.0` 语义不同（粗探针 vs 场景探针），未强行统一以免改变已验证的
   路由行为；已在代码注释中区分。
5. 可视化页面未经真实浏览器目检（本机无 Node/浏览器自动化）；资源加载与数据完整性已验证。
6. MFA–ASR 分歧等级是不确定性指标，不能当时间对齐精度；人工边界误差才是精度口径（§ROUND5 D）。

## 11. 交付物

- 代码：`feature/q1-improvement`（`q1_v2/final_pipeline.py`、`final_schema.py` 首次入库 + Round5.1 全部改动）
- 结果：`outputs/q1_v2_round5_1/`（smoke 10/10 特征包、schema、experiment_config、置信度汇总；
  中断轮产物归档于 `interrupted_run_20260925/`）
- 页面：`q1_viz/`
- 测试：92/92 通过（base python 合成测试）
