# Q1 Round 5 真实验证报告

执行分支：`feature/q1-improvement`。本轮没有修改官方 MP4、官方标签、人工 Excel、Q2/Q3 或旧版 `features_q1.py`。100 条视频只运行轻量 QA；`full_100_sample_feature_extraction = NOT_RUN`。所有人工标签仅用于事后评估，没有进入路由决策、训练或阈值调参。

## A. 自动 100 条 QA

沿用 Round 4 的预设 Correspondence QA 主阈值及已有 ASR/VAD 证据。本轮对 100 条官方视频完成基于原始 PTS 的 5 FPS MediaPipe 场景探针，并重新计算 VAD 主导的 extra-speech。Round 5 轻量 QA 成功 100/100，失败 0；这不是 100 条完整 BERT/音频/MFA/视觉特征提取。逐条记录在 `outputs/q1_v2_round5/qa/correspondence_audit_round5.csv`，配置与版本在 `outputs/q1_v2_round5/experiment_config.json`。

## B. 人工 100 条 QA

只读导入 `../data/人工核验100条.xlsx`。100 行、100 个唯一 sample ID，与官方 100 条严格 1:1；原文件未改动。人工类别：MATCHED 78、TEXT_AUDIO_MISMATCH 13、NO_SPEECH 4、NON_ENGLISH_SPEECH 3、PARTIAL_MATCH 1、UNRESOLVED 1。源 SHA256、空值与规范化规则在 `manual_qa/manual_sample_audit_import_report.json`。数字/字符串 reviewer ID 保留为字符串；实际 reviewer ID 无空值。末尾 3 行的部分人工字段原本为空，规范化结果为 null，不推断标签。

## C. QA 与人工一致性及安全路由

Round 4 原分类与人工精确一致率 90/100=90%，将 NO_AUDIO 和 NO_SPEECH 仅在粗分类层合并后为 92%。预设 `HIGH_CONFIDENCE_MATCH` 72 条，人工均为 MATCHED：precision 72/72=100%，recall 72/78=92.31%，F1=96%，危险假阳性 0，保守假阴性 6。人工记录与自动记录的所有不一致见 `manual_qa/manual_auto_disagreements.csv`；完整混淆矩阵见 `manual_qa/manual_qa_confusion_matrix.csv`。没有为提高 recall 下调主阈值。

### Extra speech

新定义对匹配句区间前/后、扣除 0.15 s 容差后的 VAD 语音区间求并集时长，累计至少 0.20 s 才判定额外讲话；ASR 未匹配词只作附加证据。没有可信匹配句区间时返回 null。人工方向为 NONE/BEFORE/AFTER/BOTH 的有 79 条，其中仅 71 条同时具有 stage-1 高置信匹配区间，另 8 条不纳入同口径准确率。

在这 71 条上：

| 指标 | Round 4 ASR 规则 | Round 5 VAD 规则 |
|---|---:|---:|
| before precision / recall | 95.65% / 36.67% | 100% / 33.33% |
| after precision / recall | 69.23% / 75.00% | 43.64% / 100% |
| 完全方向正确率 | 36.62% | 29.58% |

因此新规则**定义更符合“存在额外讲话”但在当前人工标签上的方向准确率没有提高**，尤其 after 假阳性 31 条。VAD 可把句尾延续声、噪声或非词语声段判成语音；本轮不据同一 100 条人工标签事后调阈值。详见 `qa/extra_speech_metrics.json`、`qa/extra_speech_confusion_matrix.csv`。

### 视觉轻量 QA

拆分 `any_face_detected`、`usable_face_present`、`static_visual` 和 `nonhuman_or_no_face_scene`。可用脸要求面积至少 1% 画面、合格帧覆盖至少 20%、连续可见至少 0.4 s，并排除近静态画面；这些是工程启发式，不是说话者识别。5 FPS 只用于 scene QA，后端特征对照是 10 FPS。

人工 `face_present` 100 条均可判。2 FPS 旧探针 any-face 准确 93%，5 FPS 新探针 92%，因此采样提速**未带来准确率提升**。5 FPS usable-face 的 87% 与人工 any-face 不是同一语义，不作为改进结论。场景状态：NORMAL_FACE_VIDEO 82、NON_FACE_DYNAMIC_SCENE 13、MIXED_OR_UNRESOLVED 5。人工 `static_or_nonhuman` 合并两个概念，故不计算它与纯运动 `static_visual` 的“准确率”。详见 `qa/face_qa_metrics.json` 和 `qa/face_qa_disagreements.csv`。

## D. 更新后的 `check5.xlsx` 人工边界

重新只读导入后，文件**实际为 5 条样本、73 个原词、73 个完整起止区间**，而不是假定 4/5 条。全部词索引从 0 开始且与官方词序一致，无越界、非单调、重复词索引或原词不一致；73 个 reviewer ID 单元格为空，这是来源现状，不推定多人一致性。源 SHA256 `d11de7d0e85375c775a1faaea19b9f1a3915920faf3297f6be754566b03e2055`。`inter-rater agreement = NOT_EVALUATED`。

| 样本 | 人工参考词 | 与真实 MFA 比较词 | 平均边界误差 | 平均 temporal IoU | 50/100/200 ms 双边界通过率 |
|---|---:|---:|---:|---:|---:|
| `-3g5yACwYnA__13` | 15 | 15 | 63.97 ms | 0.640 | 60.0% / 73.3% / 86.7% |
| `-3g5yACwYnA__2` | 14 | 14 | 63.29 ms | 0.645 | 28.6% / 71.4% / 92.9% |
| `-3g5yACwYnA__3` | 29 | 29 | 157.03 ms | 0.610 | 51.7% / 58.6% / 69.0% |
| `-THoVjtIkeU__2` | 10 | 10 | 43.55 ms | 0.808 | 80.0% / 90.0% / 90.0% |
| `-s9qJ7ATP7w__6` | 5 | 0 | null | null | null |

最后一条 stage-1 为 PARTIAL_MATCH/REVIEW_REQUIRED，stage-2 仍未满足保守晋级门槛，**没有运行 MFA**。人工有 5 个区间并不能替代自动对齐；该样本误差保持 null。

前 4 条宏平均（先按样本平均）边界误差 81.96 ms、IoU 0.676；68 个可比较词微平均边界误差 100.51 ms、IoU 0.653。微平均起点 MAE 80.56 ms、终点 MAE 120.47 ms，50/100/200 ms 双边界通过率 52.94% / 69.12% / 80.88%。Round 5 未改变 MFA 数值对齐算法，Round 4 vs Round 5 的数字相同；先前只用 15 词的 63.97 ms 不可与本次 68 词的 100.51 ms 当成性能变化。人工逐词详情、每样本指标、总体指标和同源对照在 `manual_boundary/`。

## E. 真正的 OpenFace68 后端

基于[官方 OpenFace Windows 2.2.0 release](https://github.com/TadasBaltrusaitis/OpenFace/releases) 的 `FeatureExtraction.exe`，按[官方模型下载脚本](https://github.com/TadasBaltrusaitis/OpenFace/blob/OpenFace_2.2.0/download_models.ps1) 补齐 CE-CLM 模型；来源、每个模型 SHA256、可执行文件 SHA256 记录于 `visual_backend/openface_download/extracted/OpenFace_2.2.0_win_x64/model_download_report.json`。真实命令及 stdout/stderr 分别保存在六条样本的 `samples/<sample_id>/openface68/`；不是从 MediaPipe 478 随意挑 68 点冒充 OpenFace。

解析 OpenFace 原生 `x_0..x_67, y_0..y_67`，帧几何维度 **68×2=136**。使用 36–41 与 42–47 两侧眼点均值之中点做平移原点，以眼中心欧氏距离做尺度归一化；不进行 roll 旋转，也不编造 z。另保留原生 35 个 AU 值、6 个 pose 值、8 个 gaze 值，独立分块，不拼接 MediaPipe blendshape。对应词级各块为均值+标准差，维度 272/70/12/16。只有 OpenFace success、confidence≥0.5、几何有限、眼距>1 像素及 PyAV/OpenFace 时间映射可靠时才标记帧有效。PyAV PTS 定位同一原始帧，时间差超 60 ms 则拒绝映射。NPZ 可用 `allow_pickle=False` 重读；词级 trace 保留真实原始帧索引。

OpenFace 只能给出被跟踪的脸，不能证明那张脸是讲话者。静态人物图中高检出率尤其不能视为动态说话者特征。多脸和身份不确定问题仍需后续人工/说话者跟踪解决。

## F. MediaPipe478 基线与 10 FPS 对照

6 条代表视频均以同一 PyAV PTS 选帧运行 MediaPipe478 与 OpenFace68，含正常单人 `-3g5yACwYnA__13`、人工无脸 `-NFrJFQijFE__1`、静态人物图 `-UuX1xuaiiE__1`、复杂画面 `-HwX2H8Z4hY__2`，以及另外两条含人工边界样本。压缩体积含当前后端的帧级和有 MFA 时的词级 NPZ；不含原视频、模型、WAV、OpenFace CSV 原文。运行时间为本机单次 CPU 测量，不是严格隔离的性能基准。

| 样本 | 选帧 | MediaPipe 有效帧 | OpenFace 有效帧 | MediaPipe/ OpenFace 压缩体积 | 词级视觉有效数 MP/OF |
|---|---:|---:|---:|---:|---:|
| `-3g5yACwYnA__13` | 55 | 55 | 55 | 449591 / 51892 B | 15/15 |
| `-s9qJ7ATP7w__6` | 25 | 19 | 24 | 109145 / 16361 B | null/null，无 MFA |
| `-UuX1xuaiiE__1` | 104 | 5 | 104 | 33486 / 59255 B | null/null，无 MFA |
| `-HwX2H8Z4hY__2` | 40 | 9 | 11 | 54106 / 9333 B | null/null，无 MFA |
| `-THoVjtIkeU__2` | 43 | 43 | 43 | 337006 / 39730 B | 9/9（10 词） |
| `-NFrJFQijFE__1` | 57 | 0 | 0 | 4267 / 2939 B | null/null，无 MFA |

MediaPipe 帧向量为原生 478×3 + 52 blendshape = 1486；OpenFace 几何块为 136，外加分离的 AU/pose/gaze，不能只比较两个数字就声称性能相同。逐样本实际 runtime、大小、真实帧数及视觉词覆盖在 `visual_backend/visual_backend_comparison.csv`。在静态图 `-UuX1xuaiiE__1`，OpenFace 104/104 帧检出脸但动态说话者身份未知，这不是“100% 视觉正确率”。

独立选择入口 `python -m q1_v2.visual_backend --backend openface68|mediapipe478 ...` 已分别在真实样本上复核，默认候选为 OpenFace68。既有融合归档仍明确是 MediaPipe478，不能因候选后端出现而改写历史输出 schema。

## G. 合成测试、真实执行与环境

新增 Round-5 合成单测 42 个参数化用例，连同既有测试共 **76/76 通过**。执行命令：

```powershell
& 'D:\anaconda\python.exe' -m pytest q1_v2\tests -q -p no:cacheprovider
```

测试覆盖 100 行映射、重复/缺失/额外 ID、大小写/布尔/reviewer 规范化、多样本边界、索引/单词错误、时间越界/非单调与缺词、VAD 前后额外讲话、无 ASR 词仍有 VAD、any-face 与 usable-face、OpenFace 68 维/失败帧/PTS 拒绝、ASR–MFA 分歧等级、stage-2 不覆盖 stage-1、NPZ 禁止 pickle。合成输入仅验证接口，不充当真实实验。

真实运行的项目环境为 `D:\anaconda\envs\q1-v2\python.exe`，Python 3.11.16、NumPy 2.4.6、PyAV 18.1.0、MediaPipe 1.0.1、Torch 2.8.0+cpu、Transformers 5.17.0、librosa 0.11.0、openpyxl 3.1.5。该环境没有 pytest；尝试 `pip install pytest==8.4.2` 时受限包索引返回“无可用版本”，故用已有 base Python 的 pytest 7.4.4 跑纯合成测试。此失败不影响项目环境真实 QA/视觉运行，但需要在可联网环境中补齐测试依赖。OpenFace release/模型 hashes 和每条运行命令均有日志；OpenFace 参数依据[官方命令行说明](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments)及[输出格式](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)。

复现实验示例（仓库根目录，Windows PowerShell）：

```powershell
$py = 'D:\anaconda\envs\q1-v2\python.exe'
$data = '..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条'
$out = 'outputs\q1_v2_round5'
& $py -m q1_v2.manual_sample_audit_import --manual-xlsx '..\data\人工核验100条.xlsx' --data-root $data --output-dir $out --auto-audit 'outputs\q1_v2_round4\correspondence_audit.csv'
& $py -m q1_v2.manual_boundary_import --manual-xlsx '..\data\附件1-数据集原始多模态样本\check5.xlsx' --data-root $data --output-dir $out --round4-root 'outputs\q1_v2_round4'
& $py -m q1_v2.round5_qa --data-root $data --round4-root 'outputs\q1_v2_round4' --output-dir $out --manual-normalized-csv "$out\manual_qa\manual_sample_audit_normalized.csv" --face-model 'outputs\q1_v2_round2\models\face_landmarker.task' --all-qa
& $py -m q1_v2.alignment_confidence --data-root $data --round4-root 'outputs\q1_v2_round4' --output-dir $out --manual-detail-csv "$out\manual_boundary\manual_alignment_detail.csv"
& $py -m q1_v2.stage2_qa --data-root $data --round4-root 'outputs\q1_v2_round4' --output-dir $out --asr-python 'outputs\q1_v2_round4\asr_env_py312\Scripts\python.exe' --asr-model 'outputs\q1_v2_round4\models\faster-whisper-base' --manual-csv "$out\manual_qa\manual_sample_audit_normalized.csv"
```

OpenFace 安装和代表视频运行的确切命令与下载 hash 记录在各 `openface_command.json`、`openface_stdout.log`、`openface_stderr.log` 和 `visual_backend/openface_download/`；`visual_backend_comparison.csv` 由 `python -m q1_v2.visual_backend_compare` 生成，`quality_summary.json` 经 `python -m q1_v2.round5_finalize --output-dir $out` 做真实 NPZ 完整性校验。模型二进制及大体积研究输出留在 Git 忽略的 `outputs/` 中，不提交仓库。

## H. 未运行和剩余风险

1. **没有运行完整 100 条最终特征提取**，也没有在本轮训练新的情感模型。既有完整融合输出仍是 Round 4 的少量样本；Round 5 的 OpenFace 帧/词块单独保存，尚未替换旧融合接口，以免影响 Q2/Q3。
2. 更新后的 `check5` 第五条虽有 5 个完整人工区间，但自动对应关系未被保守路由放行，MFA 不运行，人工边界准确率为 null。
3. 新 VAD extra-speech 语义正确但现有规则的人工方向指标下降；5 FPS 场景 QA 的 any-face 准确率也未胜出。需要额外独立样本和听辨诊断，不能在这 100 条上调完参数再称独立验证。
4. OpenFace 在静态人物图上高频检脸，仍未确定说话者身份。OpenFace 与 MediaPipe 特征没有可直接互换的维度；需要后续单独审批再确定最终后端及其融合 schema。
5. `MFA–ASR disagreement` 68 词中 HIGH 27、MEDIUM 24、LOW 15、UNAVAILABLE 2。人工边界最大误差 >200 ms 的 13 词仅 8 词被 LOW/UNAVAILABLE 标出，不能把一致性等级当真实时间对齐准确率。
6. 二阶段同一 faster-whisper-base 的 beam 10 复核 9 条，晋级 0；安全路由 recall 未提升，也未引入新的危险假阳性。它不是独立模型或独立测试集证据。

停止于此，等待人工复核与下一步授权，不启动 full-100 完整特征处理。
