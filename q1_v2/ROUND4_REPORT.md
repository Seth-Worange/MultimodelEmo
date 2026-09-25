# Q1 第四轮：先人工核验入口，再对应关系 QA

执行日期：2026-09-25。工作分支：`feature/q1-improvement`。本报告中的“100 条”仅指轻量对应关系 QA；**没有**对 100 条运行 MFA、BERT、声学和视觉完整特征提取。所有自动分类均待人工复核，也不是情感识别或时间边界准确率。

## 顺序和数据边界

1. 先生成 `outputs/q1_v2_round4/manual_sample_audit_template.csv`：来自附件 1 Excel 的 100 个不同 `(video_id, clip_id)`，手工判断列留空，自动列初始为 `NOT_EVALUATED`。
2. 再实现并运行 VAD、ASR、文本局部匹配、视觉探针和轻量 QA。重跑只刷新自动列，不清除任何手工列。目前 100 条自动列均有结果，**人工样本级审阅仍为 0 条**。
3. 用户提供的 `C:\Users\Lenovo\Desktop\-3g5yACwYnA__13_mannual.xlsx` 只作为 `-3g5yACwYnA__13` 的词边界参考。原样副本保存于 `outputs/q1_v2_round4/manual_validation/`，当时记录的 SHA-256 为 `b66df9ed495d57fd3a9866baa6fdd95d6b793cae49c3d2b2c2badf64d4dd5d98`；副本复核哈希相同。15 行均有数值边界，但 `reviewer_id` 均为空；不能把此表推及其他 99 条，也不能推断标注者身份。原桌面文件在最终检查时被其他程序占用，未再次读取，计算基于已保存且校验过的副本。

官方 Excel 文本、原 MP4、标签和旧版 `features_q1.py` 均未由本轮改动。预训练 ASR 仅用于语音存在性、语言和官方文本所在区间的粗定位；高置信路由才把官方文本对应的真实音频局部片段送入 MFA。未匹配、无语音或语言不确定时保留官方词位及原生声学/视觉帧，不编造词时间。

## 算法和实现

- `manual_sample_audit.py`：创建/刷新 100 行人工快审 CSV，只更新自动字段。
- `asr_worker.py`：独立 Python 3.12 进程运行 Silero VAD 和 `Systran/faster-whisper-base`，保留真实 ASR 词时间、原始转录、语言概率、stdout/stderr。
- `correspondence_qa.py`：源媒体时间、VAD、ASR 局部词匹配和视觉 2 FPS 探针；缩写展开仅用于比较，官方词和顺序不变。重复匹配候选、低覆盖率、非英语、不确定状态进入复核或无效路由。`NO_FACE` 独立于文本/语音对应关系。
- `round4.py`：隔离逐条失败，输出 100 条轻量 `correspondence_audit.csv`、`quality_summary.json` 和实验配置。`round4_reclassify.py` 用保存的 ASR 证据重评分，不重跑推理；仅 `-3g5yACwYnA__9` 由 `MATCHED` 改为 `PARTIAL_MATCH`，前值留存。
- `round4_features.py`：只允许显式选样，最多 12 条，**无 `--all`**。高置信匹配在 ASR 区间两侧各留 0.30 s，真实 MFA 局部时间按 `t_global=t_crop_start+t_MFA` 映射。失败路由词时间 NaN、mask 0，不做伪对齐。默认视觉 10 FPS，可选 5 FPS。
- `fusion_alignment.py` 保持原输出向后兼容，附加 QA 状态/掩码。`round4_quality.py` 区分 QA 统计、选样完整特征统计、人工边界误差，并生成逐词/逐样本复核候选。`round4_integrity.py` 检查 NPZ 无 pickle 读取、词数/维度/掩码、原始视频 PTS 和词帧关联。
- `landmark_definition.py` 输出原生 MediaPipe 478 点定义。官方 [MediaPipe 文档](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)给出 478 个三维点；[OpenFace 输出说明](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)定义的是另一套 68 点。没有经核验的两者映射，本轮明确**没有实现 OpenFace 68**，也不把 478 点冒称 68 点。帧维度 `478×3+52=1486`，词级均值/标准差 `2972`。

真实 BERT 词向量与真实声学帧在完成最终特征文件时，从此前同一不可变 MP4/同一官方文本的成功运行结果复用：逐样本核对 MP4 SHA-256、文本、样本 ID、16 kHz、时间原点和时长；在**本轮 MFA 新词区间**上重算声学词均值、标准差和掩码。来源哈希在各样本的 `native_feature_reuse_provenance.json` 和根目录 `feature_experiment_manifest.json`；这不是本轮最终尝试重新运行 BERT/librosa 的声称。视觉为本轮真实 MediaPipe 10 FPS 提取。

## 实际环境与可复现命令

主环境：Windows，Conda `q1-v2`，Python 3.11.16；MFA 3.4.2、FFmpeg 8.1.2、PyAV 18.1.0、librosa 0.11.0、PyTorch 2.8.0+cpu、Transformers 5.17.0、MediaPipe 1.0.1、NumPy 2.4.6。ASR 隔离环境：Python 3.12.7、faster-whisper 1.2.1、ONNX Runtime 1.23.2、CTranslate2 4.8.2、NumPy 2.2.6，CPU/int8。[faster-whisper 官方实现](https://github.com/SYSTRAN/faster-whisper)与[模型卡](https://huggingface.co/Systran/faster-whisper-base)为模型/API 来源。MFA 声学模型 `english_us_arpa`，发音词典为第三轮已记录的 `outputs/q1_v2_round3/dictionary/english_us_arpa_plus_two.dict`；文本模型 `google-bert/bert-base-uncased`；人脸模型 `outputs/q1_v2_round2/models/face_landmarker.task`。模型哈希和 QA 阈值见 `experiment_config.json`，MFA/视觉参数见 `feature_experiment_manifest.json`。

从仓库根目录的 PowerShell 复现，路径中的中文与空格须以变量或引号传递；以 `-` 开头的 sample ID 用 `--sample-id=...`：

```powershell
$mainPy = 'D:\anaconda\envs\q1-v2\python.exe'
$asrPy = 'outputs\q1_v2_round4\asr_env_py312\Scripts\python.exe'
$dataRoot = '..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条'
$out = 'outputs\q1_v2_round4'
$face = 'outputs\q1_v2_round2\models\face_landmarker.task'
$asrModel = 'outputs\q1_v2_round4\models\faster-whisper-base'
$dict = 'outputs\q1_v2_round3\dictionary\english_us_arpa_plus_two.dict'
$env:MFA_ROOT_DIR = 'D:\q1_v2_mfa_root'
$env:PATH = 'D:\anaconda\envs\q1-v2\Library\bin;D:\anaconda\envs\q1-v2\Scripts;' + $env:PATH
& 'D:\anaconda\envs\q1-v2\Scripts\mfa.exe' version
& $mainPy -m q1_v2.manual_sample_audit --data-root $dataRoot --output-dir $out
& $mainPy -m q1_v2.round4 --data-root $dataRoot --output-dir $out --asr-model $asrModel --asr-python $asrPy --face-model $face --all-qa
& $mainPy -m q1_v2.round4_quality --data-root $dataRoot --output-dir $out
& $mainPy -m q1_v2.round4_integrity --data-root $dataRoot --output-dir $out --sample-id=-3g5yACwYnA__13 --sample-id=-3g5yACwYnA__2 --sample-id=-3g5yACwYnA__3 --sample-id=-THoVjtIkeU__2 --sample-id=-NFrJFQijFE__1
& $mainPy -m unittest discover -s q1_v2\tests -v
```

代表性完整特征运行使用 `python -m q1_v2.round4_features`，必须传 `--data-root $dataRoot --output-dir $out --sample-id=<ID> --face-model $face --mfa-dictionary $dict --mfa-executable D:\anaconda\envs\q1-v2\Scripts\mfa.exe --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work`。`--reuse-native-source <既有真实同源样本目录>` 与 `--alignment-source <已验证真实 MFA 原始输出目录>` 仅在有明确来源时使用；各样本的真实调用/来源由 `mfa_command.json`、`alignment_import_provenance.json`、`native_feature_reuse_provenance.json`、`mfa_crop_metadata.json` 追溯。初次 MFA 探针超时和中断尝试保存在 `attempts/`，成功的独立 MFA 输出保存在 `mfa_verified/`。

人工表对比命令：

```powershell
& $mainPy -m q1_v2.manual_validation `
  --manual-xlsx 'C:\Users\Lenovo\Desktop\-3g5yACwYnA__13_mannual.xlsx' `
  --old-alignment-csv 'outputs\q1_v2_smoke\samples\-3g5yACwYnA__13\word_alignment.csv' `
  --new-alignment-csv 'outputs\q1_v2_round4\samples\-3g5yACwYnA__13\word_alignment.csv' `
  --output-dir 'outputs\q1_v2_round4\manual_validation'
```

## 真实结果：自动 QA 与完整选样严格分开

100/100 条**轻量 QA** 的自动类别：`MATCHED=72`、`PARTIAL_MATCH=8`、`TEXT_AUDIO_MISMATCH=12`、`NO_SPEECH=2`、`NON_ENGLISH_SPEECH=3`、`NO_AUDIO=2`、`UNRESOLVED=1`。2 FPS 视觉探针判 `face_present=False` 12 条。上述是未经人工确认的模型判定，不能用作准确率；`manual_sample_audit_template.csv` 的手工列全部留空。异常候选 64 条见 `sample_review_candidates.csv`。

只对下列 **5 条真实视频**完成完整特征，合计 84 个原始词、68 个自动时间戳、84 个文本有效词、68 个语音有效词、65 个视觉有效词：

| sample_id | 路由 | 原词/对齐 | 文本/语音/视觉有效词 | 视觉帧 | 说明 |
|---|---|---:|---:|---:|---|
| `-3g5yACwYnA__13` | HIGH_CONFIDENCE_MATCH | 15/15 | 15/15/15 | 55 | 前有额外语音，真实局部 MFA；已与用户边界表对照 |
| `-3g5yACwYnA__2` | HIGH_CONFIDENCE_MATCH | 14/14 | 14/14/12 | 94 | `Polymer` 新区间 3.07–3.34 s；仍待人工听辨 |
| `-3g5yACwYnA__3` | HIGH_CONFIDENCE_MATCH | 29/29 | 29/29/29 | 144 | `adhesives` 新区间 2.48–3.32 s；仍待人工听辨 |
| `-THoVjtIkeU__2` | HIGH_CONFIDENCE_MATCH | 10/10 | 10/10/9 | 43 | 不同 `video_id`；1 个短词无区间内采样帧 |
| `-NFrJFQijFE__1` | INVALID_CORRESPONDENCE | 16/0 | 16/0/0 | 57 | 有音频信号、VAD 无语音；保留原生 564 声学帧/57 视觉帧和官方 16 词，不做 MFA |

这些 5 条均经 `round4_integrity.py` 验证：`allow_pickle=False` 可读，原词数=词级序列长度，维度为文本 768、声学帧 69/词 138、视觉帧 1486/词 2972，词时间在可解释媒体范围，掩码与实际值对应，视觉原帧索引/PTS 可追溯；测试报告 `integrity_report.json` 为 5/5 通过。真实对齐图和逐词图表：`visualizations/-3g5yACwYnA__13_alignment.png` 及相邻 CSV。图中 MFA 为自动估计，不是真值。

无脸代表样本 `-HwX2H8Z4hY__9` 另作真实 10 FPS 视觉探针：78 个采样帧、0 人脸；尚未对它运行完整 BERT/声学/融合。完整 100 样本 MFA/三模态特征与情感训练**未运行**。

## 用户人工边界表：仅 1 条、15 词

比较规则：每词起点/终点绝对误差；容差通过要求该词**两个边界同时**在容差内。旧整段 MFA 与新局部 MFA 对同一份用户边界表的结果：

| 指标 | 旧整段 MFA | 新局部 MFA |
|---|---:|---:|
| 起点 MAE | 277.47 ms | 70.80 ms |
| 终点 MAE | 220.47 ms | 57.13 ms |
| 平均边界误差 | 248.97 ms | 63.97 ms |
| 50 ms 双边界通过率 | 40.0% | 60.0% |
| 100 ms 双边界通过率 | 53.3% | 73.3% |
| 200 ms 双边界通过率 | 73.3% | 86.7% |

`They've`、`been` 与人工边界的最大单边误差均为 374 ms，列入 `word_review_candidates.csv`；局部匹配并未使开头完全准确。其余词的逐项误差在 `manual_validation/manual_alignment_detail.csv`。人工表尚未提供审核者 ID，本表只称“用户提供数值边界”，不写成已独立复核真值。对没有人工边界的其他样本，时间边界准确率保持 `null`。

## 已知失败、警告与未完成事项

- 初次把 ASR/VAD 与主 Conda 环境混进同一进程触发 OpenMP Error #15（`libomp.dll`/`libiomp5md.dll`）；改用独立 Python 3.12 子进程，不使用不安全的 `KMP_DUPLICATE_LIB_OK`。
- 主环境在未设置 `MFA_ROOT_DIR` 时的通用环境探针把 MFA 记为不可用；这是 QA 进程探针的**环境配置失败**，并非实际完整特征中 MFA 未运行。设置 `MFA_ROOT_DIR=D:\q1_v2_mfa_root` 并补充 Conda DLL/脚本路径后，`mfa version` 实测 `3.4.2`，4 条高置信选样都有真实 MFA 原始 JSON。解释该差异时以每条 `mfa_command.json`/`mfa_raw.json`/退出记录为准。
- `-3g5yACwYnA__13` 初次 MFA 探针 60 s 超时；超时失败的 `_MFA_FAILED.json`、stdout/stderr 与后来成功的独立重试均保留在 `attempts/`/`mfa_verified/`。探针限时提高到 180 s。另一次在 librosa.pyin 上耗时异常的重提取被停止；最终选样依赖有哈希/时轴校验的既有真实原生 BERT/声学输出，**不是模拟特征**。
- OpenFace/iBUG 68 后端未实现；只提供定义明确的 MediaPipe 原生 478。多脸时说话者身份仍不确定，不能由检测顺序推断。
- 72 条 `MATCHED` 是自动 QA 标签，尚无人审阅；ASR 语言、局部匹配阈值可能产生误判。只有 1 条有用户数值边界，且开头两词仍有明显残差。需要继续填写 100 行人工快审表、优先核验 64 个样本级候选及 5 个逐词候选，再决定是否做大规模完整提取。
- 精简提交体积与全量 50 MB 可行性未由本轮 100 条完整特征验证。第三轮的 `submission_export` 可复用，但不得把 5 条结果外推为 100 条体积保证。

本轮单元测试命令 `python -m unittest discover -s q1_v2/tests -v`：34/34 通过（合成测试）；真实选样完整性：5/5 通过。合成测试不计入真实特征性能。
