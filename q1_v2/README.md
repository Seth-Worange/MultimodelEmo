# 问题一 v2：可核验的三模态特征与显式时序对齐

本目录是独立基础系统，不调用 `features_q1.py` 的特征、视觉采样或词级对齐算法，也不改动问题二/三接口。系统只使用预训练基础工具做推理，不读取标签来提取特征，不训练或微调情感模型。

## 方法依据与已核实 API

- [MMTA 原始论文](https://doi.org/10.1007/s44443-025-00094-3) 的 Fine-grained Alignment 用 MFA 从音频和转写得到每个词的起止时间，再把词作为多模态公共粒度。本实现只借鉴该显式对齐思想，不实现其 MLF/AFN 训练网络。
- [MFA 3.4 对齐文档](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html) 核实了 `mfa align_one SOUND TEXT DICTIONARY ACOUSTIC OUTPUT --output_format json`。3.4 已将 legacy 签名标记为未来迁移对象，所以每次运行均保存 MFA 版本、完整 argv、stdout、stderr 和原始 JSON，不把 MFA 成功或覆盖率叫作准确率。
- [MFA First steps](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/) 核实了 `english_us_arpa` 声学模型与同名发音词典的下载、检查和配套使用方式。
- [PyAV Frame](https://pyav.org/docs/stable/api/frame.html)、[Time](https://pyav.org/docs/stable/api/time.html) 和 [Stream](https://pyav.org/docs/stable/api/stream.html) 文档规定显示时间为 `frame.pts * frame.time_base`，且容器不保证固定帧率。本实现从不使用 `frame_index/fps` 推造时间。
- [MediaPipe Python Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) 核实了 `RunningMode.VIDEO`、`detect_for_video(image, timestamp_ms)`、单调时间戳以及 Blendshape 开关；[模型页](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index) 说明输出 478 个点和 52 个 Blendshape。
- [Transformers tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer) 核实了 Fast Tokenizer 的预分词/词元映射接口；模型使用 [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)，不做情感微调。
- [librosa 1.0 Mel 频谱](https://librosa.org/doc/main/api/generated/librosa.feature.melspectrogram.html) 及特征 API 用于带显式窗口/步长的基础声学描述符。

版本基线（2026-09-23 核实）：Python 3.12、MFA 3.4.2、PyAV 18.1.0、MediaPipe 1.0.1、Transformers 5.17.0、librosa 1.0.0、BERT `google-bert/bert-base-uncased`、MFA `english_us_arpa` acoustic/dictionary、官方 `face_landmarker.task`。精确 Python 版本见 `requirements-q1-v2.txt`；实际运行版本还会写入 `experiment_config.json`。

## 时间轴和数学定义

令媒体时间为 `t_media`。样本时间原点 `t0` 取可观测的容器起点、音视频流起点、首个有效音视频 PTS 中的最早值：

```text
t_sample = t_media - t0
t_video(i) = PTS_i * time_base_i - t0
t_sample_from_MFA = t_MFA_WAV + t_WAV_origin_media - t0
```

若 PTS 缺失、倒序或音频重采样块无法映射到媒体时间，样本标为不可可靠对齐并停止，不用均匀词时长补齐。音频输出为 16 kHz、单声道、PCM16 WAV；零填充只用于忠实保留可观测 PTS 间隙，并写明填充样本数。

原始转写按非空白片段保留词序、大小写、标点和字符区间。另建小写规范词串用于 MFA 结果的单调编辑距离回映射。第 `i` 个词的区间为 `[s_i,e_i)`。声学或视觉帧 `k` 仅当其真实中心/显示时间满足 `s_i <= t_k < e_i` 时算作区间内观测：

```text
I_i = { k | s_i <= t_k < e_i }
mu_i,d = mean(x_k,d : k in I_i and finite(x_k,d))
sigma_i,d = std(x_k,d : k in I_i and finite(x_k,d))
```

无词时间、无区间内采样帧、有人脸采样但检测失败、多脸说话者不确定分别编码。可选邻帧只带 `estimated_mask=1` 输出，`visual_mask` 仍为 0，不冒充区间内真实采样。

## 特征定义

文本使用 BERT 最后一层，单词的所有 WordPiece 向量作算术平均。超过 512 位置时按完整词边界分窗并保留重叠上下文；极端超长单词按其全部 subword 分窗，绝不静默截断。即使 MFA 失败，文本词表示仍保留。

音频默认 25 ms Hann 窗、10 ms 步长、`center=False`，包含 64 维 Log-Mel、RMS、F0、过零率、谱质心、谱带宽，共 69 维。F0 无效为 NaN，另存 `f0_valid_mask`。词级输出为逐维均值与标准差（138 维）、逐维有效帧数、F0 有效帧数和原始帧索引。

视觉按真实 PTS 就近采样 5 FPS 或 10 FPS，并去掉重复源帧。选择 38 个关键点（精确编号在 `config.py` 和元数据中），以鼻尖点 1 为中心、点 33–263 的眼间距为尺度，输出 114 维归一化 `(x,y,z)`，再拼接官方 52 个 Blendshape，共 166 维；词级均值和标准差为 332 维。最多检测两张脸；用最大脸生成描述符但绝不声明它就是说话者，身份掩码保持 0，多脸情况单独标记。

## 安装（Windows PowerShell）

推荐全新环境，避免已有 NumPy 与旧二进制轮子 ABI 混用：

```powershell
conda create -n q1-v2 -c conda-forge python=3.12 montreal-forced-aligner=3.4.2
conda activate q1-v2
python -m pip install -r .\q1_v2\requirements-q1-v2.txt
mfa version
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model inspect acoustic english_us_arpa
mfa model inspect dictionary english_us_arpa
```

从 MediaPipe [官方模型表](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models) 下载 `FaceLandmarker` 的 Latest float16 bundle，实际官方链接为：

```text
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

把模型放到仓库外或被 `.gitignore` 忽略的目录。PyAV wheel 自带所需 FFmpeg 库；系统 `ffmpeg`/`ffprobe` 可选装用于环境审计，核心时间读取使用 PyAV 暴露的 FFmpeg PTS/time_base。下载的 BERT、MFA、MediaPipe 权重和缓存均不得提交。

## 运行

`--data-root` 指向同时包含 `label-100.xlsx` 和 37 个 `video_id` 文件夹的附件1根目录；若工作簿另放，用 `--labels` 显式指定。

```powershell
python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> --face-model <face_landmarker.task> --max-samples 1
python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> --face-model <face_landmarker.task> --max-samples 3
python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> --face-model <face_landmarker.task> --all
```

10 FPS 对照实验加 `--visual-fps 10`，其配置哈希不同，不应与 5 FPS 共用同一输出目录。成功样本经完整性检查后原子写入 `samples/<sample_id>/`；相同配置重跑会跳过完整样本。失败会保留 `_FAILED.json` 和 traceback；修复环境后使用 `--retry-failed`，旧失败尝试移入 `archived_attempts/`，不会混入新结果。

典型样本作图：

```powershell
python -m q1_v2.visualize --video <原视频.mp4> --sample-dir <输出路径>\samples\<sample_id> --output <图.png>
```

## 输出与核验

全局输出含 `input_manifest.csv`、`data_loading.log`、`experiment_config.json`、`manifest.csv`、`quality_by_sample.csv`、`quality_summary.json`、`review_candidates.csv` 和 `manual_alignment_reference_template.csv`。人工表填好后以 `--manual-reference-csv` 重跑报告；仅有人工参考时才计算边界 MAE 与 50/100/200 ms 通过率。

每个成功样本含原始输入哈希、媒体时间元数据、MFA 原始 JSON/命令/日志、逐词 CSV、文本/音频/视觉独立 NPZ 与元数据、融合 NPZ。`fused_features.npz` 同时保存帧级数组、词级数组、三模态掩码和索引追溯关系。NPZ 不使用 pickle。所有样本保持可变词长；未来组批只对当前 batch 的最大词长 Padding，特征填充后必须配合 0 mask，禁止截断原词序列。

质量报告中的自动对齐覆盖率、特征有效率、人脸检测率都不是时间对齐准确率，更不是情感识别准确率。`quality_summary.json` 还统计所有结果体积，可据此规划竞赛 50 MB 附件。

## 测试

```powershell
python -m unittest discover -s q1_v2\tests -v
```

测试只使用合成短序列，覆盖 PTS/time_base 转换、音频/视觉区间索引、短词无视觉帧、词对齐失败但保留、越界/零时长、无人脸与无采样帧区分、估计邻帧掩码、融合维度和样本 ID。合成测试结果不属于附件1实验结果。
