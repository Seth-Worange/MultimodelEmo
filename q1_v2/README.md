# 问题一 v2：可核验的三模态特征与显式时间对齐

`q1_v2` 是独立基础系统，不复用 `features_q1.py` 的特征、视觉采样或词级对齐算法，也不修改问题二、问题三接口。标签只用于样本对应核查，不进入特征提取。系统仅使用预训练基础模型做推理，不训练或微调情感模型。

## 已核实的方法与 API

- [MMTA 原始论文](https://doi.org/10.1007/s44443-025-00094-3)：借鉴 Fine-grained Alignment 中用强制对齐建立词—音频时间区间的思想，不实现其完整训练网络。
- [MFA 3.x alignment](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html)：实际调用 `mfa align_one SOUND TEXT DICTIONARY ACOUSTIC OUTPUT --output_format json`。MFA 3.4 已提示该 legacy 签名未来会迁移，因此每次保存版本、完整 argv、stdout、stderr 和原始 JSON。
- [MFA first steps](https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/)：使用官方 `english_us_arpa` 声学模型和同名词典；用 `mfa model list acoustic/dictionary` 核验精确模型名。
- [PyAV Frame](https://pyav.org/docs/stable/api/frame.html)、[Time](https://pyav.org/docs/stable/api/time.html)：显示时间取 `frame.pts * frame.time_base`，不使用 `frame_index/fps` 假设恒定帧率。
- [MediaPipe Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)：使用 `RunningMode.VIDEO` 和单调递增的 `detect_for_video(image, timestamp_ms)`；[官方模型页](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index)说明模型输出 478 个关键点和 52 个 Blendshape。
- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)：只取预训练 BERT 最后一层上下文表示，不做情感微调。
- [librosa pYIN](https://librosa.org/doc/main/generated/librosa.pyin.html) 与 [Mel 频谱](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)：用于基础声学描述符和显式有效性掩码。

## 实际验证环境（2026-09-24）

Windows 11、Conda 24.9.2、Python 3.11.16。主要版本如下：

| 组件 | 实际版本 |
|---|---:|
| Montreal Forced Aligner | 3.4.2 |
| Kaldi / Kalpy / OpenFST | 5.5.1172 / 0.10.5 / 1.8.4 |
| FFmpeg | 8.1.2 |
| NumPy / SciPy | 2.4.6 / 1.17.1 |
| PyAV | 18.1.0 |
| librosa / SoundFile | 0.11.0 / 0.14.0 |
| PyTorch | 2.8.0+cpu |
| Transformers / Tokenizers | 5.17.0 / 0.23.2 |
| MediaPipe / OpenCV contrib | 1.0.1 / 5.0.0.93 |
| Matplotlib / seaborn | 3.11.2 / 0.13.2 |
| openpyxl | 3.1.5 |

模型与来源：

- MFA：`english_us_arpa` acoustic + dictionary，MFA 官方模型仓库。
- BERT：`google-bert/bert-base-uncased`，本次缓存的配置 revision 为 `86b5e0934494bd15c9632b12f734a8a67f723594`，词级维度 768。
- Face Landmarker：官方 float16 latest bundle；本次文件 SHA-256 为 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`。

## Windows PowerShell 安装

不要混装到既有 pytorch/base 环境。以下是本轮真实采用的安装方式：

```powershell
conda create -n q1-v2 --override-channels -c conda-forge python=3.11 montreal-forced-aligner=3.4.2 -y
conda activate q1-v2

# 明确安装 CPU wheel，CUDA 不是端到端验证的前提
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
python -m pip install openpyxl==3.1.5 av==18.1.0 transformers==5.17.0 mediapipe==1.0.1 seaborn==0.13.2
python -m pip check
```

MFA/Kaldi/OpenFST 的 Windows 原生层不能可靠读写含中文字符的路径。因此模型根目录和原生工作目录必须为纯 ASCII；路径可以包含空格，代码以 argv 列表调用，不拼接 shell 字符串。

```powershell
$env:MFA_ROOT_DIR = 'D:\q1_v2_mfa_root'
mfa version
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model list acoustic
mfa model list dictionary
```

MFA 3.4.2 在 Windows 上执行 `mfa model inspect dictionary english_us_arpa` 会出现内部 `WindowsPath` 异常；系统用官方 `model list` 的精确名称匹配进行环境探针，最终以真实 `align_one` 成功生成 JSON 作为有效性判据。

下载 MediaPipe 官方模型（模型、缓存和权重不得提交 Git）：

```powershell
New-Item -ItemType Directory -Force .\outputs\q1_v2_models
curl.exe -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -o ".\outputs\q1_v2_models\face_landmarker.task"
```

MediaPipe 1.0.1 的 Windows 原生文件加载器也可能无法打开含中文字符的绝对模型路径。本实现读取模型字节并使用官方 `BaseOptions(model_asset_buffer=...)`，已在中文仓库路径下真实初始化 VIDEO 模式。

## 时间轴与特征定义

令媒体时间为 `t_media`，公共样本时间原点 `t0` 取容器/流/首个可信 PTS 中的最早值：

```text
t_sample = t_media - t0
t_video(i) = PTS_i * time_base_i - t0
t_sample_from_MFA = t_MFA_WAV + t_WAV_origin_media - t0
```

原始转写按非空白片段保留词序、大小写、标点和字符区间。MFA 结果只用于回映射；没有时间戳的词保留在原位置，`alignment_mask=0` 且时间为 NaN。

词区间采用半开区间 `[s_i,e_i)`。音频或视频帧仅在真实中心/显示时间满足 `s_i <= t_k < e_i` 时算作区间内观测：

```text
I_i = { k | s_i <= t_k < e_i }
mu_i,d = mean(x_k,d : k in I_i and finite(x_k,d))
sigma_i,d = std(x_k,d : k in I_i and finite(x_k,d))
```

- 文本：一个原始词的全部 WordPiece 最后一层向量取均值；长文本按完整词边界分窗，不静默截断。Transformers 5.x 对 BERT 显式构造 `[CLS] tokens [SEP]`，并检查模型类型和特殊 token 数量。
- 声学：16 kHz 单声道；Log-Mel/RMS/ZCR/谱质心/谱带宽使用 25 ms 窗、10 ms 步长；pYIN 使用独立 64 ms 窗、10 ms 步长，再按真实中心时间最近映射到基础帧轴。F0 缺失为 NaN，并保留 `f0_valid_mask`，不把无声当作有效 0 Hz。帧维 69，词级均值+标准差维 138。
- 视觉：按真实 PTS 在 5 FPS 或 10 FPS 时间网格上选最近已解码帧并去重；38 个指定关键点以鼻尖点 1 为中心、点 33—263 的眼间距为尺度，得到 114 维，再拼接 52 个 Blendshape，帧维 166、词级均值+标准差维 332。
- 多人脸时用最大脸生成描述符，但不宣称其为说话者；所有 `speaker_identity_mask` 默认保持 0。无采样帧、检测失败、多人脸身份不确定、词时间缺失分别记录。

## 运行

`--data-root` 指向包含 `label-100.xlsx` 与 37 个 `video_id` 文件夹的附件 1 根目录。

```powershell
python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> `
  --face-model <face_landmarker.task> --mfa-root-dir D:\q1_v2_mfa_root `
  --mfa-work-dir D:\q1_v2_mfa_work --device cpu --max-samples 1

python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> `
  --face-model <face_landmarker.task> --mfa-root-dir D:\q1_v2_mfa_root `
  --mfa-work-dir D:\q1_v2_mfa_work --device cpu --max-samples 3

python -m q1_v2.run --data-root <附件1路径> --output-dir <输出路径> `
  --face-model <face_landmarker.task> --mfa-root-dir D:\q1_v2_mfa_root `
  --mfa-work-dir D:\q1_v2_mfa_work --device cpu --all
```

环境修复后重试失败样本加 `--retry-failed`；旧尝试移动到 `archived_attempts/`。完整且配置哈希相同的输出会跳过。10 FPS 对照实验加 `--visual-fps 10`，应使用不同输出目录。

单独验证 MFA（样本 ID 以连字符开头时必须使用等号形式）：

```powershell
python -m q1_v2.mfa_smoke --data-root <附件1路径> `
  --sample-id=-3g5yACwYnA__13 --output-dir <MFA烟测输出> `
  --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work
```

生成真实对齐图和逐词复核表：

```powershell
python -m q1_v2.visualize --video <原视频.mp4> `
  --sample-dir <输出路径>\samples\<sample_id> --output <图.png>
```

## 输出和质量语义

全局输出包括 `input_manifest.csv`、`data_loading.log`、`experiment_config.json`、`manifest.csv`、`quality_by_sample.csv`、`quality_summary.json`、`review_candidates.csv` 和人工参考模板。每个成功样本保留媒体 PTS、MFA 原始输出/命令/日志、逐词 CSV、各模态独立 NPZ/元数据及 `fused_features.npz`。

所有 NPZ 只保存非 object 数组，可用 `allow_pickle=False` 重读。可变词长不截断；未来组批仅对当前 batch 最大长度 Padding，并始终配套 mask。

阶段未执行时状态为 `not_evaluated`、对应比率为 `null`；阶段真实执行但无有效结果时才可为 0。文件覆盖率、正式处理成功率、自动时间戳覆盖率、模态有效率分别统计。MFA 成功、自动覆盖率、非零特征率都不是时间对齐准确率或情感识别准确率；没有人工参考时边界 MAE 与容差通过率保持 `null`。

## 测试

```powershell
python -m unittest discover -s q1_v2\tests -v
```

合成测试与附件 1 真实实验严格分开。测试覆盖 PTS/time_base、音视频帧区间关联、短词无视觉帧、无人脸与无采样帧区分、失败保词、异常时间戳、Padding/mask、Transformers 5 BERT 特殊 token 兼容、独立 F0 时间轴和 NPZ 维度。
