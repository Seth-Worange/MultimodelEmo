# Q1 v2 第三轮真实实验记录（2026-09-24）

仅处理问题一。原始 100 条样本、Excel 转写和标签未改动；`feature/q1-improvement` 上保留第一轮/第二轮输出。运行环境沿用独立 `q1-v2` Conda 环境：Windows、Python 3.11.16、MFA 3.4.2、PyAV 18.1.0、librosa 0.11.0、PyTorch 2.8.0+cpu、Transformers 5.17.0、MediaPipe 1.0.1。完整环境记录在 `outputs/q1_v2_round3/round3_experiment_config.json`，版本安装方法见 [README.md](README.md)。

## 词典依据和 MFA 修正

[MFA 官方词典文档](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html)说明，普通词未在词典中时可输出 `<unk>`，MFA 会去掉词首尾标点并以小写查词；[官方 align_one 文档](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html)核实了 `DICTIONARY_PATH`、`--output_format json` 和独立临时目录的调用。实际使用的 `D:\q1_v2_mfa_root\pretrained_models\dictionary\english_us_arpa.dict` 有 `adhesive`，没有 `polymer`、`adhesives`。这两个词在赛题文本中无须特殊规范化，原始 `<unk>` 是实际词典外词。

从 [CMU 官方发音词典](https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict)核对并向**实验副本**追加：

```text
polymer    P AA1 L AH0 M ER0
adhesives  AE0 D HH IY1 S IH0 V Z
adhesives  AH0 D HH IY1 S IH0 V Z
```

实际文件采用 MFA 要求的 tab 分隔词与发音。原词典 SHA-256 为 `e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0`；副本为 `e4d74cce7fad9bb09ed6c8837be617651b533bdfb2cc9a76f4cfed797e7a901c`。原词典和其他样本没有修改。具体条目见 `outputs/q1_v2_round3/dictionary/dictionary_change.json`。只重跑了 `-3g5yACwYnA__2`、`-3g5yACwYnA__3`。

| 样本/词 | 原 MFA `<unk>` 区间（仅诊断） | 新 MFA 自动区间 | 新词掩码 | 人工边界核验 |
|---|---:|---:|---:|---|
| `__2` / `Polymer` | 0.04–0.45 s | 1.20–1.59 s | 1 | 未执行 |
| `__3` / `adhesives` | 1.34–1.89 s | 1.17–1.75 s | 1 | 未执行 |

两样本 43 个原词均保留且新 MFA 均给出时间戳；连同未变更的 `__13`，三样本自动时间戳覆盖由 56/58 变为 58/58。**这是输出覆盖率，不是时间对齐准确率。** `Polymer` 新区间与旧 `<unk>` 位置相距较大，且 13 个原本有时间的词在重跑后至少一个边界移动超过 200 ms。完整差异在 `realignment_word_deltas.csv`，已进入人工复核。不能把旧 `<unk>` 区间当标准答案，也不能因新词典有发音就宣布新边界准确。人工参考数量为 0，边界误差和容差通过率仍为 `null`。

每个问题样本的 `outputs/q1_v2_round3/realigned/samples/<sample_id>/` 内保留 `before_mfa_raw.json`、`before_word_alignment.csv`、新 `mfa_raw.json`、新 `word_alignment.csv`、`mfa_command.json`、`mfa_stdout.log`、`mfa_stderr.log` 和 `target_word_diagnostic.json`。初次探针超时和受限环境拒绝写入 ASCII 工作目录的尝试保留在 `realigned/failed_attempts/`；最终成功两次运行的 `mfa_stderr.log` 均为空。MFA Windows 原生路径需要 ASCII 工作目录 `D:\q1_v2_mfa_work`。

## 固定对齐的 5/10 FPS 对照

两组使用相同的词级 MFA CSV、同一 MediaPipe Face Landmarker 权重（SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`）、相同的实际视频 PTS；邻近帧估计关闭。`__2`、`__3` 使用上表新对齐，`__13` 使用旧对齐。输出在互不覆盖的 `visual_5fps/`、`visual_10fps/`；每词的实际采样时间、原视频帧索引、有效帧数和失败原因见各自 `visual_word_audit.csv`。

| 样本 | 5 FPS 有效词 / 无采样帧词 / 实采帧 / 耗时 s | 10 FPS 有效词 / 无采样帧词 / 实采帧 / 耗时 s |
|---|---:|---:|
| `__13` | 14 / 1 / 28 / 12.79 | 15 / 0 / 55 / 3.06 |
| `__2` | 12 / 2 / 47 / 3.19 | 13 / 1 / 94 / 5.01 |
| `__3` | 25 / 4 / 72 / 4.00 | 28 / 1 / 144 / 6.57 |
| 合计 | 51 / 7 / 147 | 56 / 2 / 293 |

词区间内采样帧总计分别为 106 与 207。两组真实采样帧的人脸检测失败数均为 0。按词长 `[0,0.2)`、`[0.2,0.5)`、`[0.5,∞)` 秒统计：短词覆盖 5 FPS 为 15/22、10 FPS 为 20/22；中词两组均 19/19；长词两组均 17/17。耗时是一次顺序运行，首个 5 FPS 调用含模型冷启动，不作严格运行效率推断。详细数值见 `visual_5_10_comparison.csv`。MediaPipe 在终端打印 XNNPACK 默认加速与 feedback tensor 单签名警告；本轮没有把这些非致命终端输出转存成完整单独日志。

额外样本端到端运行还在终端显示 Hugging Face 未认证请求提示，以及从 `google-bert/bert-base-uncased` 加载 `BertModel` 时预训练任务头参数为 `UNEXPECTED` 的报告；模型主体成功推理。上述终端输出未保存为独立逐字日志，因此不能把本报告当作完整 stderr 转录。MFA 子进程的原始 stdout/stderr、命令和失败 traceback 则逐样本保存。

`revised_fused_5fps/` 和 `revised_fused_10fps/` 复用原始文本向量及原始带时间戳声学帧，按新 MFA 时间重新计算词级声学统计，并结合各自的真实视觉帧重新生成三模态 NPZ。每个词和帧的长度、时间范围、有效掩码、原视频帧索引、`allow_pickle=False` 读取已由 `revised_fused_integrity_audit.json` 验证：6/6 通过、0 错误。复用来源哈希写于每样本 `rebuild_provenance.json`。这些自动边界仍待人工核验。

## 逐词复核与人工标注

`outputs/q1_v2_round3/word_review_candidates.csv` 基于**修订后的 5 FPS 三模态输出**，含当前 `start_s/end_s`、当前音频/视觉掩码、历史 `<unk>`、旧新边界大幅移动以及当前无帧等问题；21 条候选。旧基线的 2 个 `<unk>`、6 个无采样帧词及长时词原貌保存在 `baseline_review/word_review_candidates.csv`。历史问题不会因为新自动对齐给出时间戳而从复核队列消失。

`word_review_reference_template.csv` 已预填样本 ID、原词位、原词与当前自动边界；人工填写 `reference_start_s`、`reference_end_s`、`reviewer_id`，并将 `boundary_status` 从 `unreviewed` 改为 `reviewed` 或 `cannot_determine`，可附 `review_note`。未听辨或无法确定边界的记录不参与人工误差计算。

## 精简导出体积

`submission_export.py` 每样本仅拷贝一份可用 `allow_pickle=False` 读取的 `fused_features.npz`、词帧映射元数据、词级 MFA CSV 与视频哈希/来源记录。时间戳、原词位、三模态词级和帧级特征、掩码及原视频帧索引均在。完整 WAV、MFA 原始日志、模型权重、缓存、各模态重复 NPZ 继续保留在研究目录。

| 样本 | 10 FPS 相关完整研究组件 B | 精简导出 B | 精简比例 |
|---|---:|---:|---:|
| `__13` | 1,056,226 | 249,121 | 23.59% |
| `__2` | 1,601,422 | 364,579 | 22.77% |
| `__3` | 2,581,408 | 592,414 | 22.95% |

研究组件按该样本的旧完整研究目录、新 MFA 目录（如有）、10 FPS 视觉目录及新融合目录的实际字节数相加；明细见 `submission_export_10fps/composite_size_report.csv`。另有旧 5 FPS 完整研究输出与独立精简导出的逐条对比 `submission_export/size_report.csv`。这里只验证了 3 条，100 条及赛题 50 MB 限制**未验证**。

## 额外 `video_id` 验证与失败诊断

- `-THoVjtIkeU__2`（新 `video_id`）默认 MFA 端到端成功：10/10 词有自动区间、文本/语音 10/10、视觉 8/10；结果在 `outputs/q1_v2_round3_extra/`。
- `-NFrJFQijFE__1`（另一个新 `video_id`）默认 MFA beam=10 失败，实际 stderr 为 `Could not align the file with the current beam size (10...)`；原样保留 `_FAILED.json`、`mfa_stderr.log`、命令及原始视频探针于 `outputs/q1_v2_round3_extra/samples/-NFrJFQijFE__1/`。该失败样本的 16 个原词进入逐词复核，未执行的文本/语音/视觉阶段在样本级质量报告中为 `null/not_evaluated`。
- 根据 [MFA 官方故障排查建议](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/troubleshooting.html)，仅对该失败样本做 beam=100 独立诊断；MFA-only 16/16 自动区间，随后在 `outputs/q1_v2_round3_extra_beam100_full/` 完成三模态处理。该样本 16/16 文本和语音有效，但 29/29 采样帧均未检出人脸，因此视觉 0/16；6 个词无采样帧，另 10 个词有采样帧但无人脸。beam=100 属于更宽松的自动对齐，不能据此推断边界准确。

额外样本并未并入 5/10 FPS 主对照；未启动 100 条全量实验。所有人工边界误差均未执行。

## 实际运行命令（PowerShell，在仓库根目录）

在独立环境激活后执行；受限终端未激活 Conda 时，以下命令实际以 `D:\anaconda\envs\q1-v2\python.exe` 代替 `python`，并把该环境、`Scripts`、`Library\bin` 加入进程 PATH。MFA 的本机 `mfa.exe` 路径由每样本 `mfa_command.json` 记录。

```powershell
python -m unittest discover -s q1_v2\tests -v
python -m q1_v2.round3 --phase dictionary --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3 --base-dictionary D:\q1_v2_mfa_root\pretrained_models\dictionary\english_us_arpa.dict
python -m q1_v2.round3 --phase realign --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3 --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work
python -m q1_v2.round3 --phase visual --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3 --face-model outputs\q1_v2_round2\models\face_landmarker.task
python -m q1_v2.round3 --phase fusion --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3
python -m q1_v2.round3 --phase review --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3
python -m q1_v2.round3 --phase verify --baseline-root outputs\q1_v2_smoke --output-dir outputs\q1_v2_round3
python -m q1_v2.submission_export --source-root outputs\q1_v2_round3\revised_fused_10fps --output-dir outputs\q1_v2_round3\submission_export_10fps --sample-id=-3g5yACwYnA__13 --sample-id=-3g5yACwYnA__2 --sample-id=-3g5yACwYnA__3
python -m q1_v2.run --data-root "..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条" --output-dir outputs\q1_v2_round3_extra --sample-id=-THoVjtIkeU__2 --sample-id=-NFrJFQijFE__1 --face-model outputs\q1_v2_round2\models\face_landmarker.task --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work --device cpu --skip-decode-probe
python -m q1_v2.mfa_smoke --data-root "..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条" --sample-id=-NFrJFQijFE__1 --output-dir outputs\q1_v2_round3_extra_beam100_mfa --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work --beam 100
python -m q1_v2.run --data-root "..\data\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条" --output-dir outputs\q1_v2_round3_extra_beam100_full --sample-id=-NFrJFQijFE__1 --face-model outputs\q1_v2_round2\models\face_landmarker.task --mfa-root-dir D:\q1_v2_mfa_root --mfa-work-dir D:\q1_v2_mfa_work --mfa-beam 100 --device cpu --skip-decode-probe
```

最后一次单元测试为 24/24 通过（合成接口测试）；附件 1 真实输出结果以本报告及输出目录的 CSV、NPZ、原始 MFA JSON/日志为准。未创建新的情感模型或引入外部情感数据。
