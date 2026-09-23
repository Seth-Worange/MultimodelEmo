# 问题一第二轮真实运行报告（2026-09-24）

本报告只记录真实执行结果。合成单元测试不计入附件 1 性能。完整大文件位于被 Git 忽略的 `outputs/q1_v2_smoke/` 和 `outputs/q1_v2_round2/`。

## 结论

- 新建独立 Conda 环境 `q1-v2`，未修改 base/pytorch 环境。
- MFA 3.4.2、`english_us_arpa` 声学模型和词典均已真实验证；可执行路径为 `D:\anaconda\envs\q1-v2\Scripts\mfa.EXE`。
- 指定样本 `-3g5yACwYnA__13` 的 MFA-only 结果为 15/15 词；`They've` 保持一个原始词并映射成功。
- 3 条真实样本均完成 MFA、BERT、声学、MediaPipe 和融合输出；没有运行 100 条全量处理。
- 17/17 合成单元测试通过；3 个 `fused_features.npz` 完整性验证均为 `(True, [])`。
- 没有人工参考边界，因此时间对齐 MAE 与 50/100/200 ms 通过率均保持 `null`。

## 三条真实样本

| sample_id | 原词/有时间词 | 文本有效 | 语音有效 | 视觉有效 | 媒体/解码音频时长(s) | 声学帧/F0有效帧 | 视觉帧/检测成功帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-3g5yACwYnA__13` | 15/15 | 15 | 15 | 14 | 5.513997/5.399125 | 538/213 | 28/28 |
| `-3g5yACwYnA__3` | 29/28 | 29 | 28 | 25 | 14.388997/14.327875 | 1431/692 | 72/72 |
| `-3g5yACwYnA__2` | 14/13 | 14 | 13 | 11 | 9.394010/9.231313 | 921/456 | 47/47 |

三条合计 58 个原始词，自动时间戳覆盖 56/58（96.55%，不是准确率）；文本、语音、视觉词级有效率分别为 100%、96.55%、86.21%。视觉共 147 帧，人脸检测失败 0 帧；身份掩码仍全部为 0，因为未做说话者确认。所有时间戳越界、非单调、异常重叠、零时长计数均为 0。

未获得时间戳但仍完整保留的词：

- `-3g5yACwYnA__3` 的 word 5 `adhesives`；
- `-3g5yACwYnA__2` 的 word 1 `Polymer`。

两者均为 `alignment_mask=0`、时间 NaN、`failure_reason=not_returned_or_not_matched_by_mfa`，文本向量仍有效，没有删词或均匀伪对齐。

## 首条样本细节

- MFA 原始 JSON 音频范围为 `[0, 5.399125]` 秒，词区间范围 `[0, 4.87]` 秒。
- `They've` 的 BERT WordPiece 为 `they`、`'`、`ve`，三个向量均值形成原始 word 0 的 768 维表示。
- 文本、语音、视觉词特征形状为 `(15,768)`、`(15,138)`、`(15,332)`。
- 帧级声学形状 `(538,69)`；视觉形状 `(28,166)`，原始视频帧索引范围 0—161。
- 词 13 `the` 因 5 FPS 下无真实采样帧落入短区间而视觉无效；这不是人脸检测失败，也没有复制邻近帧冒充区间内观测。
- 视频流声明时长 5.5139973958 秒，音频流声明时长 5.3990022676 秒，解码音频为 5.399125 秒；起点均为 0。差异属于原媒体流正常结束时间差异，没有补音频到视频长度。

## 实际错误与处理

1. 第一轮旧失败：`MFAUnavailableError: mfa executable not found on PATH`。原 `_FAILED.json` 已由 `--retry-failed` 移入 `outputs/q1_v2_smoke/archived_attempts/`。
2. MFA 模型最初放在中文路径时，OpenFST 写 `L.fst` 报 `_pywrapfst.FstIOError`；临时目录改为 ASCII 后，Kaldi 读取中文模型根目录又报 `_kalpy.util.KaldiFatalError`。最终固定使用 `D:\q1_v2_mfa_root` 与 `D:\q1_v2_mfa_work`。
3. MFA 3.4.2 的 `model inspect dictionary` 在 Windows 报 `AttributeError: 'WindowsPath' object has no attribute 'load_dictionary_paths'`；改用官方 `model list dictionary` 精确匹配，并以真实 `align_one` 为最终验证。
4. MediaPipe 原生层不能打开中文绝对模型路径；改用官方 `model_asset_buffer` 后 VIDEO 模式真实初始化成功。
5. Transformers 5.17 移除了本项目旧调用的 `build_inputs_with_special_tokens`。首次重试在 `text` 阶段失败，归档 `_FAILED.json` 保留；随后为 BERT 显式、可审计地构造 `[CLS] tokens [SEP]`，未知模型类型不猜测。

观察到但不影响结果的警告：Hugging Face 未登录限速提示、Windows 缓存不支持 symlink 的空间退化提示、从含 MLM/NSP 头的 BERT checkpoint 加载 `BertModel` 时报告预期的 unused head keys、MediaPipe XNNPACK 与 feedback tensor 提示。`pip check` 最终为 `No broken requirements found.`

## 结果位置

- 环境检查：`outputs/q1_v2_round2/environment_check.json`
- MFA-only 原始输出：`outputs/q1_v2_round2/mfa_smoke/samples/-3g5yACwYnA__13/`
- 三条完整样本：`outputs/q1_v2_smoke/samples/<sample_id>/`
- 质量汇总：`outputs/q1_v2_smoke/quality_summary.json`
- 逐样本质量：`outputs/q1_v2_smoke/quality_by_sample.csv`
- 首条真实图：`outputs/q1_v2_smoke/visualizations/-3g5yACwYnA__13_alignment.png`
- 首条详细复核表：`outputs/q1_v2_smoke/visualizations/-3g5yACwYnA__13_alignment_alignment_detail.csv`

输出树约 446.7 MB，其中 Hugging Face 缓存约 441.2 MB；三个样本目录合计约 3.69 MB。准备竞赛 50 MB 附件时必须排除 `cache/`、模型权重和临时 MFA 原生工作目录。
