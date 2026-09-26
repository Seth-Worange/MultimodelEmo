"""在 aligned_50.pkl 上训练并验证缺失鲁棒模型。"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.nn import functional as F

from utils.augmentation import INTERVAL_RATIOS, drop_modalities, mask_batch
from utils.config import parse_config_args
from utils.data import load_main
from utils.selection import MAIN_VIEWS, relative_degradation, selection_score
from utils.text import (DEFAULT_BERT, DEFAULT_BERT_REVISION, encode_text, load_text_encoder,
                        load_text_tokenizer, prepare_bert_inputs)
from model import AffectiveModel
from model.fuse_net import FactorizedAffectiveModel, build_fuse_regularization
from model.complementary_net import ComplementaryAffectiveModel
from model.q3_temporal import Q3TemporalModel
from model.q3_residual import Q3ResidualModel

ARCHITECTURES = ("baseline", "fuse", "complementary", "q3_temporal", "q3_residual")


def build_model(args) -> torch.nn.Module:
    """按配置构造情感模型；fuse 为 FUSE-Net 风格的三因子分解架构。"""
    if args.architecture == "q3_residual":
        if args.source_checkpoint is None:
            raise ValueError("q3_residual requires --source-checkpoint")
        source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
        return Q3ResidualModel(source["model_config"], dropout=args.dropout,
                               heads=args.transformer_heads)
    if args.architecture == "q3_temporal":
        return Q3TemporalModel(
            text_mode=args.text_mode, audio_dynamics=args.audio_dynamics,
            regression_mode=args.regression_mode, dropout=args.dropout,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads, normalize_inputs=args.normalize_inputs,
            pack_aligned_grus=args.pack_aligned_grus)
    if args.architecture == "complementary":
        return ComplementaryAffectiveModel(
            hidden_dim=64, dropout=args.dropout, audio_dynamics=args.audio_dynamics,
            regression_mode=args.regression_mode, normalize_inputs=args.normalize_inputs,
            pack_aligned_grus=args.pack_aligned_grus, text_mode=args.text_mode)
    if args.architecture == "fuse":
        return FactorizedAffectiveModel(
            text_mode=args.text_mode, audio_dynamics=args.audio_dynamics,
            regression_mode=args.regression_mode, dropout=args.dropout, tau=args.tau,
            encoder_type=args.encoder_type, transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads, normalize_inputs=args.normalize_inputs,
            pack_aligned_grus=args.pack_aligned_grus, bert_finetune=args.bert_finetune,
            bert_model_name=args.bert_model_name,
            bert_model_revision=args.bert_model_revision,
            bert_freeze_bottom_layers=args.bert_freeze_bottom_layers,
            bert_gradient_checkpointing=args.bert_gradient_checkpointing,
            bert_max_length=args.bert_max_length, bert_input_source=args.bert_input_source,
            text_residual=args.text_residual, hierarchical_head=args.hierarchical_head,
            text_polarity_head=args.text_polarity_head)
    return AffectiveModel(fusion=args.fusion, text_mode=args.text_mode,
                          audio_dynamics=args.audio_dynamics,
                          regression_mode=args.regression_mode, dropout=args.dropout,
                          encoder_type=args.encoder_type, transformer_layers=args.transformer_layers,
                          transformer_heads=args.transformer_heads, normalize_inputs=args.normalize_inputs,
                          pack_aligned_grus=args.pack_aligned_grus, bert_finetune=args.bert_finetune,
                          bert_model_name=args.bert_model_name,
                          bert_model_revision=args.bert_model_revision,
                          bert_freeze_bottom_layers=args.bert_freeze_bottom_layers,
                          bert_gradient_checkpointing=args.bert_gradient_checkpointing,
                          bert_max_length=args.bert_max_length, bert_input_source=args.bert_input_source)


def model_config(args) -> dict:
    """保存到检查点、供 load_model 还原结构的配置。"""
    if args.architecture == "q3_residual":
        source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
        return {"architecture": "q3_residual", "base_config": source["model_config"],
                "dropout": args.dropout, "heads": args.transformer_heads}
    common = {"architecture": args.architecture, "text_mode": args.text_mode,
              "audio_dynamics": args.audio_dynamics, "regression_mode": args.regression_mode,
              "dropout": args.dropout, "encoder_type": args.encoder_type,
              "transformer_layers": args.transformer_layers,
              "transformer_heads": args.transformer_heads,
              "normalize_inputs": args.normalize_inputs,
              "pack_aligned_grus": args.pack_aligned_grus,
              "bert_finetune": args.bert_finetune,
              "bert_model_name": args.bert_model_name,
              "bert_model_revision": args.bert_model_revision,
              "bert_freeze_bottom_layers": args.bert_freeze_bottom_layers,
              "bert_gradient_checkpointing": args.bert_gradient_checkpointing,
              "bert_max_length": args.bert_max_length, "bert_input_source": args.bert_input_source}
    if args.architecture == "fuse":
        return {**common, "tau": args.tau, "text_residual": args.text_residual,
                "hierarchical_head": args.hierarchical_head,
                "text_polarity_head": args.text_polarity_head}
    if args.architecture == "complementary":
        return {key: common[key] for key in (
            "text_mode", "audio_dynamics", "regression_mode", "dropout",
            "normalize_inputs", "pack_aligned_grus")} | {"architecture": "complementary"}
    if args.architecture == "q3_temporal":
        return {key: common[key] for key in (
            "text_mode", "audio_dynamics", "regression_mode", "dropout",
            "transformer_layers", "transformer_heads", "normalize_inputs",
            "pack_aligned_grus")} | {"architecture": "q3_temporal"}
    return {**common, "fusion": args.fusion}


REG_FIELDS = ("contrast", "info", "dual", "mrc", "kl", "cross")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def as_tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """数值字段转张量；样本标识等非数值字段原样保留。"""
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
            out[key] = torch.from_numpy(np.ascontiguousarray(value))
        else:
            out[key] = value
    return out


def make_batch(data: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in data.items():
        out[key] = value.index_select(0, indices) if isinstance(value, torch.Tensor) \
            else [value[i] for i in indices.tolist()]
    return out


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
           for key, value in batch.items()}
    if "bert_attention_mask" in out:
        active = torch.where(out["bert_attention_mask"].bool(),
                             torch.arange(out["bert_attention_mask"].shape[1], device=device), -1)
        width = int(active.max().item()) + 1
        for key in ("bert_input_ids", "bert_attention_mask", "bert_token_type_ids"):
            if key in out:
                out[key] = out[key][:, :width]
    return out


def model_inputs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    inputs = (batch["tokens"][:, 0], batch["lengths"], batch["audio"], batch["vision"],
              batch["text_mask"], batch["audio_mask"], batch["vision_mask"], batch.get("teacher"))
    if "bert_input_ids" in batch:
        inputs += (batch["bert_input_ids"], batch["bert_attention_mask"],
                   batch.get("bert_token_type_ids"))
    return inputs


def attach_full_text_inputs(data: dict, tokenizer, max_length: int, source: str = "raw_text") -> dict:
    """把完整转写的 BERT 输入加入切分数据。"""
    prepared = prepare_bert_inputs(data, tokenizer, max_length, source)
    stats = prepared.pop("bert_text_stats")
    data.update({key: torch.from_numpy(value) for key, value in prepared.items()})
    return stats


def checkpoint_state_dict(model: torch.nn.Module, compact_bert: bool) -> dict[str, torch.Tensor]:
    """只保存可训练BERT层，并用半精度减小检查点。"""
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {}
    prefix = "bert_text_encoder.backbone."
    for name, value in model.state_dict().items():
        if compact_bert and name.startswith(prefix):
            if name not in trainable:
                continue
            value = value.to(dtype=torch.float16)
        state[name] = value.detach().cpu().clone()
    return state


def initialize_from_frozen_checkpoint(model: torch.nn.Module, path: Path,
                                      expected_config: dict) -> dict:
    """加载同结构的冻结BERT融合模型，仅让新BERT模块保留公开预训练权重。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    previous = checkpoint["model_config"]
    current = {"embedding_dim": 96, "hidden_dim": 64, **expected_config}
    allowed_changes = {"bert_finetune", "bert_freeze_bottom_layers",
                       "bert_gradient_checkpointing", "bert_max_length", "bert_input_source"}
    legacy_defaults = {"text_residual": False, "hierarchical_head": False,
                       "text_polarity_head": False}
    for key in current.keys() | previous.keys():
        if key not in allowed_changes and current.get(key, legacy_defaults.get(key)) != previous.get(
                key, legacy_defaults.get(key)):
            raise ValueError(f"Warm-start checkpoint architecture mismatch: {key}")
    if previous.get("bert_finetune", False) or not current["bert_finetune"]:
        raise ValueError("Warm-start requires frozen-BERT source and trainable-BERT target")
    if current["bert_input_source"] != "text_bert":
        raise ValueError("Warm-start requires text_bert to match the cached alignment")
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {name for name in model.state_dict()
                        if name.startswith("bert_text_encoder.") or name.startswith("long_text_")}
    if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
        raise RuntimeError("Warm-start checkpoint state does not match the target model")
    return {"path": str(path), "epoch": checkpoint.get("epoch")}


def supervised_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                    class_weights: torch.Tensor | None = None,
                    regression_loss: str = "smooth_l1",
                    magnitude_weight: float = 0.0) -> torch.Tensor:
    # L1直接优化绝对误差；默认保留历史训练目标。
    if regression_loss not in {"smooth_l1", "l1"}:
        raise ValueError(f"Unknown regression loss: {regression_loss}")
    regression = F.l1_loss if regression_loss == "l1" else F.smooth_l1_loss
    loss = F.cross_entropy(output["logits"], batch["classes"], weight=class_weights) + regression(
        output["sentiment"], batch["sentiment"]
    )
    if magnitude_weight:
        nonzero = batch["classes"] != 1
        if nonzero.any():
            loss = loss + magnitude_weight * F.smooth_l1_loss(
                output["magnitude"][nonzero], batch["sentiment"][nonzero].abs())
    return loss


def distillation_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    predicted = F.normalize(output["teacher_pred"], dim=-1)
    target = F.normalize(batch["teacher"], dim=-1)
    valid = batch["text_mask"].to(predicted.dtype)
    cosine_error = 1 - (predicted * target).sum(dim=-1)
    return (cosine_error * valid).sum() / valid.sum().clamp_min(1)


def consistency_loss(clean: dict[str, torch.Tensor], masked: dict[str, torch.Tensor]) -> torch.Tensor:
    teacher_prob = torch.softmax(clean["logits"].detach() / 2, dim=-1)
    student_log_prob = torch.log_softmax(masked["logits"] / 2, dim=-1)
    classification = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * 4
    regression = F.smooth_l1_loss(masked["sentiment"], clean["sentiment"].detach())
    return classification + regression


def supervised_contrastive_loss(clean: torch.Tensor, masked: torch.Tensor,
                                labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """把同类融合表示及同一样本的两种视图作为正例。"""
    features = F.normalize(torch.cat((clean, masked), 0).float(), dim=-1)
    targets = labels.repeat(2)
    similarity = features @ features.T / temperature
    eye = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positives = (targets[:, None] == targets[None, :]) & ~eye
    log_probability = similarity - torch.logsumexp(similarity.masked_fill(eye, -1e4), dim=1,
                                                    keepdim=True)
    return -(log_probability * positives).sum(1).div(positives.sum(1).clamp_min(1)).mean()


def metrics(classes: np.ndarray, sentiment: np.ndarray, logits: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    guess = logits.argmax(axis=1)
    f1s = []
    for label in range(3):
        tp = np.sum((guess == label) & (classes == label))
        fp = np.sum((guess == label) & (classes != label))
        fn = np.sum((guess != label) & (classes == label))
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    corr = np.corrcoef(sentiment, predicted)[0, 1] if len(sentiment) > 1 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "accuracy": float(np.mean(guess == classes)),
        "macro_f1": float(np.mean(f1s)),
        "mae": float(np.mean(np.abs(sentiment - predicted))),
        "pearson": float(corr),
    }


VIEWS = ("clean", "local", "whole", "interval")
LOCAL_RATE_RANGE = (0.05, 0.45)


def build_view(cpu_batch: dict[str, torch.Tensor], view: str, seed: int,
               local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
               ratios: tuple[float, ...] = INTERVAL_RATIOS) -> dict[str, torch.Tensor]:
    """构造一个验证视图。

    - ``clean``：完整输入。
    - ``local``：语音与视觉在同词位上出现多段短游程缺失（附件3的实测形态）。
    - ``whole``：整段丢弃语音与视觉（极端压力测试）。
    - ``interval``：多尺度连续缺失区间，比例在候选集合中抽取。
    """
    if view == "clean":
        return cpu_batch
    if view == "whole":
        return mask_batch(cpu_batch, seed=seed, probability=1.0,
                          missing_modalities=("audio", "vision"))
    if view == "local":
        rate = random.Random(seed).uniform(*local_rate_range)
        return mask_batch(cpu_batch, seed=seed, probability=1.0, local_rate=rate)
    if view == "interval":
        return mask_batch(cpu_batch, seed=seed, probability=1.0, pattern="interval", ratios=ratios)
    raise ValueError(f"Unknown view: {view}")


def build_fixed_view(cpu_batch: dict, view: str, seed: int,
                     local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE) -> dict:
    """按样本标识固定缺失位置，不随批量大小、顺序或训练种子改变。"""
    if view == "clean":
        return cpu_batch
    ids = cpu_batch.get("ids")
    if ids is None or any(not str(value).strip() for value in ids):
        raise ValueError("sample_v2 requires nonempty sample IDs")
    keys = ["tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask"]
    if "bert_attention_mask" in cpu_batch:
        keys.append("bert_attention_mask")
    parts = {key: [] for key in keys}
    for row, sample_id in enumerate(ids):
        identity = json.dumps([seed, view, str(sample_id)], ensure_ascii=False).encode("utf-8")
        sample_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
        sample = make_batch(cpu_batch, torch.tensor([row]))
        masked = build_view(sample, view, sample_seed, local_rate_range)
        for key in keys:
            parts[key].append(masked[key])
    return {**cpu_batch, **{key: torch.cat(values) for key, values in parts.items()}}


def evaluation_seed(args) -> int:
    """新协议使用独立验证种子，旧协议保留历史行为。"""
    if args.evaluation_seed is not None:
        return args.evaluation_seed
    return 2026 if args.evaluation_protocol == "sample_v2" else args.seed + 9001


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    seed: int,
    text_encoder=None,
    views: tuple[str, ...] = VIEWS,
    local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
    drop: tuple[str, ...] = (),
    evaluation_protocol: str = "legacy_batch",
) -> dict[str, dict[str, float]]:
    predicted = predict_split(model, data, batch_size, device, text_encoder, views,
                              local_rate_range, seed, drop, evaluation_protocol)
    classes = data["classes"].numpy()
    sentiment = data["sentiment"].numpy()
    return {view: metrics(classes, sentiment, predicted[view]["logits"],
                          predicted[view]["sentiment"])
            for view in views}


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    text_encoder=None,
    views: tuple[str, ...] = VIEWS,
    local_rate_range: tuple[float, float] = LOCAL_RATE_RANGE,
    seed: int = 0,
    drop: tuple[str, ...] = (),
    evaluation_protocol: str = "legacy_batch",
) -> dict[str, dict[str, np.ndarray]]:
    """逐样本预测，用于错误归因、混淆矩阵与可视化。"""
    if evaluation_protocol not in ("legacy_batch", "sample_v2"):
        raise ValueError(f"Unknown evaluation protocol: {evaluation_protocol}")
    if evaluation_protocol == "sample_v2":
        ids = [str(value) for value in data.get("ids", [])]
        if len(ids) != len(data["tokens"]) or len(set(ids)) != len(ids):
            raise ValueError("sample_v2 requires unique sample IDs")
    model.eval()
    rng = random.Random(seed)
    gathered = {view: {"logits": [], "sentiment": []} for view in views}
    starts = range(0, len(data["tokens"]), batch_size)
    batch_bar = tqdm(starts, desc="验证", unit="batch", leave=False)
    for start in batch_bar:
        ix = torch.arange(start, min(start + batch_size, len(data["tokens"])))
        clean_cpu = make_batch(data, ix)
        if drop:
            clean_cpu = drop_modalities(clean_cpu, drop)
        seeds = {view: rng.randrange(2**31) for view in views if view != "clean"}
        for view in views:
            cpu_batch = (build_fixed_view(clean_cpu, view, seed, local_rate_range)
                         if evaluation_protocol == "sample_v2" else
                         build_view(clean_cpu, view, seeds.get(view, 0), local_rate_range))
            batch = move_inputs(cpu_batch, device)
            if (view != "clean" and text_encoder is not None
                    and not torch.equal(cpu_batch["text_mask"], clean_cpu["text_mask"])):
                batch["teacher"] = encode_text(batch, text_encoder)
            output = model(*model_inputs(batch))
            gathered[view]["logits"].append(output["logits"].cpu().numpy())
            gathered[view]["sentiment"].append(output["sentiment"].cpu().numpy())
            batch_bar.set_postfix(view=view)
    return {view: {name: np.concatenate(parts) for name, parts in values.items()}
            for view, values in gathered.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "outputs" / "runs" / "main")
    parser.add_argument("--device", default="auto", help="auto、cuda 或 cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--evaluation-protocol", choices=("legacy_batch", "sample_v2"), default="legacy_batch")
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--corruption-probability", type=float, default=0.75)
    parser.add_argument("--clean-supervision-weight", type=float, default=0.35)
    parser.add_argument("--selection-scope", choices=("task", "all", "clean"), default="task",
                        help="task 用 clean/local/interval 选模；all 保留旧四视图规则")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="大于0时用参数指数滑动平均权重验证和保存检查点")
    parser.add_argument("--fusion", choices=("gate", "concat"), default="gate")
    parser.add_argument("--text-mode", choices=("tokens", "bert"), default="bert",
                        help="使用可训练词嵌入或题目提供的上下文 BERT 特征")
    parser.add_argument("--bert-finetune", action=argparse.BooleanOptionalAction, default=False,
                        help="以 text_bert 或 raw_text token 微调 BERT；只用附件2训练标签更新权重")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="从同结构的冻结BERT检查点初始化融合层，再微调BERT")
    parser.add_argument("--source-checkpoint", type=Path, default=None,
                        help="从同结构的已训练模型初始化，继续用当前任务数据训练")
    parser.add_argument("--bert-model-name", default=DEFAULT_BERT)
    parser.add_argument("--bert-model-revision", default=DEFAULT_BERT_REVISION)
    parser.add_argument("--bert-freeze-bottom-layers", type=int, default=8)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bert-max-length", type=int, default=512)
    parser.add_argument("--bert-input-source", choices=("text_bert", "raw_text"), default="raw_text")
    parser.add_argument("--bert-warmup-epochs", type=int, default=0)
    parser.add_argument("--bert-update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bert-gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                        help="CUDA 下使用 FP16 混合精度")
    parser.add_argument("--audio-dynamics", action="store_true",
                        help="加入相邻有效词位的声学变化特征")
    parser.add_argument("--regression-mode", choices=("signed", "soft", "hard"), default="soft",
                        help="signed 直接回归强度；soft 将强度与类别概率关联")
    parser.add_argument("--encoder-type", choices=("bigru", "transformer"), default="bigru",
                        help="三种模态的序列特征编码器")
    parser.add_argument("--transformer-layers", type=int, default=1)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--normalize-inputs", action="store_true",
                        help="用训练集有效位置统计量标准化音频与视觉特征")
    parser.add_argument("--pack-aligned-grus", action="store_true",
                        help="按原始对齐序列长度打包音视频和融合GRU")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                        help="类别权重指数，设为 0 可关闭")
    parser.add_argument("--whole-probability", type=float, default=0.3,
                        help="auto 模式下整段丢弃音视频（而非局部/区间缺失）的概率")
    parser.add_argument("--corruption-mode", choices=("auto", "whole", "local", "interval", "mixed"),
                        default="auto",
                        help="缺失形态：auto 混合三者；whole 整段；local 零散短游程；interval 连续区间")
    parser.add_argument("--interval-ratios", type=float, nargs="+", default=list(INTERVAL_RATIOS),
                        help="连续缺失区间占有效序列长度的候选比例")
    parser.add_argument("--overlap-probability", type=float, default=0.5,
                        help="多模态缺失区间部分重叠的概率")
    parser.add_argument("--interval-modalities", nargs="+", default=None,
                        choices=["text", "audio", "vision"],
                        help="限定连续区间族作用的模态；缺省按权重抽样（含文本）")
    parser.add_argument("--local-rate-min", type=float, default=LOCAL_RATE_RANGE[0],
                        help="局部缺失率下界，默认取自附件3实测分布")
    parser.add_argument("--local-rate-max", type=float, default=LOCAL_RATE_RANGE[1],
                        help="局部缺失率上界，默认取自附件3实测分布")
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="baseline",
                        help="baseline 为门控融合；fuse 为 FUSE-Net 风格三因子分解")
    parser.add_argument("--text-residual", action=argparse.BooleanOptionalAction, default=False,
                        help="在FUSE分类头保留文本编码器的直接证据")
    parser.add_argument("--hierarchical-head", action=argparse.BooleanOptionalAction, default=False,
                        help="先预测中性，再预测非中性极性")
    parser.add_argument("--text-polarity-head", action=argparse.BooleanOptionalAction, default=False,
                        help="层级头用文本表征判断极性，缺失文本时回退到融合表征")
    parser.add_argument("--drop-modalities", nargs="+", default=None,
                        choices=["text", "audio", "vision"],
                        help="永久遮蔽指定模态，用于单模态基线")
    parser.add_argument("--regression-loss", choices=("smooth_l1", "l1"), default="smooth_l1")
    parser.add_argument("--magnitude-weight", type=float, default=0.0)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--auxiliary-scale", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1, help="对比分离的温标（仅 fuse）")
    parser.add_argument("--kl-beta", type=float, default=1e-2, help="变分信息瓶颈 KL 权重（仅 fuse）")
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--info-weight", type=float, default=0.1)
    parser.add_argument("--dual-weight", type=float, default=0.05)
    parser.add_argument("--mrc-weight", type=float, default=0.1)
    parser.add_argument("--cross-weight", type=float, default=0.1,
                        help="用可用模态重建缺失模态的交叉重建权重（仅 fuse）")
    args = parse_config_args(parser, "train")
    selection_views = {"task": MAIN_VIEWS, "all": VIEWS, "clean": ("clean",)}[args.selection_scope]
    if (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0
            or not 0 <= args.dropout < 1 or args.class_weight_power < 0
            or args.transformer_layers < 1 or args.transformer_heads < 1
            or 128 % args.transformer_heads != 0
            or not 0 <= args.whole_probability <= 1
            or not 0 <= args.clean_supervision_weight <= 1
            or not 0 <= args.ema_decay < 1
            or not 0 <= args.overlap_probability <= 1
            or any(not 0 < ratio <= 1 for ratio in args.interval_ratios)
            or not 0 <= args.local_rate_min <= args.local_rate_max <= 1
            or args.auxiliary_scale < 0 or args.magnitude_weight < 0
            or args.supcon_weight < 0 or args.supcon_temperature <= 0
            or args.bert_warmup_epochs < 0
            or args.bert_learning_rate <= 0 or args.bert_freeze_bottom_layers < 0
            or args.bert_max_length < 50 or args.gradient_accumulation_steps < 1
            or (args.bert_finetune and args.text_mode != "bert")
            or (args.init_checkpoint is not None and not args.bert_finetune)
            or (args.source_checkpoint is not None and args.init_checkpoint is not None)
            or (args.text_polarity_head and not args.hierarchical_head)
            or (args.architecture == "q3_temporal" and
                (args.text_mode != "bert" or args.bert_finetune))):
        parser.error("epochs, batch-size, patience and lr must be positive; dropout in [0, 1); "
                     "whole/overlap probability in [0, 1]; interval ratios in (0, 1]; "
                     "local rate range must satisfy 0 <= min <= max <= 1; BERT settings must be valid")

    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    print("loading aligned_50.pkl train/valid; this file is about 1 GB")
    need_teacher = not args.bert_finetune
    train_data = as_tensors(load_main(args.data_root, "train", need_teacher=need_teacher))
    valid_data = as_tensors(load_main(args.data_root, "valid", need_teacher=need_teacher))
    bert_text_stats = None
    if args.bert_finetune:
        tokenizer = (load_text_tokenizer(args.bert_model_name, args.bert_model_revision)
                     if args.bert_input_source == "raw_text" else None)
        bert_text_stats = {
            "train": attach_full_text_inputs(train_data, tokenizer, args.bert_max_length, args.bert_input_source),
            "valid": attach_full_text_inputs(valid_data, tokenizer, args.bert_max_length, args.bert_input_source),
        }
        print("BERT inputs prepared: " + json.dumps(bert_text_stats, ensure_ascii=False))
    model = build_model(args).to(device)
    if args.normalize_inputs:
        model.fit_input_stats(train_data)
        print("fitted audio/vision normalization from training split only")
    warm_start = None
    if args.init_checkpoint is not None:
        warm_start = initialize_from_frozen_checkpoint(model, args.init_checkpoint, model_config(args))
        print(f"warm-started fusion model from {warm_start['path']} (epoch={warm_start['epoch']})")
    if args.source_checkpoint is not None:
        checkpoint = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
        source_model = checkpoint.get("model_config", {})
        if (source_model.get("architecture", "baseline") != args.architecture
                and not (args.architecture == "q3_residual"
                         and source_model.get("architecture") == "fuse")):
            raise ValueError("source checkpoint architecture must match the target")
        if args.architecture == "q3_residual":
            model.base.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
        warm_start = {"path": str(args.source_checkpoint.resolve()),
                      "epoch": checkpoint.get("epoch"), "mode": "q3_clean_adaptation"}
        print(f"initialized Q3 backbone from {warm_start['path']}")
    drop = tuple(args.drop_modalities or ())
    if drop:
        print(f"permanently dropping modalities: {', '.join(drop)}")
    text_encoder = load_text_encoder(device) if args.text_mode == "bert" and need_teacher else None
    counts = np.bincount(train_data["classes"].numpy(), minlength=3)
    class_weights = torch.as_tensor((counts.max() / counts) ** args.class_weight_power,
                                    dtype=torch.float32, device=device)
    class_weights /= class_weights.mean()
    if args.bert_finetune:
        bert_parameters = [parameter for name, parameter in model.named_parameters()
                           if name.startswith("bert_text_encoder.") and parameter.requires_grad]
        task_parameters = [parameter for name, parameter in model.named_parameters()
                           if not name.startswith("bert_text_encoder.") and parameter.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": bert_parameters, "lr": args.bert_learning_rate},
            {"params": task_parameters, "lr": args.lr},
        ], weight_decay=1e-4)
        print(f"BERT trainable tensors={len(bert_parameters)}; frozen bottom layers="
              f"{args.bert_freeze_bottom_layers}; BERT lr={args.bert_learning_rate:g}; "
              f"task lr={args.lr:g}")
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=1e-4)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        print("AMP is enabled only on CUDA; continuing in full precision")
    ema_parameters = [p for p in model.parameters() if p.requires_grad] if args.ema_decay else []
    ema_values = [p.detach().clone() for p in ema_parameters]
    raw_before_ema = None
    if ema_values:
        print(f"EMA validation enabled; decay={args.ema_decay:g}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    best_score, best_epoch, stale = float("inf"), 0, 0
    started = time.time()

    reg_fields = [f"reg_{name}" for name in REG_FIELDS] if args.architecture == "fuse" else []
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score", "bert_phase", "bert_lr"] + reg_fields +
                                [f"{view}_{m}" for view in VIEWS
                                 for m in ("accuracy", "macro_f1", "mae", "pearson")] +
                                [f"{view}_delta_{metric}" for view in ("local", "interval")
                                 for metric in ("macro_f1", "mae")])
        writer.writeheader()
        for epoch in tqdm(range(1, args.epochs + 1), desc="训练轮次", unit="epoch"):
            if raw_before_ema is not None:
                with torch.no_grad():
                    for parameter, value in zip(ema_parameters, raw_before_ema):
                        parameter.copy_(value)
                raw_before_ema = None
            phase = "cached"
            if args.bert_finetune:
                adapting = args.bert_update and epoch > args.bert_warmup_epochs
                model.bert_text_encoder.set_adaptation_enabled(adapting)
                phase = "joint" if adapting else ("warmup" if args.bert_update else "frozen_control")
                tqdm.write(f"BERT phase={phase}")
            model.train()
            order = torch.randperm(len(train_data["tokens"]))
            total_loss, batches = 0.0, 0
            reg_totals = {name: 0.0 for name in REG_FIELDS}
            batch_offsets = list(range(0, len(order), args.batch_size))
            optimizer.zero_grad(set_to_none=True)
            batch_bar = tqdm(batch_offsets, desc=f"Epoch {epoch}/{args.epochs}",
                             unit="batch", leave=False)
            for batch_index, offset in enumerate(batch_bar):
                indices = order[offset:offset + args.batch_size]
                clean_cpu = make_batch(train_data, indices)
                if drop:
                    clean_cpu = drop_modalities(clean_cpu, drop)
                masked_cpu = mask_batch(
                    clean_cpu,
                    seed=args.seed + epoch * 100_003 + offset,
                    probability=args.corruption_probability,
                    pattern=args.corruption_mode,
                    ratios=tuple(args.interval_ratios),
                    interval_modalities=tuple(args.interval_modalities) if args.interval_modalities else None,
                    overlap_probability=args.overlap_probability,
                    whole_probability=args.whole_probability,
                    local_rate_range=(args.local_rate_min, args.local_rate_max),
                )
                clean = move_inputs(clean_cpu, device)
                masked = move_inputs(masked_cpu, device)
                if text_encoder is not None and not torch.equal(masked_cpu["text_mask"], clean_cpu["text_mask"]):
                    masked["teacher"] = encode_text(masked, text_encoder)
                with (torch.autocast(device_type="cuda", dtype=torch.float16)
                      if use_amp else contextlib.nullcontext()):
                    clean_out = model(*model_inputs(clean))
                    if args.architecture == "q3_temporal" and args.corruption_probability == 0:
                        masked_out = clean_out
                        loss = supervised_loss(clean_out, clean, class_weights,
                                               args.regression_loss, args.magnitude_weight)
                    else:
                        masked_out = model(*model_inputs(masked))
                        loss = (args.clean_supervision_weight * supervised_loss(
                                    clean_out, clean, class_weights, args.regression_loss,
                                                       args.magnitude_weight)
                                + (1 - args.clean_supervision_weight) * supervised_loss(
                                    masked_out, masked, class_weights, args.regression_loss,
                                                       args.magnitude_weight)
                                + 0.05 * consistency_loss(clean_out, masked_out))
                    if args.text_mode == "tokens":
                        loss += args.distill_weight * distillation_loss(clean_out, clean)
                    if args.architecture == "fuse" and args.auxiliary_scale > 0:
                        reg = build_fuse_regularization(
                            model, masked_out, clean_out, masked["classes"], class_weights,
                            tau=args.tau, kl_beta=args.kl_beta, contrast_weight=args.contrast_weight,
                            info_weight=args.info_weight, dual_weight=args.dual_weight,
                            mrc_weight=args.mrc_weight, cross_weight=args.cross_weight)
                        loss = loss + args.auxiliary_scale * reg["total"]
                        for name in REG_FIELDS:
                            reg_totals[name] += float(reg.get(name, reg["total"].new_zeros(())).detach())
                        if args.supcon_weight:
                            contrastive = supervised_contrastive_loss(
                                clean_out["pooled"], masked_out["pooled"],
                                masked["classes"], args.supcon_temperature)
                            loss = loss + args.supcon_weight * contrastive
                group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                accumulation = min(args.gradient_accumulation_steps,
                                   len(batch_offsets) - group_start)
                scaler.scale(loss / accumulation).backward()
                should_step = ((batch_index + 1) % args.gradient_accumulation_steps == 0
                               or batch_index + 1 == len(batch_offsets))
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema_values:
                        with torch.no_grad():
                            for averaged, parameter in zip(ema_values, ema_parameters):
                                averaged.mul_(args.ema_decay).add_(
                                    parameter.detach(), alpha=1 - args.ema_decay)
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.detach())
                batches += 1
                batch_bar.set_postfix(loss=f"{total_loss / batches:.4f}")

            if ema_values:
                with torch.no_grad():
                    raw_before_ema = [parameter.detach().clone() for parameter in ema_parameters]
                    for parameter, averaged in zip(ema_parameters, ema_values):
                        parameter.copy_(averaged)
            view_metrics = evaluate(
                model, valid_data, args.batch_size, device, evaluation_seed(args), text_encoder,
                evaluation_protocol=args.evaluation_protocol,
                local_rate_range=(args.local_rate_min, args.local_rate_max), drop=drop)
            score = selection_score(view_metrics, selection_views)
            row = {"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_score": score,
                   "bert_phase": phase, "bert_lr": args.bert_learning_rate if phase == "joint" else 0.0}
            reg_summary = ""
            if reg_fields:
                for name in REG_FIELDS:
                    row[f"reg_{name}"] = round(reg_totals[name] / max(1, batches), 5)
                reg_summary = " reg=[" + " ".join(f"{name}:{row[f'reg_{name}']:.3f}"
                                                  for name in REG_FIELDS) + "]"
            for view in VIEWS:
                row.update({f"{view}_{k}": v for k, v in view_metrics[view].items()})
            for view, changes in relative_degradation(view_metrics).items():
                row.update({f"{view}_{key}": value for key, value in changes.items()})
            writer.writerow(row)
            f.flush()
            tqdm.write(f"epoch={epoch:03d} loss={row['train_loss']:.4f} score={score:.4f} "
                       f"clean_f1={view_metrics['clean']['macro_f1']:.4f} "
                       f"local_f1={view_metrics['local']['macro_f1']:.4f} "
                       f"whole_f1={view_metrics['whole']['macro_f1']:.4f} "
                       f"interval_f1={view_metrics['interval']['macro_f1']:.4f} "
                       f"clean_mae={view_metrics['clean']['mae']:.4f} "
                       f"local_mae={view_metrics['local']['mae']:.4f} "
                       f"whole_mae={view_metrics['whole']['mae']:.4f}"
                       f"{reg_summary}")

            if score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                torch.save({
                    "model": checkpoint_state_dict(model, args.bert_finetune),
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_config": {"embedding_dim": 96, "hidden_dim": 64, **model_config(args)},
                    "val_views": view_metrics,
                    "val_score": score,
                    "selection_views": list(selection_views),
                    "evaluation_protocol": args.evaluation_protocol,
                    "evaluation_seed": evaluation_seed(args),
                    "bert_text_stats": bert_text_stats,
                    "compact_bert_state": args.bert_finetune,
                    "bert_phase": phase,
                    "ema_decay": args.ema_decay,
                    "warm_start": warm_start,
                }, args.output_dir / "best.pt")
            else:
                stale += 1
            if args.bert_finetune and args.bert_update and epoch <= args.bert_warmup_epochs:
                stale = 0
            if stale >= args.patience:
                tqdm.write(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
                break

    run = vars(args).copy()
    run["data_root"] = str(args.data_root) if args.data_root else "default"
    run["config"] = str(args.config) if args.config else None
    run["output_dir"] = str(args.output_dir)
    run["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint else None
    run["source_checkpoint"] = str(args.source_checkpoint) if args.source_checkpoint else None
    run.update({"device_used": str(device), "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda, "best_epoch": best_epoch,
                "selection_views": list(selection_views),
                "warm_start": warm_start,
                "bert_text_stats": bert_text_stats,
                "elapsed_seconds": round(time.time() - started, 2)})
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"saved {args.output_dir / 'best.pt'}; best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
