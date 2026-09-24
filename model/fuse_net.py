"""FUSE-Net 风格的三因子分解 + 变分重建 + 多因子动态融合。

参考 Yang & Li, "Factorize, Reconstruct, Enhance: A Unified Framework for
Multimodal Sentiment Analysis" (CVPR 2026) 的三个模块：

- HMF：把每个模态的上下文表示分解为共享 ``h`` / 私有 ``s`` / 噪声 ``n`` 三个子空间，
  用对比分离、信息增益与对偶一致性三个约束学习分解。
- MRC：对三支拼接做变分信息瓶颈重建，保证隐表示"充分且最小"。
- MDF：把融合权重分解为样本调制 α、因子类型系数 β、分支注意力 γ 三个尺度相乘，
  并对噪声分支加门控抑制。

按本题要求补上缺失感知：所有模态级运算都受可用性掩码约束（不可用模态不参与
融合与对比），重建只落在可用词位；另加"用可用模态重建缺失模态"的交叉重建项，
使共享子空间必须真正跨模态，从而支撑问题二的局部缺失场景。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.data import AUDIO_DIM, SEQ_LEN, TEXT_DIM, VISION_DIM, VOCAB_SIZE
from .transformer_encoder import TemporalTransformerEncoder
from .input_processing import AlignedInputProcessing
from .bert_text import FineTunedBertTextEncoder

MODALITIES = ("text", "audio", "vision")
FACTORS = ("shared", "private", "noise")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按词位掩码求均值；全掩码的样本返回 0 而不是 NaN。"""
    weight = mask.unsqueeze(-1).to(values.dtype)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class FactorizedAffectiveModel(nn.Module, AlignedInputProcessing):
    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64, text_mode: str = "bert",
                 audio_dynamics: bool = True, regression_mode: str = "soft", dropout: float = 0.2,
                 tau: float = 0.1, use_availability_embedding: bool = True,
                 encoder_type: str = "bigru", transformer_layers: int = 1,
                 transformer_heads: int = 4, normalize_inputs: bool = False,
                 pack_aligned_grus: bool = False, bert_finetune: bool = False,
                 bert_model_name: str = "google-bert/bert-base-uncased",
                 bert_model_revision: str = "86b5e0934494bd15c9632b12f734a8a67f723594",
                 bert_freeze_bottom_layers: int = 8,
                 bert_gradient_checkpointing: bool = True,
                 bert_max_length: int = 512, bert_input_source: str = "raw_text"):
        super().__init__()
        if text_mode not in {"tokens", "bert"}:
            raise ValueError(f"Unknown text mode: {text_mode}")
        if regression_mode not in {"hard", "soft", "signed"}:
            raise ValueError(f"Unknown regression mode: {regression_mode}")
        if encoder_type not in {"bigru", "transformer"}:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        self.architecture = "fuse"
        self.text_mode = text_mode
        self.audio_dynamics = audio_dynamics
        self.regression_mode = regression_mode
        self.encoder_type = encoder_type
        self.bert_finetune = bool(bert_finetune)
        self.bert_model_name = bert_model_name
        self.bert_model_revision = bert_model_revision
        self.bert_freeze_bottom_layers = int(bert_freeze_bottom_layers)
        self.bert_max_length = int(bert_max_length)
        self.bert_input_source = bert_input_source
        if self.bert_finetune and text_mode != "bert":
            raise ValueError("bert_finetune requires text_mode='bert'")
        self.init_input_processing(normalize_inputs, pack_aligned_grus, AUDIO_DIM, VISION_DIM)
        self.tau = float(tau)
        width = hidden_dim * 2

        if text_mode == "tokens":
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
            text_input_dim = embedding_dim
            self.text_teacher = nn.Linear(width, TEXT_DIM)
        else:
            self.text_input = nn.Sequential(nn.Linear(TEXT_DIM, width), nn.LayerNorm(width), nn.GELU())
            text_input_dim = width
            if self.bert_finetune:
                self.bert_text_encoder = FineTunedBertTextEncoder(
                    bert_model_name, bert_model_revision, bert_freeze_bottom_layers,
                    bert_gradient_checkpointing)
                if self.bert_max_length > self.bert_text_encoder.backbone.config.max_position_embeddings:
                    raise ValueError("bert_max_length exceeds the pretrained model position limit")
                self.long_text_projection = nn.Linear(TEXT_DIM, width)
                self.long_text_scale = nn.Parameter(torch.tensor(0.0))
        audio_dim = AUDIO_DIM * (2 if audio_dynamics else 1)
        self.audio_input = nn.Sequential(nn.Linear(audio_dim, width), nn.LayerNorm(width), nn.GELU())
        self.vision_input = nn.Sequential(nn.Linear(VISION_DIM, width), nn.LayerNorm(width), nn.GELU())
        if encoder_type == "bigru":
            self.text_gru = nn.GRU(text_input_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.audio_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
            self.vision_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        else:
            self.text_transformer = TemporalTransformerEncoder(
                text_input_dim, hidden_dim, transformer_heads, transformer_layers, dropout)
            self.audio_transformer = TemporalTransformerEncoder(
                width, hidden_dim, transformer_heads, transformer_layers, dropout)
            self.vision_transformer = TemporalTransformerEncoder(
                width, hidden_dim, transformer_heads, transformer_layers, dropout)

        # HMF：每个模态三个子空间投影。
        self.factor_proj = nn.ModuleDict({
            name: nn.ModuleDict({
                factor: nn.Sequential(nn.Linear(width, width), nn.LayerNorm(width), nn.GELU())
                for factor in FACTORS
            }) for name in MODALITIES
        })
        # 对偶一致性：共享与私有之间的双向映射。
        self.dual_forward = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})
        self.dual_backward = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})
        # MRC：变分编码器（对角高斯）与解码器。
        self.vib_encoder = nn.ModuleDict({name: nn.Linear(width * 3, width) for name in MODALITIES})
        self.vib_mu = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})
        self.vib_logvar = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})
        self.vib_decoder = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})
        # 交叉重建：用共享+私有融合表示重建模态表示。
        self.cross_decoder = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, width))
            for name in MODALITIES
        })
        # 信息增益约束：每个子空间挂一个辅助情感估计头。
        self.info_head = nn.ModuleDict({
            f"{name}_{factor}": nn.Linear(width, 3) for name in MODALITIES for factor in FACTORS
        })

        # MDF：样本调制 α、因子类型系数 β、分支注意力 γ 与噪声门控。
        self.sample_modulation = nn.Sequential(nn.Linear(width * 2, width // 2), nn.GELU(),
                                               nn.Linear(width // 2, 1))
        # β 必须初始化为非零：三尺度是相乘关系，β=0 会让 α 与 γ 一并拿到零梯度，
        # 融合在初始阶段退化为“三因子均匀平均”。
        self.factor_coefficient = nn.Parameter(torch.ones(len(FACTORS)))
        self.branch_attention = nn.ModuleDict({
            f"{name}_{factor}": nn.Linear(width, 1) for name in MODALITIES for factor in FACTORS
        })
        self.noise_gate = nn.ModuleDict({name: nn.Linear(width, width) for name in MODALITIES})

        self.fuse_proj = nn.Sequential(nn.Linear(width * 3, width), nn.LayerNorm(width), nn.GELU())
        self.use_availability_embedding = use_availability_embedding
        if use_availability_embedding:
            self.availability_embedding = nn.Embedding(8, width, padding_idx=0)
            nn.init.zeros_(self.availability_embedding.weight)
        self.fusion_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.pool_score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 3)
        self.regressor = nn.Linear(width, 1)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=dim) * mask.to(scores.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _audio_features(self, audio: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        if not self.audio_dynamics:
            return audio
        delta = torch.zeros_like(audio)
        adjacent = audio_mask[:, 1:] & audio_mask[:, :-1]
        delta[:, 1:] = (audio[:, 1:] - audio[:, :-1]) * adjacent.unsqueeze(-1).to(audio.dtype)
        return torch.cat((audio, delta), dim=-1)

    def encode(self, tokens, lengths, audio, vision, audio_mask, vision_mask, text_features=None,
               bert_input_ids=None, bert_attention_mask=None, bert_token_type_ids=None):
        steps = tokens.shape[1]
        lengths = lengths.clamp(1, steps).to("cpu", dtype=torch.int64)
        audio, vision = self.prepare_aligned_inputs(audio, vision, audio_mask, vision_mask)
        if self.text_mode == "tokens":
            text_input = self.embedding(tokens)
            tail_context = tail_available = None
        elif self.bert_finetune:
            if bert_input_ids is None or bert_attention_mask is None:
                raise ValueError("bert_finetune requires full-text BERT input ids and attention mask")
            aligned_text, tail_context, tail_available = self.bert_text_encoder(
                bert_input_ids, bert_attention_mask, bert_token_type_ids)
            text_input = self.text_input(aligned_text)
        else:
            if text_features is None or text_features.shape[:2] != tokens.shape:
                raise ValueError("bert text_mode requires (batch, steps, 768) text_features")
            text_input = self.text_input(text_features)
            tail_context = tail_available = None
        if self.encoder_type == "bigru":
            packed = nn.utils.rnn.pack_padded_sequence(
                text_input, lengths, batch_first=True, enforce_sorted=False)
            text_packed, _ = self.text_gru(packed)
            text_state, _ = nn.utils.rnn.pad_packed_sequence(
                text_packed, batch_first=True, total_length=steps)
            audio_state = self.run_aligned_gru(
                self.audio_gru, self.audio_input(self._audio_features(audio, audio_mask)), lengths)
            vision_state = self.run_aligned_gru(self.vision_gru, self.vision_input(vision), lengths)
        else:
            text_state = self.text_transformer(text_input, tokens.ne(0))
            audio_state = self.audio_transformer(
                self.audio_input(self._audio_features(audio, audio_mask)), audio_mask)
            vision_state = self.vision_transformer(self.vision_input(vision), vision_mask)
        encoded = {"text": text_state, "audio": audio_state, "vision": vision_state}
        if tail_context is not None:
            encoded["_tail_context"] = tail_context
            encoded["_tail_available"] = tail_available
        return encoded

    def factorize(self, encoded, available, tau=None):
        """HMF + MRC：返回三因子、重建结果与 KL 项（不可用词位一律置零）。"""
        tau = self.tau if tau is None else tau
        factors, recon, kl = {}, {}, {}
        batched_available = {}
        for name in MODALITIES:
            values = encoded[name]
            mask = available[name]
            batched_available[name] = mask
            gate = mask.unsqueeze(-1).to(values.dtype)
            parts = {factor: self.factor_proj[name][factor](values) * gate for factor in FACTORS}
            factors[name] = parts
            concat = torch.cat([parts[factor] for factor in FACTORS], dim=-1)
            hidden = torch.tanh(self.vib_encoder[name](concat))
            mu, logvar = self.vib_mu[name](hidden), self.vib_logvar[name](hidden).clamp(-8.0, 8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                sampled = mu + std * torch.randn_like(std)
            else:
                sampled = mu
            recon[name] = self.vib_decoder[name](sampled) * gate
            kl[name] = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar).mean()
        return factors, recon, kl, batched_available

    def dynamic_fusion(self, factors, available):
        """MDF：样本调制 × 因子系数 × 分支注意力，逐模态 softmax 后聚合，噪声分支加门控。"""
        # (B, T, 3 modalities, 3 factors)；α 是逐模态的样本调制分数，γ 是逐分支注意力。
        modulation = torch.stack([
            self.sample_modulation(torch.cat([factors[name]["shared"], factors[name]["private"]], -1)).squeeze(-1)
            for name in MODALITIES], dim=-1)
        attention = torch.stack([self.branch_attention[f"{name}_{factor}"](factors[name][factor]).squeeze(-1)
                                 for name in MODALITIES for factor in FACTORS], dim=-1)
        attention = attention.view(*attention.shape[:2], len(MODALITIES), len(FACTORS))
        logits = (modulation.unsqueeze(-1) + attention) * self.factor_coefficient.view(1, 1, 1, -1)
        weights = torch.softmax(logits, dim=-1)  # 在每个模态内部对三个因子归一化
        # 模态不可用时权重清零后重新归一化，保持聚合尺度稳定。
        gate = torch.stack([available[name] for name in MODALITIES], dim=-1).to(weights.dtype)
        weights = weights * gate.unsqueeze(-1)

        aggregated = {}
        for index, factor in enumerate(FACTORS):
            stacked = torch.stack([factors[name][factor] for name in MODALITIES], dim=2)
            part = weights[:, :, :, index].unsqueeze(-1)
            if factor == "noise":
                gated = torch.stack([factors[name]["noise"] * torch.sigmoid(self.noise_gate[name](factors[name]["noise"]))
                                     for name in MODALITIES], dim=2)
                stacked = gated
            denominator = part.sum(dim=2, keepdim=True).clamp_min(1e-6)
            aggregated[factor] = (stacked * part).sum(dim=2) / denominator.squeeze(-1)
        # 模态归因：每个因子内部先跨模态归一化，再对三个因子取平均。
        # 注意不能对因子维求和——softmax 后逐模态恒为 1，那样得到的只是可用性指示。
        normalized = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        modality_share = normalized.mean(dim=-1)
        return aggregated, modality_share

    def forward(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                text_features=None, bert_input_ids=None, bert_attention_mask=None,
                bert_token_type_ids=None):
        steps = tokens.shape[1]
        if steps != SEQ_LEN:
            raise ValueError(f"Expected {SEQ_LEN} sequence positions, got {steps}")
        lengths = lengths.clamp(1, steps).to("cpu", dtype=torch.int64)
        encoded = self.encode(tokens, lengths, audio, vision, audio_mask, vision_mask, text_features,
                              bert_input_ids, bert_attention_mask, bert_token_type_ids)
        tail_context = encoded.pop("_tail_context", None)
        tail_available = encoded.pop("_tail_available", None)
        available = {"text": text_mask.bool(), "audio": audio_mask.bool(), "vision": vision_mask.bool()}
        factors, recon, kl, _ = self.factorize(encoded, available)
        aggregated, modality_share = self.dynamic_fusion(factors, available)

        fused = self.fuse_proj(torch.cat([aggregated["shared"], aggregated["private"], aggregated["noise"]], dim=-1))
        if self.use_availability_embedding:
            bits = torch.stack([available[name] for name in MODALITIES], dim=2).to(torch.long)
            combined = (bits * bits.new_tensor((1, 2, 4))).sum(dim=2)
            fused = fused + self.availability_embedding(combined)

        valid_steps = torch.stack([available[name] for name in MODALITIES], dim=2).any(dim=2)
        fused_state = self.run_aligned_gru(self.fusion_gru, fused, lengths)
        time_weights = self._masked_softmax(self.pool_score(fused_state).squeeze(-1), valid_steps, dim=1)
        pooled = (fused_state * time_weights.unsqueeze(-1)).sum(dim=1)
        if tail_context is not None:
            tail = self.long_text_projection(tail_context)
            pooled = pooled + torch.tanh(self.long_text_scale) * tail * tail_available.unsqueeze(-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        raw_strength = self.regressor(pooled).squeeze(-1)
        if self.regression_mode == "signed":
            sentiment = 3.0 * torch.tanh(raw_strength)
            magnitude = sentiment.abs()
        else:
            magnitude = 3.0 * torch.sigmoid(raw_strength)
        if self.regression_mode == "soft":
            probabilities = logits.softmax(dim=-1)
            sentiment = magnitude * (probabilities[:, 2] - probabilities[:, 0])
        elif self.regression_mode == "hard":
            sentiment = magnitude * (logits.argmax(dim=-1) - 1).to(magnitude.dtype)

        shared = {name: factors[name]["shared"] for name in MODALITIES}
        private = {name: factors[name]["private"] for name in MODALITIES}
        noise = {name: factors[name]["noise"] for name in MODALITIES}
        cross = {}
        bridge = torch.cat([aggregated["shared"], aggregated["private"]], dim=-1)
        for name in MODALITIES:
            cross[name] = self.cross_decoder[name](bridge)
        info_logits = {
            f"{name}_{factor}": self.info_head[f"{name}_{factor}"](masked_mean(factors[name][factor], available[name]))
            for name in MODALITIES for factor in FACTORS
        }
        result = {
            "logits": logits,
            "magnitude": magnitude,
            "sentiment": sentiment,
            "modality_weights": modality_share,
            "time_weights": time_weights,
            "encoded": encoded,
            "shared": shared,
            "private": private,
            "noise": noise,
            "recon": recon,
            "cross": cross,
            "kl": kl,
            "info_logits": info_logits,
            "available": available,
        }
        if self.text_mode == "tokens":
            result["teacher_pred"] = self.text_teacher(encoded["text"])
        return result


def fuse_regularization(masked: dict, clean: dict, classes: torch.Tensor, class_weights=None,
                        *, tau: float = 0.1, kl_beta: float = 1e-3,
                        contrast_weight: float = 0.1, info_weight: float = 0.1,
                        dual_weight: float = 0.05, mrc_weight: float = 0.1,
                        cross_weight: float = 0.1) -> dict[str, torch.Tensor]:
    """HMF / MRC 正则项 + 缺失感知的交叉重建项。

    所有项都只在满足可用性条件的词位上求均值，因此整段或局部缺失的样本不会
    被零填充位置污染。交叉重建以**完整分支**的编码结果为教师，监督"用可用模态
    重建缺失模态"，这是原论文没有、但问题二必需的缺失感知机制。
    """
    total = masked["logits"].new_zeros(())
    stats: dict[str, torch.Tensor] = {}

    # HMF-1 对比分离：拉近跨模态共享子空间，推远同一模态的共享与私有。
    contrast_terms, contrast_count = [], 0
    shared, private, available = masked["shared"], masked["private"], masked["available"]
    for first in MODALITIES:
        for second in MODALITIES:
            if first == second:
                continue
            pair_mask = available[first] & available[second]
            if not pair_mask.any():
                continue
            positive = F.cosine_similarity(shared[first], shared[second], dim=-1) / tau
            negative = F.cosine_similarity(shared[first], private[first], dim=-1) / tau
            stacked = torch.stack([positive, negative], dim=-1)
            term = -torch.log_softmax(stacked, dim=-1)[..., 0]
            contrast_terms.append(term[pair_mask].mean())
            contrast_count += 1
    if contrast_count:
        stats["contrast"] = torch.stack(contrast_terms).mean()
        total = total + contrast_weight * stats["contrast"]

    # HMF-2 信息增益：共享/私有分支要能预测情感，噪声分支不得优于随机猜。
    # 论文用信息得分 ℓ，直接最大化 ℓ_noise 的相反数会让噪声头无限发散，
    # 因此这里改为有界形式：共享/私有取交叉熵（越小越好），噪声取
    # relu(log3 - CE)，即"一旦不比随机猜好就不再产生梯度"。信息头不使用
    # 类别权重，保证随机猜基准恰为 log3。
    info_terms = []
    chance = math.log(3.0)
    for name in MODALITIES:
        has = available[name].any(dim=1)
        if not has.any():
            continue
        weight = has.to(masked["logits"].dtype)
        denominator = has.sum().clamp_min(1)
        for factor in ("shared", "private"):
            logits = masked["info_logits"][f"{name}_{factor}"]
            loss = F.cross_entropy(logits, classes, reduction="none")
            info_terms.append((loss * weight).sum() / denominator)
        noise_logits = masked["info_logits"][f"{name}_noise"]
        noise_loss = F.cross_entropy(noise_logits, classes, reduction="none")
        info_terms.append((F.relu(chance - noise_loss) * weight).sum() / denominator)
    if info_terms:
        stats["info"] = torch.stack(info_terms).sum() / len(MODALITIES)
        total = total + info_weight * stats["info"]

    # HMF-3 对偶一致性需要模型参数，由 build_fuse_regularization 追加。

    # MRC 变分重建（论文 L2 形式，按元素取均值以免量级压过任务损失）+ KL 最小化。
    mrc_terms = []
    for name in MODALITIES:
        error = (masked["recon"][name] - masked["encoded"][name].detach()).pow(2).mean(dim=-1)
        mrc_terms.append((error * available[name]).sum() / available[name].sum().clamp_min(1.0))
    stats["mrc"] = torch.stack(mrc_terms).mean()
    stats["kl"] = torch.stack([masked["kl"][name] for name in MODALITIES]).mean()
    total = total + mrc_weight * stats["mrc"] + kl_beta * stats["kl"]

    # 缺失感知交叉重建：只用可用模态的融合表示去重建"完整分支里存在、本分支缺失"的模态。
    cross_terms = []
    for name in MODALITIES:
        target = clean["encoded"][name].detach()
        missing = clean["available"][name] & ~available[name]
        if not missing.any():
            continue
        predicted = masked["cross"][name]
        distance = 1.0 - F.cosine_similarity(predicted, target, dim=-1)
        cross_terms.append(distance[missing].mean())
    if cross_terms:
        stats["cross"] = torch.stack(cross_terms).mean()
        total = total + cross_weight * stats["cross"]
    else:
        stats["cross"] = masked["logits"].new_zeros(())

    stats.setdefault("contrast", masked["logits"].new_zeros(()))
    stats.setdefault("info", masked["logits"].new_zeros(()))
    stats["total"] = total
    return stats


def build_fuse_regularization(model, masked: dict, clean: dict, classes: torch.Tensor, class_weights=None,
                              **kwargs) -> dict[str, torch.Tensor]:
    """在 ``fuse_regularization`` 基础上补入需要模型参数的对偶一致性项。"""
    stats = fuse_regularization(masked, clean, classes, class_weights, **kwargs)
    dual_weight = float(kwargs.get("dual_weight", 0.05))
    dual_terms = []
    for name in MODALITIES:
        gate = masked["available"][name].to(masked["shared"][name].dtype)
        forward = model.dual_forward[name](masked["private"][name])
        backward = model.dual_backward[name](masked["shared"][name])
        term = ((masked["shared"][name] - forward).pow(2).mean(dim=-1)
                + (masked["private"][name] - backward).pow(2).mean(dim=-1))
        dual_terms.append((term * gate).sum() / gate.sum().clamp_min(1.0))
    stats["dual"] = torch.stack(dual_terms).mean()
    stats["total"] = stats["total"] + dual_weight * stats["dual"]
    return stats


def demo() -> None:
    torch.manual_seed(0)
    model = FactorizedAffectiveModel(text_mode="bert", dropout=0.0)
    batch = 2
    tokens = torch.zeros(batch, SEQ_LEN, dtype=torch.long)
    tokens[:, :4] = torch.tensor([101, 100, 200, 102])
    text = torch.randn(batch, SEQ_LEN, TEXT_DIM)
    audio = torch.randn(batch, SEQ_LEN, AUDIO_DIM)
    vision = torch.randn(batch, SEQ_LEN, VISION_DIM)
    mask = torch.zeros(batch, SEQ_LEN, dtype=torch.bool)
    mask[:, :10] = True
    out = model(tokens, torch.full((batch,), 10), audio, vision, mask, mask, mask, text)
    assert out["logits"].shape == (batch, 3) and out["sentiment"].shape == (batch,)
    assert set(out["shared"]) == set(MODALITIES)
    assert out["modality_weights"].shape == (batch, SEQ_LEN, 3)
    # 三模态全部无效的词位按构造取 0；只要有一个模态可用，份额就归一化为 1。
    share = out["modality_weights"][mask]
    assert torch.allclose(share.sum(-1), torch.ones(share.shape[0]), atol=1e-4)
    assert torch.allclose(out["modality_weights"][~mask], torch.zeros_like(out["modality_weights"][~mask]))
    # 归因必须随参数变化，而不是退化成可用性指示。
    other = FactorizedAffectiveModel(text_mode="bert", dropout=0.0)
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in other.branch_attention.parameters():
            parameter.add_(0.5)
        changed = other(tokens, torch.full((batch,), 10), audio, vision, mask, mask, mask, text)
    assert not torch.allclose(out["modality_weights"], changed["modality_weights"], atol=1e-5), \
        "modality_weights 未随参数变化，仍是可用性指示"
    assert all(out["recon"][name].shape == (batch, SEQ_LEN, 128) for name in MODALITIES)
    assert torch.isfinite(out["logits"]).all()

    # 整段缺失：音频与视觉全掩码时不应产生 NaN，且模态份额集中到文本。
    none = torch.zeros(batch, SEQ_LEN, dtype=torch.bool)
    missing = model(tokens, torch.full((batch,), 10), torch.zeros_like(audio),
                    torch.zeros_like(vision), mask, none, none, text)
    assert torch.isfinite(missing["logits"]).all()
    share = missing["modality_weights"][:, :10, 0]
    assert share.mean() > 0.99, f"text share should dominate when only text is available: {share.mean()}"

    # 局部缺失：音频在部分词位缺失仍可计算。
    partial = mask.clone()
    partial[:, 3:6] = False
    part = model(tokens, torch.full((batch,), 10), audio, vision, mask, partial, mask, text)
    assert torch.isfinite(part["logits"]).all()

    clean = model(tokens, torch.full((batch,), 10), audio, vision, mask, mask, mask, text)
    stats = build_fuse_regularization(model, part, clean, torch.tensor([0, 2]))
    for key in ("contrast", "info", "dual", "mrc", "kl", "cross", "total"):
        assert key in stats, key
        assert torch.isfinite(stats[key]), f"{key} not finite: {stats[key]}"
    stats["total"].backward()
    assert model.vib_decoder["text"].weight.grad is not None, "MRC 未回传梯度"
    assert model.info_head["text_shared"].weight.grad is not None, "信息增益约束未回传梯度"
    assert model.dual_forward["text"].weight.grad is not None, "对偶一致性未回传梯度"
    # 任务损失路径（正则项本身不经过预测头，需重新前向一次）。
    model.zero_grad(set_to_none=True)
    fresh = model(tokens, torch.full((batch,), 10), audio, vision, mask, partial, mask, text)
    task = (F.cross_entropy(fresh["logits"], torch.tensor([0, 2]))
            + F.smooth_l1_loss(fresh["sentiment"], torch.tensor([-1.0, 1.0])))
    task.backward()
    assert model.classifier.weight.grad is not None
    print("fuse_net demo passed;", {k: round(float(v.detach()), 5) for k, v in stats.items()})


if __name__ == "__main__":
    demo()
