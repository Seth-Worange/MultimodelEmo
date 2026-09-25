from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.data import AUDIO_DIM, SEQ_LEN, TEXT_DIM, VISION_DIM, VOCAB_SIZE
from .transformer_encoder import TemporalTransformerEncoder
from .input_processing import AlignedInputProcessing

MODALITIES = ("text", "audio", "vision")


class ConfidenceCalibration(nn.Module):
    """CICA 的分段置信度校准损失。"""

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        if temperature <= 0:
            raise ValueError("confidence calibration temperature must be positive")
        self.temperature = temperature
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_weights = nn.Parameter(torch.zeros(3))

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.raw_beta)
        alpha = beta + (1.0 - beta) * torch.sigmoid(self.raw_alpha)
        # 用软分段保留边界梯度，同时维持置信度区间的顺序。
        weights = 1.0 + F.softplus(self.raw_weights)
        low = torch.sigmoid((beta - confidence) / self.temperature)
        high = torch.sigmoid((confidence - alpha) / self.temperature)
        middle = (1.0 - low - high).clamp_min(0.0)
        penalty = low * weights[0] + middle * weights[1] + high * weights[2]
        return (penalty * (confidence - 1.0).square()).mean()


class CICAAffectiveModel(nn.Module, AlignedInputProcessing):
    """CICA 启发的两阶段置信度预训练与可靠性门控模型。"""

    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64,
                 text_mode: str = "bert", audio_dynamics: bool = True,
                 regression_mode: str = "soft", dropout: float = 0.2,
                 encoder_type: str = "bigru", transformer_layers: int = 1,
                 transformer_heads: int = 4, ca_temperature: float = 0.05,
                 normalize_inputs: bool = False, pack_aligned_grus: bool = False):
        super().__init__()
        if text_mode not in {"tokens", "bert"}:
            raise ValueError(f"Unknown text mode: {text_mode}")
        if regression_mode not in {"hard", "soft", "signed"}:
            raise ValueError(f"Unknown regression mode: {regression_mode}")
        if encoder_type not in {"bigru", "transformer"}:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        if ca_temperature <= 0:
            raise ValueError("ca_temperature must be positive")
        self.architecture = "cica"
        self.text_mode = text_mode
        self.audio_dynamics = audio_dynamics
        self.regression_mode = regression_mode
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.init_input_processing(normalize_inputs, pack_aligned_grus, AUDIO_DIM, VISION_DIM)
        width = hidden_dim * 2

        if text_mode == "tokens":
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
            text_input_dim = embedding_dim
        else:
            self.text_input = nn.Sequential(nn.Linear(TEXT_DIM, width), nn.LayerNorm(width), nn.GELU())
            text_input_dim = width
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

        self.modal_heads = nn.ModuleDict({
            name: nn.ModuleDict({
                "classifier": nn.Linear(width, 3),
                "regressor": nn.Linear(width, 1),
                "confidence": nn.Linear(width, 1),
                "uncertainty": nn.Linear(width, 1),
            }) for name in MODALITIES
        })
        self.confidence_calibration = nn.ModuleDict({
            name: ConfidenceCalibration(ca_temperature) for name in MODALITIES
        })

        self.gate = nn.Linear(width, 1)
        self.availability_embedding = nn.Embedding(8, width, padding_idx=0)
        nn.init.zeros_(self.availability_embedding.weight)
        self.fusion_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.pool_score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)
        self.modal_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 3)
        self.regressor = nn.Linear(width, 1)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=dim)
        weights = weights * mask.to(scores.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _audio_features(self, audio: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        if not self.audio_dynamics:
            return audio
        delta = torch.zeros_like(audio)
        adjacent = audio_mask[:, 1:] & audio_mask[:, :-1]
        delta[:, 1:] = (audio[:, 1:] - audio[:, :-1]) * adjacent.unsqueeze(-1).to(audio.dtype)
        return torch.cat((audio, delta), dim=-1)

    def encode_modalities(self, tokens, lengths, audio, vision, audio_mask, vision_mask,
                          text_features=None):
        steps = tokens.shape[1]
        lengths = lengths.clamp(1, steps).to("cpu", dtype=torch.int64)
        audio, vision = self.prepare_aligned_inputs(audio, vision, audio_mask, vision_mask)
        if self.text_mode == "tokens":
            text_input = self.embedding(tokens)
        else:
            if text_features is None or text_features.shape[:2] != tokens.shape:
                raise ValueError("bert text_mode requires (batch, steps, 768) text_features")
            text_input = self.text_input(text_features)
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
        return {"text": text_state, "audio": audio_state, "vision": vision_state}

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.unsqueeze(-1).to(values.dtype)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _branch_outputs(self, states, available):
        pooled, logits, sentiment, confidence, uncertainty, magnitude = [], [], [], [], [], []
        for name in MODALITIES:
            vector = self._masked_mean(states[name], available[name])
            heads = self.modal_heads[name]
            head_input = self.modal_dropout(vector)
            branch_logits = heads["classifier"](head_input)
            raw_strength = heads["regressor"](head_input).squeeze(-1)
            branch_magnitude = 3.0 * torch.sigmoid(raw_strength)
            probability = branch_logits.softmax(dim=-1)
            if self.regression_mode == "soft":
                branch_sentiment = branch_magnitude * (probability[:, 2] - probability[:, 0])
            elif self.regression_mode == "hard":
                branch_sentiment = branch_magnitude * (branch_logits.argmax(-1) - 1).to(vector.dtype)
            else:
                branch_sentiment = 3.0 * torch.tanh(raw_strength)
                branch_magnitude = branch_sentiment.abs()
            valid = available[name].any(dim=1)
            branch_confidence = torch.sigmoid(heads["confidence"](head_input).squeeze(-1))
            branch_uncertainty = torch.sigmoid(heads["uncertainty"](head_input).squeeze(-1))
            # 全模态缺失时按 CICA 约定输出低置信度和高不确定性。
            branch_confidence = torch.where(valid, branch_confidence, torch.zeros_like(branch_confidence))
            branch_uncertainty = torch.where(valid, branch_uncertainty, torch.ones_like(branch_uncertainty))
            pooled.append(vector)
            logits.append(branch_logits)
            sentiment.append(branch_sentiment)
            confidence.append(branch_confidence)
            uncertainty.append(branch_uncertainty)
            magnitude.append(branch_magnitude)
        return {
            "modal_pooled": torch.stack(pooled, dim=1),
            "modal_logits": torch.stack(logits, dim=1),
            "modal_sentiment": torch.stack(sentiment, dim=1),
            "confidence": torch.stack(confidence, dim=1),
            "uncertainty": torch.stack(uncertainty, dim=1),
            "modal_magnitude": torch.stack(magnitude, dim=1),
        }

    def _availability(self, tokens, text_mask, audio_mask, vision_mask):
        if tokens.shape[1] != SEQ_LEN:
            raise ValueError(f"Expected {SEQ_LEN} sequence positions, got {tokens.shape[1]}")
        return {"text": text_mask.bool(), "audio": audio_mask.bool(),
                "vision": vision_mask.bool()}

    def _fuse(self, states, branch, available, lengths):
        stacked = torch.stack([states[name] for name in MODALITIES], dim=2)
        masks = torch.stack([available[name] for name in MODALITIES], dim=2)

        gate_scores = self.gate(stacked).squeeze(-1)
        base_weights = self._masked_softmax(gate_scores, masks, dim=2)
        reliability = F.relu(1.0 + branch["confidence"] - branch["uncertainty"])
        scaled = base_weights * reliability[:, None, :]
        denominator = scaled.sum(dim=2, keepdim=True)
        weights = torch.where(denominator > 1e-8,
                              scaled / denominator.clamp_min(1e-8), base_weights)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)

        bits = masks.to(torch.long) * masks.new_tensor((1, 2, 4), dtype=torch.long)
        pattern = bits.sum(dim=2)
        fused = fused + self.availability_embedding(pattern)
        valid_steps = masks.any(dim=2)
        fused_state = self.run_aligned_gru(self.fusion_gru, fused, lengths)
        time_weights = self._masked_softmax(self.pool_score(fused_state).squeeze(-1),
                                            valid_steps, dim=1)
        pooled = self.dropout((fused_state * time_weights.unsqueeze(-1)).sum(dim=1))
        logits = self.classifier(pooled)
        raw_strength = self.regressor(pooled).squeeze(-1)
        magnitude = 3.0 * torch.sigmoid(raw_strength)
        if self.regression_mode == "soft":
            probability = logits.softmax(dim=-1)
            sentiment = magnitude * (probability[:, 2] - probability[:, 0])
        elif self.regression_mode == "hard":
            sentiment = magnitude * (logits.argmax(-1) - 1).to(magnitude.dtype)
        else:
            sentiment = 3.0 * torch.tanh(raw_strength)
            magnitude = sentiment.abs()
        return {
            **branch,
            "logits": logits,
            "magnitude": magnitude,
            "sentiment": sentiment,
            "modality_weights": weights,
            "time_weights": time_weights,
            "pooled": pooled,
            "modal_states": stacked,
            "reliability": reliability,
        }

    def forward_cap(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                    text_features=None):
        """CAP 阶段只执行单模态编码和预测，不计算融合层。"""
        available = self._availability(tokens, text_mask, audio_mask, vision_mask)
        states = self.encode_modalities(
            tokens, lengths, audio, vision, audio_mask, vision_mask, text_features)
        branch = self._branch_outputs(states, available)
        return {**branch, "modal_states": torch.stack([states[name] for name in MODALITIES], dim=2)}

    def forward_fusion(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                       text_features=None):
        """CIF 阶段以无梯度方式提取已冻结的单模态表征，只训练融合层。"""
        available = self._availability(tokens, text_mask, audio_mask, vision_mask)
        with torch.no_grad():
            states = self.encode_modalities(
                tokens, lengths, audio, vision, audio_mask, vision_mask, text_features)
            branch = self._branch_outputs(states, available)
        return self._fuse(states, branch, available, lengths)

    def forward(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                text_features=None):
        available = self._availability(tokens, text_mask, audio_mask, vision_mask)
        states = self.encode_modalities(
            tokens, lengths, audio, vision, audio_mask, vision_mask, text_features)
        branch = self._branch_outputs(states, available)
        return self._fuse(states, branch, available, lengths)

    def confidence_loss(self, modality: str, confidence: torch.Tensor) -> torch.Tensor:
        return self.confidence_calibration[modality](confidence)
