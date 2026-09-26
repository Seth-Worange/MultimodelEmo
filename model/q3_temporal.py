"""问题三的完整序列交互预测骨干。"""

from __future__ import annotations

import torch
from torch import nn

from utils.data import AUDIO_DIM, SEQ_LEN, TEXT_DIM, VISION_DIM
from .input_processing import AlignedInputProcessing
from .transformer_encoder import TemporalTransformerEncoder


class Q3TemporalModel(nn.Module, AlignedInputProcessing):
    """以文本时序为主轴，在整段语音与视觉中检索相关线索。"""

    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64,
                 text_mode: str = "bert", audio_dynamics: bool = True,
                 regression_mode: str = "signed", dropout: float = 0.2,
                 transformer_layers: int = 1, transformer_heads: int = 4,
                 normalize_inputs: bool = True, pack_aligned_grus: bool = True):
        super().__init__()
        if text_mode != "bert":
            raise ValueError("Q3TemporalModel requires frozen BERT features")
        if regression_mode not in {"signed", "soft"}:
            raise ValueError("Q3TemporalModel supports signed or soft regression")
        self.architecture = "q3_temporal"
        self.text_mode = text_mode
        self.audio_dynamics = audio_dynamics
        self.regression_mode = regression_mode
        self.bert_finetune = False
        self.init_input_processing(normalize_inputs, pack_aligned_grus, AUDIO_DIM, VISION_DIM)
        width = hidden_dim * 2
        self.text_input = nn.Sequential(nn.Linear(TEXT_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.audio_input = nn.Sequential(
            nn.Linear(AUDIO_DIM * (2 if audio_dynamics else 1), width), nn.LayerNorm(width), nn.GELU())
        self.vision_input = nn.Sequential(nn.Linear(VISION_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.encoders = nn.ModuleDict({name: nn.GRU(width, hidden_dim, batch_first=True,
                                                    bidirectional=True)
                                       for name in ("text", "audio", "vision")})
        self.modality_position = nn.Parameter(torch.empty(1, SEQ_LEN, width))
        nn.init.normal_(self.modality_position, std=0.02)
        self.audio_attention = nn.MultiheadAttention(width, transformer_heads, dropout=dropout,
                                                      batch_first=True)
        self.vision_attention = nn.MultiheadAttention(width, transformer_heads, dropout=dropout,
                                                       batch_first=True)
        self.local_gate = nn.Linear(width * 3, 2)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.fusion_norm = nn.LayerNorm(width)
        self.temporal = TemporalTransformerEncoder(width, hidden_dim, transformer_heads,
                                                    transformer_layers, dropout)
        self.pool_score = nn.Linear(width, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 3)
        self.regressor = nn.Linear(width, 1)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=dim) * mask.to(scores.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)

    def _audio_features(self, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.audio_dynamics:
            return audio
        delta = torch.zeros_like(audio)
        adjacent = mask[:, 1:] & mask[:, :-1]
        delta[:, 1:] = (audio[:, 1:] - audio[:, :-1]) * adjacent.unsqueeze(-1)
        return torch.cat((audio, delta), dim=-1)

    def _attend(self, query: torch.Tensor, state: torch.Tensor, mask: torch.Tensor,
                attention: nn.MultiheadAttention) -> torch.Tensor:
        safe = mask.clone()
        empty = ~safe.any(dim=1)
        safe[empty, 0] = True
        keys = state + self.modality_position[:, :state.shape[1]]
        context = attention(query, keys, state, key_padding_mask=~safe, need_weights=False)[0]
        return context * (~empty).to(context.dtype)[:, None, None]

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor, text_mask: torch.Tensor, audio_mask: torch.Tensor,
                vision_mask: torch.Tensor, text_features: torch.Tensor | None = None,
                bert_input_ids: torch.Tensor | None = None,
                bert_attention_mask: torch.Tensor | None = None,
                bert_token_type_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if tokens.shape[1] != SEQ_LEN or text_features is None or text_features.shape[:2] != tokens.shape:
            raise ValueError("Expected aligned (batch, 50, 768) BERT features")
        text_mask, audio_mask, vision_mask = (mask.bool() for mask in
                                              (text_mask, audio_mask, vision_mask))
        audio, vision = self.prepare_aligned_inputs(audio, vision, audio_mask, vision_mask)
        inputs = {
            "text": self.text_input(text_features),
            "audio": self.audio_input(self._audio_features(audio, audio_mask)),
            "vision": self.vision_input(vision),
        }
        states = {name: self.run_aligned_gru(self.encoders[name], values, lengths)
                  for name, values in inputs.items()}
        text, acoustic, visual = (states[name] for name in ("text", "audio", "vision"))
        local_available = torch.stack((audio_mask, vision_mask), dim=-1)
        local_weights = self._masked_softmax(
            self.local_gate(torch.cat((text, acoustic, visual), dim=-1)), local_available, -1)
        local = (local_weights[..., 0:1] * acoustic + local_weights[..., 1:2] * visual)
        acoustic_context = self._attend(text, acoustic, audio_mask, self.audio_attention)
        visual_context = self._attend(text, visual, vision_mask, self.vision_attention)
        context = (acoustic_context * audio_mask.any(1)[:, None, None]
                   + visual_context * vision_mask.any(1)[:, None, None]) / (
                       audio_mask.any(1).to(text.dtype) + vision_mask.any(1).to(text.dtype)
                   ).clamp_min(1)[:, None, None]
        fused = self.fusion_norm(text + local + torch.tanh(self.context_scale) * context)
        fused = self.temporal(fused, text_mask)
        time_weights = self._masked_softmax(self.pool_score(fused).squeeze(-1), text_mask, 1)
        pooled = self.dropout((fused * time_weights.unsqueeze(-1)).sum(dim=1))
        logits = self.classifier(pooled)
        raw = self.regressor(pooled).squeeze(-1)
        if self.regression_mode == "signed":
            sentiment = 3.0 * torch.tanh(raw)
            magnitude = sentiment.abs()
        else:
            magnitude = 3.0 * torch.sigmoid(raw)
            probabilities = logits.softmax(dim=-1)
            sentiment = magnitude * (probabilities[:, 2] - probabilities[:, 0])
        return {"logits": logits, "sentiment": sentiment, "magnitude": magnitude,
                "time_weights": time_weights, "modality_weights": local_weights}
