"""保留文本语义、由声画补充的轻量晚期融合模型。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.data import AUDIO_DIM, TEXT_DIM, VISION_DIM
from .input_processing import AlignedInputProcessing


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(values.dtype)
    return (values * weight).sum(1) / weight.sum(1).clamp_min(1.0)


class ComplementaryAffectiveModel(nn.Module, AlignedInputProcessing):
    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64, dropout: float = 0.2,
                 audio_dynamics: bool = True, regression_mode: str = "soft",
                 normalize_inputs: bool = True, pack_aligned_grus: bool = False,
                 text_mode: str = "bert"):
        super().__init__()
        if text_mode != "bert":
            raise ValueError("Complementary model requires cached BERT features")
        if regression_mode not in {"soft", "signed", "hard"}:
            raise ValueError(f"Unknown regression mode: {regression_mode}")
        self.text_mode = text_mode
        self.bert_finetune = False
        self.regression_mode = regression_mode
        self.audio_dynamics = audio_dynamics
        self.init_input_processing(normalize_inputs, pack_aligned_grus, AUDIO_DIM, VISION_DIM)
        width = hidden_dim * 2
        self.text_input = nn.Sequential(nn.Linear(TEXT_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.audio_input = nn.Sequential(
            nn.Linear(AUDIO_DIM * (2 if audio_dynamics else 1), width), nn.LayerNorm(width), nn.GELU())
        self.vision_input = nn.Sequential(nn.Linear(VISION_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.text_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.audio_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.vision_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.text_summary = nn.Sequential(nn.Linear(width * 2, width), nn.LayerNorm(width), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(width * 6 + 3, width * 2), nn.LayerNorm(width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 3)
        self.regressor = nn.Linear(width, 1)

    def forward(self, tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask,
                text_features=None, bert_input_ids=None, bert_attention_mask=None,
                bert_token_type_ids=None):
        if text_features is None:
            raise ValueError("Cached BERT features are required")
        lengths = lengths.clamp(1, tokens.shape[1]).to("cpu", dtype=torch.int64)
        audio, vision = self.prepare_aligned_inputs(audio, vision, audio_mask, vision_mask)
        if self.audio_dynamics:
            delta = torch.zeros_like(audio)
            adjacent = audio_mask[:, 1:] & audio_mask[:, :-1]
            delta[:, 1:] = (audio[:, 1:] - audio[:, :-1]) * adjacent.unsqueeze(-1)
            audio = torch.cat((audio, delta), -1)
        text_projected = self.text_input(text_features)
        text_state = self.run_aligned_gru(self.text_gru, text_projected, lengths)
        audio_state = self.run_aligned_gru(self.audio_gru, self.audio_input(audio), lengths)
        vision_state = self.run_aligned_gru(self.vision_gru, self.vision_input(vision), lengths)
        present = torch.stack([mask.any(1) for mask in (text_mask, audio_mask, vision_mask)], 1)
        text_body = masked_mean(text_state, text_mask)
        cls = text_projected[:, 0] * present[:, 0:1].to(text_body.dtype)
        text_summary = self.text_summary(torch.cat((text_body, cls), -1))
        text_summary = text_summary * present[:, 0:1]
        audio_summary = masked_mean(audio_state, audio_mask)
        vision_summary = masked_mean(vision_state, vision_mask)
        t, a, v = text_summary, audio_summary, vision_summary
        combined = torch.cat((t, a, v, t * a, t * v, a * v, present.to(t.dtype)), -1)
        pooled = self.dropout(self.fusion(combined))
        logits = self.classifier(pooled)
        raw_strength = self.regressor(pooled).squeeze(-1)
        if self.regression_mode == "signed":
            sentiment = 3 * torch.tanh(raw_strength)
            magnitude = sentiment.abs()
        else:
            magnitude = 3 * torch.sigmoid(raw_strength)
            if self.regression_mode == "soft":
                probabilities = logits.softmax(-1)
                sentiment = magnitude * (probabilities[:, 2] - probabilities[:, 0])
            else:
                sentiment = magnitude * (logits.argmax(-1) - 1)
        return {"logits": logits, "sentiment": sentiment, "magnitude": magnitude}
