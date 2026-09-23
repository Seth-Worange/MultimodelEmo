'''
Author: Orange
Date: 2026-09-23 08:54
LastEditors: Orange
LastEditTime: 2026-09-23 09:19
FilePath: model.py
Description: 

'''
from __future__ import annotations

"""Masked temporal fusion with token-distillation and contextual-BERT text modes."""

import torch
from torch import nn
from torch.nn import functional as F

from data import AUDIO_DIM, SEQ_LEN, TEXT_DIM, VISION_DIM, VOCAB_SIZE


class AffectiveModel(nn.Module):
    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64, fusion: str = "gate",
                 text_mode: str = "tokens", audio_dynamics: bool = False):
        super().__init__()
        if fusion not in {"gate", "concat"}:
            raise ValueError(f"Unknown fusion mode: {fusion}")
        if text_mode not in {"tokens", "bert"}:
            raise ValueError(f"Unknown text mode: {text_mode}")
        self.fusion = fusion
        self.text_mode = text_mode
        self.audio_dynamics = audio_dynamics
        width = hidden_dim * 2
        if text_mode == "tokens":
            self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=0)
            text_input_dim = embedding_dim
            self.text_teacher = nn.Linear(width, TEXT_DIM)
        else:
            self.text_input = nn.Sequential(nn.Linear(TEXT_DIM, width), nn.LayerNorm(width), nn.GELU())
            text_input_dim = width
        self.text_gru = nn.GRU(text_input_dim, hidden_dim, batch_first=True, bidirectional=True)
        audio_dim = AUDIO_DIM * (2 if audio_dynamics else 1)
        self.audio_input = nn.Sequential(nn.Linear(audio_dim, width), nn.LayerNorm(width), nn.GELU())
        self.vision_input = nn.Sequential(nn.Linear(VISION_DIM, width), nn.LayerNorm(width), nn.GELU())
        self.audio_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.vision_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        if fusion == "gate":
            self.gate = nn.Linear(width, 1)
        else:
            self.fusion_input = nn.Sequential(
                nn.Linear(width * 3 + 3, width), nn.LayerNorm(width), nn.GELU()
            )
        self.fusion_gru = nn.GRU(width, hidden_dim, batch_first=True, bidirectional=True)
        self.pool_score = nn.Linear(width, 1)
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

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        vision_mask: torch.Tensor,
        text_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, steps = tokens.shape
        if steps != SEQ_LEN:
            raise ValueError(f"Expected {SEQ_LEN} sequence positions, got {steps}")
        lengths = lengths.clamp(1, steps).to("cpu", dtype=torch.int64)
        if self.text_mode == "tokens":
            text_input = self.embedding(tokens)
        else:
            if text_features is None or text_features.shape[:2] != tokens.shape:
                raise ValueError("bert text_mode requires (batch, steps, 768) text_features")
            text_input = self.text_input(text_features)
        packed = nn.utils.rnn.pack_padded_sequence(text_input, lengths, batch_first=True, enforce_sorted=False)
        text_packed, _ = self.text_gru(packed)
        text_state, _ = nn.utils.rnn.pad_packed_sequence(text_packed, batch_first=True, total_length=steps)
        audio_values = self._audio_features(audio, audio_mask)
        audio_state, _ = self.audio_gru(self.audio_input(audio_values))
        vision_state, _ = self.vision_gru(self.vision_input(vision))

        states = torch.stack((text_state, audio_state, vision_state), dim=2)
        available = torch.stack((text_mask, audio_mask, vision_mask), dim=2).bool()
        if self.fusion == "gate":
            modality_weights = self._masked_softmax(self.gate(states).squeeze(-1), available, dim=2)
            fused = (states * modality_weights.unsqueeze(-1)).sum(dim=2)
        else:
            masked_states = states * available.unsqueeze(-1).to(states.dtype)
            fused = self.fusion_input(torch.cat((masked_states.flatten(2), available.to(states.dtype)), dim=2))
            modality_weights = available.to(states.dtype) / available.sum(dim=2, keepdim=True).clamp_min(1)
        valid_steps = available.any(dim=2)
        fused_state, _ = self.fusion_gru(fused)
        time_weights = self._masked_softmax(self.pool_score(fused_state).squeeze(-1), valid_steps, dim=1)
        pooled = (fused_state * time_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled)
        magnitude = 3.0 * torch.sigmoid(self.regressor(pooled).squeeze(-1))
        sentiment = magnitude * (logits.argmax(dim=-1) - 1).to(magnitude.dtype)
        result = {
            "logits": logits,
            "magnitude": magnitude,
            "sentiment": sentiment,
            "modality_weights": modality_weights,
            "time_weights": time_weights,
        }
        if self.text_mode == "tokens":
            result["teacher_pred"] = self.text_teacher(text_state)
        return result


def demo() -> None:
    model = AffectiveModel().eval()
    tokens = torch.zeros(2, SEQ_LEN, dtype=torch.long)
    tokens[:, :4] = torch.tensor([101, 100, 200, 102])
    mask = torch.zeros(2, SEQ_LEN, dtype=torch.bool)
    output = model(tokens, torch.full((2,), 4), torch.zeros(2, SEQ_LEN, AUDIO_DIM),
                   torch.zeros(2, SEQ_LEN, VISION_DIM), mask, mask, mask)
    assert output["logits"].shape == (2, 3)
    assert output["magnitude"].shape == (2,)
    assert output["sentiment"].shape == (2,)
    assert torch.isfinite(output["logits"]).all()
    assert torch.all(output["sentiment"] * (output["logits"].argmax(dim=-1) - 1) >= 0)
    assert torch.isfinite(output["modality_weights"]).all()
    concat_model = AffectiveModel(fusion="concat").eval()
    concat_out = concat_model(tokens, torch.full((2,), 4), torch.zeros(2, SEQ_LEN, AUDIO_DIM),
                              torch.zeros(2, SEQ_LEN, VISION_DIM), mask, mask, mask)
    assert concat_out["logits"].shape == (2, 3)
    bert_model = AffectiveModel(text_mode="bert").eval()
    bert_out = bert_model(tokens, torch.full((2,), 4), torch.zeros(2, SEQ_LEN, AUDIO_DIM),
                          torch.zeros(2, SEQ_LEN, VISION_DIM), mask, mask, mask,
                          torch.zeros(2, SEQ_LEN, TEXT_DIM))
    assert bert_out["logits"].shape == (2, 3)
    dynamic_model = AffectiveModel(text_mode="bert", audio_dynamics=True).eval()
    audio = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, SEQ_LEN, 1).expand(2, -1, AUDIO_DIM)
    audio_mask = torch.ones(2, SEQ_LEN, dtype=torch.bool)
    audio_mask[:, 3] = False
    dynamics = dynamic_model._audio_features(audio, audio_mask)
    assert dynamics.shape == (2, SEQ_LEN, AUDIO_DIM * 2)
    assert torch.equal(dynamics[:, 1, AUDIO_DIM:], torch.ones(2, AUDIO_DIM))
    assert torch.count_nonzero(dynamics[:, 3:5, AUDIO_DIM:]) == 0


if __name__ == "__main__":
    demo()
    print("model.py checks passed")
