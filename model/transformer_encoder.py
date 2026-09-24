from __future__ import annotations

import torch
from torch import nn

from utils.data import SEQ_LEN


class TemporalTransformerEncoder(nn.Module):
    """带可学习位置编码与缺失位置掩码的序列编码器。"""

    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 4,
                 layers: int = 1, dropout: float = 0.2):
        super().__init__()
        width = hidden_dim * 2
        if width % heads:
            raise ValueError(f"Transformer width {width} must be divisible by heads={heads}")
        if layers < 1:
            raise ValueError("Transformer layers must be positive")
        self.input_projection = (nn.Identity() if input_dim == width
                                 else nn.Linear(input_dim, width))
        self.position = nn.Parameter(torch.empty(1, SEQ_LEN, width))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if values.shape[1] != self.position.shape[1]:
            raise ValueError(f"Expected {self.position.shape[1]} sequence positions, got {values.shape[1]}")
        valid = valid.bool()
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        encoded = self.input_projection(values) + self.position[:, :values.shape[1]]
        encoded = self.encoder(self.dropout(encoded), src_key_padding_mask=~safe_valid)
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)
