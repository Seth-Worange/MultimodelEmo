"""对齐音视频输入的训练集标准化与打包序列运算。"""

from __future__ import annotations

import torch
from torch import nn


class AlignedInputProcessing:
    """供三个模型共用；均值方差保存在检查点中。"""

    def init_input_processing(self, normalize_inputs: bool, pack_aligned_grus: bool,
                              audio_dim: int, vision_dim: int) -> None:
        self.normalize_inputs = bool(normalize_inputs)
        self.pack_aligned_grus = bool(pack_aligned_grus)
        self.register_buffer("audio_mean", torch.zeros(audio_dim))
        self.register_buffer("audio_scale", torch.ones(audio_dim))
        self.register_buffer("vision_mean", torch.zeros(vision_dim))
        self.register_buffer("vision_scale", torch.ones(vision_dim))

    @torch.no_grad()
    def fit_input_stats(self, train: dict[str, torch.Tensor]) -> None:
        """仅以训练集可用时间步拟合逐维标准化参数。"""
        for name in ("audio", "vision"):
            values = train[name].to(dtype=torch.float64)
            valid = train[f"{name}_mask"].bool()
            selected = values[valid]
            if not selected.numel():
                raise ValueError(f"No valid training values for {name}")
            mean = selected.mean(dim=0)
            scale = selected.std(dim=0, unbiased=False)
            scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
            getattr(self, f"{name}_mean").copy_(mean.to(getattr(self, f"{name}_mean")))
            getattr(self, f"{name}_scale").copy_(scale.to(getattr(self, f"{name}_scale")))

    def prepare_aligned_inputs(self, audio: torch.Tensor, vision: torch.Tensor,
                               audio_mask: torch.Tensor, vision_mask: torch.Tensor):
        if not self.normalize_inputs:
            return audio, vision
        audio = (audio - self.audio_mean) / self.audio_scale
        vision = (vision - self.vision_mean) / self.vision_scale
        audio = audio * audio_mask.unsqueeze(-1).to(audio.dtype)
        vision = vision * vision_mask.unsqueeze(-1).to(vision.dtype)
        return audio, vision

    def run_aligned_gru(self, gru: nn.GRU, values: torch.Tensor,
                        lengths: torch.Tensor) -> torch.Tensor:
        if not self.pack_aligned_grus:
            return gru(values)[0]
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        steps = values.shape[1]
        packed_values = pack_padded_sequence(values, lengths.clamp(1, steps).to("cpu", dtype=torch.int64),
                                             batch_first=True, enforce_sorted=False)
        encoded, _ = gru(packed_values)
        return pad_packed_sequence(encoded, batch_first=True, total_length=steps)[0]


def fill_legacy_input_buffers(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """读取旧检查点时补齐新增的默认标准化缓冲区。"""
    result = dict(state)
    for name, value in model.state_dict().items():
        if name not in result and name.rsplit(".", 1)[-1] in {
                "audio_mean", "audio_scale", "vision_mean", "vision_scale"}:
            result[name] = value
    return result
