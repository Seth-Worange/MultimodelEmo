"""CMAD-inspired output, feature, and correlation distillation losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DistillationConfig:
    """Independent switches and weights for the three KD levels."""

    temperature: float = 2.0
    output_enabled: bool = True
    output_weight: float = 0.2
    feature_enabled: bool = True
    feature_weight: float = 0.1
    correlation_enabled: bool = True
    correlation_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        for name in ("output_weight", "feature_weight", "correlation_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def classification_kd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL divergence, with teacher probabilities detached."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * temperature**2


def regression_kd_loss(teacher_reg: torch.Tensor, student_reg: torch.Tensor) -> torch.Tensor:
    """SmoothL1 regression distillation against a detached teacher target."""
    return F.smooth_l1_loss(student_reg, teacher_reg.detach())


def _normalized_features(feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim != 2:
        raise ValueError(f"features must have shape [B, D], got {tuple(feature.shape)}")
    return F.normalize(feature, p=2, dim=-1)


def feature_kd_loss(teacher_feature: torch.Tensor, student_feature: torch.Tensor) -> torch.Tensor:
    """MSE between L2-normalized teacher and student batch features."""
    teacher_norm = _normalized_features(teacher_feature.detach())
    student_norm = _normalized_features(student_feature)
    if teacher_norm.shape != student_norm.shape:
        raise ValueError("teacher and student feature shapes must match")
    return F.mse_loss(student_norm, teacher_norm)


def correlation_matrix(feature: torch.Tensor) -> torch.Tensor:
    """Return the normalized batch relation matrix with shape [B, B]."""
    normalized = _normalized_features(feature)
    return normalized @ normalized.transpose(0, 1)


def correlation_kd_loss(teacher_feature: torch.Tensor, student_feature: torch.Tensor) -> torch.Tensor:
    """Match pairwise cosine relations in a batch; B=1 is well-defined."""
    teacher_corr = correlation_matrix(teacher_feature.detach())
    student_corr = correlation_matrix(student_feature)
    if teacher_corr.shape != student_corr.shape:
        raise ValueError("teacher and student correlation shapes must match")
    return F.mse_loss(student_corr, teacher_corr)


def compute_distillation_losses(
    teacher: dict[str, torch.Tensor],
    student: dict[str, torch.Tensor],
    config: DistillationConfig = DistillationConfig(),
) -> dict[str, torch.Tensor]:
    """Compute enabled KD terms and their weighted total.

    Expected output keys are ``logits``, ``sentiment``, and ``feature``.
    Disabled terms are returned as differentiable zero tensors on the student
    output device so logging and total-loss code need no special branches.
    """
    zero = student["logits"].sum() * 0.0
    output_cls = classification_kd_loss(
        teacher["logits"], student["logits"], config.temperature
    ) if config.output_enabled else zero
    output_reg = regression_kd_loss(
        teacher["sentiment"], student["sentiment"]
    ) if config.output_enabled else zero
    feature = feature_kd_loss(
        teacher["feature"], student["feature"]
    ) if config.feature_enabled else zero
    correlation = correlation_kd_loss(
        teacher["feature"], student["feature"]
    ) if config.correlation_enabled else zero
    output = output_cls + output_reg
    total = (config.output_weight * output
             + config.feature_weight * feature
             + config.correlation_weight * correlation)
    return {
        "kd_cls_loss": output_cls,
        "kd_reg_loss": output_reg,
        "feature_kd_loss": feature,
        "correlation_kd_loss": correlation,
        "output_kd_loss": output,
        "total_kd_loss": total,
    }


__all__ = [
    "DistillationConfig",
    "classification_kd_loss",
    "regression_kd_loss",
    "feature_kd_loss",
    "correlation_matrix",
    "correlation_kd_loss",
    "compute_distillation_losses",
]
