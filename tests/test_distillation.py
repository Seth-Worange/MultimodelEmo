import unittest

import torch

from model.distillation import (
    DistillationConfig,
    compute_distillation_losses,
    correlation_kd_loss,
    correlation_matrix,
    feature_kd_loss,
)


class DistillationChecks(unittest.TestCase):
    def _outputs(self, batch: int = 4):
        teacher_feature = torch.randn(batch, 8)
        student_feature = torch.randn(batch, 8, requires_grad=True)
        return (
            {"logits": torch.randn(batch, 3), "sentiment": torch.randn(batch),
             "feature": teacher_feature},
            {"logits": torch.randn(batch, 3, requires_grad=True),
             "sentiment": torch.randn(batch, requires_grad=True),
             "feature": student_feature},
        )

    def test_losses_are_finite_and_student_gets_gradient(self):
        teacher, student = self._outputs()
        losses = compute_distillation_losses(teacher, student)
        for value in losses.values():
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))
        losses["total_kd_loss"].backward()
        self.assertIsNotNone(student["feature"].grad)
        self.assertIsNone(teacher["feature"].grad)

    def test_identity_feature_and_correlation_are_zero(self):
        feature = torch.randn(4, 8)
        self.assertLess(float(feature_kd_loss(feature, feature)), 1e-6)
        self.assertLess(float(correlation_kd_loss(feature, feature)), 1e-6)

    def test_correlation_shape_and_batch_one(self):
        feature = torch.randn(1, 8, requires_grad=True)
        self.assertEqual(tuple(correlation_matrix(feature).shape), (1, 1))
        loss = correlation_kd_loss(feature.detach(), feature)
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertIsNotNone(feature.grad)

    def test_ablation_switches_return_zero_terms(self):
        teacher, student = self._outputs()
        config = DistillationConfig(
            output_enabled=False, feature_enabled=False, correlation_enabled=False
        )
        losses = compute_distillation_losses(teacher, student, config)
        for value in losses.values():
            self.assertEqual(float(value), 0.0)


if __name__ == "__main__":
    unittest.main()
