import unittest

import torch
from torch.nn import functional as F

from model import AffectiveModel
from model.distillation import DistillationConfig, compute_distillation_losses
from scripts.train import supervised_loss, supervised_loss_components


class PostSmokeFixChecks(unittest.TestCase):
    def test_valid_only_modality_mean_ignores_padding(self):
        weights = torch.tensor([[[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.0, 0.0, 0.0]],
                                [[0.6, 0.1, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        valid = torch.tensor([[True, True, False], [True, False, False]])
        mean = (weights * valid.unsqueeze(-1)).sum(dim=(0, 1)) / valid.sum()
        self.assertTrue(torch.allclose(mean, torch.tensor([0.4, 0.26666668, 0.33333334])))
        self.assertTrue(torch.allclose(mean.sum(), torch.ones(())))

    def test_task_loss_decomposition(self):
        output = {"logits": torch.randn(4, 3), "sentiment": torch.randn(4)}
        batch = {"classes": torch.tensor([0, 1, 2, 1]), "sentiment": torch.randn(4)}
        cls, reg = supervised_loss_components(output, batch)
        self.assertTrue(torch.allclose(cls + reg, supervised_loss(output, batch)))

    def test_no_kd_is_zero_and_task_only(self):
        teacher = {"logits": torch.randn(3, 3), "sentiment": torch.randn(3),
                   "feature": torch.randn(3, 5)}
        student = {"logits": torch.randn(3, 3, requires_grad=True),
                   "sentiment": torch.randn(3, requires_grad=True),
                   "feature": torch.randn(3, 5, requires_grad=True)}
        config = DistillationConfig(output_enabled=False, feature_enabled=False,
                                    correlation_enabled=False)
        losses = compute_distillation_losses(teacher, student, config)
        self.assertEqual(float(losses["total_kd_loss"]), 0.0)

    def test_model_prediction_unchanged_by_valid_steps_field(self):
        torch.manual_seed(4)
        model = AffectiveModel(text_mode="bert").eval()
        tokens = torch.zeros(2, 50, dtype=torch.long)
        lengths = torch.tensor([4, 3])
        text = torch.randn(2, 50, 768)
        audio = torch.randn(2, 50, 74)
        vision = torch.randn(2, 50, 35)
        mask = torch.zeros(2, 50, dtype=torch.bool)
        mask[:, :4] = True
        output = model(tokens, lengths, audio, vision, mask, mask, mask, text)
        self.assertTrue(torch.allclose(output["modality_weights"].sum(dim=-1),
                                       output["valid_steps"].to(torch.float32)))
        self.assertTrue(torch.isfinite(output["logits"]).all())


if __name__ == "__main__":
    unittest.main()
