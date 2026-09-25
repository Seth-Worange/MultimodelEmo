"""非中性样本的强度监督检查。"""
import unittest

import torch
from torch.nn import functional as F

from scripts.train import supervised_loss


class MagnitudeSupervisionTest(unittest.TestCase):
    def test_only_non_neutral_samples_supervise_magnitude(self):
        magnitude = torch.tensor([0.2, 2.0, 0.5], requires_grad=True)
        output = {
            "logits": torch.zeros(3, 3),
            "sentiment": torch.zeros(3),
            "magnitude": magnitude,
        }
        batch = {
            "classes": torch.tensor([0, 1, 2]),
            "sentiment": torch.tensor([-1.0, 0.0, 2.0]),
        }
        base = supervised_loss(output, batch)
        weighted = supervised_loss(output, batch, magnitude_weight=0.3)
        expected = 0.3 * F.smooth_l1_loss(
            magnitude[[0, 2]], batch["sentiment"][[0, 2]].abs())
        self.assertTrue(torch.allclose(weighted - base, expected))
        weighted.backward()
        self.assertEqual(magnitude.grad[1].item(), 0.0)
        self.assertGreater(magnitude.grad[0].abs().item(), 0)
        self.assertGreater(magnitude.grad[2].abs().item(), 0)
