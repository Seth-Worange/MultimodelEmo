"""融合表征监督对比损失检查。"""

import unittest

import torch

from scripts.train import supervised_contrastive_loss


class SupervisedContrastiveTest(unittest.TestCase):
    def test_paired_views_are_positive(self):
        labels = torch.tensor([0, 1])
        separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        mixed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        good = supervised_contrastive_loss(separated, separated, labels, 0.1)
        bad = supervised_contrastive_loss(mixed, mixed, labels, 0.1)
        self.assertLess(good.item(), bad.item())
        good.backward()
        self.assertTrue(torch.isfinite(separated.grad).all())
