"""轻量晚期融合的前向、反向与文本缺失检查。"""

import unittest

import torch

from model.complementary_net import ComplementaryAffectiveModel
from scripts.infer import build_from_config


class ComplementaryNetTest(unittest.TestCase):
    def test_forward_and_missing_text(self):
        torch.manual_seed(13)
        model = build_from_config({"architecture": "complementary", "dropout": 0.0}).eval()
        self.assertIsInstance(model, ComplementaryAffectiveModel)
        tokens = torch.zeros(2, 50, dtype=torch.long)
        tokens[:, :4] = torch.tensor([101, 200, 300, 102])
        active = torch.zeros(2, 50, dtype=torch.bool)
        active[:, 1:3] = True
        absent = torch.zeros_like(active)
        inputs = (tokens, torch.full((2,), 4), torch.randn(2, 50, 74),
                  torch.randn(2, 50, 35), absent, active, active)
        text = torch.randn(2, 50, 768)
        one = model(*inputs, text)
        two = model(*inputs, text + 10)
        self.assertTrue(torch.allclose(one["logits"], two["logits"], atol=1e-6))
        self.assertEqual(one["logits"].shape, (2, 3))
        one["sentiment"].sum().backward()
        self.assertGreater(model.regressor.weight.grad.abs().sum().item(), 0)
