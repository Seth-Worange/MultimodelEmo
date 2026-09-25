"""文本残差头对缺失掩码的行为检查。"""
import unittest

import torch

from model.fuse_net import FactorizedAffectiveModel


class TextResidualTest(unittest.TestCase):
    def test_missing_text_cannot_contribute(self):
        torch.manual_seed(7)
        model = FactorizedAffectiveModel(text_residual=True).eval()
        with torch.no_grad():
            model.text_residual_classifier.weight.fill_(0.1)
            model.text_residual_classifier.bias.fill_(0.2)
        tokens = torch.zeros(2, 50, dtype=torch.long)
        tokens[:, :4] = torch.tensor([101, 200, 300, 102])
        text = torch.randn(2, 50, 768)
        audio = torch.randn(2, 50, 74)
        vision = torch.randn(2, 50, 35)
        active = torch.zeros(2, 50, dtype=torch.bool)
        active[:, 1:3] = True
        missing = torch.zeros_like(active)
        lengths = torch.full((2,), 4)
        with torch.no_grad():
            missing_logits = model(tokens, lengths, audio, vision, missing, active, active, text)["logits"]
            model.text_residual_classifier.weight.zero_()
            model.text_residual_classifier.bias.zero_()
            missing_baseline = model(tokens, lengths, audio, vision, missing, active, active, text)["logits"]
        self.assertTrue(torch.allclose(missing_logits, missing_baseline, atol=1e-6))
