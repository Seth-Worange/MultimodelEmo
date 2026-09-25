"""中性与极性层级分类头检查。"""
import unittest

import torch

from model.fuse_net import FactorizedAffectiveModel


class HierarchicalHeadTest(unittest.TestCase):
    def test_probabilities_and_gradients(self):
        torch.manual_seed(11)
        model = FactorizedAffectiveModel(hierarchical_head=True).eval()
        tokens = torch.zeros(3, 50, dtype=torch.long)
        tokens[:, :4] = torch.tensor([101, 200, 300, 102])
        active = torch.zeros(3, 50, dtype=torch.bool)
        active[:, 1:3] = True
        result = model(
            tokens, torch.full((3,), 4), torch.randn(3, 50, 74),
            torch.randn(3, 50, 35), active, active, active,
            torch.randn(3, 50, 768),
        )
        self.assertTrue(torch.allclose(result["logits"].exp().sum(-1), torch.ones(3), atol=1e-5))
        torch.nn.functional.cross_entropy(result["logits"], torch.tensor([0, 1, 2])).backward()
        self.assertGreater(model.neutral_classifier.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.polarity_classifier.weight.grad.abs().sum().item(), 0)

    def test_text_polarity_falls_back_when_text_is_missing(self):
        torch.manual_seed(12)
        model = FactorizedAffectiveModel(hierarchical_head=True,
                                         text_polarity_head=True, dropout=0.0).eval()
        tokens = torch.zeros(2, 50, dtype=torch.long)
        tokens[:, :4] = torch.tensor([101, 200, 300, 102])
        present = torch.zeros(2, 50, dtype=torch.bool)
        present[:, 1:3] = True
        absent = torch.zeros_like(present)
        audio = torch.randn(2, 50, 74)
        vision = torch.randn(2, 50, 35)
        teacher = torch.randn(2, 50, 768)
        full = model(tokens, torch.full((2,), 4), audio, vision,
                     present, present, present, teacher)
        no_text = model(tokens, torch.full((2,), 4), audio, vision,
                        absent, present, present, teacher)
        self.assertTrue(torch.isfinite(no_text["logits"]).all())
        self.assertFalse(torch.allclose(full["logits"], no_text["logits"]))
        torch.nn.functional.cross_entropy(full["logits"], torch.tensor([0, 2])).backward()
        self.assertGreater(model.polarity_classifier.weight.grad.abs().sum().item(), 0)
