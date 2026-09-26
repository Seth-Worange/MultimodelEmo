"""问题三骨干的初始化与真实缺失边界。"""

import unittest

import torch

from model.q3_residual import Q3ResidualModel
from model.q3_temporal import Q3TemporalModel


def sample():
    tokens = torch.zeros(2, 50, dtype=torch.long)
    tokens[:, :8] = 1
    lengths = torch.full((2,), 8)
    audio = torch.randn(2, 50, 74)
    vision = torch.randn(2, 50, 35)
    text = torch.randn(2, 50, 768)
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[:, 1:7] = True
    missing = mask.clone()
    missing[1] = False
    return tokens, lengths, audio, vision, mask, missing, missing, text


class Q3BackboneChecks(unittest.TestCase):
    def test_temporal_model_handles_missing_audio_and_vision(self):
        model = Q3TemporalModel().eval()
        with torch.no_grad():
            output = model(*sample())
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertEqual(output["modality_weights"][1].sum().item(), 0.0)

    def test_residual_starts_from_exact_base_prediction(self):
        config = {"architecture": "fuse", "text_mode": "bert", "dropout": 0.0}
        model = Q3ResidualModel(config, dropout=0.0).eval()
        inputs = sample()
        with torch.no_grad():
            output = model(*inputs)
            base = model.base(*inputs)
        torch.testing.assert_close(output["logits"], base["logits"])
        torch.testing.assert_close(output["sentiment"], base["sentiment"])
        self.assertFalse(any(parameter.requires_grad for parameter in model.base.parameters()))


if __name__ == "__main__":
    unittest.main()
