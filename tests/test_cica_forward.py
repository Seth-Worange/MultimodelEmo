"""CICA 融合前向与独立评估路径检查。"""

import unittest

import torch

from model.cica_net import CICAAffectiveModel


class CicaForwardTest(unittest.TestCase):
    def test_fusion_receives_sequence_lengths(self):
        model = CICAAffectiveModel(dropout=0.0).eval()
        tokens = torch.zeros(2, 50, dtype=torch.long)
        tokens[:, 0] = 101
        tokens[:, 1:6] = 200
        lengths = torch.tensor([4, 7])
        text_mask = torch.zeros(2, 50, dtype=torch.bool)
        text_mask[:, 1:6] = True
        audio_mask = text_mask.clone()
        vision_mask = text_mask.clone()
        inputs = (tokens, lengths, torch.zeros(2, 50, 74), torch.zeros(2, 50, 35),
                  text_mask, audio_mask, vision_mask, torch.zeros(2, 50, 768))
        with torch.no_grad():
            direct = model(*inputs)
            frozen = model.forward_fusion(*inputs)
        self.assertEqual(direct["logits"].shape, (2, 3))
        self.assertTrue(torch.allclose(direct["logits"], frozen["logits"]))


if __name__ == "__main__":
    unittest.main()
