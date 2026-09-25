"""语境对照组、序数强度与切片加权目标的检查。"""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train import (context_ranking_loss, ordinal_intensity_loss, ordinal_thresholds,
                           supervised_loss)
from utils.slices import (mine_context_pairs, negation_family_mask, negation_tags,
                          sample_weights, slice_masks)

TEXTS = [
    'If I blow it at the team exercise, should I kiss my chances of cheering "GO BLUE" goodbye?] Absolutely not.',
    "It's definitely not worth the money.",
    "There is nothing wrong with that.",
    "The movie was great.",
]
CLASSES = np.array([2, 0, 1, 2])
SENTIMENT = np.array([1.0, -2.0, 0.0, 2.0], dtype=np.float32)


class SliceMiningTest(unittest.TestCase):
    def test_negation_families_detected(self):
        self.assertEqual(negation_tags(TEXTS), ["answer_neg", "answer_neg", "neg_idiom", None])
        self.assertEqual(negation_family_mask(TEXTS).tolist(), [True, True, True, False])

    def test_context_pairs_link_opposite_labels(self):
        pairs = mine_context_pairs(TEXTS, CLASSES, SENTIMENT)
        self.assertIn((0, 1), pairs)
        for high, low in pairs:
            self.assertGreaterEqual(SENTIMENT[high], SENTIMENT[low])

    def test_sample_weights_boost_slices(self):
        weights = sample_weights(TEXTS, SENTIMENT, negation_boost=1.0, boundary_boost=0.5)
        self.assertEqual(weights.tolist(), [2.0, 2.0, 2.5, 1.0])
        masks = slice_masks(TEXTS, CLASSES, SENTIMENT)
        self.assertEqual(masks["boundary"].tolist(), [False, False, True, False])
        self.assertEqual(masks["strong"].tolist(), [False, True, False, True])


class OrdinalObjectiveTest(unittest.TestCase):
    def test_thresholds_are_half_integers(self):
        self.assertTrue(torch.allclose(ordinal_thresholds(7),
                                       torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])))
        self.assertTrue(torch.allclose(ordinal_thresholds(3), torch.tensor([-0.5, 0.5])))
        with self.assertRaises(ValueError):
            ordinal_thresholds(4)

    def test_loss_supervises_cumulative_order(self):
        output = {"ordinal_logits": torch.zeros(2, 6, requires_grad=True)}
        batch = {"sentiment": torch.tensor([1.0, -2.0])}
        loss = ordinal_intensity_loss(output, batch, 7)
        targets = torch.tensor([[1., 1., 1., 1., 0., 0.], [1., 0., 0., 0., 0., 0.]])
        expected = F.binary_cross_entropy_with_logits(output["ordinal_logits"], targets)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertIsNotNone(output["ordinal_logits"].grad)

    def test_disabled_when_no_head(self):
        output = {"logits": torch.zeros(2, 3), "sentiment": torch.zeros(2)}
        batch = {"classes": torch.tensor([0, 2]), "sentiment": torch.tensor([-1.0, 1.0])}
        base = supervised_loss(output, batch)
        self.assertTrue(torch.allclose(
            supervised_loss(output, batch, ordinal_bins=7, ordinal_weight=1.0), base))


class ContextRankingTest(unittest.TestCase):
    def test_satisfied_pair_has_zero_loss(self):
        pred = torch.tensor([1.5, -1.5])
        target = torch.tensor([2.0, -2.0])
        self.assertEqual(context_ranking_loss(pred, target, margin=1.0).item(), 0.0)

    def test_wrong_order_penalised(self):
        pred = torch.tensor([-2.0, 2.0], requires_grad=True)
        target = torch.tensor([2.0, -2.0])
        loss = context_ranking_loss(pred, target, margin=1.0)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertGreater(pred.grad[0].abs().item(), 0)


class WeightedSupervisionTest(unittest.TestCase):
    def test_weighted_mean_matches_manual(self):
        output = {"logits": torch.zeros(3, 3, requires_grad=True),
                  "sentiment": torch.tensor([0.2, -0.4, 0.9], requires_grad=True)}
        batch = {"classes": torch.tensor([1, 0, 2]), "sentiment": torch.tensor([0., -2., 1.])}
        weights = torch.tensor([2.0, 1.0, 1.5])
        loss = supervised_loss(output, batch, sample_weights=weights)
        per_sample = (F.cross_entropy(output["logits"], batch["classes"], reduction="none")
                      + F.smooth_l1_loss(output["sentiment"], batch["sentiment"], reduction="none"))
        expected = (per_sample * weights).sum() / weights.sum()
        self.assertTrue(torch.allclose(loss, expected))
        unit = supervised_loss(output, batch, sample_weights=torch.ones(3))
        self.assertTrue(torch.allclose(unit, supervised_loss(output, batch)))


class OrdinalHeadTest(unittest.TestCase):
    def test_fuse_model_exposes_ordinal_logits(self):
        from model.fuse_net import FactorizedAffectiveModel
        from utils.data import AUDIO_DIM, SEQ_LEN, TEXT_DIM, VISION_DIM
        model = FactorizedAffectiveModel(text_mode="bert", dropout=0.0, ordinal_bins=7)
        batch = 2
        tokens = torch.zeros(batch, SEQ_LEN, dtype=torch.long)
        tokens[:, :4] = torch.tensor([101, 100, 200, 102])
        mask = torch.zeros(batch, SEQ_LEN, dtype=torch.bool)
        mask[:, :10] = True
        out = model(tokens, torch.full((batch,), 10), torch.randn(batch, SEQ_LEN, AUDIO_DIM),
                    torch.randn(batch, SEQ_LEN, VISION_DIM), mask, mask, mask,
                    torch.randn(batch, SEQ_LEN, TEXT_DIM))
        self.assertEqual(tuple(out["ordinal_logits"].shape), (batch, 6))
        # 旧结构缺省关闭序数头，加载历史检查点不受影响。
        legacy = FactorizedAffectiveModel(text_mode="bert", dropout=0.0)
        self.assertIsNone(legacy(tokens, torch.full((batch,), 10), torch.randn(batch, SEQ_LEN, AUDIO_DIM),
                                 torch.randn(batch, SEQ_LEN, VISION_DIM), mask, mask, mask,
                                 torch.randn(batch, SEQ_LEN, TEXT_DIM))["ordinal_logits"])

    def test_even_bins_rejected(self):
        from model.fuse_net import FactorizedAffectiveModel
        with self.assertRaises(ValueError):
            FactorizedAffectiveModel(text_mode="bert", ordinal_bins=4)


if __name__ == "__main__":
    unittest.main()
