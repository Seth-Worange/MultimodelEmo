"""验证缺失样本可复现，以及中性决策不会改动分类。"""

import unittest
from types import SimpleNamespace

import torch

from scripts.infer import ModelEnsemble, NeutralZeroModel
from scripts.train import build_fixed_view, evaluation_seed, make_batch
from utils.alignment import alignment_coverage, mapped_word_text
from model.fuse_net import FactorizedAffectiveModel
from utils.data import AUDIO_DIM, SEQ_LEN, VISION_DIM


class FixedModel(torch.nn.Module):
    text_mode = "bert"
    regression_mode = "soft"

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self):
        probabilities = self.logits.softmax(-1)
        magnitude = torch.ones(len(self.logits))
        return {"logits": self.logits, "magnitude": magnitude,
                "sentiment": magnitude * (probabilities[:, 2] - probabilities[:, 0])}


class EvaluationChecks(unittest.TestCase):
    def test_train_only_stats_and_packed_padding_invariance(self):
        torch.manual_seed(3)
        model = FactorizedAffectiveModel(text_mode="bert", hidden_dim=4, dropout=0.0,
                                         normalize_inputs=True, pack_aligned_grus=True).eval()
        count, steps = 2, SEQ_LEN
        audio = torch.randn(count, steps, AUDIO_DIM)
        vision = torch.randn(count, steps, VISION_DIM)
        text = torch.randn(count, steps, 768)
        lengths = torch.tensor([7, 11])
        tokens = torch.zeros(count, steps, dtype=torch.long)
        tokens[:, 0] = 101
        tokens[:, 1:6] = torch.tensor([12, 13, 14, 102, 0])
        tokens[1, 1:10] = torch.tensor([12, 13, 14, 15, 16, 17, 18, 102, 0])
        text_mask = tokens.gt(0) & tokens.ne(101) & tokens.ne(102)
        audio_mask = torch.zeros(count, steps, dtype=torch.bool)
        vision_mask = torch.zeros_like(audio_mask)
        audio_mask[0, 1:7] = True
        audio_mask[1, 1:11] = True
        vision_mask[0, 1:7] = True
        vision_mask[1, 1:11] = True
        # 内部缺口保留在原位置，padding的大值不会参与统计。
        audio_mask[:, 3] = False
        vision_mask[:, 4] = False
        model.fit_input_stats({"audio": audio, "vision": vision,
                               "audio_mask": audio_mask, "vision_mask": vision_mask})
        expected_audio_mean = audio[audio_mask].mean(0)
        self.assertTrue(torch.allclose(model.audio_mean, expected_audio_mean, atol=1e-6))
        audio_alt, vision_alt, text_alt = audio.clone(), vision.clone(), text.clone()
        for row, length in enumerate(lengths.tolist()):
            audio_alt[row, length:] = 1e4
            vision_alt[row, length:] = -1e4
            text_alt[row, length:] = 1e4
        args = (tokens, lengths, audio, vision, text_mask, audio_mask, vision_mask, text)
        alt = (tokens, lengths, audio_alt, vision_alt, text_mask, audio_mask, vision_mask, text_alt)
        with torch.no_grad():
            first, second = model(*args), model(*alt)
        self.assertTrue(torch.allclose(first["logits"], second["logits"], atol=1e-6))
        self.assertTrue(torch.allclose(first["sentiment"], second["sentiment"], atol=1e-6))

    def test_token_match_does_not_imply_full_coverage(self):
        first = {"word_index": 0, "word": "very", "start": 0.0, "end": 0.3}
        second = {"word_index": 1, "word": "very", "start": 0.3, "end": 0.6}
        mapping = {"token_match_fraction": 1.0, "positions": [None, first, first, second],
                   "words": [first, second, {"word": "good", "end": 1.0}]}
        coverage = alignment_coverage(mapping)
        self.assertEqual(coverage["evidence_scope"], "partial_aligned_transcript")
        self.assertEqual(coverage["aligned_word_coverage"], 2 / 3)
        self.assertEqual(coverage["mapped_time_end"], 0.6)
        self.assertEqual(mapped_word_text([first, first, second]), "very very")

    def test_fixed_masks_ignore_batch_size_and_order(self):
        count, length = 9, 20
        tokens = torch.ones(count, 3, length, dtype=torch.long)
        tokens[:, 0] = torch.arange(10, 10 + length)
        data = {"tokens": tokens, "ids": [f"sample-{i}" for i in range(count)],
                "lengths": torch.full((count,), length),
                "audio": torch.ones(count, length, 3), "vision": torch.ones(count, length, 2)}
        for modality in ("text", "audio", "vision"):
            data[f"{modality}_mask"] = torch.ones(count, length, dtype=torch.bool)
        for view in ("clean", "local", "whole", "interval"):
            expected = build_fixed_view(data, view, 2026)
            for order in (torch.arange(count), torch.arange(count - 1, -1, -1)):
                for size in (1, 4, 9):
                    for indices in order.split(size):
                        actual = build_fixed_view(make_batch(data, indices), view, 2026)
                        for key in ("tokens", "audio", "vision", "text_mask", "audio_mask", "vision_mask"):
                            self.assertTrue(torch.equal(actual[key], expected[key][indices]), (view, size, key))
        self.assertTrue(torch.all(data["audio"] == 1))
        self.assertTrue(torch.all(data["text_mask"]))
        first = build_fixed_view(data, "interval", 2026)
        second = build_fixed_view(data, "interval", 2027)
        self.assertTrue(any(not torch.equal(first[f"{m}_mask"], second[f"{m}_mask"])
                            for m in ("text", "audio", "vision")))

    def test_evaluation_seed_is_separate(self):
        for seed in (2026, 2027):
            args = SimpleNamespace(seed=seed, evaluation_seed=None, evaluation_protocol="sample_v2")
            self.assertEqual(evaluation_seed(args), 2026)
        args.evaluation_protocol = "legacy_batch"
        self.assertEqual(evaluation_seed(args), 11028)

    def test_neutral_zero_after_ensemble(self):
        first = FixedModel([[3, 0, 1], [0, 2, 3], [0, 1, 3]])
        second = FixedModel([[4, 0, 1], [0, 5, 1], [1, 0, 3]])
        ensemble = ModelEnsemble([first, second])
        original = ensemble()
        actual = NeutralZeroModel(ensemble)()
        self.assertTrue(torch.equal(actual["logits"], original["logits"]))
        self.assertTrue(torch.equal(actual["raw_sentiment"], original["sentiment"]))
        self.assertEqual(float(actual["sentiment"][1]), 0.0)
        self.assertNotEqual(float(original["sentiment"][1]), 0.0)
        self.assertTrue(torch.equal(actual["sentiment"][[0, 2]], original["sentiment"][[0, 2]]))


if __name__ == "__main__":
    unittest.main()
