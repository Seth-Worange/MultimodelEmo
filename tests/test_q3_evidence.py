"""检查上下文闭合与部分时间映射。"""

import unittest
from types import SimpleNamespace

import torch

from scripts.refine_q3_evidence import score_mask
from utils.q3_evidence import context_positions, mapped_event


class Q3EvidenceTest(unittest.TestCase):
    def test_answer_keeps_preceding_question(self):
        words = ["Should", "I", "leave?]", "Absolutely", "not."]
        mapping = {"words": [{"word": word} for word in words],
                   "positions": [None] + [{"word_index": i, "word": word,
                                               "start": i, "end": i + 0.5}
                                              for i, word in enumerate(words)]}
        self.assertEqual(context_positions(mapping, 4, 6), [1, 2, 3])

    def test_partial_mapping_does_not_invent_time(self):
        mapping = {"positions": [None, {"start": 1.2, "end": 1.5}]}
        self.assertEqual(mapped_event(mapping, [0, 1]),
                         {"start_seconds": 1.2, "end_seconds": 1.5,
                          "mapping_status": "partial"})
        self.assertEqual(mapped_event(mapping, [0])["mapping_status"], "unmapped")

    def test_early_mask_preserves_bert_special_tokens(self):
        class Encoder:
            def __call__(self, input_ids, attention_mask, token_type_ids):
                self.ids = input_ids.clone()
                return SimpleNamespace(last_hidden_state=torch.zeros(1, 4, 768))

        class Model:
            def __call__(self, *args):
                return {"logits": torch.zeros(1, 3)}

        tokens = torch.tensor([[[101, 12, 13, 102], [1, 1, 1, 1], [0, 0, 0, 0]]])
        batch = {"tokens": tokens, "lengths": torch.tensor([4]),
                 "audio": torch.zeros(1, 4, 1), "vision": torch.zeros(1, 4, 1),
                 "text_mask": torch.tensor([[False, True, True, False]]),
                 "audio_mask": torch.ones(1, 4, dtype=torch.bool),
                 "vision_mask": torch.ones(1, 4, dtype=torch.bool)}
        encoder = Encoder()
        score_mask(Model(), batch, encoder, "text", torch.tensor([[False, True, False, False]]))
        self.assertEqual(encoder.ids[0].tolist(), [101, 12, 0, 102])


if __name__ == "__main__":
    unittest.main()
