import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.features_q1 import pool_intervals
from utils.text import encode_text


class ContextEncoder:
    def __call__(self, input_ids, attention_mask, token_type_ids):
        context = (input_ids * attention_mask).sum(dim=1, keepdim=True).float()
        return SimpleNamespace(last_hidden_state=context[:, None].expand(-1, input_ids.shape[1], -1))


class PipelineChecks(unittest.TestCase):
    def test_hidden_text_cannot_reach_context(self):
        tokens = torch.tensor([[[101, 10, 20, 102, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 0]]])
        batch = {"tokens": tokens, "text_mask": torch.tensor([[False, True, True, False, False]])}
        clean = encode_text(batch, ContextEncoder())
        hidden_mask = batch["text_mask"].clone()
        hidden_mask[0, 1] = False
        hidden = encode_text(batch, ContextEncoder(), hidden_mask)
        tokens[0, 0, 1] = 999
        hidden_after_change = encode_text(batch, ContextEncoder(), hidden_mask)
        self.assertFalse(torch.equal(clean, hidden))
        self.assertTrue(torch.equal(hidden, hidden_after_change))

    def test_nearest_face_frame_is_marked(self):
        times = np.array([0.0, 0.2, 0.4])
        features = np.arange(3, dtype=np.float32)[:, None]
        words = [{"start": 0.11, "end": 0.16}]
        _, valid, nearest = pool_intervals(times, features, words, nearest_tolerance=0.1)
        self.assertEqual((valid[0], nearest[0]), (1, 1))


if __name__ == "__main__":
    unittest.main()
