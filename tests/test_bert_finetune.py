import unittest

import numpy as np
import torch

from model.bert_text import align_bert_outputs
from scripts.infer import load_checkpoint_state
from scripts.train import checkpoint_state_dict
from utils.augmentation import mask_batch
from utils.text import mask_full_text_attention, prepare_full_text_inputs


class TinyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        rows = {
            "short": [101, 7, 102],
            "long": [101, *range(10, 68), 102],
        }
        return {"input_ids": [rows[text] for text in texts],
                "token_type_ids": [[0] * len(rows[text]) for text in texts]}


class BertFineTuneChecks(unittest.TestCase):
    def test_compact_checkpoint_keeps_only_trainable_bert_weights(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert_text_encoder = torch.nn.Module()
                self.bert_text_encoder.backbone = torch.nn.Module()
                self.bert_text_encoder.backbone.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
                self.bert_text_encoder.backbone.tuned = torch.nn.Parameter(torch.ones(4))
                self.head = torch.nn.Linear(4, 2)

        model = TinyModel()
        state = checkpoint_state_dict(model, compact_bert=True)
        self.assertNotIn("bert_text_encoder.backbone.frozen", state)
        self.assertEqual(state["bert_text_encoder.backbone.tuned"].dtype, torch.float16)
        model.bert_text_encoder.backbone.tuned.data.zero_()
        load_checkpoint_state(model, state, compact_bert=True)
        self.assertTrue(torch.equal(model.bert_text_encoder.backbone.tuned,
                                    torch.ones(4, dtype=torch.float32)))

    def test_full_sequence_keeps_aligned_prefix_and_pools_tail(self):
        hidden = torch.arange(60, dtype=torch.float32).view(1, 60, 1).requires_grad_()
        attention = torch.ones(1, 60, dtype=torch.long)
        aligned, tail, available = align_bert_outputs(hidden, attention)
        self.assertEqual(aligned.shape, (1, 50, 1))
        self.assertTrue(torch.equal(aligned[0, :49, 0], hidden[0, :49, 0]))
        self.assertEqual(aligned[0, 49, 0].item(), 59.0)
        self.assertEqual(tail.item(), float(hidden[0, 49:59].mean().detach()))
        self.assertTrue(available.item())
        (aligned.sum() + tail.sum()).backward()
        self.assertIsNotNone(hidden.grad)

    def test_text_mask_maps_to_full_bert_attention(self):
        attention = torch.ones(1, 60, dtype=torch.long)
        text_mask = torch.ones(1, 50, dtype=torch.bool)
        text_mask[:, 4:7] = False
        result = mask_full_text_attention(attention, text_mask)
        self.assertFalse(result[0, 4:7].any())
        self.assertTrue(result[0, 1:4].all())
        self.assertFalse(result[0, 49:59].any())
        self.assertTrue(result[0, 0] and result[0, 59])

        text_mask.zero_()
        result = mask_full_text_attention(attention, text_mask)
        self.assertEqual(int(result.sum()), 2)

    def test_local_text_mask_also_hides_unaligned_tail(self):
        tokens = torch.zeros(1, 3, 50, dtype=torch.long)
        tokens[0, 0, :49] = torch.arange(101, 150)
        tokens[0, 0, 0] = 101
        tokens[0, 0, 49] = 102
        tokens[0, 1, :] = 1
        text_mask = torch.ones(1, 50, dtype=torch.bool)
        text_mask[:, 0] = False
        text_mask[:, 49] = False
        batch = {"tokens": tokens, "text_mask": text_mask,
                 "audio": torch.ones(1, 50, 2), "vision": torch.ones(1, 50, 2),
                 "audio_mask": torch.ones(1, 50, dtype=torch.bool),
                 "vision_mask": torch.ones(1, 50, dtype=torch.bool),
                 "bert_input_ids": torch.ones(1, 60, dtype=torch.long),
                 "bert_attention_mask": torch.ones(1, 60, dtype=torch.long),
                 "bert_token_type_ids": torch.zeros(1, 60, dtype=torch.long)}
        masked = mask_batch(batch, seed=7, probability=1.0, pattern="interval",
                            ratios=(0.2,), interval_modalities=("text",))
        self.assertFalse(masked["bert_attention_mask"][0, 49:59].any())
        self.assertTrue(masked["bert_attention_mask"][0, 59])

    def test_full_text_inputs_preserve_prefix_and_add_head_tail(self):
        tokens = np.zeros((2, 3, 50), dtype=np.int64)
        tokens[0, 0, :3] = [101, 7, 102]
        tokens[0, 1, :3] = 1
        long_ids = [101, *range(10, 68), 102]
        tokens[1, 0, :49] = long_ids[:49]
        tokens[1, 0, 49] = 102
        tokens[1, 1, :] = 1
        data = {"tokens": tokens, "raw_text": ["short", "long"], "ids": ["s", "l"]}
        prepared = prepare_full_text_inputs(data, TinyTokenizer(), max_length=52)
        self.assertEqual(prepared["bert_input_ids"].shape, (2, 52))
        long_row = prepared["bert_input_ids"][1, prepared["bert_attention_mask"][1].astype(bool)]
        self.assertEqual(int(long_row[0]), 101)
        self.assertEqual(int(long_row[-1]), 102)
        self.assertTrue(np.array_equal(long_row[1:26], np.arange(10, 35)))
        self.assertEqual(prepared["bert_text_stats"]["n_head_tail_truncated"], 1)


if __name__ == "__main__":
    unittest.main()
