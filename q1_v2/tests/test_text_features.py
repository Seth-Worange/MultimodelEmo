from __future__ import annotations

import unittest

from q1_v2.text_features import _build_model_inputs


class _Transformers5BertTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    @staticmethod
    def num_special_tokens_to_add(pair: bool = False) -> int:
        return 2


class TextFeatureCompatibilityTests(unittest.TestCase):
    def test_transformers5_bert_special_tokens_are_explicit(self) -> None:
        ids, mask, token_types = _build_model_inputs(
            _Transformers5BertTokenizer(), [2027, 1005, 2310], "bert",
        )
        self.assertEqual(ids, [101, 2027, 1005, 2310, 102])
        self.assertEqual(mask, [1, 0, 0, 0, 1])
        self.assertEqual(token_types, [0, 0, 0, 0, 0])

    def test_unknown_transformers5_model_does_not_guess_special_tokens(self) -> None:
        with self.assertRaises(RuntimeError):
            _build_model_inputs(_Transformers5BertTokenizer(), [1], "unknown")


if __name__ == "__main__":
    unittest.main()
