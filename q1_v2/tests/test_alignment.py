from __future__ import annotations

import math
import unittest
from pathlib import Path

from q1_v2.alignment import MFAWord, normalize_word, parse_mfa_json, remap_mfa_words
from q1_v2.io_utils import write_json
from q1_v2.data_loader import split_original_words


class AlignmentTests(unittest.TestCase):
    def test_curly_apostrophe_normalizes_for_matching(self) -> None:
        self.assertEqual(normalize_word("They\u2019ve"), "they've")

    def test_failed_alignment_preserves_every_original_word(self) -> None:
        original = split_original_words("Hello, mystery world!")
        aligned = [MFAWord("hello", 0.1, 0.3), MFAWord("world", 0.6, 0.9)]
        result = remap_mfa_words(
            "sample", original, aligned,
            wav_origin_media_s=0.0, timeline_origin_media_s=0.0, duration_s=1.0,
        )
        self.assertEqual(len(result), 3)
        self.assertEqual([word.original_word for word in result], ["Hello,", "mystery", "world!"])
        self.assertEqual([word.alignment_mask for word in result], [1, 0, 1])
        self.assertTrue(math.isnan(result[1].start_s))

    def test_punctuation_only_word_is_retained(self) -> None:
        original = split_original_words("yes ... no")
        aligned = [MFAWord("yes", 0.0, 0.2), MFAWord("no", 0.3, 0.5)]
        result = remap_mfa_words(
            "sample", original, aligned,
            wav_origin_media_s=0.0, timeline_origin_media_s=0.0, duration_s=1.0,
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1].failure_reason, "punctuation_only_no_spoken_form")

    def test_out_of_bounds_overlap_and_zero_duration_are_rejected(self) -> None:
        original = split_original_words("one two three")
        aligned = [
            MFAWord("one", -0.2, 0.2),
            MFAWord("two", 0.3, 0.3),
            MFAWord("three", 0.1, 0.5),
        ]
        result = remap_mfa_words(
            "sample", original, aligned,
            wav_origin_media_s=0.0, timeline_origin_media_s=0.0, duration_s=0.4,
        )
        reasons = " | ".join(word.failure_reason for word in result)
        self.assertIn("interval_out_of_bounds", reasons)
        self.assertIn("zero_or_negative_duration", reasons)
        self.assertIn("non_monotonic_interval", reasons)

    def test_mock_mfa_json_parser_is_explicitly_synthetic(self) -> None:
        path = Path(__file__).parent / "_synthetic_mfa.json"
        try:
            write_json(path, {
                "tiers": {
                    "speaker - words": {"entries": [[0.0, 0.2, "hello"], [0.2, 0.5, "world"]]},
                    "speaker - phones": {"entries": [[0.0, 0.1, "HH"], [0.1, 0.2, "AH"]]},
                },
                "synthetic_test_only": True,
            })
            words = parse_mfa_json(path)
            self.assertEqual([word.label for word in words], ["hello", "world"])
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
