from __future__ import annotations

import unittest

from q1_v2.quality import _aggregate


class QualitySemanticsTests(unittest.TestCase):
    def test_unexecuted_modalities_are_not_reported_as_zero(self) -> None:
        row = {
            "sample_id": "video__1", "read_status": "ok",
            "processing_status": "not_processed", "original_word_count": 3,
            "alignment_status": "not_evaluated", "text_status": "not_evaluated",
            "audio_status": "not_evaluated", "visual_status": "not_evaluated",
        }
        summary = _aggregate([row], None)
        self.assertIsNone(summary["automatic_alignment_coverage_not_accuracy"])
        self.assertIsNone(summary["text_word_valid_rate"])
        self.assertIsNone(summary["audio_word_valid_rate"])
        self.assertIsNone(summary["visual_word_valid_rate"])

    def test_failed_attempt_is_distinct_from_later_unexecuted_stages(self) -> None:
        row = {
            "sample_id": "video__1", "read_status": "ok",
            "processing_status": "failed", "original_word_count": 3,
            "alignment_status": "failed", "aligned_word_count": 0,
            "text_status": "not_evaluated", "audio_status": "not_evaluated",
            "visual_status": "not_evaluated",
        }
        summary = _aggregate([row], None)
        self.assertEqual(summary["alignment_evaluated_sample_count"], 1)
        self.assertEqual(summary["automatic_alignment_coverage_not_accuracy"], 0.0)
        self.assertIsNone(summary["text_word_valid_rate"])


if __name__ == "__main__":
    unittest.main()
