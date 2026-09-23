from __future__ import annotations

import unittest

import numpy as np

from q1_v2.alignment import WordAlignment
from q1_v2.audio_features import aggregate_audio_to_words
from q1_v2.visual_features import aggregate_visual_to_words


def word(index: int, start: float, end: float, mask: int = 1) -> WordAlignment:
    return WordAlignment("s", index, f"w{index}", f"w{index}", start, end, mask, "" if mask else "missing")


class AggregationTests(unittest.TestCase):
    def test_audio_frame_association_and_missing_mask(self) -> None:
        times = np.asarray([0.05, 0.15, 0.25, 0.35])
        features = np.asarray([[1.0, np.nan], [3.0, 100.0], [5.0, 110.0], [7.0, np.nan]], dtype=np.float32)
        f0_valid = np.asarray([0, 1, 1, 0], dtype=np.uint8)
        words = [word(0, 0.0, 0.2), word(1, 0.2, 0.4), word(2, np.nan, np.nan, 0)]
        result = aggregate_audio_to_words(features, times, f0_valid, words)
        self.assertEqual(result[2], [[0, 1], [2, 3], []])
        self.assertEqual(result[1].tolist(), [1, 1, 0])
        self.assertEqual(result[4].tolist(), [1, 1, 0])
        self.assertTrue(np.isnan(result[0][2]).all())

    def test_short_word_no_visual_sample_is_not_face_failure(self) -> None:
        times = np.asarray([0.0, 0.2, 0.4])
        features = np.ones((3, 2), dtype=np.float32)
        detection = np.asarray([1, 0, 1], dtype=np.uint8)
        face_counts = np.asarray([1, 0, 2])
        originals = np.asarray([10, 20, 30])
        words = [word(0, 0.05, 0.10), word(1, 0.19, 0.21), word(2, 0.39, 0.41), word(3, np.nan, np.nan, 0)]
        result = aggregate_visual_to_words(features, times, detection, face_counts, originals, words)
        statuses = result[8]
        self.assertEqual(statuses[0], "no_sampled_frame_in_interval")
        self.assertEqual(statuses[1], "sampled_frames_present_but_all_face_detections_failed")
        self.assertIn("multiple_faces", statuses[2])
        self.assertEqual(statuses[3], "word_time_missing")
        self.assertEqual(result[1].tolist(), [0, 0, 1, 0])

    def test_nearest_frame_is_marked_estimated_not_observed(self) -> None:
        result = aggregate_visual_to_words(
            np.ones((1, 2), dtype=np.float32), np.asarray([0.0]), np.asarray([1]),
            np.asarray([1]), np.asarray([9]), [word(0, 0.04, 0.06)],
            nearest_max_distance_s=0.1,
        )
        self.assertEqual(result[1].tolist(), [0])
        self.assertEqual(result[3].tolist(), [1])
        self.assertEqual(result[6].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
