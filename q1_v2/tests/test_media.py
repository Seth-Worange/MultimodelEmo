from __future__ import annotations

import math
import unittest
from fractions import Fraction

from q1_v2.media import (
    VideoFrameTiming,
    media_to_sample_time,
    pts_to_seconds,
    sample_to_media_time,
    sample_to_wav_time,
    wav_to_sample_time,
)
from q1_v2.visual_features import choose_sampled_frames


class MediaTimeTests(unittest.TestCase):
    def test_time_base_round_trip(self) -> None:
        self.assertAlmostEqual(pts_to_seconds(512, Fraction(1, 15360)), 1 / 30)
        sample = media_to_sample_time(3.25, 2.0)
        self.assertAlmostEqual(sample, 1.25)
        self.assertAlmostEqual(sample_to_media_time(sample, 2.0), 3.25)
        mapped = wav_to_sample_time(0.5, 2.2, 2.0)
        self.assertAlmostEqual(mapped, 0.7)
        self.assertAlmostEqual(sample_to_wav_time(mapped, 2.2, 2.0), 0.5)
        self.assertTrue(math.isnan(pts_to_seconds(None, Fraction(1, 25))))

    def test_sampling_uses_pts_not_index_over_fps(self) -> None:
        times = [0.0, 0.03, 0.09, 0.21, 0.39, 0.42]
        frames = [
            VideoFrameTiming(i, i, "1/100", value, value, False, True)
            for i, value in enumerate(times)
        ]
        selected = choose_sampled_frames(frames, 5.0)
        self.assertEqual([item["frame_index"] for item in selected], [0, 3, 4])
        self.assertEqual([round(float(item["sample_time_s"]), 2) for item in selected], [0.0, 0.21, 0.39])

    def test_non_monotonic_video_timestamps_are_rejected(self) -> None:
        frames = [
            VideoFrameTiming(0, 0, "1/100", 0.2, 0.2, False, True),
            VideoFrameTiming(1, 1, "1/100", 0.1, 0.1, False, True),
        ]
        with self.assertRaisesRegex(ValueError, "not strictly increasing"):
            choose_sampled_frames(frames, 5.0)


if __name__ == "__main__":
    unittest.main()
