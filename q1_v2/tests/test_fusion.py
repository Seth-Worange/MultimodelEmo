from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from q1_v2.alignment import WordAlignment
from q1_v2.audio_features import AudioFeatureResult
from q1_v2.data_loader import split_original_words
from q1_v2.fusion_alignment import build_fused_sample, pad_word_sequences, validate_fused_output
from q1_v2.io_utils import write_json
from q1_v2.text_features import TextFeatureResult
from q1_v2.visual_features import VisualFeatureResult


class FusionTests(unittest.TestCase):
    def _features(self):
        words = split_original_words("A test")
        align = [
            WordAlignment("id", 0, "A", "a", 0.0, 0.2, 1, ""),
            WordAlignment("id", 1, "test", "test", np.nan, np.nan, 0, "missing"),
        ]
        text = TextFeatureResult(np.ones((2, 3), np.float32), np.ones(2, np.uint8), [], 3, 2, "mock", None, "mean", [])
        audio = AudioFeatureResult(
            np.ones((2, 2), np.float32), np.asarray([0.05, 0.15]), np.asarray([0, 1]),
            np.asarray([1, 0], np.uint8), ["rms", "f0"], np.ones((2, 4), np.float32),
            np.asarray([1, 0], np.uint8), [[0, 1], []], np.asarray([2, 0]), np.asarray([1, 0]),
            np.asarray([[2, 1], [0, 0]]), ["", "word_time_missing"], 16000, 400, 160,
        )
        visual = VisualFeatureResult(
            np.ones((1, 2), np.float32), np.asarray([0.1]), np.asarray([5]), np.asarray([0.0]),
            np.asarray([0.1]), np.asarray([0]), np.asarray([1], np.uint8), np.asarray([1]),
            np.asarray([0], np.uint8), ["single_face_speaker_unverified"], np.ones((2, 4), np.float32),
            np.asarray([1, 0], np.uint8), np.asarray([0, 0], np.uint8), np.asarray([0, 0], np.uint8),
            [[0], []], [[5], []], np.asarray([1, 0]), np.asarray([1, 0]),
            ["face_features_observed_single_face_speaker_unverified", "word_time_missing"],
            [1], ["_neutral"], 5.0,
        )
        return words, align, text, audio, visual

    def test_output_dimensions_and_sample_id_integrity(self) -> None:
        words, align, text, audio, visual = self._features()
        path = Path(__file__).parent / "_test_output"
        path.mkdir(exist_ok=False)
        try:
            fused = build_fused_sample("id", words, align, text, audio, visual, output_dir=path)
            write_json(path / "_SUCCESS.json", {"sample_id": "id"})
            self.assertEqual(fused.text_features.shape, (2, 3))
            self.assertEqual(fused.audio_word_features.shape, (2, 4))
            valid, errors = validate_fused_output(path, "id")
            self.assertTrue(valid, errors)
        finally:
            for child in path.iterdir():
                child.unlink()
            path.rmdir()

    def test_padding_does_not_truncate(self) -> None:
        sequences = [np.ones((2, 3)), np.ones((4, 3)) * 2]
        masks = [np.asarray([1, 1]), np.asarray([1, 1, 1, 0])]
        padded, mask = pad_word_sequences(sequences, masks)
        self.assertEqual(padded.shape, (2, 4, 3))
        self.assertEqual(mask.tolist(), [[1, 1, 0, 0], [1, 1, 1, 0]])
        self.assertTrue(np.all(padded[1, 3] == 2))


if __name__ == "__main__":
    unittest.main()
