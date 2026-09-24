from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from q1_v2.io_utils import write_json
from q1_v2.submission_export import export_sample


class SubmissionExportTests(unittest.TestCase):
    def test_keeps_words_masks_frame_times_and_traceability_without_wav(self) -> None:
        temporary = Path.cwd() / f".q1_export_test_{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            source = temporary / "source" / "video__1"
            target = temporary / "export" / "video__1"
            source.mkdir(parents=True)
            np.savez_compressed(
                source / "fused_features.npz",
                sample_id=np.asarray("video__1"), original_words=np.asarray(["one", "two"]),
                word_start_s=np.asarray([0.0, np.nan]), word_end_s=np.asarray([.2, np.nan]),
                text_word_features=np.ones((2, 3)), audio_word_features=np.ones((2, 2)),
                visual_word_features=np.ones((2, 2)), text_mask=np.asarray([1, 1]),
                audio_mask=np.asarray([1, 0]), visual_mask=np.asarray([0, 0]),
                alignment_mask=np.asarray([1, 0]), audio_frame_features=np.ones((2, 2)),
                audio_frame_times_s=np.asarray([.05, .15]),
                audio_frame_f0_valid_mask=np.asarray([1, 0]),
                visual_frame_features=np.ones((1, 2)), visual_frame_times_s=np.asarray([.3]),
                visual_original_frame_indices=np.asarray([7]), visual_detection_mask=np.asarray([1]),
                audio_word_frame_counts=np.asarray([2, 0]), visual_word_frame_counts=np.asarray([0, 0]),
            )
            write_json(source / "fused_metadata.json", {"word_count": 2, "audio_word_frame_indices": [[0, 1], []], "visual_word_original_frame_indices": [[], []]})
            write_json(source / "_SUCCESS.json", {"status": "success"})
            write_json(source / "sample_input.json", {"video_id": "video", "clip_id": "1", "video_sha256": "abc"})
            write_json(source / "mfa_raw.json", {"tiers": {}})
            (source / "word_alignment.csv").write_text("word_index\n0\n1\n", encoding="utf-8")
            (source / "audio_mfa_16k_mono.wav").write_bytes(b"large waveform placeholder")
            report = export_sample(source, target)
            self.assertEqual(report["word_count"], 2)
            self.assertFalse((target / "audio_mfa_16k_mono.wav").exists())
            with np.load(target / "fused_features.npz", allow_pickle=False) as arrays:
                self.assertEqual(len(arrays["original_words"]), 2)
                self.assertEqual(arrays["visual_original_frame_indices"].tolist(), [7])
                self.assertEqual(arrays["alignment_mask"].tolist(), [1, 0])
        finally:
            shutil.rmtree(temporary)


if __name__ == "__main__":
    unittest.main()
