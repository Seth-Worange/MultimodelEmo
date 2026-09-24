from __future__ import annotations

import unittest
import shutil
import uuid
from pathlib import Path

import numpy as np

from q1_v2.io_utils import write_json
from q1_v2.quality import _aggregate, word_review_rows


class QualitySemanticsTests(unittest.TestCase):
    def test_failed_alignment_lists_each_original_word_without_claiming_other_modalities(self) -> None:
        sample_dir = Path.cwd() / f".q1_failed_review_test_{uuid.uuid4().hex}"
        sample_dir.mkdir()
        try:
            write_json(sample_dir / "sample_input.json", {
                "sample_id": "video__1", "original_words": ["one", "two"],
            })
            write_json(sample_dir / "_FAILED.json", {
                "failure_stage": "alignment", "error": "beam exhausted",
            })
            rows = word_review_rows(sample_dir)
            self.assertEqual([row["original_word"] for row in rows], ["one", "two"])
            self.assertTrue(all(row["alignment_mask"] == 0 for row in rows))
            self.assertTrue(all(row["visual_mask"] == "" for row in rows))
        finally:
            shutil.rmtree(sample_dir)

    def test_word_review_reports_unknown_and_no_frame_without_sample_threshold(self) -> None:
        sample_dir = Path.cwd() / f".q1_review_test_{uuid.uuid4().hex}"
        sample_dir.mkdir()
        try:
            (sample_dir / "transcript_original.txt").write_text("Key Polymer brings", encoding="utf-8")
            write_json(sample_dir / "mfa_raw.json", {"tiers": {"words": {"entries": [
                [0, .1, "key"], [.1, .4, "<unk>"], [.4, .8, "brings"],
            ]}}})
            write_json(sample_dir / "fused_metadata.json", {
                "visual_word_status": ["observed", "word_time_missing", "no_sampled_frame_in_interval"],
            })
            np.savez_compressed(
                sample_dir / "fused_features.npz",
                sample_id=np.asarray("video__1"),
                original_words=np.asarray(["Key", "Polymer", "brings"]),
                word_start_s=np.asarray([0.0, np.nan, .4]),
                word_end_s=np.asarray([.1, np.nan, .8]),
                alignment_mask=np.asarray([1, 0, 1]),
                audio_mask=np.asarray([1, 0, 1]),
                visual_mask=np.asarray([1, 0, 0]),
                visual_word_frame_counts=np.asarray([1, 0, 0]),
            )
            rows = word_review_rows(sample_dir)
            self.assertEqual([row["word_index"] for row in rows], [1, 2])
            self.assertIn("mfa_unknown", rows[0]["issue_type"])
            self.assertIn("word_time_missing", rows[0]["issue_type"])
            self.assertEqual(rows[0]["start_s"], "")
            self.assertIn("no_video_frame_in_interval", rows[1]["issue_type"])
        finally:
            shutil.rmtree(sample_dir)

    def test_face_failure_and_no_sample_are_distinct_word_issues(self) -> None:
        sample_dir = Path.cwd() / f".q1_visual_review_test_{uuid.uuid4().hex}"
        sample_dir.mkdir()
        try:
            write_json(sample_dir / "fused_metadata.json", {
                "visual_word_status": [
                    "sampled_frames_present_but_all_face_detections_failed",
                    "no_sampled_frame_in_interval",
                ],
            })
            np.savez_compressed(
                sample_dir / "fused_features.npz",
                sample_id=np.asarray("video__1"), original_words=np.asarray(["one", "two"]),
                word_start_s=np.asarray([0.0, .2]), word_end_s=np.asarray([.2, .4]),
                alignment_mask=np.asarray([1, 1]), audio_mask=np.asarray([1, 1]),
                visual_mask=np.asarray([0, 0]), visual_word_frame_counts=np.asarray([1, 0]),
            )
            rows = word_review_rows(sample_dir)
            self.assertEqual(rows[0]["issue_type"], "sampled_frames_without_valid_face")
            self.assertEqual(rows[1]["issue_type"], "no_video_frame_in_interval")
        finally:
            shutil.rmtree(sample_dir)

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
