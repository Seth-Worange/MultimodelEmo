from __future__ import annotations

import csv
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from q1_v2.alignment import WordAlignment
from q1_v2.correspondence_qa import ASRWord, QAConfig, decide_correspondence, local_word_match
from q1_v2.data_loader import split_original_words
from q1_v2.manual_sample_audit import create_manual_sample_audit
from q1_v2.round4_features import crop_for_local_mfa, unavailable_alignment
from q1_v2.visual_features import BLENDSHAPE_NAMES, aggregate_visual_to_words, normalize_landmarks


class Round4SyntheticTests(unittest.TestCase):
    def _words(self, phrase: str) -> list[ASRWord]:
        return [ASRWord(token, i * .2, (i + 1) * .2, .9) for i, token in enumerate(phrase.split())]

    def _decide(self, match: dict, *, speech: bool = True, language: str = "en", probability: float = .99):
        return decide_correspondence(
            audio_signal_present=True, speech_present=speech,
            detected_language=language, language_probability=probability,
            match=match, config=QAConfig(),
        )

    def test_exact_text_is_high_confidence(self) -> None:
        match = local_word_match("one two three four", self._words("one two three four"))
        self.assertEqual(match["token_recall"], 1.0)
        self.assertEqual(self._decide(match)[1], "HIGH_CONFIDENCE_MATCH")

    def test_extra_before_after_and_both_are_locally_detected(self) -> None:
        official = "they've been able to find solutions"
        for prefix, suffix in (("extra words ", ""), ("", " extra words"), ("extra words ", " extra words")):
            with self.subTest(prefix=prefix, suffix=suffix):
                words = self._words(prefix + "they've been able to find solutions" + suffix)
                match = local_word_match(official, words)
                self.assertEqual(match["token_recall"], 1.0)
                self.assertEqual(match["extra_speech_before"], bool(prefix))
                self.assertEqual(match["extra_speech_after"], bool(suffix))
                self.assertEqual(self._decide(match)[1], "HIGH_CONFIDENCE_MATCH")

    def test_partial_and_total_mismatch_do_not_force_align(self) -> None:
        partial = local_word_match("one two three four five six", self._words("two three four"))
        mismatch = local_word_match("one two three four", self._words("red green blue"))
        self.assertEqual(self._decide(partial)[1], "REVIEW_REQUIRED")
        self.assertEqual(self._decide(mismatch)[1], "INVALID_CORRESPONDENCE")

    def test_repeated_full_sentence_requires_review(self) -> None:
        match = local_word_match(
            "one two three four", self._words("one two three four other one two three four"),
        )
        self.assertTrue(match["local_match_ambiguous"])
        self.assertEqual(self._decide(match)[1], "REVIEW_REQUIRED")

    def test_no_speech_and_non_english_are_invalid(self) -> None:
        match = local_word_match("one two three", self._words("one two three"))
        self.assertEqual(self._decide(match, speech=False)[0], "NO_SPEECH")
        self.assertEqual(self._decide(match, language="fr")[0], "NON_ENGLISH_SPEECH")

    def test_invalid_correspondence_keeps_all_official_words(self) -> None:
        record = SimpleNamespace(sample_id="a__1", original_words=split_original_words("One two three"))
        aligned = unavailable_alignment(record, "text_audio_mismatch")
        self.assertEqual([item.original_word for item in aligned], ["One", "two", "three"])
        self.assertTrue(all(item.alignment_mask == 0 and np.isnan(item.start_s) for item in aligned))

    def test_crop_maps_local_mfa_time_back_to_global_sample_time(self) -> None:
        root = Path.cwd() / f"_round4_synthetic_{uuid.uuid4().hex}"
        root.mkdir()
        try:
            media = SimpleNamespace(
                audio_timestamp_reliable=True, wav_origin_sample_s=.2,
                wav_origin_media_s=1.2, timeline_origin_media_s=1.0,
                waveform=np.zeros(16000 * 3, dtype=np.float32),
                sample_rate=16000, wav_path="original.wav",
            )
            crop = crop_for_local_mfa(media, 1.0, 2.0, margin_s=.3, output_path=root / "crop.wav")
            self.assertAlmostEqual(crop["crop_start_global_s"], .7, places=4)
            self.assertAlmostEqual(crop["crop_wav_origin_media_s"], 1.7, places=4)
            self.assertAlmostEqual(crop["crop_start_global_s"] + .2, .9, places=4)
        finally:
            self.assertTrue(root.resolve().is_relative_to(Path.cwd().resolve()))
            shutil.rmtree(root)

    def test_native_mediapipe_geometry_is_not_falsely_called_openface68(self) -> None:
        points = [SimpleNamespace(x=i / 1000, y=i / 1500, z=i / 2000) for i in range(478)]
        geometry = normalize_landmarks(points, tuple(range(478)))
        self.assertEqual(geometry.shape, (1434,))
        self.assertEqual(len(BLENDSHAPE_NAMES), 52)
        self.assertEqual(len(geometry) + len(BLENDSHAPE_NAMES), 1486)

    def test_short_word_without_sample_and_no_face_are_distinct(self) -> None:
        words = [
            WordAlignment("a__1", 0, "a", "a", .02, .07, 1),
            WordAlignment("a__1", 1, "b", "b", .10, .15, 1),
        ]
        result = aggregate_visual_to_words(
            np.full((2, 2), np.nan, dtype=np.float32), np.asarray([0., .1]),
            np.asarray([0, 0], dtype=np.uint8), np.asarray([0, 0]),
            np.asarray([0, 3]), words,
        )
        self.assertEqual(result[-1], ["no_sampled_frame_in_interval", "sampled_frames_present_but_all_face_detections_failed"])

    def test_manual_sample_audit_preserves_human_cells(self) -> None:
        root = Path.cwd() / f"_round4_audit_{uuid.uuid4().hex}"
        root.mkdir()
        path = root / "audit.csv"
        records = [SimpleNamespace(sample_id=f"v__{i}", video_id="v", clip_id=str(i)) for i in range(100)]
        try:
            create_manual_sample_audit(records, path)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["automatic_correspondence_status"], "NOT_EVALUATED")
            rows[0]["reviewer_id"] = "reviewer-x"
            rows[0]["manual_speech_present"] = "yes"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            create_manual_sample_audit(records, path, automatic_by_id={"v__0": {
                "automatic_correspondence_status": "MATCHED",
            }})
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                updated = list(csv.DictReader(handle))
            self.assertEqual(updated[0]["reviewer_id"], "reviewer-x")
            self.assertEqual(updated[0]["manual_speech_present"], "yes")
            self.assertEqual(updated[0]["automatic_correspondence_status"], "MATCHED")
        finally:
            self.assertTrue(root.resolve().is_relative_to(Path.cwd().resolve()))
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
