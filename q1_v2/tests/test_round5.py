"""Synthetic-only Round-5 contracts; no synthetic observation is reported as real QA."""

import csv
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from q1_v2.alignment_confidence import confidence, map_official_to_asr
from q1_v2.config import Q1Config
from q1_v2.correspondence_qa import vad_extra_speech
from q1_v2.data_loader import OriginalWord, SampleRecord
from q1_v2.manual_boundary_import import normalize_boundaries
from q1_v2.manual_sample_audit_import import _boolean, _category, normalize_rows
from q1_v2.openface_backend import associate_openface_with_pts, feature_schema, normalize_openface68, parse_openface_csv
from q1_v2.stage2_qa import second_stage_decision
from q1_v2.visual_scene_qa import visual_scene_decision
from q1_v2.visual_backend import extract_with_backend


def _official(count=100):
    return [SampleRecord(sample_id=f"video{i}__0", video_id=f"video{i}", clip_id="0",
                         text="Hello world", video_path=Path(f"video{i}.mp4"), source_row=i + 2,
                         original_words=[OriginalWord(0, "Hello", 0, 5), OriginalWord(1, "world", 6, 11)])
            for i in range(count)]


def _qa_table(count=100):
    header = ["sample_id", "manual_speech_present", "manual_language",
              "manual_official_text_present", "manual_extra_speech", "manual_face_present",
              "manual_static_or_nonhuman", "manual_correspondence_class", "reviewer_id"]
    return [header] + [[f"video{i}__0", True, "en", "full", "none", True, False, "matched", i % 2 or "r"]
                       for i in range(count)]


def _boundary_table():
    return [["sample_id", "word_index", "original words", "reference_start_s", "reference_end_s", "reviewer_id"],
            ["video0__0", 0, "Hello", 0.1, .4, None],
            ["video0__0", 1, "world", .4, .8, None],
            ["video1__0", 0, "Hello", .1, .3, "r"]]


def test_manual_100_import():
    rows, report = normalize_rows(_qa_table(), _official())
    assert len(rows) == report["matched_official_sample_count"] == 100


def test_manual_duplicate():
    table = _qa_table()
    table[-1][0] = "video0__0"
    _, report = normalize_rows(table, _official())
    assert report["duplicate_sample_ids"] == ["video0__0"]


def test_manual_missing():
    _, report = normalize_rows(_qa_table(99), _official())
    assert report["missing_sample_ids"] == ["video99__0"]


def test_manual_extra():
    table = _qa_table()
    table[-1][0] = "extraneous__0"
    _, report = normalize_rows(table, _official())
    assert report["extra_sample_ids"] == ["extraneous__0"]


def test_manual_case_normalization():
    rows, _ = normalize_rows(_qa_table(), _official())
    assert rows[0]["manual_correspondence_class"] == "MATCHED"


@pytest.mark.parametrize("value,expected", [(2, "2"), (" x ", "x"), (None, "")])
def test_reviewer_normalization(value, expected):
    table = _qa_table()
    table[1][-1] = value
    rows, _ = normalize_rows(table, _official())
    assert rows[0]["reviewer_id"] == expected


@pytest.mark.parametrize("value,expected", [(True, True), (0, False), (" uncertain ", None)])
def test_excel_boolean_normalization(value, expected):
    assert _boolean(value, "test") is expected


def test_manual_optional_blank():
    table = _qa_table()
    table[1][4] = None
    rows, _ = normalize_rows(table, _official())
    assert rows[0]["manual_extra_speech"] is None


def test_boundary_multi_sample():
    rows, report = normalize_boundaries(_boundary_table(), _official(), {"video0__0": 2, "video1__0": 2})
    assert len(rows) == 3 and report["distinct_sample_count"] == 2


def test_boundary_word_index_mismatch():
    table = _boundary_table()
    table[1][1] = 1
    table.pop(2)
    rows, report = normalize_boundaries(table, _official(), {"video0__0": 2, "video1__0": 2})
    assert rows[0]["word_mapping_status"] == "WORD_MISMATCH"
    assert report["word_mapping_error_count"] == 1


def test_boundary_original_word_mismatch():
    table = _boundary_table()
    table[1][2] = "wrong"
    rows, _ = normalize_boundaries(table, _official(), {"video0__0": 2, "video1__0": 2})
    assert rows[0]["word_mapping_status"] == "WORD_MISMATCH"


@pytest.mark.parametrize("start,end", [(.5, .4), (-.1, .4), (.1, 2.2)])
def test_boundary_invalid(start, end):
    table = _boundary_table()
    table[1][3:5] = [start, end]
    with pytest.raises(ValueError):
        normalize_boundaries(table, _official(), {"video0__0": 2, "video1__0": 2})


def test_boundary_partial_coverage():
    table = _boundary_table()
    table.pop(2)
    _, report = normalize_boundaries(table, _official(), {"video0__0": 2, "video1__0": 2})
    assert report["manual_row_coverage_by_sample"]["video0__0"] == .5


def test_boundary_missing_end_not_imputed():
    table = _boundary_table()
    table[1][4] = None
    rows, _ = normalize_boundaries(table, _official(), {"video0__0": 2, "video1__0": 2})
    assert rows[0]["reference_end_s"] is None
    assert rows[0]["boundary_status"] == "MISSING_END"


@pytest.mark.parametrize("regions,expected", [
    ([{"start_s": .1, "end_s": .6}], (True, False)),
    ([{"start_s": 3.4, "end_s": 4}], (False, True)),
    ([{"start_s": .1, "end_s": .6}, {"start_s": 3.4, "end_s": 4}], (True, True)),
    ([], (False, False)),
])
def test_vad_extra_directions(regions, expected):
    result = vad_extra_speech(regions, matched_start_s=1, matched_end_s=3,
                              audio_start_s=0, audio_end_s=4, tolerance_s=.15,
                              min_duration_s=.2)
    assert (result["extra_speech_before"], result["extra_speech_after"]) == expected


def test_vad_extra_without_asr_words():
    result = vad_extra_speech([{"start_s": .1, "end_s": .6}], matched_start_s=1,
                              matched_end_s=3, audio_start_s=0, audio_end_s=4,
                              tolerance_s=.15, min_duration_s=.2,
                              asr_unmatched_before=False)
    assert result["extra_speech_before"] is True


def test_vad_extra_without_span():
    result = vad_extra_speech([], matched_start_s=None, matched_end_s=None,
                              audio_start_s=0, audio_end_s=4, tolerance_s=.15, min_duration_s=.2)
    assert result["extra_speech_before"] is None


def test_any_face_not_usable_face():
    row = {"decoded": True, "face_count": 1, "area_qualified_face": False,
           "sample_time_s": 0., "frame_mean_absolute_difference": None}
    result = visual_scene_decision([row], fps=5, config=Q1Config())
    assert result["any_face_detected"] and not result["usable_face_present"]


def test_no_face_scene():
    row = {"decoded": True, "face_count": 0, "area_qualified_face": False,
           "sample_time_s": 0., "frame_mean_absolute_difference": .2}
    assert visual_scene_decision([row], fps=5, config=Q1Config())["visual_scene_status"] == "NON_FACE_DYNAMIC_SCENE"


def _geometry():
    xy = np.zeros((68, 2), dtype=np.float64)
    xy[:, 0] = np.arange(68)
    xy[36:42, 0] = 20
    xy[42:48, 0] = 40
    return xy


def test_openface68_geometry_shape():
    vector = normalize_openface68(_geometry())
    assert vector.shape == (136,) and np.isfinite(vector).all()


def test_openface68_schema_not_mediapipe():
    schema = feature_schema()
    assert schema["geometry_frame_dimension"] == 136 and not schema["mediapipe_blendshape_concatenated"]


def test_openface68_failed_frame():
    path = Path("synthetic_openface.csv")
    names = ["frame", "timestamp", "confidence", "success"] + [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=names)
    writer.writeheader()
    writer.writerow({"frame": 1, "timestamp": 0, "confidence": .1, "success": 0})
    with patch.object(Path, "open", return_value=io.StringIO(handle.getvalue())):
        rows, _ = parse_openface_csv(path)
    assert not rows[0]["face_valid"] and np.isnan(rows[0]["geometry"]).all()


def test_openface_timestamp_unreliable_mask():
    parsed = [{"openface_frame": 1, "openface_timestamp_s": 1., "face_valid": True,
               "failure_reason": "", "geometry": np.ones(136, dtype=np.float32),
               "action_units": np.ones(2, dtype=np.float32),
               "head_pose": np.ones(1, dtype=np.float32), "gaze": np.ones(1, dtype=np.float32)}]
    media = [{"frame_index": 0, "pts": 0, "time_base": "1/1000", "media_time_s": 0.,
              "sample_time_s": 0., "corrupt": False, "timestamp_valid": True}]
    result = associate_openface_with_pts(parsed, media, fps=10)
    assert not result[0]["face_valid"]
    assert np.isnan(result[0]["action_units"]).all()


def test_asr_mfa_word_mapping():
    mapping = map_official_to_asr(["They've", "been"],
                                  [{"word": "They"}, {"word": "have"}, {"word": "been"}],
                                  first_asr_index=0, last_asr_index=2)
    assert mapping[0]["indices"] == [0, 1] and mapping[1]["indices"] == [2]


@pytest.mark.parametrize("value,expected", [(None, "UNAVAILABLE"), (.05, "HIGH"), (.15, "MEDIUM"), (.3, "LOW")])
def test_mfa_asr_confidence(value, expected):
    assert confidence(value, Q1Config()) == expected


def test_stage2_does_not_override_stage1():
    first = {"route": "INVALID_CORRESPONDENCE", "correspondence_status": "TEXT_AUDIO_MISMATCH",
             "matched_audio_start_s": 1., "matched_audio_end_s": 2., "token_recall": 1.}
    match = {"matched_audio_start_s": 1., "matched_audio_end_s": 2.}
    assert not second_stage_decision(first, "MATCHED", "HIGH_CONFIDENCE_MATCH", match)[0]


def test_stage2_span_disagreement_blocks_promotion():
    first = {"route": "REVIEW_REQUIRED", "correspondence_status": "PARTIAL_MATCH",
             "matched_audio_start_s": 1., "matched_audio_end_s": 2., "token_recall": .9,
             "local_match_ambiguous": False}
    match = {"matched_audio_start_s": 1.5, "matched_audio_end_s": 2.5}
    assert second_stage_decision(first, "MATCHED", "HIGH_CONFIDENCE_MATCH", match)[1] == "two_decodes_disagree_on_span"


def test_config_native478_distinct_from_legacy38():
    config = Q1Config()
    assert len(config.mediapipe_native_landmark_indices) == 478
    assert len(config.selected_landmarks) == 38
    assert config.crop_margin_s == .30 and config.visual_probe_fps == 5.
    assert config.final_visual_backend_candidate == "openface68"


def test_visual_backend_unknown_rejected():
    with pytest.raises(ValueError):
        extract_with_backend(backend="invented68", video_path=Path("x.mp4"),
                             media_metadata=Path("x.json"), output_dir=Path("unused"),
                             alignments=[], config=Q1Config())


def test_npz_no_pickle():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, geometry=np.ones((2, 136), dtype=np.float32), mask=np.array([True, False]))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        assert archive["geometry"].shape == (2, 136)
