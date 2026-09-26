"""Synthetic-only Round-5 contracts; no synthetic observation is reported as real QA."""

import csv
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from q1_v2.alignment import WordAlignment
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
from q1_v2.visual_backend import MediaPipe478Backend, OpenFace68Backend, make_visual_backend
from q1_v2.final_schema import feature_schema as frozen_feature_schema, validate_package
from q1_v2.final_schema import visual_schema as frozen_visual_schema
from q1_v2.final_pipeline import _nullable_bool, _package, _quality_block, select_smoke_ids


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


def test_frozen_default_openface68():
    config = Q1Config()
    assert config.visual_backend == "openface68"
    assert config.alignment_margin_s == config.crop_margin_s == .30
    assert config.openface_confidence_threshold == config.face_validity_threshold == .5


def test_visual_backend_factory_modes():
    assert isinstance(make_visual_backend(Q1Config(), face_model=None,
                                          openface_executable=Path("openface.exe")), OpenFace68Backend)
    from dataclasses import replace
    assert isinstance(make_visual_backend(replace(Q1Config(), visual_backend="mediapipe478"),
                                          face_model=Path("face.task"), openface_executable=None),
                      MediaPipe478Backend)


def test_frozen_openface_block_dimensions():
    schema = frozen_visual_schema("openface68", Q1Config())
    assert [block["word_dimension"] for block in schema["blocks"]] == [272, 70, 12, 16]
    assert sum(block["word_dimension"] for block in schema["blocks"]) == 370
    assert schema["stored_as_separate_blocks"]


def test_frozen_mediapipe_baseline_schema():
    schema = frozen_visual_schema("mediapipe478", Q1Config())
    assert [block["frame_dimension"] for block in schema["blocks"]] == [1434, 52]


def test_frozen_package_schema_no_pickle():
    schema = frozen_feature_schema("openface68", Q1Config())
    assert schema["npz_loading"] == "np.load(path, allow_pickle=False)"
    assert schema["speaker_identity_verified"] is False


@pytest.mark.parametrize("source,expected", [(None, -1), ("", -1), (False, 0), (True, 1), ("false", 0), ("true", 1)])
def test_frozen_nullable_quality_flag(source, expected):
    assert int(_nullable_bool(source)) == expected


def test_npz_no_pickle():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, geometry=np.ones((2, 136), dtype=np.float32), mask=np.array([True, False]))
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as archive:
        assert archive["geometry"].shape == (2, 136)


@pytest.mark.parametrize("qa,expected", [
    ({"correspondence_status": "MATCHED", "speech_present": True,
      "english_speech_likely": True, "face_present": True},
     {"correspondence_status": "MATCHED", "speech_status": "SPEECH_PRESENT",
      "language_status": "ENGLISH", "face_status": "FACE_PRESENT"}),
    ({"correspondence_status": "NO_SPEECH", "speech_present": False,
      "english_speech_likely": False, "face_present": False},
     {"correspondence_status": "NO_SPEECH", "speech_status": "NO_SPEECH",
      "language_status": "NOT_APPLICABLE", "face_status": "NO_FACE"}),
    ({"correspondence_status": "NON_ENGLISH_SPEECH", "speech_present": "True",
      "english_speech_likely": "False", "face_present": None},
     {"correspondence_status": "NON_ENGLISH_SPEECH", "speech_status": "SPEECH_PRESENT",
      "language_status": "NON_ENGLISH", "face_status": "UNKNOWN"}),
])
def test_frozen_quality_block(qa, expected):
    assert _quality_block(qa) == expected


def _synthetic_package(target: Path, sample_id: str = "videoX__0") -> None:
    record = SampleRecord(
        sample_id=sample_id, video_id="videoX", clip_id="0", text="Hello world",
        video_path=target / "source.mp4", source_row=2,
        original_words=[OriginalWord(0, "Hello", 0, 5), OriginalWord(1, "world", 6, 11)])
    record.video_path.write_bytes(b"synthetic")
    words = 2
    alignments = [WordAlignment(sample_id, 0, "Hello", "hello", 0.1, 0.4, 1, ""),
                  WordAlignment(sample_id, 1, "world", "world", 0.5, 0.9, 1, "")]
    text = SimpleNamespace(features=np.ones((words, 768), dtype=np.float32),
                           mask=np.ones(words, dtype=np.uint8))
    audio = SimpleNamespace(
        word_features=np.ones((words, 138), dtype=np.float32),
        word_mask=np.ones(words, dtype=np.uint8),
        frame_features=np.ones((3, 69), dtype=np.float32),
        frame_times_s=np.array([0.0, 0.01, 0.02]), frame_start_samples=np.array([0, 160, 320]),
        f0_valid_mask=np.ones(3, dtype=np.uint8),
        word_frame_indices=[[0, 1], [2]], word_failure_reasons=["", ""])
    frames = 4
    visual = SimpleNamespace(
        backend="openface68",
        frame_blocks={"landmark_geometry": np.ones((frames, 136), dtype=np.float32),
                      "action_units": np.ones((frames, 35), dtype=np.float32),
                      "head_pose": np.ones((frames, 6), dtype=np.float32),
                      "gaze": np.ones((frames, 8), dtype=np.float32)},
        word_blocks={"landmark_geometry": np.ones((words, 272), dtype=np.float32),
                     "action_units": np.ones((words, 70), dtype=np.float32),
                     "head_pose": np.ones((words, 12), dtype=np.float32),
                     "gaze": np.ones((words, 16), dtype=np.float32)},
        frame_indices=np.arange(frames, dtype=np.int32),
        frame_times_s=np.array([0.1, 0.2, 0.3, 0.4]),
        frame_mask=np.ones(frames, dtype=np.uint8), word_mask=np.ones(words, dtype=np.uint8),
        word_sampled_frame_counts=np.full(words, 2, dtype=np.int32),
        word_valid_face_counts=np.full(words, 2, dtype=np.int32),
        word_original_frame_indices=[[0, 1], [2, 3]])
    qa = {"correspondence_status": "MATCHED", "route": "HIGH_CONFIDENCE_MATCH",
          "speech_present": True, "english_speech_likely": True, "face_present": True}
    _package(record, target, qa, {"usable_face_present": True},
             alignments, ["LOW", "HIGH"], text, audio, visual,
             mfa_policy="REUSED_VERIFIED_REAL_MFA")


def test_frozen_package_validate_round_trip(tmp_path):
    _synthetic_package(tmp_path)
    assert validate_package(tmp_path, "videoX__0", backend="openface68") == []
    with np.load(tmp_path / "feature_package.npz", allow_pickle=False) as archive:
        assert str(archive["speech_status"].item()) == "SPEECH_PRESENT"
        assert str(archive["language_status"].item()) == "ENGLISH"
        assert str(archive["face_status"].item()) == "FACE_PRESENT"
    metadata = (tmp_path / "feature_package_metadata.json").read_text(encoding="utf-8")
    assert '"correspondence_status"' in metadata
    broken = tmp_path / "broken"
    broken.mkdir()
    _synthetic_package(broken)
    with np.load(broken / "feature_package.npz", allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["word_text"] = payload["word_text"][:1]
    np.savez_compressed(broken / "feature_package.npz", **payload)
    assert any(item.startswith("word_length_mismatch") for item in
               validate_package(broken, "videoX__0", backend="openface68"))


def test_select_smoke_ids_includes_non_english(tmp_path):
    qa_root, round5_root = tmp_path / "r4", tmp_path / "r5"
    (round5_root / "qa").mkdir(parents=True)
    qa_root.mkdir()
    with (qa_root / "correspondence_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "correspondence_status", "route"])
        writer.writeheader()
        writer.writerows([
            {"sample_id": "ne__0", "correspondence_status": "NON_ENGLISH_SPEECH",
             "route": "INVALID_CORRESPONDENCE"},
            {"sample_id": "ok__0", "correspondence_status": "MATCHED",
             "route": "HIGH_CONFIDENCE_MATCH"},
            {"sample_id": "ok__1", "correspondence_status": "MATCHED",
             "route": "HIGH_CONFIDENCE_MATCH"},
        ])
    with (round5_root / "qa" / "correspondence_audit_round5.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "visual_scene_status"])
        writer.writeheader()
        writer.writerows([{"sample_id": "ok__0", "visual_scene_status": "NORMAL_FACE_VIDEO"},
                          {"sample_id": "ok__1", "visual_scene_status": "NORMAL_FACE_VIDEO"}])
    selected = select_smoke_ids(qa_root, round5_root)
    assert len(selected) == len(set(selected)) == 10
    assert selected[8] == "ne__0"
    assert selected[9] in {"ok__0", "ok__1"}


def test_partial_review_number_normalization():
    from q1_v2.partial_review import normalize_words
    units = normalize_words(["twelve", "hundred", "square", "feet", "2008"])
    assert [u.text for u in units] == ["1200", "square", "feet", "2008"]
    assert units[0].word_indices == (0, 1)


def test_partial_review_blockwise_pure_insertion():
    from q1_v2.partial_review import review_partial_match
    from q1_v2.data_loader import OriginalWord
    words = [OriginalWord(i, t, 0, 0) for i, t in
             enumerate(["in", "kenya", "following", "election"])]
    asr = ([{"word": "in", "start_s": "0.1", "end_s": "0.3"},
            {"word": "kenya", "start_s": "0.3", "end_s": "0.6"},
            {"word": "and", "start_s": "0.7", "end_s": "0.8"},
            {"word": "even", "start_s": "0.8", "end_s": "0.9"},
            {"word": "following", "start_s": "1.0", "end_s": "1.2"},
            {"word": "election", "start_s": "1.3", "end_s": "1.6"}])
    out = review_partial_match(sample_id="x__0", official_text="in kenya following election",
                               original_words=words, asr_rows=asr,
                               stage1_ambiguous=False,
                               thresholds={"match_recall_min": 0.8, "match_precision_min": 0.75,
                                           "match_edit_similarity_min": 0.7, "min_exact_tokens": 3})
    assert out["decision"]["passed"] is True
    assert out["decision"]["policy"] == "REVIEWED_BLOCKWISE_MFA"
    assert out["gaps"][0]["pure_inserted_speech"] is True
    assert out["segments"][1]["official_text"] == "following election"


def test_partial_review_missing_text_stays_blocked():
    from q1_v2.partial_review import review_partial_match
    from q1_v2.data_loader import OriginalWord
    words = [OriginalWord(i, t, 0, 0) for i, t in
             enumerate(["look", "at", "the", "record", "today", "please"])]
    asr = [{"word": w, "start_s": str(i * 0.2), "end_s": str(i * 0.2 + 0.2)}
           for i, w in enumerate(["look", "at", "the", "record"])]
    out = review_partial_match(sample_id="x__1", official_text="look at the record today please",
                               original_words=words, asr_rows=asr,
                               stage1_ambiguous=False,
                               thresholds={"match_recall_min": 0.8, "match_precision_min": 0.75,
                                           "match_edit_similarity_min": 0.7, "min_exact_tokens": 3})
    assert out["decision"]["passed"] is False
    assert out["decision"]["policy"] is None
