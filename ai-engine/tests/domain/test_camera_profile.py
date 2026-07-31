from __future__ import annotations

import pytest

from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES, CameraProfile


def test_defaults_match_the_spec_table() -> None:
    profile = CameraProfile(camera_id="cam-1")
    assert profile.min_track_frames == 8
    assert profile.scene_delta_threshold == 0.35
    assert profile.scene_delta_frames == 5
    assert profile.dwell_radius_px == 48.0
    assert profile.dwell_seconds == 30.0
    assert profile.speed_percentile == 0.95
    assert profile.speed_fallback_px_s == 180.0
    assert profile.track_count_baseline == 6
    assert profile.summary_interval_seconds == 45.0
    assert profile.bucket_capacity == 2
    assert profile.bucket_refill_seconds == 10.0
    assert profile.vlm_enabled is True


def test_default_salient_classes_match_the_spec() -> None:
    expected = frozenset(
        {
            "person",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "backpack",
            "handbag",
            "suitcase",
        }
    )
    assert expected == DEFAULT_SALIENT_CLASSES
    assert CameraProfile(camera_id="cam-1").salient_classes is DEFAULT_SALIENT_CLASSES


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("min_track_frames", 0, "min_track_frames"),
        ("scene_delta_threshold", 1.5, "scene_delta_threshold"),
        ("scene_delta_frames", 0, "scene_delta_frames"),
        ("dwell_radius_px", -1.0, "dwell_radius_px"),
        ("dwell_seconds", 0.0, "dwell_seconds"),
        ("speed_percentile", 1.0, "speed_percentile"),
        ("speed_fallback_px_s", 0.0, "speed_fallback_px_s"),
        ("track_count_baseline", 0, "track_count_baseline"),
        ("summary_interval_seconds", 0.0, "summary_interval_seconds"),
        ("bucket_capacity", 0, "bucket_capacity"),
        ("bucket_refill_seconds", 0.0, "bucket_refill_seconds"),
    ],
)
def test_invalid_values_are_rejected_at_construction(
    field_name: str, bad_value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CameraProfile(camera_id="cam-1", **{field_name: bad_value})


def test_profile_is_immutable() -> None:
    profile = CameraProfile(camera_id="cam-1")
    with pytest.raises(AttributeError):
        profile.dwell_seconds = 5.0  # type: ignore[misc]
