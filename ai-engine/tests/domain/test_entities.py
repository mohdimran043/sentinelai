from __future__ import annotations

import pytest

from sentinel_ai.domain.entities import (
    BBox,
    Detection,
    EscalationReason,
    SceneState,
    Severity,
    ThreatScore,
    Track,
)


def _scene(signature: tuple[float, ...]) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=(),
        motion_energy=0.0,
        scene_signature=signature,
    )


class TestBBox:
    def test_centroid_is_the_box_centre(self) -> None:
        box = BBox(10.0, 20.0, 30.0, 60.0)
        assert box.cx == 20.0
        assert box.cy == 40.0

    def test_area_is_width_times_height(self) -> None:
        assert BBox(0.0, 0.0, 4.0, 5.0).area == 20.0

    def test_inverted_box_has_zero_area_not_negative(self) -> None:
        assert BBox(10.0, 10.0, 4.0, 4.0).area == 0.0

    def test_is_immutable(self) -> None:
        box = BBox(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(AttributeError):
            box.x1 = 5.0  # type: ignore[misc]


class TestSignatureDelta:
    def test_identical_signatures_have_zero_delta(self) -> None:
        assert _scene((0.5, 0.5)).signature_delta((0.5, 0.5)) == 0.0

    def test_missing_previous_signature_is_treated_as_no_change(self) -> None:
        """First frame must not fire SceneChange — there is nothing to compare to."""
        assert _scene((0.5, 0.5)).signature_delta(None) == 0.0

    def test_disjoint_signatures_have_delta_of_one(self) -> None:
        assert _scene((1.0, 0.0)).signature_delta((0.0, 1.0)) == pytest.approx(1.0)

    def test_delta_is_half_l1_distance(self) -> None:
        # |0.6-0.4| + |0.4-0.6| = 0.4 ; halved = 0.2
        assert _scene((0.6, 0.4)).signature_delta((0.4, 0.6)) == pytest.approx(0.2)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _scene((0.5, 0.5)).signature_delta((0.3, 0.3, 0.4))


class TestSceneStateHelpers:
    def test_tracks_of_returns_only_matching_labels(self) -> None:
        box = BBox(0.0, 0.0, 1.0, 1.0)
        scene = SceneState(
            camera_id="cam-1",
            frame_index=3,
            timestamp=1.0,
            detections=(Detection("person", 0.9, box), Detection("dog", 0.8, box)),
            tracks=(
                Track(1, "person", box, age_frames=10, speed_px_s=0.0),
                Track(2, "dog", box, age_frames=4, speed_px_s=5.0),
            ),
            motion_energy=0.1,
            scene_signature=(1.0,),
        )
        assert [t.track_id for t in scene.tracks_of({"person"})] == [1]

    def test_tracks_of_on_an_empty_scene_yields_nothing(self) -> None:
        assert list(_scene((1.0,)).tracks_of({"person"})) == []


class TestThreatScore:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, Severity.INFO),
            (0.19, Severity.INFO),
            (0.2, Severity.LOW),
            (0.4, Severity.MEDIUM),
            (0.6, Severity.HIGH),
            (0.8, Severity.CRITICAL),
            (1.0, Severity.CRITICAL),
        ],
    )
    def test_severity_is_derived_from_value(self, value: float, expected: Severity) -> None:
        assert ThreatScore.from_value(value).severity is expected

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_out_of_range_values_are_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            ThreatScore.from_value(value)


def test_escalation_reasons_cover_all_seven_spec_triggers() -> None:
    assert {r.value for r in EscalationReason} == {
        "new_salient_track",
        "scene_change",
        "dwell_exceeded",
        "speed_anomaly",
        "track_count_spike",
        "periodic_summary",
        "user_requested",
    }
