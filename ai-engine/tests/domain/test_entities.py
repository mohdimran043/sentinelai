from __future__ import annotations

from uuid import uuid4

import pytest

from sentinel_ai.domain.entities import (
    BBox,
    Detection,
    EscalationReason,
    Event,
    SceneState,
    Severity,
    ThreatScore,
    Track,
)
from sentinel_ai.domain.welfare import WelfareAssessment


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


class TestEventWelfareDefault:
    """Task 2: `Event.welfare` must default to `WelfareAssessment.none()` so every
    existing construction site (every call in this repo predates the field) keeps
    working untouched."""

    def test_welfare_defaults_to_none(self) -> None:
        event = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=0.0,
            reason=EscalationReason.SPEED_ANOMALY,
            threat=ThreatScore.from_value(0.5),
            description="A person is running toward the gate.",
            suggested_action="Review the clip.",
        )
        assert event.welfare == WelfareAssessment.none()

    def test_two_default_constructed_events_do_not_share_a_welfare_instance(self) -> None:
        """Pins the mutable-default trap: `welfare: WelfareAssessment = WelfareAssessment.none()`
        would evaluate the default once at class-definition time and share it across
        every instance. `WelfareAssessment` is frozen, so aliasing would not corrupt
        state here today — but `field(default_factory=...)` is still the correct
        mechanism, and this test would fail loudly if a future edit swapped it for a
        bare shared default that stopped being safe to alias."""
        first = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=0.0,
            reason=EscalationReason.SPEED_ANOMALY,
            threat=ThreatScore.from_value(0.5),
            description="",
            suggested_action="",
        )
        second = Event(
            event_id=uuid4(),
            camera_id="cam-2",
            occurred_at=0.0,
            reason=EscalationReason.SPEED_ANOMALY,
            threat=ThreatScore.from_value(0.5),
            description="",
            suggested_action="",
        )
        assert first.welfare is not second.welfare


def test_escalation_reasons_are_exactly_the_documented_set() -> None:
    """Pinned as a whole set, not a subset, so a member added without a published
    contract change fails here rather than at a consumer.

    `reason` is in the event schema's enum, so every addition is a wire change: a
    consumer validating against the committed schema rejects an unknown value, which
    is the correct failure but a remote one. This is the local one.
    """
    assert {r.value for r in EscalationReason} == {
        # Spec §4.1's seven.
        "new_salient_track",
        "scene_change",
        "dwell_exceeded",
        "speed_anomaly",
        "track_count_spike",
        "periodic_summary",
        "user_requested",
        # Spec §6 and §7. Not per-frame predicates like the six automatic ones above,
        # but temporal state machines' reports — see `domain/behaviour/`.
        "fall_suspected",
        "abandoned_object",
        "camera_tamper",
        "zone_intrusion",
        "line_crossing",
        # Spec §8-§12. Also a temporal machine's report rather than a per-frame
        # predicate — see `domain/policy/authorization.py`.
        "unauthorized_person",
    }


def test_every_behaviour_kind_has_a_reason_and_no_reason_is_orphaned() -> None:
    """The mapping the pipeline publishes through must be total in both directions.

    A kind with no reason cannot be escalated at all, and a behaviour reason with no
    kind is a value on the wire nothing can produce — a filter option in a console that
    never matches. `_BEHAVIOUR_REASONS` is indexed rather than `.get`-with-a-default
    precisely so the first is a build failure; this pins the second.
    """
    from sentinel_ai.domain.behaviour.candidate import BehaviourKind
    from sentinel_ai.pipeline.runner import _BEHAVIOUR_REASONS

    assert set(_BEHAVIOUR_REASONS) == set(BehaviourKind)
    # `unauthorized_person` is deliberately absent from `BehaviourKind`: it is not a
    # behaviour detector. It runs on a face pipeline rather than on tracks and geometry,
    # and it is raised on its own path — see `CameraRunner._detect_unauthorized`.
    behaviour_reasons = set(_BEHAVIOUR_REASONS.values())
    assert len(behaviour_reasons) == len(BehaviourKind), "two kinds share one reason"


def test_the_fall_reason_is_named_suspected_rather_than_detected() -> None:
    """ADR 10, applied to the reason vocabulary. `fall_detected` would read to every
    downstream consumer as a trained classifier's verdict; what is actually behind
    this is a geometry state machine awaiting vision-language confirmation."""
    assert EscalationReason.FALL_SUSPECTED.value == "fall_suspected"
    assert "fall_detected" not in {r.value for r in EscalationReason}
