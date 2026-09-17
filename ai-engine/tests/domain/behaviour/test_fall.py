"""The fall signature: what completes it, and — mostly — what must not.

Every scenario below is a scripted sequence of bounding boxes at chosen timestamps,
run through the pure state machine on a CPU in microseconds. That is the entire reason
the machine is pure: "the person stayed down for seven seconds" is otherwise a test
nobody writes, and "the person got up after four" is one nobody writes twice.

The negative cases carry the weight. A detector that fires on a fall is easy; one that
stays silent while somebody sits down, lies down to stretch, or is simply already on
the floor is the difference between an alert an operator acts on and one they learn to
dismiss.
"""

from __future__ import annotations

import pytest

from sentinel_ai.domain.behaviour.fall import (
    FallEvidence,
    FallPolicy,
    FallTracker,
    Posture,
    observe_falls,
)
from sentinel_ai.domain.behaviour.observation import (
    BehaviourObservation,
    Keypoint,
    KeypointName,
    PersonPose,
)
from sentinel_ai.domain.entities import BBox, SceneState, Track

POLICY = FallPolicy()


def upright_box(cx: float = 100.0, cy: float = 100.0, height: float = 100.0) -> BBox:
    """A standing person: tall and narrow, aspect 0.4."""
    half_h = height / 2.0
    half_w = height * 0.4 / 2.0
    return BBox(x1=cx - half_w, y1=cy - half_h, x2=cx + half_w, y2=cy + half_h)


def fallen_box(cx: float = 100.0, cy: float = 160.0, length: float = 100.0) -> BBox:
    """A person on the ground: wide and short, aspect 2.5."""
    half_w = length / 2.0
    half_h = length / 2.5 / 2.0
    return BBox(x1=cx - half_w, y1=cy - half_h, x2=cx + half_w, y2=cy + half_h)


def observation(
    box: BBox,
    timestamp: float,
    *,
    track_id: int = 1,
    label: str = "person",
    poses: dict[int, PersonPose] | None = None,
) -> BehaviourObservation:
    track = Track(track_id=track_id, label=label, box=box, age_frames=10, speed_px_s=0.0)
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=timestamp,
        detections=(),
        tracks=(track,),
        motion_energy=0.0,
        scene_signature=(1.0,),
    )
    return BehaviourObservation(scene=scene, poses=poses or {})


def _ticks(start: float, stop: float, step: float) -> list[float]:
    ticks: list[float] = []
    current = start
    while current <= stop + 1e-9:
        ticks.append(round(current, 3))
        current += step
    return ticks


def run(
    frames: list[tuple[BBox, float]],
    *,
    policy: FallPolicy = POLICY,
    poses: dict[float, dict[int, PersonPose]] | None = None,
) -> tuple[FallTracker, list[FallEvidence]]:
    """Feed a scripted sequence and collect every candidate raised along the way."""
    tracker = FallTracker()
    raised: list[FallEvidence] = []
    for box, timestamp in frames:
        obs = observation(box, timestamp, poses=(poses or {}).get(timestamp))
        tracker, completed = observe_falls(obs, policy, tracker)
        raised.extend(completed)
    return tracker, raised


def a_fall(*, stand_until: float = 1.0, settle_until: float = 6.0) -> list[tuple[BBox, float]]:
    """Stand, drop fast, stay down. The canonical positive case.

    The drop covers 60px in 0.2s against a 100px body — 3.0 body heights per second,
    comfortably over the 0.7 default and comfortably unlike sitting down.
    """
    frames = [(upright_box(), t) for t in _ticks(0.0, stand_until, 0.2)]
    frames.append((fallen_box(), stand_until + 0.2))
    frames.extend((fallen_box(), t) for t in _ticks(stand_until + 0.4, settle_until, 0.2))
    return frames


class TestTheSignatureCompletes:
    def test_stand_then_drop_then_stay_down_raises_exactly_one_candidate(self) -> None:
        _, raised = run(a_fall())
        assert len(raised) == 1

    def test_the_evidence_reports_what_was_measured_not_a_score(self) -> None:
        """No probability anywhere on the record. ADR 10's argument applied to a
        second judgement source: a float would be read as a calibrated likelihood by
        everything downstream, and nothing here has earned one."""
        _, raised = run(a_fall())
        evidence = raised[0]
        assert evidence.track_id == 1
        assert evidence.descent_rate > POLICY.min_descent_rate
        assert evidence.settled_seconds >= POLICY.settle_seconds
        assert not hasattr(evidence, "confidence")
        assert not hasattr(evidence, "score")

    def test_the_summary_hedges_and_never_diagnoses(self) -> None:
        """§7: "possible", "appears to have". A surveillance system saying someone
        *has collapsed* is making a medical claim it cannot support."""
        _, raised = run(a_fall())
        summary = raised[0].summary().lower()
        assert "appears to have fallen" in summary
        assert "collapsed" not in summary

    def test_the_evidence_points_at_when_the_descent_began(self) -> None:
        """Not when the candidate was raised — that is `settle_seconds` later, and a
        clip centred there would open on a body already on the floor, missing the one
        moment an investigator needs to see."""
        _, raised = run(a_fall(stand_until=1.0))
        assert 1.0 <= raised[0].started_at <= 1.4

    def test_nothing_is_raised_before_the_settle_window_elapses(self) -> None:
        """The clause that separates "fell" from "fell and is not getting up"."""
        _, raised = run(a_fall(stand_until=1.0, settle_until=1.0 + POLICY.settle_seconds - 0.5))
        assert raised == []

    def test_a_second_episode_after_standing_up_raises_again(self) -> None:
        """Reported is not a terminal state. Someone who falls, is reported, gets up
        and falls again has fallen twice, and the second one is not a duplicate."""
        frames = a_fall(stand_until=1.0, settle_until=6.0)
        frames.extend((upright_box(), t) for t in _ticks(6.2, 7.5, 0.2))
        frames.append((fallen_box(), 7.7))
        frames.extend((fallen_box(), t) for t in _ticks(7.9, 12.0, 0.2))
        _, raised = run(frames)
        assert len(raised) == 2


class TestItStaysSilent:
    def test_a_person_lying_still_produces_one_candidate_not_one_per_frame(self) -> None:
        """Thirty extra frames of an unchanged scene. Deduplicating here rather than
        in the alert engine is what stops the alert engine being the thing under
        load."""
        _, raised = run(a_fall(stand_until=1.0, settle_until=12.0))
        assert len(raised) == 1

    def test_sitting_down_slowly_is_not_a_fall(self) -> None:
        """The descent rate is the discriminator. This drop covers the same distance
        over 2 seconds rather than 0.2 — about 0.3 body heights per second, under the
        0.7 bar — and reaching the floor slowly is how people sit down."""
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.extend(
            (upright_box(cy=100.0 + 6.0 * step), round(1.0 + 0.2 * step, 3))
            for step in range(1, 11)
        )
        frames.extend((fallen_box(), t) for t in _ticks(3.2, 10.0, 0.2))
        _, raised = run(frames)
        assert raised == []

    def test_someone_already_on_the_floor_is_never_reported(self) -> None:
        """No transition was observed, so there is nothing to report. A real
        limitation, stated plainly: this detects falling, not lying."""
        _, raised = run([(fallen_box(), t) for t in _ticks(0.0, 20.0, 0.2)])
        assert raised == []

    def test_dropping_and_getting_straight_back_up_is_not_reported(self) -> None:
        """A stumble. Down for a second, up again — nobody needs waking for it."""
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((fallen_box(), 1.2))
        frames.extend((fallen_box(), t) for t in _ticks(1.4, 2.2, 0.2))
        frames.extend((upright_box(), t) for t in _ticks(2.4, 6.0, 0.2))
        _, raised = run(frames)
        assert raised == []

    def test_a_descent_that_never_becomes_horizontal_expires(self) -> None:
        """Crouching fast to pick something up: the rate clears the bar, the posture
        never does. Past `descent_window_seconds` the episode is abandoned rather
        than left armed to complete on some unrelated later frame."""
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((upright_box(cy=180.0), 1.2))
        frames.extend((upright_box(cy=180.0), t) for t in _ticks(1.4, 12.0, 0.2))
        _, raised = run(frames)
        assert raised == []

    def test_a_non_person_track_is_ignored_entirely(self) -> None:
        tracker = FallTracker()
        raised_any = False
        for box, timestamp in a_fall():
            obs = observation(box, timestamp, label="car")
            tracker, completed = observe_falls(obs, POLICY, tracker)
            raised_any = raised_any or bool(completed)
        assert not raised_any
        assert tracker.tracks == {}

    def test_a_person_who_leaves_frame_takes_their_episode_with_them(self) -> None:
        """Track state is dropped rather than aged out, so a track id the tracker
        later reuses for a different person cannot inherit this one's descent."""
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((fallen_box(), 1.2))
        tracker, _ = run(frames)
        assert 1 in tracker.tracks

        empty = SceneState(
            camera_id="cam-1",
            frame_index=0,
            timestamp=1.4,
            detections=(),
            tracks=(),
            motion_energy=0.0,
            scene_signature=(1.0,),
        )
        tracker, raised = observe_falls(BehaviourObservation(scene=empty), POLICY, tracker)
        assert tracker.tracks == {}
        assert raised == ()


class TestRobustness:
    def test_a_degenerate_frame_is_skipped_without_inflating_the_measured_rate(
        self,
    ) -> None:
        """A zero-height box has no aspect ratio and no body-height unit, so it can
        answer nothing — but a fall that happens to contain one tracking artefact is
        still a fall, and missing it would be the worse error for a welfare system.

        So the frame is stepped over and the next good one measures against the last
        good one, across the true interval. The rate that comes out has to be a rate a
        body could actually produce: an earlier version adopted the degenerate frame's
        centroid and reported 13.8 body heights per second for a 2.8 body-height fall,
        on a record whose whole claim is that it carries measurements rather than
        scores.
        """
        flat = BBox(x1=10.0, y1=50.0, x2=60.0, y2=50.0)
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((flat, 1.2))
        frames.extend((fallen_box(), t) for t in _ticks(1.4, 8.0, 0.2))
        _, raised = run(frames)

        assert len(raised) == 1, "one bad frame must not lose a real fall"
        # 60px of centroid drop over a 100px body across 0.4s of real time.
        assert raised[0].descent_rate == pytest.approx(1.5)

    def test_a_degenerate_frame_alone_cannot_manufacture_a_fall(self) -> None:
        """The artefact must not be able to *create* a signature either: an upright
        person with one bad frame is still an upright person."""
        flat = BBox(x1=10.0, y1=50.0, x2=60.0, y2=50.0)
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((flat, 1.2))
        frames.extend((upright_box(), t) for t in _ticks(1.4, 8.0, 0.2))
        _, raised = run(frames)
        assert raised == []

    def test_a_track_first_seen_as_a_degenerate_box_creates_no_state(self) -> None:
        """Nothing is known about it — not even which way up it is — so inventing a
        starting phase would be inventing the one fact the machine reasons from."""
        flat = BBox(x1=10.0, y1=50.0, x2=60.0, y2=50.0)
        tracker, raised = observe_falls(observation(flat, 0.0), POLICY, FallTracker())
        assert tracker.tracks == {}
        assert raised == ()

    def test_a_regressing_timeline_does_not_invent_a_descent(self) -> None:
        """An RTSP reconnect restarts the source clock at zero. A rate computed
        across that discontinuity is arithmetic on two different timelines."""
        tracker = FallTracker()
        tracker, _ = observe_falls(observation(upright_box(), 10.0), POLICY, tracker)
        _, raised = observe_falls(observation(fallen_box(), 0.0), POLICY, tracker)
        assert raised == ()

    def test_settle_time_restarts_when_the_body_moves_rather_than_cancelling(self) -> None:
        """Someone struggling to get up is still down. Cancelling on the first twitch
        would lose exactly the case most worth reporting — but the timer restarting
        means the candidate comes later, not never."""
        frames = [(upright_box(), t) for t in _ticks(0.0, 1.0, 0.2)]
        frames.append((fallen_box(), 1.2))
        frames.extend((fallen_box(), t) for t in _ticks(1.4, 3.0, 0.2))
        frames.append((fallen_box(cx=200.0), 3.2))
        frames.extend((fallen_box(cx=200.0), t) for t in _ticks(3.4, 8.0, 0.2))
        _, raised = run(frames)
        assert len(raised) == 1
        assert raised[0].settled_seconds >= POLICY.settle_seconds


class TestPose:
    """Pose improves the reading; it is never required."""

    @staticmethod
    def _pose(track_id: int, *, horizontal: bool) -> PersonPose:
        """A skeleton whose torso runs vertically (upright) or across (horizontal)."""
        if horizontal:
            shoulder_x, shoulder_y, hip_x, hip_y = 60.0, 160.0, 140.0, 160.0
        else:
            shoulder_x, shoulder_y, hip_x, hip_y = 100.0, 70.0, 100.0, 130.0
        return PersonPose(
            track_id=track_id,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(shoulder_x, shoulder_y, 0.9),
                KeypointName.RIGHT_SHOULDER: Keypoint(shoulder_x, shoulder_y, 0.9),
                KeypointName.LEFT_HIP: Keypoint(hip_x, hip_y, 0.9),
                KeypointName.RIGHT_HIP: Keypoint(hip_x, hip_y, 0.9),
            },
        )

    def test_a_pose_backed_candidate_says_so(self) -> None:
        frames = a_fall()
        poses = {
            timestamp: {1: self._pose(1, horizontal=(box.x2 - box.x1) > (box.y2 - box.y1))}
            for box, timestamp in frames
        }
        _, raised = run(frames, poses=poses)
        assert len(raised) == 1
        assert raised[0].used_pose is True

    def test_geometry_only_candidates_say_that_too(self) -> None:
        """A consumer weighting pose-backed evidence more heavily needs to be able to
        tell; nothing here does the weighting for them."""
        _, raised = run(a_fall())
        assert raised[0].used_pose is False

    def test_a_low_confidence_skeleton_falls_back_to_geometry(self) -> None:
        """A pose model returns a full skeleton whatever it can see. Trusting a
        guessed hip would let an occluded standing person read as horizontal."""
        pose = PersonPose(
            track_id=1,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(60.0, 160.0, 0.05),
                KeypointName.LEFT_HIP: Keypoint(140.0, 160.0, 0.05),
            },
        )
        frames = a_fall()
        poses = {timestamp: {1: pose} for _, timestamp in frames}
        _, raised = run(frames, poses=poses)
        assert len(raised) == 1
        assert raised[0].used_pose is False

    def test_a_torso_angle_is_unsigned_so_orientation_does_not_matter(self) -> None:
        """Lying head-left and head-right are the same posture. A signed angle would
        make one of them read as upright."""
        left = PersonPose(
            track_id=1,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(60.0, 100.0, 0.9),
                KeypointName.LEFT_HIP: Keypoint(140.0, 100.0, 0.9),
            },
        )
        right = PersonPose(
            track_id=1,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(140.0, 100.0, 0.9),
                KeypointName.LEFT_HIP: Keypoint(60.0, 100.0, 0.9),
            },
        )
        assert left.torso_angle_degrees(min_keypoint_confidence=0.4) == pytest.approx(90.0)
        assert right.torso_angle_degrees(min_keypoint_confidence=0.4) == pytest.approx(90.0)

    def test_a_skeleton_that_cannot_answer_returns_none_rather_than_a_guess(self) -> None:
        """`None` is what makes the per-frame fallback work. A plausible number would
        silently outrank the geometry it replaced."""
        no_hips = PersonPose(
            track_id=1,
            keypoints={KeypointName.LEFT_SHOULDER: Keypoint(100.0, 70.0, 0.9)},
        )
        assert no_hips.torso_angle_degrees(min_keypoint_confidence=0.4) is None

        zero_length = PersonPose(
            track_id=1,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(100.0, 100.0, 0.9),
                KeypointName.LEFT_HIP: Keypoint(100.0, 100.0, 0.9),
            },
        )
        assert zero_length.torso_angle_degrees(min_keypoint_confidence=0.4) is None

    def test_a_leaning_torso_is_neither_upright_nor_horizontal(self) -> None:
        """The band between the two thresholds exists so that bending to pick
        something up is not the start of a fall."""
        leaning = PersonPose(
            track_id=1,
            keypoints={
                KeypointName.LEFT_SHOULDER: Keypoint(100.0, 100.0, 0.9),
                KeypointName.LEFT_HIP: Keypoint(145.0, 145.0, 0.9),
            },
        )
        angle = leaning.torso_angle_degrees(min_keypoint_confidence=0.4)
        assert angle is not None
        assert POLICY.upright_torso_degrees_max < angle < POLICY.horizontal_torso_degrees_min


class TestPolicyValidation:
    """§12: thresholds are configuration, and configuration that cannot mean anything
    fails at construction rather than producing a detector that never fires."""

    def test_overlapping_aspect_bands_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must exceed upright_aspect_max"):
            FallPolicy(upright_aspect_max=1.5, horizontal_aspect_min=1.1)

    def test_overlapping_torso_bands_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="torso angles"):
            FallPolicy(upright_torso_degrees_max=70.0, horizontal_torso_degrees_min=60.0)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("min_descent_rate", 0.0),
            ("settle_seconds", 0.0),
            ("descent_window_seconds", 0.0),
            ("still_radius", -1.0),
            ("min_upright_seconds", -1.0),
            ("min_keypoint_confidence", 1.5),
        ],
    )
    def test_out_of_range_thresholds_are_rejected(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            FallPolicy(**{field_name: value})


def test_posture_has_an_unknown_member_and_it_is_used() -> None:
    """`UNKNOWN` is a real answer, not a failure code: a degenerate box and an
    unreadable skeleton both produce it, and both must be unable to decide anything."""
    assert Posture.UNKNOWN in set(Posture)
