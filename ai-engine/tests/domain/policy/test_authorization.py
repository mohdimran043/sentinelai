"""§11, as executable prose: a single uncertain frame must never raise an alert.

§11 is a whole section of the specification saying one thing, and the first class below
is that thing. Everything else is the ways of not-quite-satisfying it that must stay
silent — which carry more weight here than in any other detector, because a false
unauthorised-person alert is an accusation about a specific identifiable individual.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from sentinel_ai.domain.identity import (
    FaceEmbedding,
    FaceObservation,
    FaceQuality,
    MatchCandidate,
)
from sentinel_ai.domain.policy.authorization import (
    AuthorizationPolicy,
    AuthorizationTracker,
    UnauthorizedFinding,
    observe_authorization,
)

POLICY = AuthorizationPolicy()
GOOD = FaceQuality(box_pixels=120, detector_confidence=0.95, frontality=0.8)
POOR = FaceQuality(box_pixels=12, detector_confidence=0.95, frontality=0.8)
SIDE_ON = FaceQuality(box_pixels=120, detector_confidence=0.95, frontality=0.05)

EMBEDDING = FaceEmbedding.of((1.0, 0.0, 0.0))


def face(
    *,
    track_id: int = 1,
    quality: FaceQuality = GOOD,
    similarity: float | None = None,
    authorized_here: bool = True,
    name: str = "Employee A",
    timestamp: float = 0.0,
) -> FaceObservation:
    candidates = (
        ()
        if similarity is None
        else (
            MatchCandidate(
                person_id=uuid4(),
                display_name=name,
                similarity=similarity,
                authorized_here=authorized_here,
            ),
        )
    )
    return FaceObservation(
        track_id=track_id,
        embedding=EMBEDDING,
        quality=quality,
        timestamp=timestamp,
        candidates=candidates,
    )


def run(
    script: list[tuple[FaceObservation, float]], policy: AuthorizationPolicy = POLICY
) -> tuple[AuthorizationTracker, list[UnauthorizedFinding]]:
    tracker = AuthorizationTracker()
    findings: list[UnauthorizedFinding] = []
    for observation, now in script:
        tracker, found = observe_authorization((observation,), policy, tracker, now=now)
        findings.extend(found)
    return tracker, findings


def unknown_for(
    seconds: float, *, step: float = 0.5, track_id: int = 1
) -> list[tuple[FaceObservation, float]]:
    """One unrecognised person, seen clearly, for `seconds`."""
    script: list[tuple[FaceObservation, float]] = []
    now = 0.0
    while now <= seconds + 1e-9:
        script.append((face(track_id=track_id, similarity=0.1), now))
        now = round(now + step, 3)
    return script


class TestTheSpecRequirement:
    """§11: four consistent unknown frames of one tracked person, above the quality
    floor, over a minimum duration — then, and only then, an alert."""

    def test_a_person_who_stays_unrecognised_is_eventually_reported(self) -> None:
        _, findings = run(unknown_for(3.0))
        assert len(findings) == 1
        assert findings[0].observations >= POLICY.min_observations
        assert findings[0].duration_seconds >= POLICY.min_duration_seconds

    def test_a_single_unknown_frame_reports_nothing(self) -> None:
        """The requirement, in its simplest form."""
        _, findings = run([(face(similarity=0.1), 0.0)])
        assert findings == []

    def test_enough_frames_but_not_enough_time_reports_nothing(self) -> None:
        """Frame count alone is a frame-rate measurement — four frames is a sixth of a
        second at 25 fps."""
        script = [(face(similarity=0.1), round(0.01 * i, 3)) for i in range(10)]
        _, findings = run(script)
        assert findings == []

    def test_enough_time_but_not_enough_frames_reports_nothing(self) -> None:
        script = [(face(similarity=0.1), 0.0), (face(similarity=0.1), 10.0)]
        _, findings = run(script)
        assert findings == []

    def test_it_is_reported_once_not_once_per_frame(self) -> None:
        _, findings = run(unknown_for(60.0))
        assert len(findings) == 1


class TestItStaysSilent:
    def test_one_good_recognition_immunises_the_track(self) -> None:
        """An authorised employee is face-on perhaps one frame in three. If the frames
        after a recognition re-accumulated, the system would alert on the people it had
        just recognised — the single worst failure this feature can have, and one an
        early version of this module actually had."""
        script = [
            (face(similarity=0.1), 0.0),
            (face(similarity=0.1), 0.5),
            (face(similarity=0.9), 1.0),
        ]
        script.extend((face(similarity=0.1), 1.5 + 0.5 * i) for i in range(10))
        _, findings = run(script)
        assert findings == []

    def test_a_face_too_small_to_read_is_discarded_not_counted(self) -> None:
        """A 12-pixel face embeds near the middle of the space and is mildly similar
        to everybody. Counting it as evidence of *not* matching is counting noise."""
        script = [(face(quality=POOR, similarity=0.1), round(0.5 * i, 3)) for i in range(20)]
        _, findings = run(script)
        assert findings == []

    def test_a_face_turned_away_is_discarded_not_counted(self) -> None:
        script = [(face(quality=SIDE_ON, similarity=0.1), round(0.5 * i, 3)) for i in range(20)]
        _, findings = run(script)
        assert findings == []

    def test_a_low_confidence_detection_is_discarded(self) -> None:
        unsure = FaceQuality(box_pixels=120, detector_confidence=0.2, frontality=0.8)
        script = [(face(quality=unsure, similarity=0.1), round(0.5 * i, 3)) for i in range(20)]
        _, findings = run(script)
        assert findings == []

    def test_four_different_people_seen_once_each_report_nothing(self) -> None:
        """Four unknown faces belonging to four people walking past is not evidence
        about any of them."""
        script = [(face(track_id=i, similarity=0.1), round(0.5 * i, 3)) for i in range(1, 9)]
        _, findings = run(script)
        assert findings == []

    def test_a_recognised_authorised_person_is_never_reported(self) -> None:
        script = [(face(similarity=0.95), round(0.5 * i, 3)) for i in range(40)]
        _, findings = run(script)
        assert findings == []


class TestWhatTheOperatorIsTold:
    def test_the_near_match_is_carried_not_just_the_verdict(self) -> None:
        """§10 asks for the reference and the confidence. "Nobody matched" and
        "matched at 0.38 against a 0.42 threshold" are very different things, and the
        second frequently means the threshold is wrong rather than the person is."""
        script = [(face(similarity=0.38, name="Employee B"), round(0.5 * i, 3)) for i in range(10)]
        _, findings = run(script)
        assert findings[0].best_person_name == "Employee B"
        assert findings[0].best_similarity == pytest.approx(0.38)

    def test_the_strongest_look_is_kept_not_the_latest(self) -> None:
        """The best look this camera ever got is the most informative thing to hand an
        operator; the last frame is often the worst."""
        script = [
            (face(similarity=0.10), 0.0),
            (face(similarity=0.39), 0.5),
            (face(similarity=0.05), 1.0),
            (face(similarity=0.05), 1.5),
            (face(similarity=0.05), 2.0),
            (face(similarity=0.05), 2.5),
        ]
        _, findings = run(script)
        assert findings[0].best_similarity == pytest.approx(0.39)

    def test_an_enrolled_person_in_the_wrong_place_is_named_as_such(self) -> None:
        """A known person somewhere they should not be reads very differently from an
        unknown face, and an operator acts on it differently."""
        script = [
            (face(similarity=0.9, authorized_here=False, name="Employee C"), round(0.5 * i, 3))
            for i in range(10)
        ]
        _, findings = run(script)
        assert findings[0].best_person_authorized_elsewhere is True
        assert "not authorised on this camera" in findings[0].summary()

    def test_a_weak_near_match_is_not_called_an_enrolled_person(self) -> None:
        """Below the threshold the closest face is just the closest face. Naming them
        as an unauthorised *person* would be an accusation the evidence cannot
        support."""
        script = [
            (face(similarity=0.2, authorized_here=False), round(0.5 * i, 3)) for i in range(10)
        ]
        _, findings = run(script)
        assert findings[0].best_person_authorized_elsewhere is False

    def test_the_summary_claims_only_what_the_system_can_know(self) -> None:
        """ "Did not match any face authorised on this camera" is a statement about
        this system's enrolled faces. It is not a statement about whether the person
        belongs, and it must not read like one."""
        _, findings = run(unknown_for(3.0))
        summary = findings[0].summary().lower()
        assert "did not match" in summary
        assert "intruder" not in summary
        assert "trespass" not in summary

    def test_the_finding_points_at_when_they_were_first_seen(self) -> None:
        """Not when the evidence completed — a clip centred there opens after the
        moment worth reviewing."""
        _, findings = run(unknown_for(3.0))
        assert findings[0].first_seen == 0.0


class TestHeadTurns:
    def test_frames_where_the_face_is_unreadable_do_not_reset_the_evidence(self) -> None:
        """A person walking down a corridor is face-on perhaps one frame in three.
        Dropping the evidence every time they turn their head means
        `min_observations` is never reached on any real camera."""
        script = [
            (face(similarity=0.1), 0.0),
            (face(quality=SIDE_ON, similarity=0.1), 0.5),
            (face(similarity=0.1), 1.0),
            (face(quality=SIDE_ON, similarity=0.1), 1.5),
            (face(similarity=0.1), 2.0),
            (face(similarity=0.1), 2.5),
        ]
        _, findings = run(script)
        assert len(findings) == 1


class TestPolicyValidation:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("match_threshold", 1.5),
            ("match_threshold", -2.0),
            ("min_observations", 0),
            ("min_duration_seconds", -1.0),
            ("min_box_pixels", 0),
            ("min_detector_confidence", 1.5),
            ("min_frontality", -0.1),
            ("cooldown_seconds", -1.0),
        ],
    )
    def test_out_of_range_thresholds_are_rejected(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            AuthorizationPolicy(**{field_name: value})  # type: ignore[arg-type]


class TestRecognitionIsSticky:
    """The failure mode that matters most, pinned directly.

    Caught by a test rather than by a site: an early version deleted the track's state
    on recognition, so the unreadable frames that follow began a fresh episode and the
    system alerted on the authorised employee it had just recognised.
    """

    def test_a_long_run_of_unreadable_frames_after_recognition_never_alerts(self) -> None:
        script = [(face(similarity=0.95), 0.0)]
        script.extend((face(similarity=0.05), round(0.5 * i, 3)) for i in range(1, 60))
        _, findings = run(script)
        assert findings == []

    def test_recognition_survives_frames_where_no_face_was_readable_at_all(self) -> None:
        script = [(face(similarity=0.95), 0.0)]
        script.extend(
            (face(quality=SIDE_ON, similarity=0.05), round(0.5 * i, 3)) for i in range(1, 30)
        )
        script.extend((face(similarity=0.05), round(15.0 + 0.5 * i, 3)) for i in range(20))
        _, findings = run(script)
        assert findings == []

    def test_an_unrecognised_person_is_still_reported_normally(self) -> None:
        """The stickiness must not have disabled the feature."""
        _, findings = run(unknown_for(3.0))
        assert len(findings) == 1
