"""The welfare-notification routing rule (spec §5, Task 10).

Every case below is expressed against `concerns_to_notify` alone — no scheduler,
no notifier, no event loop — because the rule is policy and policy lives in
`domain/`. The scheduler's own tests then prove the rule is *reached*; these
prove it is *right*.
"""

from __future__ import annotations

import pytest

from sentinel_ai.domain.entities import Severity, ThreatScore
from sentinel_ai.domain.policy.notification import (
    NOTIFYING_SEVERITIES,
    concerns_to_notify,
)
from sentinel_ai.domain.welfare import (
    ConcernKind,
    Confidence,
    WelfareAssessment,
    WelfareConcern,
)

ALL_KINDS = frozenset(ConcernKind)


def a_concern(
    kind: ConcernKind = ConcernKind.COLLAPSE,
    confidence: Confidence = Confidence.LIKELY,
    evidence_stated: bool = True,
) -> WelfareConcern:
    return WelfareConcern(
        kind=kind,
        confidence=confidence,
        evidence="lying motionless near the wall, not moving",
        evidence_stated=evidence_stated,
    )


def route(
    *concerns: WelfareConcern,
    severity: Severity = Severity.HIGH,
    notify_on: frozenset[ConcernKind] = ALL_KINDS,
    min_confidence: Confidence = Confidence.LIKELY,
) -> tuple[WelfareConcern, ...]:
    return concerns_to_notify(
        WelfareAssessment(concerns=concerns),
        severity=severity,
        notify_on=notify_on,
        min_confidence=min_confidence,
    )


def test_a_likely_concern_a_camera_watches_for_is_routed() -> None:
    concern = a_concern()
    assert route(concern, notify_on=frozenset({ConcernKind.COLLAPSE})) == (concern,)


def test_a_kind_the_camera_does_not_watch_for_is_not_routed() -> None:
    """The first half of the rule. A camera configured only for `collapse` must
    stay silent about an altercation however confident the model is."""
    assert (
        route(a_concern(kind=ConcernKind.ALTERCATION), notify_on=frozenset({ConcernKind.COLLAPSE}))
        == ()
    )


def test_an_empty_notify_on_routes_nothing() -> None:
    """`notify_on: []` is a stored instruction meaning "notify nobody about this
    camera" (see `CameraConfig.notify_on`), not an unconfigured field. A rule that
    treated the empty set as "no filter configured, so allow everything" would turn
    the one setting an operator uses to mute a camera into the one that unmutes it.
    """
    assert route(a_concern(), severity=Severity.CRITICAL, notify_on=frozenset()) == ()


def test_a_possible_concern_below_the_cameras_minimum_confidence_is_not_routed() -> None:
    """The second half of the rule, isolated from the severity band: the camera's
    own bar is `likely`, and a `possible` concern misses it even at CRITICAL — the
    band can only ever *withhold* a `possible` concern, never promote one past the
    bar the operator set."""
    assert route(a_concern(confidence=Confidence.POSSIBLE), severity=Severity.CRITICAL) == ()


def test_a_possible_concern_at_the_cameras_minimum_still_needs_the_caution_band() -> None:
    """A camera that has explicitly lowered its bar to `possible` still does not
    notify on a `possible` concern alone — spec §5's "`possible` alone does not
    notify unless the threat score is already in the caution band or above". A
    single frame's maybe, with nothing else agreeing, is not worth waking someone."""
    assert (
        route(
            a_concern(confidence=Confidence.POSSIBLE),
            severity=Severity.LOW,
            min_confidence=Confidence.POSSIBLE,
        )
        == ()
    )


def test_a_possible_concern_at_the_cameras_minimum_routes_inside_the_caution_band() -> None:
    concern = a_concern(confidence=Confidence.POSSIBLE)
    assert route(concern, severity=Severity.MEDIUM, min_confidence=Confidence.POSSIBLE) == (
        concern,
    )


@pytest.mark.parametrize(
    ("severity", "routed"),
    [
        (Severity.INFO, False),
        (Severity.LOW, False),
        (Severity.MEDIUM, True),
        (Severity.HIGH, True),
        (Severity.CRITICAL, True),
    ],
)
def test_the_caution_band_begins_at_medium(severity: Severity, routed: bool) -> None:
    """Pins the exact boundary "caution band or above" resolves to, in both
    directions — a rule that read the band as `high` and above, or as `low` and
    above, fails one half of this table."""
    concern = a_concern(confidence=Confidence.POSSIBLE)
    expected = (concern,) if routed else ()
    assert route(concern, severity=severity, min_confidence=Confidence.POSSIBLE) == expected


def test_a_likely_concern_routes_below_the_caution_band() -> None:
    """The band gate applies to `possible` and to nothing else. A model that says
    `likely` about a collapse must reach a human even when the threat score that
    same describe produced was mild — the welfare opinion and the threat score are
    two different judgements, and this system exists to act on the first."""
    concern = a_concern(confidence=Confidence.LIKELY)
    assert route(concern, severity=Severity.INFO) == (concern,)


def test_the_threshold_matches_the_severity_band_the_scores_fall_into() -> None:
    """The resolution is stated as a severity, but spec §5 states it as a threat
    score. This pins the two together against `_SEVERITY_BANDS`, so a change to
    either one that broke the correspondence would fail here rather than silently
    move the notification threshold."""
    assert ThreatScore.from_value(0.4).severity in NOTIFYING_SEVERITIES
    assert ThreatScore.from_value(0.39).severity not in NOTIFYING_SEVERITIES


def test_only_the_concerns_the_camera_watches_for_are_carried() -> None:
    """One note per event, carrying the concerns that routed — not the whole
    assessment. A note that carried a kind the operator muted would leak exactly
    what `notify_on` exists to suppress."""
    collapse = a_concern(kind=ConcernKind.COLLAPSE)
    altercation = a_concern(kind=ConcernKind.ALTERCATION)
    routed = route(collapse, altercation, notify_on=frozenset({ConcernKind.COLLAPSE}))
    assert routed == (collapse,)


def test_an_assessment_with_no_concerns_routes_nothing() -> None:
    assert (
        concerns_to_notify(
            WelfareAssessment.none(),
            severity=Severity.CRITICAL,
            notify_on=ALL_KINDS,
            min_confidence=Confidence.POSSIBLE,
        )
        == ()
    )


def test_a_concern_the_model_never_evidenced_is_still_routed() -> None:
    """`evidence_stated=False` means the model named a concern and described
    nothing (`domain/welfare.py`). It is deliberately *not* a routing filter: a
    muted alarm is the worst outcome this system can produce, and "likely collapse,
    no detail given" is still a person worth looking at. The flag travels on the
    note instead, so whoever reads it can see the difference."""
    concern = a_concern(evidence_stated=False)
    assert route(concern) == (concern,)
