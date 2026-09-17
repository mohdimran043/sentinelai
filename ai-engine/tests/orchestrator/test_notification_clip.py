"""Which clip a notification carries.

The short one when somebody is about to be woken up, the full one otherwise. The
decision is the point: sending the long clip to a phone at 3am does not fail loudly, it
just gets watched less, and sending the short one to an investigator loses the context
they came for.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore
from sentinel_ai.domain.policy.alerting import severity_rank

FULL = "s3://sentinel-clips/cam-1/evt.mp4"
SHORT = "s3://sentinel-clips/cam-1/evt-notify.mp4"


def an_event(severity: Severity, clip_uri: str | None = FULL) -> Event:
    return Event(
        event_id=uuid4(),
        camera_id="cam-1",
        occurred_at=1_700_000_000.0,
        source_timestamp=1.0,
        reason=EscalationReason.FALL_SUSPECTED,
        threat=ThreatScore(value=0.9, severity=severity),
        description="a person appears to have fallen",
        suggested_action="check on them",
        labels=("person",),
        track_ids=(1,),
        subject_track_ids=(1,),
        clip_uri=clip_uri,
    )


class TestSeverityRank:
    def test_critical_outranks_high_outranks_low(self) -> None:
        """`Severity` is a `StrEnum`, so `>` on it compares alphabetically and puts
        `critical` *below* `low`. Every comparison goes through `severity_rank` because
        that bug is silent and points the wrong way."""
        assert severity_rank(Severity.CRITICAL) > severity_rank(Severity.HIGH)
        assert severity_rank(Severity.HIGH) > severity_rank(Severity.MEDIUM)
        assert severity_rank(Severity.MEDIUM) > severity_rank(Severity.LOW)

    def test_the_alphabetical_trap_is_real(self) -> None:
        # Documents why the helper exists at all, so nobody "simplifies" it away.
        assert Severity.CRITICAL.value < Severity.LOW.value


class _Stub:
    def __init__(self, minimum: Severity) -> None:
        self._notify_clip_min_severity = minimum


def _choose(minimum: Severity, request: object, event: Event) -> str | None:
    from sentinel_ai.orchestrator.scheduler import VlmScheduler

    return VlmScheduler._clip_for_note(_Stub(minimum), request, event)  # type: ignore[arg-type]


class TestChoice:
    @pytest.mark.parametrize("severity", [Severity.CRITICAL, Severity.HIGH])
    def test_urgent_notes_get_the_short_clip(self, severity: Severity) -> None:
        assert _choose(Severity.HIGH, _RequestWith(SHORT), an_event(severity)) == SHORT

    @pytest.mark.parametrize("severity", [Severity.MEDIUM, Severity.LOW])
    def test_unhurried_notes_keep_the_full_clip(self, severity: Severity) -> None:
        """Nobody is running anywhere, and the extra context is worth more than the
        shorter download."""
        assert _choose(Severity.HIGH, _RequestWith(SHORT), an_event(severity)) == FULL

    def test_the_full_clip_is_the_fallback_when_no_short_one_was_made(self) -> None:
        """A writer that could not cut one, or was configured not to. The note still
        goes, carrying what exists."""
        assert _choose(Severity.HIGH, _RequestWith(None), an_event(Severity.CRITICAL)) == FULL

    def test_no_clip_at_all_stays_no_clip(self) -> None:
        """Spec §9: a clip that never finished must not suppress the notification."""
        event = an_event(Severity.CRITICAL, clip_uri=None)
        assert _choose(Severity.HIGH, _RequestWith(SHORT), event) is None

    def test_a_lower_threshold_includes_more(self) -> None:
        assert _choose(Severity.LOW, _RequestWith(SHORT), an_event(Severity.LOW)) == SHORT


class _RequestWith:
    def __init__(self, notify_uri: str | None) -> None:
        self.clip = _HandleWith(notify_uri)


class _HandleWith:
    def __init__(self, notify_uri: str | None) -> None:
        self.notify_uri = notify_uri
