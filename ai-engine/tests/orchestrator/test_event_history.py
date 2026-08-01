"""Tests for the console's bounded per-camera event ring.

The point of every test here is the *bound*. A ring test that only checks that
what went in came back out passes just as happily against an unbounded list, and
an unbounded list in a process that runs for months is the leak this module
exists to prevent — so each test below is written to fail against
`list.append` with no eviction, or against a single global ring shared by every
camera.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore
from sentinel_ai.orchestrator.event_history import (
    RECENT_EVENTS_PER_CAMERA,
    RecentEventLog,
)


def _event(
    camera_id: str = "cam-1",
    *,
    occurred_at: float = 0.0,
    threat: float = 0.5,
    description: str = "A person walks by.",
    description_unavailable: bool = False,
    event_id: UUID | None = None,
) -> Event:
    return Event(
        event_id=event_id or uuid4(),
        camera_id=camera_id,
        occurred_at=occurred_at,
        source_timestamp=occurred_at,
        reason=EscalationReason.NEW_SALIENT_TRACK,
        threat=ThreatScore.from_value(threat),
        description=description,
        suggested_action="Review the clip.",
        labels=("person",),
        track_ids=(7,),
        description_unavailable=description_unavailable,
    )


def test_an_empty_log_reports_a_camera_with_no_events_rather_than_failing() -> None:
    history = RecentEventLog(capacity=3).history("cam-1")
    assert history.events == ()
    assert history.latest is None
    assert history.camera_id == "cam-1"
    assert history.capacity == 3


def test_events_come_back_oldest_first() -> None:
    """Chronological order is what a threat-over-time chart plots directly. A log
    that returned newest-first would draw every chart backwards."""
    log = RecentEventLog(capacity=10)
    for index in range(3):
        log.record(_event(occurred_at=float(index)))
    assert [event.occurred_at for event in log.history("cam-1").events] == [0.0, 1.0, 2.0]


def test_the_ring_evicts_the_oldest_once_it_is_full() -> None:
    """The bound is enforced, and it is enforced by dropping the *oldest*.

    Fails against an unbounded list (which returns all five) and against a ring
    that drops the newest instead (which returns 0, 1, 2).
    """
    log = RecentEventLog(capacity=3)
    for index in range(5):
        log.record(_event(occurred_at=float(index)))

    history = log.history("cam-1")
    assert len(history.events) == 3, "the bound was not enforced — this is the leak"
    assert [event.occurred_at for event in history.events] == [2.0, 3.0, 4.0]
    assert history.latest is not None
    assert history.latest.occurred_at == 4.0


def test_the_ring_stays_bounded_over_many_more_events_than_its_capacity() -> None:
    """A single overflow can be survived by an off-by-one; ten times the capacity
    cannot. This is the property that matters for a long-running process."""
    log = RecentEventLog(capacity=4)
    for index in range(40):
        log.record(_event(occurred_at=float(index)))
    assert len(log.history("cam-1").events) == 4


def test_a_busy_camera_cannot_evict_a_quiet_camera_s_history() -> None:
    """The reason the ring is per camera and not global.

    A single shared ring of the same capacity would answer this with an empty
    list for `cam-quiet`: its one event is the oldest of the eleven recorded and
    would have been evicted by `cam-busy`'s tenth. That is precisely the history
    an operator wants — the corridor that escalated once — and it is the first
    thing a global ring throws away.
    """
    log = RecentEventLog(capacity=3)
    log.record(_event("cam-quiet", occurred_at=0.0, description="The quiet one."))
    for index in range(10):
        log.record(_event("cam-busy", occurred_at=float(index + 1)))

    quiet = log.history("cam-quiet")
    assert len(quiet.events) == 1, "a busy camera evicted a quiet camera's history"
    assert quiet.events[0].description == "The quiet one."
    assert len(log.history("cam-busy").events) == 3, "the busy camera is still bounded"


def test_each_camera_gets_its_own_full_capacity_and_only_its_own_events() -> None:
    """The mirror of the isolation test: per camera means the capacity is per
    camera, not a global budget divided between them — and a camera's history must
    contain that camera's events and nobody else's.

    A single global ring of capacity 3 passes a bare length check here (three
    entries come back for both cameras) and fails this one: the ring would hold
    only `cam-2`'s three events, and `cam-1` would be served its neighbour's.
    """
    log = RecentEventLog(capacity=3)
    for camera_id in ("cam-1", "cam-2"):
        for index in range(3):
            log.record(_event(camera_id, occurred_at=float(index)))

    for camera_id in ("cam-1", "cam-2"):
        events = log.history(camera_id).events
        assert len(events) == 3
        assert {event.camera_id for event in events} == {camera_id}


def test_a_recorded_event_keeps_the_fields_the_console_plots_and_shows() -> None:
    event = _event(occurred_at=12.5, threat=0.9, description="Two people fighting.")
    log = RecentEventLog(capacity=3)
    log.record(event)

    recorded = log.history("cam-1").events[0]
    assert recorded.event_id == event.event_id
    assert recorded.camera_id == "cam-1"
    assert recorded.occurred_at == 12.5
    assert recorded.source_timestamp == 12.5
    assert recorded.threat_score == 0.9
    assert recorded.severity is Severity.CRITICAL
    assert recorded.description == "Two people fighting."
    assert recorded.reason is EscalationReason.NEW_SALIENT_TRACK
    assert recorded.labels == ("person",)
    assert recorded.track_ids == (7,)
    assert recorded.description_unavailable is False


def test_a_degraded_event_is_kept_and_stays_flagged() -> None:
    """A §9 event the vision model never described is exactly the one an operator
    needs in the console, so it is retained — and it must arrive still saying so,
    or the console shows a metadata stand-in as if it were a scene description."""
    log = RecentEventLog(capacity=3)
    log.record(_event(description_unavailable=True, description="new_salient_track: person"))
    latest = log.history("cam-1").latest
    assert latest is not None
    assert latest.description_unavailable is True


def test_the_history_snapshot_does_not_change_under_later_recording() -> None:
    """The API renders `latest` and `events` from one snapshot so the live panel and
    the chart cannot disagree. That only holds if the snapshot is a copy."""
    log = RecentEventLog(capacity=5)
    log.record(_event(occurred_at=1.0))
    snapshot = log.history("cam-1")
    log.record(_event(occurred_at=2.0))

    assert len(snapshot.events) == 1
    assert snapshot.latest is not None
    assert snapshot.latest.occurred_at == 1.0


class TestClipUri:
    """T3. The ring is written at assembly, before the clip exists; the URI has to
    reach it afterwards or the console can never link an event to its footage."""

    def test_an_event_starts_with_no_clip_uri(self) -> None:
        """Assembly precedes `_attach_clip`, and that ordering is load-bearing: spec
        §9 requires the event to be recorded even when the clip never finishes."""
        log = RecentEventLog(capacity=3)
        log.record(_event())
        assert log.history("cam-1").events[0].clip_uri is None

    def test_attaching_a_clip_backfills_the_entry_that_is_already_there(self) -> None:
        log = RecentEventLog(capacity=3)
        event = _event(occurred_at=5.0)
        log.record(event)

        assert log.attach_clip("cam-1", event.event_id, "s3://clips/a.mp4") is True

        (entry,) = log.history("cam-1").events
        assert entry.clip_uri == "s3://clips/a.mp4"
        assert entry.event_id == event.event_id
        assert entry.occurred_at == 5.0, "back-filling must not rewrite anything else"

    def test_the_backfill_does_not_append_a_second_copy_or_reorder_the_ring(self) -> None:
        """A ring that grows an entry per clip would double-count every event on the
        console's chart, and would evict twice as fast as its documented bound."""
        log = RecentEventLog(capacity=10)
        events = [_event(occurred_at=float(index)) for index in range(3)]
        for event in events:
            log.record(event)

        log.attach_clip("cam-1", events[0].event_id, "s3://clips/first.mp4")

        entries = log.history("cam-1").events
        assert [entry.event_id for entry in entries] == [event.event_id for event in events]
        assert [entry.clip_uri for entry in entries] == ["s3://clips/first.mp4", None, None]

    def test_the_backfill_bumps_the_sequence_so_a_live_client_sees_the_update(self) -> None:
        """The updated entry is a *new version* of the same event, not a duplicate of
        it. A client that already holds the pre-clip copy must be able to tell it is
        looking at something newer — see `TestLiveStream`."""
        log = RecentEventLog(capacity=3)
        event = _event()
        log.record(event)
        before = log.history("cam-1").events[0].sequence

        log.attach_clip("cam-1", event.event_id, "s3://clips/a.mp4")

        assert log.history("cam-1").events[0].sequence > before

    def test_attaching_to_an_event_that_has_aged_out_is_a_no_op(self) -> None:
        """The ring is bounded and silently lossy by design; a late clip for an
        evicted event must not resurrect it, and must not raise into the worker."""
        log = RecentEventLog(capacity=1)
        first = _event(occurred_at=1.0)
        log.record(first)
        log.record(_event(occurred_at=2.0))

        assert log.attach_clip("cam-1", first.event_id, "s3://clips/gone.mp4") is False
        assert len(log.history("cam-1").events) == 1
        assert log.history("cam-1").events[0].occurred_at == 2.0

    def test_a_clip_only_lands_on_its_own_cameras_ring(self) -> None:
        log = RecentEventLog(capacity=3)
        event = _event("cam-1")
        log.record(event)
        log.record(_event("cam-2"))

        assert log.attach_clip("cam-2", event.event_id, "s3://clips/a.mp4") is False
        assert log.history("cam-2").events[0].clip_uri is None


def test_a_capacity_below_one_is_rejected() -> None:
    """Zero would silently discard every event and report an always-empty console."""
    with pytest.raises(ValueError, match="capacity"):
        RecentEventLog(capacity=0)


def test_the_shipped_default_bound_is_finite_and_documented() -> None:
    """The default is what production actually runs with; a test that only ever
    exercises a capacity of 3 says nothing about it."""
    assert isinstance(RECENT_EVENTS_PER_CAMERA, int)
    assert 50 <= RECENT_EVENTS_PER_CAMERA <= 1000
    assert RecentEventLog().capacity == RECENT_EVENTS_PER_CAMERA
