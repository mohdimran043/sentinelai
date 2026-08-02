"""`DeadLetterSpool` — spec §9's last resort."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from sentinel_ai.adapters.publishers.dead_letter import DeadLetterSpool
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.ports.event_publisher import FailedEventSink
from sentinel_ai.ports.notifier import WelfareNote


def an_event(camera_id: str = "cam-1") -> Event:
    return Event(
        event_id=uuid4(),
        camera_id=camera_id,
        occurred_at=12.5,
        reason=EscalationReason.NEW_SALIENT_TRACK,
        threat=ThreatScore.from_value(0.4),
        description="A person is near the door.",
        suggested_action="Monitor.",
        labels=("person",),
        track_ids=(1,),
    )


def a_welfare_note(camera_id: str = "cam-1") -> WelfareNote:
    return WelfareNote(
        event_id=uuid4(),
        camera_id=camera_id,
        label="Front Door",
        zone="entrance",
        occurred_at=12.5,
        severity="high",
        description="A person is lying motionless on the floor.",
        concerns=(
            WelfareConcern(
                kind=ConcernKind.COLLAPSE,
                confidence=Confidence.LIKELY,
                evidence="Person is prone and not moving.",
            ),
        ),
        clip_uri=None,
    )


def test_it_is_a_failed_event_sink() -> None:
    assert issubclass(DeadLetterSpool, FailedEventSink)


async def test_the_event_and_the_reason_it_failed_are_both_recorded(tmp_path: Path) -> None:
    sink = DeadLetterSpool(tmp_path)
    event = an_event()

    await sink.store(event, ValueError("'' is too short"))

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "ValueError"
    assert "is too short" in record["error"]
    assert record["event"]["camera_id"] == "cam-1"
    assert record["event"]["description"] == "A person is near the door."
    assert record["event"]["event_id"] == str(event.event_id)


async def test_it_does_not_use_encode_event(tmp_path: Path) -> None:
    """A payload the schema rejects is one of the exact cases that reaches this sink.
    Validating on the way in would throw the evidence away for the same reason it was
    lost before — so an event `encode_event` refuses must still be written."""
    sink = DeadLetterSpool(tmp_path)

    await sink.store(an_event(camera_id=""), ValueError("schema"))

    (path,) = sorted(tmp_path.glob("*.json"))
    assert json.loads(path.read_text(encoding="utf-8"))["event"]["camera_id"] == ""


async def test_the_record_carries_a_type_discriminator_for_events(tmp_path: Path) -> None:
    """The spool now holds two record shapes (`Event`, from the publisher
    path, and `WelfareNote`, from Task 6's webhook notifier), distinguished
    only by which fields happen to be present. This module's own docstring
    calls the output "for an operator (or a repair script)" — a repair
    script written against the `Event` shape would `KeyError` on `reason`
    for a welfare note it did not know to expect. `record_type` makes the
    shape explicit instead of inferred."""
    sink = DeadLetterSpool(tmp_path)

    await sink.store(an_event(), ValueError("one"))

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["record_type"] == "Event"


async def test_the_record_carries_a_type_discriminator_for_welfare_notes(tmp_path: Path) -> None:
    sink = DeadLetterSpool(tmp_path)

    await sink.store(a_welfare_note(), RuntimeError("webhook responded with HTTP 503"))

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["record_type"] == "WelfareNote"


async def test_two_failures_in_the_same_nanosecond_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    sink = DeadLetterSpool(tmp_path)
    await sink.store(an_event(), ValueError("one"))
    await sink.store(an_event(), ValueError("two"))
    assert len(list(tmp_path.glob("*.json"))) == 2


async def test_no_partial_file_is_ever_left_behind(tmp_path: Path) -> None:
    """Write-then-rename, same as the publisher's spool: a `.json` is complete or
    absent, never half-written."""
    sink = DeadLetterSpool(tmp_path)
    await sink.store(an_event(), ValueError("one"))
    assert list(tmp_path.glob("*.json.tmp")) == []


async def test_an_unwritable_directory_never_raises_at_the_caller(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The port forbids raising: this runs inside the escalation worker's own error
    handling, where an exception would poison the worker iteration on top of losing
    the event. The log line becomes the last copy."""
    sink = DeadLetterSpool(tmp_path)
    tmp_path.chmod(0o500)
    try:
        with caplog.at_level("ERROR"):
            await sink.store(an_event(), ValueError("one"))
    finally:
        tmp_path.chmod(0o700)

    assert "the event is lost" in caplog.text
    assert list(tmp_path.glob("*.json")) == []
