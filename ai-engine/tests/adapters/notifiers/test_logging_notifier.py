"""`LoggingNotifier` — the default `Notifier` (Task 5).

A deployment that configures no webhook still needs an observable trail, and
every later test of notification routing needs an adapter that touches no
network. Both are satisfied by logging one structured line per note at INFO.

Field selection is deliberate, not incidental: `logger.info("%s", note)` or
`logger.info(repr(note))` would both pass a naive "does it log" test while
silently widening what gets logged the moment a field is added to
`WelfareNote` later. So several tests here assert the *absence* of a
repr-shaped dump, not just the presence of the fields that matter.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest

from sentinel_ai.adapters.notifiers.logging import LoggingNotifier
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.ports.notifier import Notifier, WelfareNote


def a_note(**overrides: object) -> WelfareNote:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "label": "Front Door",
        "zone": "entrance",
        "occurred_at": 1_700_000_000.0,
        "severity": "high",
        "description": "A person is lying motionless on the floor.",
        "concerns": (
            WelfareConcern(
                kind=ConcernKind.COLLAPSE,
                confidence=Confidence.LIKELY,
                evidence="Person is prone and not moving.",
            ),
        ),
        "clip_uri": "s3://sentinel-clips/cam-1/evt.mp4",
    }
    defaults.update(overrides)
    return WelfareNote(**defaults)  # type: ignore[arg-type]


def test_it_is_a_notifier() -> None:
    assert issubclass(LoggingNotifier, Notifier)


async def test_it_logs_one_info_line_with_camera_id_severity_and_concern_kind(
    caplog: pytest.LogCaptureFixture,
) -> None:
    notifier = LoggingNotifier()
    note = a_note(camera_id="cam-7", severity="critical")

    with caplog.at_level(logging.INFO):
        await notifier.notify(note)

    records = [r for r in caplog.records if r.name.startswith("sentinel_ai")]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert "cam-7" in message
    assert "critical" in message
    assert "collapse" in message


async def test_a_note_with_no_concerns_still_logs(caplog: pytest.LogCaptureFixture) -> None:
    """A note with no concerns is still a real notification — see Task 4's
    reasoning about `clip_uri=None` for the same principle applied to the
    clip. It must not be treated as a no-op."""
    notifier = LoggingNotifier()
    note = a_note(concerns=())

    with caplog.at_level(logging.INFO):
        await notifier.notify(note)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO


async def test_unevidenced_concerns_are_visibly_distinguished_from_evidenced_ones(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`evidence_stated=False` means the model named a concern without
    describing what it saw. The log line must make that visible rather than
    presenting an unevidenced concern as though it had been described."""
    notifier = LoggingNotifier()
    unevidenced = WelfareConcern(
        kind=ConcernKind.DISTRESS,
        confidence=Confidence.POSSIBLE,
        evidence="reported, not described",
        evidence_stated=False,
    )
    evidenced = WelfareConcern(
        kind=ConcernKind.COLLAPSE,
        confidence=Confidence.LIKELY,
        evidence="Person is prone and not moving.",
        evidence_stated=True,
    )
    note = a_note(concerns=(unevidenced, evidenced))

    with caplog.at_level(logging.INFO):
        await notifier.notify(note)

    message = caplog.records[0].getMessage()
    assert "distress" in message
    assert "collapse" in message
    # Isolate each concern's own rendering (the list is comma-separated) so the
    # unevidenced marker's placement can be checked per-concern, not just for
    # presence anywhere in the line.
    segments = [segment.strip() for segment in message.split(",")]
    distress_segment = next(s for s in segments if "distress" in s)
    collapse_segment = next(s for s in segments if "collapse" in s)
    assert "unevidenced" in distress_segment
    assert "unevidenced" not in collapse_segment


async def test_the_note_is_never_logged_as_a_whole(caplog: pytest.LogCaptureFixture) -> None:
    """Never `repr(note)` or `%s` of the whole note: if a field is added to
    `WelfareNote` later it must not start appearing in logs without someone
    deciding it should."""
    notifier = LoggingNotifier()
    note = a_note(description="a very distinctive description sentinel xyzzy")

    with caplog.at_level(logging.INFO):
        await notifier.notify(note)

    message = caplog.records[0].getMessage()
    assert "WelfareNote(" not in message
    # `description` is on the port but is not one of the fields this adapter
    # is asked to log (camera id, severity, concern kinds) — its absence is
    # evidence the note was not dumped wholesale.
    assert "xyzzy" not in message


async def test_notify_never_raises_even_for_a_malformed_note(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A notifier that throws would take down the pipeline it exists to
    observe. Even a malformed note must produce a log line, not an
    exception."""
    notifier = LoggingNotifier()

    class NotActuallyAConcern:
        kind = "collapse"
        # Deliberately missing `confidence`/`evidence_stated` to provoke an
        # AttributeError inside formatting.

    malformed = a_note(concerns=(NotActuallyAConcern(),))

    with caplog.at_level(logging.INFO):
        await notifier.notify(malformed)  # must not raise

    assert len(caplog.records) == 1
