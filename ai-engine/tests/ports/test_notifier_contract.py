"""Contract tests for the `Notifier` port (Task 4).

Follows the shape of `tests/ports/test_port_contracts.py`: the port itself is
abstract-and-uninstantiable, and its `FakeNotifier` is exercised directly
rather than mocked, so these tests would catch a fake that drifted from the
port it claims to satisfy.
"""

from __future__ import annotations

import inspect
from uuid import uuid4

import pytest

from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.ports.notifier import Notifier, WelfareNote
from tests.fakes.io import FakeNotifier


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


def test_notifier_is_abstract_and_cannot_be_instantiated() -> None:
    assert inspect.isabstract(Notifier)
    with pytest.raises(TypeError):
        Notifier()  # type: ignore[abstract]


def test_fake_notifier_satisfies_notifier() -> None:
    assert issubclass(FakeNotifier, Notifier)


def test_welfare_note_is_frozen() -> None:
    note = a_note()
    with pytest.raises(AttributeError):
        note.severity = "critical"  # type: ignore[misc]


def test_a_note_with_no_clip_is_valid() -> None:
    """A notification can legitimately precede a clip that failed to write;
    losing the alert because the evidence upload failed would be exactly
    backwards for a welfare system."""
    note = a_note(clip_uri=None)
    assert note.clip_uri is None


class TestFakeNotifier:
    async def test_it_records_notes_in_order(self) -> None:
        notifier = FakeNotifier()
        first = a_note(description="first")
        second = a_note(description="second")

        await notifier.notify(first)
        await notifier.notify(second)

        assert notifier.notes == [first, second]

    async def test_it_can_be_configured_to_fail(self) -> None:
        """Task 6's webhook adapter and Task 10's dispatch both need to prove
        that a failing notifier does not take down the pipeline; the fake
        must be able to simulate that failure."""
        notifier = FakeNotifier(error=ConnectionError("webhook unreachable"))

        with pytest.raises(ConnectionError):
            await notifier.notify(a_note())

        assert notifier.notes == [], "a failed notify must not be recorded as sent"
