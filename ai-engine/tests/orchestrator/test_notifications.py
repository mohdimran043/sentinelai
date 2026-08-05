"""The welfare-notification dispatcher (Task 10).

Delivery is deliberately not on the escalation worker's path: a webhook that
hangs must cost the notification it belongs to and nothing else — not the GPU
admission slot, not the next describe, not a camera's clip. These tests pin
that separation, the bound on how much can queue up behind a dead endpoint, and
the shutdown that drains rather than leaks.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import TracebackType
from uuid import uuid4

import pytest

from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.orchestrator.notifications import NotificationDispatcher
from sentinel_ai.ports.notifier import Notifier, WelfareNote
from tests.fakes.io import FakeNotifier, HangingNotifier


def a_note(camera_id: str = "cam-1") -> WelfareNote:
    return WelfareNote(
        event_id=uuid4(),
        camera_id=camera_id,
        label="Front Door",
        zone="corridor",
        occurred_at=1_700_000_000.0,
        severity="high",
        description="A person is lying motionless on the floor.",
        concerns=(
            WelfareConcern(
                kind=ConcernKind.COLLAPSE,
                confidence=Confidence.LIKELY,
                evidence="prone and not moving",
            ),
        ),
        clip_uri="s3://sentinel-clips/cam-1/evt.mp4",
    )


class FlakyNotifier(Notifier):
    """Fails the first `fail_first` deliveries, then records — one instance across
    a whole test, mirroring `test_scheduler.FlakyPublisher`, so no test has to reach
    into the dispatcher to swap its notifier out mid-run."""

    def __init__(self, fail_first: int) -> None:
        self.notes: list[WelfareNote] = []
        self.attempts = 0
        self._fail_first = fail_first

    async def notify(self, note: WelfareNote) -> None:
        self.attempts += 1
        if self.attempts <= self._fail_first:
            raise RuntimeError("endpoint refused")
        self.notes.append(note)


class Worker:
    """Runs the dispatcher's single worker for the duration of an `async with`."""

    def __init__(self, dispatcher: NotificationDispatcher) -> None:
        self._dispatcher = dispatcher
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> NotificationDispatcher:
        self._task = asyncio.create_task(self._dispatcher.run())
        return self._dispatcher

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


async def test_a_submitted_note_reaches_the_notifier() -> None:
    notifier = FakeNotifier()
    note = a_note()
    async with Worker(NotificationDispatcher(notifier)) as dispatcher:
        assert dispatcher.submit(note) is True
        await dispatcher.drain()

    assert notifier.notes == [note]
    assert dispatcher.delivered == 1


async def test_a_notifier_that_raises_does_not_stop_the_worker() -> None:
    """The port asks implementations to raise on failure so the caller can decide
    what a failed notification means; this is that decision. One dead delivery must
    not take the worker down, or the first webhook outage silently ends every
    notification for the lifetime of the process."""
    notifier = FlakyNotifier(fail_first=1)
    dispatcher = NotificationDispatcher(notifier)
    second = a_note("cam-2")
    async with Worker(dispatcher):
        dispatcher.submit(a_note())
        await dispatcher.drain()
        dispatcher.submit(second)
        await dispatcher.drain()

    assert dispatcher.failures == 1
    assert notifier.notes == [second], "the worker survived the failure and kept delivering"


async def test_a_notifier_that_raises_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A swallowed failure with no trace is indistinguishable from a delivered one.
    The event id has to be in the line: it is what ties a missing alert back to the
    event it was about."""
    note = a_note()
    dispatcher = NotificationDispatcher(FakeNotifier(error=RuntimeError("endpoint refused")))
    with caplog.at_level(logging.WARNING, logger="sentinel_ai.orchestrator.notifications"):
        async with Worker(dispatcher):
            dispatcher.submit(note)
            await dispatcher.drain()

    assert any(str(note.event_id) in record.getMessage() for record in caplog.records)


async def test_a_failure_whose_message_is_empty_is_still_named_by_type(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`str(TimeoutError())` is `""`, and the dispatch deadline is the failure this
    frame catches most often against the shipped webhook adapter — which swallows
    every other delivery failure itself. A line reading `... (camera cam-1): ` with
    nothing after the colon names no cause and reads like a formatting bug, so the
    exception's `repr` — which always names the type — is what goes in the line."""
    note = a_note()
    dispatcher = NotificationDispatcher(HangingNotifier(), timeout_seconds=0.0)
    with caplog.at_level(logging.WARNING, logger="sentinel_ai.orchestrator.notifications"):
        async with Worker(dispatcher):
            dispatcher.submit(note)
            await asyncio.wait_for(dispatcher.drain(), timeout=5.0)

    assert dispatcher.failures == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any("TimeoutError" in message for message in messages), messages
    assert all(not message.rstrip().endswith(":") for message in messages), (
        "the reason is empty for this exception; the type is what makes the line legible"
    )


async def test_a_hanging_notifier_is_timed_out_and_the_worker_moves_on() -> None:
    """`timeout_seconds=0.0` costs no wall-clock time: `asyncio.timeout` schedules
    the deadline at `loop.time()`, which the loop reaches on its next pass. The
    `wait_for` is the failure path only — against a dispatcher with no timeout the
    worker never finishes this note and this test must fail rather than hang."""
    notifier = HangingNotifier()
    dispatcher = NotificationDispatcher(notifier, timeout_seconds=0.0)
    async with Worker(dispatcher):
        dispatcher.submit(a_note())
        try:
            await asyncio.wait_for(dispatcher.drain(), timeout=5.0)
        except TimeoutError:
            pytest.fail("the hung notify() was never timed out: the dispatch timeout is dead")

    assert notifier.started.is_set(), "test setup: notify() must actually have been entered"
    assert notifier.cancelled is True, "the timeout must cancel the call, not merely abandon it"
    assert dispatcher.failures == 1


async def test_the_queue_drops_and_counts_when_it_is_full() -> None:
    """Bounded, like the escalation queue and for the same reason: with no worker
    draining it — a dead endpoint holding every delivery open — an unbounded queue
    grows for as long as the process runs. A drop is counted and logged, never
    raised, because `submit` is called from the escalation worker and must not be
    able to cost the event."""
    dispatcher = NotificationDispatcher(FakeNotifier(), maxsize=1)
    assert dispatcher.submit(a_note()) is True
    assert dispatcher.submit(a_note()) is False
    assert dispatcher.dropped == 1


async def test_submit_never_awaits_the_delivery() -> None:
    """The property the whole class exists for. `submit` is a `def`, not an `async
    def`, so "the escalation worker awaits a webhook" is unrepresentable rather
    than merely discouraged — and a note handed to a notifier that never returns
    still comes straight back to the caller."""
    dispatcher = NotificationDispatcher(HangingNotifier())
    async with Worker(dispatcher):
        assert dispatcher.submit(a_note()) is True
        assert dispatcher.delivered == 0


async def test_drain_returns_once_the_queue_is_empty() -> None:
    notifier = FakeNotifier()
    dispatcher = NotificationDispatcher(notifier)
    async with Worker(dispatcher):
        for index in range(3):
            dispatcher.submit(a_note(f"cam-{index}"))
        await dispatcher.drain()

    assert len(notifier.notes) == 3
