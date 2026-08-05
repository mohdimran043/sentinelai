"""Welfare notification delivery, off the escalation worker's path (Task 10).

Spec's order is process -> describe -> clip finalised -> publish -> notify, and
the last step is the only one whose failure mode is "somebody outside this
process did not answer". `VlmScheduler` decides *whether* a published event is
worth a person's attention (`domain/policy/notification.py` is the rule) and
hands the resulting `WelfareNote` here; everything after that — the network, the
timeout, the retry the adapter does on its own, the failure — happens on this
worker and nowhere else.

**Why it is not simply awaited where the event is published.** There is one VLM
worker and one GPU admission slot. An `await notifier.notify(...)` inside
`VlmScheduler._process` would hold that slot for the duration of an HTTP call to
an endpoint this process does not control — `WebhookNotifier`'s own documented
worst case is 18 seconds against its defaults, and a black-holed TCP connection
is worse still. At concurrency 1 that is every camera in the process waiting on
one webhook, and the clip whose escalation is next in the queue waiting with it.
Notification is best-effort; the pipeline is not.

**Why a queue and a worker rather than a task per note.** This codebase has
already been bitten by unmanaged `asyncio.create_task`, and a task per note is
unbounded by construction: a dead endpoint that holds every delivery open until
the timeout accumulates one live task per escalation for as long as the process
runs. The same shape `VlmScheduler` already uses — a bounded queue, drop-and-
count when it is full, exactly one worker, `drain()` for a shutdown that waits —
bounds the memory, gives shutdown one task to unwind instead of a set, and
delivers notes in the order the events were published. `EngineService` owns the
worker task and drains it on the way out, in the same phase ordering it already
applies to the escalation queue.

Nothing here decides *what* to notify about: that is policy and lives in
`domain/policy/notification.py`.
"""

from __future__ import annotations

import asyncio
import logging

from sentinel_ai.ports.notifier import Notifier, WelfareNote

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_NOTIFY_QUEUE_MAXSIZE", "NotificationDispatcher"]

DEFAULT_NOTIFY_QUEUE_MAXSIZE = 32
"""How many notes may wait on a notifier that is not keeping up.

Not a `Settings` field, unlike `vlm_queue_maxsize`: the escalation queue's bound is
a load-shedding decision an operator tunes against their camera count, while this
one exists only so a dead endpoint cannot grow the process without limit. Deep
enough that the ordinary burst — every camera in a site escalating at once — queues
rather than drops, since the arrival rate here is already bounded by the single VLM
worker upstream.
"""

DEFAULT_NOTIFY_TIMEOUT_SECONDS = 20.0
"""Wall-clock ceiling on one whole delivery. See `Settings.notifier_timeout_seconds`,
which is what a deployment actually sets; this default exists so a `NotificationDispatcher`
built without one is still bounded."""


class NotificationDispatcher:
    def __init__(
        self,
        notifier: Notifier,
        *,
        maxsize: int = DEFAULT_NOTIFY_QUEUE_MAXSIZE,
        timeout_seconds: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        self._notifier = notifier
        self._timeout_seconds = timeout_seconds
        self._queue: asyncio.Queue[WelfareNote] = asyncio.Queue(maxsize=maxsize)
        self._delivered = 0
        self._dropped = 0
        self._failures = 0

    def submit(self, note: WelfareNote) -> bool:
        """Non-blocking. Returns False and counts a drop when the queue is full.

        Deliberately not a coroutine, for the same reason `VlmScheduler.submit` is
        not: the caller is the escalation worker, and "the pipeline awaits a
        webhook" has to be unrepresentable rather than merely discouraged.
        """
        try:
            self._queue.put_nowait(note)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "notification queue full: dropping the welfare note for event %s (camera %s)",
                note.event_id,
                note.camera_id,
            )
            return False
        return True

    async def run(self) -> None:
        """The single worker loop. Cancel to stop."""
        while True:
            note = await self._queue.get()
            try:
                await self._deliver(note)
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            self._queue.task_done()

    async def drain(self) -> None:
        """Await delivery of everything currently queued.

        Two callers: tests, and `EngineService.stop()`, which drains this between
        cancelling the escalation worker and cancelling this one so that the notes
        the last escalations produced are actually delivered. It only returns once
        the queue is empty, so the shutdown caller bounds it with a timeout rather
        than trusting a remote endpoint to answer.
        """
        await self._queue.join()

    @property
    def delivered(self) -> int:
        return self._delivered

    @property
    def dropped(self) -> int:
        """Notes the queue refused under load. Distinct from `failures`, which is a
        note that reached the notifier and did not get through: one is this process
        shedding work, the other is somebody else's endpoint, and an operator needs
        to know which."""
        return self._dropped

    @property
    def failures(self) -> int:
        return self._failures

    async def _deliver(self, note: WelfareNote) -> None:
        """One delivery: bounded, counted, and never raising past this frame.

        The port asks implementations to raise on failure precisely so this caller
        can decide what a failed notification means for the pipeline. The decision
        is: nothing. A welfare notifier exists to observe the system, and an
        unreachable endpoint is an ordinary, expected state of the world — letting
        it out of here would kill the worker and silently end every notification for
        the lifetime of the process. So it is logged, counted, and dropped.

        `asyncio.timeout` is a wall-clock bound over whatever the adapter does
        internally, including its own retries. `WebhookNotifier` already bounds
        itself and never raises; this is not redundant with that, because `Notifier`
        the *port* promises neither — a future adapter, or a bug in this one, must
        not be able to park this worker forever.
        """
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._notifier.notify(note)
        except asyncio.CancelledError:
            # Shutdown, not a delivery failure. `asyncio.timeout` converts only its
            # *own* deadline into `TimeoutError`; a cancellation from outside stays a
            # `CancelledError` and belongs to `run()`, which unwinds the worker.
            raise
        except Exception as error:
            self._failures += 1
            # `%r`, not `%s`: the exception this frame catches most often is the
            # `TimeoutError` from the deadline just above it, and `str(TimeoutError())`
            # is the empty string — the message alone renders "... (camera cam-1): "
            # and names no cause at all, which is indistinguishable from a log line
            # with a formatting bug. `repr` always names the type and never leaves a
            # dangling colon, the same reason `WebhookNotifier` logs its connection and
            # timeout failures by exception type.
            logger.warning(
                "welfare notification failed for event %s (camera %s): %r",
                note.event_id,
                note.camera_id,
                error,
            )
        else:
            self._delivered += 1
