"""The single entry point the API delegates to (spec §5.4, §5.7)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from uuid import UUID

from sentinel_ai.orchestrator.event_history import CameraEventHistory
from sentinel_ai.orchestrator.registry import ModelRegistry
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport

logger = logging.getLogger(__name__)

__all__ = ["EngineService", "UnknownCameraError"]

_DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS = 60.0
"""How often `ResidentSet.sweep_idle` is polled, not the idle-unload window itself
(that is `ModelSpec.idle_unload_seconds`, spec §5.4's 600s for the VLM). Comfortably
below 600s so the unload fires within a bounded margin of its deadline rather than
being detected up to a whole interval late."""

_DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 10.0
"""Ceiling on how long `stop()` waits for the escalation queue to empty.

Bounded on purpose: the drain runs the VLM, and a wedged or unusually slow describe
must not be able to hold a `docker compose down` open indefinitely. Wide enough that
the ordinary case — a handful of queued escalations, each already covered by
`vlm_timeout_seconds` — completes well inside it."""


class _IdleSweeper:
    """Periodically calls `sweep`; owns no task or model-registry state of its own.

    Split out exactly like `_ReconnectLoop` (Task 14, adapters/sources/rtsp.py): a
    plain "wait, then act" sequencer, so it is unit-testable with an injected clock
    and sleep function — no real wall-clock time, no asyncio task, no GPU. Composed
    with `ResidentSet.sweep_idle` as `sweep` here; nothing about it is specific to
    resident-set eviction.
    """

    def __init__(
        self,
        sweep: Callable[[float], Awaitable[None]],
        *,
        interval_seconds: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sweep = sweep
        self._interval = interval_seconds
        self._clock = clock
        self._sleep = sleep

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            await self._sleep(self._interval)
            if should_stop():
                return
            await self._sweep(self._clock())


class UnknownCameraError(KeyError):
    """Raised by `telemetry()`/`describe_now()` for an unregistered camera id.

    A `KeyError` subclass so existing `except KeyError` call sites keep working, and
    a distinct type so the API layer (Task 10) can map it to HTTP 404 without
    catching every other `KeyError` in the process too.
    """

    def __init__(self, camera_id: str) -> None:
        super().__init__(f"unknown camera: {camera_id}")
        self.camera_id = camera_id


class EngineService:
    """Owns the process's cameras, models, and scheduler; the API talks to nothing else."""

    def __init__(
        self,
        cameras: Mapping[str, CameraRunner],
        registry: ModelRegistry,
        resident_set: ResidentSet,
        scheduler: VlmScheduler,
        required_model_keys: tuple[str, ...],
        clock: Callable[[], float],
        idle_sweep_interval_seconds: float = _DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        shutdown_drain_timeout_seconds: float = _DEFAULT_SHUTDOWN_DRAIN_TIMEOUT_SECONDS,
    ) -> None:
        self._cameras = dict(cameras)
        self._registry = registry
        self._resident_set = resident_set
        self._scheduler = scheduler
        self._required_model_keys = required_model_keys
        self._clock = clock
        self._shutdown_drain_timeout_seconds = shutdown_drain_timeout_seconds
        # Held apart rather than in one list: `stop()` has to unwind them in a
        # specific order, and a flat list of tasks cannot express that order.
        self._scheduler_task: asyncio.Task[None] | None = None
        self._camera_tasks: list[asyncio.Task[None]] = []
        self._sweeper_task: asyncio.Task[None] | None = None
        self._idle_sweeper = _IdleSweeper(
            resident_set.sweep_idle,
            interval_seconds=idle_sweep_interval_seconds,
            clock=clock,
            sleep=sleep,
        )

    async def start(self) -> None:
        """Load the required models, then start the VLM worker, every camera, and the
        periodic idle sweep (spec §5.4's 600s VLM idle-unload, S7-deferred to here)."""
        await self._resident_set.ensure(self._required_model_keys, self._clock())
        self._scheduler_task = asyncio.create_task(self._scheduler.run())
        self._camera_tasks = [
            asyncio.create_task(runner.run()) for runner in self._cameras.values()
        ]
        self._sweeper_task = asyncio.create_task(
            self._idle_sweeper.run_forever(should_stop=lambda: False)
        )

    async def stop(self) -> None:
        """Unwind in phase order, and dead-letter the queue if we are cut short.

        A cancelled `stop()` **stays** cut short: the `CancelledError` propagates, the
        remaining phases do not run, and this never degrades into a slow graceful
        shutdown. What it must not do is take the queue down with it. Spec §9 says an
        anomaly event is never lost to an infrastructure failure, and a forced shutdown
        — uvicorn's `--timeout-graceful-shutdown`, a supervisor kill, a timed-out ASGI
        lifespan — is one: an operator who SIGTERMs the service does not thereby consent
        to losing the anomaly it was midway through recording. So on the way out,
        whatever is still queued or in flight is spilled to the same `DeadLetterSpool`
        the drain-cap path already uses, and is recoverable from disk afterwards.

        See `_unwind` for the phase ordering itself and `_spill_to_dead_letter` for why
        the spill is safe to run from a cancelled coroutine.
        """
        try:
            await self._unwind()
        except asyncio.CancelledError as cancellation:
            await self._spill_to_dead_letter(cancellation)
            raise

    async def _spill_to_dead_letter(self, reason: BaseException) -> None:
        """Dead-letter whatever the cancelled shutdown was still holding.

        Cancellation can arrive in any phase, so the worker may well still be running
        and `abandon_pending()` may not be called against a live worker — it empties the
        queue, and racing the worker for the same request could publish and dead-letter
        it twice. Hence cancel-then-await here before spilling. That await is wrapped
        rather than left bare because the caller is already being cancelled: a second
        cancellation delivered while we wait must not cost us the spill, which is the
        whole point of being here.

        This cannot itself turn into a slow shutdown. Cancelling a worker that is
        awaiting a describe unwinds it at its next suspension point, and
        `DeadLetterSpool.store` is a synchronous local write behind an `async def` — no
        broker, no network, nothing that can hang. That is exactly why §9's last resort
        is the right mechanism here and a publish attempt would not be.
        """
        task, self._scheduler_task = self._scheduler_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        await self._scheduler.abandon_pending(reason)

    async def _unwind(self) -> None:
        """Unwind in producer-then-consumer order, draining the queue in between.

        The order is the whole point. `CameraRunner.run()`'s `finally` deliberately
        *submits* the escalation whose clip was still recording rather than dropping
        it — spec §9 forbids losing an anomaly event to a lifecycle failure, and a
        shutdown is one. Cancelling every task in one pass, as this used to, issued
        the scheduler worker's `cancel()` before any runner had unwound, so that
        carefully preserved event landed in a queue whose only consumer was already
        dead. It was counted as an escalation, never published, and never counted as
        dropped. So:

          1. the idle sweeper first — it has no ordering constraint of its own, but
             stopping it here means it cannot evict the VLM out from under the drain;
          2. then the cameras, awaited to completion, so every `finally` runs and
             every in-flight escalation is on the queue while the worker still lives;
          3. then the drain, bounded, so a wedged VLM cannot hold shutdown open;
          4. then the worker itself;
          5. and if the cap in (3) expired, whatever the worker never got to is
             dead-lettered instead of evaporating with it.

        Step 5 exists because the cap in step 3 reproduced the very defect steps 1-4
        were added to fix. `suppress(TimeoutError)` followed by cancelling the worker
        left the still-queued escalations counted, unpublished, and absent from every
        counter — C3's signature again, through a different door. See
        `VlmScheduler.abandon_pending`.

        Resident models are deliberately *not* unloaded: process exit reclaims the
        VRAM, and an unload here would only slow shutdown down.
        """
        await self._cancel(self._sweeper_task)
        self._sweeper_task = None

        await self._cancel(*self._camera_tasks)
        self._camera_tasks = []

        expiry: TimeoutError | None = None
        if self._scheduler_task is not None and not self._scheduler_task.done():
            try:
                await asyncio.wait_for(
                    self._scheduler.drain(), timeout=self._shutdown_drain_timeout_seconds
                )
            except TimeoutError as error:
                expiry = error
                logger.error(
                    "the escalation queue did not drain within %.1fs; "
                    "dead-lettering whatever is left rather than losing it",
                    self._shutdown_drain_timeout_seconds,
                )
        await self._cancel(self._scheduler_task)
        self._scheduler_task = None
        if expiry is not None:
            # After the worker is cancelled, never before: `abandon_pending()` empties
            # the queue, so running it against a live worker would race it for the same
            # request and could publish and dead-letter the same event.
            await self._scheduler.abandon_pending(expiry)

    @staticmethod
    async def _cancel(*tasks: asyncio.Task[None] | None) -> None:
        """Cancel all, then await all — never cancel-and-await one at a time.

        Awaiting each task before cancelling the next would let a still-running task
        keep producing work for a phase that has already been torn down.

        The `cancelling()` bookkeeping is what makes the ordering in `stop()` mean
        anything. A blanket `suppress(CancelledError)` here cannot tell "the task I
        awaited ended via cancellation" — expected, keep unwinding — from "*my own*
        `stop()` was cancelled while I awaited it". Swallowing the second case lets
        `stop()` silently skip the remaining phases and cancel the scheduler while a
        camera's `finally` is still running unawaited, which drops exactly the event
        that ordering exists to preserve. That is reachable in production: uvicorn's
        `--timeout-graceful-shutdown` cancels the ASGI lifespan's shutdown task, and
        this phase deliberately takes longer than the old one-pass teardown did.

        `Task.cancelling()` counts cancellations requested *of this task*, so a rise
        across the await means the cancellation was aimed at us, not delivered by the
        task we were waiting on. Re-raise so the caller learns shutdown was cut short.

        An ordinary exception out of an awaited task is a different thing entirely and
        must **not** propagate. `CameraRunner.run()` deliberately re-raises a producer
        failure so a mid-stream decode error is not mistaken for a clean end of stream
        (`runner.py`; `FileSource` surfaces `_pump_error` the same way, and a
        `detector.detect()` CUDA fault does it on RTSP). Letting that out of here
        aborted phases 2-4 of `stop()`: the scheduler worker was never cancelled and
        leaked, and every *other* camera's `finally` never ran — so its recording clip
        was never aborted and the in-flight escalation that `finally` exists to submit
        was never submitted. One camera's decode error losing another camera's evidence
        is exactly what spec §9 forbids, so it is logged and shutdown carries on.
        """
        live = [task for task in tasks if task is not None]
        for task in live:
            task.cancel()

        current = asyncio.current_task()
        for task in live:
            requested_before = current.cancelling() if current is not None else 0
            try:
                await task
            except asyncio.CancelledError:
                if current is not None and current.cancelling() > requested_before:
                    raise
            except Exception:
                logger.exception("a task failed during shutdown; continuing to unwind")

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return tuple(runner.telemetry() for runner in self._cameras.values())

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        return self._get_runner(camera_id).telemetry()

    def event_history(self, camera_id: str) -> CameraEventHistory:
        """The console's bounded, volatile view of what this camera recently produced.

        Not the event store — RabbitMQ plus the Phase 1C consumer is that; see
        `orchestrator/event_history.py`.

        `_get_runner` is called for its side effect: the history lives on the
        scheduler, which is shared by every camera and knows nothing about which ids
        are configured, so without this an id nobody ever heard of would return a
        cheerful empty list instead of the 404 every other camera route gives. A
        *configured* camera that has simply not escalated yet still returns an empty
        list, which is a different answer to a different question.
        """
        self._get_runner(camera_id)
        return self._scheduler.event_history(camera_id)

    def health(self) -> dict[str, HealthReport]:
        return self._registry.health()

    async def describe_now(self, camera_id: str) -> UUID:
        return await self._get_runner(camera_id).describe_now()

    def _get_runner(self, camera_id: str) -> CameraRunner:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise UnknownCameraError(camera_id) from None
