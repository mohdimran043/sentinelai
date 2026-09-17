"""The single entry point the API delegates to (spec §5.4, §5.7)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from uuid import UUID

from sentinel_ai.adapters.config.camera_file import CameraConfig
from sentinel_ai.domain.alert import Alert
from sentinel_ai.domain.capabilities import CameraCapabilities, ModelRole, required_roles
from sentinel_ai.domain.identity import AuthorizedPerson, EnrolledFace, FaceEmbedding
from sentinel_ai.domain.welfare import ConcernKind, Confidence
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.alert_persistence import AlertPersistence
from sentinel_ai.orchestrator.alerts import AlertRegister, UnknownAlertError
from sentinel_ai.orchestrator.event_history import CameraEventHistory, EventSubscription
from sentinel_ai.orchestrator.face_crop import reference_image
from sentinel_ai.orchestrator.registry import ModelRegistry
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.ports.clip_index import ClipIndex, ClipRecord, StorageUsage
from sentinel_ai.ports.clip_reader import ClipReader
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.face import FaceDetector, FaceStore
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import HealthReport

logger = logging.getLogger(__name__)

__all__ = ["EngineNotComposedError", "EngineService", "UnknownCameraError"]

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


def _stopped_telemetry(config: CameraConfig) -> CameraTelemetry:
    """What a camera that has never run this process looks like.

    Every counter zero, and here that is the truth rather than a placeholder: this
    process has not watched a single frame from it. The identity and the configuration
    are real, which is what makes the camera still findable, still labelled and still
    editable while it is switched off.
    """
    return CameraTelemetry(
        camera_id=config.camera_id,
        label=config.label,
        enabled=False,
        frames_seen=0,
        frames_dropped=0,
        detections_run=0,
        escalations=0,
        escalations_dropped=0,
        discontinuities=0,
        last_frame_at=None,
        last_escalation_at=None,
        zone=config.zone,
        capabilities=config.capabilities,
        notify_on=config.notify_on,
        notify_min_confidence=config.notify_min_confidence,
        clip_preroll_seconds=config.clip_preroll_seconds,
        clip_postroll_seconds=config.clip_postroll_seconds,
        summary_interval_seconds=config.summary_interval_seconds,
    )


class FaceCapabilityUnavailableError(RuntimeError):
    """A person-management request arrived at an engine with no face pipeline.

    Distinct from an unknown person: this is "nothing here can answer that", which the
    API reports as 503 rather than 404. A 404 would tell an operator the person does
    not exist, when the truth is that no camera on this deployment enables
    `person_authorization` and nobody is enrolled anywhere.
    """

    def __init__(self) -> None:
        super().__init__(
            "person authorization is not enabled on this engine: no camera has the "
            "capability, so no face model and no biometric store were built. Enable it "
            "on a camera in cameras.json, set SENTINEL_FACE_ENCRYPTION_KEY, and restart."
        )


class UnknownPersonError(KeyError):
    """No enrolled person with this id."""

    def __init__(self, person_id: UUID) -> None:
        self.person_id = person_id
        super().__init__(f"unknown person: {person_id}")


class CapabilityUnavailableError(RuntimeError):
    """An edit asked to enable a capability whose model this process never loaded.

    Which models exist is decided once, at startup, from the union over every
    configured camera (`main.build_models`). That is not a caching decision that
    could be relaxed: placing a 3B vision model is a multi-second download-and-place
    against a GPU that every other camera is sharing, and doing it inside an HTTP
    request would stall the whole site for the duration — on a request that a console
    would reasonably retry.

    So the honest answer is a refusal that names the restart, rather than a 200 that
    wrote the file and left the running camera unable to honour it. The file is *not*
    written: this is raised before the store is touched, so a refused edit leaves the
    record and the running camera agreeing, which is the invariant every other
    rejection on this path also keeps.
    """

    def __init__(self, camera_id: str, missing: frozenset[ModelRole]) -> None:
        self.camera_id = camera_id
        self.missing = missing
        names = ", ".join(sorted(role.value for role in missing))
        super().__init__(
            f"camera {camera_id!r}: this engine did not load the {names} model, so that "
            f"capability cannot be enabled without a restart. Which models load is "
            f"decided at startup from every camera's capabilities; add it to "
            f"cameras.json and restart the engine. Nothing was changed."
        )


class DuplicateCameraError(ValueError):
    """A camera with this id is already running. The API turns it into a 409.

    Here rather than in the composition root for `EngineNotComposedError`'s reason: the
    API layer maps it, and the API layer may not import `main`.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        super().__init__(f"camera {camera_id!r} already exists")


class UnknownCameraError(KeyError):
    """Raised by `telemetry()`/`describe_now()` for an unregistered camera id.

    A `KeyError` subclass so existing `except KeyError` call sites keep working, and
    a distinct type so the API layer (Task 10) can map it to HTTP 404 without
    catching every other `KeyError` in the process too.
    """

    def __init__(self, camera_id: str) -> None:
        super().__init__(f"unknown camera: {camera_id}")
        self.camera_id = camera_id


class EngineNotComposedError(RuntimeError):
    """Raised for a request that arrives before the engine has been composed.

    Lives here beside `UnknownCameraError` because `main.ComposedService` raises both
    and the API layer maps both, and the API layer may not import the composition root.

    Only `subscribe_events()` needs it: the read-only endpoints can answer an
    uncomposed engine honestly (no cameras, no models) and the per-camera ones already
    have `UnknownCameraError`, but "a live stream over a ring that does not exist yet"
    has no honest empty value — an open stream that can never carry anything is
    indistinguishable, to a console, from a quiet site. The API turns this into a 503,
    which is what it is: try again shortly. Unreachable through uvicorn, which
    completes lifespan startup before it serves a request.
    """


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
        available_roles: frozenset[ModelRole],
        detector: ObjectDetector | None,
        face: FaceDetector | None = None,
        face_store: FaceStore | None = None,
        alert_persistence: AlertPersistence | None = None,
        clip_reader: ClipReader | None = None,
        clip_index: ClipIndex | None = None,
        disabled: Sequence[CameraConfig] = (),
        wall_clock: Callable[[], float] = time.time,
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
        # Which model roles this process actually built, decided once at startup from
        # the union over every configured camera (spec §13). A capability edit may
        # only enable what is already here: loading a 3B vision model is a
        # multi-second download-and-place, and doing it under an HTTP request would
        # stall every camera sharing the GPU. See `update_camera_metadata`.
        # Required rather than defaulted, and the absence of a default is the point.
        # An `available_roles=frozenset()` default makes every capability edit refuse
        # itself, and a `detector=None` default silently detaches the detector from
        # any camera that is *edited* — both are failures a composition root can
        # forget into existence, and neither shows up until a camera has quietly
        # stopped seeing anything. Making them explicit turns both into a
        # construction error the type checker reports.
        self._available_roles = available_roles
        # Optional, and its absence is a supported configuration rather than a
        # degraded one: without a store the register behaves exactly as it did before
        # durability existed. Every call site below therefore has to cope with `None`,
        # which is why the flush is a helper rather than an inline await.
        self._alert_persistence = alert_persistence
        # The one shared detector (ADR 4), kept so a capability edit that newly needs
        # inference can hand it to the runner. `None` when no role needed one.
        self._detector = detector
        # Both `None` unless some camera enabled person authorisation. Held so the
        # people-management endpoints can enrol against the same store the cameras
        # search — two stores would mean enrolling somebody nobody recognises.
        self._face = face
        self._face_store = face_store
        # Optional for the same reason `alert_persistence` is: an engine with no object
        # store still detects, still escalates and still raises alerts — the rows just
        # carry no playable clip. `None` here is a deployment choice, not a fault, so
        # every read of it answers 404 rather than raising.
        self._clip_reader = clip_reader
        self._clip_index = clip_index
        # Cameras that are configured and deliberately not running, holding the
        # telemetry each had when it was stopped. Keyed the same way `_cameras` is, and
        # the two are disjoint by construction — `disable_camera` moves an id from one
        # to the other, `forget_disabled` moves it back.
        self._disabled: dict[str, CameraTelemetry] = {
            config.camera_id: _stopped_telemetry(config) for config in disabled
        }
        # `camera_id -> (source timestamp, wall time we saw it advance)`. See
        # `last_frame_epoch` for why a camera's own timeline cannot answer
        # "is this camera still delivering".
        self._frame_observations: dict[str, tuple[float, float]] = {}
        self._wall_clock = wall_clock
        self._shutdown_drain_timeout_seconds = shutdown_drain_timeout_seconds
        # Held apart rather than in one list: `stop()` has to unwind them in a
        # specific order, and a flat list of tasks cannot express that order.
        self._scheduler_task: asyncio.Task[None] | None = None
        # Keyed by camera id rather than a bare list, so one camera can be stopped
        # without stopping the rest — which is what `remove_camera` needs and what a
        # list made impossible.
        self._camera_tasks: dict[str, asyncio.Task[None]] = {}
        self._sweeper_task: asyncio.Task[None] | None = None
        self._notifications_task: asyncio.Task[None] | None = None
        self._idle_sweeper = _IdleSweeper(
            resident_set.sweep_idle,
            interval_seconds=idle_sweep_interval_seconds,
            clock=clock,
            sleep=sleep,
        )

    async def start(self) -> None:
        """Load the required models, then start the VLM worker, the notification
        worker, every camera, and the periodic idle sweep (spec §5.4's 600s VLM
        idle-unload, S7-deferred to here)."""
        # Before anything can write to the register: restoring into a register that a
        # camera had already added an alert to is refused outright (`restore` is not a
        # merge), and this is the only ordering that guarantees it is still empty.
        if self._alert_persistence is not None:
            await self._alert_persistence.restore()
            self._alert_persistence.start()
        await self._resident_set.ensure(self._required_model_keys, self._clock())
        self._scheduler_task = asyncio.create_task(self._scheduler.run())
        # Started here rather than left to whoever built the dispatcher: an unrun
        # notification worker is silent by construction — `submit()` still returns
        # True, the queue simply fills, and nothing anywhere reports that no note
        # has left the process. Reached through the scheduler so it is necessarily
        # the same dispatcher that receives the notes (T10).
        self._notifications_task = asyncio.create_task(self._scheduler.notifications.run())
        self._camera_tasks = {
            camera_id: asyncio.create_task(runner.run())
            for camera_id, runner in self._cameras.items()
        }
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
        # The notification worker goes too, and synchronously: `cancel()` cannot
        # itself be interrupted, so it holds even if the spill below never returns.
        # It is cancelled rather than drained because we are already being cancelled
        # — there is no time budget left to spend on a remote endpoint, and the
        # events themselves are what §9 protects, not the notes about them.
        notifications, self._notifications_task = self._notifications_task, None
        if notifications is not None:
            notifications.cancel()
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
             dead-lettered instead of evaporating with it;
          6. then, last, the notification queue those publishes filled — see
             `_drain_notifications` for why it can only be last.

        Step 5 exists because the cap in step 3 reproduced the very defect steps 1-4
        were added to fix. `suppress(TimeoutError)` followed by cancelling the worker
        left the still-queued escalations counted, unpublished, and absent from every
        counter — C3's signature again, through a different door. See
        `VlmScheduler.abandon_pending`.

        Resident models are deliberately *not* unloaded: process exit reclaims the
        VRAM, and an unload here would only slow shutdown down.

        A seventh phase writes the alert store, after step 6 for step 6's own reason:
        publishing is what opens an alert, so the register is not settled until the
        last publish has happened.
        """
        await self._cancel(self._sweeper_task)
        self._sweeper_task = None

        await self._cancel(*self._camera_tasks.values())
        self._camera_tasks = {}

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

        await self._drain_notifications()
        await self._stop_alert_persistence()

    async def _stop_alert_persistence(self) -> None:
        """Phase 7: stop the flush timer and write the register one last time.

        Last, because every phase above can still change the register — a camera's
        `finally` submits an escalation, the drain publishes it, and the publish is
        what opens an alert. Flushing before that would persist a register that was
        about to change and then not write the change.

        Never allowed to fail the shutdown. `AlertPersistence.flush` already swallows
        and logs a store error; this guard is for the store raising somewhere it does
        not, and the trade is the same one: triage state is worth a lot and it is not
        worth a shutdown that hangs or a traceback that hides the real cause.
        """
        if self._alert_persistence is None:
            return
        try:
            await self._alert_persistence.stop()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("could not write the alert store on shutdown")

    async def _drain_notifications(self) -> None:
        """Phase 6: deliver the notes the last escalations produced, then stop the
        worker.

        After step 5 and not before. The escalation worker is what *creates* notes,
        so draining this queue while it still lives would drain a queue that is still
        being filled; and `abandon_pending()` publishes nothing — its events are the
        §9 degraded kind, which carry no welfare opinion and so route to nobody —
        so nothing is added after it either.

        Bounded by the same cap as the escalation drain, for the same reason: the
        thing on the other end of a notification is somebody else's HTTP endpoint,
        and a dead one holding a socket open must not be able to hold a
        `docker compose down` open with it. Past the cap the delivery is cancelled
        rather than waited on — unlike an escalation there is no §9 last resort to
        spill to here, and there should not be: `WebhookNotifier` already
        dead-letters its own undeliverable notes, and a second spool in this layer
        would duplicate every one it wrote.
        """
        task, self._notifications_task = self._notifications_task, None
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(
                    self._scheduler.notifications.drain(),
                    timeout=self._shutdown_drain_timeout_seconds,
                )
            except TimeoutError:
                logger.error(
                    "welfare notifications did not drain within %.1fs; "
                    "abandoning whatever is still in flight",
                    self._shutdown_drain_timeout_seconds,
                )
        await self._cancel(task)

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

    def last_frame_epoch(self, camera_id: str) -> float | None:
        """When this camera last delivered a frame, in Unix epoch seconds.

        **Not derivable from `CameraTelemetry.last_frame_at`**, which is the
        camera's own source timeline — `time.monotonic()` for an RTSP camera,
        seconds-from-start-of-file for a replayed one. A console that read that as
        an epoch showed "20712d ago" against every camera and reported "0 of 3
        delivering" while all three were, which is precisely the kind of wrong a
        surveillance console may not be.

        Answered here because this is the layer that legitimately holds a clock.
        `CameraRunner` deliberately has none (see `pipeline/runner.py` on why a
        second clock in the frame loop is a real bug), so the translation cannot
        happen there — and it cannot be a fixed offset either, because a file
        source starts at 0.0 while a live one starts at process uptime.

        The mechanism is observational: every time telemetry is read, a camera
        whose source timestamp has **advanced** since the last read is stamped with
        the wall clock now. A camera that has stopped delivering keeps its last
        stamp and therefore ages, which is exactly the signal a liveness indicator
        needs. The cost is that it is only as fresh as the last read — irrelevant
        while a console polls every few seconds, and honest either way: a camera
        nobody has asked about has no observation to report.
        """
        observed = self._frame_observations.get(camera_id)
        return None if observed is None else observed[1]

    def _observe_frames(self, telemetry: CameraTelemetry) -> None:
        """Record that this camera's source timeline moved, and when we noticed."""
        if telemetry.last_frame_at is None:
            return
        previous = self._frame_observations.get(telemetry.camera_id)
        if previous is None or telemetry.last_frame_at > previous[0]:
            self._frame_observations[telemetry.camera_id] = (
                telemetry.last_frame_at,
                self._wall_clock(),
            )

    @property
    def scheduler(self) -> VlmScheduler:
        """The one scheduler every camera submits to. Exposed so the composition root
        can build a runner for a camera added after startup and hand it the same one —
        a second scheduler would be a second VLM queue with its own admission gate,
        quietly doubling the GPU concurrency the whole design is built to bound."""
        return self._scheduler

    def add_camera(self, runner: CameraRunner) -> None:
        """Start watching a camera that was not in the file when the engine started.

        Additive, and that is why it is allowed where changing a running camera's `url`
        is not: there is no `CameraRunner` to tear down, no pre-roll ring to discard and
        no clip mid-recording, so the objection that makes `PATCH url` a restart simply
        does not arise.

        Started immediately rather than on the next `start()`. A camera that appears in
        `cameras.json` and in `/cameras` but delivers no frames until somebody restarts
        the engine is worse than one that was refused — it looks like it is working.
        """
        if runner.camera_id in self._cameras:
            raise ValueError(f"camera {runner.camera_id!r} is already running")
        self._cameras[runner.camera_id] = runner
        self._camera_tasks[runner.camera_id] = asyncio.create_task(runner.run())
        logger.info("started camera %s", runner.camera_id)

    async def disable_camera(self, camera_id: str) -> CameraTelemetry:
        """Stop watching a camera without forgetting it. Returns its final telemetry.

        The counters are **frozen, not zeroed**. A disabled camera reporting zero frames
        would read as "this saw nothing", when what actually happened is "this saw
        forty thousand frames and then somebody switched it off" — and on a wall where
        an operator is deciding what to look at, those two are not close.

        The frozen snapshot lives only as long as this process. After a restart a
        disabled camera reports zeros, which is then the true answer: nothing has been
        watched this run.

        Idempotent. Disabling an already-disabled camera returns the snapshot it
        already has rather than raising — two operators clicking the same switch is an
        ordinary race, not a mistake either of them made.
        """
        stopped = self._disabled.get(camera_id)
        if stopped is not None:
            return stopped
        # Before the cancel, while the runner still exists: afterwards there is nothing
        # left to ask.
        final = replace(self.telemetry(camera_id), enabled=False)
        await self.remove_camera(camera_id)
        self._disabled[camera_id] = final
        logger.info("disabled camera %s", camera_id)
        return final

    def is_running(self, camera_id: str) -> bool:
        """Whether a `CameraRunner` exists for this id right now.

        Distinct from "does this camera exist", which `telemetry` answers for both the
        running and the stopped, and from "is it healthy", which nothing here answers.
        """
        return camera_id in self._cameras

    def forget_disabled(self, camera_id: str) -> None:
        """Drop a disabled camera's frozen snapshot.

        Called when the camera is enabled again or deleted outright. Without it a
        re-enabled camera would appear twice — once live, once as the corpse of its
        previous run — and `cameras()` would report a site with more cameras than it
        has.
        """
        self._disabled.pop(camera_id, None)

    async def remove_camera(self, camera_id: str) -> None:
        """Stop watching a camera and forget it.

        Cancelling the task is the whole stop: `CameraRunner.run`'s own `finally`
        abandons any half-recorded clip and closes the source, and it does so for a
        cancellation exactly as it does for a shutdown. Awaited, not fired and forgotten
        — returning before the source is closed would leave a socket open against a
        camera the operator has been told is gone.
        """
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        task = self._camera_tasks.pop(camera_id, None)
        await self._cancel(task)
        del self._cameras[camera_id]
        logger.info("stopped and removed camera %s", camera_id)

    async def snapshot(self, camera_id: str) -> bytes | None:
        """The most recent frame from one camera, as a JPEG.

        `None` when nothing has arrived yet — a camera that has just been added, or one
        whose source is down. The API turns that into a 404 rather than an empty image,
        so a console can tell "not yet" from "here is a black frame".
        """
        return await self._get_runner(camera_id).snapshot_jpeg()

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        """Every configured camera, running or deliberately stopped.

        Disabled cameras are included rather than omitted, and that is the point of
        disabling rather than deleting: a camera an operator switched off must stay
        visible, or the only way to see it is to notice something missing. They are
        sorted in with the rest by id so the list does not reorder itself as cameras
        are toggled.
        """
        snapshots = tuple(runner.telemetry() for runner in self._cameras.values())
        for snapshot in snapshots:
            self._observe_frames(snapshot)
        return tuple(
            sorted((*snapshots, *self._disabled.values()), key=lambda camera: camera.camera_id)
        )

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        stopped = self._disabled.get(camera_id)
        if stopped is not None:
            # Not passed through `_observe_frames`: nothing is delivering, and feeding
            # a frozen timestamp to the liveness tracker would make a switched-off
            # camera look like one that had just gone stale.
            return stopped
        snapshot = self._get_runner(camera_id).telemetry()
        self._observe_frames(snapshot)
        return snapshot

    def ensure_capabilities_available(
        self, camera_id: str, capabilities: CameraCapabilities
    ) -> None:
        """Refuse a capability set this process has no model for — **before** anything
        is written.

        Deliberately a separate call rather than a check inside
        `update_camera_metadata`, and the ordering is the whole reason. The write path
        is: validate, persist to `cameras.json`, then apply to the running camera
        (`main.ComposedService.update_camera` documents why that order and not the
        reverse). A check living in the apply step would fire *after* the file had
        already been rewritten, leaving the record saying one thing and the running
        camera doing another — the exact half-applied state that path is built to make
        impossible, and it would make this error's "nothing was changed" a lie.

        So the caller asks this first, and a refusal costs nothing.
        """
        # Raise `UnknownCameraError` for an unknown camera rather than answering
        # "available", so an unknown id fails the same way here as everywhere else on
        # this path instead of surfacing two requests later.
        self._get_runner(camera_id)
        self.ensure_roles_available(camera_id, capabilities)

    def ensure_roles_available(self, camera_id: str, capabilities: CameraCapabilities) -> None:
        """The model check on its own, for a camera that does not exist yet.

        `ensure_capabilities_available` asks this *after* checking the camera is running,
        which is right for an edit and wrong for a create: a camera being added has no
        runner, and the whole point of checking before the write is that it happens
        before anything exists. Same refusal, same message, one fewer precondition.
        """
        missing = required_roles(capabilities) - self._available_roles
        if missing:
            raise CapabilityUnavailableError(camera_id, frozenset(missing))

    def update_camera_metadata(
        self,
        camera_id: str,
        *,
        label: str,
        zone: Zone | None,
        capabilities: CameraCapabilities,
        notify_on: frozenset[ConcernKind],
        notify_min_confidence: Confidence,
        clip_preroll_seconds: float | None,
        clip_postroll_seconds: float | None,
        summary_interval_seconds: float | None,
    ) -> None:
        """Apply an already-persisted change of a camera's editable record to the
        running camera.

        Deliberately knows nothing about `cameras.json`: this service owns running
        cameras, not the record of configured ones, and the caller that owns both
        (`main.ComposedService.update_camera`) is the one that orders the write
        before this call. Keeping the file out of here is what stops a future
        change from making an in-memory edit that never reaches disk.

        The whole editable record rather than a diff, for the reason
        `CameraRunner.apply_metadata` gives: the caller has just read the persisted
        record back, and applying all of it cannot leave some fields applied and
        others not.

        Synchronous and total: `apply_metadata` resolves every value that can be
        rejected before it writes a single field, so a rejected edit leaves the
        camera exactly as it was and there is no partial-application case for a
        caller to unwind. Both configuration edges validate the same bounds, so
        nothing reaching here can be rejected in the first place — but the totality
        is a property of `apply_metadata`, not of that argument, and a caller may
        rely on it as such.
        """
        self._get_runner(camera_id).apply_metadata(
            label=label,
            zone=zone,
            capabilities=capabilities,
            # Resolved here rather than in the runner, because "does a detector
            # exist in this process" is not something one camera can know. Handed
            # `None` when this camera no longer needs one, so switching a camera off
            # stops it paying for a forward pass per frame rather than merely
            # discarding the results.
            detector=(
                self._detector if ModelRole.DETECTOR in required_roles(capabilities) else None
            ),
            notify_on=notify_on,
            notify_min_confidence=notify_min_confidence,
            clip_preroll_seconds=clip_preroll_seconds,
            clip_postroll_seconds=clip_postroll_seconds,
            summary_interval_seconds=summary_interval_seconds,
        )

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

    def subscribe_events(self) -> EventSubscription:
        """A live handle on every camera's recent-event ring, for `GET /events/stream`.

        Site-wide rather than per camera, and so takes no camera id and cannot 404: a
        console watching a building wants one connection, not one per camera, and the
        entries carry `camera_id` for the caller to split on.
        """
        return self._scheduler.subscribe_events()

    def close_event_streams(self) -> int:
        """End every live stream; returns how many there were.

        Deliberately **not** part of `stop()`. That method's phase ordering exists to
        get the last escalations published, and those publishes assemble events that
        the streams should still carry — closing them from inside it would cut the
        console off from precisely the shutdown-time events spec §9 goes to such
        lengths to preserve. The API lifespan calls this after `stop()` returns, and
        in a `finally`, so a cut-short shutdown still releases its readers.
        """
        return self._scheduler.close_event_streams()

    async def list_people(self) -> tuple[tuple[AuthorizedPerson, int], ...]:
        """Every enrolled person with their reference-face count.

        The count rather than the faces: §12 asks that biometric information is not
        exposed unnecessarily, and a console listing people needs to know that somebody
        has only one reference (usually why they are not being recognised) without
        needing the vector.
        """
        store = self._face_store
        if store is None:
            return ()
        people = await store.list_people()
        # A list comprehension rather than a generator into `tuple()`: the count is an
        # await, and an async generator is not an `Iterable` the constructor accepts.
        counted: list[tuple[AuthorizedPerson, int]] = []
        for person in people:
            counted.append((person, await store.reference_count(person.person_id)))
        return tuple(counted)

    async def upsert_person(self, person: AuthorizedPerson) -> AuthorizedPerson:
        store = self._require_face_store()
        await store.add_person(person)
        return person

    async def delete_person(self, person_id: UUID) -> bool:
        store = self._require_face_store()
        return await store.delete_person(person_id)

    async def enroll_face(
        self, person_id: UUID, image: object, *, original: bytes | None = None
    ) -> int:
        """Detect a face in `image`, check its quality, embed it, and store it (§9).

        Returns how many reference faces the person now has. Raises `ValueError` when
        the image carries no usable face — which the API turns into a 422 naming the
        reason, because "we accepted your photograph and silently enrolled nothing" is
        the failure that makes somebody unrecognisable and nobody able to say why.
        """
        store = self._require_face_store()
        if self._face is None:
            raise FaceCapabilityUnavailableError()
        if await store.get_person(person_id) is None:
            raise UnknownPersonError(person_id)

        frame = FrameData(
            camera_id="enrolment",
            frame_index=0,
            timestamp=0.0,
            width=0,
            height=0,
            pixels=image,
        )
        faces = await self._face.detect(frame)
        if not faces:
            raise ValueError("no face was found in that image")
        # The largest face, on the assumption that an enrolment photograph is *of*
        # somebody rather than a crowd scene. Choosing by detector confidence instead
        # would sometimes pick a sharp bystander over the slightly soft subject.
        best = max(faces, key=lambda face: face.box.area)
        if not isinstance(best.aligned, FaceEmbedding):
            raise ValueError("the face pipeline produced no embedding for that image")
        # The photograph as uploaded, not the crop the detector found. An operator
        # enrolling somebody wants to see the picture they chose — is it the right
        # person, is it a good photo — and a 112px face crop answers neither well. The
        # crop is still what gets embedded; this is only what gets shown.
        await store.add_embedding(
            person_id,
            best.aligned,
            image=None if original is None else reference_image(original),
        )
        return await store.reference_count(person_id)

    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        """Every reference on a person's record. Raises if the person is unknown."""
        store = self._require_face_store()
        if await store.get_person(person_id) is None:
            raise UnknownPersonError(person_id)
        return await store.list_faces(person_id)

    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        """One stored face crop, or `None` when there is not one.

        Does *not* raise on an unknown person, unlike `list_faces`. The caller is
        rendering a picture or a placeholder either way, and a 404 that distinguished
        "no such person" from "no such photograph" would let anybody with the endpoint
        enumerate who is enrolled.
        """
        return await self._require_face_store().read_face_image(person_id, face_id)

    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        return await self._require_face_store().delete_face(person_id, face_id)

    def _require_face_store(self) -> FaceStore:
        if self._face_store is None:
            raise FaceCapabilityUnavailableError()
        return self._face_store

    def alerts(self) -> tuple[Alert, ...]:
        """Every alert this process is holding, worst first.

        Reached through the scheduler rather than held here, so what the API reads is
        necessarily the same register the events are going into — the arrangement
        `notifications` already uses and for the same reason.
        """
        register = self._scheduler.alerts
        return () if register is None else register.snapshot()

    async def flush_alerts(self) -> None:
        """Write the register now, if a store is configured and anything changed.

        Awaited by the acknowledge and resolve handlers, so a 200 from those endpoints
        means an operator's decision reached the disk rather than only the heap. The
        machine-driven path does not call this — see `AlertPersistence` on why the two
        are paced differently.
        """
        if self._alert_persistence is not None:
            await self._alert_persistence.flush()

    def clear_alerts(self) -> int:
        """Empty the alert register. Returns how many were dropped."""
        register = self._scheduler.alerts
        return 0 if register is None else register.clear()

    def acknowledge_alert(self, alert_id: UUID, *, by: str, at: float) -> Alert:
        """Record that a person has seen an alert.

        `UnknownAlertError` for an id the register does not hold, and `ValueError` for
        one already resolved, both propagate: the API maps them to 404 and 409. An
        engine with no register raises `UnknownAlertError` too, which is the truthful
        answer — that alert really is not here.
        """
        register = self._scheduler.alerts
        if register is None:
            raise UnknownAlertError(alert_id)
        return register.acknowledge(alert_id, by=by, at=at)

    def resolve_alert(self, alert_id: UUID) -> Alert:
        register = self._scheduler.alerts
        if register is None:
            raise UnknownAlertError(alert_id)
        return register.resolve(alert_id)

    async def storage_usage(self) -> StorageUsage:
        """What the clip bucket holds. Never raises — see `ClipIndex.usage`."""
        if self._clip_index is None:
            return StorageUsage(
                bucket="",
                clips=0,
                objects=0,
                bytes_used=0,
                retention_days=0,
                per_camera=(),
                reachable=False,
            )
        return await self._clip_index.usage()

    async def list_clips(self, camera_id: str, *, limit: int) -> tuple[ClipRecord, ...]:
        """One camera's clips, newest first. `UnknownCameraError` for an id this engine
        does not have — a footage browser asking about a camera that was deleted should
        be told that, not handed an empty list it would render as "nothing recorded"."""
        self.telemetry(camera_id)
        if self._clip_index is None:
            return ()
        return await self._clip_index.list_clips(camera_id, limit=limit)

    async def read_camera_clip(self, camera_id: str, event_id: UUID, *, short: bool) -> bytes | None:
        """One clip of one camera, by the event it was recorded for.

        The object path is built by the adapter from two validated components — a
        camera this engine has, and a `UUID` — never from a caller's string. The reader
        then re-checks the bucket, as it does for the alert route.

        Falls back to the full recording when no short copy exists, for
        `read_alert_clip`'s reason.
        """
        self.telemetry(camera_id)
        if self._clip_index is None or self._clip_reader is None:
            return None
        clip = await self._clip_reader.read(self._clip_index.clip_uri(camera_id, event_id, short=short))
        if clip is None and short:
            return await self._clip_reader.read(
                self._clip_index.clip_uri(camera_id, event_id, short=False)
            )
        return clip

    async def read_alert_clip(self, alert_id: UUID, *, short: bool) -> bytes | None:
        """The recording behind one alert, or `None` if there is nothing to play.

        The alert id is the whole of the authorisation, and that is deliberate: the URI
        is looked up here rather than accepted from the caller, so no request can name
        an object this engine did not itself attach to an alert.

        `short` picks the notification-length copy, falling back to the full clip when
        the writer never made one — a fallback rather than a 404 because the operator
        asked to see what happened, and a longer answer to that question is still an
        answer. `UnknownAlertError` propagates for an id the register does not hold.
        """
        alert = self._require_register(alert_id).get(alert_id)
        uri = (alert.notify_clip_uri or alert.clip_uri) if short else alert.clip_uri
        if uri is None or self._clip_reader is None:
            return None
        return await self._clip_reader.read(uri)

    def _require_register(self, alert_id: UUID) -> AlertRegister:
        register = self._scheduler.alerts
        if register is None:
            raise UnknownAlertError(alert_id)
        return register

    def health(self) -> dict[str, HealthReport]:
        return self._registry.health()

    async def describe_now(self, camera_id: str) -> UUID:
        return await self._get_runner(camera_id).describe_now()

    def _get_runner(self, camera_id: str) -> CameraRunner:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise UnknownCameraError(camera_id) from None
