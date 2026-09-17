"""The VLM escalation queue (spec §5.4) and the S14 event-assembly worker (spec §6, §9).

Exactly one worker processes escalations: there is one GPU and one set of resident
VLM weights, so concurrent `describe()` calls would contend for the same VRAM.
Drop-on-full is safe because the token bucket and the cooldown already bound the
camera-side arrival rate (spec §4.1) — a full queue should be rare in practice, and
when it happens the drop is counted, never raised.

Event assembly happens here and only here (S14): this is the one place in the system
that constructs an `Event`. Every error path below still produces one — spec §9's
governing rule is that an anomaly event is never lost to an infrastructure failure.
That rule is enforced in four places:

  * a failed or timed-out `describe()` -- or one that returns an unusable threat
    value -- yields a metadata-derived description flagged
    `description_unavailable=True`;
  * a failed `ClipHandle.finish()` yields an event with `clip_uri=None`;
  * a `publish()` that *raises* hands the event to the `FailedEventSink` instead of
    dropping it, and counts it in `publish_failures`;
  * a failure anywhere still releases the admission slot in a `finally` and leaves the
    worker alive for the next escalation. A leaked slot at concurrency 1 wedges the
    GPU for the lifetime of the process, and a worker that dies on one poisoned
    request silently stops describing everything after it.

The third of those is the boundary worth being precise about. `RabbitMQPublisher`
already spools when the *broker* is unreachable, and that is the common case; it
returns normally, so nothing here fires and the event is not handled twice. But three
reachable paths raise out of `publish()` instead — a schema-invalid event
(`encode_event` validates before the transport guard), a spool directory that cannot
be written, and any unexpected transport error — and each of those used to end with
the event existing nowhere. The sink is here rather than inside the publisher because
spec §9 is a property of the system, not of RabbitMQ: it has to hold for whatever
`EventPublisher` is wired in.

VLM residency
-------------
Every describe is preceded by `ResidentSet.ensure((vlm_model_key,), now)`. That single
call does two jobs, and the system is broken without either: it *reloads* a VLM the
600s idle sweep has evicted, and it stamps `last_used_at`, which is what makes the
sweep's window mean "idle" rather than "600s since boot". It goes through
`ResidentSet` rather than calling `ModelRuntime.initialize()` directly so that
`plan_residency`'s admission arithmetic and eviction bookkeeping stay authoritative.

VLM out-of-memory
-----------------
Spec §9 gives this failure its own row — *evict LRU, retry once, mark unhealthy,
pipeline continues detection-only* — because it is the one the camera can survive if
it is handled and cannot if it is not. `_describe_surviving_one_oom` implements it,
and only it: every other describe failure is one bad escalation and takes the §9
fallback below. See that method for why the model evicted is the VLM and not the
literal least-recently-used one.

Clock
-----
`clock` here is *real elapsed time* — `time.monotonic` in production. It has to be:
`AdmissionGate` spaces GPU admissions with `asyncio.sleep`, which runs on the wall
clock, so a scheduler clock that did not advance with it would compound the gate's
deficit without bound. This is deliberately **not** the same clock as the camera
pipeline's: `CameraRunner` has no clock of its own and runs entirely on the source's
timeline (see `pipeline/runner.py`). The two agree only for a live RTSP source and
must not be conflated.

Event timestamps
----------------
Which is exactly why an event carries two of them. `Event.source_timestamp` is the
scene's own timeline, unmodified — the thing a clip's pts and the camera telemetry
share. `Event.occurred_at` is Unix epoch seconds, rebased onto the wall clock through
`_to_epoch` so a consumer can sort, display and compare events across restarts and
across cameras. The rebasing is anchored **per camera**, lazily, from that camera's
own first-seen `source_timestamp` — not from a single process-wide anchor — because
`source_timestamp` does not share one timeline across sources: `RtspSource` stamps
`time.monotonic()`, `FileSource` stamps container pts starting at 0.0 per file. See
`_to_epoch` for the arithmetic and for the precision this trades away in exchange.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import (
    EscalationReason,
    Event,
    SceneState,
    Severity,
    ThreatScore,
)
from sentinel_ai.domain.policy.alerting import severity_rank
from sentinel_ai.domain.policy.notification import concerns_to_notify
from sentinel_ai.domain.policy.priority import priority_of, rank
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareAssessment
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.alerts import AlertRegister
from sentinel_ai.orchestrator.escalation_queue import PriorityWorkQueue
from sentinel_ai.orchestrator.event_history import (
    RECENT_EVENTS_PER_CAMERA,
    CameraEventHistory,
    EventSubscription,
    RecentEventLog,
)
from sentinel_ai.orchestrator.notifications import NotificationDispatcher
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher, FailedEventSink
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.notifier import WelfareNote
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest

logger = logging.getLogger(__name__)

__all__ = ["EscalationRequest", "VlmScheduler"]


@dataclass(frozen=True, slots=True)
class EscalationRequest:
    """What the runner hands the scheduler.

    Carries everything the VLM and the event need, so the worker never reaches back
    into the pipeline for anything — the camera is free to move on to the next frame
    the instant this is queued.
    """

    camera_id: str
    event_id: UUID
    reason: EscalationReason
    detail: str
    scene: SceneState
    keyframe: FrameData
    profile: CameraProfile
    camera_label: str
    history: tuple[str, ...]
    clip: ClipHandle | None

    zone: Zone | None
    """Where this camera watches, for the note a human reads (T10).

    Carried here rather than looked up: this object is the runner's whole
    handover, and a scheduler that reached back for a camera's zone would have to
    hold a reference to the pipeline it exists to be decoupled from.
    """

    notify_on: frozenset[ConcernKind]
    notify_min_confidence: Confidence
    """This camera's welfare-notification policy, read off the runner at the moment
    the escalation was decided (T10).

    Snapshotted with the rest of the request rather than consulted at notify time,
    for `_escalate`'s stated reason: an escalation half on the old policy and half
    on the new one has no defensible meaning, and `apply_metadata` can land at any
    await between here and the published event. `CameraRunner` still branches on
    neither — the frame loop carries them, and the rule that reads them
    (`domain/policy/notification.py`) runs here, downstream of the publish.

    No defaults, like every other field on this record: a request that silently
    defaulted to "notify about everything" would notify about a camera an operator
    had muted, and one that defaulted to the empty set would mute a camera nobody
    muted. Both are wrong, and neither is visible without a test that looks for it.
    """

    subject_track_ids: tuple[int, ...] = ()
    """Whose behaviour this escalation is about, when a detector knew.

    Carried from `BehaviourCandidate.track_ids` and empty for the six automatic
    triggers, which describe a scene rather than a subject. Ends up on
    `Event.subject_track_ids`, where it is what `AlertKey` deduplicates on — see that
    field for why keying on the scene's tracks instead did not work.
    """

    describe: bool = True
    """Whether this camera asked for a vision-language description (spec §13).

    False is not a failure and not an error: it is a camera configured for
    `anomaly_detection` without `scene_description`, which is the cheap tier that
    makes many cameras affordable on one GPU. The escalation still becomes a
    published `Event` — the fact that a track count spiked is information on its own
    — it simply carries a metadata-derived description and never reaches the model.

    Defaulted to `True`, unlike every other field here, because this one has a
    genuinely safe default and the alternative is worse: a construction site that
    forgot it would silently stop describing a camera an operator never reconfigured.
    The failure modes are asymmetric in a way the welfare fields' are not — there,
    either default is capable of being wrong in a direction nobody asked for; here,
    "describe unless told otherwise" is exactly the pre-capabilities behaviour every
    existing caller means.
    """


_UNAVAILABLE_THREAT_VALUE = 0.5
"""A conservative mid-range placeholder: severity truly is unknown without a
description, and 0.5 neither over- nor under-states it for downstream triage."""

_UNAVAILABLE_ACTION = "Review the clip when available."

_SKIPPED_ACTION = "Review the clip; scene description is not enabled on this camera."

DESCRIPTION_SKIPPED_METADATA_KEY = "description_skipped"
"""`Event.metadata` key naming *why* a description is absent, when the reason is
configuration rather than failure.

`description_unavailable` is a bool, and it has always meant one thing: the model was
asked and did not answer. A camera with `scene_description` switched off produces an
event with no description either, but for the opposite reason — nothing went wrong,
and rendering it as a model failure would teach an operator to ignore a flag that
otherwise means a real fault. Rather than widen the bool into a tri-state (a required
schema field, so widening it is a breaking change for every consumer already reading
it), the distinction rides in `metadata`, which is free-form `dict[str, str]` in the
published schema and already exists for exactly this: operational detail about one
event. Absent means the ordinary reading of `description_unavailable` applies.
"""


_OOM_TYPE_NAMES = frozenset({"OutOfMemoryError", "CudaOutOfMemoryError", "OutOfMemory"})
"""Class names that mean "the accelerator ran out of memory".

Matched by *name* across the MRO rather than by importing `torch.cuda.OutOfMemoryError`:
this module is in the orchestrator layer and must keep running on a CI box with no
`gpu` extra installed, where that import does not exist. Backed up by a substring
check on the message, because several runtimes raise a plain `RuntimeError`
("CUDA out of memory. Tried to allocate ...") rather than a distinct type.
"""


def _is_out_of_memory(error: BaseException) -> bool:
    """Spec §9's VLM-OOM row applies to this failure and not to any other one.

    Deliberately generous: treating an unrelated failure as an OOM costs one extra
    reload, while missing a real OOM costs the recovery the spec requires.
    """
    if isinstance(error, MemoryError):
        return True
    if any(cls.__name__ in _OOM_TYPE_NAMES for cls in type(error).__mro__):
        return True
    return "out of memory" in str(error).lower()


def _labels_and_tracks(scene: SceneState) -> tuple[tuple[str, ...], tuple[int, ...]]:
    labels = tuple(sorted({track.label for track in scene.tracks}))
    track_ids = tuple(track.track_id for track in scene.tracks)
    return labels, track_ids


def _basis_for(request: EscalationRequest, welfare: WelfareAssessment) -> WelfareAssessment:
    """Stamp the assessment with what it actually rests on (ADR 10, spec §7).

    A welfare opinion reached from a single still frame and one reached from a frame
    *plus* a multi-second geometry state machine that watched someone go down and stay
    down are different amounts of evidence, and ADR 10's closing paragraph is explicit
    about the consequence: a second source "gets its own `basis` value rather than
    silently widening what this one means". This is the one place that promise is kept.

    The condition is the escalation reason, because that is the only thing here that
    knows a temporal machine was involved — `FALL_SUSPECTED` is raised by
    `domain/behaviour/fall.py` and by nothing else (`pipeline/runner.py`).

    An assessment with no concerns is left alone. `single_frame_vlm` is the default and
    an empty assessment is omitted from the published event entirely, so stamping one
    would be recording corroboration of a concern that does not exist — and would put a
    `temporal_pose_vlm` marker on payloads where the model declined to agree, which is
    the opposite of what the value is for.
    """
    if request.reason is not EscalationReason.FALL_SUSPECTED or not welfare.concerns:
        return welfare
    return replace(welfare, basis="temporal_pose_vlm")


def _metadata_description(request: EscalationRequest) -> str:
    """Fallback description built from cheap signals alone — no VLM call required."""
    labels, _ = _labels_and_tracks(request.scene)
    what = ", ".join(labels) if labels else "motion"
    return f"{request.reason.value}: {what} ({request.detail})"


class VlmScheduler:
    def __init__(
        self,
        vlm: VisionLanguageModel | None,
        publisher: EventPublisher,
        admission: AdmissionGate,
        *,
        resident_set: ResidentSet,
        vlm_model_key: str | None,
        dead_letter: FailedEventSink,
        notifications: NotificationDispatcher,
        alerts: AlertRegister | None = None,
        notify_clip_min_severity: Severity = Severity.HIGH,
        maxsize: int,
        timeout_seconds: float,
        clock: Callable[[], float],
        wall_clock: Callable[[], float] = time.time,
        recent_events_per_camera: int = RECENT_EVENTS_PER_CAMERA,
    ) -> None:
        # `None` when no configured camera enabled `scene_description`, so the
        # composition root never built a VLM at all (spec §13). Every escalation then
        # takes the skipped-description path below. Optional here rather than a null
        # object because a null `VisionLanguageModel` would still be *registered*,
        # still appear in `/health`, and still have to answer `describe()` with
        # something — three lies to avoid one `if`.
        self._vlm = vlm
        if (vlm is None) != (vlm_model_key is None):
            # The two travel together or not at all: a key with no runtime makes
            # `_ensure_vlm_resident` ask the registry for a model nobody registered,
            # and a runtime with no key makes it impossible to keep resident. Either
            # way the failure surfaces much later, as a describe that mysteriously
            # degrades, so it is caught at construction instead.
            raise ValueError(
                "vlm and vlm_model_key must both be set or both be None; got "
                f"vlm={'set' if vlm is not None else 'None'}, vlm_model_key={vlm_model_key!r}"
            )
        self._publisher = publisher
        self._admission = admission
        self._resident_set = resident_set
        self._vlm_model_key = vlm_model_key
        # Required, not optional, for the same reason `resident_set` is: an
        # unwired last resort is indistinguishable from no last resort, and the
        # failure it guards against is silent by construction.
        self._dead_letter = dead_letter
        # Required for the same reason `dead_letter` is. A deployment that
        # configures nothing still gets `LoggingNotifier` (see `main.build_notifier`),
        # so "no notifier" is never a real state — and an optional parameter here
        # would make an unwired dispatch indistinguishable from a site where nothing
        # was worth notifying about, which is the one failure a welfare system may
        # not have.
        self._notifications = notifications
        # Optional, unlike `dead_letter` and `notifications`, because an alert register
        # is a *view* rather than a guarantee: an engine without one still publishes
        # every event, still spools, still notifies. What it loses is the operator's
        # triage list, and a composition that wants only the event stream (the
        # benchmark harness, most tests) should not have to construct one.
        self._alerts = alerts
        # The band at or above which a notification carries the short clip rather than
        # the evidence one. See `_clip_for_note`.
        self._notify_clip_min_severity = notify_clip_min_severity
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        # Injected rather than taken from `time` directly so no test needs a real
        # clock to pin the arithmetic — see `_to_epoch` for how it is used.
        self._wall_clock = wall_clock
        # One anchor per camera, established lazily from that camera's own first
        # event rather than once for the whole process — see `_to_epoch`.
        self._camera_anchors: dict[str, tuple[float, float]] = {}
        # A volatile console cache, not the event store — see
        # `orchestrator/event_history.py`. It lives here rather than beside the
        # publisher because it must capture events the publisher never sees: a
        # dead-lettered one, and one a cancelled shutdown abandoned.
        self._recent = RecentEventLog(recent_events_per_camera)
        # Priority-ordered rather than FIFO, and evicting the least urgent rather
        # than the newest — see `orchestrator/escalation_queue.py` for both
        # arguments. A suspected fall queued behind four periodic summaries is
        # thirteen seconds at the measured describe latency.
        self._queue: PriorityWorkQueue[EscalationRequest] = PriorityWorkQueue(maxsize=maxsize)
        self._dropped = 0
        self._publish_failures = 0
        self._abandoned = 0
        self._in_flight: EscalationRequest | None = None
        """The request the worker is currently describing, kept so that a shutdown
        which cancels the worker mid-describe can still account for it. Cleared on
        every normal outcome; deliberately *not* cleared on cancellation — see
        `run()` and `abandon_pending()`."""

    def submit(self, request: EscalationRequest) -> bool:
        """Non-blocking. Returns False and counts a drop when the queue is full.

        Deliberately not a coroutine: the camera pipeline must never await the VLM,
        and a `def` makes that unrepresentable rather than merely discouraged.
        """
        accepted, evicted = self._queue.submit(request, rank(priority_of(request.reason)))
        if evicted is not None:
            self._dropped += 1
            logger.warning(
                "vlm queue full: dropped %s escalation for camera %s to make room for %s",
                evicted.reason.value,
                evicted.camera_id,
                request.reason.value,
            )
        return accepted

    async def run(self) -> None:
        """The single worker loop. Cancel to stop."""
        while True:
            request = await self._queue.get()
            self._in_flight = request
            try:
                await self._process(request)
            except asyncio.CancelledError:
                # Shutdown cut this describe short. `_in_flight` stays set on purpose:
                # the request is no longer on the queue and the worker will never
                # finish it, so `abandon_pending()` is the only thing left that can
                # keep spec §9 true for it.
                self._queue.task_done()
                raise
            except Exception:
                # `asyncio.CancelledError` is a `BaseException`, so cancelling the
                # worker still stops it cleanly; anything else is one bad escalation,
                # and killing the process's only VLM worker over it would silently
                # stop every camera from ever being described again.
                logger.exception(
                    "escalation failed for camera %s event %s",
                    request.camera_id,
                    request.event_id,
                )
            self._in_flight = None
            self._queue.task_done()

    async def abandon_pending(self, reason: BaseException) -> int:
        """Dead-letter every escalation this worker will now never process (spec §9).

        Called by `EngineService.stop()` when the bounded drain expires. Without it
        that cap reproduced C3's exact signature through a different door: the
        escalations still queued (and the one the worker was midway through) were
        counted in `escalations`, never published, never counted in
        `escalations_dropped`, and invisible in every telemetry read — the reviewer's
        probe measured `escalations 1 / published 0 / dropped 0 / dead-lettered 0 /
        publish_failures 0 / warnings []`. Reachable with the shipped numbers: a cap of
        10s against `vlm_timeout_seconds` 30s and a queue of four real Qwen describes
        spaced by `vlm_global_min_interval_seconds`, and the team's own live runs used
        `--timeout-graceful-shutdown 2`, which cuts `stop()` off well before the cap.

        Each abandoned request becomes the same §9 fallback event a failed describe
        produces — `description_unavailable=True` — and goes to the `FailedEventSink`
        rather than the publisher: past the cap there is no time budget left to spend
        on a broker that may itself be the reason we are late, and the sink's whole
        job is the event that has nowhere else to go. Any clip handle still riding on
        the request is aborted, or it would leak the same PyAV container and temp file
        a failed pre-roll seed used to.

        Must be called only once the worker is no longer running, or it races the
        worker for the queue.
        """
        pending: list[EscalationRequest] = []
        if self._in_flight is not None:
            pending.append(self._in_flight)
            self._in_flight = None
        for drained in self._queue.drain_pending():
            pending.append(drained)
            self._queue.task_done()

        for request in pending:
            self._abandoned += 1
            logger.error(
                "shutdown abandoned escalation for camera %s event %s before it could be "
                "published; dead-lettering it (%s)",
                request.camera_id,
                request.event_id,
                reason,
            )
            if request.clip is not None:
                with contextlib.suppress(Exception):
                    await request.clip.abort()
            await self._dead_letter.store(self._unavailable_event(request), reason)
        return len(pending)

    async def drain(self) -> None:
        """Await completion of everything currently queued.

        Two callers: tests, and `EngineService.stop()`, which drains between
        cancelling the cameras and cancelling this worker so that the escalations a
        runner submits from its shutdown path are actually published (spec §9). It
        only ever returns once the queue is empty, so the shutdown caller bounds it
        with a timeout rather than trusting the VLM to finish.
        """
        await self._queue.join()

    @property
    def notifications(self) -> NotificationDispatcher:
        """The dispatcher this scheduler hands routed notes to.

        Exposed so `EngineService` can own its worker task and drain it in the same
        shutdown phase ordering it already applies to the escalation queue, without
        a second constructor parameter that could be wired to a *different*
        dispatcher than the one actually receiving notes.
        """
        return self._notifications

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def publish_failures(self) -> int:
        """Events the publisher refused and the dead-letter sink took instead.

        A counter distinct from `dropped`: a drop is the queue rejecting an
        escalation under load, which the governors make an expected event, while
        this is an assembled event that could not be delivered. All three of the
        paths this covers previously left `dropped == 0`, so the loss showed up in
        no counter at all.
        """
        return self._publish_failures

    @property
    def abandoned(self) -> int:
        """Escalations `abandon_pending()` dead-lettered because shutdown ran out of
        drain budget. A third counter rather than a reuse of either of the others: this
        is neither load-shedding (`dropped`) nor a publisher refusing an assembled event
        (`publish_failures`), and conflating it would hide the one number an operator
        needs to see after a `docker compose down` cut a queue short."""
        return self._abandoned

    def event_history(self, camera_id: str) -> CameraEventHistory:
        """This camera's bounded, volatile console history — see
        `orchestrator/event_history.py`, and note that it is not the event store.

        An unconfigured camera id is not this object's concern and yields an empty
        snapshot; `EngineService` is what turns an unknown camera into an
        `UnknownCameraError`, so an unknown camera 404s and a quiet one returns an
        empty list.
        """
        return self._recent.history(camera_id)

    def subscribe_events(self) -> EventSubscription:
        """A live client's handle on the same ring `event_history` snapshots.

        Registered here and now, before the caller takes its backlog: see
        `EventSubscription` for why that order is what makes the handover gapless.
        """
        return self._recent.subscribe()

    def close_event_streams(self) -> int:
        """End every live stream; returns how many there were. Called at shutdown so a
        subscriber parked on an engine that has stopped producing is released rather
        than left waiting."""
        return self._recent.close_all()

    async def _process(self, request: EscalationRequest) -> None:
        await self._admission.acquire(self._clock())
        try:
            await self._ensure_vlm_resident()
            event = await self._describe(request)
            event = await self._attach_clip(event, request)
            await self._publish(event)
            # After the publish, before the notify, and deliberately between them.
            # After, because §9's rule is that an event is never lost to an
            # infrastructure failure and the register is not that record — folding into
            # an alert before the event is safely away would put triage ahead of
            # durability. Before the notify, because a notification carries a human-
            # facing summary and the alert is what an operator will open from it.
            #
            # `absorb` is pure bookkeeping over a pure rule and cannot block, so it
            # cannot hold the admission slot the `finally` below releases.
            self._record_alert(event, request)
            # Spec's order: describe -> clip finalised -> publish -> notify. Last,
            # and only after `_attach_clip`, so the note carries the clip uri the
            # responder will want; and non-blocking, so the endpoint on the other
            # end of it cannot hold the admission slot this `finally` releases.
            self._notify(event, request)
        finally:
            self._admission.release(self._clock())

    async def _publish(self, event: Event) -> None:
        """Publish, and keep the event if the publisher will not take it (spec §9).

        A raise from `publish()` means the publisher handled nothing — a spooling
        publisher returns normally, so the disk spool and this sink never both act on
        the same event. The exception is swallowed after the event is safe: re-raising
        would abandon the admission slot to the `finally` above and log the same
        failure twice, and there is nothing further the worker could do about it.
        """
        try:
            await self._publisher.publish(event)
        except Exception as error:
            self._publish_failures += 1
            await self._dead_letter.store(event, error)

    def _record_alert(self, event: Event, request: EscalationRequest) -> None:
        """Fold this event into the operator's alert list, if there is one.

        Never raises. The register is a view over events that have already been
        published, so a failure here costs a row in a console — losing the escalation
        over it would trade the durable thing for the convenient one. The same
        reasoning `Notifier` documents, one step earlier in the sequence.
        """
        if self._alerts is None:
            return
        try:
            self._alerts.absorb(
                event,
                camera_label=request.camera_label,
                zone=request.zone,
                # `occurred_at` rather than the scheduler's own clock: the merge window
                # has to be measured on the same timeline the alert's `first_seen` and
                # `last_seen` are reported on, or a replayed camera would open a new
                # episode for every event.
                now=event.occurred_at,
                # The short clip the notification would carry, offered to the console
                # too. Read off the handle rather than the event because the event does
                # not carry it: `_attach_clip` puts only the full clip on the event, and
                # this is the same recording trimmed. `None` whenever no clip was
                # recorded, the writer makes no short copy, or the trim failed.
                notify_clip_uri=(request.clip.notify_uri if request.clip is not None else None),
            )
        except Exception:
            logger.exception(
                "could not record an alert for camera %s event %s; the event was still published",
                event.camera_id,
                event.event_id,
            )

    @property
    def alerts(self) -> AlertRegister | None:
        """The register this scheduler folds events into, for the API to read.

        Reached through the scheduler rather than held separately by `EngineService`
        for `notifications`' reason: it guarantees the thing being read is the same
        object the events are going into.
        """
        return self._alerts

    def _notify(self, event: Event, request: EscalationRequest) -> None:
        """Route this event's welfare concerns to a human, if any of them qualify.

        Synchronous on purpose. Everything expensive happens on the dispatcher's own
        worker (`orchestrator/notifications.py`); what runs here is the rule and a
        `put_nowait`, so the escalation worker never awaits a network it does not
        control. `submit()` cannot raise and its return value is deliberately
        ignored — a full queue is already counted and logged there, and there is
        nothing this frame could do about it that would not cost the pipeline more
        than the note is worth.

        Deliberately runs even when `_publish` fell back to the dead-letter sink. A
        broker outage is the case *most* worth reaching a person over: the event is
        safe on disk, nobody is looking at a console that has stopped receiving, and
        muting the alert because the wrong pipe broke would be exactly backwards.
        """
        concerns = concerns_to_notify(
            event.welfare,
            severity=event.threat.severity,
            notify_on=request.notify_on,
            min_confidence=request.notify_min_confidence,
        )
        if not concerns:
            return
        self._notifications.submit(
            WelfareNote(
                event_id=event.event_id,
                camera_id=event.camera_id,
                label=request.camera_label,
                zone=None if request.zone is None else request.zone.value,
                occurred_at=event.occurred_at,
                # The band this note was routed at, as a plain string an adapter
                # serialises verbatim rather than reasons about — see `WelfareNote`.
                severity=event.threat.severity.value,
                description=event.description,
                # Only the concerns that routed, never the whole assessment: a note
                # naming a kind the operator muted would leak exactly what
                # `notify_on` exists to suppress.
                concerns=concerns,
                # The short clip when this note is urgent enough to be read on a
                # phone, the full one otherwise. `notify_uri` is `None` unless the
                # writer made one, so the fallback is the ordinary case rather than an
                # error path — see `ClipHandle.notify_uri`.
                clip_uri=self._clip_for_note(request, event),
            )
        )

    def _clip_for_note(self, request: EscalationRequest, event: Event) -> str | None:
        """Which clip a notification carries.

        Only above the threshold, and only if a short one exists. Below it the full clip
        is the better link: nobody is running anywhere, and the extra seconds of context
        are worth more than the shorter download.
        """
        if event.clip_uri is None or request.clip is None:
            return event.clip_uri
        if severity_rank(event.threat.severity) < severity_rank(self._notify_clip_min_severity):
            return event.clip_uri
        return request.clip.notify_uri or event.clip_uri

    async def _ensure_vlm_resident(self) -> None:
        """Reload an evicted VLM, and freshen its idle clock so it is not evicted again
        while it is genuinely in use.

        Without this the system describes correctly for exactly `idle_unload_seconds`
        after boot and then never again: the idle sweeper evicts the VLM (600s of no
        activity being the normal state of a camera watching an empty corridor), and
        nothing on the describe path ever brings it back. Every subsequent event
        publishes `description_unavailable=True` with a flat 0.5 threat score, which
        makes triage meaningless, while `/health` reports the VLM as UNLOADED — the
        state an operator reads as *correctly idle*, not as a fault.

        Inside the admission-gated section on purpose: a reload is a multi-second
        `from_pretrained`, and running it here means only the one escalation waits for
        it, never the camera pipeline. A failure is logged and swallowed rather than
        raised, so it cannot wedge the gate or cost the event — `describe()` then fails
        on its own load-state guard and §9's fallback publishes as usual.
        """
        if self._vlm_model_key is None:
            return
        try:
            await self._resident_set.ensure((self._vlm_model_key,), self._clock())
        except Exception as error:
            logger.warning(
                "could not make vlm %s resident; the describe will fall back: %s",
                self._vlm_model_key,
                error,
            )

    def _require_vlm(self) -> tuple[VisionLanguageModel, str]:
        """The VLM and its model key, on the paths that only run once one exists.

        `self._vlm` and `self._vlm_model_key` are optional together (see `__init__`),
        and `_describe` returns the skipped-description event before reaching anything
        below. That makes every caller of this method unreachable with no VLM — but
        "unreachable" is a claim about today's call graph, and a future path into the
        describe machinery that forgot the guard would otherwise fail with an
        `AttributeError` on `None` several frames deep.

        Raising here instead keeps that failure legible *and* keeps it inside
        `_describe`'s `except Exception` — so even the bug degrades to §9's published
        fallback event rather than losing the escalation.
        """
        if self._vlm is None or self._vlm_model_key is None:
            raise RuntimeError(
                "the describe path was reached with no vision-language model loaded; "
                "no configured camera enabled scene_description"
            )
        return self._vlm, self._vlm_model_key

    async def _run_vlm(self, vlm_request: VisionRequest) -> SceneDescription:
        vlm, _ = self._require_vlm()
        async with asyncio.timeout(self._timeout_seconds):
            return await vlm.describe(vlm_request)

    async def _describe_surviving_one_oom(self, vlm_request: VisionRequest) -> SceneDescription:
        """Spec §9's VLM-OOM row: evict, retry once, mark unhealthy, carry on.

        Only the OOM case takes this path. Everything else — a timeout, a malformed
        reply, a load-state guard — is one bad describe and is handled by the caller's
        §9 fallback; retrying those would just spend the GPU slot twice.

        **Which model is evicted, and why it is not the LRU.** The spec says "evict
        LRU". Taken literally here that evicts the *detector*: its idle clock is only
        stamped at boot (the scheduler freshens the VLM's before every describe, and
        nothing freshens the detector's), so the detector is permanently the
        least-recently-used model. Unloading it would stop detection outright — the
        exact opposite of the same row's "pipeline continues detection-only". So the
        VLM is what gets evicted: it is the allocator that failed, and it is the only
        eviction that frees VRAM without blinding the camera. With two models
        configured these are the only two choices; a future third model would want a
        genuine LRU pass over everything *except* the always-on detector.

        The eviction is what makes the retry worth attempting: `shutdown()` runs
        `gc.collect()` then `torch.cuda.empty_cache()` (measured at ~2.4 GB -> ~54 MiB
        in Task 13), so the reload starts from a compacted pool rather than the
        fragmented one that just failed.

        A second OOM means the card genuinely cannot hold this model right now. The
        model is evicted again and marked `UNHEALTHY`, so `/health` says so instead of
        reporting the post-eviction `UNLOADED` an operator reads as *correctly idle*,
        and the error is re-raised into the caller's §9 fallback so the event still
        publishes with `description_unavailable=True`. Recovery is not permanent-off:
        the next escalation's `_ensure_vlm_resident()` will try to load it again, and
        succeeds if whatever else was on the card has gone. Detection never stopped.
        """
        _, model_key = self._require_vlm()
        try:
            return await self._run_vlm(vlm_request)
        except Exception as error:
            if not _is_out_of_memory(error):
                raise
            logger.warning(
                "vlm %s ran out of memory; evicting and retrying once: %s",
                model_key,
                error,
            )
        await self._resident_set.evict(model_key)
        try:
            await self._resident_set.ensure((model_key,), self._clock())
            return await self._run_vlm(vlm_request)
        except Exception as retry_error:
            detail = f"out of memory on two consecutive describes: {retry_error}"
            logger.error("vlm %s %s", model_key, detail)
            # Evict *before* marking: `shutdown()` sets the runtime back to UNLOADED,
            # so the other order would erase the very state this is recording.
            with contextlib.suppress(Exception):
                await self._resident_set.evict(model_key)
            self._resident_set.mark_unhealthy(model_key, detail)
            raise

    def _to_epoch(self, camera_id: str, source_timestamp: float) -> float:
        """Rebase a source timestamp onto Unix epoch seconds through a per-camera anchor.

        `occurred_at` used to be `scene.timestamp` verbatim, which for an RTSP camera is
        `time.monotonic()` — a live run published `"occurred_at": 51181.128868795`. That
        value resets on every restart, does not share an origin with a replay camera in
        the same process, and cannot be turned into a wall time by a consumer, while the
        codec's docstring tells the Go consumer to sort by it.

        **Why the anchor is per camera, not per process.** A first fix anchored the
        whole process once, at construction, to `(wall_clock(), clock())`, and rebased
        every event as `wall_at_anchor + (source_timestamp - mono_at_anchor)`. That is
        exact only when a camera's `source_timestamp` shares `clock`'s timeline — true
        for `RtspSource`, whose frames carry `time.monotonic()`, and false for
        `FileSource`, whose frames carry the container's own pts starting at 0.0 per
        file. `time.monotonic()` counts from boot, so a replay event's `occurred_at`
        came out `wall_at_anchor - mono_at_anchor`, i.e. roughly one uptime in the past
        — about 14 hours on the box the live run came from — and a replay camera and an
        RTSP camera in the same process landed hours apart for events observed seconds
        apart, even though both anchors were read correctly.

        The fix is to stop assuming a shared timeline at all: each camera's anchor is
        established lazily, from *that camera's own* first-seen `source_timestamp`
        paired with the wall clock reading taken at that moment, and cached in
        `_camera_anchors`. `clock()` never enters this calculation — only `wall_clock`
        does, and only once per camera. A camera's first event therefore always lands
        at the wall time it was assembled, whatever timeline its source counts on and
        whatever `clock()` happens to read.

        **Why the anchor is read once per camera, and never again.** The obvious
        alternative — `time.time()` per event — makes `occurred_at` a fresh sample of a
        clock that `ntpd`, `chronyd` or a hypervisor can step backwards at any moment. Two
        events a second apart could then land out of order relative to each other, which
        is precisely the ordering the consumer is told to rely on. Offsetting from one
        anchor instead makes the difference between any two `occurred_at` values from the
        same camera *exactly* the difference between their source timestamps: the
        absolute value is as accurate as that camera's anchor was, and the ordering is as
        reliable as the source's own clock, which is to say perfectly.

        **The consequence, honestly stated.** A camera's first event is anchored at
        that event's assembly, not at some shared moment — so comparing `occurred_at`
        across two different cameras is only precise to when each camera's anchor was
        established, not to true simultaneity. That is a far smaller error than one
        uptime (typically well under a second, since a camera's first escalation is
        assembled shortly after the engine starts observing it), and unlike the
        process-wide anchor it degrades gracefully: it is never off by more than each
        camera's own anchoring delay, never by an unrelated system's uptime.
        """
        anchor = self._camera_anchors.get(camera_id)
        if anchor is None:
            anchor = (self._wall_clock(), source_timestamp)
            self._camera_anchors[camera_id] = anchor
        wall_at_anchor, source_at_anchor = anchor
        return wall_at_anchor + (source_timestamp - source_at_anchor)

    def _assemble(
        self,
        request: EscalationRequest,
        *,
        threat: ThreatScore,
        description: str,
        suggested_action: str,
        description_unavailable: bool,
        welfare: WelfareAssessment,
        metadata: dict[str, str] | None = None,
    ) -> Event:
        """The one place in the system that constructs an `Event` (S14).

        Every path — a good describe, a failed one, and a shutdown that abandoned the
        escalation before it was ever described — comes through here, precisely so no
        error path can produce a differently-shaped event, or none at all.

        `welfare` is keyword-only with no default, on purpose (T3 review): a default
        of `WelfareAssessment.none()` here would silently reproduce the exact bug this
        parameter exists to fix — `_describe`'s success path forgetting to carry
        `SceneDescription.welfare` across would still type-check, still pass every
        existing test, and `Event.welfare` would stay empty in production no matter
        how loudly the model reported a concern. Every caller must say explicitly
        which welfare opinion (or the deliberate absence of one) this event carries.

        Which is also why the console's recent-event ring is *first* written here:
        hanging it off the publish path instead would omit exactly the events an
        operator most needs to see in the console — the §9 degraded one, and the one a
        cancelled shutdown dead-lettered rather than published. The ring is bounded,
        per camera, and volatile; see `orchestrator/event_history.py` for the bound.

        The one later write is `_attach_clip`'s back-fill of `clip_uri`, which rewrites
        this entry rather than appending another. It has to come after: the clip is not
        finished at assembly, and delaying the record until it is would lose the event
        whenever the clip does not finish at all — the one thing §9 forbids.
        """
        labels, track_ids = _labels_and_tracks(request.scene)
        event = Event(
            event_id=request.event_id,
            camera_id=request.camera_id,
            occurred_at=self._to_epoch(request.camera_id, request.scene.timestamp),
            source_timestamp=request.scene.timestamp,
            reason=request.reason,
            threat=threat,
            description=description,
            suggested_action=suggested_action,
            labels=labels,
            track_ids=track_ids,
            subject_track_ids=request.subject_track_ids,
            description_unavailable=description_unavailable,
            welfare=welfare,
            # `Event.metadata` defaults via `field(default_factory=dict)`; passing
            # `{}` rather than `None` keeps every existing caller's event byte-identical
            # to what it was before this parameter existed.
            metadata=dict(metadata) if metadata else {},
        )
        self._recent.record(event)
        return event

    def _unavailable_event(self, request: EscalationRequest) -> Event:
        """Spec §9's degraded event: everything the metadata already knows, and an
        honest flag saying the VLM never spoke. There is no VLM opinion to carry on
        this path, so `welfare` is explicitly `WelfareAssessment.none()` rather than
        anything derived from a description that was never produced."""
        return self._assemble(
            request,
            threat=ThreatScore.from_value(_UNAVAILABLE_THREAT_VALUE),
            description=_metadata_description(request),
            suggested_action=_UNAVAILABLE_ACTION,
            description_unavailable=True,
            welfare=WelfareAssessment.none(),
        )

    def _skipped_event(self, request: EscalationRequest) -> Event:
        """The event for an escalation this camera never wanted described (spec §13).

        Shaped exactly like `_unavailable_event` — same metadata-derived description,
        same conservative mid-range threat — because from a consumer's point of view
        the description really is absent and `description_unavailable` really is true.
        What differs is *why*, and that rides in `metadata` under
        `DESCRIPTION_SKIPPED_METADATA_KEY` so a console can say "description is off for
        this camera" instead of "the model failed", and so an operator never learns to
        ignore a flag that otherwise means a genuine fault. See that constant for why
        the distinction is not a third state on the bool.

        No welfare opinion, for `_unavailable_event`'s reason and more strongly: no
        model looked at this frame at all.
        """
        return self._assemble(
            request,
            threat=ThreatScore.from_value(_UNAVAILABLE_THREAT_VALUE),
            description=_metadata_description(request),
            suggested_action=_SKIPPED_ACTION,
            description_unavailable=True,
            welfare=WelfareAssessment.none(),
            metadata={DESCRIPTION_SKIPPED_METADATA_KEY: "scene_description_disabled"},
        )

    async def _describe(self, request: EscalationRequest) -> Event:
        # Two ways to arrive here with nothing to ask: this camera did not enable
        # `scene_description`, or no camera did and the process never built a VLM.
        # Both are configuration, not failure, and both produce the same event.
        if not request.describe or self._vlm is None:
            return self._skipped_event(request)
        vlm_request = VisionRequest(
            keyframe=request.keyframe,
            scene=request.scene,
            history=request.history,
            camera_label=request.camera_label,
            reason_detail=request.detail,
        )
        try:
            description = await self._describe_surviving_one_oom(vlm_request)
            # Inside the guard on purpose. `SceneDescription.threat_value` is a bare
            # unvalidated float and `ThreatScore.from_value` rejects anything outside
            # [0, 1]; Task 13's Qwen adapter parses that number out of generated model
            # text, so an out-of-range value is an ordinary infrastructure failure. Left
            # outside, it raised straight out of the success path and the event was
            # dropped entirely -- the one thing spec §9 forbids. Here it degrades to the
            # same `description_unavailable=True` fallback as any other VLM failure.
            threat = ThreatScore.from_value(description.threat_value)
        except Exception as error:
            logger.warning(
                "vlm describe failed for camera %s event %s: %s",
                request.camera_id,
                request.event_id,
                error,
            )
            return self._unavailable_event(request)
        return self._assemble(
            request,
            threat=threat,
            description=description.description,
            suggested_action=description.suggested_action,
            description_unavailable=False,
            # T3: the VLM's own opinion, carried across rather than dropped — see
            # `_assemble`'s docstring for why `welfare` has no default that would let
            # this be forgotten silently.
            welfare=_basis_for(request, description.welfare),
        )

    async def _attach_clip(self, event: Event, request: EscalationRequest) -> Event:
        """Finish the clip and put its URI on the event — and on the console's copy.

        The ring is written at assembly, before this runs, which is deliberate and
        stays that way: spec §9 requires the event to survive a clip that never
        finishes, and it does — the failure path below leaves both the published event
        and the ring entry with `clip_uri=None`. Only the success path writes again,
        back-filling the one entry that is already there. See
        `orchestrator/event_history.py` for why that is a rewrite rather than a second
        append, and for the sequence bump that lets a live subscriber tell the update
        apart from a redelivery.
        """
        if request.clip is None:
            return event
        try:
            clip_uri = await request.clip.finish()
        except Exception as error:
            logger.warning(
                "clip finish failed for camera %s event %s: %s",
                request.camera_id,
                request.event_id,
                error,
            )
            # A failed finish() can leave the muxer's container open and its temp file
            # on disk. abort() is contractually idempotent and never raises, so this is
            # always safe — and without it every failed clip leaks a file handle.
            await request.clip.abort()
            return event
        if not self._recent.attach_clip(event.camera_id, event.event_id, clip_uri):
            # The ring wrapped between assembly and upload. Bounded and volatile by
            # construction, so this is a lost console link, never a lost event.
            logger.debug(
                "clip %s arrived after event %s had aged out of camera %s's ring",
                clip_uri,
                event.event_id,
                event.camera_id,
            )
        return replace(event, clip_uri=clip_uri)
