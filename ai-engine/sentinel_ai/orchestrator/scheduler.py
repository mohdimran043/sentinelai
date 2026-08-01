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
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, Event, SceneState, ThreatScore
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher, FailedEventSink
from sentinel_ai.ports.frame_source import FrameData
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


_UNAVAILABLE_THREAT_VALUE = 0.5
"""A conservative mid-range placeholder: severity truly is unknown without a
description, and 0.5 neither over- nor under-states it for downstream triage."""

_UNAVAILABLE_ACTION = "Review the clip when available."

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


def _metadata_description(request: EscalationRequest) -> str:
    """Fallback description built from cheap signals alone — no VLM call required."""
    labels, _ = _labels_and_tracks(request.scene)
    what = ", ".join(labels) if labels else "motion"
    return f"{request.reason.value}: {what} ({request.detail})"


class VlmScheduler:
    def __init__(
        self,
        vlm: VisionLanguageModel,
        publisher: EventPublisher,
        admission: AdmissionGate,
        *,
        resident_set: ResidentSet,
        vlm_model_key: str,
        dead_letter: FailedEventSink,
        maxsize: int,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._vlm = vlm
        self._publisher = publisher
        self._admission = admission
        self._resident_set = resident_set
        self._vlm_model_key = vlm_model_key
        # Required, not optional, for the same reason `resident_set` is: an
        # unwired last resort is indistinguishable from no last resort, and the
        # failure it guards against is silent by construction.
        self._dead_letter = dead_letter
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._queue: asyncio.Queue[EscalationRequest] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0
        self._publish_failures = 0

    def submit(self, request: EscalationRequest) -> bool:
        """Non-blocking. Returns False and counts a drop when the queue is full.

        Deliberately not a coroutine: the camera pipeline must never await the VLM,
        and a `def` makes that unrepresentable rather than merely discouraged.
        """
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("vlm queue full: dropping escalation for camera %s", request.camera_id)
            return False
        return True

    async def run(self) -> None:
        """The single worker loop. Cancel to stop."""
        while True:
            request = await self._queue.get()
            try:
                await self._process(request)
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
            finally:
                self._queue.task_done()

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

    async def _process(self, request: EscalationRequest) -> None:
        await self._admission.acquire(self._clock())
        try:
            await self._ensure_vlm_resident()
            event = await self._describe(request)
            event = await self._attach_clip(event, request)
            await self._publish(event)
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
        try:
            await self._resident_set.ensure((self._vlm_model_key,), self._clock())
        except Exception as error:
            logger.warning(
                "could not make vlm %s resident; the describe will fall back: %s",
                self._vlm_model_key,
                error,
            )

    async def _run_vlm(self, vlm_request: VisionRequest) -> SceneDescription:
        async with asyncio.timeout(self._timeout_seconds):
            return await self._vlm.describe(vlm_request)

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
        try:
            return await self._run_vlm(vlm_request)
        except Exception as error:
            if not _is_out_of_memory(error):
                raise
            logger.warning(
                "vlm %s ran out of memory; evicting and retrying once: %s",
                self._vlm_model_key,
                error,
            )
        await self._resident_set.evict(self._vlm_model_key)
        try:
            await self._resident_set.ensure((self._vlm_model_key,), self._clock())
            return await self._run_vlm(vlm_request)
        except Exception as retry_error:
            detail = f"out of memory on two consecutive describes: {retry_error}"
            logger.error("vlm %s %s", self._vlm_model_key, detail)
            # Evict *before* marking: `shutdown()` sets the runtime back to UNLOADED,
            # so the other order would erase the very state this is recording.
            with contextlib.suppress(Exception):
                await self._resident_set.evict(self._vlm_model_key)
            self._resident_set.mark_unhealthy(self._vlm_model_key, detail)
            raise

    async def _describe(self, request: EscalationRequest) -> Event:
        labels, track_ids = _labels_and_tracks(request.scene)
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
            description_text = description.description
            suggested_action = description.suggested_action
            unavailable = False
        except Exception as error:
            logger.warning(
                "vlm describe failed for camera %s event %s: %s",
                request.camera_id,
                request.event_id,
                error,
            )
            threat = ThreatScore.from_value(_UNAVAILABLE_THREAT_VALUE)
            description_text = _metadata_description(request)
            suggested_action = _UNAVAILABLE_ACTION
            unavailable = True
        # One construction site, both paths: S14 owns event assembly precisely so that
        # no error path can produce a differently-shaped event, or none at all.
        return Event(
            event_id=request.event_id,
            camera_id=request.camera_id,
            occurred_at=request.scene.timestamp,
            reason=request.reason,
            threat=threat,
            description=description_text,
            suggested_action=suggested_action,
            labels=labels,
            track_ids=track_ids,
            description_unavailable=unavailable,
        )

    async def _attach_clip(self, event: Event, request: EscalationRequest) -> Event:
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
        return replace(event, clip_uri=clip_uri)
