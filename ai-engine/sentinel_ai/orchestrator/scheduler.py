"""The VLM escalation queue (spec §5.4) and the S14 event-assembly worker (spec §6, §9).

Exactly one worker processes escalations: there is one GPU and one set of resident
VLM weights, so concurrent `describe()` calls would contend for the same VRAM.
Drop-on-full is safe because the token bucket and the cooldown already bound the
camera-side arrival rate (spec §4.1) — a full queue should be rare in practice, and
when it happens the drop is counted, never raised.

Event assembly happens here and only here (S14): this is the one place in the system
that constructs an `Event`. Every error path below still produces one — spec §9's
governing rule is that an anomaly event is never lost to an infrastructure failure.
That rule is enforced in three places:

  * a failed or timed-out `describe()` -- or one that returns an unusable threat
    value -- yields a metadata-derived description flagged
    `description_unavailable=True`;
  * a failed `ClipHandle.finish()` yields an event with `clip_uri=None`;
  * a failure anywhere — including the publisher itself — still releases the admission
    slot in a `finally` and leaves the worker alive for the next escalation. A leaked
    slot at concurrency 1 wedges the GPU for the lifetime of the process, and a worker
    that dies on one poisoned request silently stops describing everything after it.

VLM residency
-------------
Every describe is preceded by `ResidentSet.ensure((vlm_model_key,), now)`. That single
call does two jobs, and the system is broken without either: it *reloads* a VLM the
600s idle sweep has evicted, and it stamps `last_used_at`, which is what makes the
sweep's window mean "idle" rather than "600s since boot". It goes through
`ResidentSet` rather than calling `ModelRuntime.initialize()` directly so that
`plan_residency`'s admission arithmetic and eviction bookkeeping stay authoritative.

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
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, Event, SceneState, ThreatScore
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.vision_llm import VisionLanguageModel, VisionRequest

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
        maxsize: int,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._vlm = vlm
        self._publisher = publisher
        self._admission = admission
        self._resident_set = resident_set
        self._vlm_model_key = vlm_model_key
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._queue: asyncio.Queue[EscalationRequest] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0

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

    async def _process(self, request: EscalationRequest) -> None:
        await self._admission.acquire(self._clock())
        try:
            await self._ensure_vlm_resident()
            event = await self._describe(request)
            event = await self._attach_clip(event, request)
            await self._publisher.publish(event)
        finally:
            self._admission.release(self._clock())

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
            async with asyncio.timeout(self._timeout_seconds):
                description = await self._vlm.describe(vlm_request)
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
