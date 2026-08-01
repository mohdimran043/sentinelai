"""One asyncio task per camera (spec §5.3): decode → detect → track → motion → gate.

Threading
---------
`FrameSource.__aiter__`/`.packets()` and `ObjectDetector.detect` are `async` by contract
precisely because PyAV demux/decode and YOLO inference are blocking C-extension calls.
The adapters that perform them (`FileSource`/`RtspSource`, `Yolo11Detector`) hide the
offload behind that `async def` — a background thread feeding a queue, or
`await asyncio.to_thread(...)`. The runner's obligation is therefore only to `await`
these calls and never to add a second, redundant executor hop around an already-async
port method. `Tracker.update` and `MotionAnalyzer.analyze` are cheap, pure CPU (the port
keeps them synchronous deliberately) and the escalation gate is a pure function, so all
three run inline on the loop.

Backpressure
------------
A `_LatestSlot` mailbox decouples frame *arrival* from frame *processing*. A `_produce`
task drains the source eagerly and always overwrites the slot with the newest frame; an
overwrite before the consumer collects the previous frame is a drop, counted in
`CameraTelemetry.frames_dropped`. This is what makes "drop the stale frame, process the
newest" real: a naive `async for frame in source: await detector.detect(frame)` loop
never drops anything, because it only asks the source for a new frame once the previous
one is fully processed — producing a delayed complete record where surveillance wants
current reality.

Both `_produce` and `_packet_loop` yield once per item (`asyncio.sleep(0)`, zero
wall-clock time). Real sources already suspend per item, so the yield changes nothing
for them; a source that never suspends would otherwise run its entire stream inside a
single event-loop step, starving the consumer and turning every frame but the last into
a drop.

Clip lifecycle
--------------
Only the escalation that *opens* a clip ever carries its `ClipHandle` on an
`EscalationRequest`. A later escalation while the clip is still recording only extends
`_ActiveClip.deadline` — submitting its own request with `clip=None` — because finishing
a shared handle twice, or before its (possibly extended) post-roll window has closed, is
undefined. The clip-owning request is queued lazily, from the packet loop, the moment
`packet.pts` first reaches the (possibly-extended) deadline; never from the frame loop,
so escalation submission stays non-blocking and immediate for every *other* request.

Deadlock analysis
-----------------
Three concurrent flows share one lock (`_clip_lock`) and one mailbox:

  * `_produce` never blocks on the consumer (`_LatestSlot.put` is synchronous and
    overwrites), so the producer cannot be starved by a slow pipeline, and `close()`
    guarantees the consumer always terminates rather than waiting forever.
  * `_escalate` (consumer side) and `_packet_loop` both take `_clip_lock` and neither
    acquires a second lock while holding it, so there is no lock-order cycle. Each
    holds it only across clip-writer I/O, which depends on neither of the other flows.
  * `VlmScheduler.submit` is synchronous and drops when full, so a stalled VLM can
    never apply back-pressure to the camera.

`run()` waits for the packet loop to end naturally rather than cancelling it, so an open
clip's post-roll is not truncated by our own shutdown; that is safe because the frame
stream and the packet stream come from the same demux pass, so the frame stream ending
means the packet stream is ending too.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import get_settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.domain.policy.escalation import GateState, decide, force
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData, FrameSource
from sentinel_ai.ports.tracker import Tracker

logger = logging.getLogger(__name__)

__all__ = ["CameraRunner", "CameraTelemetry"]

_DEFAULT_FPS = 10.0
"""Used only until two frames have been seen and a real interval can be measured."""

_HISTORY_MAXLEN = 5
"""How many previous escalation details the VLM gets as context (spec §22)."""


@dataclass(frozen=True, slots=True)
class CameraTelemetry:
    camera_id: str
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    last_frame_at: float | None
    last_escalation_at: float | None


@dataclass(frozen=True, slots=True)
class _ActiveClip:
    """A clip currently recording, and the escalation that will carry it."""

    handle: ClipHandle
    deadline: float
    request: EscalationRequest


class _LatestSlot:
    """Single-slot mailbox: only the newest unconsumed frame survives.

    An overwrite before the consumer collects the previous frame is exactly the
    backpressure drop spec §5.3 requires. `close()` lets the producer signal
    end-of-stream so the consumer stops instead of waiting forever.
    """

    def __init__(self) -> None:
        self._frame: FrameData | None = None
        self._closed = False
        self._event = asyncio.Event()
        self.dropped = 0

    def put(self, frame: FrameData) -> None:
        if self._frame is not None:
            self.dropped += 1
        self._frame = frame
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def get(self) -> FrameData | None:
        """The newest frame, or None once the stream has closed and drained."""
        while True:
            if self._frame is not None:
                frame, self._frame = self._frame, None
                return frame
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()


class CameraRunner:
    def __init__(
        self,
        *,
        camera_id: str,
        camera_label: str,
        source: FrameSource,
        detector: ObjectDetector,
        tracker: Tracker,
        motion: MotionAnalyzer,
        profile: CameraProfile,
        scheduler: VlmScheduler,
        clip_writer: ClipWriter | None,
        preroll: PreRollBuffer,
        clock: Callable[[], float],
        detect_every_n_frames: int = 1,
    ) -> None:
        if detect_every_n_frames < 1:
            raise ValueError(f"detect_every_n_frames must be >= 1, got {detect_every_n_frames}")
        self._camera_id = camera_id
        self._camera_label = camera_label
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._motion = motion
        self._profile = profile
        self._scheduler = scheduler
        self._clip_writer = clip_writer
        self._preroll = preroll
        self._clock = clock
        self._detect_every_n_frames = detect_every_n_frames

        # `clip_postroll_seconds` has no S11 constructor slot: it is a process-wide
        # tuning value (S1), read once here rather than threaded through every
        # `CameraRunner` construction site.
        self._clip_postroll_seconds = get_settings().clip_postroll_seconds

        self._gate_state = GateState.initial(profile, clock())
        self._history: deque[str] = deque(maxlen=_HISTORY_MAXLEN)
        self._active_clip: _ActiveClip | None = None
        self._clip_lock = asyncio.Lock()
        self._slot: _LatestSlot | None = None

        self._last_scene: SceneState | None = None
        self._last_keyframe: FrameData | None = None
        self._last_signature_len: int | None = None
        self._last_frame_index: int | None = None
        self._last_processed_timestamp: float | None = None
        self._last_two_timestamps: tuple[float, float] | None = None

        self._frames_seen = 0
        self._detections_run = 0
        self._escalations = 0
        self._escalations_dropped = 0
        self._discontinuities = 0
        self._last_frame_at: float | None = None
        self._last_escalation_at: float | None = None

    # -- lifecycle ------------------------------------------------------------------

    async def run(self) -> None:
        slot = _LatestSlot()
        self._slot = slot
        produce_task = asyncio.create_task(self._produce(slot))
        packets_task = asyncio.create_task(self._packet_loop())
        try:
            await self._consume(slot)
            # The frame stream has ended. Re-raise whatever the producer failed with —
            # a decode error must not be mistaken for a clean end of stream — then let
            # the packet stream drain so an open clip's post-roll survives our shutdown.
            await produce_task
            await packets_task
        finally:
            produce_task.cancel()
            packets_task.cancel()
            await asyncio.gather(produce_task, packets_task, return_exceptions=True)
            await self._abandon_open_clip()
            await self._source.close()

    async def describe_now(self) -> UUID:
        """USER_REQUESTED path (spec §6) — uses domain `force()`; returns the event id."""
        scene, keyframe = self._last_scene, self._last_keyframe
        if scene is None or keyframe is None:
            raise RuntimeError(f"camera {self._camera_id!r} has not processed a frame yet")
        now = self._clock()
        outcome = force(EscalationReason.USER_REQUESTED, self._gate_state, now)
        self._gate_state = outcome.state
        return await self._escalate(
            scene=scene,
            keyframe=keyframe,
            reason=EscalationReason.USER_REQUESTED,
            detail=outcome.decision.detail,
            now=now,
        )

    def telemetry(self) -> CameraTelemetry:
        return CameraTelemetry(
            camera_id=self._camera_id,
            frames_seen=self._frames_seen,
            frames_dropped=self._slot.dropped if self._slot is not None else 0,
            detections_run=self._detections_run,
            escalations=self._escalations,
            escalations_dropped=self._escalations_dropped,
            discontinuities=self._discontinuities,
            last_frame_at=self._last_frame_at,
            last_escalation_at=self._last_escalation_at,
        )

    # -- stage 1: frame arrival, with backpressure -----------------------------------

    async def _produce(self, slot: _LatestSlot) -> None:
        try:
            async for frame in self._source:
                self._frames_seen += 1
                self._last_frame_at = frame.timestamp
                if frame.frame_index % self._detect_every_n_frames == 0:
                    slot.put(frame)
                await asyncio.sleep(0)  # cooperative yield; see the module docstring
        finally:
            slot.close()

    async def _consume(self, slot: _LatestSlot) -> None:
        while True:
            frame = await slot.get()
            if frame is None:
                return
            await self._process_frame(frame)

    # -- stages 2-5: detect, track, motion, gate -------------------------------------

    async def _process_frame(self, frame: FrameData) -> None:
        detections = await self._detector.detect(frame)
        self._detections_run += 1
        tracks = self._tracker.update(detections, frame.timestamp)
        signals = self._motion.analyze(frame)

        if self._is_discontinuous(frame, signals):
            await self._on_discontinuity(frame)

        self._last_signature_len = len(signals.scene_signature)
        self._last_frame_index = frame.frame_index
        if self._last_processed_timestamp is not None:
            self._last_two_timestamps = (self._last_processed_timestamp, frame.timestamp)
        self._last_processed_timestamp = frame.timestamp

        scene = SceneState(
            camera_id=self._camera_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            detections=detections,
            tracks=tracks,
            motion_energy=signals.motion_energy,
            scene_signature=signals.scene_signature,
        )
        self._last_scene = scene
        self._last_keyframe = frame

        outcome = decide(scene, self._profile, self._gate_state)
        self._gate_state = outcome.state
        decision = outcome.decision
        if decision.should_escalate and decision.reason is not None:
            await self._escalate(
                scene=scene,
                keyframe=frame,
                reason=decision.reason,
                detail=decision.detail,
                now=frame.timestamp,
            )

    def _is_discontinuous(self, frame: FrameData, signals: MotionSignals) -> bool:
        """Spec §5.2/§6: a stream discontinuity, not a comparable `SceneState`.

        Two observable signals, both of which invalidate everything derived from
        frame-to-frame history:

          * a `frame_index` or timestamp regression — an RTSP reconnect (Task 14)
            restarts both at zero;
          * a scene-signature length change — a mid-stream resolution renegotiation
            leaves two histograms with different bin counts.

        The domain already returns 0.0/None rather than raising on the second case
        (Phase 1A review finding B1); resetting the derived state so the next frame
        starts clean is the caller's job, and this is the caller.
        """
        if self._last_frame_index is not None and frame.frame_index < self._last_frame_index:
            return True
        if (
            self._last_processed_timestamp is not None
            and frame.timestamp < self._last_processed_timestamp
        ):
            return True
        return (
            self._last_signature_len is not None
            and len(signals.scene_signature) != self._last_signature_len
        )

    async def _on_discontinuity(self, frame: FrameData) -> None:
        logger.info(
            "stream discontinuity on camera %s at frame %d (t=%.3f): resetting derived state",
            self._camera_id,
            frame.frame_index,
            frame.timestamp,
        )
        self._discontinuities += 1
        self._tracker.reset()
        self._motion.reset()
        # Buffered packets belong to the old timeline: their pts no longer order
        # against the new stream's, and after a resolution change they cannot even be
        # remuxed into the same clip.
        self._preroll.clear()
        self._gate_state = GateState.initial(self._profile, frame.timestamp)
        self._last_two_timestamps = None
        await self._reanchor_open_clip(frame.timestamp)

    # -- escalation, clip lifecycle, scheduler submission ----------------------------

    async def _escalate(
        self,
        *,
        scene: SceneState,
        keyframe: FrameData,
        reason: EscalationReason,
        detail: str,
        now: float,
    ) -> UUID:
        event_id = uuid4()
        history = tuple(self._history)
        self._history.append(detail)
        self._last_escalation_at = now

        def request_with(clip: ClipHandle | None) -> EscalationRequest:
            return EscalationRequest(
                camera_id=self._camera_id,
                event_id=event_id,
                reason=reason,
                detail=detail,
                scene=scene,
                keyframe=keyframe,
                profile=self._profile,
                camera_label=self._camera_label,
                history=history,
                clip=clip,
            )

        if self._clip_writer is None:
            self._submit(request_with(None))
            return event_id

        async with self._clip_lock:
            if self._active_clip is None:
                handle = await self._clip_writer.open(
                    self._camera_id, event_id, self._estimated_fps()
                )
                for packet in self._preroll.flush():
                    await handle.append(packet)
                self._active_clip = _ActiveClip(
                    handle=handle,
                    deadline=now + self._clip_postroll_seconds,
                    request=request_with(handle),
                )
            else:
                # A clip is already recording: extend its post-roll rather than open a
                # second, overlapping one (spec §5.5). This escalation still gets its
                # own event — just no clip of its own, since a `ClipHandle` may only
                # ever be finished once, and only after its window has actually closed.
                self._active_clip = replace(
                    self._active_clip, deadline=now + self._clip_postroll_seconds
                )
                self._submit(request_with(None))
        return event_id

    async def _packet_loop(self) -> None:
        """Feeds the pre-roll ring and any recording clip from the same demux pass.

        The lock is held across `append`, so a live packet arriving mid-pre-roll-flush
        (itself inside the same lock, in `_escalate`) waits rather than racing ahead of
        older, still-buffered packets: the clip stays pts-ordered without a second queue.
        """
        async for packet in self._source.packets():
            self._preroll.append(packet)
            async with self._clip_lock:
                active = self._active_clip
                if active is not None:
                    await active.handle.append(packet)
                    if packet.pts >= active.deadline:
                        self._active_clip = None
                        self._submit(active.request)
            await asyncio.sleep(0)  # cooperative yield; see the module docstring

    async def _reanchor_open_clip(self, now: float) -> None:
        """Move a recording clip's deadline onto the post-discontinuity timeline.

        After a reconnect restarts pts at zero, a deadline expressed in the old
        timeline would never be reached and the clip would record until the camera
        stops.
        """
        async with self._clip_lock:
            if self._active_clip is not None:
                self._active_clip = replace(
                    self._active_clip, deadline=now + self._clip_postroll_seconds
                )

    async def _abandon_open_clip(self) -> None:
        """Shutdown with a clip still recording: discard the partial file, keep the event.

        The post-roll never closed, so the clip is incomplete — `abort()` is
        contractually infallible, which is why it is safe to await here, possibly while
        already being cancelled. The escalation is still submitted, with `clip=None`:
        spec §9 forbids losing an anomaly event to an infrastructure or lifecycle
        failure, and a camera stopping mid-post-roll is one.
        """
        pending, self._active_clip = self._active_clip, None
        if pending is None:
            return
        with contextlib.suppress(Exception):
            await pending.handle.abort()
        self._submit(replace(pending.request, clip=None))

    def _submit(self, request: EscalationRequest) -> None:
        if self._scheduler.submit(request):
            self._escalations += 1
        else:
            self._escalations_dropped += 1

    def _estimated_fps(self) -> float:
        if self._last_two_timestamps is None:
            return _DEFAULT_FPS
        previous, current = self._last_two_timestamps
        delta = current - previous
        return 1.0 / delta if delta > 0 else _DEFAULT_FPS
