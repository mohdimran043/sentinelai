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

Time base
---------
There is exactly **one** clock per camera pipeline, and it is the source's:
`FrameData.timestamp` and `EncodedPacket.pts`, which `FrameSource` guarantees are the
same timeline as each other. `CameraRunner` deliberately takes no `Callable[[], float]`
of its own. Every time-valued decision here reads the frame or packet that occasioned
it — the gate's initial `TokenBucket` stamp comes from the first frame, `describe_now()`
takes its `now` from the last processed scene, and a clip's post-roll deadline is
compared against `packet.pts` and so must be computed from it.

This is not stylistic. A runner that carried a second clock would compare the two
without either being wrong on its own: `GateState.initial` would stamp
`bucket.updated_at` from clock B while `decide()` refills against `scene.timestamp` on
clock A, and once B reads ahead of A — guaranteed for `FileSource`, whose timeline
starts at 0.0, against any `time.monotonic` runner clock — `TokenBucket.refilled`'s
regressing-clock guard makes every refill a no-op. The camera spends its burst tokens
and is then rate-limited into silence for the lifetime of the process. The same
divergence stops any clip ever reaching its deadline, and since an already-recording
clip only ever extends, the camera stops producing evidence too.

The scheduler's clock is a genuinely different thing and stays: `AdmissionGate` spaces
GPU admissions with `asyncio.sleep`, so it needs real elapsed time. The two agree only
for a live RTSP source and must not be conflated.

Clip lifecycle
--------------
Only the escalation that *opens* a clip ever carries its `ClipHandle` on an
`EscalationRequest`. A later escalation while the clip is still recording only extends
`_ActiveClip.deadline` — submitting its own request with `clip=None` — because finishing
a shared handle twice, or before its (possibly extended) post-roll window has closed, is
undefined. The clip-owning request is queued lazily, from the packet loop, the moment
`packet.pts` first reaches the (possibly-extended) deadline; never from the frame loop,
so escalation submission stays non-blocking and immediate for every *other* request.

A clip only ever ends one of two ways: `finish()` at its deadline, or `abort()` — on
shutdown, or on a discontinuity, both of which leave it unable to be completed. It is
never carried across a discontinuity, for the same reason the pre-roll is not.

Deadlock analysis
-----------------
Three concurrent flows share one lock (`_clip_lock`) and one mailbox:

  * `_produce` never blocks on the consumer (`_LatestSlot.put` is synchronous and
    overwrites), so the producer cannot be starved by a slow pipeline, and `close()`
    guarantees the consumer always terminates rather than waiting forever.
  * `_escalate` and `_on_discontinuity` (both consumer side, never nested) and
    `_packet_loop` take `_clip_lock`, and none acquires a second lock while holding it,
    so there is no lock-order cycle. Each holds it only across clip-writer I/O, which
    depends on neither of the other flows.
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
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import get_settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.domain.policy.escalation import GateState, decide, force
from sentinel_ai.domain.zone import Zone
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
    zone: Zone | None = None
    """Which zone this camera watches, or None when nobody has grouped it (T1).

    Static configuration riding on a counters record, deliberately: `GET /cameras` is
    built from these snapshots, so a console that groups cameras reads the grouping
    from the same object it reads the liveness from — one request, and no way for the
    two to describe different sets of cameras. Defaulted so that every existing
    construction site keeps working and an ungrouped camera stays representable.
    """


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
        detect_every_n_frames: int = 1,
        zone: Zone | None = None,
    ) -> None:
        if detect_every_n_frames < 1:
            raise ValueError(f"detect_every_n_frames must be >= 1, got {detect_every_n_frames}")
        self._camera_id = camera_id
        self._camera_label = camera_label
        # Carried, never read by the pipeline: the runner is the only per-camera object
        # the API can reach, so the zone travels here to reach `telemetry()`. Behaviour
        # does not branch on it — a zone changes how a console groups a camera, not how
        # this loop watches one.
        self._zone = zone
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._motion = motion
        self._profile = profile
        self._scheduler = scheduler
        self._clip_writer = clip_writer
        self._preroll = preroll
        self._detect_every_n_frames = detect_every_n_frames

        # `clip_postroll_seconds` has no S11 constructor slot: it is a process-wide
        # tuning value (S1), read once here rather than threaded through every
        # `CameraRunner` construction site.
        self._clip_postroll_seconds = get_settings().clip_postroll_seconds

        # Deferred to the first frame, which is the earliest point at which a time on
        # the only timeline this pipeline has is available. `_on_discontinuity`
        # already re-initialises it from `frame.timestamp`; this makes construction
        # agree with it instead of seeding the token bucket from somewhere else.
        self._gate_state: GateState | None = None
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
        """USER_REQUESTED path (spec §6) — uses domain `force()`; returns the event id.

        `now` is the last processed scene's timestamp, not a separate clock reading:
        `force()` writes it to `GateState.last_escalation_at`, which the cooldown then
        compares against `scene.timestamp`, and the clip deadline derived from it is
        compared against `packet.pts`. Both are the source's timeline. The guard below
        already guarantees the value exists.
        """
        scene, keyframe = self._last_scene, self._last_keyframe
        if scene is None or keyframe is None or self._gate_state is None:
            raise RuntimeError(f"camera {self._camera_id!r} has not processed a frame yet")
        now = scene.timestamp
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
            zone=self._zone,
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
        if self._gate_state is None:
            self._gate_state = GateState.initial(self._profile, frame.timestamp)

        detections = await self._detector.detect(frame)
        self._detections_run += 1

        # Before `update()`/`analyze()`, not after. Both of the regression signals are
        # readable off the frame alone, and everything those two stages derive is
        # meaningless across the break: track ids matched against pre-reconnect tracks
        # (with inflated `age_frames`), a motion energy computed as a delta against a
        # frame from the old timeline. Resetting afterwards reset the gate correctly and
        # then immediately fed it a `SceneState` built from the state just declared
        # invalid — and `min_track_frames` and the speed trigger both read those fields,
        # so the first frame after a reconnect could fire a spurious escalation.
        discontinuous = self._is_timeline_regression(frame)
        if discontinuous:
            await self._on_discontinuity(frame)

        tracks = self._tracker.update(detections, frame.timestamp)
        signals = self._motion.analyze(frame)

        # The third signal is the only one that needs the analyzer's output, so it can
        # only be handled here, after the fact: this frame's `signals` are already
        # computed by the time we learn they are incomparable, and the reset protects
        # the frames after it. A timeline regression has already reset everything, so
        # a signature length that also moved is the same discontinuity, not a second one.
        if not discontinuous and self._is_signature_change(signals):
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

    def _is_timeline_regression(self, frame: FrameData) -> bool:
        """Spec §5.2/§6: the stream restarted — an RTSP reconnect (Task 14) restarts
        `frame_index` and timestamps at zero.

        Readable off the frame alone, which is what lets it be checked before anything
        is derived from frame-to-frame history.
        """
        if self._last_frame_index is not None and frame.frame_index < self._last_frame_index:
            return True
        return (
            self._last_processed_timestamp is not None
            and frame.timestamp < self._last_processed_timestamp
        )

    def _is_signature_change(self, signals: MotionSignals) -> bool:
        """Spec §5.2/§6: a mid-stream resolution renegotiation leaves two histograms
        with different bin counts, so the two `SceneState`s are not comparable.

        Unlike a timeline regression this is only visible in the analyzer's output. The
        domain already returns 0.0/None rather than raising on it (Phase 1A review
        finding B1); resetting the derived state so the *next* frame starts clean is the
        caller's job, and this is the caller.
        """
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
        # An already-recording clip is the same problem as the pre-roll, one step later:
        # left open it goes on appending new-timeline packets directly onto old-timeline
        # ones, producing a file with a non-monotonic pts sequence after a reconnect and
        # an unmuxable one after a resolution change. That is worse than no clip at all,
        # because it looks like evidence until someone tries to play it. Finishing early
        # is not an option: the packet loop runs concurrently with this one and may
        # already have appended post-discontinuity packets, so there is no clean cut
        # point left to finish at. Abort it and let the escalation it belonged to
        # publish with no clip — spec §9 keeps the event either way.
        async with self._clip_lock:
            # `_abandon_open_clip` does not take the lock itself: its other caller is
            # `run()`'s `finally`, by which point the packet loop is already cancelled.
            await self._abandon_open_clip()

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
                handle = None
                try:
                    handle = await self._clip_writer.open(
                        self._camera_id, event_id, self._estimated_fps()
                    )
                    for packet in self._preroll.flush():
                        await handle.append(packet)
                except Exception:
                    # Opening or seeding a clip is I/O — MinIO's bucket check, a PyAV
                    # muxer, a temp file. Any of it can fail, and none of it may cost
                    # the event (spec §9) or the camera: this runs on the frame loop,
                    # so an escaping exception ends `run()` and the camera with it.
                    logger.warning(
                        "could not start a clip for camera %s event %s; publishing "
                        "the event without one",
                        self._camera_id,
                        event_id,
                        exc_info=True,
                    )
                    # The half-built clip is nobody's to finish. `open()` succeeding and
                    # a pre-roll `append()` then failing is the ordinary shape of this —
                    # it is the path B4's sub-tick remux fault takes when it lands during
                    # the flush — and without this the handle is simply dropped: with
                    # `MinioClipHandle` that is a live PyAV container plus a temp .mp4,
                    # and since `_active_clip` is never set, *every* subsequent
                    # escalation opens another one. Same discipline as the sibling
                    # failure path in `_packet_loop`, which has always aborted.
                    if handle is not None:
                        with contextlib.suppress(Exception):
                            await handle.abort()
                    self._submit(request_with(None))
                    return event_id
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

        A failure while writing *one* packet into a clip abandons that clip and nothing
        else. Letting it out of this coroutine kills the packet loop, and killing the
        packet loop silently kills the whole camera: `run()` is still awaiting the frame
        loop and never observes the dead task, while the source's packet queue — which
        the demux thread pushes into with a blocking put — fills and stops the demux,
        so frames stop too. The camera then reports a frozen `frames_seen` and no error
        anywhere. Observed live against a real RTSP stream through the composition root
        (`sentinel_ai/main.py`), after roughly 5 000 frames. The clip is best-effort
        evidence; the camera and the event are not.
        """
        async for packet in self._source.packets():
            self._preroll.append(packet)
            async with self._clip_lock:
                active = self._active_clip
                if active is not None:
                    try:
                        await active.handle.append(packet)
                        closed = packet.pts >= active.deadline
                    except Exception:
                        logger.warning(
                            "clip append failed for camera %s event %s; abandoning the clip "
                            "and keeping the camera running",
                            self._camera_id,
                            active.request.event_id,
                            exc_info=True,
                        )
                        self._active_clip = None
                        # Same rule as `_abandon_open_clip`: the file goes, the event
                        # stays (spec §9). `abort()` is contractually idempotent and
                        # never raises.
                        await active.handle.abort()
                        self._submit(replace(active.request, clip=None))
                    else:
                        if closed:
                            self._active_clip = None
                            self._submit(active.request)
            await asyncio.sleep(0)  # cooperative yield; see the module docstring

    async def _abandon_open_clip(self) -> None:
        """Discard a still-recording clip's partial file, keep its event.

        Two callers, one rule: the clip cannot be completed, so the file goes and the
        escalation is submitted anyway with `clip=None`. Shutdown reaches here because
        the post-roll never closed; a discontinuity reaches here because the recording
        would otherwise splice two timelines. Spec §9 forbids losing an anomaly event to
        an infrastructure or lifecycle failure, and both of those are one.

        `abort()` is contractually infallible, which is why it is safe to await here,
        possibly while already being cancelled. Caller-owned locking: `run()`'s `finally`
        has already cancelled the packet loop, while `_on_discontinuity` races it and
        holds `_clip_lock` across the call.
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
