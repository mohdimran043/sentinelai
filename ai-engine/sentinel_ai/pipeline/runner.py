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

Its two lengths, and the gate's forced-look interval, are per camera: `cameras.json`
may set `clip_preroll_seconds`, `clip_postroll_seconds` and `summary_interval_seconds`,
and an unset one falls back to the default it overrides: the horizon the composer sized
the pre-roll ring at, `Settings.clip_postroll_seconds`, and the camera profile's own
interval. All three are editable on a running camera through `apply_metadata`,
which documents field by field when each new value is first read. The short version:
on the next escalation. A clip already recording keeps the deadline it opened with.

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
from sentinel_ai.domain.welfare import ConcernKind, Confidence
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
    label: str
    """The camera's display name — `cameras.json`'s `label`, defaulting to the id.

    Here for the same reason `zone` is, and now for a second one: it is editable at
    runtime (`PATCH /cameras/{id}`), and a console that can change a label has to be
    able to read back the label the engine is actually using. Required rather than
    defaulted precisely because of that — a telemetry record whose label silently
    fell back to `""` would render as a nameless camera and look like a save that
    half-worked.
    """
    zone: Zone | None = None
    """Which zone this camera watches, or None when nobody has grouped it (T1).

    Static configuration riding on a counters record, deliberately: `GET /cameras` is
    built from these snapshots, so a console that groups cameras reads the grouping
    from the same object it reads the liveness from — one request, and no way for the
    two to describe different sets of cameras. Defaulted so that every existing
    construction site keeps working and an ungrouped camera stays representable.
    """

    notify_on: frozenset[ConcernKind] = frozenset(ConcernKind)
    """Which welfare concern kinds this camera notifies a human about.

    Here for `zone`'s reason exactly — `GET /cameras` is built from this record, and a
    console that can edit the policy has to be able to read it back — and carried, not
    read, by this module: nothing in the frame loop branches on it. Routing is a later
    task's, downstream of the published event.

    Empty means "notify nobody about this camera", and that is a real, storable
    instruction rather than an unconfigured field; see `CameraConfig.notify_on`.
    """

    notify_min_confidence: Confidence = Confidence.LIKELY
    """The lowest confidence tier that may notify. Carried for the same reason and read
    by the same nobody as `notify_on`."""

    clip_preroll_seconds: float | None = None
    """The camera's own pre-roll length, or None when it follows
    `Settings.clip_preroll_seconds`.

    **The stored override, not the resolved value.** None here means "this camera
    follows the engine-wide default", which is the same answer `CameraEditResponse`
    gives for the same field, so a console can compare what it wrote with what it
    reads back. Reporting the resolved 3.0 instead would make a console that
    re-submits the record it just read pin the camera to a number nobody chose, and
    silently opt it out of any future change to the default.
    """

    clip_postroll_seconds: float | None = None
    """The camera's own post-roll length, or None for `Settings.clip_postroll_seconds`.
    Stored-not-resolved, as `clip_preroll_seconds`."""

    summary_interval_seconds: float | None = None
    """The camera's own forced-look interval, or None for the profile's. Stored-not-
    resolved, as `clip_preroll_seconds` — and note the fallback here is the camera
    profile rather than a setting."""


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
        notify_on: frozenset[ConcernKind] = frozenset(ConcernKind),
        notify_min_confidence: Confidence = Confidence.LIKELY,
        clip_preroll_seconds: float | None = None,
        clip_postroll_seconds: float | None = None,
        summary_interval_seconds: float | None = None,
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
        # Carried for the same reason and read by the same nobody: welfare notification
        # routing happens downstream of the published event, not in this loop. Since
        # T10 they also ride out on every `EscalationRequest` (see `_escalate`) — still
        # carried, still never branched on here.
        self._notify_on = notify_on
        self._notify_min_confidence = notify_min_confidence
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._motion = motion
        # The file's profile, kept apart from `self._profile` (set below) because
        # `summary_interval_seconds` overrides one field of it and reverting that
        # override has to restore the profile's own value rather than a remembered
        # copy of it. Everything else about the profile is restart-only.
        self._base_profile = profile
        self._scheduler = scheduler
        self._clip_writer = clip_writer
        self._preroll = preroll
        # This camera's pre-roll default, captured *before* the override below narrows
        # the ring: the horizon the ring arrived with is, by construction, the value
        # whoever composed this runner chose for it (`main.compose` sizes it from its
        # own `settings` argument). Reverting the override to `None` has to return the
        # ring to that number and no other. Deriving it from `get_settings()` at edit
        # time instead would answer with the process-wide default, which is the same
        # number only when the composer happened to be handed `get_settings()`'s own
        # object — true of `create_default_app`, false of any other composition, and
        # the divergence is silent because both answers are plausible floats.
        self._default_preroll_seconds = preroll.preroll_seconds
        self._detect_every_n_frames = detect_every_n_frames

        # The three duration overrides, exactly as `cameras.json` stores them: `None`
        # is "follow the default", and `telemetry()` reports them unresolved so the
        # console reads back the record rather than the number in force. The resolved
        # values live where they are used — on the pre-roll ring, in
        # `_clip_postroll_seconds` and in `_profile` — and only this block and
        # `apply_metadata` write either half, so the two cannot drift.
        self._clip_preroll_override = clip_preroll_seconds
        self._clip_postroll_override = clip_postroll_seconds
        self._summary_interval_override = summary_interval_seconds
        # An absent override leaves the ring exactly as given rather than re-asserting
        # the size it already has.
        if clip_preroll_seconds is not None:
            preroll.preroll_seconds = clip_preroll_seconds
        self._clip_postroll_seconds = self._postroll_for(clip_postroll_seconds)
        self._profile = self._profile_for(summary_interval_seconds)

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

    def apply_metadata(
        self,
        *,
        label: str,
        zone: Zone | None,
        notify_on: frozenset[ConcernKind],
        notify_min_confidence: Confidence,
        clip_preroll_seconds: float | None,
        clip_postroll_seconds: float | None,
        summary_interval_seconds: float | None,
    ) -> None:
        """Apply an already-persisted edit of this camera's record to the running
        camera. **The new values take effect on the next escalation, never
        retroactively.**

        Whole-record rather than per-field on purpose: the caller has just read the
        persisted record back, and passing all of it keeps "apply what the file now
        says" a single statement instead of seven conditionals that could apply some
        and skip others. `url` and `profile` have no equivalent here and must not
        grow one — see `adapters/config/camera_file.py`.

        Safety, field by field. Every read below is a synchronous attribute read of a
        value this method replaces wholesale, and this method is itself synchronous
        and runs on the event loop (`main.ComposedService.update_camera` awaits the
        store and then calls straight through), so no read can observe a half-applied
        edit. What differs between the fields is *when* the new value is first read:

        * `label` and `zone` are not policy at all. The loop never branches on
          either; they are read by `telemetry()` and by `_escalate`'s request
          assembly, and either read simply returns whichever value is current.
        * `notify_on` and `notify_min_confidence` are policy, but not *this*
          component's: nothing here branches on them. They are carried to
          `telemetry()` so the console can read back what it wrote, and — since
          T10 — onto the `EscalationRequest`, still unread by this loop, for
          `VlmScheduler` to route on downstream of the published event. Both are
          read once per escalation, in `_escalate`, before its first await, and
          the escalation that is already in flight keeps the policy it was decided
          under: a note routed half on the old `notify_on` and half on the new one
          is the same indefensible split the post-roll snapshot exists to avoid,
          except that here one of the halves is an operator's decision to mute a
          camera. The edit governs the next escalation, per this method's opening
          promise.
        * `summary_interval_seconds` is read by the gate, once per frame, inside
          `decide()`. Swapping `_profile` between frames is safe because no gate
          state is derived from the interval: `GateState` remembers *when* the last
          summary happened, and `periodic_summary` compares that against whatever
          interval is current. So a shortened interval can make the next frame due
          for a summary and a lengthened one can make an already-due camera wait —
          which is what an operator asking for a different interval means — and
          neither invalidates a decision already taken.
        * `clip_postroll_seconds` is read once per escalation, in `_escalate`,
          before its first await, and written into `_ActiveClip.deadline`. **An
          in-flight clip keeps the deadline it opened with**: nothing recomputes it,
          because a deadline moved backwards past packets already written lands in
          the clip's own past, and one moved forwards past the end of the stream
          leaves the recording to be aborted instead of finished. A second
          escalation while that clip is still recording extends the deadline, and
          *that* extension uses the new value — it is the next escalation.
        * `clip_preroll_seconds` is the one duration read *after* an await: the ring's
          horizon is read inside `flush()`, which `_escalate` calls after opening the
          clip. An edit landing in that window gives the clip a lead-in somewhere
          between the old and the new length, and cannot do worse than that —
          `flush()` only ever returns already-buffered packets, all of them older
          than the escalation, so every reachable outcome is a valid clip. A
          narrowing takes effect on the next `flush()` or `append`, whichever comes
          first: `PreRollBuffer._anchor_index` recomputes the horizon from the
          current value on every call, so the narrower one is honoured by the very
          next read and not only by the next eviction. A widening takes effect as the
          ring refills, since packets already evicted are gone (see
          `PreRollBuffer.preroll_seconds`). An override reverted to `None` restores
          the horizon this runner was *composed* with — `_default_preroll_seconds`,
          captured in `__init__` — not whatever `get_settings()` says now.

        Applied all-or-nothing. Two of the values are fallible — a negative pre-roll
        is refused by `PreRollBuffer`, a non-positive interval by
        `CameraProfile.__post_init__` — so both are resolved, and the ring (the only
        object here that is not ours) is written, before a single attribute of this
        runner moves. Neither raise is reachable through `PATCH /cameras/{id}`, whose
        request model and file loader both enforce the same bounds; the ordering is
        what makes "totally applied or not applied at all" a property of this method
        rather than of who happens to call it. `main.ComposedService.update_camera`
        has already persisted the file by the time this runs and has nothing to
        unwind with, and it should not need one.
        """
        new_profile = self._profile_for(summary_interval_seconds)
        new_postroll = self._postroll_for(clip_postroll_seconds)
        new_horizon = (
            self._default_preroll_seconds if clip_preroll_seconds is None else clip_preroll_seconds
        )
        # First and only write that can raise; the setter validates before it assigns,
        # so a refused horizon leaves the ring on its old one and this runner untouched.
        self._preroll.preroll_seconds = new_horizon

        self._camera_label = label
        self._zone = zone
        self._notify_on = notify_on
        self._notify_min_confidence = notify_min_confidence
        self._clip_preroll_override = clip_preroll_seconds
        self._clip_postroll_override = clip_postroll_seconds
        self._summary_interval_override = summary_interval_seconds
        self._clip_postroll_seconds = new_postroll
        self._profile = new_profile

    def _postroll_for(self, override: float | None) -> float:
        """The given per-camera post-roll if set, else the process-wide default.

        `Settings` rather than a constructor slot for the default: it is a
        process-wide tuning value (S1), and reading it here is what lets an edit that
        reverts the override to `None` restore it without the runner having to
        remember what it was built with. The pre-roll cannot do the same — its
        default reaches the runner through the ring the composer sized, not through
        the settings object — which is why `_default_preroll_seconds` exists and this
        does not need an equivalent.

        Takes the override as an argument rather than reading
        `self._clip_postroll_override`, so `apply_metadata` can resolve a new value
        before it commits the field that value came from.
        """
        if override is not None:
            return override
        return get_settings().clip_postroll_seconds

    def _profile_for(self, override: float | None) -> CameraProfile:
        """`_base_profile`, with `summary_interval_seconds` replaced when the camera
        overrides it.

        A copy rather than a mutation because `CameraProfile` is frozen, and only
        this one field because it is the only one `cameras.json` lets an operator
        change without a restart — the gate is part-way through applying every other
        one. `dataclasses.replace` re-runs `__post_init__`, so this raises on an
        interval of zero or below; both configuration edges already refuse it (`> 0`
        in `load_cameras` and in `CameraEditRequest`), so no caller can reach that,
        and `apply_metadata` calls this before it writes anything so that the
        unreachable case would still be harmless.

        Takes the override as an argument for the reason `_postroll_for` gives.
        """
        if override is None:
            return self._base_profile
        return replace(self._base_profile, summary_interval_seconds=override)

    def telemetry(self) -> CameraTelemetry:
        return CameraTelemetry(
            camera_id=self._camera_id,
            label=self._camera_label,
            frames_seen=self._frames_seen,
            frames_dropped=self._slot.dropped if self._slot is not None else 0,
            detections_run=self._detections_run,
            escalations=self._escalations,
            escalations_dropped=self._escalations_dropped,
            discontinuities=self._discontinuities,
            last_frame_at=self._last_frame_at,
            last_escalation_at=self._last_escalation_at,
            zone=self._zone,
            notify_on=self._notify_on,
            notify_min_confidence=self._notify_min_confidence,
            # The stored overrides, not the resolved values — see the field docstrings
            # on `CameraTelemetry`.
            clip_preroll_seconds=self._clip_preroll_override,
            clip_postroll_seconds=self._clip_postroll_override,
            summary_interval_seconds=self._summary_interval_override,
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
        # Read here, before the first await, so this escalation runs on the policy that
        # was in force when it was decided: `decide()` has just read `_profile` and
        # returns to this method without suspending, and the deadline below has to be
        # the post-roll the operator had configured at that instant. `apply_metadata`
        # can land during the clip-open await otherwise, and an escalation half on the
        # old policy and half on the new one is the one outcome with no defensible
        # meaning. See `apply_metadata` for the field-by-field argument.
        profile = self._profile
        postroll_seconds = self._clip_postroll_seconds
        # Snapshotted here for the same reason, and it matters more for these two
        # than for the durations above: `notify_on` is the operator's mute switch,
        # and an escalation that read it after an `apply_metadata` landed mid-clip
        # would notify under a policy that was never in force when the scene was
        # judged. Still not a branch — nothing in this loop reads either value; they
        # ride to `VlmScheduler`, which routes downstream of the published event.
        zone = self._zone
        notify_on = self._notify_on
        notify_min_confidence = self._notify_min_confidence

        def request_with(clip: ClipHandle | None) -> EscalationRequest:
            return EscalationRequest(
                camera_id=self._camera_id,
                event_id=event_id,
                reason=reason,
                detail=detail,
                scene=scene,
                keyframe=keyframe,
                profile=profile,
                camera_label=self._camera_label,
                history=history,
                clip=clip,
                zone=zone,
                notify_on=notify_on,
                notify_min_confidence=notify_min_confidence,
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
                    deadline=now + postroll_seconds,
                    request=request_with(handle),
                )
            else:
                # A clip is already recording: extend its post-roll rather than open a
                # second, overlapping one (spec §5.5). This escalation still gets its
                # own event — just no clip of its own, since a `ClipHandle` may only
                # ever be finished once, and only after its window has actually closed.
                # The extension uses *this* escalation's post-roll, which is the point
                # of "the next escalation": an edit that landed after the clip opened
                # lengthens or shortens the window from here, without ever moving the
                # deadline the clip already had out from under the packets in it.
                self._active_clip = replace(self._active_clip, deadline=now + postroll_seconds)
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
