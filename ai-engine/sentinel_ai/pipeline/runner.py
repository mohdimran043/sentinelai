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
import io
import logging
from collections import deque
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import get_settings
from sentinel_ai.domain.behaviour.candidate import BehaviourKind
from sentinel_ai.domain.behaviour.observation import BehaviourObservation, PersonPose
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.capabilities import (
    DEFAULT_CAPABILITIES,
    CameraCapabilities,
    Capability,
)
from sentinel_ai.domain.entities import BBox, Detection, EscalationReason, SceneState
from sentinel_ai.domain.identity import FaceEmbedding, FaceObservation
from sentinel_ai.domain.policy.authorization import (
    AuthorizationPolicy,
    AuthorizationTracker,
    observe_authorization,
)
from sentinel_ai.domain.policy.escalation import GateState, decide, force
from sentinel_ai.domain.welfare import ConcernKind, Confidence
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.pipeline.behaviour import BehaviourEngine
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.face import DetectedFace, FaceDetector, FaceStore
from sentinel_ai.ports.frame_source import FrameData, FrameSource
from sentinel_ai.ports.pose import PoseEstimator
from sentinel_ai.ports.tracker import Tracker

logger = logging.getLogger(__name__)

__all__ = ["CameraRunner", "CameraTelemetry"]

_DEFAULT_FPS = 10.0
"""Used only until two frames have been seen and a real interval can be measured."""

_HISTORY_MAXLEN = 5
"""How many previous escalation details the VLM gets as context (spec §22)."""

_BEHAVIOUR_REASONS: dict[BehaviourKind, EscalationReason] = {
    BehaviourKind.FALL: EscalationReason.FALL_SUSPECTED,
    BehaviourKind.ABANDONED_OBJECT: EscalationReason.ABANDONED_OBJECT,
    BehaviourKind.CAMERA_TAMPER: EscalationReason.CAMERA_TAMPER,
    BehaviourKind.ZONE_INTRUSION: EscalationReason.ZONE_INTRUSION,
    BehaviourKind.LINE_CROSSING: EscalationReason.LINE_CROSSING,
}
"""What a behaviour becomes on the wire.

A total mapping, indexed rather than `.get`-with-a-default: a kind with no reason would
otherwise be published under some plausible-looking fallback, and `reason` is the field
every consumer filters and prioritises on. A `KeyError` here is a build failure, which
is the right cost for adding a detector without deciding what it is called.
"""


@dataclass(frozen=True, slots=True)
class CameraTelemetry:
    camera_id: str
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    """Escalations this camera raised that the queue refused outright.

    Narrower than it looks, and the gap is worth knowing. Since the queue became
    priority-ordered (§16) a submission is refused only when it is *itself* the least
    urgent thing present. An escalation that was accepted and later evicted to make
    room for a more urgent one is **not** counted here — this camera cannot know that
    happened, because it happens after the submission returned. `VlmScheduler.dropped`
    is the process-wide count that does include it.

    So: this answers "how often was this camera turned away", and the scheduler's
    counter answers "how much work was discarded". On a saturated site the second is
    larger, and neither is wrong.
    """
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
    falls_suspected: int = 0
    """How many fall signatures have completed on this camera since the engine started.

    Defaulted — and therefore placed here, after the last required field — so every
    existing construction site keeps working. Counted on the runner rather than derived
    from the event ring because that ring is bounded and volatile: a camera that has
    produced more events than it holds would under-report, and a number an operator
    reads as "how often has this happened here" must not silently shrink.
    """

    enabled: bool = True
    """Whether this camera is being watched at all.

    `False` describes a camera that is configured and deliberately stopped: its record
    is still in `cameras.json`, it still has a label, a zone and a capability set, and
    no `CameraRunner` exists for it. That is a different thing from a camera that is
    running badly, and the distinction is the whole reason the field exists — an
    operator looking at a wall needs "nobody is watching this on purpose" to be
    unmistakable from "this has gone quiet".

    A disabled camera's counters are the ones it had when it was stopped, frozen. They
    are not zeroed, because zero would read as "saw nothing" when the truth is "saw
    this much, then was switched off"; see `EngineService.disable_camera`.
    """

    zone: Zone | None = None
    """Which zone this camera watches, or None when nobody has grouped it (T1).

    Static configuration riding on a counters record, deliberately: `GET /cameras` is
    built from these snapshots, so a console that groups cameras reads the grouping
    from the same object it reads the liveness from — one request, and no way for the
    two to describe different sets of cameras. Defaulted so that every existing
    construction site keeps working and an ungrouped camera stays representable.
    """

    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES
    """Which AI capabilities are running on this camera (spec §13).

    Reported for `zone`'s reason — `GET /cameras` is built from this record, so a
    console reads what a camera is actually doing from the same object it reads the
    camera's liveness from, and the two cannot describe different sets of cameras.

    This is what makes the console's capability checkboxes honest: they render the
    engine's own answer about what it is running, not the file's answer about what
    was asked for. The two agree today because `main.compose` derives one from the
    other, and this field is how that stays checkable from outside.
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


def _encode_jpeg(frame: FrameData, max_edge: int) -> bytes | None:
    """One frame as a JPEG, or `None` if it cannot be made.

    `None` rather than raising: a snapshot is a convenience on a page that has plenty
    else to show, and a camera whose page 500s because an encoder complained is worse
    than one whose picture is missing.
    """
    try:
        import numpy as np
        from PIL import Image

        pixels = frame.pixel_array()
        if not isinstance(pixels, np.ndarray):
            return None
        picture = Image.fromarray(pixels[:, :, ::-1].astype(np.uint8))  # BGR -> RGB
        picture.thumbnail((max_edge, max_edge))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
    except Exception:
        logger.debug("could not encode a snapshot", exc_info=True)
        return None


class CameraRunner:
    def __init__(
        self,
        *,
        camera_id: str,
        camera_label: str,
        source: FrameSource,
        detector: ObjectDetector | None,
        tracker: Tracker,
        motion: MotionAnalyzer,
        profile: CameraProfile,
        scheduler: VlmScheduler,
        clip_writer: ClipWriter | None,
        preroll: PreRollBuffer,
        detect_every_n_frames: int = 1,
        pose: PoseEstimator | None = None,
        behaviour: BehaviourEngine | None = None,
        face: FaceDetector | None = None,
        face_store: FaceStore | None = None,
        authorization: AuthorizationPolicy | None = None,
        describe_scenes: bool = True,
        capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
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
        # `None` when this camera enabled no capability that needs one (spec §13). The
        # frame loop then decodes, tracks liveness and buffers pre-roll, but runs no
        # inference at all — which is what "watch this camera, run nothing on it" has
        # to mean if it is to cost anything less than a camera that does. Skipping is
        # per camera rather than per process on purpose: the detector is shared (ADR 4),
        # so its mere existence for one camera must not oblige every other camera to
        # pay for a forward pass.
        self._detector = detector
        # Carried, never branched on by the gate: whether a described escalation is
        # wanted is the scheduler's business, and it rides out on every
        # `EscalationRequest` so the decision is snapshotted with the rest of the
        # request rather than re-read at describe time.
        self._describe_scenes = describe_scenes
        # Every behaviour machine this camera enabled, with its state. An empty engine
        # (`enabled` False) is the normal case and costs a boolean per frame.
        # Constructed by the composition root rather than here, because which detectors
        # a camera runs is a capability question and `domain/capabilities.py` is where
        # that is answered.
        self._behaviour = behaviour if behaviour is not None else BehaviourEngine()
        # The shared pose model, or `None` when no camera asked for one. Optional
        # independently of the engine: a camera can run fall detection with pose
        # unavailable and the state machine falls back to bounding-box geometry,
        # recording which it used (`domain/behaviour/fall.py`).
        self._pose = pose
        self._falls_suspected = 0
        self._behaviours_raised = 0
        # All three are `None` unless this camera enabled `person_authorization`. They
        # travel together: a policy with no face pipeline would be inert configuration,
        # and a pipeline with no roster to search would have nothing to compare against.
        # `main.compose` sets all three or none.
        self._face = face
        self._face_store = face_store
        self._authorization = authorization
        self._authorization_tracker = AuthorizationTracker()
        self._unauthorized_raised = 0
        # Carried for reporting only — `telemetry()` publishes it so a console can
        # render what is actually running on this camera rather than what it hopes is.
        self._capabilities = capabilities
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
        # The most recent decoded frame, for `snapshot_jpeg`. One per camera,
        # replaced each time, so it pins a single decode buffer and never grows.
        self._latest_frame: FrameData | None = None
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
        capabilities: CameraCapabilities,
        notify_on: frozenset[ConcernKind],
        notify_min_confidence: Confidence,
        clip_preroll_seconds: float | None,
        clip_postroll_seconds: float | None,
        summary_interval_seconds: float | None,
        detector: ObjectDetector | None,
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
        self._capabilities = capabilities
        # Both derived from `capabilities`, never stored alongside it, for the reason
        # `main.compose` derives them at startup: two fields that can disagree about
        # whether a camera describes scenes is a camera whose behaviour depends on
        # which one a reader happened to consult.
        #
        # Read points, in the same "when does the new value first apply" terms as the
        # durations below: `_describe_scenes` is read once per escalation in
        # `_escalate`, before its first await, so the escalation already in flight
        # keeps the setting it was decided under. `auto_escalation_enabled` is read by
        # the gate once per frame inside `decide()`, and swapping `_profile` between
        # frames is safe for `summary_interval_seconds`'s reason — no `GateState` is
        # derived from it, so turning triggers off simply stops the next frame firing
        # one and turning them on lets the next frame fire normally.
        self._describe_scenes = capabilities.enabled(Capability.SCENE_DESCRIPTION)
        # The caller resolves this: whether a detector *exists* is a property of the
        # process (which roles were built at startup), and this runner cannot know it.
        # `None` means this camera now runs no inference — `_process_frame` returns
        # after counting the frame, so liveness stays accurate while nothing is spent.
        self._detector = detector
        self._notify_on = notify_on
        self._notify_min_confidence = notify_min_confidence
        self._clip_preroll_override = clip_preroll_seconds
        self._clip_postroll_override = clip_postroll_seconds
        self._summary_interval_override = summary_interval_seconds
        self._clip_postroll_seconds = new_postroll
        self._profile = replace(
            new_profile,
            auto_escalation_enabled=capabilities.enabled(Capability.ANOMALY_DETECTION),
        )

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

    async def snapshot_jpeg(self, max_edge: int = 960) -> bytes | None:
        """The most recent frame as a JPEG, or `None` if none has arrived yet.

        Not video, and the console says so where it shows one. It exists because live
        video is mediamtx's job and only cameras *published to* mediamtx have a playlist
        — an EarthCam page and a file do not, and before this their camera page showed
        nothing at all while the engine was demonstrably decoding them.

        Encoded on demand and off the event loop. Encoding every frame against the
        chance somebody is looking would be a JPEG per frame per camera forever; this
        costs one encode per request instead, and a request only happens while a page is
        open.
        """
        frame = self._latest_frame
        if frame is None:
            return None
        return await asyncio.to_thread(_encode_jpeg, frame, max_edge)

    @property
    def camera_id(self) -> str:
        """The camera this runner watches. Read-only: the id is what every other index
        in the system keys on, and a runner that could be renamed underneath them would
        strand its telemetry, its alerts and its events under the old one."""
        return self._camera_id

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
            falls_suspected=self._falls_suspected,
            last_frame_at=self._last_frame_at,
            last_escalation_at=self._last_escalation_at,
            zone=self._zone,
            capabilities=self._capabilities,
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

        if self._detector is None and not self._behaviour.enabled:
            # Nothing on this camera reads a frame. Liveness still has to be accurate,
            # so the frame is counted and timestamped exactly as it would be otherwise —
            # a camera running nothing must still be visibly *up*, or an operator
            # cannot tell it apart from one that has stopped delivering.
            self._last_frame_index = frame.frame_index
            self._last_processed_timestamp = frame.timestamp
            self._latest_frame = frame
            return

        # `()` rather than a skipped stage when no detector was built for this camera.
        # `Capability.CAMERA_TAMPER` needs no model at all — it reads the luma histogram
        # the motion stage computes — so a camera can legitimately reach here with
        # nothing to detect *with* and still have something to watch *for*. The tracker
        # handed an empty tuple yields no tracks, which is the truth.
        # Held for `snapshot_jpeg`, which is how a camera with no mediamtx path — an
        # EarthCam page, a file — gets a picture on its page at all. One frame per
        # camera, replaced every time, so it pins one decode buffer and never grows.
        self._latest_frame = frame

        detections: tuple[Detection, ...] = ()
        if self._detector is not None:
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

        # Behaviour detectors run before the gate and, when one fires, **instead of**
        # it. A completed signature is already deduplicated to one report per episode by
        # its own state machine, so it bypasses the gate's three governors exactly as a
        # user request does — see `EscalationReason.FALL_SUSPECTED`. Letting a cooldown
        # window swallow it would lose the one escalation this subsystem exists to
        # produce, and running the gate *as well* would spend a second VLM call
        # describing the same frame.
        if await self._detect_behaviours(scene, frame):
            return

        # After the behaviour machines and before the gate, for the same reason they
        # come before it: a completed authorisation episode is already deduplicated to
        # one report per tracked person, so the gate's governors have nothing left to
        # protect against. It runs *after* them because a person on the floor matters
        # more than a person who is unrecognised, and only one escalation per frame is
        # possible — the keyframe is shared.
        if await self._detect_unauthorized(scene, frame):
            return

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

    async def _detect_behaviours(self, scene: SceneState, frame: FrameData) -> bool:
        """Advance every behaviour machine; escalate and return True if one completed.

        Returns a bool rather than escalating silently so `_process_frame` can see that
        this frame is already spoken for. The alternative — letting the gate run too —
        would describe the same keyframe twice, once as a behaviour and once as
        whatever trigger the same unusual scene also fired.
        """
        if not self._behaviour.enabled:
            return False

        poses: dict[int, PersonPose] = {}
        if self._pose is not None and self._behaviour.needs_pose:
            try:
                poses = await self._pose.estimate(frame, scene.tracks)
            except Exception as error:
                # Pose is an enhancement with a working fallback, so a failure here
                # degrades the reading rather than costing the frame. Raising would
                # take down a camera pipeline over an optional model; skipping the
                # whole check would silently disable fall detection on a camera whose
                # operator enabled it. Neither is acceptable, and the geometry path is
                # exactly what `fall.py` keeps for this case.
                logger.warning(
                    "pose estimation failed for camera %s; falling back to geometry: %s",
                    self._camera_id,
                    error,
                )

        candidates = self._behaviour.observe(
            BehaviourObservation(
                scene=scene,
                poses=poses,
                # Zones are stored as fractions of the frame so they survive an RTSP
                # resolution renegotiation; this is what turns them back into pixels.
                frame_width=frame.width,
                frame_height=frame.height,
            )
        )
        if not candidates:
            return False

        # At most one escalation per frame even when two machines finish together: the
        # keyframe is shared, so a second request would describe the same image.
        # `BehaviourEngine.observe` has already ordered them, so the first is the one
        # whose consequences are worst — see `_PRIORITY` there. The others stay in
        # their `REPORTED` phase and will not raise again, which is a real limitation
        # recorded here rather than hidden; the alternative is N escalations for one
        # frame, flooding the queue the gate exists to protect.
        winner = candidates[0]
        self._behaviours_raised += 1
        if winner.kind is BehaviourKind.FALL:
            self._falls_suspected += 1
        await self._escalate(
            scene=scene,
            keyframe=frame,
            reason=_BEHAVIOUR_REASONS[winner.kind],
            detail=winner.summary,
            now=frame.timestamp,
            subject_tracks=winner.track_ids,
        )
        return True

    async def _detect_unauthorized(self, scene: SceneState, frame: FrameData) -> bool:
        """Run the face pipeline and advance the authorisation machine (§8-§12).

        Returns True when a finding was escalated, so `_process_frame` knows this frame
        is already spoken for.

        Every failure here degrades rather than propagates. Face recognition is an
        optional capability layered on a pipeline that works without it, and taking a
        camera down because a face model faulted would trade the whole of surveillance
        for one feature. A failure is logged and the frame carries on to the gate.
        """
        if self._face is None or self._face_store is None or self._authorization is None:
            return False
        if not scene.tracks:
            # No people, so no faces worth looking for. The common case, and skipping
            # it is most of why this capability is affordable at all.
            return False

        try:
            detected = await self._face.detect(frame)
            if not detected:
                return False
            observations = await self._build_face_observations(detected, scene, frame)
        except Exception as error:
            logger.warning(
                "face pipeline failed for camera %s; authorisation is not checked on "
                "this frame: %s",
                self._camera_id,
                error,
            )
            return False

        if not observations:
            return False

        self._authorization_tracker, findings = observe_authorization(
            observations,
            self._authorization,
            self._authorization_tracker,
            now=frame.timestamp,
        )
        if not findings:
            return False

        finding = findings[0]
        self._unauthorized_raised += 1
        await self._escalate(
            scene=scene,
            keyframe=frame,
            reason=EscalationReason.UNAUTHORIZED_PERSON,
            detail=finding.summary(),
            now=frame.timestamp,
            subject_tracks=(finding.track_id,),
        )
        return True

    async def _build_face_observations(
        self,
        detected: tuple[DetectedFace, ...],
        scene: SceneState,
        frame: FrameData,
    ) -> tuple[FaceObservation, ...]:
        """Attach each detected face to the person track it sits inside, and search.

        A face is attributed to the tracked person whose box **contains its centre**,
        rather than by overlap: a face box is a small region entirely inside a person
        box, so IoU between them is tiny even when the attribution is obvious.
        Containment is the right test for a part-of relationship.

        A face inside no tracked person is dropped. It is usually a reflection, a
        photograph on a wall, or a person the object detector missed — and without a
        track there is nothing for §11's temporal confirmation to accumulate against,
        so it could never contribute to a finding anyway.
        """
        assert self._face_store is not None
        observations: list[FaceObservation] = []
        for face in detected:
            track_id = self._track_containing(face.box, scene)
            if track_id is None:
                continue
            if not isinstance(face.aligned, FaceEmbedding):
                # This detector did not produce an embedding alongside the detection.
                # A separate embedder would be wired here; until one is, saying so
                # beats guessing.
                continue
            candidates = await self._face_store.search(
                face.aligned,
                camera_id=self._camera_id,
                zone=self._zone.value if self._zone is not None else None,
                now=frame.timestamp,
            )
            observations.append(
                FaceObservation(
                    track_id=track_id,
                    embedding=face.aligned,
                    quality=face.quality,
                    timestamp=frame.timestamp,
                    candidates=candidates,
                )
            )
        return tuple(observations)

    @staticmethod
    def _track_containing(face_box: BBox, scene: SceneState) -> int | None:
        """The person track whose box contains this face's centre, if any."""
        centre_x, centre_y = face_box.cx, face_box.cy
        for track in scene.tracks:
            if track.label != "person":
                continue
            box = track.box
            if box.x1 <= centre_x <= box.x2 and box.y1 <= centre_y <= box.y2:
                return track.track_id
        return None

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
        # Every behaviour machine is reset for the tracker's reason, one layer up: each
        # phase they hold is keyed by a track id the tracker has just invalidated, and
        # every timestamp in them is on a timeline that no longer exists. Carried
        # across a reconnect, a track that was `UPRIGHT` before the break would measure
        # its first post-break frame against a centroid from the old stream — arbitrary
        # pixels over an arbitrary interval — which is exactly how a reconnect becomes
        # a reported fall that never happened.
        self._behaviour.reset()
        # The authorisation machine holds per-track evidence keyed by ids the tracker
        # has just invalidated. Carried across, a reconnect could complete somebody
        # else's episode — and an accusation assembled from two different people's
        # frames is the worst thing this feature could produce.
        self._authorization_tracker = AuthorizationTracker()
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
        subject_tracks: tuple[int, ...] = (),
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
        describe = self._describe_scenes
        # Empty unless a behaviour detector named a subject. `_escalate` is shared
        # between the gate's trigger path and the behaviour path, and only the second
        # knows whose behaviour it is reporting.
        subject_track_ids = subject_tracks

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
                # Snapshotted with the rest of the request, for the reason the welfare
                # policy above is: `apply_metadata` can land at any await between here
                # and the published event, and an escalation half on the old capability
                # set and half on the new one has no defensible meaning.
                describe=describe,
                subject_track_ids=subject_track_ids,
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
