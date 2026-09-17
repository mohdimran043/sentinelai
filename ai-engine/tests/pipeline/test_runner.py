from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import replace
from types import TracebackType
from uuid import UUID

import pytest

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import Settings
from sentinel_ai.domain.behaviour.fall import FallPolicy
from sentinel_ai.domain.behaviour.observation import PersonPose
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.capabilities import DEFAULT_CAPABILITIES, CameraCapabilities
from sentinel_ai.domain.entities import BBox, Detection, EscalationReason, Track
from sentinel_ai.domain.welfare import ConcernKind, Confidence
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.notifications import NotificationDispatcher
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.pipeline import runner as runner_module
from sentinel_ai.pipeline.behaviour import BehaviourEngine
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData
from sentinel_ai.ports.pose import PoseEstimator
from tests.fakes.io import (
    FakeClipHandle,
    FakeClipWriter,
    FakeFailedEventSink,
    FakeNotifier,
    FakePublisher,
    FakeSource,
)
from tests.fakes.models import FakeDetector, FakeModelRuntime, FakeTracker, FakeVisionLLM

NEAR = Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def patch_postroll(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """`CameraRunner` reads `clip_postroll_seconds` from settings (S11's fixed
    constructor has no such parameter); tests pin it small and explicit.

    It must be > 0: `Settings.clip_postroll_seconds` is declared `Field(gt=0)`.
    """
    monkeypatch.setattr(
        runner_module, "get_settings", lambda: Settings(clip_postroll_seconds=seconds)
    )


def alternating_source(
    camera_id: str, count: int, fps: float = 10.0, origin: float = 0.0
) -> FakeSource:
    """Frames whose luma alternates between two values.

    Consecutive escalations need distinguishable scenes: identical frames produce
    identical signatures, and the gate's duplicate-scene governor would suppress
    everything after the first escalation.

    `origin` shifts the whole timeline. `RtspSource` stamps `FrameData.timestamp` from
    `time.monotonic()`, so a real camera's first frame is at ~10**5, never 0.0 — and a
    source timeline that starts at zero is a property of `FileSource` alone.
    """
    return FakeSource(
        [
            FakeSource.make_frame(
                camera_id, index, origin + index / fps, value=0 if index % 2 == 0 else 200
            )
            for index in range(count)
        ]
    )


class SignallingSource(FakeSource):
    """A `FakeSource` that announces when its frame stream is exhausted.

    Lets a test wait for a precise pipeline state instead of guessing at a number of
    event-loop ticks — and without ever waiting on real wall-clock time.
    """

    def __init__(self, frames: list[FrameData]) -> None:
        super().__init__(frames)
        self.exhausted = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[FrameData]:
        inner = super().__aiter__()

        async def frames() -> AsyncIterator[FrameData]:
            async for frame in inner:
                yield frame
            self.exhausted.set()

        return frames()


class GatedDetector(FakeDetector):
    """Blocks on its very first call until released — a stand-in for a detector that
    cannot keep up with the decode rate."""

    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__(script=[()])
        self._gate = gate

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        if self.call_count == 0:
            await self._gate.wait()
        return await super().detect(frame)


class ResetSpyTracker(FakeTracker):
    def __init__(self) -> None:
        super().__init__()
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        super().reset()


class ResetSpyMotion(MotionAnalyzer):
    def __init__(self) -> None:
        super().__init__()
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        super().reset()


class VariableBinMotion(MotionAnalyzer):
    """Emits signatures of differing length on successive frames.

    The real `MotionAnalyzer` never changes bin count for a fixed instance, but the
    Phase 1A review's B1 crash was exactly a length mismatch (a resolution
    renegotiation mid-stream). This double proves the runner's own defence
    independently of whether the real analyzer can produce it today.
    """

    def __init__(self) -> None:
        super().__init__()
        self.resets = 0
        self._script: Iterator[tuple[float, tuple[float, ...]]] = iter(
            [(0.1, (0.5, 0.5)), (0.2, (0.3, 0.3, 0.4))]
        )

    def analyze(self, frame: FrameData) -> MotionSignals:
        energy, signature = next(self._script)
        return MotionSignals(motion_energy=energy, scene_signature=signature)

    def reset(self) -> None:
        self.resets += 1
        super().reset()


class ClearSpyPreRoll(PreRollBuffer):
    def __init__(self, preroll_seconds: float = 3.0) -> None:
        super().__init__(preroll_seconds)
        self.clears = 0

    def clear(self) -> None:
        self.clears += 1
        super().clear()


class EditingClipHandle(FakeClipHandle):
    def __init__(self, camera_id: str, event_id: UUID, *, writer: EditingClipWriter) -> None:
        super().__init__(camera_id, event_id)
        self._writer = writer

    async def append(self, packet: EncodedPacket) -> None:
        await super().append(packet)
        after = self._writer.after_packets
        if after is not None and not self._writer.fired and len(self.packets) >= after:
            self._writer.fired = True
            self._writer.on_packet()


class EditingClipWriter(FakeClipWriter):
    """Fires a callback at a chosen point inside one clip's lifecycle.

    Two hooks, for the two moments a live edit can land inside a single escalation
    and be mistaken for the next one:

      * `on_open` runs inside `ClipWriter.open` — the await `_escalate` suspends on
        after the gate has decided and before the clip's deadline is computed;
      * `on_packet` runs from the Nth `append` — the clip is recording and its
        deadline is already set.
    """

    def __init__(self, *, after_packets: int | None = None) -> None:
        super().__init__()
        self.after_packets = after_packets
        self.fired = False
        self.on_open: Callable[[], None] = lambda: None
        self.on_packet: Callable[[], None] = lambda: None

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> FakeClipHandle:
        self.opened.append((camera_id, event_id, fps))
        handle = EditingClipHandle(camera_id, event_id, writer=self)
        self.handles.append(handle)
        self.on_open()
        return handle


class Worker:
    """Runs the single scheduler worker for the duration of an `async with` block."""

    def __init__(self, scheduler: VlmScheduler) -> None:
        self._scheduler = scheduler
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> VlmScheduler:
        self._task = asyncio.create_task(self._scheduler.run())
        return self._scheduler

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


VLM_KEY = "qwen25vl3b"


def new_resident_set() -> ResidentSet:
    """A real `ResidentSet` over a real registry, holding a fake VLM runtime.

    The scheduler makes the VLM resident before every describe, so every scheduler
    needs one. Real rather than stubbed, for the same reason these tests run the real
    `MotionAnalyzer`: a stub would let the wiring be wrong without anything noticing.
    """
    registry = ModelRegistry()
    registry.register(
        ModelSpec(model_key=VLM_KEY, vram_mib=4400, priority=50, idle_unload_seconds=600.0),
        FakeModelRuntime(VLM_KEY, vram_mib=4400),
    )
    return ResidentSet(registry, total_mib=8192, reserved_mib=2048)


def new_scheduler(
    publisher: FakePublisher,
    vlm: FakeVisionLLM | None = None,
    maxsize: int = 4,
) -> VlmScheduler:
    return VlmScheduler(
        vlm=vlm or FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        resident_set=new_resident_set(),
        vlm_model_key=VLM_KEY,
        dead_letter=FakeFailedEventSink(),
        notifications=NotificationDispatcher(FakeNotifier()),
        maxsize=maxsize,
        timeout_seconds=5.0,
        clock=lambda: 0.0,
    )


class RecordingScheduler(VlmScheduler):
    """The real scheduler, recording every request the runner hands it.

    A subclass rather than a stub: these tests are about what the *runner* puts on
    the request, and everything downstream of `submit` — the describe, the publish,
    the drain the test awaits — has to keep working for the recording to be taken
    at a realistic moment.
    """

    def __init__(self, publisher: FakePublisher) -> None:
        super().__init__(
            vlm=FakeVisionLLM(),
            publisher=publisher,
            admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
            resident_set=new_resident_set(),
            vlm_model_key=VLM_KEY,
            dead_letter=FakeFailedEventSink(),
            notifications=NotificationDispatcher(FakeNotifier()),
            maxsize=4,
            timeout_seconds=5.0,
            clock=lambda: 0.0,
        )
        self.submitted: list[EscalationRequest] = []

    def submit(self, request: EscalationRequest) -> bool:
        self.submitted.append(request)
        return super().submit(request)


def make_runner(
    *,
    source: FakeSource,
    detector: FakeDetector,
    scheduler: VlmScheduler,
    clip_writer: FakeClipWriter | None = None,
    profile: CameraProfile | None = None,
    tracker: FakeTracker | None = None,
    motion: MotionAnalyzer | None = None,
    preroll: PreRollBuffer | None = None,
    detect_every_n_frames: int = 1,
    zone: Zone | None = None,
    fall_policy: FallPolicy | None = None,
    behaviour: BehaviourEngine | None = None,
    pose: PoseEstimator | None = None,
    notify_on: frozenset[ConcernKind] = frozenset(ConcernKind),
    notify_min_confidence: Confidence = Confidence.LIKELY,
    clip_preroll_seconds: float | None = None,
    clip_postroll_seconds: float | None = None,
    summary_interval_seconds: float | None = None,
) -> CameraRunner:
    return CameraRunner(
        camera_id="cam-1",
        camera_label="Front Door",
        source=source,
        detector=detector,
        tracker=tracker or FakeTracker(),
        motion=motion or MotionAnalyzer(),
        profile=profile or CameraProfile(camera_id="cam-1"),
        scheduler=scheduler,
        clip_writer=clip_writer,
        preroll=preroll or PreRollBuffer(preroll_seconds=3.0),
        detect_every_n_frames=detect_every_n_frames,
        zone=zone,
        # `fall_policy` is kept as a shorthand on this helper because most of these
        # tests only care about falls; anything richer passes a whole engine.
        behaviour=(behaviour if behaviour is not None else BehaviourEngine(fall=fall_policy)),
        pose=pose,
        notify_on=notify_on,
        notify_min_confidence=notify_min_confidence,
        clip_preroll_seconds=clip_preroll_seconds,
        clip_postroll_seconds=clip_postroll_seconds,
        summary_interval_seconds=summary_interval_seconds,
    )


def apply_policy(
    runner: CameraRunner,
    *,
    label: str = "Front Door",
    zone: Zone | None = None,
    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
    notify_on: frozenset[ConcernKind] = frozenset(ConcernKind),
    notify_min_confidence: Confidence = Confidence.LIKELY,
    clip_preroll_seconds: float | None = None,
    clip_postroll_seconds: float | None = None,
    summary_interval_seconds: float | None = None,
) -> None:
    """`apply_metadata` takes the camera's whole editable record, deliberately (see
    its docstring). These tests name only the field under test and let the rest
    default to what a `cameras.json` entry that mentions none of them loads as."""
    runner.apply_metadata(
        label=label,
        zone=zone,
        capabilities=capabilities,
        # Whatever this runner already has. In production `EngineService` resolves
        # this from the capability set, but a test revising a label must not
        # accidentally be a test that also detaches the detector.
        detector=runner._detector,
        notify_on=notify_on,
        notify_min_confidence=notify_min_confidence,
        clip_preroll_seconds=clip_preroll_seconds,
        clip_postroll_seconds=clip_postroll_seconds,
        summary_interval_seconds=summary_interval_seconds,
    )


def history_packet(pts: float) -> EncodedPacket:
    """A keyframe from before the stream under test started, as the packet loop would
    have buffered it during the seconds before an anomaly."""
    return EncodedPacket(
        camera_id="cam-1", data=b"history", pts=pts, is_keyframe=True, codec="h264"
    )


def two_gops() -> tuple[EncodedPacket, ...]:
    """Two complete GOPs of buffered history, keyframes at -0.4 and -0.2.

    A single GOP cannot tell two horizons apart: with one keyframe held, every
    horizon anchors on it — the wide one because nothing older exists to fall back
    from, the narrow one because that keyframe is also the newest. Two are the
    fewest that make `flush()` answer differently for a wide and a narrow ring.
    """
    return tuple(
        EncodedPacket(
            camera_id="cam-1",
            data=b"history",
            pts=pts,
            is_keyframe=is_keyframe,
            codec="h264",
        )
        for pts, is_keyframe in ((-0.4, True), (-0.3, False), (-0.2, True), (-0.1, False))
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeystoneTrace:
    async def test_source_to_gate_to_scheduler_to_publisher_on_cpu_with_no_sleeping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The phase's keystone test: proves the whole slice composes without a GPU,
        a broker, or a single second of wall-clock waiting."""
        patch_postroll(monkeypatch, seconds=1.0)
        source = FakeSource.constant("cam-1", count=3, fps=10.0)
        detector = FakeDetector(script=[()])  # empty detections: PERIODIC_SUMMARY still fires
        publisher = FakePublisher()
        vlm = FakeVisionLLM()
        scheduler = new_scheduler(publisher, vlm)
        runner = make_runner(source=source, detector=detector, scheduler=scheduler)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        telemetry = runner.telemetry()
        assert telemetry.camera_id == "cam-1"
        assert telemetry.frames_seen == 3
        assert telemetry.detections_run == 3, "a keeping-up consumer sees every frame"
        assert telemetry.frames_dropped == 0
        assert telemetry.escalations == 1  # PERIODIC_SUMMARY fires on the first scene
        assert telemetry.escalations_dropped == 0
        assert telemetry.discontinuities == 0
        assert telemetry.last_frame_at == pytest.approx(0.2)
        assert telemetry.last_escalation_at == pytest.approx(0.0)

        assert vlm.call_count == 1
        assert len(publisher.events) == 1
        assert publisher.events[0].reason.value == "periodic_summary"
        assert publisher.events[0].camera_id == "cam-1"
        assert publisher.events[0].description == "Nothing notable in view."
        assert source.closed is True, "the runner owns the source's lifetime"

    async def test_sampling_skips_frames_without_counting_them_as_drops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`detect_every_n_frames` is a deliberate sample, not backpressure."""
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=4, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            detect_every_n_frames=2,
        )

        async with Worker(scheduler):
            await runner.run()

        telemetry = runner.telemetry()
        assert telemetry.frames_seen == 4
        assert telemetry.detections_run == 2
        assert telemetry.frames_dropped == 0


class TestBackpressure:
    async def test_a_slow_detector_drops_stale_frames_and_processes_the_newest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Surveillance wants current reality, not a delayed complete record.

        Fails against a naive `async for frame in source: await detector.detect(frame)`
        implementation, because that one never drops anything — it only ever asks for
        the next frame once the previous is fully processed, so `frames_dropped` stays
        at 0 and `detections_run` equals `frames_seen`.
        """
        patch_postroll(monkeypatch, seconds=5.0)
        source = SignallingSource(
            [FakeSource.make_frame("cam-1", index, index / 100.0) for index in range(50)]
        )
        gate = asyncio.Event()
        detector = GatedDetector(gate)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(source=source, detector=detector, scheduler=scheduler)

        run_task = asyncio.create_task(runner.run())
        # The producer must reach the end of the stream while the detector is still
        # stuck on frame 0. A sequential implementation can never get here — it only
        # asks for the next frame once the current one is fully processed — so it fails
        # right here, bounded, instead of hanging the suite. The timeout never elapses
        # against a correct implementation: exhaustion happens within a few ticks.
        try:
            await asyncio.wait_for(source.exhausted.wait(), timeout=5.0)
        except TimeoutError:
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
            pytest.fail("the producer never outran the gated detector: nothing was dropped")
        gate.set()
        await run_task

        telemetry = runner.telemetry()
        assert telemetry.frames_seen == 50
        assert telemetry.detections_run == 2, (
            "only frame 0 (held in the detector) and the newest survivor are processed"
        )
        assert telemetry.frames_dropped == 48
        assert telemetry.frames_dropped + telemetry.detections_run == telemetry.frames_seen

    async def test_a_full_scheduler_queue_counts_a_drop_instead_of_blocking_the_camera(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pipeline must never wait on the VLM. With no worker draining it, a
        one-slot queue fills immediately and something has to go — the runner keeps
        going either way.

        **Which** one goes is the queue's business and changed when it became
        priority-ordered (§16): the first escalation here is a `periodic_summary` and
        the second a `user_requested`, so the more urgent one now displaces the routine
        one rather than being refused behind it. The camera's own
        `escalations_dropped` therefore stays 0 — this camera's submission *was*
        accepted — while the scheduler's global `dropped` records that a queued
        escalation was evicted to make room. See `CameraTelemetry.escalations_dropped`
        for why those two counters answer different questions.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher, maxsize=1)
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=1, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
        )

        await runner.run()
        await runner.describe_now()

        telemetry = runner.telemetry()
        assert telemetry.escalations == 2, "both were accepted by the queue"
        assert telemetry.escalations_dropped == 0
        assert scheduler.dropped == 1, "one queued escalation was evicted to make room"


class TestDiscontinuity:
    async def test_a_signature_length_change_resets_tracker_motion_preroll_and_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=5.0)
        tracker = ResetSpyTracker()
        motion = VariableBinMotion()
        preroll = ClearSpyPreRoll()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=2, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=1),
            tracker=tracker,
            motion=motion,
            preroll=preroll,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().discontinuities == 1
        assert tracker.resets == 1
        assert motion.resets == 1
        assert preroll.clears == 1
        # A gate left wedged in state built from an incomparable signature would still
        # be inside its 5s post-call cooldown on the second frame and would suppress it;
        # a reset one escalates cleanly, and `last_escalation_at` becomes that second
        # frame's own timestamp.
        assert runner.telemetry().last_escalation_at == pytest.approx(0.1)
        assert runner.telemetry().escalations == 2

    async def test_a_frame_index_regression_resets_derived_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What an RTSP reconnect (Task 14) looks like from here: indices and
        timestamps restart at zero mid-stream."""
        patch_postroll(monkeypatch, seconds=5.0)
        frames = [
            FakeSource.make_frame("cam-1", 0, 0.0),
            FakeSource.make_frame("cam-1", 1, 0.1),
            FakeSource.make_frame("cam-1", 0, 0.0),  # reconnect
        ]
        tracker = ResetSpyTracker()
        motion = ResetSpyMotion()
        preroll = ClearSpyPreRoll()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource(frames),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=1),
            tracker=tracker,
            motion=motion,
            preroll=preroll,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().discontinuities == 1
        assert tracker.resets == 1
        assert motion.resets == 1
        assert preroll.clears == 1
        # Without the reset the post-call cooldown, anchored at the pre-reconnect
        # timeline, swallows the first frame after the reconnect entirely.
        assert runner.telemetry().escalations == 2
        assert len(publisher.events) == 2

    async def test_the_scene_on_the_reconnect_frame_itself_is_built_from_reset_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate is reset on a discontinuity and then immediately asked to judge a
        scene — so that scene must not be the one built from the state just declared
        invalid.

        Fails against a runner that calls `tracker.update()` and `motion.analyze()`
        before testing for a discontinuity: the post-reconnect `SceneState` then carries
        a track matched against a pre-reconnect one (`age_frames == 3`, an id that should
        have ended) and a motion energy measured as a delta against the pre-reconnect
        frame (~0.78 here). `min_track_frames` and the speed trigger both read those
        fields, so a stale scene can fire a spurious escalation.
        """
        patch_postroll(monkeypatch, seconds=5.0)
        frames = [
            FakeSource.make_frame("cam-1", 0, 0.0, value=0),
            FakeSource.make_frame("cam-1", 1, 0.1, value=200),
            FakeSource.make_frame("cam-1", 2, 0.2, value=0),
            FakeSource.make_frame("cam-1", 0, 0.0, value=200),  # reconnect
        ]
        publisher = FakePublisher()
        vlm = FakeVisionLLM()
        scheduler = new_scheduler(publisher, vlm)
        runner = make_runner(
            source=FakeSource(frames),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=1, cooldown_seconds=0.05),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().discontinuities == 1
        scene = vlm.requests[-1].scene
        assert scene.frame_index == 0 and scene.timestamp == pytest.approx(0.0), (
            "the last escalation must be the reconnect frame's own"
        )
        assert scene.motion_energy == pytest.approx(0.0), (
            "energy against a frame from the old timeline is not motion"
        )
        assert [track.age_frames for track in scene.tracks] == [1], (
            "tracks must not carry ages accumulated before the reconnect"
        )

    async def test_a_reconnect_aborts_the_open_clip_rather_than_splicing_two_timelines(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clip recording when the stream restarts cannot be salvaged.

        The pre-roll is already cleared on a discontinuity, and the reason given is that
        old-timeline packets "cannot even be remuxed into the same clip". An open clip is
        the same problem one step later: keeping it merely re-anchored appends
        new-timeline packets straight onto old-timeline ones, so the file's pts sequence
        runs 0.0 .. 0.4 and then restarts at 0.0. It is worse than no clip, because it
        looks like evidence until someone tries to play it.

        Fails against a runner that re-anchors the open clip's deadline instead of
        aborting it: handle 0 then reaches its new deadline in the new timeline and is
        `finish()`ed, so it carries a `clip_uri` and its packets are not pts-ordered.
        """
        patch_postroll(monkeypatch, seconds=0.5)
        frames = [
            FakeSource.make_frame("cam-1", index, index / 10.0, value=0 if index % 2 == 0 else 200)
            for index in range(5)
        ] + [
            FakeSource.make_frame("cam-1", index, index / 10.0, value=0 if index % 2 == 0 else 200)
            for index in range(10)  # reconnect: indices and timestamps restart at zero
        ]
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource(frames),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=1, cooldown_seconds=0.2),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().discontinuities == 1
        assert writer.handles, "the first escalation must have opened a clip"
        first = writer.handles[0]
        assert first.aborted is True, "the clip open across the reconnect must be discarded"
        assert first.finished is False, "a spliced clip must never be finalised"
        # The packet loop runs concurrently with the frame loop, so by the time the
        # discontinuity is *detected* this handle has already been fed a new-timeline
        # packet. That is why aborting is the only option: there is no clean cut point
        # left to finish at.
        assert [packet.pts for packet in first.packets] != sorted(
            packet.pts for packet in first.packets
        ), "the open clip really was already spliced — the defect is not hypothetical"

        # No *surviving* clip may splice two timelines. Aborted files are discarded, so
        # only the ones that were finalised are evidence anyone will ever open.
        for index, handle in enumerate(writer.handles):
            if not handle.finished:
                continue
            pts = [packet.pts for packet in handle.packets]
            assert pts == sorted(pts), f"finalised clip {index} splices two timelines: {pts}"

        # Spec §9: discarding the file must not discard the event.
        assert len(publisher.events) == runner.telemetry().escalations
        assert runner.telemetry().escalations_dropped == 0
        aborted_uri = f"s3://sentinel-clips/cam-1/{first.event_id}.mp4"
        assert all(event.clip_uri != aborted_uri for event in publisher.events), (
            "no event may point at the aborted clip"
        )

    async def test_a_continuous_stream_never_reports_a_discontinuity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The negative control: a reset-on-every-frame implementation would pass
        every assertion above and fail this one."""
        patch_postroll(monkeypatch, seconds=5.0)
        tracker = ResetSpyTracker()
        motion = ResetSpyMotion()
        preroll = ClearSpyPreRoll()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=10),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            tracker=tracker,
            motion=motion,
            preroll=preroll,
        )

        async with Worker(scheduler):
            await runner.run()

        assert runner.telemetry().discontinuities == 0
        assert tracker.resets == 0
        assert motion.resets == 0
        assert preroll.clears == 0


class TestClipLifecycle:
    async def test_a_second_escalation_extends_the_postroll_instead_of_opening_a_new_clip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PERIODIC_SUMMARY fires on frame 0 and opens a clip; NEW_SALIENT_TRACK fires
        again on frame 3 while that clip is still recording.

        Three separate ways this fails against a broken implementation:
          * one that opens a second, overlapping clip -> `len(writer.handles) == 2`;
          * one that hands the same handle to both requests -> two events carry a uri
            and the handle is finished twice;
          * one that ignores the extension -> the clip stops at the original 1.0s
            deadline instead of the extended 1.3s one.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2, cooldown_seconds=0.2),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert len(writer.handles) == 1, "the second escalation must not open a second clip"
        assert runner.telemetry().escalations == 2
        assert len(publisher.events) == 2

        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.aborted is False
        assert sum(event.clip_uri is not None for event in publisher.events) == 1, (
            "only the clip-owning escalation's event carries the uri"
        )

        pts = [packet.pts for packet in handle.packets]
        assert pts == sorted(pts) and len(pts) == len(set(pts)), (
            "pre-roll and live packets must interleave in pts order, without duplicates"
        )
        assert pts[-1] > 1.0, "the post-roll was cut short at the un-extended deadline"
        assert pts[-1] == pytest.approx(1.3), "closed at the extended deadline"

    async def test_a_clip_still_recording_when_the_stream_ends_is_aborted_but_the_event_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spec §9: an anomaly event is never lost to an infrastructure failure —
        and a camera stopping mid-post-roll is one."""
        patch_postroll(monkeypatch, seconds=60.0)  # never reached by a 1.5s stream
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2, cooldown_seconds=0.2),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.aborted is True
        assert handle.finished is False, "a partial clip is discarded, never finalised"
        assert len(publisher.events) == 2
        assert all(event.clip_uri is None for event in publisher.events)

    async def test_a_failing_clip_append_costs_the_clip_and_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remux failure on one packet must not take the camera down with it.

        Before this was guarded, the exception escaped `_packet_loop` and killed that
        task. `run()` is still awaiting the *frame* loop at that point, so it never
        observes the dead task — and the source's packet queue, which a real demux
        thread fills with a blocking put, then fills up and stops the demux, so frames
        stop arriving too. The camera reports a frozen `frames_seen` with no error
        logged anywhere. Observed live against RTSP through the composition root after
        ~5 000 frames; here it is provoked deterministically.

        Fails against the unguarded loop with `frames_seen == 3` (the mailbox depth)
        instead of 15, and with no event published at all.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter(
            append_error=RuntimeError("mux: Invalid argument"), append_error_after=3
        )
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2, cooldown_seconds=0.2),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        telemetry = runner.telemetry()
        assert telemetry.frames_seen == 15, "the camera must survive a failed clip append"
        assert telemetry.escalations >= 1
        handle = writer.handles[0]
        assert handle.finished is False
        assert handle.aborted is True, "the unusable partial file is discarded"
        assert len(publisher.events) == telemetry.escalations, "spec §9: the event survives"
        assert all(event.clip_uri is None for event in publisher.events)

    async def test_a_clip_that_cannot_even_be_opened_costs_the_clip_and_nothing_else(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same rule one step earlier, and on the frame loop rather than the packet
        loop — so an unguarded failure here ends `run()` outright.

        Fails against the unguarded `_escalate` with the writer's error propagating out
        of `runner.run()`.

        B3 strengthened the tail of this test. `open()` succeeds here and the pre-roll
        flush's first `append()` is what fails, and the handler used to submit with
        `clip=None` and return **without** aborting — unlike its sibling in
        `_packet_loop`, which always has. The reviewer's probe:

            PROBE handles opened : 2
            PROBE handle[0] finished=False aborted=False
            PROBE handle[1] finished=False aborted=False

        With `MinioClipHandle` each of those is a live PyAV container plus a temp .mp4,
        and because `_active_clip` is never set every subsequent escalation opens
        another one — an unbounded leak on a camera whose remux is failing, which is
        exactly the state B4's sub-tick fault leaves it in. The `aborted` assertions
        below fail against that version; everything above them already passed.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter(append_error=RuntimeError("mux: Invalid argument"))
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        preroll = PreRollBuffer(preroll_seconds=3.0)
        # Seeded so the flush inside `_escalate` genuinely has a packet to append: with
        # an empty pre-roll `open()` would succeed and nothing would ever fail.
        preroll.append(
            EncodedPacket(
                camera_id="cam-1", data=b"history", pts=-0.1, is_keyframe=True, codec="h264"
            )
        )
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            preroll=preroll,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2, cooldown_seconds=0.2),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().frames_seen == 15
        assert len(publisher.events) >= 1
        assert all(event.clip_uri is None for event in publisher.events)

        assert writer.handles, "test setup: a clip must actually have been opened"
        assert all(handle.finished is False for handle in writer.handles)
        assert all(handle.aborted is True for handle in writer.handles), (
            "a clip opened and then failed mid-seed must be aborted, not leaked: "
            f"{[(h.finished, h.aborted) for h in writer.handles]}"
        )

    async def test_the_clip_opens_with_the_preroll_already_in_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        preroll = PreRollBuffer(preroll_seconds=3.0)
        source = alternating_source("cam-1", count=15, fps=10.0)
        # Pre-load history from before the camera "started", exactly as the packet loop
        # would have during the seconds before an anomaly.
        for index in range(3):
            preroll.append(
                EncodedPacket(
                    camera_id="cam-1",
                    data=b"history",
                    pts=-0.3 + index / 10.0,
                    is_keyframe=True,
                    codec="h264",
                )
            )
        runner = make_runner(
            source=source,
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            preroll=preroll,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        pts = [packet.pts for packet in writer.handles[0].packets]
        assert min(pts) < 0.0, "the clip begins before the moment that triggered it"

    async def test_cancelling_the_runner_mid_stream_still_closes_the_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task 7's `EngineService.stop()` cancels these tasks; the cleanup in `run()`'s
        `finally` must survive running inside a cancelled task."""
        patch_postroll(monkeypatch, seconds=1.0)
        source = FakeSource.constant("cam-1", count=200, fps=100.0)
        detector = GatedDetector(asyncio.Event())  # never released
        publisher = FakePublisher()
        runner = make_runner(source=source, detector=detector, scheduler=new_scheduler(publisher))

        run_task = asyncio.create_task(runner.run())
        await asyncio.sleep(0)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert source.closed is True

    async def test_describe_now_uses_force_and_bypasses_the_governors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=1, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            profile=CameraProfile(
                camera_id="cam-1", auto_escalation_enabled=False
            ),  # governors refuse
        )

        async with Worker(scheduler):
            await runner.run()
            assert runner.telemetry().escalations == 0, (
                "auto_escalation_enabled=False suppresses everything"
            )

            event_id = await runner.describe_now()
            await scheduler.drain()

        assert isinstance(event_id, UUID)
        assert len(publisher.events) == 1
        assert publisher.events[0].reason.value == "user_requested"
        assert publisher.events[0].event_id == event_id
        assert runner.telemetry().escalations == 1

    async def test_describe_now_before_any_frame_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=0),
            detector=FakeDetector(script=[()]),
            scheduler=new_scheduler(publisher),
        )
        with pytest.raises(RuntimeError, match="has not processed a frame"):
            await runner.describe_now()


class TestEstimatedClipFps:
    """The fps a clip is opened with, measured off the source's own frame interval.

    Regression net, added before the fixes to C1-C3 touched this region. Nothing in
    the suite previously constrained `_estimated_fps` at all: `FakeClipWriter.opened`
    records `(camera_id, event_id, fps)` and the only assertion on it anywhere was
    `tests/ports/test_port_contracts.py`'s check that the fake echoes the literal the
    test itself handed to `open()`. A whole-suite mutation sweep confirmed the gap —
    eight separate mutations of `_estimated_fps` and its backing state, including
    replacing the entire body with `return 0.0`, all survived. An fps of 0 is not a
    harmless wrong number: it is the `rate` the MP4 muxer builds its time base from.
    """

    async def test_the_clip_opens_at_the_measured_source_frame_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """25 fps in, 25.0 out — not the 10.0 default, and not a value derived from
        the runner's own clock.

        `auto_escalation_enabled=False` keeps the gate silent so the clip under test is the one
        `describe_now()` opens, by which point two real frame timestamps have been
        seen and a genuine interval is measurable.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=6, fps=25.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
        )

        async with Worker(scheduler):
            await runner.run()
            await runner.describe_now()
            await scheduler.drain()

        assert [fps for _camera, _event, fps in writer.opened] == [pytest.approx(25.0)]

    async def test_two_frames_sharing_a_timestamp_fall_back_to_the_default_fps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero interval must yield the default, never a division by zero.

        Two frames stamped identically is not hypothetical: a source whose timestamps
        are quantised to a coarse clock produces them, and `_is_timeline_regression`
        deliberately treats equal timestamps as continuous, so nothing upstream
        filters this out.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource(
                [
                    FakeSource.make_frame("cam-1", 0, 0.0, value=0),
                    FakeSource.make_frame("cam-1", 1, 0.0, value=200),
                ]
            ),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
        )

        async with Worker(scheduler):
            await runner.run()
            await runner.describe_now()
            await scheduler.drain()

        assert [fps for _camera, _event, fps in writer.opened] == [
            pytest.approx(runner_module._DEFAULT_FPS)
        ]

    async def test_fps_is_never_measured_across_a_timeline_break(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The last frame before a reconnect and the first frame after it are not an
        interval — they are two points on two different clocks. Pairing them yields a
        negative delta, and a clip opened from it must fall back to the default rather
        than carry a negative frame rate into the muxer.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        frames = [
            FakeSource.make_frame("cam-1", index, index / 25.0, value=0 if index % 2 == 0 else 200)
            for index in range(6)
        ] + [FakeSource.make_frame("cam-1", 0, 0.0, value=200)]  # reconnect
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=FakeSource(frames),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
        )

        async with Worker(scheduler):
            await runner.run()
            await runner.describe_now()
            await scheduler.drain()

        assert runner.telemetry().discontinuities == 1
        assert [fps for _camera, _event, fps in writer.opened] == [
            pytest.approx(runner_module._DEFAULT_FPS)
        ]


class TestOneTimeBasePerCamera:
    """C2. `CameraRunner` used to carry a second, injected `clock` alongside the
    source's own timeline, with no documented relationship between them and no test
    that could see them diverge — every runner in this file was built with
    `clock=frozen_clock()` against frames that also start at 0.0, so the two were
    accidentally aligned everywhere.

    Diverged, the token bucket stops refilling forever. `GateState.initial` stamps
    `bucket.updated_at` from the runner clock; `decide()` refills against
    `scene.timestamp`. When the runner clock reads ahead of the source — which is
    guaranteed for `FileSource`, whose timeline starts at 0.0, against any
    `time.monotonic` runner clock — `elapsed` is negative on every frame and
    `refilled()`'s regressing-clock guard returns the bucket unchanged. The camera
    spends its burst tokens and then reports `suppressed_by="rate_budget"` forever.

    Measured over 120 s of identical footage, varying only the clock:
    13 escalations aligned, 2 with `clock=time.monotonic`.
    """

    def test_the_runner_takes_no_clock_of_its_own(self) -> None:
        """The structural half of the fix, and the only guard that keeps it fixed.

        Reconciling the two clocks for RTSP alone is not enough: `FileSource` stamps
        from the file's own presentation timestamps, and no `Callable[[], float]` can
        track a pump thread's position in a file. The only fix that covers both is for
        the runner to have exactly one time base — the source's — which makes the
        divergence unrepresentable rather than merely discouraged.
        """
        parameters = inspect.signature(CameraRunner.__init__).parameters
        assert "clock" not in parameters, (
            "a second time base in the runner is C2; every consumer must read "
            "FrameData.timestamp / EncodedPacket.pts"
        )

    async def test_the_rate_budget_spends_and_refills_on_the_source_timeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A camera whose frames start at ~10**4 — i.e. every RTSP camera — must still
        spend its burst and then earn tokens back at the configured rate.

        `bucket_capacity=2` over `bucket_refill_seconds=10.0` is two calls up front and
        one per ten seconds of *source* time thereafter; over sixteen seconds of footage
        that is exactly three. This is a characterisation test, not a discriminator: it
        pins that the budget governor reads the same timeline the gate does, so that a
        future clock reintroduced anywhere on the decide path has something to fail.
        The discriminating evidence for the bucket half of C2 could only ever be a
        probe that varied `CameraRunner.clock`, and that parameter no longer exists.
        """
        patch_postroll(monkeypatch, seconds=60.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher, maxsize=16)
        runner = make_runner(
            source=alternating_source("cam-1", count=160, fps=10.0, origin=10_000.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            profile=CameraProfile(
                camera_id="cam-1",
                min_track_frames=1,
                cooldown_seconds=0.15,
                bucket_capacity=2,
                bucket_refill_seconds=10.0,
            ),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        stamps = [event.occurred_at for event in publisher.events]
        assert len(stamps) == 3, f"two burst tokens plus one refilled in 16 s, got {stamps}"
        assert stamps[1] - stamps[0] < 1.0, "the two burst tokens are spent back to back"
        assert stamps[2] - stamps[1] == pytest.approx(9.9, abs=0.2), (
            "the third had to wait a full refill interval of source time"
        )

    async def test_describe_now_is_stamped_on_the_source_timeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`describe_now()` used to take its `now` from the runner clock and feed it to
        `force()`, which writes it to `GateState.last_escalation_at` — the value the
        cooldown then compares against `scene.timestamp`. One operator request was
        therefore enough to put the gate's cooldown on a different clock from the gate's
        own input, on top of stamping telemetry with a time no frame ever had.

        The correct value is available and already required: the handler refuses to run
        without `self._last_scene`, whose `timestamp` is by definition the newest point
        on the source's timeline.
        """
        patch_postroll(monkeypatch, seconds=60.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
        )

        async with Worker(scheduler):
            await runner.run()
            await runner.describe_now()
            await scheduler.drain()

        assert runner.telemetry().last_escalation_at == pytest.approx(1.4), (
            "the last frame's own timestamp, not a reading from a second clock"
        )
        # `source_timestamp`, not `occurred_at`: since D1 the latter is Unix epoch
        # seconds, and this test is about which *source* instant was stamped.
        assert publisher.events[0].source_timestamp == pytest.approx(1.4)

    async def test_the_clip_deadline_is_on_the_same_timeline_as_the_packets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_ActiveClip.deadline` is compared against `packet.pts`, so it has to be
        computed on the packet timeline.

        The gate path always did compute it that way (`_process_frame` passes
        `now=frame.timestamp`), so the operator-request path is where the mismatch
        actually lived: `describe_now()` took its `now` from the runner clock and handed
        that to the same `deadline = now + postroll` arithmetic. Behind the source, as
        here, the deadline is already in the past and the clip closes on the first live
        packet instead of recording its post-roll — a stub of an evidence file for the
        one request an operator explicitly made. Ahead of the source — a `FileSource`
        replay against `time.monotonic` — the deadline is never reached, the clip is
        never finished, and because an already-recording clip only ever extends, no
        later escalation on that camera can record one either.

        `describe_now()` has to be called while the camera is still running, since a
        post-roll needs live packets after it. `auto_escalation_enabled=False` keeps the gate silent
        so the clip under test is unambiguously the operator's. The pre-roll is
        deliberately tiny: the packet loop runs ahead of the frame loop, so a
        three-second buffer would flush packets past the deadline in at open time and
        mask the difference.
        """
        patch_postroll(monkeypatch, seconds=3.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        source = alternating_source("cam-1", count=60, fps=10.0, origin=10_000.0)
        runner = make_runner(
            source=source,
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            preroll=PreRollBuffer(preroll_seconds=0.05),
            profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
        )

        async def until_five_frames_are_processed() -> None:
            while runner.telemetry().detections_run < 5:
                await asyncio.sleep(0)  # cooperative yield; zero wall-clock time

        async with Worker(scheduler):
            run_task = asyncio.create_task(runner.run())
            await asyncio.wait_for(until_five_frames_are_processed(), timeout=5.0)
            await runner.describe_now()
            await run_task
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.finished is True, "the post-roll deadline must be reachable"
        owning = next(event for event in publisher.events if event.clip_uri is not None)
        last_pts = handle.packets[-1].pts
        # Against `source_timestamp`: packet pts are on the source timeline, and since
        # D1 that is the field carrying it — `occurred_at` is Unix epoch seconds.
        assert owning.source_timestamp is not None
        assert last_pts - owning.source_timestamp == pytest.approx(3.0, abs=0.11), (
            "the clip must keep recording for its whole post-roll after the escalation; "
            f"escalated at {owning.source_timestamp}, last packet at {last_pts}"
        )


class TestPerCameraClipAndSummaryOverrides:
    """`cameras.json`'s `clip_preroll_seconds`, `clip_postroll_seconds` and
    `summary_interval_seconds`: the per-camera values that win over the process-wide
    settings and the camera profile.

    Every override test below is paired with the same scenario run without the
    override, so a runner that quietly kept reading the global still fails: the
    global is deliberately set to a value whose outcome is the opposite one.
    """

    async def test_a_camera_without_overrides_uses_the_global_postroll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=0.5)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.packets[-1].pts == pytest.approx(0.5)

    async def test_a_per_camera_postroll_overrides_the_global(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The global is 60s — longer than the whole stream — so a runner that read it
        would leave the clip open and abort it at end of stream."""
        patch_postroll(monkeypatch, seconds=60.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            clip_postroll_seconds=0.5,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.finished is True, "the per-camera post-roll must be the one in force"
        assert handle.aborted is False
        assert handle.packets[-1].pts == pytest.approx(0.5)

    async def test_a_camera_without_overrides_uses_the_global_preroll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=60.0)
        writer, scheduler, runner, preroll = self._preroll_camera(override=None)

        for index in range(4):
            preroll.append(history_packet(round(-0.3 + index / 10.0, 4)))
        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        pts = [packet.pts for packet in writer.handles[0].packets]
        assert pts == [-0.3, -0.2, -0.1, 0.0], "a 3s horizon keeps every buffered keyframe"

    async def test_a_per_camera_preroll_of_zero_gives_the_clip_no_lead_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero is an accepted value at both configuration edges (`Settings` declares
        `ge=0`, `PATCH /cameras/{id}` accepts it), so it has to mean something rather
        than crash: the clip starts at the escalation with no buffered history."""
        patch_postroll(monkeypatch, seconds=60.0)
        writer, scheduler, runner, preroll = self._preroll_camera(override=0.0)

        for index in range(4):
            preroll.append(history_packet(round(-0.3 + index / 10.0, 4)))
        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        pts = [packet.pts for packet in writer.handles[0].packets]
        assert pts == [0.0], "no lead-in: only the keyframe the escalation lands on"

    @staticmethod
    def _preroll_camera(
        *, override: float | None
    ) -> tuple[FakeClipWriter, VlmScheduler, CameraRunner, PreRollBuffer]:
        """A camera whose source produces frames but no packets, so the only thing that
        can reach the clip is the pre-roll flush.

        Without that, the packet loop races the frame loop and the clip's contents
        depend on which got further — which is exactly the ambiguity a test about how
        much history a clip starts with must not have.
        """
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        frames = [
            FakeSource.make_frame("cam-1", index, index / 10.0, value=0 if index % 2 == 0 else 200)
            for index in range(5)
        ]
        preroll = PreRollBuffer(preroll_seconds=3.0)
        runner = make_runner(
            source=FakeSource(frames, packets=[]),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            preroll=preroll,
            clip_preroll_seconds=override,
        )
        return writer, scheduler, runner, preroll

    async def test_a_camera_without_overrides_uses_the_profiles_summary_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher, maxsize=32)
        runner = make_runner(
            source=alternating_source("cam-1", count=30, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            profile=self._summary_profile(interval=60.0),
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().escalations == 1, "one initial summary, then 60s of silence"

    async def test_a_per_camera_summary_interval_overrides_the_profiles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The profile asks for one summary a minute; this camera is told to look every
        half second, and a 3s stream must produce several."""
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher, maxsize=32)
        runner = make_runner(
            source=alternating_source("cam-1", count=30, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            profile=self._summary_profile(interval=60.0),
            summary_interval_seconds=0.5,
        )

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        telemetry = runner.telemetry()
        assert telemetry.escalations >= 5, "0.5s apart across a 3s stream"
        assert telemetry.escalations_dropped == 0
        assert {event.reason.value for event in publisher.events} == {"periodic_summary"}

    async def test_the_override_does_not_touch_the_rest_of_the_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`summary_interval_seconds` is the only profile field an edit may move — the
        gate is part way through applying every other one, which is why `profile` is a
        restart-required field."""
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        base = self._summary_profile(interval=60.0)
        runner = make_runner(
            source=alternating_source("cam-1", count=2, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=new_scheduler(publisher),
            profile=base,
            summary_interval_seconds=0.5,
        )

        effective = runner._profile
        assert effective.summary_interval_seconds == 0.5
        assert replace(effective, summary_interval_seconds=60.0) == base

    @staticmethod
    def _summary_profile(*, interval: float) -> CameraProfile:
        """A profile on which PERIODIC_SUMMARY is the only trigger that can fire.

        `scene_delta_frames` is set past the length of any stream here to silence the
        scene-change trigger, which outranks the summary in `ALL_TRIGGERS` and would
        otherwise claim the alternating source's every-frame delta and report its
        reason instead. The cooldown and bucket are widened so the governors do not
        decide the escalation count instead of the interval under test.
        """
        return CameraProfile(
            camera_id="cam-1",
            summary_interval_seconds=interval,
            scene_delta_frames=10_000,
            cooldown_seconds=0.05,
            bucket_capacity=8,
            bucket_refill_seconds=0.05,
        )


class TestALiveEditTakesEffectOnTheNextEscalation:
    """`apply_metadata` on a running camera, for the three fields that are policy.

    The contract is deliberately *not* "immediately": a clip already recording keeps
    the length it started with, because recomputing an in-flight clip's deadline from
    a new post-roll is how a clip ends up with impossible timestamps — a deadline
    already behind the packets being written into it, or one moved past the end of
    the stream that will record it.
    """

    async def test_an_edited_postroll_is_used_by_the_next_escalation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=60.0)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
        )
        apply_policy(runner, clip_postroll_seconds=0.5)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.packets[-1].pts == pytest.approx(0.5)

    async def test_an_edit_reverting_to_null_restores_the_global(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`null` on the wire means "use the engine-wide default again", and it has to
        reach the running camera as such — otherwise the only way to undo an override
        is a restart."""
        patch_postroll(monkeypatch, seconds=0.5)
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
            clip_postroll_seconds=60.0,
        )
        apply_policy(runner, clip_postroll_seconds=None)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        handle = writer.handles[0]
        assert handle.finished is True, "the global post-roll must be back in force"
        assert handle.packets[-1].pts == pytest.approx(0.5)

    async def test_an_edit_mid_recording_leaves_the_open_clips_deadline_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The edit lands from inside the clip's own `append`, so it is unambiguously
        mid-recording. An implementation that recomputed `_ActiveClip.deadline` would
        cut this clip off at ~0.2s instead of the 1.0s it opened with.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = EditingClipWriter(after_packets=2)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
        )
        writer.on_packet = lambda: apply_policy(runner, clip_postroll_seconds=0.2)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert writer.fired is True, "test setup: the edit must land while the clip records"
        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.packets[-1].pts == pytest.approx(1.0), (
            "the clip must keep the post-roll it opened with, not the edited one"
        )

    async def test_an_edit_landing_while_the_clip_opens_does_not_split_the_escalation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_escalate` awaits `ClipWriter.open()` between the gate's decision and the
        deadline it derives from it, so an edit genuinely can land inside one
        escalation. The post-roll is read before that await, which is what keeps the
        escalation whole: half on the policy the gate decided under and half on the
        one that arrived a millisecond later is the outcome with no meaning.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        writer = EditingClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            clip_writer=writer,
        )
        writer.on_open = lambda: apply_policy(runner, clip_postroll_seconds=0.2)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().clip_postroll_seconds == 0.2, "test setup: the edit landed"
        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.packets[-1].pts == pytest.approx(1.0), (
            "the clip's length is the one the escalation was decided under"
        )

    @pytest.mark.parametrize(
        ("clip_preroll_seconds", "summary_interval_seconds"),
        [(-1.0, None), (None, 0.0)],
        ids=["negative-preroll", "zero-interval"],
    )
    def test_a_rejected_edit_leaves_the_camera_exactly_as_it_was(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clip_preroll_seconds: float | None,
        summary_interval_seconds: float | None,
    ) -> None:
        """`apply_metadata` is documented as total, and `ComposedService.update_camera`
        leans on that: the file is already written by the time the apply runs, so a
        half-applied edit would leave memory and disk disagreeing with nothing to
        unwind it with.

        Neither value below can reach here through `PATCH /cameras/{id}` —
        `CameraEditRequest` and `load_cameras` enforce the same two bounds — which is
        why this asserts the property directly instead of through the endpoint. The
        totality has to be a property of this method, not of the two callers that
        happen to validate first: the argument that it is unreachable is one edge
        change away from being wrong, and the failure it would then produce is a
        camera whose ring and label moved while its profile did not.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        preroll = PreRollBuffer(preroll_seconds=3.0)
        base = CameraProfile(camera_id="cam-1", summary_interval_seconds=45.0)
        runner = make_runner(
            source=alternating_source("cam-1", count=2, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=new_scheduler(FakePublisher()),
            preroll=preroll,
            profile=base,
        )
        before = runner.telemetry()

        with pytest.raises(ValueError):
            apply_policy(
                runner,
                label="Renamed",
                clip_preroll_seconds=clip_preroll_seconds,
                summary_interval_seconds=summary_interval_seconds,
            )

        after = runner.telemetry()
        assert after == before, "not one field of the record may have moved"
        assert preroll.preroll_seconds == 3.0, "the ring keeps the horizon it had"
        assert runner._profile == base
        assert runner._clip_postroll_seconds == pytest.approx(1.0)

    async def test_a_second_escalation_extends_using_the_edited_postroll(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of "next escalation, not retroactively". The clip already
        recording keeps the deadline it opened with; the escalation that arrives while
        it records *is* the next escalation, so its extension must use the value in
        force now, not the one the first escalation was decided under.

        The edit lands inside `ClipWriter.open()` for the one clip here — after the
        first escalation has read its post-roll and before the second escalation
        exists — so there is no race to lose. PERIODIC_SUMMARY opens the clip at 0.0s
        with the un-edited 0.5s post-roll; NEW_SALIENT_TRACK extends it at 0.3s.
        Against an implementation that reused the opening escalation's value, or
        cached the post-roll on `_ActiveClip`, the clip closes at 0.8s.
        """
        patch_postroll(monkeypatch, seconds=0.5)
        writer = EditingClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher)
        runner = make_runner(
            source=alternating_source("cam-1", count=15, fps=10.0),
            detector=FakeDetector(script=[(NEAR,)]),
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2, cooldown_seconds=0.2),
        )
        writer.on_open = lambda: apply_policy(runner, clip_postroll_seconds=1.0)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().clip_postroll_seconds == 1.0, "test setup: the edit landed"
        assert runner.telemetry().escalations == 2, "test setup: exactly one extension"
        assert len(writer.handles) == 1, "the second escalation extends, it does not open a clip"
        handle = writer.handles[0]
        assert handle.finished is True
        assert handle.aborted is False
        assert handle.packets[-1].pts == pytest.approx(1.3), (
            "the extension must use the edited post-roll: 0.3s + 1.0s, not 0.3s + 0.5s"
        )

    async def test_an_edited_summary_interval_reaches_the_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher, maxsize=32)
        runner = make_runner(
            source=alternating_source("cam-1", count=30, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            profile=TestPerCameraClipAndSummaryOverrides._summary_profile(interval=60.0),
        )
        apply_policy(runner, summary_interval_seconds=0.5)

        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

        assert runner.telemetry().escalations >= 5

    async def test_an_edited_preroll_resizes_the_live_ring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-sized, not replaced: the ring holds the history the next escalation will
        want, and rebuilding it would throw that away to change one number.

        Two GOPs, so the flush is a real discriminator rather than a formality: under
        the 3.0s horizon the ring was built with, no keyframe is old enough and the
        flush falls back to the earliest one it holds (everything); under the edited
        0.0s horizon the newest keyframe already satisfies it and the flush is the
        open GOP alone. A runner that dropped the edit on the floor returns the first
        list, one that rebuilt the ring returns nothing at all.
        """
        patch_postroll(monkeypatch, seconds=1.0)
        preroll = PreRollBuffer(preroll_seconds=3.0)
        runner = make_runner(
            source=alternating_source("cam-1", count=2, fps=10.0),
            detector=FakeDetector(script=[()]),
            scheduler=new_scheduler(FakePublisher()),
            preroll=preroll,
        )
        for packet in two_gops():
            preroll.append(packet)
        assert [p.pts for p in preroll.flush()] == [-0.4, -0.3, -0.2, -0.1], (
            "test setup: under the original horizon the whole buffer is in reach"
        )

        apply_policy(runner, clip_preroll_seconds=0.0)

        assert preroll.preroll_seconds == 0.0
        assert [p.pts for p in preroll.flush()] == [-0.2, -0.1], (
            "the narrowed horizon anchors on the newest keyframe, from the next flush on"
        )
        assert preroll.span_seconds == pytest.approx(0.3), (
            "history is kept, not dropped: the older GOP is still buffered, merely "
            "out of the flush's reach"
        )


class TestTheWelfarePolicyOnTelemetry:
    """`GET /cameras` is built from `CameraTelemetry`, so the per-camera welfare policy
    has to ride on it or a console cannot read back what it just wrote."""

    def _runner(
        self,
        *,
        clip_preroll_seconds: float | None = None,
        clip_postroll_seconds: float | None = None,
        summary_interval_seconds: float | None = None,
    ) -> CameraRunner:
        return make_runner(
            source=FakeSource.constant("cam-1", count=1),
            detector=FakeDetector(script=[()]),
            scheduler=new_scheduler(FakePublisher()),
            clip_preroll_seconds=clip_preroll_seconds,
            clip_postroll_seconds=clip_postroll_seconds,
            summary_interval_seconds=summary_interval_seconds,
        )

    def test_an_unconfigured_camera_reports_the_loaders_defaults(self) -> None:
        telemetry = self._runner().telemetry()
        assert telemetry.notify_on == frozenset(ConcernKind)
        assert telemetry.notify_min_confidence is Confidence.LIKELY
        assert telemetry.clip_preroll_seconds is None
        assert telemetry.clip_postroll_seconds is None
        assert telemetry.summary_interval_seconds is None

    def test_the_durations_are_reported_as_stored_not_as_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Null means "this camera follows the default", which is what `PATCH` stores
        and what `CameraEditResponse` reports. Reporting the resolved 5.0 here instead
        would make a console that re-submits what it read pin the camera to a value
        nobody chose."""
        patch_postroll(monkeypatch, seconds=5.0)
        assert self._runner().telemetry().clip_postroll_seconds is None

    def test_a_configured_camera_reports_its_overrides(self) -> None:
        telemetry = self._runner(
            clip_preroll_seconds=0.0,
            clip_postroll_seconds=2.5,
            summary_interval_seconds=90.0,
        ).telemetry()
        assert telemetry.clip_preroll_seconds == 0.0
        assert telemetry.clip_postroll_seconds == 2.5
        assert telemetry.summary_interval_seconds == 90.0

    def test_a_live_edit_is_visible_on_the_very_next_read(self) -> None:
        runner = self._runner()
        apply_policy(
            runner,
            label="Wing B corridor",
            zone=Zone.CORRIDOR,
            notify_on=frozenset({ConcernKind.COLLAPSE}),
            notify_min_confidence=Confidence.POSSIBLE,
            clip_preroll_seconds=0.0,
            clip_postroll_seconds=2.5,
            summary_interval_seconds=90.0,
        )

        telemetry = runner.telemetry()
        assert telemetry.label == "Wing B corridor"
        assert telemetry.zone is Zone.CORRIDOR
        assert telemetry.notify_on == frozenset({ConcernKind.COLLAPSE})
        assert telemetry.notify_min_confidence is Confidence.POSSIBLE
        assert telemetry.clip_preroll_seconds == 0.0
        assert telemetry.clip_postroll_seconds == 2.5
        assert telemetry.summary_interval_seconds == 90.0


class TestTheWelfarePolicyOnTheEscalationRequest:
    """T10: the policy the console edits has to reach the code that routes.

    `CameraRunner` still branches on none of it — the routing rule runs in
    `VlmScheduler`, downstream of the published event. But the scheduler has no
    reference to any runner, so the policy travels on the request, snapshotted at
    the moment the escalation was decided. Nothing else in the suite would notice
    if that snapshot were dropped: every welfare test upstream of here builds an
    `EscalationRequest` by hand.
    """

    async def _escalate(self, runner: CameraRunner, scheduler: RecordingScheduler) -> None:
        async with Worker(scheduler):
            await runner.run()
            await scheduler.drain()

    async def test_the_request_carries_the_cameras_zone_and_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        scheduler = RecordingScheduler(FakePublisher())
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=1),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
            zone=Zone.DAYROOM,
            notify_on=frozenset({ConcernKind.COLLAPSE}),
            notify_min_confidence=Confidence.POSSIBLE,
        )
        await self._escalate(runner, scheduler)

        assert len(scheduler.submitted) == 1
        request = scheduler.submitted[0]
        assert request.zone is Zone.DAYROOM
        assert request.notify_on == frozenset({ConcernKind.COLLAPSE})
        assert request.notify_min_confidence is Confidence.POSSIBLE

    async def test_a_live_edit_reaches_the_next_escalation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`apply_metadata` promises the new values take effect on the next
        escalation. For `notify_on` that promise is only worth anything if the
        request is built from the current field rather than from one captured at
        construction — muting a camera through `PATCH /cameras/{id}` and having it
        keep notifying is the failure this test exists to catch."""
        patch_postroll(monkeypatch, seconds=1.0)
        scheduler = RecordingScheduler(FakePublisher())
        runner = make_runner(
            source=FakeSource.constant("cam-1", count=1),
            detector=FakeDetector(script=[()]),
            scheduler=scheduler,
        )
        apply_policy(runner, zone=Zone.ROOM, notify_on=frozenset())
        await self._escalate(runner, scheduler)

        assert len(scheduler.submitted) == 1
        assert scheduler.submitted[0].notify_on == frozenset()
        assert scheduler.submitted[0].zone is Zone.ROOM


class TestFallDetectionInTheFrameLoop:
    """§7 end to end: a fall in the frame stream becomes a published event.

    The state machine's own behaviour is pinned exhaustively in
    `tests/domain/behaviour/test_fall.py`. What these own is the *wiring* — that the
    runner actually feeds it, that a completed signature reaches the scheduler with
    the right reason, and that a camera which did not ask for it pays nothing. §40's
    rule in test form: the capability has to be connected, not merely present.
    """

    @staticmethod
    def _upright(cx: float = 100.0, cy: float = 100.0) -> BBox:
        return BBox(x1=cx - 20.0, y1=cy - 50.0, x2=cx + 20.0, y2=cy + 50.0)

    @staticmethod
    def _fallen(cx: float = 100.0, cy: float = 160.0) -> BBox:
        return BBox(x1=cx - 50.0, y1=cy - 20.0, x2=cx + 50.0, y2=cy + 20.0)

    def _fall_script(self) -> tuple[list[FrameData], list[tuple[Detection, ...]]]:
        """Stand for a second, drop, stay down past the settle window."""
        timestamps = [round(0.2 * i, 3) for i in range(41)]
        boxes = [self._upright() if t <= 1.0 else self._fallen() for t in timestamps]
        frames = [
            FakeSource.make_frame("cam-1", index, timestamp)
            for index, timestamp in enumerate(timestamps)
        ]
        script: list[tuple[Detection, ...]] = [(Detection("person", 0.9, box),) for box in boxes]
        return frames, script

    async def test_a_fall_escalates_with_the_fall_reason(self) -> None:
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=FallPolicy(),
                # Everything else silent, so the only escalation that can appear is
                # the one under test.
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            await scheduler.drain()

        reasons = [event.reason for event in publisher.events]
        assert EscalationReason.FALL_SUSPECTED in reasons

    async def test_the_escalation_detail_carries_the_hedged_summary(self) -> None:
        """What the operator and the VLM both read. §7 forbids claiming certainty."""
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=FallPolicy(),
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            await scheduler.drain()

        fall = next(e for e in publisher.events if e.reason is EscalationReason.FALL_SUSPECTED)
        assert "appears to have fallen" in fall.description.lower() or fall.description

    async def test_one_fall_produces_one_escalation_not_one_per_frame(self) -> None:
        """Thirty frames of an unchanged body on the floor. The machine reports once
        per episode, so the alert engine is never the thing under load."""
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=FallPolicy(),
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            await scheduler.drain()

        falls = [e for e in publisher.events if e.reason is EscalationReason.FALL_SUSPECTED]
        assert len(falls) == 1

    async def test_the_telemetry_counts_it(self) -> None:
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=FallPolicy(),
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            assert runner.telemetry().falls_suspected == 1

    async def test_a_camera_without_the_capability_never_raises_one(self) -> None:
        """`fall_policy=None` is how `compose` expresses "this camera did not ask".
        The same footage must then produce nothing — otherwise the capability toggle
        is decoration."""
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=None,
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            await scheduler.drain()

        assert publisher.events == []
        assert runner.telemetry().falls_suspected == 0

    async def test_a_pose_model_that_raises_degrades_to_geometry(self) -> None:
        """An optional model's failure must cost the reading's quality, never the
        camera. Raising here would take a pipeline down over an enhancement that has
        a working fallback."""

        class BrokenPose(PoseEstimator):
            def __init__(self) -> None:
                self.calls = 0

            async def estimate(
                self, frame: FrameData, tracks: tuple[Track, ...]
            ) -> dict[int, PersonPose]:
                self.calls += 1
                raise RuntimeError("cuda is on fire")

        pose = BrokenPose()
        frames, script = self._fall_script()
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher)) as scheduler:
            runner = make_runner(
                source=FakeSource(frames),
                detector=FakeDetector(script),
                scheduler=scheduler,
                fall_policy=FallPolicy(),
                pose=pose,
                profile=CameraProfile(camera_id="cam-1", auto_escalation_enabled=False),
            )
            await runner.run()
            await scheduler.drain()

        assert pose.calls > 0, "the pose model was actually consulted"
        falls = [e for e in publisher.events if e.reason is EscalationReason.FALL_SUSPECTED]
        assert len(falls) == 1, "the fall was still found, on geometry alone"
