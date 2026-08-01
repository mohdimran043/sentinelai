from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterator
from types import TracebackType
from uuid import UUID

import pytest

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import Settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline import runner as runner_module
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeTracker, FakeVisionLLM

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


def frozen_clock(value: float = 0.0) -> Callable[[], float]:
    """The runner's injected clock. Only `describe_now` and the gate's initial state
    read it — every other timestamp comes off the frames themselves — so freezing it
    keeps the tests independent of wall-clock time entirely."""
    return lambda: value


def alternating_source(camera_id: str, count: int, fps: float = 10.0) -> FakeSource:
    """Frames whose luma alternates between two values.

    Consecutive escalations need distinguishable scenes: identical frames produce
    identical signatures, and the gate's duplicate-scene governor would suppress
    everything after the first escalation.
    """
    return FakeSource(
        [
            FakeSource.make_frame(camera_id, index, index / fps, value=0 if index % 2 == 0 else 200)
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


def new_scheduler(
    publisher: FakePublisher,
    vlm: FakeVisionLLM | None = None,
    maxsize: int = 4,
) -> VlmScheduler:
    return VlmScheduler(
        vlm=vlm or FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=maxsize,
        timeout_seconds=5.0,
        clock=frozen_clock(),
    )


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
        clock=frozen_clock(),
        detect_every_n_frames=detect_every_n_frames,
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
        one-slot queue fills immediately and the second escalation is dropped, counted
        and forgotten — the runner keeps going."""
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
        assert telemetry.escalations == 1
        assert telemetry.escalations_dropped == 1
        assert scheduler.dropped == 1


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
            profile=CameraProfile(camera_id="cam-1", vlm_enabled=False),  # governors refuse
        )

        async with Worker(scheduler):
            await runner.run()
            assert runner.telemetry().escalations == 0, "vlm_enabled=False suppresses everything"

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
