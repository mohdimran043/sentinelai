from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pytest

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.orchestrator.service import EngineService, UnknownCameraError, _IdleSweeper
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource
from tests.fakes.io import FakeClipHandle, FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeModelRuntime, FakeTracker, FakeVisionLLM

DETECTOR_SPEC = ModelSpec(model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None)
VLM_SPEC = ModelSpec(model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0)


def clock() -> float:
    return 0.0


def build_service(monkeypatch: pytest.MonkeyPatch) -> tuple[EngineService, FakePublisher]:
    from sentinel_ai.config import Settings
    from sentinel_ai.pipeline import runner as runner_module

    monkeypatch.setattr(runner_module, "get_settings", lambda: Settings(clip_postroll_seconds=1.0))

    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    registry.register(VLM_SPEC, FakeModelRuntime("qwen25vl3b", vram_mib=4400))
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    publisher = FakePublisher()
    scheduler = VlmScheduler(
        vlm=FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )
    runner = CameraRunner(
        camera_id="cam-1",
        camera_label="Front Door",
        source=FakeSource.constant("cam-1", count=1, fps=10.0),
        detector=FakeDetector(script=[()]),
        tracker=FakeTracker(),
        motion=MotionAnalyzer(),
        profile=CameraProfile(camera_id="cam-1"),
        scheduler=scheduler,
        clip_writer=None,
        preroll=PreRollBuffer(preroll_seconds=3.0),
        clock=clock,
    )
    service = EngineService(
        cameras={"cam-1": runner},
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=clock,
    )
    return service, publisher


async def _run_to_quiescence() -> None:
    """Give every scheduled task a chance to run to its next suspension point.

    A single `asyncio.sleep(0)` only advances the loop by one ready-queue pass;
    the camera task tree here (runner -> produce/packet sub-tasks) is several
    awaits deep, so draining a handful of ticks is what actually lets the
    one-frame source finish rather than leaving it mid-flight when the test
    reads telemetry. Every yield is `asyncio.sleep(0)` -- zero wall-clock time.
    """
    for _ in range(20):
        await asyncio.sleep(0)


async def test_start_loads_the_required_models_and_runs_every_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _publisher = build_service(monkeypatch)
    await service.start()
    await _run_to_quiescence()
    await service.stop()

    assert service.health()["yolo11s"].state.value == "healthy"
    assert service.health()["qwen25vl3b"].state.value == "healthy"
    telemetry = service.telemetry("cam-1")
    assert telemetry.frames_seen == 1


async def test_cameras_lists_every_configured_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    await service.start()
    await _run_to_quiescence()
    await service.stop()
    assert {t.camera_id for t in service.cameras()} == {"cam-1"}


async def test_telemetry_of_an_unknown_camera_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    with pytest.raises(UnknownCameraError, match="cam-404"):
        service.telemetry("cam-404")


async def test_describe_now_of_an_unknown_camera_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    with pytest.raises(UnknownCameraError, match="cam-404"):
        await service.describe_now("cam-404")


async def test_describe_now_of_a_known_camera_delegates_to_its_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _publisher = build_service(monkeypatch)
    await service.start()
    await _run_to_quiescence()
    event_id = await service.describe_now("cam-1")
    await service.stop()
    assert event_id is not None


class TestIdleSweeperUnit:
    """`_IdleSweeper` owns no task/queue state of its own — like `_ReconnectLoop`
    (Task 14, adapters/sources/rtsp.py) it is a plain "wait, then act" sequencer,
    unit-testable with an injected clock and sleep: no real wall-clock time, no
    scheduler task, no eviction logic (that is `ResidentSet`'s, covered in
    test_resident_set.py)."""

    async def test_sweeps_on_the_injected_interval_using_the_injected_clock(self) -> None:
        """Fails against a sweeper that calls `sweep` immediately without sleeping, one
        that sleeps a fixed amount regardless of `interval_seconds`, or one that never
        reads the clock at all."""
        sweeps: list[float] = []
        calls = 0

        async def sweep(now: float) -> None:
            nonlocal calls
            calls += 1
            sweeps.append(now)

        clock_values = iter([10.0, 20.0, 30.0])

        def clock() -> float:
            return next(clock_values)

        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        sweeper = _IdleSweeper(sweep, interval_seconds=7.5, clock=clock, sleep=fake_sleep)
        await sweeper.run_forever(should_stop=lambda: calls >= 3)

        assert sleeps == pytest.approx([7.5, 7.5, 7.5])
        assert sweeps == pytest.approx([10.0, 20.0, 30.0])

    async def test_never_sweeps_once_should_stop_is_already_true(self) -> None:
        calls = 0

        async def sweep(_now: float) -> None:
            nonlocal calls
            calls += 1

        async def fake_sleep(_seconds: float) -> None:
            raise AssertionError("should_stop() was already true; sleep must not be called")

        sweeper = _IdleSweeper(sweep, interval_seconds=60.0, clock=lambda: 0.0, sleep=fake_sleep)
        await sweeper.run_forever(should_stop=lambda: True)
        assert calls == 0


class EndlessSource(FrameSource):
    """Frames and packets forever — i.e. every real camera.

    Every other source in the suite ends on its own, which is exactly the condition
    under which a shutdown cannot lose anything: by the time `stop()` is called there
    is nothing in flight left to lose. A camera that is still recording when the
    process is asked to exit is the ordinary case in production and the only one that
    exercises `CameraRunner`'s abandon-the-clip-but-keep-the-event path.
    """

    def __init__(self, camera_id: str, fps: float = 10.0) -> None:
        self._camera_id = camera_id
        self._fps = fps
        self.closed = False

    def _frame(self, index: int) -> FrameData:
        return FakeSource.make_frame(
            self._camera_id, index, index / self._fps, value=0 if index % 2 == 0 else 200
        )

    def __aiter__(self) -> AsyncIterator[FrameData]:
        async def frames() -> AsyncIterator[FrameData]:
            index = 0
            while True:
                yield self._frame(index)
                index += 1
                await asyncio.sleep(0)

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        async def stream() -> AsyncIterator[EncodedPacket]:
            index = 0
            while True:
                yield EncodedPacket(
                    camera_id=self._camera_id,
                    data=f"packet-{index}".encode(),
                    pts=index / self._fps,
                    is_keyframe=True,
                    codec="h264",
                )
                index += 1
                await asyncio.sleep(0)

        return stream()

    async def close(self) -> None:
        self.closed = True


class ClipOpenSignallingWriter(FakeClipWriter):
    """Announces the moment a clip starts recording, so the test can stop the service
    at a precisely known state instead of counting event-loop ticks."""

    def __init__(self) -> None:
        super().__init__()
        self.clip_open = asyncio.Event()

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> FakeClipHandle:
        handle = await super().open(camera_id, event_id, fps)
        self.clip_open.set()
        return handle


async def test_stop_publishes_the_escalation_the_runner_preserves_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §9 across the shutdown seam (C3).

    `CameraRunner.run()`'s `finally` calls `_abandon_open_clip()`, which discards the
    partial clip file and re-submits its escalation with `clip=None` precisely so the
    event is not lost. `stop()` used to cancel every task in one synchronous pass with
    the scheduler worker at index 0, so that submit reached a queue whose only consumer
    was already cancelled: the event was counted in `escalations`, never published, and
    never counted in `escalations_dropped` either, because `put_nowait` succeeded.

    Fails against that ordering with `escalations=2, published=1` — the clip-owning
    event, which is the most recent one and the one with evidence attached, is the one
    that goes missing.
    """
    from sentinel_ai.config import Settings
    from sentinel_ai.pipeline import runner as runner_module

    # Long enough that the clip is still recording when stop() arrives.
    monkeypatch.setattr(
        runner_module, "get_settings", lambda: Settings(clip_postroll_seconds=600.0)
    )

    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    registry.register(VLM_SPEC, FakeModelRuntime("qwen25vl3b", vram_mib=4400))
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    publisher = FakePublisher()
    scheduler = VlmScheduler(
        vlm=FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )
    writer = ClipOpenSignallingWriter()
    runner = CameraRunner(
        camera_id="cam-1",
        camera_label="Front Door",
        source=EndlessSource("cam-1"),
        detector=FakeDetector(script=[()]),
        tracker=FakeTracker(),
        motion=MotionAnalyzer(),
        profile=CameraProfile(camera_id="cam-1"),
        scheduler=scheduler,
        clip_writer=writer,
        preroll=PreRollBuffer(preroll_seconds=3.0),
        clock=clock,
    )
    service = EngineService(
        cameras={"cam-1": runner},
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=clock,
    )

    await service.start()
    await asyncio.wait_for(writer.clip_open.wait(), timeout=5.0)
    await service.stop()

    handle = writer.handles[0]
    assert handle.aborted is True, "test setup: the clip must still have been recording"
    assert handle.finished is False
    assert service.telemetry("cam-1").escalations_dropped == 0
    assert {event.event_id for event in publisher.events} == {handle.event_id}, (
        "the escalation the runner's shutdown path preserved must reach the publisher"
    )
    assert len(publisher.events) == service.telemetry("cam-1").escalations, (
        "every escalation the camera counted must have been published"
    )


async def test_start_wires_a_periodic_idle_sweep_that_evicts_the_idle_vlm() -> None:
    """Closes the gap Task 7 deferred: without this wiring, `ResidentSet.sweep_idle`
    (tested in isolation in test_resident_set.py) never actually runs in a live
    `EngineService`, so the 600s VLM idle-unload never fires. Fails against a
    `start()` that loads models but never schedules a sweep at all."""
    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    vlm_runtime = FakeModelRuntime("qwen25vl3b", vram_mib=4400)
    registry.register(VLM_SPEC, vlm_runtime)
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    publisher = FakePublisher()
    scheduler = VlmScheduler(
        vlm=FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )

    elapsed = {"now": 0.0}

    def ticking_clock() -> float:
        return elapsed["now"]

    async def fake_sleep(seconds: float) -> None:
        elapsed["now"] += seconds
        await asyncio.sleep(0)  # cooperative yield; zero wall-clock time

    service = EngineService(
        cameras={},
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=ticking_clock,
        idle_sweep_interval_seconds=100.0,
        sleep=fake_sleep,
    )
    await service.start()
    for _ in range(50):
        await asyncio.sleep(0)
    await service.stop()

    assert vlm_runtime.shutdown_calls >= 1
    assert resident_set.resident() == frozenset({"yolo11s"})
