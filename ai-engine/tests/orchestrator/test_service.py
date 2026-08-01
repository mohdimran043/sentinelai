from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.orchestrator.service import EngineService, UnknownCameraError, _IdleSweeper
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource
from sentinel_ai.ports.model_runtime import LifecycleState
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
from tests.fakes.io import (
    FakeClipHandle,
    FakeClipWriter,
    FakeFailedEventSink,
    FakePublisher,
    FakeSource,
)
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
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=FakeFailedEventSink(),
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
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=FakeFailedEventSink(),
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
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=FakeFailedEventSink(),
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


class LoadStateAwareVlm(VisionLanguageModel):
    """Refuses to describe while its runtime is unloaded — exactly the guard
    `Qwen25VLDescriber.describe()` opens with
    (`if self._model is None: raise RuntimeError(... called before initialize())`).

    A `FakeVisionLLM` describes happily whether or not the weights are on the card,
    which is why a composed test built on one could never have seen C1.
    """

    def __init__(self, runtime: FakeModelRuntime) -> None:
        self._runtime = runtime
        self.calls = 0

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.calls += 1
        if self._runtime.health().state is LifecycleState.UNLOADED:
            raise RuntimeError("Qwen25VLDescriber.describe called before initialize()")
        return SceneDescription(
            description="A person is standing near the door.",
            threat_value=0.3,
            suggested_action="Monitor.",
        )


def an_escalation_request() -> EscalationRequest:
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=12.5,
        detections=(),
        tracks=(),
        motion_energy=0.1,
        scene_signature=(1.0,),
    )
    return EscalationRequest(
        camera_id="cam-1",
        event_id=uuid4(),
        reason=EscalationReason.NEW_SALIENT_TRACK,
        detail="1 new salient track",
        scene=scene,
        keyframe=FakeSource.make_frame("cam-1", 0, 0.0),
        profile=CameraProfile(camera_id="cam-1"),
        camera_label="Front Door",
        history=(),
        clip=None,
    )


async def test_an_escalation_after_the_idle_unload_still_gets_a_real_description() -> None:
    """C1, at the level it actually bites: the composed system.

    The test above proves half a cycle — that eviction happens. Nothing proved the
    model ever comes back. It did not: `VlmScheduler` held no `ResidentSet` reference
    at all, so the system described correctly for exactly ten minutes after boot and
    then published `description_unavailable=True` with a flat 0.5 threat score for
    every event thereafter, for the lifetime of the process, while `/health` reported
    the VLM as UNLOADED — the state an operator reads as *correctly idle*, not as a
    fault. 600s of quiet is the normal state of a camera watching an empty corridor,
    so this fired on essentially every deployment.

    Verified against the unwired describe path: `initialize_calls = 1`,
    `description_unavailable = True`, `threat = 0.5`,
    `description = 'new_salient_track: motion (1 new salient track)'`.
    """
    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    vlm_runtime = FakeModelRuntime("qwen25vl3b", vram_mib=4400)
    registry.register(VLM_SPEC, vlm_runtime)
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    elapsed = {"now": 0.0}

    def ticking_clock() -> float:
        return elapsed["now"]

    async def fake_sleep(seconds: float) -> None:
        elapsed["now"] += seconds
        await asyncio.sleep(0)  # cooperative yield; zero wall-clock time

    publisher = FakePublisher()
    scheduler = VlmScheduler(
        vlm=LoadStateAwareVlm(vlm_runtime),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=FakeFailedEventSink(),
        maxsize=4,
        timeout_seconds=5.0,
        clock=ticking_clock,
    )
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
    assert vlm_runtime.initialize_calls == 1
    for _ in range(50):  # let the sweeper carry the clock past idle_unload_seconds=600
        await asyncio.sleep(0)
    assert vlm_runtime.shutdown_calls == 1, "test setup: the VLM must have been evicted"
    assert service.health()["qwen25vl3b"].state.value == "unloaded"

    scheduler.submit(an_escalation_request())
    await asyncio.wait_for(scheduler.drain(), timeout=5.0)
    await service.stop()

    assert vlm_runtime.initialize_calls == 2, "the escalation must have reloaded the VLM"
    assert resident_set.resident() == frozenset({"yolo11s", "qwen25vl3b"})
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert event.description_unavailable is False, (
        "an escalation after the idle window must produce a real description"
    )
    assert event.description == "A person is standing near the door."
    assert event.threat.value == pytest.approx(0.3), (
        "the 0.5 placeholder makes every event score identically and triage useless"
    )


class DecodeErrorSource(EndlessSource):
    """Endless until a decode error kills the frame stream mid-flight.

    Not hypothetical and not a test-only shape: `FileSource` surfaces `_pump_error`
    from its demux thread, `CameraRunner.run()` deliberately re-raises it so a decode
    error is never mistaken for a clean end of stream, and a `detector.detect()` CUDA
    fault reaches the same place on a live RTSP camera. Nothing in the suite exercised
    a mid-stream source error before this.
    """

    def __init__(self, camera_id: str, fail_after: int, fps: float = 10.0) -> None:
        super().__init__(camera_id, fps)
        self._fail_after = fail_after

    def __aiter__(self) -> AsyncIterator[FrameData]:
        async def frames() -> AsyncIterator[FrameData]:
            index = 0
            while index < self._fail_after:
                yield self._frame(index)
                index += 1
                await asyncio.sleep(0)
            raise RuntimeError("decode error: corrupt frame")

        return frames()


async def test_one_cameras_decode_error_does_not_abort_the_rest_of_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2 — C3 reopened through a path the progress ledger recorded as unreachable.

    `_cancel` re-raised anything that was not a `CancelledError`, so the first camera
    task that ended in an exception aborted phases 2-4 of `stop()`. The reviewer's
    probe, two cameras with one mid-stream decode error:

        PROBE stop() raised: RuntimeError decode error: corrupt frame
        PROBE scheduler task done?  False    <- worker never cancelled, leaks
        PROBE cam-b clip aborted:   False    <- its finally never ran
        PROBE escalations cam-b: 0           <- its in-flight event never submitted

    One camera's decode error costing another camera's evidence is precisely what the
    C3 ordering exists to prevent, so this asserts all three: shutdown completes, the
    healthy camera's clip is aborted rather than leaked, and the escalation its
    `finally` preserved reaches the publisher.

    Fails against the propagating `_cancel` at the very first line — `stop()` raises
    `RuntimeError: decode error: corrupt frame` out of the test.
    """
    from sentinel_ai.config import Settings
    from sentinel_ai.pipeline import runner as runner_module

    # Long enough that cam-b's clip is still recording when stop() arrives.
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
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=FakeFailedEventSink(),
        maxsize=8,
        timeout_seconds=5.0,
        clock=clock,
    )

    def build_runner(
        camera_id: str, source: FrameSource, writer: FakeClipWriter | None
    ) -> CameraRunner:
        return CameraRunner(
            camera_id=camera_id,
            camera_label=camera_id,
            source=source,
            detector=FakeDetector(script=[()]),
            tracker=FakeTracker(),
            motion=MotionAnalyzer(),
            profile=CameraProfile(camera_id=camera_id),
            scheduler=scheduler,
            clip_writer=writer,
            preroll=PreRollBuffer(preroll_seconds=3.0),
        )

    writer_b = ClipOpenSignallingWriter()
    service = EngineService(
        # cam-a first: `_cancel` awaits in this order, so the failing camera is the
        # one that used to abort the loop before cam-b was ever awaited.
        cameras={
            "cam-a": build_runner("cam-a", DecodeErrorSource("cam-a", fail_after=4), None),
            "cam-b": build_runner("cam-b", EndlessSource("cam-b"), writer_b),
        },
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=clock,
    )

    await service.start()
    scheduler_task = service._scheduler_task
    cam_a_task = service._camera_tasks[0]
    await asyncio.wait_for(writer_b.clip_open.wait(), timeout=5.0)
    await asyncio.wait({cam_a_task})
    assert isinstance(cam_a_task.exception(), RuntimeError), (
        "test setup: cam-a must have died of its decode error"
    )

    await service.stop()

    assert scheduler_task is not None and scheduler_task.done(), (
        "phase 4 must still run: an uncancelled scheduler worker is a leaked task"
    )
    handle_b = writer_b.handles[0]
    assert handle_b.aborted is True, "cam-b's finally must still run and abort its clip"
    assert handle_b.finished is False
    assert service.telemetry("cam-b").escalations_dropped == 0
    assert handle_b.event_id in {event.event_id for event in publisher.events}, (
        "cam-b's preserved escalation must still be published (spec §9)"
    )


class NeverReturningVlm(VisionLanguageModel):
    """A describe that starts and never finishes — a wedged VLM, or simply a real
    multi-second Qwen generate that the drain cap runs out of patience for."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


async def test_a_drain_that_runs_out_of_budget_dead_letters_instead_of_losing_events() -> None:
    """B5: the bounded drain reproduced C3 through its own cap.

    `stop()` wrapped `drain()` in `suppress(TimeoutError)` and then cancelled the
    worker, so everything still queued — and the request the worker was midway through
    describing, which is no longer on the queue at all — simply evaporated. The
    reviewer's probe with `shutdown_drain_timeout_seconds=0.2` against a slower
    describe:

        escalations counted : 1     published : 0
        escalations_dropped : 0     dead-lettered : 0
        scheduler.dropped   : 0     publish_failures : 0
        warnings/errors     : []

    Counted, never published, never counted as dropped, invisible to every telemetry
    read. Reachable with the shipped numbers: a 10s cap against `vlm_timeout_seconds`
    30s and a queue depth of four real Qwen describes each spaced by
    `vlm_global_min_interval_seconds`.

    The cap is pinned at 0.0 here rather than 0.2: `asyncio.wait_for` short-circuits a
    non-positive timeout without sleeping, so this costs no wall-clock time at all and
    cannot flake on a loaded CI box. It is the same expiry branch either way.

    Fails against the suppressing version with `abandoned == 0` and an empty
    dead-letter spool.
    """
    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    registry.register(VLM_SPEC, FakeModelRuntime("qwen25vl3b", vram_mib=4400))
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    publisher = FakePublisher()
    dead_letter = FakeFailedEventSink()
    vlm = NeverReturningVlm()
    scheduler = VlmScheduler(
        vlm=vlm,
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        resident_set=resident_set,
        vlm_model_key="qwen25vl3b",
        dead_letter=dead_letter,
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )
    service = EngineService(
        cameras={},
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=clock,
        shutdown_drain_timeout_seconds=0.0,
    )
    await service.start()

    in_flight = an_escalation_request()
    handle = FakeClipHandle("cam-1", uuid4())
    still_queued = replace(an_escalation_request(), clip=handle)
    assert scheduler.submit(in_flight)
    assert scheduler.submit(still_queued)
    await asyncio.wait_for(vlm.started.wait(), timeout=5.0)

    await service.stop()

    assert publisher.events == [], "test setup: the wedged describe must publish nothing"
    assert scheduler.abandoned == 2, (
        "both the mid-describe request and the queued one must be accounted for"
    )
    assert {event.event_id for event in dead_letter.events} == {
        in_flight.event_id,
        still_queued.event_id,
    }, "an escalation the drain cap cut short must reach the dead-letter sink (spec §9)"
    assert all(event.description_unavailable for event in dead_letter.events), (
        "no VLM ever spoke for these, and the event must say so"
    )
    assert handle.aborted is True, (
        "a clip riding on an abandoned request leaks a PyAV container and a temp file"
    )


class TestStopUnderExternalCancellation:
    """`stop()` must not silently skip its remaining phases when *it* is cancelled.

    The ordering in `stop()` exists so a camera's `finally` can submit its preserved
    escalation while the scheduler worker is still alive. A blanket
    `suppress(CancelledError)` around `await task` cannot tell "the task I awaited
    ended via cancellation" from "my own stop() was cancelled while awaiting it" —
    and swallowing the second lets stop() march on and cancel the scheduler while a
    camera's cleanup is still running unawaited, dropping exactly the event the
    ordering protects. Reachable in production: uvicorn's
    --timeout-graceful-shutdown cancels the ASGI lifespan's shutdown task.
    """

    async def test_cancelling_stop_propagates_instead_of_being_swallowed(self) -> None:
        started = asyncio.Event()

        async def lingering_cleanup() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Cleanup that outlives its own cancellation, like CameraRunner's
                # finally. Bounded yields, so the test cannot hang if the guard
                # regresses — it fails instead.
                for _ in range(20):
                    await asyncio.sleep(0)
                raise

        registry = ModelRegistry()
        service = EngineService(
            cameras={},
            registry=registry,
            resident_set=ResidentSet(registry, total_mib=8192, reserved_mib=2048),
            scheduler=None,  # type: ignore[arg-type]
            required_model_keys=(),
            clock=lambda: 0.0,
        )
        camera_task = asyncio.create_task(lingering_cleanup())
        service._camera_tasks = [camera_task]
        await started.wait()

        stop_task = asyncio.create_task(service.stop())
        await asyncio.sleep(0)
        stop_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await stop_task

        with contextlib.suppress(asyncio.CancelledError):
            await camera_task
