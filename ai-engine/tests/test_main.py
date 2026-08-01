"""The composition root (I1).

The point of these tests is that the assembled system can be *built and run*. Phase
1B's three Criticals were all wiring-level and all invisible to a suite in which
nothing composed more than two layers; the last test here composes the real
`EngineService`, `VlmScheduler`, `ResidentSet`, `CameraRunner`, `FileSource` and
`RabbitMQPublisher` through `main.compose` and drives them through a full
start/stop, with fakes only where CI has no GPU.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI

from sentinel_ai import main
from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector
from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.adapters.serialization.event_codec import validate_payload
from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.rtsp import RtspSource
from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber
from sentinel_ai.api.routes import EngineServiceProtocol
from sentinel_ai.config import Settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, Detection, EscalationReason, Event, ThreatScore
from sentinel_ai.main import (
    BrokerLink,
    CameraConfig,
    CameraConfigError,
    ComposedService,
    Composition,
    Models,
    build_source,
    compose,
    load_cameras,
    refresh_specs_from_capabilities,
)
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.service import UnknownCameraError
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.ports.model_runtime import ModelRuntime
from tests.fakes.models import FakeDetector, FakeModelRuntime, FakeVisionLLM

ASSET = Path(__file__).resolve().parent / "assets" / "synthetic_clip.mp4"

DETECTOR_KEY = "yolo11s.pt"
VLM_KEY = "qwen25vl3b"


def _statically_satisfies_the_api_protocol() -> None:
    """`create_app` accepts anything shaped like `EngineServiceProtocol`; this pins
    that `ComposedService` is shaped like one, checked by mypy rather than at runtime.

    The real adapters must likewise be `ModelRuntime`s or `ModelRegistry.register`
    would not take them — `main` imports them for exactly that.
    """
    service: EngineServiceProtocol = ComposedService(lambda: pytest.fail("not called"))
    detector: type[ModelRuntime] = Yolo11Detector
    vlm: type[ModelRuntime] = Qwen25VLDescriber
    del service, detector, vlm


def write_cameras(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_settings_default_camera_file_and_the_example_file_agree() -> None:
    """The shipped example must be loadable by the shipped default reader — a sample
    config that the loader rejects is worse than none."""
    example = Path(__file__).resolve().parents[1] / "cameras.example.json"
    cameras = load_cameras(example)
    assert [camera.camera_id for camera in cameras] == ["avenue_01", "replay_01"]
    assert cameras[1].profile.cooldown_seconds == 5.0


class TestCameraConfig:
    def test_a_minimal_entry_defaults_its_label_and_profile(self, tmp_path: Path) -> None:
        path = write_cameras(tmp_path, {"cameras": [{"id": "cam-1", "url": "rtsp://host/one"}]})
        (camera,) = load_cameras(path)
        assert camera == CameraConfig(
            camera_id="cam-1",
            label="cam-1",
            url="rtsp://host/one",
            profile=camera.profile,
        )
        assert camera.profile.camera_id == "cam-1"
        assert camera.profile.cooldown_seconds == 5.0  # the CameraProfile default

    def test_a_profile_override_reaches_the_profile(self, tmp_path: Path) -> None:
        path = write_cameras(
            tmp_path,
            {
                "cameras": [
                    {
                        "id": "cam-1",
                        "url": "rtsp://host/one",
                        "label": "Front door",
                        "profile": {"cooldown_seconds": 12.5, "salient_classes": ["person"]},
                    }
                ]
            },
        )
        (camera,) = load_cameras(path)
        assert camera.label == "Front door"
        assert camera.profile.cooldown_seconds == 12.5
        assert camera.profile.salient_classes == frozenset({"person"})

    def test_a_missing_file_names_itself_and_the_way_out(self, tmp_path: Path) -> None:
        with pytest.raises(CameraConfigError, match=r"cameras\.example\.json"):
            load_cameras(tmp_path / "absent.json")

    def test_invalid_json_is_a_config_error_not_a_traceback(self, tmp_path: Path) -> None:
        path = tmp_path / "cameras.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(CameraConfigError, match="not valid JSON"):
            load_cameras(path)

    def test_an_empty_camera_list_is_rejected(self, tmp_path: Path) -> None:
        """A process with no cameras starts, serves /health and watches nothing."""
        with pytest.raises(CameraConfigError, match="configures no cameras"):
            load_cameras(write_cameras(tmp_path, {"cameras": []}))

    def test_a_duplicate_id_is_rejected_rather_than_silently_replacing(
        self, tmp_path: Path
    ) -> None:
        """`EngineService` keys cameras by id, so the second entry would win and the
        first camera would never be watched — with no error anywhere."""
        path = write_cameras(
            tmp_path,
            {"cameras": [{"id": "cam-1", "url": "a"}, {"id": "cam-1", "url": "b"}]},
        )
        with pytest.raises(CameraConfigError, match="duplicate camera id"):
            load_cameras(path)

    def test_a_misspelled_profile_field_fails_at_startup(self, tmp_path: Path) -> None:
        """Otherwise the typo is accepted, ignored, and the camera runs on defaults."""
        path = write_cameras(
            tmp_path,
            {"cameras": [{"id": "cam-1", "url": "a", "profile": {"cooldown_second": 1.0}}]},
        )
        with pytest.raises(CameraConfigError, match="unknown profile field"):
            load_cameras(path)

    def test_a_profile_violating_its_own_invariants_fails_at_startup(self, tmp_path: Path) -> None:
        path = write_cameras(
            tmp_path,
            {"cameras": [{"id": "cam-1", "url": "a", "profile": {"cooldown_seconds": -1.0}}]},
        )
        with pytest.raises(CameraConfigError, match="invalid profile"):
            load_cameras(path)

    @pytest.mark.parametrize("missing", ["id", "url"])
    def test_the_two_required_keys_are_required(self, tmp_path: Path, missing: str) -> None:
        entry = {"id": "cam-1", "url": "rtsp://host/one"}
        del entry[missing]
        with pytest.raises(CameraConfigError, match=f"'{missing}'"):
            load_cameras(write_cameras(tmp_path, {"cameras": [entry]}))


class TestSourceSelection:
    """The scheme in the URL is the whole switch between a live camera and a replay."""

    @pytest.mark.parametrize("url", ["rtsp://host/one", "rtsps://host/one"])
    def test_an_rtsp_url_builds_an_rtsp_source_with_the_configured_backoff(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        """Constructed through a stub rather than for real: `RtspSource.__init__`
        starts a PyAV demux against the URL immediately, which CI must never do."""
        calls: list[dict[str, object]] = []

        class StubRtspSource:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        monkeypatch.setattr(main, "RtspSource", StubRtspSource)
        settings = Settings(rtsp_reconnect_initial_seconds=2.0, rtsp_reconnect_max_seconds=45.0)
        build_source(CameraConfig("cam-1", "Cam 1", url, _profile("cam-1")), settings)

        assert calls == [
            {
                "camera_id": "cam-1",
                "url": url,
                "reconnect_initial_seconds": 2.0,
                "reconnect_max_seconds": 45.0,
            }
        ]

    def test_anything_else_is_taken_as_a_file_path(self) -> None:
        config = CameraConfig("cam-1", "Cam 1", str(ASSET), _profile("cam-1"))
        source = build_source(config, Settings(source_realtime=False))
        assert isinstance(source, FileSource)

    def test_the_real_rtsp_source_still_matches_the_stub_signature(self) -> None:
        """Keeps the stub above honest: a renamed constructor keyword would otherwise
        make that test pass while production startup raised TypeError."""
        parameters = set(inspect.signature(RtspSource.__init__).parameters)
        assert {
            "camera_id",
            "url",
            "reconnect_initial_seconds",
            "reconnect_max_seconds",
        } <= parameters


def _profile(camera_id: str) -> CameraProfile:
    return CameraProfile(camera_id=camera_id)


class TestMeasuredVramReachesThePlanner:
    """Before the composition root, `ModelSpec` was never constructed outside tests, so
    the whole measured-VRAM correction chain (68 -> 432 MiB for the detector, ~2766 for
    the VLM) terminated in a number only `/health` ever read."""

    def test_the_measured_figure_replaces_the_configured_estimate(self) -> None:
        registry = ModelRegistry()
        runtime = FakeModelRuntime(DETECTOR_KEY, vram_mib=432)
        registry.register(
            ModelSpec(model_key=DETECTOR_KEY, vram_mib=68, priority=100, idle_unload_seconds=None),
            runtime,
        )
        assert registry.specs()[0].vram_mib == 68

        changed = refresh_specs_from_capabilities(registry)

        assert changed == {DETECTOR_KEY: 432}
        assert registry.specs()[0].vram_mib == 432, "plan_residency reads ModelSpec, not caps"

    def test_the_rest_of_the_spec_is_preserved(self) -> None:
        registry = ModelRegistry()
        registry.register(
            ModelSpec(model_key=VLM_KEY, vram_mib=4400, priority=50, idle_unload_seconds=600.0),
            FakeModelRuntime(VLM_KEY, vram_mib=2766),
        )
        refresh_specs_from_capabilities(registry)
        (spec,) = registry.specs()
        assert (spec.priority, spec.idle_unload_seconds) == (50, 600.0)

    def test_an_unmeasured_model_keeps_its_configured_estimate(self) -> None:
        """`capabilities().vram_mib` is 0 until warmup() has run. Trusting that zero
        would tell the planner the model is free and let it admit anything."""
        registry = ModelRegistry()
        registry.register(
            ModelSpec(model_key=VLM_KEY, vram_mib=2766, priority=50, idle_unload_seconds=600.0),
            FakeModelRuntime(VLM_KEY, vram_mib=0),
        )
        assert refresh_specs_from_capabilities(registry) == {}
        assert registry.specs()[0].vram_mib == 2766


class RecordingBroker:
    """A `BrokerConnection` whose link can be brought up and down under the test."""

    def __init__(self, *, up: bool = True, connect_error: Exception | None = None) -> None:
        self.up = up
        self.connect_error = connect_error
        self.connect_calls = 0
        self.replay_calls = 0
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.connect_calls += 1
        if not self.up:
            raise ConnectionError("broker unreachable")
        if self.connect_error is not None:
            raise self.connect_error
        self._connected = True

    async def replay_spool(self) -> None:
        self.replay_calls += 1
        if not self.up:
            raise ConnectionError("broker went away")


class TestBrokerLink:
    """Spec §9's spool is only half a guarantee without a component that drains it.
    Before this class existed, `connect()` and `replay_spool()` had zero production
    callers and every spooled event stayed on disk forever."""

    @staticmethod
    def _link(broker: RecordingBroker, sleeps: list[float]) -> BrokerLink:
        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) >= 5:
                raise asyncio.CancelledError

        return BrokerLink(
            broker,
            initial_seconds=1.0,
            max_seconds=8.0,
            replay_interval_seconds=30.0,
            sleep=sleep,
        )

    async def test_a_healthy_link_connects_once_and_keeps_draining_the_spool(self) -> None:
        broker = RecordingBroker()
        sleeps: list[float] = []
        with pytest.raises(asyncio.CancelledError):
            await self._link(broker, sleeps).run_forever(should_stop=lambda: False)

        assert broker.connect_calls == 1, "connect is not re-attempted on a live link"
        assert broker.replay_calls == 5, "the spool is re-drained every interval"
        assert sleeps == [30.0] * 5

    async def test_a_down_broker_backs_off_exponentially_to_the_cap(self) -> None:
        broker = RecordingBroker(up=False)
        sleeps: list[float] = []
        with pytest.raises(asyncio.CancelledError):
            await self._link(broker, sleeps).run_forever(should_stop=lambda: False)

        assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]

    async def test_the_backoff_resets_once_the_broker_returns(self) -> None:
        broker = RecordingBroker(up=False)
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            broker.up = True
            if len(sleeps) >= 3:
                raise asyncio.CancelledError

        link = BrokerLink(
            broker,
            initial_seconds=1.0,
            max_seconds=8.0,
            replay_interval_seconds=30.0,
            sleep=sleep,
        )
        with pytest.raises(asyncio.CancelledError):
            await link.run_forever(should_stop=lambda: False)

        assert sleeps == [1.0, 30.0, 30.0]
        assert broker.replay_calls >= 1

    async def test_step_never_raises_so_the_task_outlives_a_dead_broker(self) -> None:
        """A broker that is down is the ordinary case; it must not kill the only
        component that will replay the spool when it comes back."""
        broker = RecordingBroker(up=False)
        link = BrokerLink(
            broker, initial_seconds=1.0, max_seconds=8.0, replay_interval_seconds=30.0
        )
        assert await link.step() is False
        broker.up = True
        assert await link.step() is True

    async def test_a_spooled_event_is_replayed_when_the_link_comes_up(self, tmp_path: Path) -> None:
        """End to end over the real publisher: the file on disk is gone afterwards
        because something finally called `replay_spool()`."""
        publisher = RabbitMQPublisher(
            url="amqp://unused", exchange="sentinel.events", spool_dir=tmp_path / "spool"
        )
        await publisher.publish(_event())  # no connect() -> straight to the spool
        spooled = sorted((tmp_path / "spool").glob("*.json"))
        assert len(spooled) == 1

        published: list[str] = []

        class ConnectableStub(RecordingBroker):
            async def replay_spool(self) -> None:
                self.replay_calls += 1
                for path in sorted((tmp_path / "spool").glob("*.json")):
                    published.append(json.loads(path.read_text(encoding="utf-8"))["event_id"])
                    path.unlink()

        link = BrokerLink(
            ConnectableStub(),
            initial_seconds=1.0,
            max_seconds=8.0,
            replay_interval_seconds=30.0,
        )
        assert await link.step() is True
        assert published and not list((tmp_path / "spool").glob("*.json"))


def _event() -> Event:
    return Event(
        event_id=uuid4(),
        camera_id="cam-1",
        occurred_at=1.0,
        reason=EscalationReason.NEW_SALIENT_TRACK,
        threat=ThreatScore.from_value(0.4),
        description="A person is near the door.",
        suggested_action="Monitor.",
        labels=("person",),
        track_ids=(1,),
    )


# -- the composed engine ------------------------------------------------------------


def fake_models() -> Models:
    """The production `Models`, with CPU stand-ins for the two GPU adapters.

    In production `detector`/`vlm` and the registry's runtimes are the same objects
    (`Yolo11Detector` and `Qwen25VLDescriber` each implement both interfaces); the
    dataclass keeps them as separate fields precisely so CI can substitute here.
    """
    registry = ModelRegistry()
    registry.register(
        ModelSpec(model_key=DETECTOR_KEY, vram_mib=432, priority=100, idle_unload_seconds=None),
        FakeModelRuntime(DETECTOR_KEY, kind="detector", vram_mib=432),
    )
    registry.register(
        ModelSpec(model_key=VLM_KEY, vram_mib=2766, priority=50, idle_unload_seconds=600.0),
        FakeModelRuntime(VLM_KEY, vram_mib=2766),
    )
    box = BBox(x1=10.0, y1=10.0, x2=60.0, y2=120.0)
    return Models(
        detector_key=DETECTOR_KEY,
        detector=FakeDetector([(Detection(label="person", confidence=0.9, box=box),)]),
        vlm_key=VLM_KEY,
        vlm=FakeVisionLLM(),
        registry=registry,
    )


def composed(
    tmp_path: Path, *, broker: BrokerLink | None = None, **overrides: object
) -> tuple[Composition, Settings]:
    settings = Settings(
        source_realtime=False,
        event_spool_dir=str(tmp_path / "spool"),
        clip_temp_dir=str(tmp_path / "clips"),
        vlm_global_min_interval_seconds=0.0,
        **overrides,  # type: ignore[arg-type]
    )
    cameras = (CameraConfig("cam-1", "Camera One", str(ASSET), _profile("cam-1")),)
    composition = compose(
        settings,
        cameras,
        fake_models(),
        main.build_publisher(settings),
        None,
        main.build_dead_letter(settings),
    )
    # The one thing CI may not have: a broker. `compose` builds a real `BrokerLink`
    # over the real publisher (asserted structurally in `TestCompose`); here it is
    # swapped for one whose link is permanently down, which is also the state that
    # makes the publisher spool — the property the end-to-end test below asserts.
    return replace(composition, broker=broker or _offline_broker_link()), settings


def _offline_broker_link() -> BrokerLink:
    # One failing attempt, then parked on a backoff longer than any test: the task
    # exists (so `stop()` really does cancel a live one) without spinning.
    return BrokerLink(
        RecordingBroker(up=False),
        initial_seconds=3600.0,
        max_seconds=3600.0,
        replay_interval_seconds=3600.0,
    )


class StubbornBrokerLink(BrokerLink):
    """A link whose task refuses to finish cancelling.

    That makes `stop()`'s wait for it the *only* interruptible await left, which is
    what lets the test deliver a cancellation at exactly that point. It is not a
    contrived shape: uvicorn sends the graceful-shutdown cancellation and then, on a
    second signal, cancels again — so a second cancellation genuinely can land while
    `stop()` is already unwinding.
    """

    def __init__(self) -> None:
        super().__init__(
            RecordingBroker(up=False),
            initial_seconds=3600.0,
            max_seconds=3600.0,
            replay_interval_seconds=3600.0,
        )
        self.stepped = asyncio.Event()
        self._release = asyncio.Event()

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        try:
            await self._release.wait()
        except asyncio.CancelledError:
            # Deliberately ignores it, so the awaiting `stop()` stays parked.
            await self._release.wait()

    async def step(self) -> bool:
        self.stepped.set()
        return True

    def release(self) -> None:
        self._release.set()


class TestCompose:
    def test_every_configured_camera_becomes_a_runner(self, tmp_path: Path) -> None:
        composition, _ = composed(tmp_path)
        assert [t.camera_id for t in composition.service.cameras()] == ["cam-1"]

    def test_the_scheduler_is_wired_to_the_same_resident_set_the_service_owns(
        self, tmp_path: Path
    ) -> None:
        """C1's fix is only real if the scheduler's `ensure()` and the idle sweeper's
        act on one set of bookkeeping. Two `ResidentSet`s would each believe they owned
        the card."""
        composition, _ = composed(tmp_path)
        service_set = composition.service._resident_set
        scheduler_set = composition.service._scheduler._resident_set
        assert service_set is scheduler_set

    def test_the_admission_gate_and_the_engine_run_on_real_elapsed_time(
        self, tmp_path: Path
    ) -> None:
        """I6: `AdmissionGate` spaces admissions with `asyncio.sleep`, so its clock must
        advance with the loop. The camera pipeline's timeline is a different thing and
        `CameraRunner` takes no clock at all (C2) — assert that too, structurally."""
        composition, _ = composed(tmp_path)
        assert composition.service._clock is time.monotonic
        assert composition.service._scheduler._clock is time.monotonic
        assert "clock" not in inspect.signature(CameraRunner.__init__).parameters

    def test_the_broker_link_is_built_over_the_real_publisher(self, tmp_path: Path) -> None:
        """`composed()` below swaps the link out so CI never touches a broker; this is
        what keeps that substitution honest about what production wires."""
        settings = Settings(event_spool_dir=str(tmp_path / "spool"))
        publisher = main.build_publisher(settings)
        composition = compose(
            settings,
            (CameraConfig("cam-1", "Camera One", str(ASSET), _profile("cam-1")),),
            fake_models(),
            publisher,
            None,
            main.build_dead_letter(settings),
        )
        assert composition.publisher is publisher
        assert composition.broker._publisher is publisher
        assert isinstance(composition.broker, BrokerLink)


class TestComposedService:
    async def test_the_read_endpoints_report_an_uncomposed_engine_honestly(self) -> None:
        service = ComposedService(lambda: pytest.fail("must not build before start()"))
        assert service.cameras() == ()
        assert service.health() == {}
        with pytest.raises(UnknownCameraError):
            service.telemetry("cam-1")
        with pytest.raises(UnknownCameraError):
            await service.describe_now("cam-1")

    async def test_start_composes_then_starts_and_stop_closes_the_publisher(
        self, tmp_path: Path
    ) -> None:
        composition, _ = composed(tmp_path)
        service = ComposedService(lambda: composition)

        await service.start()
        try:
            assert [t.camera_id for t in service.cameras()] == ["cam-1"]
            assert set(service.health()) == {DETECTOR_KEY, VLM_KEY}
        finally:
            await service.stop()

        assert not composition.publisher.connected
        assert service.cameras() == ()

    async def test_a_cancellation_landing_in_the_teardown_is_not_swallowed(
        self, tmp_path: Path
    ) -> None:
        """`create_app`'s lifespan awaits `stop()` in exactly the position uvicorn's
        `--timeout-graceful-shutdown` cancels, and a second signal cancels again while
        that shutdown is already unwinding. `EngineService._cancel` was fixed to
        re-raise rather than march on when the cancellation is aimed at *it*; this is
        the caller that makes that reachable in production, and it must not reintroduce
        the same defect one layer out.

        The cancellation is delivered while `stop()` is waiting for the broker task,
        which is the one await in the teardown. Against a teardown that wraps that wait
        in a blanket `suppress(CancelledError)` the cancellation is swallowed, `stop()`
        runs on and returns normally, and the lifespan reports a truncated shutdown to
        uvicorn as a completed one: `stop_task.cancelled()` is False and no
        `CancelledError` is raised.

        Note the narrower case — a cancellation landing in `service.stop()` — does
        *not* discriminate: a `try/finally` re-raises it regardless of what the
        `finally` suppresses. Only a cancellation delivered inside the suppressed await
        itself tells the two apart, which is why the link below refuses to finish
        cancelling.
        """
        link = StubbornBrokerLink()
        composition, _ = composed(tmp_path, broker=link)
        service = ComposedService(lambda: composition)
        await service.start()
        try:
            stop_task = asyncio.create_task(service.stop())
            # `step()` is the last call before the teardown and does not await, so when
            # this returns `stop_task` has already suspended on the wait that follows.
            async with asyncio.timeout(10.0):
                await link.stepped.wait()

            stop_task.cancel()
            # `asyncio.wait` rather than `await stop_task`: bounded, and it separates
            # the two ways this can go wrong. A teardown that suppresses the
            # cancellation and then parks on the same never-finishing task hangs; one
            # that suppresses it and completes returns normally. Both are failures, and
            # they deserve different messages.
            _done, pending = await asyncio.wait({stop_task}, timeout=5.0)
            assert not pending, "stop() hung in its teardown instead of propagating"
            assert stop_task.cancelled(), "a shutdown cut short must not look like a clean one"
        finally:
            link.release()

    async def test_start_feeds_the_measured_vram_back_into_the_specs(self, tmp_path: Path) -> None:
        composition, _ = composed(tmp_path)
        # A runtime that only reports its VRAM once warmed up, exactly as the real
        # adapters do — `capabilities().vram_mib` is 0 until `warmup()` has run.
        composition.registry.register(
            ModelSpec(model_key=VLM_KEY, vram_mib=2766, priority=50, idle_unload_seconds=600.0),
            WarmupMeasuringRuntime(VLM_KEY, measured_mib=3100),
        )
        service = ComposedService(lambda: composition)
        await service.start()
        try:
            specs = {spec.model_key: spec.vram_mib for spec in composition.registry.specs()}
            assert specs[VLM_KEY] == 3100
        finally:
            await service.stop()

    def test_the_module_level_app_is_a_fastapi_app_with_the_four_endpoints(self) -> None:
        """`uvicorn sentinel_ai.main:app` needs something to serve, and importing this
        module must not read a camera file, open a socket or touch the GPU."""
        assert isinstance(main.app, FastAPI)
        paths = set(main.app.openapi()["paths"])
        assert {
            "/health",
            "/cameras",
            "/cameras/{camera_id}/telemetry",
            "/cameras/{camera_id}/describe",
        } <= paths


class WarmupMeasuringRuntime(FakeModelRuntime):
    def __init__(self, model_key: str, measured_mib: int) -> None:
        super().__init__(model_key, vram_mib=0)
        self._measured_mib = measured_mib

    async def warmup(self) -> None:
        await super().warmup()
        self._vram_mib = self._measured_mib


class TestTheAssembledSystemRuns:
    async def test_a_full_start_to_stop_pass_publishes_through_the_real_publisher(
        self, tmp_path: Path
    ) -> None:
        """The test this branch did not have: real `EngineService`, `VlmScheduler`,
        `AdmissionGate`, `ResidentSet`, `CameraRunner`, `MotionAnalyzer`,
        `PreRollBuffer`, `FileSource` over a committed asset and a real
        `RabbitMQPublisher`, assembled by production code rather than a test fixture.

        With no broker reachable the publisher spools, which is exactly what spec §9
        promises — so the assertion is that a schema-valid payload lands on disk.
        """
        composition, settings = composed(tmp_path)
        service = ComposedService(lambda: composition)

        await service.start()
        try:
            # Driven by the camera's own end of stream, not a tick count.
            async with asyncio.timeout(20.0):
                while service.cameras()[0].frames_seen < 50:
                    await asyncio.sleep(0.01)
            telemetry = service.cameras()[0]
        finally:
            await service.stop()

        assert telemetry.frames_seen == 50, "the whole 50-frame asset was decoded"
        assert telemetry.escalations >= 1, "a tracked person must escalate"

        spooled = sorted(Path(settings.event_spool_dir).glob("*.json"))
        assert len(spooled) == telemetry.escalations, "no escalation may go unpublished"
        for path in spooled:
            validate_payload(json.loads(path.read_text(encoding="utf-8")))
