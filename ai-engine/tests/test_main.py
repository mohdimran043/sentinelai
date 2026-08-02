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
from sentinel_ai.domain.zone import Zone
from sentinel_ai.main import (
    BrokerLink,
    CameraConfig,
    CameraConfigError,
    CameraEdit,
    ComposedService,
    Composition,
    Models,
    build_source,
    compose,
    load_cameras,
    refresh_specs_from_capabilities,
)
from sentinel_ai.orchestrator.event_history import RECENT_EVENTS_PER_CAMERA
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
    assert [camera.camera_id for camera in cameras] == ["avenue_01", "demo_live", "replay_01"]
    assert cameras[2].profile.cooldown_seconds == 5.0


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


class TestCameraZone:
    """T1. A zone groups cameras; its absence groups nothing and breaks nothing."""

    def test_a_zone_reaches_the_camera_record_as_the_enum(self, tmp_path: Path) -> None:
        path = write_cameras(
            tmp_path,
            {"cameras": [{"id": "cam-1", "url": "rtsp://host/one", "zone": "corridor"}]},
        )
        (camera,) = load_cameras(path)
        assert camera.zone is Zone.CORRIDOR

    def test_a_camera_without_a_zone_is_ungrouped_not_a_config_error(self, tmp_path: Path) -> None:
        """Every camera file written before zones existed must keep loading. An
        ungrouped camera is a camera nobody has grouped yet, which is a fact about the
        deployment; the loader's fail-loud is for input it genuinely cannot build."""
        path = write_cameras(tmp_path, {"cameras": [{"id": "cam-1", "url": "rtsp://host/one"}]})
        (camera,) = load_cameras(path)
        assert camera.zone is None

    def test_an_explicit_null_zone_is_the_same_as_omitting_it(self, tmp_path: Path) -> None:
        """`GET /cameras` renders an ungrouped camera as `"zone": null`; a config that
        is round-tripped through that shape must load again."""
        path = write_cameras(
            tmp_path, {"cameras": [{"id": "cam-1", "url": "rtsp://host/one", "zone": None}]}
        )
        (camera,) = load_cameras(path)
        assert camera.zone is None

    def test_an_unknown_zone_fails_loud_and_names_the_vocabulary(self, tmp_path: Path) -> None:
        """The crisp distinction: *absent* is ungrouped, *wrong* is unbuildable. An
        operator who typed `hallway` meant to group that camera and did not; silently
        dropping it to ungrouped hides the typo behind a plausible-looking console."""
        path = write_cameras(
            tmp_path, {"cameras": [{"id": "cam-1", "url": "rtsp://host/one", "zone": "hallway"}]}
        )
        with pytest.raises(CameraConfigError, match=r"unknown zone 'hallway'.*corridor"):
            load_cameras(path)

    def test_a_non_string_zone_fails_loud(self, tmp_path: Path) -> None:
        path = write_cameras(
            tmp_path, {"cameras": [{"id": "cam-1", "url": "rtsp://host/one", "zone": 3}]}
        )
        with pytest.raises(CameraConfigError, match="zone"):
            load_cameras(path)

    def test_the_shipped_example_shows_both_a_grouped_and_an_ungrouped_camera(self) -> None:
        """The example is the documentation an operator copies. It has to show that the
        field is optional as well as what it looks like when set."""
        example = Path(__file__).resolve().parents[1] / "cameras.example.json"
        zones = [camera.zone for camera in load_cameras(example)]
        assert Zone.CORRIDOR in zones
        assert None in zones


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

    def test_a_configured_zone_survives_composition_and_reaches_the_api(
        self, tmp_path: Path
    ) -> None:
        """T1's whole point. A zone in the file that stops at `CameraConfig` groups
        nothing: `GET /cameras` is built from `CameraRunner.telemetry()`, so the value
        has to be carried through composition to be visible at all."""
        settings = Settings(
            source_realtime=False,
            event_spool_dir=str(tmp_path / "spool"),
            clip_temp_dir=str(tmp_path / "clips"),
        )
        cameras = (
            CameraConfig("cam-1", "Camera One", str(ASSET), _profile("cam-1"), Zone.DAYROOM),
            CameraConfig("cam-2", "Camera Two", str(ASSET), _profile("cam-2")),
        )
        composition = compose(
            settings,
            cameras,
            fake_models(),
            main.build_publisher(settings),
            None,
            main.build_dead_letter(settings),
        )
        assert [t.zone for t in composition.service.cameras()] == [Zone.DAYROOM, None]

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
            # Not an empty history: before `start()` there are no cameras at all, and
            # an empty list would tell a console the camera exists and has been quiet.
            service.event_history("cam-1")
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

    def test_the_module_level_app_is_a_fastapi_app_with_the_read_and_write_endpoints(
        self,
    ) -> None:
        """`uvicorn sentinel_ai.main:app` needs something to serve, and importing this
        module must not read a camera file, open a socket or touch the GPU."""
        assert isinstance(main.app, FastAPI)
        paths = set(main.app.openapi()["paths"])
        assert {
            "/health",
            "/cameras",
            "/cameras/{camera_id}/telemetry",
            "/cameras/{camera_id}/events",
            "/cameras/{camera_id}/describe",
        } <= paths


class TestTheComposedEditPath:
    """`ComposedService.update_camera`: the file and the running camera, together.

    This is the seam the unit tests on either side cannot see. `CameraFileStore`
    knows nothing about runners and `EngineService` knows nothing about the file;
    the whole class of defect left over is "the write happened and the camera did
    not" (or the reverse), and only a test holding both can catch it. That is the
    exact shape of the three wiring-level Criticals this module exists to prevent.
    """

    @staticmethod
    def _service(tmp_path: Path, *entries: object) -> tuple[ComposedService, Path, Composition]:
        # The default entry mirrors what `composed()` puts in memory, label included:
        # a fixture whose file and runners already disagree could not tell an edit
        # that failed to land from one that was never needed.
        camera_file = tmp_path / "cameras.json"
        camera_file.write_text(
            json.dumps(
                {
                    "cameras": list(entries)
                    or [{"id": "cam-1", "label": "Camera One", "url": str(ASSET)}]
                }
            ),
            encoding="utf-8",
        )
        composition, _ = composed(tmp_path, cameras_file=str(camera_file))
        return ComposedService(lambda: composition), camera_file, composition

    async def test_an_edit_reaches_both_the_file_and_the_running_camera(
        self, tmp_path: Path
    ) -> None:
        """Both halves in one assertion, deliberately. An implementation that only
        writes the file passes every `CameraFileStore` test and leaves the console
        showing the old label until a restart; one that only mutates the runner
        passes every `EngineService` test and loses the change at that restart."""
        service, camera_file, _ = self._service(tmp_path)
        await service.start()
        try:
            record = await service.update_camera(
                "cam-1", CameraEdit(label="Wing B corridor", zone=Zone.CORRIDOR)
            )

            assert (record.label, record.zone) == ("Wing B corridor", Zone.CORRIDOR)
            live = service.telemetry("cam-1")
            assert (live.label, live.zone) == ("Wing B corridor", Zone.CORRIDOR)
            (stored,) = load_cameras(camera_file)
            assert (stored.label, stored.zone) == ("Wing B corridor", Zone.CORRIDOR)
        finally:
            await service.stop()

    async def test_the_edit_survives_a_reload_of_the_file(self, tmp_path: Path) -> None:
        """ "Persisted" means the next process reads it, which is the only definition
        an operator cares about. `load_cameras` here is the same call
        `create_default_app` makes at startup."""
        service, camera_file, _ = self._service(tmp_path)
        await service.start()
        try:
            await service.update_camera("cam-1", CameraEdit(label="Renamed"))
        finally:
            await service.stop()

        assert [camera.label for camera in load_cameras(camera_file)] == ["Renamed"]

    async def test_the_new_label_reaches_the_next_event(self, tmp_path: Path) -> None:
        """The label is not decoration: `CameraRunner` puts it on every escalation it
        assembles, so an edit that stops at `telemetry()` would rename the camera in
        the console and leave every subsequent event carrying the old name."""
        service, _, composition = self._service(tmp_path)
        await service.start()
        try:
            await service.update_camera("cam-1", CameraEdit(label="Wing B corridor"))
            async with asyncio.timeout(20.0):
                while not composition.service.event_history("cam-1").events:
                    await asyncio.sleep(0.01)
            runner = composition.service._cameras["cam-1"]
            assert runner._camera_label == "Wing B corridor"
        finally:
            await service.stop()

    async def test_an_unknown_camera_is_rejected_before_the_file_is_touched(
        self, tmp_path: Path
    ) -> None:
        """Order matters here for two reasons: a typo'd id must not be able to
        rewrite the document, and the caller must get a plain 404 rather than the
        confusing 409 `edited_document`'s own unknown-id guard would produce."""
        service, camera_file, _ = self._service(tmp_path)
        before = camera_file.read_text(encoding="utf-8")
        await service.start()
        try:
            with pytest.raises(UnknownCameraError):
                await service.update_camera("cam-9", CameraEdit(label="New"))
        finally:
            await service.stop()

        assert camera_file.read_text(encoding="utf-8") == before

    async def test_a_refused_write_leaves_the_running_camera_untouched(
        self, tmp_path: Path
    ) -> None:
        """The reason the write comes before the apply. If the file cannot take the
        edit, the operator must be looking at a console that still agrees with disk
        — not at a change that will silently vanish at the next restart."""
        service, camera_file, _ = self._service(tmp_path)
        await service.start()
        try:
            camera_file.write_text("{ not json", encoding="utf-8")
            with pytest.raises(CameraConfigError):
                await service.update_camera("cam-1", CameraEdit(label="Renamed"))

            assert service.telemetry("cam-1").label == "Camera One"
        finally:
            await service.stop()

    async def test_a_camera_removed_from_the_file_by_hand_is_a_conflict_not_an_append(
        self, tmp_path: Path
    ) -> None:
        """The file is mounted, and an operator can still edit it. The engine's
        in-memory camera list is from startup, so the two genuinely can disagree —
        and inventing an entry from memory would overwrite whoever changed it."""
        service, camera_file, _ = self._service(
            tmp_path,
            {"id": "cam-1", "url": str(ASSET)},
        )
        await service.start()
        try:
            camera_file.write_text(
                json.dumps({"cameras": [{"id": "cam-other", "url": str(ASSET)}]}),
                encoding="utf-8",
            )
            with pytest.raises(CameraConfigError, match="no camera 'cam-1'"):
                await service.update_camera("cam-1", CameraEdit(label="Renamed"))

            assert [c["id"] for c in json.loads(camera_file.read_text())["cameras"]] == [
                "cam-other"
            ]
        finally:
            await service.stop()

    async def test_an_uncomposed_engine_answers_unknown_camera(self) -> None:
        """Same honesty as every other per-camera method before `start()`: there are
        no cameras yet, so every id is unknown — and, crucially, no file is written
        for one."""
        service = ComposedService(lambda: pytest.fail("must not build before start()"))
        with pytest.raises(UnknownCameraError):
            await service.update_camera("cam-1", CameraEdit(label="Renamed"))

    def test_composition_holds_a_store_over_the_configured_camera_file(
        self, tmp_path: Path
    ) -> None:
        """The store must read the same path `load_cameras` did, or an edit would be
        written to a file nothing loads."""
        camera_file = tmp_path / "elsewhere.json"
        composition, settings = composed(tmp_path, cameras_file=str(camera_file))
        assert composition.camera_store.path == Path(settings.cameras_file) == camera_file


class TestWritesAreDisabledUnlessTurnedOn:
    def test_the_setting_defaults_to_off(self) -> None:
        """The posture a deployment gets when nobody has thought about it. This
        engine has no authentication, so an on-by-default write endpoint hands
        camera reconfiguration to anything that can reach the port."""
        assert Settings().enable_camera_writes is False

    def test_the_default_app_is_built_read_only(self) -> None:
        assert main.app.state.camera_writes_enabled is False


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

        # The console's recent-event ring, through production composition rather than
        # a hand-wired scheduler: this is what proves `GET /cameras/{id}/events` will
        # actually have something to serve in the assembled system.
        history = composition.service.event_history("cam-1")
        assert len(history.events) == len(spooled), (
            "every assembled event must be in the console ring, not just the published ones"
        )
        assert history.latest is not None
        assert history.latest.description, "the live panel needs a description to show"
        assert history.capacity == RECENT_EVENTS_PER_CAMERA
