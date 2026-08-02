"""The composition root: the one place that builds the real system.

    uvicorn sentinel_ai.main:app --host 0.0.0.0 --port 8000

Everything below this line in the import graph is written to be composed; nothing
below it composes anything. This module is the outermost layer — it may import
every other layer, and no other module may import it (`tests/test_architecture.py`
keeps `domain/` and `ports/` pure regardless).

Why this module is not optional
-------------------------------
Phase 1B's three Critical defects (the VLM never reloading after its idle unload,
`CameraRunner`'s two divergent time bases, and `EngineService.stop()` discarding the
event the runner's shutdown path exists to preserve) were all wiring-level: each was
invisible in every unit test and only appeared once the real pieces were assembled.
Nothing assembled them, so nobody saw them. Two more consequences of the same gap are
fixed here: `ModelSpec` was never constructed outside tests, so the measured-VRAM
correction chain fed nothing; and `RabbitMQPublisher.connect()`/`replay_spool()` had
no production caller at all, which made spec §9's disk spool write-only.

Camera configuration
--------------------
A JSON file, path from `SENTINEL_CAMERAS_FILE` (default `./cameras.json`). See
`cameras.example.json`:

    {
      "cameras": [
        {
          "id": "avenue_01",
          "label": "Avenue (demo)",
          "url": "rtsp://localhost:8554/avenue_01",
          "zone": "corridor",
          "profile": {"cooldown_seconds": 5.0}
        }
      ]
    }

`id` and `url` are required; `label` defaults to `id`; `profile` overrides any
`CameraProfile` field and is validated against that dataclass's own field names, so a
typo fails at startup rather than silently doing nothing. `zone` is optional and
constrained to `sentinel_ai.domain.zone.Zone` — omit it and the camera is ungrouped,
which is a legitimate deployment and not a config error; give it a value this process
does not know and startup fails, because a typo'd zone is a camera the operator meant
to group and silently did not. A `url` with an `rtsp://` or
`rtsps://` scheme builds an `RtspSource`; anything else is taken as a path to a video
file and builds a `FileSource`, which is what makes a replay deployment (and the
end-to-end demo over a downloaded clip) the same code path as a live camera.

A file rather than environment variables because a camera is a nested record with a
nested profile: flattening a list of those into `SENTINEL_*` names is a worse
interface than one small document that can be diffed, reviewed and mounted into a
container. Everything that is genuinely a process-wide scalar stays in `Settings`.

The reader, the writer and the record itself now live in
`sentinel_ai.adapters.config.camera_file`; they are re-exported here because this
module composed them first and because `load_cameras` is still called from exactly
one place, `create_default_app` below. `PATCH /cameras/{id}` edits `label` and
`zone` through `CameraFileStore` — persisted to the same file, then applied to the
running `CameraRunner` — and `url` and `profile` stay restart-only. See that
module's docstring for why the line is drawn there, and `Settings.enable_camera_writes`
for why the endpoint is off unless a deployment turns it on.

Composition happens at startup, not at import
---------------------------------------------
`RtspSource.__init__` calls `asyncio.create_task`, so a camera cannot be constructed
before an event loop is running, and `create_app` takes an already-built service.
`ComposedService` closes that gap: it satisfies `EngineServiceProtocol` immediately
and builds the real `EngineService` inside `start()`, which the FastAPI lifespan runs
on the loop. Importing this module therefore reads no camera file, opens no socket and
touches no GPU — which is also what lets CI import it with neither the `gpu` extra nor
a `cameras.json` present.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from fastapi import FastAPI

from sentinel_ai.adapters.config.camera_file import (
    CameraConfig,
    CameraConfigError,
    CameraEdit,
    CameraFileStore,
    load_cameras,
)
from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, select_device
from sentinel_ai.adapters.publishers.dead_letter import DeadLetterSpool
from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.adapters.sources.rtsp import RtspSource
from sentinel_ai.adapters.storage.minio_clips import MinioClipWriter
from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber
from sentinel_ai.api.app import create_app
from sentinel_ai.config import Settings, get_settings
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.event_history import CameraEventHistory, EventSubscription
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.orchestrator.service import (
    EngineNotComposedError,
    EngineService,
    UnknownCameraError,
)
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.event_publisher import FailedEventSink
from sentinel_ai.ports.frame_source import FrameSource
from sentinel_ai.ports.model_runtime import HealthReport
from sentinel_ai.ports.vision_llm import VisionLanguageModel

logger = logging.getLogger(__name__)

__all__ = [
    "BrokerLink",
    "CameraConfig",
    "CameraConfigError",
    "CameraEdit",
    "CameraFileStore",
    "ComposedService",
    "Composition",
    "Models",
    "app",
    "build_clip_writer",
    "build_dead_letter",
    "build_models",
    "build_publisher",
    "build_source",
    "compose",
    "create_default_app",
    "load_cameras",
    "refresh_specs_from_capabilities",
]

_RTSP_SCHEMES = ("rtsp://", "rtsps://")

_DETECTOR_PRIORITY = 100
"""Higher survives eviction. The detector is always-on: a camera without it sees
nothing at all, whereas a camera without the VLM still detects and still publishes
(spec §9's degraded, description-unavailable event)."""

_VLM_PRIORITY = 50


@dataclass(frozen=True, slots=True)
class Models:
    """The two model adapters, each under both the interface the pipeline calls and
    the `ModelRuntime` lifecycle the registry manages (S13)."""

    detector_key: str
    detector: ObjectDetector
    vlm_key: str
    vlm: VisionLanguageModel
    registry: ModelRegistry


@dataclass(frozen=True, slots=True)
class Composition:
    """Everything the process owns, assembled. Held together so `stop()` can unwind
    the parts `EngineService` does not know about (the broker link, the connection)."""

    service: EngineService
    registry: ModelRegistry
    publisher: RabbitMQPublisher
    broker: BrokerLink
    camera_store: CameraFileStore
    """The writer behind `PATCH /cameras/{id}`, over the same file `load_cameras`
    read. Held here rather than inside `EngineService` because the engine owns
    running cameras and this owns the record of configured ones; `ComposedService`
    is the only thing that needs both, and it is the only thing that has both."""


# -- adapter construction -----------------------------------------------------------


def build_source(config: CameraConfig, settings: Settings) -> FrameSource:
    """An `RtspSource` for an rtsp(s) URL, a `FileSource` for anything else.

    Must be called with a running event loop: `RtspSource.__init__` starts its
    reconnect task immediately, and begins pumping before any consumer exists.
    """
    if config.url.startswith(_RTSP_SCHEMES):
        return RtspSource(
            camera_id=config.camera_id,
            url=config.url,
            reconnect_initial_seconds=settings.rtsp_reconnect_initial_seconds,
            reconnect_max_seconds=settings.rtsp_reconnect_max_seconds,
        )
    return FileSource(config.url, camera_id=config.camera_id, realtime=settings.source_realtime)


def build_models(settings: Settings) -> Models:
    """Construct both model adapters and register them with their startup `ModelSpec`.

    Constructing is cheap and importless — `Yolo11Detector` and `Qwen25VLDescriber`
    both defer `torch`/`ultralytics`/`transformers` to `initialize()` — so this runs
    on a box with no GPU extra installed. Only `select_device()` needs torch, and only
    when `Settings.device` is unset.
    """
    device = settings.device if settings.device is not None else select_device()
    detector = Yolo11Detector(
        model_id=settings.detector_model_id,
        conf=settings.detector_conf_threshold,
        iou=settings.detector_iou_threshold,
        imgsz=settings.detector_imgsz,
        device=device,
    )
    vlm = Qwen25VLDescriber(
        model_id=settings.vlm_model_id,
        max_new_tokens=settings.vlm_max_new_tokens,
        device=device,
    )
    registry = ModelRegistry()
    registry.register(
        ModelSpec(
            model_key=settings.detector_model_id,
            vram_mib=settings.detector_vram_mib,
            priority=_DETECTOR_PRIORITY,
            idle_unload_seconds=None,
        ),
        detector,
    )
    registry.register(
        ModelSpec(
            model_key=settings.vlm_model_id,
            vram_mib=settings.vlm_vram_mib,
            priority=_VLM_PRIORITY,
            idle_unload_seconds=settings.vlm_idle_unload_seconds,
        ),
        vlm,
    )
    return Models(
        detector_key=settings.detector_model_id,
        detector=detector,
        vlm_key=settings.vlm_model_id,
        vlm=vlm,
        registry=registry,
    )


def build_publisher(settings: Settings) -> RabbitMQPublisher:
    return RabbitMQPublisher(
        url=settings.rabbitmq_url,
        exchange=settings.rabbitmq_exchange,
        spool_dir=Path(settings.event_spool_dir),
    )


def build_dead_letter(settings: Settings) -> FailedEventSink:
    return DeadLetterSpool(Path(settings.dead_letter_dir))


def build_clip_writer(settings: Settings) -> ClipWriter:
    return MinioClipWriter(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
        temp_dir=Path(settings.clip_temp_dir),
    )


def refresh_specs_from_capabilities(registry: ModelRegistry) -> dict[str, int]:
    """Replace each startup `ModelSpec.vram_mib` estimate with the measured figure.

    `plan_residency()` reads `ModelSpec.vram_mib`, a configured number, not
    `capabilities().vram_mib`, the measured one — and `capabilities().vram_mib` is 0
    until `warmup()` has run, which is after `ensure()` has already computed the first
    plan. That chicken-and-egg is why a configured seed exists at all. Calling this
    once after `EngineService.start()` closes it: from the second plan onwards (every
    idle sweep, every describe-time `ensure()`) the arithmetic runs on what the card
    actually gave the process rather than on a number in a `.env`.

    Returns the keys it changed and their new values, for logging and for tests.
    """
    changed: dict[str, int] = {}
    for spec in registry.specs():
        runtime = registry.get(spec.model_key)
        measured = runtime.capabilities().vram_mib
        # 0 means "never warmed up", not "needs no VRAM": trusting it would tell the
        # planner both models are free and let it admit anything.
        if measured > 0 and measured != spec.vram_mib:
            registry.register(replace(spec, vram_mib=measured), runtime)
            changed[spec.model_key] = measured
    if changed:
        logger.info("model vram specs updated from measured capabilities: %s", changed)
    return changed


# -- the broker link ----------------------------------------------------------------


class BrokerConnection(Protocol):
    """The slice of `RabbitMQPublisher` the reconnect loop drives."""

    @property
    def connected(self) -> bool: ...

    async def connect(self) -> None: ...

    async def replay_spool(self) -> None: ...


class BrokerLink:
    """Keeps the publisher connected and drains its disk spool.

    Spec §9 promises a "disk-backed buffer, replayed on reconnect". Before this
    existed the buffer was real and the replay was not: `connect()` and
    `replay_spool()` had zero production callers, so every event written to the spool
    while the broker was down stayed there forever.

    Structured like `rtsp._ReconnectLoop` and for the same reason — a plain
    "wait, then act" sequencer with an injected `sleep`, so its backoff is unit-
    testable with no broker, no network and no wall-clock time.
    """

    def __init__(
        self,
        publisher: BrokerConnection,
        *,
        initial_seconds: float,
        max_seconds: float,
        replay_interval_seconds: float,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._publisher = publisher
        self._initial = initial_seconds
        self._max = max_seconds
        self._replay_interval = replay_interval_seconds
        self._sleep = sleep
        self.backoff_log: list[float] = []

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        backoff = self._initial
        while not should_stop():
            if await self.step():
                backoff = self._initial
                await self._sleep(self._replay_interval)
            else:
                self.backoff_log.append(backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2.0, self._max)

    async def step(self) -> bool:
        """One connect-and-drain attempt; True when the link is healthy afterwards.

        Never raises. A broker that is down is the ordinary case — the engine keeps
        detecting and keeps spooling — so it must not be able to kill this task and
        take the replay with it.
        """
        try:
            if not self._publisher.connected:
                await self._publisher.connect()
                logger.info("broker connected; replaying spool")
            await self._publisher.replay_spool()
        except Exception as exc:
            logger.warning("broker link unavailable (%s); will retry", exc)
            return False
        return True


# -- assembly -----------------------------------------------------------------------


def compose(
    settings: Settings,
    cameras: tuple[CameraConfig, ...],
    models: Models,
    publisher: RabbitMQPublisher,
    clip_writer: ClipWriter | None,
    dead_letter: FailedEventSink,
) -> Composition:
    """Wire everything into one `EngineService`. Call with a running event loop.

    The adapters are passed in rather than built here so that a caller can compose
    the same graph over stand-ins — which is what makes this function, rather than a
    hand-written script, the thing CI can exercise.
    """
    resident_set = ResidentSet(
        models.registry,
        total_mib=settings.vram_total_mib,
        reserved_mib=settings.vram_reserved_mib,
    )
    scheduler = VlmScheduler(
        models.vlm,
        publisher,
        AdmissionGate(
            concurrency=settings.vlm_global_concurrency,
            min_interval_seconds=settings.vlm_global_min_interval_seconds,
        ),
        resident_set=resident_set,
        vlm_model_key=models.vlm_key,
        dead_letter=dead_letter,
        maxsize=settings.vlm_queue_maxsize,
        timeout_seconds=settings.vlm_timeout_seconds,
        # Real elapsed time, deliberately: `AdmissionGate` spaces admissions with
        # `asyncio.sleep`, so its clock has to advance with the loop's. This is a
        # different clock from the camera pipeline's, which has none of its own and
        # runs entirely on the source timeline (see pipeline/runner.py).
        clock=time.monotonic,
        # The wall clock `VlmScheduler._to_epoch` anchors each camera to, lazily and
        # once per camera, so `Event.occurred_at` is Unix epoch seconds that no NTP
        # step can reorder — see `_to_epoch` for why the anchor is per camera rather
        # than shared with the monotonic reading above.
        wall_clock=time.time,
    )
    runners = {
        config.camera_id: CameraRunner(
            camera_id=config.camera_id,
            camera_label=config.label,
            source=build_source(config, settings),
            detector=models.detector,
            tracker=ByteTrackTracker(),
            motion=MotionAnalyzer(),
            profile=config.profile,
            scheduler=scheduler,
            clip_writer=clip_writer,
            preroll=PreRollBuffer(preroll_seconds=settings.clip_preroll_seconds),
            detect_every_n_frames=settings.detect_every_n_frames,
            zone=config.zone,
        )
        for config in cameras
    }
    service = EngineService(
        cameras=runners,
        registry=models.registry,
        resident_set=resident_set,
        scheduler=scheduler,
        # Both resident from boot: the detector because every frame needs it, the VLM
        # so the first escalation is not paying a multi-second `from_pretrained`. The
        # 600s idle sweep may still evict the VLM later, and the scheduler's
        # `ensure()` brings it back on the next describe.
        required_model_keys=(models.detector_key, models.vlm_key),
        clock=time.monotonic,
    )
    broker = BrokerLink(
        publisher,
        initial_seconds=settings.rtsp_reconnect_initial_seconds,
        max_seconds=settings.rtsp_reconnect_max_seconds,
        replay_interval_seconds=settings.broker_replay_interval_seconds,
    )
    return Composition(
        service=service,
        registry=models.registry,
        publisher=publisher,
        broker=broker,
        # The same path `load_cameras` was handed above. Deliberately re-read on
        # every edit rather than caching the document composition already parsed:
        # the operator who mounted this file can still edit it by hand, and the
        # console must not silently overwrite what they wrote.
        camera_store=CameraFileStore(Path(settings.cameras_file)),
    )


class ComposedService:
    """`EngineServiceProtocol` that composes the real engine when the app starts.

    `create_app` takes an already-constructed service, and `RtspSource.__init__`
    needs a running loop, so the two cannot meet at import time. This defers the
    whole construction into `start()`, which the FastAPI lifespan runs on the loop.
    Before `start()` the engine genuinely has no cameras and no models, and the
    read-only endpoints say exactly that rather than pretending otherwise.
    """

    def __init__(self, build: Callable[[], Composition]) -> None:
        self._build = build
        self._composition: Composition | None = None
        self._broker_task: asyncio.Task[None] | None = None

    @property
    def composition(self) -> Composition | None:
        return self._composition

    async def start(self) -> None:
        composition = self._build()
        self._composition = composition
        # Started before the engine: the spool may already hold events from a previous
        # process, and draining them does not depend on any camera being up.
        self._broker_task = asyncio.create_task(
            composition.broker.run_forever(should_stop=lambda: False)
        )
        await composition.service.start()
        refresh_specs_from_capabilities(composition.registry)

    async def stop(self) -> None:
        """Unwind what `EngineService` does not own — and let a cut-short shutdown
        stay cut short.

        This is the caller that makes `EngineService._cancel`'s `cancelling()`
        bookkeeping matter in production: uvicorn's `--timeout-graceful-shutdown`
        cancels the ASGI lifespan's shutdown task, and `create_app`'s lifespan awaits
        this method in exactly that position. So the same discipline applies one layer
        out — nothing here may swallow a `CancelledError` aimed at *this* coroutine,
        because doing so reports a truncated shutdown to uvicorn as a clean one.

        The broker link is therefore cancelled **first and synchronously**: `cancel()`
        cannot itself be interrupted, so the link stops even if the very next `await`
        never returns. Everything after that is an await and any of them may be the one
        that is cut short, so they are ordered by what it costs to lose them. Losing
        `publisher.close()` costs an AMQP socket that the exiting process closes anyway;
        losing `service.stop()` would cost events, which is why it goes first.
        """
        composition, self._composition = self._composition, None
        broker_task, self._broker_task = self._broker_task, None
        if composition is None:
            return
        if broker_task is not None:
            broker_task.cancel()
        try:
            # The engine first: its own shutdown ordering publishes the escalations
            # the runners preserve, and those publishes must still have somewhere to
            # go. Then one last drain, so anything spooled during shutdown reaches the
            # broker instead of waiting for the next process.
            await composition.service.stop()
            await composition.broker.step()
        finally:
            if broker_task is not None:
                # `gather(..., return_exceptions=True)` rather than
                # `suppress(CancelledError)`: it absorbs the *task's* cancellation as a
                # result while still propagating one delivered to *us*. A blanket
                # suppress cannot tell those apart — the defect fixed in
                # `EngineService._cancel`, which is reachable here for the same reason.
                await asyncio.gather(broker_task, return_exceptions=True)
            await composition.publisher.close()

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return () if self._composition is None else self._composition.service.cameras()

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return self._composition.service.telemetry(camera_id)

    def event_history(self, camera_id: str) -> CameraEventHistory:
        # Same honesty as `telemetry` above: before `start()` there are no cameras at
        # all, so every id is unknown. Returning an empty history instead would tell a
        # console that a camera exists and has simply been quiet.
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return self._composition.service.event_history(camera_id)

    def subscribe_events(self) -> EventSubscription:
        if self._composition is None:
            # No ring exists yet, so there is nothing to subscribe *to*. An open stream
            # that can never carry anything is indistinguishable from a quiet site, so
            # the API answers 503 instead. See `EngineNotComposedError`.
            raise EngineNotComposedError(
                "the engine has not finished starting; no event stream is available yet"
            )
        return self._composition.service.subscribe_events()

    def close_event_streams(self) -> int:
        # Zero, not an error: the lifespan calls this unconditionally on the way out,
        # including after a startup that never composed anything.
        return 0 if self._composition is None else self._composition.service.close_event_streams()

    def health(self) -> dict[str, HealthReport]:
        return {} if self._composition is None else self._composition.service.health()

    async def describe_now(self, camera_id: str) -> UUID:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return await self._composition.service.describe_now(camera_id)

    async def update_camera(self, camera_id: str, edit: CameraEdit) -> CameraConfig:
        """Persist an edit to `cameras.json`, then apply it to the running camera.

        Composition-root work by nature: it needs the file (which `EngineService`
        knows nothing about) and the runner (which `CameraFileStore` knows nothing
        about), and this is the one object holding both.

        The order below is the whole safety property, and each step exists because
        the alternative ordering is silently wrong:

          1. **unknown camera first**, via `telemetry()`, so an id the engine never
             heard of is a 404 that never opened the file. Writing first would let
             a typo'd id rewrite a document — and, because `edited_document` also
             refuses an unknown id, would turn a plain 404 into a confusing 409;
          2. **persist second.** If the write fails, nothing has been applied: the
             running camera still matches the file, and the operator gets an error
             instead of a change that vanishes at the next restart;
          3. **apply third, from the record the store read back**, not from the
             request. Applying the request would let memory and disk disagree
             wherever the file's normalisation differs from what was asked for.

        A failure in (2) therefore leaves the system exactly as it was, and a
        success in (2) is always followed by (3) — the apply is two attribute
        writes on an object already proven to exist, with no await between them, so
        there is no window where the file has moved and the camera has not.
        """
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        before = self._composition.service.telemetry(camera_id)
        record = await self._composition.camera_store.apply(camera_id, edit)
        self._composition.service.update_camera_metadata(
            camera_id, label=record.label, zone=record.zone
        )
        # Old value logged alongside the new one: this endpoint is unauthenticated
        # (see the module docstring), so the log line is the only record of what a
        # camera used to be called or where it used to be grouped once the write
        # above overwrites both in the file. Without it, an anonymous caller could
        # rename a camera to something misleading and nobody could reconstruct what
        # the console said about that location a minute earlier.
        logger.info(
            "camera %s reconfigured: label=%r->%r zone=%s->%s",
            camera_id,
            before.label,
            record.label,
            before.zone,
            record.zone,
        )
        return record


def create_default_app() -> FastAPI:
    """The app `uvicorn sentinel_ai.main:app` serves."""
    settings = get_settings()

    def build() -> Composition:
        cameras = load_cameras(Path(settings.cameras_file))
        logger.info("composing engine for %d camera(s)", len(cameras))
        return compose(
            settings,
            cameras,
            build_models(settings),
            build_publisher(settings),
            build_clip_writer(settings),
            build_dead_letter(settings),
        )

    if settings.enable_camera_writes:
        logger.warning(
            "camera writes are ENABLED and this engine has no authentication: anyone "
            "who can reach this port can relabel and re-zone cameras. Bind it to "
            "localhost or put an authenticating proxy in front of it."
        )
    return create_app(ComposedService(build), camera_writes_enabled=settings.enable_camera_writes)


app = create_default_app()
