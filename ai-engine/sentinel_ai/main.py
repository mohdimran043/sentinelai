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

from sentinel_ai.adapters.alerts.json_store import JsonAlertStore
from sentinel_ai.adapters.config.camera_file import (
    UNSET,
    CameraConfig,
    CameraConfigError,
    CameraCreate,
    CameraEdit,
    CameraFileStore,
    load_cameras,
)
from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, select_device
from sentinel_ai.adapters.face.encrypted_store import EncryptedFaceStore
from sentinel_ai.adapters.face.insightface_pipeline import InsightFacePipeline
from sentinel_ai.adapters.notifiers.logging import LoggingNotifier
from sentinel_ai.adapters.notifiers.webhook import WebhookNotifier
from sentinel_ai.adapters.pose.yolo11_pose import Yolo11PoseEstimator
from sentinel_ai.adapters.publishers.dead_letter import DeadLetterSpool
from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.adapters.sources.earthcam import is_earthcam_url
from sentinel_ai.adapters.sources.earthcam_source import EarthCamSource
from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.adapters.sources.probe import ProbeResult, probe_source
from sentinel_ai.adapters.sources.rtsp import RtspSource
from sentinel_ai.adapters.storage.minio_clips import MinioClipReader, MinioClipWriter
from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber
from sentinel_ai.api.app import create_app
from sentinel_ai.config import NotifierKind, Settings, get_settings
from sentinel_ai.domain.alert import Alert
from sentinel_ai.domain.capabilities import Capability, ModelRole, required_roles
from sentinel_ai.domain.identity import AuthorizedPerson, EnrolledFace
from sentinel_ai.domain.welfare import ConcernKind
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.alert_persistence import AlertPersistence
from sentinel_ai.orchestrator.alerts import AlertRegister, UnknownAlertError
from sentinel_ai.orchestrator.event_history import CameraEventHistory, EventSubscription
from sentinel_ai.orchestrator.notifications import NotificationDispatcher
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.orchestrator.service import (
    DuplicateCameraError,
    EngineNotComposedError,
    EngineService,
    FaceCapabilityUnavailableError,
    UnknownCameraError,
)
from sentinel_ai.pipeline.behaviour import BehaviourEngine
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.ports.alert_store import AlertStore
from sentinel_ai.ports.clip_index import ClipRecord, StorageUsage
from sentinel_ai.ports.clip_reader import ClipReader
from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.event_publisher import FailedEventSink
from sentinel_ai.ports.face import FaceDetector, FaceStore
from sentinel_ai.ports.frame_source import FrameSource
from sentinel_ai.ports.model_runtime import HealthReport
from sentinel_ai.ports.notifier import Notifier
from sentinel_ai.ports.pose import PoseEstimator
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
    "build_clip_reader",
    "build_clip_writer",
    "build_dead_letter",
    "build_models",
    "build_notifier",
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

_FACE_PRIORITY = 85
"""Below pose, above the VLM. Face work is per person rather than per frame, so it is
less punishing to evict than pose — but evicting it to place a VLM would leave a camera
that was asked to check identities silently not checking them."""

_POSE_PRIORITY = 90
"""Below the always-on detector, above the VLM.

Ordering matters to `plan_residency()`, which evicts LRU among lower-or-equal-priority
models to make room. Pose sits above the VLM because it is read on every sampled frame
of a fall-detection camera while the VLM fires only on escalations: evicting pose to
place a VLM would blind the fall detector for the whole of a describe, and the describe
is the less time-critical of the two. It sits below the detector because nothing works
without detection."""


@dataclass(frozen=True, slots=True)
class Models:
    """The model adapters this process actually built, each under both the interface
    the pipeline calls and the `ModelRuntime` lifecycle the registry manages (S13).

    Which ones exist is decided by the union of every camera's capabilities, not by
    this class: see `build_models`. A role nobody asked for is `None` here, is absent
    from `registry`, and never appears in `/health` — which is the point. §13's
    "automatically avoid loading unnecessary models for disabled capabilities" is not
    a lazy-load optimisation; it is the difference between a 2.7 GiB resident VLM and
    no VLM at all on a box running thirty corridor cameras that only need triggers.
    """

    detector_key: str | None
    detector: ObjectDetector | None
    vlm_key: str | None
    vlm: VisionLanguageModel | None
    registry: ModelRegistry
    pose_key: str | None = None
    pose: PoseEstimator | None = None
    face_key: str | None = None
    face: FaceDetector | None = None
    """The keypoint model, when some camera enabled a capability needing one.

    Defaulted, unlike the two above, so that every existing construction site — every
    test that builds a `Models` by hand — keeps working and describes a process with no
    pose model, which is what those sites mean."""

    def available_roles(self) -> frozenset[ModelRole]:
        """The roles this process actually built.

        Read off the adapters themselves, not recomputed from the camera file. The
        two normally agree — `build_models` was handed `required_roles()` over that
        same file — but they are different facts, and only this one is true by
        construction: a composition handed stand-ins (every test that calls
        `compose`), or one whose model construction partially failed, has the roles
        it *has*, not the roles the file asked for. A capability edit is refused
        against what loaded, so it must consult the fact that cannot be stale.
        """
        return frozenset(
            role
            for role, built in (
                (ModelRole.DETECTOR, self.detector is not None),
                (ModelRole.VLM, self.vlm is not None),
                (ModelRole.POSE, self.pose is not None),
                (ModelRole.FACE, self.face is not None),
            )
            if built
        )

    def required_keys(self) -> tuple[str, ...]:
        """The model keys `EngineService` should make resident at startup.

        Only the ones that exist. Passing a `None` through to
        `ResidentSet.ensure` would ask the registry for a model nobody registered
        and fail startup on a configuration that is perfectly legal.
        """
        return tuple(
            key
            for key in (self.detector_key, self.vlm_key, self.pose_key, self.face_key)
            if key is not None
        )


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

    settings: Settings
    models: Models
    face_store: FaceStore | None
    clip_writer: ClipWriter | None
    """The four things `build_runner` needs, held so a camera added after startup is
    built from exactly what the cameras at startup were built from. Without them
    `create_camera` would have to reconstruct them, and a second construction site is a
    second place for a capability to be wired to the wrong model."""

    notifier: Notifier
    """The welfare notifier, held for the same reason `publisher` is: it owns
    process-wide state `EngineService` knows nothing about and cannot unwind — a
    pooled HTTP client, and a filter attached to the process's `httpx` logger. See
    `ComposedService.stop`."""


# -- adapter construction -----------------------------------------------------------


def build_source(config: CameraConfig, settings: Settings) -> FrameSource:
    """An `EarthCamSource` for an earthcam.com page, an `RtspSource` for an rtsp(s) URL,
    a `FileSource` for anything else.

    Must be called with a running event loop: `RtspSource.__init__` starts its
    reconnect task immediately, and begins pumping before any consumer exists.
    """
    # Before the scheme check: an EarthCam camera is configured as the page a person
    # would open in a browser, which is an https:// URL and would otherwise fall through
    # to `FileSource` and be opened as a path.
    if is_earthcam_url(config.url):
        return EarthCamSource(
            camera_id=config.camera_id,
            page_url=config.url,
            reconnect_initial_seconds=settings.rtsp_reconnect_initial_seconds,
            reconnect_max_seconds=settings.rtsp_reconnect_max_seconds,
            hwaccel_device=settings.decode_hwaccel,
            max_fps=settings.source_max_fps,
        )
    if config.url.startswith(_RTSP_SCHEMES):
        return RtspSource(
            camera_id=config.camera_id,
            url=config.url,
            reconnect_initial_seconds=settings.rtsp_reconnect_initial_seconds,
            reconnect_max_seconds=settings.rtsp_reconnect_max_seconds,
            hwaccel_device=settings.decode_hwaccel,
            max_fps=settings.source_max_fps,
        )
    return FileSource(
        config.url,
        camera_id=config.camera_id,
        realtime=settings.source_realtime,
        hwaccel_device=settings.decode_hwaccel,
    )


def build_models(settings: Settings, roles: frozenset[ModelRole]) -> Models:
    """Construct and register exactly the model adapters `roles` asks for (spec §13).

    `roles` comes from `required_roles()` over every configured camera's capabilities,
    so a process whose cameras all run triggers-only builds no VLM: no checkpoint
    download, no VRAM, no entry in `/health`, and `VlmScheduler` takes its
    skipped-description path for every escalation. This is the only place that
    decision is made, and it is made *before* construction rather than by leaving an
    unused object loaded, because a model that exists is a model the resident set will
    dutifully keep resident.

    Binding roles to checkpoints happens here and only here. `domain/capabilities.py`
    names `ModelRole.VLM`; which checkpoint fills it is `Settings.vlm_model_id`, and
    the pure layer never learns it — see that module on why it may not import `config`.

    Constructing is cheap and importless — `Yolo11Detector` and `Qwen25VLDescriber`
    both defer `torch`/`ultralytics`/`transformers` to `initialize()` — so this runs
    on a box with no GPU extra installed. Only `select_device()` needs torch, and only
    when `Settings.device` is unset, which is why even that is skipped when no role
    needs a device at all.
    """
    registry = ModelRegistry()
    if not roles:
        # No camera enabled anything. Nothing to place, and nothing to ask torch about.
        return Models(detector_key=None, detector=None, vlm_key=None, vlm=None, registry=registry)

    device = settings.device if settings.device is not None else select_device()

    detector: ObjectDetector | None = None
    detector_key: str | None = None
    if ModelRole.DETECTOR in roles:
        detector_key = settings.detector_model_id
        concrete_detector = Yolo11Detector(
            model_id=settings.detector_model_id,
            conf=settings.detector_conf_threshold,
            iou=settings.detector_iou_threshold,
            imgsz=settings.detector_imgsz,
            device=device,
        )
        detector = concrete_detector
        registry.register(
            ModelSpec(
                model_key=settings.detector_model_id,
                vram_mib=settings.detector_vram_mib,
                priority=_DETECTOR_PRIORITY,
                idle_unload_seconds=None,
            ),
            concrete_detector,
        )

    vlm: VisionLanguageModel | None = None
    vlm_key: str | None = None
    if ModelRole.VLM in roles:
        vlm_key = settings.vlm_model_id
        concrete_vlm = Qwen25VLDescriber(
            model_id=settings.vlm_model_id,
            max_new_tokens=settings.vlm_max_new_tokens,
            quantization=settings.vlm_quantization,
            device=device,
        )
        vlm = concrete_vlm
        registry.register(
            ModelSpec(
                model_key=settings.vlm_model_id,
                vram_mib=settings.vlm_vram_mib,
                priority=_VLM_PRIORITY,
                idle_unload_seconds=settings.vlm_idle_unload_seconds,
            ),
            concrete_vlm,
        )

    pose: PoseEstimator | None = None
    pose_key: str | None = None
    if ModelRole.POSE in roles:
        pose_key = settings.pose_model_id
        concrete_pose = Yolo11PoseEstimator(
            model_id=settings.pose_model_id,
            conf=settings.pose_conf_threshold,
            imgsz=settings.pose_imgsz,
            device=device,
        )
        pose = concrete_pose
        registry.register(
            ModelSpec(
                model_key=settings.pose_model_id,
                vram_mib=settings.pose_vram_mib,
                priority=_POSE_PRIORITY,
                idle_unload_seconds=settings.pose_idle_unload_seconds,
            ),
            concrete_pose,
        )

    face: FaceDetector | None = None
    face_key: str | None = None
    if ModelRole.FACE in roles:
        face_key = settings.face_model_name
        concrete_face = InsightFacePipeline(model_name=settings.face_model_name, device=device)
        face = concrete_face
        registry.register(
            ModelSpec(
                model_key=settings.face_model_name,
                vram_mib=settings.face_vram_mib,
                priority=_FACE_PRIORITY,
                # Never idle-evicted, for the pose model's reason and more sharply: it
                # is idle exactly when nobody is about, and the reload would land as
                # somebody walks in — the moment authorisation matters.
                idle_unload_seconds=None,
            ),
            concrete_face,
        )

    return Models(
        detector_key=detector_key,
        detector=detector,
        vlm_key=vlm_key,
        vlm=vlm,
        registry=registry,
        pose_key=pose_key,
        pose=pose,
        face_key=face_key,
        face=face,
    )


def build_behaviour_engine(config: CameraConfig) -> BehaviourEngine:
    """The behaviour machines one camera's capabilities ask for.

    Each policy is passed only when its capability is enabled, so the engine's own
    `enabled`/`needs_pose` answers are derived from what the operator actually switched
    on rather than from what happens to be configured. A camera with zones drawn but
    `zone_monitoring` off runs nothing, which is what "off" has to mean.
    """
    enabled = config.capabilities.enabled
    return BehaviourEngine(
        fall=config.fall_policy if enabled(Capability.FALL_DETECTION) else None,
        abandonment=config.abandonment_policy if enabled(Capability.ABANDONED_OBJECT) else None,
        tamper=config.tamper_policy if enabled(Capability.CAMERA_TAMPER) else None,
        zones=config.zone_policy if enabled(Capability.ZONE_MONITORING) else None,
    )


def build_alert_store(settings: Settings) -> AlertStore | None:
    """The durable half of the alert register, unless the deployment turned it off.

    `SENTINEL_ALERT_STORE_PATH=null` is a supported configuration and returns `None`:
    the register then behaves exactly as it did before durability existed. Every other
    value is a path, and the directory is created on the first write rather than here,
    so a process that never raises an alert never creates a file.

    Unlike `build_face_store`, this one cannot refuse to start over a missing key,
    because there is no key: what it holds is operational state about a site, not
    biometric data about a person. See `adapters/alerts/json_store.py`.
    """
    if settings.alert_store_path is None:
        return None
    return JsonAlertStore(Path(settings.alert_store_path))


def build_face_store(settings: Settings, roles: frozenset[ModelRole]) -> FaceStore | None:
    """The biometric store, when any camera asked for person authorisation.

    **Refuses to start without an encryption key**, rather than falling back to writing
    biometric data in the clear. §12 requires encryption at rest, and a fallback would
    mean the property holds only on deployments that happened to configure it — which
    is the same as not holding.

    `None` when no camera enabled the capability: no store is opened, no file is
    created, and nothing on disk suggests this site does face recognition.
    """
    if ModelRole.FACE not in roles:
        return None
    if not settings.face_encryption_key:
        raise RuntimeError(
            "a camera enables person_authorization but SENTINEL_FACE_ENCRYPTION_KEY is "
            "not set. Refusing to start: face embeddings are biometric data and this "
            "engine will not write them unencrypted. Generate a key with "
            "`python -m sentinel_ai.adapters.face.encrypted_store`."
        )
    return EncryptedFaceStore(
        Path(settings.face_store_path), encryption_key=settings.face_encryption_key
    )


def build_publisher(settings: Settings) -> RabbitMQPublisher:
    return RabbitMQPublisher(
        url=settings.rabbitmq_url,
        exchange=settings.rabbitmq_exchange,
        spool_dir=Path(settings.event_spool_dir),
    )


def build_dead_letter(settings: Settings) -> DeadLetterSpool:
    """The concrete spool, not the `FailedEventSink` port it satisfies.

    Two things depend on it and they need different halves: `VlmScheduler` wants
    the port (an `Event` sink), while `WebhookNotifier` wants the wider
    `store(Event | WelfareNote, ...)` this class actually offers. Returning the
    concrete type lets one directory serve both without a second spool — see
    `build_notifier`.
    """
    return DeadLetterSpool(Path(settings.dead_letter_dir))


def build_notifier(settings: Settings, dead_letter: DeadLetterSpool) -> Notifier:
    """The welfare notifier this deployment delivers through (T10).

    `LoggingNotifier` is the default and needs nothing configured, so there is no
    composition in which welfare notification is simply absent — a route that
    reaches nobody and a site with nothing to report look identical from outside,
    and only one of them is acceptable.

    A `webhook` kind with no URL **raises** rather than falling back. The fallback
    is the tempting choice and the wrong one: an operator who set
    `SENTINEL_NOTIFIER_KIND=webhook` and mistyped the URL variable would get a
    process that starts cleanly, logs cheerfully, and never sends the one alert the
    whole feature exists for. Failing at startup is the only outcome they can act
    on.

    **A dispatch ceiling that cannot clear the adapter's retries raises too.**
    `notifier_timeout_seconds` is the *outer* deadline `NotificationDispatcher`
    puts around one whole delivery, and nothing in its name stops an operator from
    reading it as the per-request HTTP timeout and setting 5.0. That combination
    cannot deliver anything that needs a retry and does not merely fail — it
    *destroys*: `asyncio.timeout` cancels `WebhookNotifier.notify` mid-retry, the
    `CancelledError` misses that adapter's `except Exception`, and the note is lost
    with no dead-letter record and a WARNING naming a `TimeoutError` whose `str()`
    is empty. So the two numbers are compared here, where an operator can still act
    on it, against `retry_worst_case_seconds` read off the adapter rather than a
    second copy of `18.0`.

    Raising rather than warning, on the same reasoning as the missing URL and as
    `load_cameras`' typo'd `profile` and `zone` fields: this codebase fails a
    misconfiguration at startup rather than running on with it. The failure being
    guarded against here is silent by construction — no delivery, no spool, no
    stated reason — so a WARNING would be a log line whose only reader is somebody
    already looking for a problem they have no other evidence of. It is one
    environment variable away from fixed at exactly the moment this raises, and it
    costs a deployment nothing to unset `SENTINEL_NOTIFIER_TIMEOUT_SECONDS` and get
    the default that was sized for this adapter. A ceiling *equal* to the worst
    case fails too: it makes the deadline and the adapter's last attempt race, and
    a coin-flip between delivery and destruction is not a configuration to accept.

    **`install_httpx_log_redaction()` is called here, and only here.** Task 6
    exposed it as an explicit named function so it would not be a hidden side
    effect of constructing an adapter — which left it with no caller at all.
    httpx logs `HTTP Request: POST <url>` at INFO on the process-wide `httpx`
    logger, independently of anything `WebhookNotifier` logs itself, and a webhook
    URL routinely carries a credential in its path (ntfy's topic token, Slack's
    incoming-webhook token). Without this call every deployment that configures a
    webhook writes that credential into its own logs. The composition root is the
    right caller precisely because the filter is process-wide state: this is the
    one object that owns the process, and `ComposedService.stop()` is what takes it
    back off again.
    """
    if settings.notifier_kind is NotifierKind.WEBHOOK:
        if not settings.notifier_webhook_url:
            raise ValueError(
                "notifier_kind is 'webhook' but notifier_webhook_url is not set; "
                "set SENTINEL_NOTIFIER_WEBHOOK_URL or choose notifier_kind='logging'"
            )
        notifier = WebhookNotifier(settings.notifier_webhook_url, dead_letter)
        # Before `install_httpx_log_redaction`, so the raising path leaves no
        # filter attached to the process-wide `httpx` logger behind it.
        if settings.notifier_timeout_seconds <= notifier.retry_worst_case_seconds:
            raise ValueError(
                f"notifier_timeout_seconds ({settings.notifier_timeout_seconds}) does not "
                f"exceed the webhook notifier's retry worst case "
                f"({notifier.retry_worst_case_seconds}s); it is the outer ceiling on one "
                "whole delivery, retries included, not the per-request HTTP timeout, and "
                "below that figure it cancels a delivery mid-retry and destroys the note "
                "without a dead-letter record. Raise SENTINEL_NOTIFIER_TIMEOUT_SECONDS "
                "above it or leave it unset"
            )
        notifier.install_httpx_log_redaction()
        # Deliberately no URL in this line, for the reason the redaction exists.
        logger.info("welfare notifications will be delivered by webhook")
        return notifier
    return LoggingNotifier()


def build_clip_reader(settings: Settings) -> MinioClipReader:
    """Reads back what `build_clip_writer`'s writer wrote.

    The same endpoint, bucket and credentials, deliberately not the same object: the
    writer is held by the pipeline and the reader by the API, and keeping them separate
    is what lets a deployment give the reader read-only credentials without touching the
    path that records evidence.
    """
    return MinioClipReader(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
        # Reported by `GET /storage`, not enforced here — the expiry is a bucket
        # lifecycle rule the object store applies. Passed so a storage screen can say
        # how long clips last without a second round trip.
        retention_days=settings.clip_retention_days,
    )


def build_clip_writer(settings: Settings) -> ClipWriter:
    return MinioClipWriter(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
        temp_dir=Path(settings.clip_temp_dir),
        retention_days=settings.clip_retention_days,
        notify_seconds=settings.notify_clip_seconds,
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


def build_runner(
    config: CameraConfig,
    *,
    settings: Settings,
    models: Models,
    face_store: FaceStore | None,
    scheduler: VlmScheduler,
    clip_writer: ClipWriter | None,
) -> CameraRunner:
    """One camera, fully wired. Extracted from `compose` so a camera added at
    runtime is built by exactly the same code as one read from the file at startup.

    That sameness is the point rather than tidiness: a second construction site is a
    second place for a capability to be wired to the wrong model, and it would be
    reached only on the path nothing tests as heavily — the one that adds a camera to
    a running engine."""
    return CameraRunner(
        camera_id=config.camera_id,
        camera_label=config.label,
        source=build_source(config, settings),
        # Per camera, not per process. `models.detector` exists as soon as *any*
        # camera needs one (roles are a union), so handing it to a camera that
        # enabled nothing would make that camera pay for a forward pass per frame
        # to produce detections nothing reads. `None` here is what makes a
        # zero-capability camera actually cheap rather than merely quiet.
        detector=(
            models.detector if ModelRole.DETECTOR in required_roles(config.capabilities) else None
        ),
        tracker=ByteTrackTracker(),
        motion=MotionAnalyzer(),
        # The gate reads `auto_escalation_enabled`; capabilities are what an
        # operator actually set. Derived here, in the composition root, rather
        # than stored twice — a profile field and a capability that could
        # disagree is a camera whose behaviour depends on which one a reader
        # happened to look at. `domain/` never learns what a capability is.
        profile=replace(
            config.profile,
            auto_escalation_enabled=config.capabilities.enabled(Capability.ANOMALY_DETECTION),
        ),
        describe_scenes=config.capabilities.enabled(Capability.SCENE_DESCRIPTION),
        capabilities=config.capabilities,
        # One machine per capability this camera actually enabled. Built here
        # rather than in the runner because which detectors run is a capability
        # question, and `domain/capabilities.py` is where that is answered.
        behaviour=build_behaviour_engine(config),
        # All three or none, per camera. A face model exists as soon as *any*
        # camera asked for authorisation, so handing it to a camera that did not
        # would run face detection on a stream whose operator deliberately
        # excluded it — the §8 decision this capability exists to make enforceable.
        face=(
            models.face if config.capabilities.enabled(Capability.PERSON_AUTHORIZATION) else None
        ),
        face_store=(
            face_store if config.capabilities.enabled(Capability.PERSON_AUTHORIZATION) else None
        ),
        authorization=(
            config.authorization_policy
            if config.capabilities.enabled(Capability.PERSON_AUTHORIZATION)
            else None
        ),
        # Shared, and handed over only to cameras that can use it, so a camera
        # without fall detection never pays for a keypoint pass even when another
        # camera loaded the model.
        pose=(models.pose if config.capabilities.enabled(Capability.FALL_DETECTION) else None),
        scheduler=scheduler,
        clip_writer=clip_writer,
        preroll=PreRollBuffer(preroll_seconds=settings.clip_preroll_seconds),
        detect_every_n_frames=settings.detect_every_n_frames,
        zone=config.zone,
        # The per-camera welfare policy, passed as stored rather than resolved:
        # `None` means "follow the engine-wide default", and `CameraRunner` is what
        # turns that into a number — so a camera whose file says nothing keeps
        # getting whatever the settings say, and one whose override is reverted at
        # runtime gets the setting back rather than the value it booted with. The
        # pre-roll is the one whose default arrives by another route: the ring above
        # is built at `settings.clip_preroll_seconds` and narrowed by the runner when
        # this camera overrides it, and the runner records the size it was handed as
        # the value a reverted override returns to — so a composition given settings
        # other than `get_settings()`'s (every test that calls this function) reverts
        # to *its* default rather than the process-wide one.
        notify_on=config.notify_on,
        notify_min_confidence=config.notify_min_confidence,
        clip_preroll_seconds=config.clip_preroll_seconds,
        clip_postroll_seconds=config.clip_postroll_seconds,
        summary_interval_seconds=config.summary_interval_seconds,
    )


def compose(
    settings: Settings,
    cameras: tuple[CameraConfig, ...],
    models: Models,
    publisher: RabbitMQPublisher,
    clip_writer: ClipWriter | None,
    dead_letter: FailedEventSink,
    notifier: Notifier | None = None,
    face_store: FaceStore | None = None,
    alert_store: AlertStore | None = None,
) -> Composition:
    """Wire everything into one `EngineService`. Call with a running event loop.

    The adapters are passed in rather than built here so that a caller can compose
    the same graph over stand-ins — which is what makes this function, rather than a
    hand-written script, the thing CI can exercise.

    `notifier` is the one adapter with a default, and the default is
    `LoggingNotifier()`: it needs no configuration and touches no network, so
    "composed without a notifier" is a state that cannot exist rather than one every
    caller has to remember to avoid. A composition that notified nobody would be
    indistinguishable, from outside, from a site with nothing to report.
    """
    resident_set = ResidentSet(
        models.registry,
        total_mib=settings.vram_total_mib,
        reserved_mib=settings.vram_reserved_mib,
    )
    notifier = notifier if notifier is not None else LoggingNotifier()
    # Hoisted out of the `VlmScheduler(...)` call below rather than inlined, because
    # two things now need this exact object: the scheduler that fills it, and the
    # persistence coordinator that mirrors it to disk. Reading it back through
    # `VlmScheduler.alerts` would work at runtime and not type-check — that attribute is
    # optional, since a scheduler can legitimately be built without a register — and
    # silencing that would be silencing the one check that catches the two diverging.
    alert_register = AlertRegister(
        capacity=settings.alert_register_capacity,
        merge_window_seconds=settings.alert_merge_window_seconds,
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
        # The dispatcher is built here, next to its only submitter, rather than
        # passed in: nothing outside this function has a use for one, and
        # `EngineService` reaches the same object back through
        # `VlmScheduler.notifications` to own its worker task, so there is no way to
        # end up running a *different* dispatcher than the one receiving notes.
        notifications=NotificationDispatcher(
            notifier, timeout_seconds=settings.notifier_timeout_seconds
        ),
        # Built here, beside the scheduler that fills it, for the dispatcher's reason:
        # nothing outside this function needs one, and reaching it back through
        # `VlmScheduler.alerts` is what guarantees the API reads the same register the
        # events are going into.
        alerts=alert_register,
        notify_clip_min_severity=settings.notify_clip_min_severity,
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
    # Only the enabled ones get a runner. A disabled camera opens no socket and
    # decodes nothing — that is the whole point of the flag, and building a runner and
    # then not starting it would spend the connection anyway.
    runners = {
        config.camera_id: build_runner(
            config,
            settings=settings,
            models=models,
            face_store=face_store,
            scheduler=scheduler,
            clip_writer=clip_writer,
        )
        for config in cameras
        if config.enabled
    }
    clip_reader = build_clip_reader(settings)
    service = EngineService(
        cameras=runners,
        # The rest of the file: configured, switched off, and still part of the site.
        # Without this a restart would make every disabled camera vanish from
        # `/cameras`, and the only evidence it still existed would be the file.
        disabled=tuple(config for config in cameras if not config.enabled),
        registry=models.registry,
        resident_set=resident_set,
        scheduler=scheduler,
        # Both resident from boot: the detector because every frame needs it, the VLM
        # so the first escalation is not paying a multi-second `from_pretrained`. The
        # 600s idle sweep may still evict the VLM later, and the scheduler's
        # `ensure()` brings it back on the next describe.
        required_model_keys=models.required_keys(),
        # What this process can honour a capability edit for: what was actually
        # built, never what the camera file asked for. See `Models.available_roles`.
        available_roles=models.available_roles(),
        detector=models.detector,
        # The same objects the cameras use. Two stores would mean enrolling somebody
        # into a register nobody searches.
        face=models.face,
        face_store=face_store,
        # Built here beside the register it mirrors, for `NotificationDispatcher`'s
        # reason: reaching the register back through `VlmScheduler.alerts` is what
        # guarantees the thing being persisted is the thing the API reads.
        alert_persistence=(
            None
            if alert_store is None
            else AlertPersistence(
                alert_register,
                alert_store,
                flush_interval_seconds=settings.alert_flush_interval_seconds,
            )
        ),
        # Serves `GET /alerts/{id}/clip`. Built unconditionally rather than only when
        # something has already recorded: the register outlives any one clip, and an
        # engine that could not read back a clip it wrote last week would be a strange
        # thing to have configured away.
        clip_reader=clip_reader,
        # The same object. One MinIO client serving both ports rather than two, because
        # a second would double the connection pool to answer questions about the same
        # bucket.
        clip_index=clip_reader,
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
        settings=settings,
        models=models,
        face_store=face_store,
        clip_writer=clip_writer,
        notifier=notifier,
    )


async def _close_notifier(notifier: Notifier) -> None:
    """Release whatever process-wide state a notifier took, if it took any.

    `isinstance` rather than a method on the port: `Notifier` is one method wide on
    purpose, and `LoggingNotifier` genuinely has nothing to release — widening the
    port to give it an empty `aclose()` would make every future adapter implement a
    no-op to satisfy a need only one of them has. Same reasoning, and same shape, as
    `RabbitMQPublisher.close()` being the composition root's business rather than
    `EventPublisher`'s.

    Redaction first, then the client: removing the filter can never fail, while
    `aclose()` touches a real connection pool, and losing the filter would leave a
    credential-bearing closure attached to the process's `httpx` logger for the rest
    of its life.
    """
    if isinstance(notifier, WebhookNotifier):
        notifier.remove_httpx_log_redaction()
        await notifier.aclose()


def _rendered_kinds(kinds: frozenset[ConcernKind]) -> str:
    """`notify_on` as a stable, unambiguous string for the audit log below.

    Sorted, because `frozenset` iteration order follows the process's hash seed and an
    audit line that renders the same policy differently on every run cannot be compared
    with the one before it — the same reason `edited_document` writes the field sorted.
    Bracketed, because the empty set is the "notify nobody" instruction rather than an
    absent value, and unbracketed it would render as nothing at all between the `=` and
    the `->` and read as a field the log forgot.
    """
    return f"[{','.join(sorted(kinds))}]"


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
            # Last, and after `service.stop()` has drained the notification queue:
            # closing the HTTP client first would fail the very deliveries that
            # drain exists to complete. Losing this to a cut-short shutdown costs a
            # socket the exiting process closes anyway — and the redaction filter,
            # which dies with the process too. Both are cheaper than anything above
            # them here, which is why they go last.
            await _close_notifier(composition.notifier)

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

    async def list_people(self) -> tuple[tuple[AuthorizedPerson, int], ...]:
        composition = self._composition
        return () if composition is None else await composition.service.list_people()

    async def upsert_person(self, person: AuthorizedPerson) -> AuthorizedPerson:
        return await self._composed().upsert_person(person)

    async def delete_person(self, person_id: UUID) -> bool:
        return await self._composed().delete_person(person_id)

    async def enroll_face(
        self, person_id: UUID, image: object, *, original: bytes | None = None
    ) -> int:
        return await self._composed().enroll_face(person_id, image, original=original)

    def _composed(self) -> EngineService:
        """The running engine, or the same refusal an engine with no face pipeline gives.

        Before composition there is no store and no model, which is
        indistinguishable — from a caller's point of view — from a deployment that
        never enabled the capability. Reporting it the same way is the honest answer
        and avoids a second error class for a state uvicorn does not produce.
        """
        composition = self._composition
        if composition is None:
            raise FaceCapabilityUnavailableError()
        return composition.service

    def alerts(self) -> tuple[Alert, ...]:
        """The operator's alert list. Empty before the engine has composed, which is
        the truthful answer — nothing has happened yet."""
        composition = self._composition
        return () if composition is None else composition.service.alerts()

    def acknowledge_alert(self, alert_id: UUID, *, by: str, at: float) -> Alert:
        composition = self._composition
        if composition is None:
            # Before composition there is no register, so this alert genuinely is not
            # here. 404 is the honest answer rather than a 503 about startup.
            raise UnknownAlertError(alert_id)
        return composition.service.acknowledge_alert(alert_id, by=by, at=at)

    def resolve_alert(self, alert_id: UUID) -> Alert:
        composition = self._composition
        if composition is None:
            raise UnknownAlertError(alert_id)
        return composition.service.resolve_alert(alert_id)

    async def storage_usage(self) -> StorageUsage:
        """Answerable before composition finishes, and deliberately so: the object store
        is not the engine's to start, and an operator watching a slow boot should be
        able to see whether their clips are there."""
        if self._composition is None:
            return StorageUsage(
                bucket="",
                clips=0,
                objects=0,
                bytes_used=0,
                retention_days=0,
                per_camera=(),
                reachable=False,
            )
        return await self._composition.service.storage_usage()

    async def list_clips(self, camera_id: str, *, limit: int) -> tuple[ClipRecord, ...]:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return await self._composition.service.list_clips(camera_id, limit=limit)

    async def read_camera_clip(
        self, camera_id: str, event_id: UUID, *, short: bool
    ) -> bytes | None:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return await self._composition.service.read_camera_clip(camera_id, event_id, short=short)

    async def read_alert_clip(self, alert_id: UUID, *, short: bool) -> bytes | None:
        composition = self._composition
        if composition is None:
            # Before `start()` there is no register, so there is no alert with this id
            # — the same answer the other alert delegates give, for the same reason.
            raise UnknownAlertError(alert_id)
        return await composition.service.read_alert_clip(alert_id, short=short)

    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        return await self._face_service().list_faces(person_id)

    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        return await self._face_service().read_face_image(person_id, face_id)

    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        return await self._face_service().delete_face(person_id, face_id)

    def _face_service(self) -> EngineService:
        """The composed service, or the same refusal an engine without faces gives.

        Before composition there is no face store, which is exactly the state an engine
        with no `person_authorization` camera is in permanently — so the honest answer
        is the one that state already has, rather than a second error meaning "not yet".
        """
        composition = self._composition
        if composition is None:
            raise FaceCapabilityUnavailableError()
        return composition.service

    def clear_alerts(self) -> int:
        composition = self._composition
        return 0 if composition is None else composition.service.clear_alerts()

    async def flush_alerts(self) -> None:
        """No-op before composition, because there is nothing to write: the two callers
        both reach this only after an acknowledge or resolve, and both of those have
        already raised `UnknownAlertError` on an uncomposed engine."""
        composition = self._composition
        if composition is not None:
            await composition.service.flush_alerts()

    def last_frame_epoch(self, camera_id: str) -> float | None:
        composition = self._composition
        return None if composition is None else composition.service.last_frame_epoch(camera_id)

    def health(self) -> dict[str, HealthReport]:
        return {} if self._composition is None else self._composition.service.health()

    async def describe_now(self, camera_id: str) -> UUID:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return await self._composition.service.describe_now(camera_id)

    async def create_camera(self, create: CameraCreate) -> CameraConfig:
        """Add a camera, persist it, and start watching — no restart.

        `update_camera`'s ordering, for `update_camera`'s reasons, with one extra step
        at the front:

          1. **refuse a duplicate id first**, from memory, so a clash is an error that
             never opened the file;
          2. **refuse capabilities this process cannot honour**, before the write. Which
             models exist is fixed at startup, so a camera asking for one that was never
             loaded must be refused rather than added in a state where it silently
             detects nothing;
          3. **persist**, and only then;
          4. **build and start the runner, from the record the store read back** — not
             from the request, so memory and file cannot disagree about normalisation.

        A failure at (3) leaves nothing started. A failure at (4) leaves a camera in the
        file that is not running, which is the one asymmetry here and is why it is last:
        everything that can be checked cheaply has been, so reaching (4) and failing
        means the source itself would not build, and the record on disk is what an
        operator needs in order to fix it.
        """
        if self._composition is None:
            raise EngineNotComposedError(
                "the engine is still starting; a camera cannot be added yet"
            )
        if create.camera_id in {camera.camera_id for camera in self._composition.service.cameras()}:
            raise DuplicateCameraError(create.camera_id)
        if create.capabilities is not None:
            # `ensure_roles_available`, not `ensure_capabilities_available`: the
            # latter asks whether the camera is running first, and this one is not
            # running because it does not exist yet.
            self._composition.service.ensure_roles_available(create.camera_id, create.capabilities)

        record = await self._composition.camera_store.create(create)
        runner = build_runner(
            record,
            settings=self._composition.settings,
            models=self._composition.models,
            face_store=self._composition.face_store,
            scheduler=self._composition.service.scheduler,
            clip_writer=self._composition.clip_writer,
        )
        self._composition.service.add_camera(runner)
        logger.info(
            "added camera %s (%s) with capabilities %s",
            record.camera_id,
            record.label,
            sorted(c.value for c in Capability if record.capabilities.enabled(c)) or ["none"],
        )
        return record

    async def snapshot(self, camera_id: str) -> bytes | None:
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        return await self._composition.service.snapshot(camera_id)

    async def probe_source(self, url: str) -> ProbeResult:
        """Look at a URL without adding it. Available before composition finishes,
        deliberately: probing touches nothing the engine owns, and an operator filling in
        the add-camera form during a slow startup should not be made to wait."""
        return await probe_source(url)

    async def set_camera_enabled(self, camera_id: str, enabled: bool) -> CameraConfig:
        """Start or stop watching a camera, and write that decision down.

        The two directions are deliberately not mirror images, because the failure that
        matters is different in each:

        * **Disabling stops first, then writes.** `delete_camera`'s rule — never leave
          something running that the file does not describe. A write-first failure
          would record the camera as off while it kept watching and publishing.
        * **Enabling writes first, then starts.** `create_camera`'s rule — never start
          something the file does not describe. A start-first failure would leave a
          camera watching that the next restart would not bring back, which looks like
          it worked until somebody restarts.

        Both leave the file and the running set consistent on failure, in the direction
        that under-promises rather than over-promises.

        Idempotent in both directions: the file is rewritten to say what it already
        said, and the service's own toggles absorb the no-op.
        """
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        service = self._composition.service
        # Before anything: an id neither running nor stopped is a 404 that never opened
        # the file. `telemetry` answers for both sets.
        service.telemetry(camera_id)

        if not enabled:
            await service.disable_camera(camera_id)
            return await self._composition.camera_store.set_enabled(camera_id, False)

        record = await self._composition.camera_store.set_enabled(camera_id, True)
        # The frozen snapshot goes before the runner is added, or `cameras()` would
        # briefly report this camera twice.
        service.forget_disabled(camera_id)
        service.add_camera(
            build_runner(
                record,
                settings=self._composition.settings,
                models=self._composition.models,
                face_store=self._composition.face_store,
                scheduler=service.scheduler,
                clip_writer=self._composition.clip_writer,
            )
        )
        logger.info("enabled camera %s (%s)", record.camera_id, record.label)
        return record

    async def delete_camera(self, camera_id: str) -> None:
        """Stop a camera and remove it from `cameras.json`.

        Stop **before** the write, the opposite order to `create_camera`, and for the
        same underlying rule: leave nothing running that the file does not describe. A
        write-first failure would remove the record while the camera kept watching,
        publishing events under an id nothing can look up.
        """
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        service = self._composition.service
        # A disabled camera has no runner to stop — `remove_camera` would raise
        # `UnknownCameraError` and turn deleting a switched-off camera into a 404 for a
        # camera the operator can plainly see in the list.
        service.telemetry(camera_id)
        if service.is_running(camera_id):
            await service.remove_camera(camera_id)
        service.forget_disabled(camera_id)
        await self._composition.camera_store.remove(camera_id)
        logger.info("deleted camera %s", camera_id)

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
        success in (2) is always followed by (3) — the apply is synchronous, has no
        await inside it, and is all-or-nothing by construction (`apply_metadata`
        resolves every fallible value before it writes any field), so there is no
        window where the file has moved and the camera has not, and no half-applied
        state for a failure here to leave behind.
        """
        if self._composition is None:
            raise UnknownCameraError(camera_id)
        before = self._composition.service.telemetry(camera_id)
        # Before the write, not after: a capability whose model this process never
        # loaded cannot be honoured, and discovering that after `cameras.json` has
        # been rewritten would leave the record and the running camera disagreeing.
        # See `EngineService.ensure_capabilities_available`.
        if edit.capabilities is not UNSET:
            self._composition.service.ensure_capabilities_available(camera_id, edit.capabilities)
        record = await self._composition.camera_store.apply(camera_id, edit)
        self._composition.service.update_camera_metadata(
            camera_id,
            label=record.label,
            zone=record.zone,
            capabilities=record.capabilities,
            notify_on=record.notify_on,
            notify_min_confidence=record.notify_min_confidence,
            clip_preroll_seconds=record.clip_preroll_seconds,
            clip_postroll_seconds=record.clip_postroll_seconds,
            summary_interval_seconds=record.summary_interval_seconds,
        )
        # Old value logged alongside the new one, for every field this endpoint can
        # change: this endpoint is unauthenticated (see the module docstring), so the
        # log line is the only record of what a camera used to be once the write above
        # overwrites it in the file. Without it, an anonymous caller could rename a
        # camera to something misleading and nobody could reconstruct what the console
        # said about that location a minute earlier — and, since Task 8, could mute a
        # camera's welfare notifications with even less trace, because a camera that
        # has stopped telling anyone about a collapse looks exactly like a camera with
        # nothing to report.
        logger.info(
            "camera %s reconfigured: label=%r->%r zone=%s->%s notify_on=%s->%s "
            "notify_min_confidence=%s->%s clip_preroll_seconds=%s->%s "
            "clip_postroll_seconds=%s->%s summary_interval_seconds=%s->%s",
            camera_id,
            before.label,
            record.label,
            before.zone,
            record.zone,
            _rendered_kinds(before.notify_on),
            _rendered_kinds(record.notify_on),
            before.notify_min_confidence,
            record.notify_min_confidence,
            before.clip_preroll_seconds,
            record.clip_preroll_seconds,
            before.clip_postroll_seconds,
            record.clip_postroll_seconds,
            before.summary_interval_seconds,
            record.summary_interval_seconds,
        )
        return record


def create_default_app() -> FastAPI:
    """The app `uvicorn sentinel_ai.main:app` serves."""
    settings = get_settings()

    def build() -> Composition:
        cameras = load_cameras(Path(settings.cameras_file))
        # The union over every camera, asked *before* anything is constructed: this
        # is the whole of §13's "avoid loading unnecessary models". A site whose
        # cameras all run triggers-only never downloads a VLM checkpoint.
        # Every camera's capabilities, including the disabled ones. Loading only what
        # the running cameras need would mean re-enabling a camera could ask for a
        # model this process never loaded — and the refusal would arrive at the moment
        # an operator was switching a camera back on, which is the worst time to
        # discover it. A disabled camera's models cost VRAM they are not using; that is
        # the price of being able to switch it on without a restart.
        roles = required_roles(*(camera.capabilities for camera in cameras))
        logger.info(
            "composing engine for %d camera(s); model roles required: %s",
            len(cameras),
            ", ".join(sorted(role.value for role in roles)) or "none",
        )
        dead_letter = build_dead_letter(settings)
        return compose(
            settings,
            cameras,
            build_models(settings, roles),
            build_publisher(settings),
            build_clip_writer(settings),
            dead_letter,
            face_store=build_face_store(settings, roles),
            alert_store=build_alert_store(settings),
            # The same spool the scheduler dead-letters events to. One directory,
            # two record shapes: `DeadLetterSpool.store` already accepts either, and
            # a second spool would only split the "nothing else worked" pile in two.
            notifier=build_notifier(settings, dead_letter),
        )

    if settings.enable_camera_writes:
        logger.warning(
            "camera writes are ENABLED and this engine has no authentication: anyone "
            "who can reach this port can relabel and re-zone cameras. Bind it to "
            "localhost or put an authenticating proxy in front of it."
        )
    return create_app(ComposedService(build), camera_writes_enabled=settings.enable_camera_writes)


app = create_default_app()
