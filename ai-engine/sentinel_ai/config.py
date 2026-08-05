"""Runtime configuration. This is the ONLY place the Development/Production seam is chosen."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(StrEnum):
    """Spec §3.1/§3.2. Development binds in-process transport; Production binds gRPC."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class NotifierKind(StrEnum):
    """Which `Notifier` the composition root builds (spec §5, T10).

    An enum rather than "a webhook URL is set, so use a webhook": those are two
    different facts, and conflating them makes turning notifications off require
    deleting the URL — so an operator silencing a site for an afternoon has to
    keep the credential somewhere else and paste it back. It also gives the
    unreachable-webhook case a name: `webhook` with no URL is a configuration
    error a deployment can be told about, where the implicit form would silently
    fall back to logging and look like it worked.
    """

    LOGGING = "logging"
    WEBHOOK = "webhook"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    mode: Mode = Mode.DEVELOPMENT

    cameras_file: str = "./cameras.json"
    """Where `sentinel_ai.main` reads its camera list from (see that module's
    docstring for the file's shape). A file rather than environment variables:
    a camera is a nested record with a per-camera `CameraProfile`, and flattening
    a list of those into `SENTINEL_*` names is a worse interface than one small
    JSON document that can be diffed, reviewed and mounted into a container."""

    enable_camera_writes: bool = False
    """Whether `PATCH /cameras/{id}` may change anything. **Off by default.**

    This engine has no authentication — the console's login screen is a shell and
    JWT is Phase 1C — so every endpoint is open to whatever can reach the port. For
    the read endpoints that is a disclosure question; for a write endpoint it is a
    control question, and the two are not the same size. A camera's label and zone
    are what an operator navigates by and what a console groups by, so an anonymous
    caller who can rewrite them can make a camera look like a different camera, and
    the change persists to `cameras.json` and survives the restart that would
    otherwise undo it.

    Default-off means a deployment that has not thought about this is not writable
    by anyone who finds it, and the operator who turns it on is the one who decided
    the port is reachable only by people who should be able to do this. The route
    still exists in the OpenAPI document either way — a contract that changes shape
    with a runtime flag is worse than a documented 403 — and answers 403 while this
    is false. See `docs/operations.md`.
    """

    device: str | None = None
    """Torch device for both models. `None` auto-detects via
    `yolo11.select_device()` (cuda when visible, else cpu). Set it explicitly to
    pin a device or to force CPU on a box that has a GPU."""

    vram_total_mib: int = Field(default=8192, gt=0)
    vram_reserved_mib: int = Field(default=2048, ge=0)

    detector_model_id: str = "yolo11s.pt"
    detector_vram_mib: int = Field(default=432, gt=0)
    """Startup estimate for `ModelSpec.vram_mib`, replaced by the measured
    `capabilities().vram_mib` once `warmup()` has run (see
    `main.refresh_specs_from_capabilities`). A configured seed is unavoidable:
    `plan_residency()` has to decide whether a model fits *before* it is loaded,
    and nothing can measure it before then. The default is the figure Task 11
    measured on an RTX 4060 (`memory_reserved()` + CUDA-context overhead)."""

    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    vlm_vram_mib: int = Field(default=2766, gt=0)
    """The same startup estimate for the VLM; Task 13's measured figure."""

    vlm_idle_unload_seconds: float = Field(default=600.0, gt=0)

    decode_hwaccel: str | None = None

    rabbitmq_url: str = "amqp://sentinel:sentinel@localhost:5672/"
    rabbitmq_exchange: str = "sentinel.events"
    event_spool_dir: str = "./var/spool/events"
    broker_replay_interval_seconds: float = Field(default=30.0, gt=0)
    """How often `main.BrokerLink` re-drains the spool while the broker is up.
    Spec §9's disk buffer is only half a guarantee without something that replays
    it; this is the period of the component that does."""

    dead_letter_dir: str = "./var/spool/dead-letter"
    """Last resort for an event the publisher itself would not take (spec §9).
    Distinct from `event_spool_dir`, which holds validated payloads waiting for a
    broker that is merely down; this one holds events that failed for a reason
    replay cannot fix, so mixing them would strand the healthy spool."""

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "sentinel"
    minio_secret_key: str = "sentinel123"
    minio_bucket: str = "sentinel-clips"
    minio_secure: bool = False

    clip_preroll_seconds: float = Field(default=3.0, ge=0)
    clip_postroll_seconds: float = Field(default=5.0, gt=0)

    detector_conf_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    detector_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    detector_imgsz: int = Field(default=640, gt=0)

    vlm_queue_maxsize: int = Field(default=4, ge=1)
    vlm_timeout_seconds: float = Field(default=30.0, gt=0)
    vlm_max_new_tokens: int = Field(default=256, gt=0)
    vlm_global_concurrency: int = Field(default=1, ge=1)
    vlm_global_min_interval_seconds: float = Field(default=2.0, ge=0)

    detect_every_n_frames: int = Field(default=1, ge=1)
    source_realtime: bool = True
    rtsp_reconnect_initial_seconds: float = Field(default=1.0, gt=0)
    rtsp_reconnect_max_seconds: float = Field(default=30.0, gt=0)

    clip_temp_dir: str = "./var/clips"

    notifier_kind: NotifierKind = NotifierKind.LOGGING
    """Which welfare notifier the engine delivers through. **Logging by default.**

    Not `None`: there is no "no notifier" state. `LoggingNotifier` touches no
    network and needs no configuration, so a deployment that has thought about
    nothing still leaves a trail an operator can tail — and every routing decision
    the engine makes is observable somewhere rather than only in the absence of an
    alert nobody was expecting."""

    notifier_webhook_url: str | None = None
    """Where `notifier_kind=webhook` POSTs. Required by that kind and ignored by
    every other.

    **Treat this as a credential.** ntfy and Slack both put a per-recipient token
    in the URL path, which is why `WebhookNotifier` never logs it, never follows a
    redirect that could re-send it elsewhere, and why the composition root installs
    httpx's log redaction around it (see `main.build_notifier`)."""

    notifier_timeout_seconds: float = Field(default=20.0, gt=0)
    """Wall-clock ceiling on **one whole notification**, retries included — not the
    per-request HTTP timeout.

    The distinction matters, because the two numbers pull opposite ways.
    `WebhookNotifier` bounds each individual attempt at its own 5s and retries a
    transient failure up to three times with backoff, a documented worst case of
    18.0 seconds; this is the outer deadline `NotificationDispatcher` enforces
    around all of that, so it must sit *above* the adapter's worst case or it
    cancels the retries midway and turns every transient 429 into a lost note.
    Hence 20.0 rather than the 5.0 an operator reading "timeout" as "HTTP timeout"
    would reach for.

    It exists at all because `Notifier` the port promises no bound of its own: the
    webhook adapter happens to bound itself, a future adapter need not, and neither
    may park the single notification worker forever."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
