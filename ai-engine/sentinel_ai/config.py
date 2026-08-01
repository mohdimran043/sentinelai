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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    mode: Mode = Mode.DEVELOPMENT

    cameras_file: str = "./cameras.json"
    """Where `sentinel_ai.main` reads its camera list from (see that module's
    docstring for the file's shape). A file rather than environment variables:
    a camera is a nested record with a per-camera `CameraProfile`, and flattening
    a list of those into `SENTINEL_*` names is a worse interface than one small
    JSON document that can be diffed, reviewed and mounted into a container."""

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
