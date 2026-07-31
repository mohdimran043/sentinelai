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

    vram_total_mib: int = Field(default=8192, gt=0)
    vram_reserved_mib: int = Field(default=2048, ge=0)

    detector_model_id: str = "yolo11s.pt"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    vlm_idle_unload_seconds: float = Field(default=600.0, gt=0)

    decode_hwaccel: str | None = None

    rabbitmq_url: str = "amqp://sentinel:sentinel@localhost:5672/"
    rabbitmq_exchange: str = "sentinel.events"
    event_spool_dir: str = "./var/spool/events"

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
