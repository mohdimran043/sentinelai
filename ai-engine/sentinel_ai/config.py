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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
