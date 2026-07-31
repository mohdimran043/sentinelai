"""Smoke test for deploy/compose/docker-compose.core.yml.

Excluded from CI: CI runs `-m "not gpu and not integration"` (see pyproject.toml's
`[tool.pytest.ini_options]` markers and the Global Constraints). Run manually after
`make up`:

    cd ai-engine && . .venv/bin/activate && python -m pytest -m integration \
        tests/integration/test_compose_services.py -v
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse

import pytest

from sentinel_ai.config import get_settings

# Postgres and Redis have no sentinel_ai.config settings yet (spec: the Go backend owns
# them from Phase 1C onward) — these mirror docker-compose.core.yml's port mappings
# directly rather than a setting that does not exist.
POSTGRES_HOST_PORT = ("localhost", 5432)
REDIS_HOST_PORT = ("localhost", 6379)
RABBITMQ_MANAGEMENT_HOST_PORT = ("localhost", 15672)
MEDIAMTX_RTSP_HOST_PORT = ("localhost", 8554)

TIMEOUT_SECONDS = 3.0


def _assert_reachable(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=TIMEOUT_SECONDS):
        pass


@pytest.mark.integration
def test_postgres_is_reachable() -> None:
    _assert_reachable(*POSTGRES_HOST_PORT)


@pytest.mark.integration
def test_redis_is_reachable() -> None:
    _assert_reachable(*REDIS_HOST_PORT)


@pytest.mark.integration
def test_rabbitmq_amqp_is_reachable() -> None:
    settings = get_settings()
    parsed = urlparse(settings.rabbitmq_url)
    assert parsed.hostname is not None, f"unparseable rabbitmq_url: {settings.rabbitmq_url!r}"
    _assert_reachable(parsed.hostname, parsed.port or 5672)


@pytest.mark.integration
def test_rabbitmq_management_ui_is_reachable() -> None:
    _assert_reachable(*RABBITMQ_MANAGEMENT_HOST_PORT)


@pytest.mark.integration
def test_minio_is_reachable() -> None:
    settings = get_settings()
    host, _, port = settings.minio_endpoint.partition(":")
    assert port, f"minio_endpoint has no port: {settings.minio_endpoint!r}"
    _assert_reachable(host, int(port))


@pytest.mark.integration
def test_mediamtx_rtsp_port_is_reachable() -> None:
    _assert_reachable(*MEDIAMTX_RTSP_HOST_PORT)
