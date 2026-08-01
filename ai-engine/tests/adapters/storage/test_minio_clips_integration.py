"""Live-MinIO round trip: requires `docker compose -f deploy/compose/docker-compose.core.yml
up -d minio` (spec §5.5). Excluded from CI by `-m "not integration"`, matching how
`test_rabbitmq_integration.py` is split out from `test_rabbitmq_spool.py` for RabbitMQ.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from sentinel_ai.adapters.storage.minio_clips import MinioClipWriter
from sentinel_ai.config import get_settings

from .test_minio_clips import _annexb_packets_from_fixture

pytestmark = pytest.mark.integration


async def test_finish_returns_a_uri_the_object_exists_at(tmp_path: Path) -> None:
    settings = get_settings()
    writer = MinioClipWriter(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        bucket=settings.minio_bucket,
        secure=settings.minio_secure,
        temp_dir=tmp_path,
    )
    camera_id, event_id = "cam-integration", uuid4()
    handle = await writer.open(camera_id, event_id, fps=25.0)
    for packet in _annexb_packets_from_fixture(camera_id):
        await handle.append(packet)
    uri = await handle.finish()

    assert uri == f"s3://{settings.minio_bucket}/{camera_id}/{event_id}.mp4"
    stat = await asyncio.to_thread(
        writer._client.stat_object, settings.minio_bucket, f"{camera_id}/{event_id}.mp4"
    )
    assert stat.size is not None and stat.size > 0
