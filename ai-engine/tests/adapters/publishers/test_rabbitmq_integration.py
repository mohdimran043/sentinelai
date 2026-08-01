"""Live-broker round trip: requires `docker compose -f deploy/compose/docker-compose.core.yml
up -d rabbitmq` (spec §2, Task 2). Excluded from CI by `-m "not integration"`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import pytest

from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.config import get_settings
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore

pytestmark = pytest.mark.integration


def _event() -> Event:
    return Event(
        event_id=uuid4(),
        camera_id="cam-integration",
        occurred_at=1_700_000_000.0,
        reason=EscalationReason.SPEED_ANOMALY,
        threat=ThreatScore.from_value(0.7),
        description="A person is running.",
        suggested_action="Review the clip.",
    )


async def test_publish_reaches_a_bound_queue(tmp_path: Path) -> None:
    settings = get_settings()
    publisher = RabbitMQPublisher(
        url=settings.rabbitmq_url, exchange=settings.rabbitmq_exchange, spool_dir=tmp_path
    )
    await publisher.connect()
    try:
        assert publisher._channel is not None
        assert publisher._exchange is not None
        queue = await publisher._channel.declare_queue(exclusive=True)
        await queue.bind(publisher._exchange, routing_key="anomaly.#")

        event = _event()
        await publisher.publish(event)

        message = await asyncio.wait_for(queue.get(), timeout=5.0)
        body = json.loads(message.body.decode("utf-8"))
        assert body["event_id"] == str(event.event_id)
        await message.ack()
    finally:
        await publisher.close()


async def test_spooled_events_replay_once_the_broker_is_reachable(tmp_path: Path) -> None:
    settings = get_settings()

    # An unreachable port forces every publish to spool, without touching the
    # real broker's lifecycle.
    down_publisher = RabbitMQPublisher(
        url="amqp://sentinel:sentinel@localhost:1/",
        exchange=settings.rabbitmq_exchange,
        spool_dir=tmp_path,
    )
    event = _event()
    await down_publisher.publish(event)
    assert len(list(tmp_path.glob("*.json"))) == 1

    up_publisher = RabbitMQPublisher(
        url=settings.rabbitmq_url, exchange=settings.rabbitmq_exchange, spool_dir=tmp_path
    )
    await up_publisher.connect()
    try:
        assert up_publisher._channel is not None
        assert up_publisher._exchange is not None
        queue = await up_publisher._channel.declare_queue(exclusive=True)
        await queue.bind(up_publisher._exchange, routing_key="anomaly.#")

        await up_publisher.replay_spool()

        message = await asyncio.wait_for(queue.get(), timeout=5.0)
        body = json.loads(message.body.decode("utf-8"))
        assert body["event_id"] == str(event.event_id)
        await message.ack()
    finally:
        await up_publisher.close()
    assert list(tmp_path.glob("*.json")) == []
