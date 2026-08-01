"""RabbitMQPublisher's disk spool, exercised with no broker running (spec
§6, §9): everything that must survive a dead broker and a process restart.
A live-broker round trip is `@pytest.mark.integration`
(test_rabbitmq_integration.py) and excluded from CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import aio_pika
import pytest

from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore


class _FakeExchange:
    """Stands in for `aio_pika`'s exchange: records publishes, never touches
    a socket. Assigned directly to `_exchange` so `replay_spool()` and
    `publish()` are testable without a real `connect()`."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def publish(self, message: aio_pika.Message, routing_key: str) -> None:
        body = json.loads(message.body.decode("utf-8"))
        self.published.append((routing_key, body))


def _event(camera_id: str = "cam-1", **overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": camera_id,
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.7),
        "description": "A person is running.",
        "suggested_action": "Review the clip.",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


async def test_publish_without_a_connection_spools_to_disk(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )

    await publisher.publish(_event())

    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_two_events_in_the_same_instant_replay_in_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-second (here, same-nanosecond) collision: the filename's
    sequence tie-breaker, not wall-clock resolution, must decide order.

    The two event_ids are pinned so that plain hex-lexical order is the
    *reverse* of call order (`ff...` > `00...`). A spool naming scheme that
    dropped the sequence number and fell back to `<ts_ns>-<event_id.hex>`
    would sort the second-published event first here, deterministically
    failing this assertion instead of merely doing so by chance of a random
    UUID4 draw.
    """
    import sentinel_ai.adapters.publishers.rabbitmq as rabbitmq_module

    monkeypatch.setattr(rabbitmq_module.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    first_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    second_id = UUID("00000000-0000-0000-0000-000000000000")
    first = _event(occurred_at=1.0, event_id=first_id)
    second = _event(occurred_at=2.0, event_id=second_id)

    await publisher.publish(first)
    await publisher.publish(second)

    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]
    await publisher.replay_spool()

    replayed_ids = [body["event_id"] for _routing_key, body in fake_exchange.published]
    assert replayed_ids == [str(first.event_id), str(second.event_id)]
    assert list(tmp_path.glob("*.json")) == [], "successfully replayed spool files are removed"


async def test_a_corrupt_spool_file_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    good = tmp_path / "10000000000000000000-00000001-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json"
    good.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_id": str(uuid4()),
                "camera_id": "cam-1",
                "occurred_at": 1.0,
                "reason": "speed_anomaly",
                "threat_score": 0.7,
                "severity": "high",
                "description": "d",
                "suggested_action": "a",
                "labels": [],
                "track_ids": [],
                "keyframe_uri": None,
                "clip_uri": None,
                "description_unavailable": False,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    corrupt = tmp_path / "20000000000000000000-00000001-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.json"
    corrupt.write_text("{not valid json", encoding="utf-8")

    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]

    await publisher.replay_spool()  # must not raise

    assert len(fake_exchange.published) == 1, "the valid file replayed"
    assert not good.exists(), "a replayed file is deleted"
    assert corrupt.with_suffix(".json.corrupt").exists(), (
        "the corrupt file is set aside, not retried forever"
    )


async def test_replay_before_connect_raises_a_clear_error(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="connect"):
        await publisher.replay_spool()


async def test_routing_key_carries_camera_and_reason(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]

    await publisher.publish(_event(camera_id="cam-7", reason=EscalationReason.DWELL_EXCEEDED))

    routing_key, _body = fake_exchange.published[0]
    assert routing_key == "anomaly.cam-7.dwell_exceeded"


async def test_a_broker_failure_mid_replay_leaves_remaining_files_for_next_time(
    tmp_path: Path,
) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    await publisher.publish(_event())
    await publisher.publish(_event())

    class _FlakyExchange(_FakeExchange):
        async def publish(self, message: aio_pika.Message, routing_key: str) -> None:
            raise ConnectionError("broker dropped again")

    publisher._exchange = _FlakyExchange()  # type: ignore[assignment]
    await publisher.replay_spool()  # must not raise

    assert len(list(tmp_path.glob("*.json"))) == 2, "nothing was deleted on a failed replay"


async def test_a_wrong_shape_spool_file_is_set_aside_not_left_to_wedge_the_spool(
    tmp_path: Path,
) -> None:
    """Valid JSON of the wrong shape must not strand every event behind it.

    `["abc"]` parses cleanly, then raises ValueError inside validate_payload's
    dict() conversion. Before the guard widened, that escaped replay_spool AND
    left the file in place — so it sorted first every time and re-crashed on
    every subsequent attempt, permanently blocking the healthy events queued
    behind it. Spec §9 says an event is never lost to an infrastructure
    failure; a spool that can never drain loses all of them.
    """
    publisher = RabbitMQPublisher(
        url="amqp://unused", exchange="sentinel.events", spool_dir=tmp_path
    )
    # Sorts first: an all-zero timestamp prefix puts it ahead of the real event.
    (tmp_path / "00000000000000000000-00000000-deadbeef.json").write_text(
        '["abc"]', encoding="utf-8"
    )
    await publisher.publish(_event())
    assert len(list(tmp_path.glob("*.json"))) == 2

    exchange = _FakeExchange()
    publisher._exchange = exchange  # type: ignore[assignment]
    await publisher.replay_spool()

    assert len(exchange.published) == 1, "the healthy event was stranded behind the bad file"
    assert list(tmp_path.glob("*.json")) == [], "spool did not drain"
    assert len(list(tmp_path.glob("*.json.corrupt"))) == 1, "bad file was not set aside"
