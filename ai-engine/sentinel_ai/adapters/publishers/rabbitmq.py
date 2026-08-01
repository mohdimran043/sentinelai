"""RabbitMQ EventPublisher with a disk-backed spool (spec §5.6, §9).

Governing rule (spec §9): an anomaly event is never lost to an
infrastructure failure. When the broker is unreachable, the already-validated
payload is written to `spool_dir` and replayed once `connect()` succeeds
again. `connect()` and `replay_spool()` are separate, explicit calls (S13) —
this publisher never retries on its own; the caller (a reconnect loop with
its own backoff policy) decides when each runs.

Spool file naming: `<ts_ns:020d>-<seq:08d>-<event_id.hex>.json`. The
zero-padded nanosecond wall-clock timestamp sorts chronologically across
process restarts — wall time keeps advancing across a crash, unlike the
domain's injected monotonic clock, which resets to 0 on the next process. The
sequence number breaks ties when two events land in the same nanosecond
(a real possibility: two cameras can both escalate inside one scheduler
tick). Both are baked into the filename, so `sorted(glob(...))` alone gives
replay order — no separate index file to keep consistent with the spool
directory.

Partial-write detection is write-then-rename, not a checksum: the file is
written to a `.json.tmp` sibling and moved into place with `Path.replace`,
which is an atomic rename on the same filesystem. A `.json` file therefore
either doesn't exist yet or is complete — `replay_spool`'s `*.json` glob can
never observe a half-written one. A checksum would also work, but it requires
computing and storing a hash for every payload just to protect against a
window that write-then-rename closes for free.
"""

from __future__ import annotations

import json
import logging
import time as time  # re-exported: tests monkeypatch `rabbitmq.time.time_ns`
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection
from aio_pika.exceptions import AMQPError
from jsonschema import ValidationError

from sentinel_ai.adapters.serialization.event_codec import encode_event, validate_payload
from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.event_publisher import EventPublisher

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (AMQPError, OSError, ConnectionError)


def _routing_key(camera_id: str, reason: str) -> str:
    """`anomaly.<camera_id>.<reason>` on the topic exchange — lets a consumer
    bind on one camera, one reason across all cameras, or everything
    (`anomaly.#`) without the publisher knowing which."""
    return f"anomaly.{camera_id}.{reason}"


class RabbitMQPublisher(EventPublisher):
    def __init__(self, url: str, exchange: str, spool_dir: Path) -> None:
        self._url = url
        self._exchange_name = exchange
        self._spool_dir = spool_dir
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._spool_seq = 0

    async def connect(self) -> None:
        """Open the connection, channel and durable topic exchange.

        Raises on failure; the caller decides whether and when to retry.
        Durable exchange + persistent messages (in `_publish_payload`) mean a
        broker restart drops neither the topology nor an in-flight message.
        """
        connection = await aio_pika.connect_robust(self._url)
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            self._exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        self._connection = connection
        self._channel = channel
        self._exchange = exchange

    async def publish(self, event: Event) -> None:
        payload = encode_event(event)  # validates against the committed schema
        routing_key = _routing_key(event.camera_id, event.reason.value)
        if self._exchange is None:
            self._spool(event.event_id, payload)
            return
        try:
            await self._publish_payload(routing_key, payload)
        except _TRANSPORT_ERRORS:
            logger.warning("broker unreachable, spooling event %s", event.event_id)
            self._spool(event.event_id, payload)

    async def replay_spool(self) -> None:
        """Publish every spooled payload, oldest first, deleting each as it
        succeeds. Stops at the first broker failure so the remaining files
        wait for the next call, and sets aside (rather than crashes on) a
        corrupt file so it is not retried forever.
        """
        if self._exchange is None:
            raise RuntimeError("replay_spool() called before connect() succeeded")
        for path in sorted(self._spool_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    # Valid JSON of the wrong shape: `["abc"]` parses fine, then blows up
                    # inside validate_payload's dict() conversion. Rejecting it here keeps
                    # the guard about the shape rather than about which exception type a
                    # downstream call happens to raise.
                    raise TypeError(f"spool payload must be a JSON object, got {type(payload)}")
                validate_payload(payload)
            except (
                json.JSONDecodeError,
                ValidationError,
                KeyError,
                TypeError,
                ValueError,
                OSError,
            ):
                # ValueError matters as much as the rest: without it a wrong-shape file
                # crashed replay AND was left in place, so it stayed first in sort order
                # and re-crashed on every later attempt — permanently stranding every
                # healthy event queued behind it. Spec §9 says events are never lost to an
                # infrastructure failure, and an un-drainable spool loses all of them.
                logger.warning("skipping corrupt spool file %s", path.name, exc_info=True)
                path.rename(path.with_suffix(".json.corrupt"))
                continue

            routing_key = _routing_key(str(payload["camera_id"]), str(payload["reason"]))
            try:
                await self._publish_payload(routing_key, payload)
            except _TRANSPORT_ERRORS:
                logger.warning("broker dropped mid-replay, stopping at %s", path.name)
                return
            path.unlink(missing_ok=True)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def _publish_payload(self, routing_key: str, payload: Mapping[str, object]) -> None:
        assert self._exchange is not None
        message = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
        )
        await self._exchange.publish(message, routing_key=routing_key)

    def _spool(self, event_id: UUID, payload: Mapping[str, object]) -> None:
        """Persist a payload for later replay.

        Ordering caveat: the filename's primary sort key is wall-clock
        `time.time_ns()`, because it is the only clock that keeps advancing across a
        process restart — the injected monotonic clock the rest of the system uses
        restarts at zero. The cost is that an NTP step correction can move the wall
        clock backwards mid-run, so events spooled after the jump may sort before
        events spooled before it. No event is lost — every file is still replayed —
        but replay order can differ from occurrence order across such a jump.
        `_spool_seq` only breaks ties within an identical nanosecond, so it does not
        rescue this case.
        """
        self._spool_seq += 1
        base = f"{time.time_ns():020d}-{self._spool_seq:08d}-{event_id.hex}"
        final_path = self._spool_dir / f"{base}.json"
        tmp_path = self._spool_dir / f"{base}.json.tmp"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(final_path)  # atomic rename on the same filesystem
