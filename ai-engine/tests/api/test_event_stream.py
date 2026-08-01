"""`GET /events/stream` (T2): the SSE bridge until Phase 1C's WebSocket hub exists.

Nothing here sleeps. Every test either arranges the writes it cares about *before*
reading a frame — a queued write is returned without ever awaiting — or yields to the
loop explicitly with `asyncio.sleep(0)` when it needs a parked reader to observe
something. A stream test that leans on wall-clock time is a stream test that passes
for the wrong reason on a loaded CI box.

The properties under test are the ones that make a stream usable rather than a demo:
a client connecting mid-life gets the backlog and then live events with no gap and no
duplicate; a client that stops reading is bounded and cut off rather than allowed to
grow the engine's heap or wedge the writer; a stream ends when its client leaves and
when the engine shuts down; and an event that is legitimately re-sent (its clip
arrived) is not mistaken for a duplicate.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from sentinel_ai.api.app import create_app
from sentinel_ai.api.sse import (
    SSE_MEDIA_TYPE,
    event_stream_body,
    format_comment,
    format_frame,
)
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore
from sentinel_ai.orchestrator.event_history import (
    CameraEventHistory,
    EventSubscription,
    RecentEventLog,
)
from sentinel_ai.orchestrator.service import EngineNotComposedError, UnknownCameraError
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport


def _event(
    camera_id: str = "cam-1",
    *,
    occurred_at: float = 0.0,
    event_id: UUID | None = None,
    clip_uri: str | None = None,
) -> Event:
    return Event(
        event_id=event_id or uuid4(),
        camera_id=camera_id,
        occurred_at=occurred_at,
        source_timestamp=occurred_at,
        reason=EscalationReason.NEW_SALIENT_TRACK,
        threat=ThreatScore.from_value(0.5),
        description="A person walks by.",
        suggested_action="Review the clip.",
        labels=("person",),
        track_ids=(7,),
        clip_uri=clip_uri,
    )


def parse(frame: str) -> tuple[str, object]:
    """(event name, decoded payload) from one frame. Rejects anything malformed so a
    test cannot pass against a frame no client could read."""
    lines = frame.split("\n")
    assert lines[-2:] == ["", ""], f"a frame must end with a blank line: {frame!r}"
    assert lines[0].startswith("event: "), f"no event name: {frame!r}"
    data = [line.removeprefix("data: ") for line in lines[1:-2]]
    assert data, f"no data lines: {frame!r}"
    return lines[0].removeprefix("event: "), json.loads("\n".join(data))


async def next_frame(body: AsyncGenerator[str, None]) -> str:
    """The next frame, bounded.

    A stream test that waits forever for a frame that never comes hangs the suite
    instead of failing it, and a hung suite reports nothing. The bound never elapses
    while the behaviour holds — every frame these tests expect is already queued when
    they ask for it, so nothing here waits on real time.
    """
    return await asyncio.wait_for(anext(body), timeout=1.0)


async def frames(body: AsyncGenerator[str, None], count: int) -> list[str]:
    return [await next_frame(body) for _ in range(count)]


async def yield_to_the_loop() -> None:
    """One turn of the event loop — not a wait. `asyncio.sleep(0)` reschedules
    immediately, so a parked reader gets to run without any time passing."""
    await asyncio.sleep(0)


class TestFraming:
    """The bytes themselves. The console's parser (`web/src/recorder/sse.ts`) splits on
    a blank line and reads `event:`/`data:` fields, so a frame that gets this wrong is
    invisible or, worse, silently merged into its neighbour."""

    def test_a_frame_carries_its_name_and_payload_and_ends_with_a_blank_line(self) -> None:
        assert format_frame("anomaly", '{"a":1}') == 'event: anomaly\ndata: {"a":1}\n\n'

    def test_a_multi_line_payload_becomes_several_data_lines(self) -> None:
        """A raw newline inside a payload would end the field and split one frame into
        two malformed ones."""
        assert format_frame("backlog", "one\ntwo") == "event: backlog\ndata: one\ndata: two\n\n"

    def test_a_comment_carries_no_data_field(self) -> None:
        assert format_comment("keepalive") == ": keepalive\n\n"


class TestBacklogThenLive:
    async def test_the_stream_opens_with_a_backlog_array_then_streams_new_events(self) -> None:
        log = RecentEventLog(capacity=10)
        first = _event(occurred_at=1.0)
        log.record(first)

        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        name, backlog = await _first(body)
        assert name == "backlog"
        assert isinstance(backlog, list)
        assert [entry["event_id"] for entry in backlog] == [str(first.event_id)]

        live = _event(occurred_at=2.0)
        log.record(live)
        name, payload = parse(await next_frame(body))
        assert name == "anomaly"
        assert isinstance(payload, dict)
        assert payload["event_id"] == str(live.event_id)
        await body.aclose()

    async def test_a_client_connecting_mid_life_sees_every_event_exactly_once(self) -> None:
        """The handover, and the whole point of T2's sequencing.

        The window between subscribing and reading the backlog is real: the route
        handler subscribes, and Starlette starts the response body afterwards. An
        event written in that window is written into this client's queue *and* into
        the backlog it is about to be handed.

        Fails against a snapshot-then-register order (the event vanishes — a gap, and
        a silent one) and against dropping the watermark filter (it arrives twice).
        """
        log = RecentEventLog(capacity=50)
        before = [_event(occurred_at=float(index)) for index in range(3)]
        for event in before:
            log.record(event)

        subscription = log.subscribe()  # the handler registers...
        during = _event(occurred_at=3.0)
        log.record(during)  # ...and this lands before the body has run

        body = event_stream_body(subscription, keepalive_seconds=None)
        _, backlog = await _first(body)
        assert isinstance(backlog, list)

        after = _event(occurred_at=4.0)
        log.record(after)
        live = [parse(frame)[1] for frame in await frames(body, 1)]

        seen = [entry["event_id"] for entry in backlog] + [
            entry["event_id"] for entry in live if isinstance(entry, dict)
        ]
        expected = [str(event.event_id) for event in [*before, during, after]]
        assert seen == expected, "no gap, no duplicate, and in order"
        assert len(set(seen)) == len(seen)
        assert subscription.suppressed_duplicates == 1, (
            "the copy of `during` that was queued before the backlog was taken must be "
            "the one that is discarded"
        )
        await body.aclose()

    async def test_a_second_client_starts_from_its_own_backlog(self) -> None:
        """Two consoles must not share a cursor: one opening late must not skip what
        it has never seen, and must not steal events from the other."""
        log = RecentEventLog(capacity=10)
        log.record(_event(occurred_at=1.0))
        first = event_stream_body(log.subscribe(), keepalive_seconds=None)
        await next_frame(first)

        log.record(_event(occurred_at=2.0))
        second = event_stream_body(log.subscribe(), keepalive_seconds=None)
        _, backlog = await _first(second)
        assert isinstance(backlog, list)
        assert len(backlog) == 2, "the late client's backlog is the whole current window"

        # ...and the first client is still live and unaffected.
        name, payload = parse(await next_frame(first))
        assert name == "anomaly"
        assert isinstance(payload, dict)
        assert payload["occurred_at"] == 2.0
        await first.aclose()
        await second.aclose()

    async def test_the_backlog_is_empty_rather_than_absent_on_a_quiet_engine(self) -> None:
        """An empty array is an answer. Omitting the frame would leave a console unable
        to tell "connected, nothing to show" from "still connecting"."""
        body = event_stream_body(RecentEventLog(capacity=5).subscribe(), keepalive_seconds=None)
        name, backlog = await _first(body)
        assert (name, backlog) == ("backlog", [])
        await body.aclose()

    async def test_every_cameras_events_arrive_on_the_one_stream(self) -> None:
        log = RecentEventLog(capacity=10)
        log.record(_event("cam-1", occurred_at=1.0))
        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        await next_frame(body)

        log.record(_event("cam-2", occurred_at=2.0))
        _, payload = parse(await next_frame(body))
        assert isinstance(payload, dict)
        assert payload["camera_id"] == "cam-2", (
            "a site-wide stream is useless if the consumer cannot tell which camera "
            "an event came from"
        )
        await body.aclose()


class TestRedeliveryVersusDuplicate:
    async def test_a_clip_arriving_later_is_re_sent_rather_than_suppressed(self) -> None:
        """T3 seen from the stream. A live client holds the pre-clip copy; unless the
        update reaches it, it shows an event with no footage forever while a client
        that connected a second later shows the link.

        This is the case a de-duplicator keyed on `event_id` gets wrong, which is why
        the wire carries `sequence`.
        """
        log = RecentEventLog(capacity=10)
        event = _event(occurred_at=1.0)
        log.record(event)
        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        _, backlog = await _first(body)
        assert isinstance(backlog, list)
        assert backlog[0]["clip_uri"] is None

        log.attach_clip("cam-1", event.event_id, "s3://clips/a.mp4")

        name, payload = parse(await next_frame(body))
        assert name == "anomaly"
        assert isinstance(payload, dict)
        assert payload["event_id"] == str(event.event_id), "the same event..."
        assert payload["clip_uri"] == "s3://clips/a.mp4", "...in a newer version"
        assert payload["sequence"] > backlog[0]["sequence"], (
            "the higher sequence is what tells a client to replace its copy rather "
            "than draw a second event"
        )
        await body.aclose()

    async def test_a_copy_the_backlog_already_carried_is_dropped(self) -> None:
        """The other half of the same rule, exercised directly on the subscription so
        the filter itself is pinned and not merely the arrangement that avoids needing
        it."""
        log = RecentEventLog(capacity=10)
        subscription = log.subscribe()
        log.record(_event(occurred_at=1.0))
        backlog = subscription.open()
        assert len(backlog) == 1
        assert subscription.pending == 1, "test setup: the same write is also queued"

        with pytest.raises(TimeoutError):
            # Zero, so nothing waits: the queued copy is discarded and the subscription
            # then has genuinely nothing to send.
            await subscription.next_event(timeout=0.0)

        assert subscription.suppressed_duplicates == 1
        assert subscription.pending == 0


class TestASlowOrVanishedClient:
    async def test_a_client_that_stops_reading_is_bounded_and_then_cut_off(self) -> None:
        """Not "buffered until the process dies". A suspended laptop's tab is
        indistinguishable from a slow one, so the engine must survive both."""
        log = RecentEventLog(capacity=500)
        subscription = log.subscribe(maxsize=4)
        body = event_stream_body(subscription, keepalive_seconds=None)
        await next_frame(body)  # the backlog; after this the client reads nothing

        for index in range(50):
            log.record(_event(occurred_at=float(index)))

        assert subscription.overflowed is True
        assert subscription.pending <= 4, "the queue is bounded by construction"

        name, payload = parse(await next_frame(body))
        assert name == "overflow"
        assert isinstance(payload, dict)
        assert "reconnect" in str(payload).lower(), "tell the client how to recover"
        with pytest.raises(StopAsyncIteration):
            await next_frame(body)

    async def test_a_stalled_client_never_wedges_or_breaks_the_writer(self) -> None:
        """`record()` runs on the scheduler's worker, between assembling an event and
        publishing it. A subscriber that cannot keep up must not be able to slow that
        down, and must certainly not raise into it."""
        log = RecentEventLog(capacity=500)
        stalled = log.subscribe(maxsize=2)
        stalled.open()
        healthy = log.subscribe(maxsize=100)
        healthy.open()

        for index in range(20):
            log.record(_event(occurred_at=float(index)))

        assert stalled.overflowed is True
        assert healthy.overflowed is False, "one bad client must not cut off a good one"
        assert len(log.history("cam-1").events) == 20, "and must not cost the ring an event"
        assert await healthy.next_event(timeout=None) is not None

    async def test_a_client_that_disconnects_stops_being_fed(self) -> None:
        """Starlette closes the generator when the client goes; the `finally` has to
        unregister, or every request leaves a queue behind for the life of the
        process."""
        log = RecentEventLog(capacity=10)
        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        await next_frame(body)
        assert log.subscribers == 1

        await body.aclose()

        assert log.subscribers == 0
        log.record(_event(occurred_at=1.0))  # must not raise, must not queue anywhere


class TestKeepalive:
    async def test_an_idle_stream_writes_a_comment_instead_of_nothing(self) -> None:
        """A zero interval, so this proves the behaviour without waiting for it: a
        proxy that sees no bytes for its idle timeout closes the connection, and the
        console then reconnects every timeout period forever."""
        log = RecentEventLog(capacity=10)
        body = event_stream_body(log.subscribe(), keepalive_seconds=0.0)
        await next_frame(body)

        assert await next_frame(body) == format_comment("keepalive")
        assert await next_frame(body) == format_comment("keepalive")

        log.record(_event(occurred_at=1.0))
        name, _ = parse(await next_frame(body))
        assert name == "anomaly", "a queued event is sent before any further keepalive"
        await body.aclose()


class TestShutdown:
    async def test_closing_the_streams_ends_a_parked_reader(self) -> None:
        """Shutdown must close streams, not leave them hanging. A reader parked on an
        engine that has stopped producing would otherwise wait until a proxy or the
        browser gave up."""
        log = RecentEventLog(capacity=10)
        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        await next_frame(body)

        reader: asyncio.Task[str] = asyncio.create_task(anext(body))  # deliberately unbounded
        await yield_to_the_loop()
        assert not reader.done(), "test setup: the reader must actually be parked"

        assert log.close_all() == 1

        with pytest.raises(StopAsyncIteration):
            # Bounded so that a stream which does *not* close fails this test in a
            # second instead of hanging the suite. The wait never elapses while the
            # behaviour holds: `close()` releases the reader synchronously.
            await asyncio.wait_for(reader, timeout=1.0)
        assert log.subscribers == 0

    async def test_closing_is_idempotent_and_survives_a_stream_that_already_ended(
        self,
    ) -> None:
        log = RecentEventLog(capacity=10)
        body = event_stream_body(log.subscribe(), keepalive_seconds=None)
        await next_frame(body)
        await body.aclose()

        assert log.close_all() == 0


class TestSubscriptionMisuse:
    async def test_opening_twice_is_rejected(self) -> None:
        """A second `open()` would move the watermark forward and silently skip
        everything queued in between."""
        subscription = RecentEventLog(capacity=5).subscribe()
        subscription.open()
        with pytest.raises(RuntimeError, match="already been opened"):
            subscription.open()

    async def test_reading_an_unopened_subscription_is_rejected(self) -> None:
        subscription = RecentEventLog(capacity=5).subscribe()
        with pytest.raises(RuntimeError, match="open"):
            await subscription.next_event(timeout=None)


# -- the HTTP surface ---------------------------------------------------------------


class _StreamingFakeService:
    """A service whose ring is real — the stream is only worth testing over the real
    thing — and whose streams are pre-closed so a `TestClient` request terminates."""

    def __init__(self, *, log: RecentEventLog | None = None, composed: bool = True) -> None:
        self.log = log or RecentEventLog(capacity=10)
        self._composed = composed
        self.closed_streams = 0

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return ()

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        raise UnknownCameraError(camera_id)

    def event_history(self, camera_id: str) -> CameraEventHistory:
        raise UnknownCameraError(camera_id)

    def subscribe_events(self) -> EventSubscription:
        if not self._composed:
            raise EngineNotComposedError("the engine has not finished starting")
        subscription = self.log.subscribe()
        # Closed immediately: the backlog still renders (it is read from the log, not
        # from the queue), and the body then ends instead of holding the test client's
        # thread open forever.
        subscription.close()
        return subscription

    def close_event_streams(self) -> int:
        self.closed_streams += 1
        return self.log.close_all()

    def health(self) -> dict[str, HealthReport]:
        return {}

    async def describe_now(self, camera_id: str) -> UUID:
        raise UnknownCameraError(camera_id)


class TestTheEndpoint:
    def test_the_response_is_an_event_stream_that_opens_with_the_backlog(self) -> None:
        service = _StreamingFakeService()
        service.log.record(_event(occurred_at=1.0, clip_uri="s3://clips/a.mp4"))
        with TestClient(create_app(service)) as client:
            response = client.get("/events/stream")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(SSE_MEDIA_TYPE)
        assert response.headers["cache-control"] == "no-cache, no-transform"
        name, backlog = parse(response.text)
        assert name == "backlog"
        assert isinstance(backlog, list)
        assert backlog[0]["clip_uri"] == "s3://clips/a.mp4"
        assert backlog[0]["camera_id"] == "cam-1"

    def test_an_engine_that_has_not_composed_yet_says_503_not_an_empty_stream(self) -> None:
        """An open stream that can never carry anything is indistinguishable from a
        quiet site, and a console would sit on it forever."""
        with TestClient(create_app(_StreamingFakeService(composed=False))) as client:
            response = client.get("/events/stream")

        assert response.status_code == 503

    def test_the_lifespan_closes_every_stream_on_the_way_out(self) -> None:
        """Shutdown closes streams rather than hanging — and does so *after* `stop()`,
        so the escalations that shutdown ordering exists to publish still reach a
        console that is watching."""
        service = _StreamingFakeService()
        live = service.log.subscribe()
        with TestClient(create_app(service)) as client:
            client.get("/health")
            assert live.closed is False

        assert service.closed_streams == 1
        assert live.closed is True

    def test_the_contract_documents_the_history_a_long_disconnect_loses(self) -> None:
        """The integrator reads the generated contract, not this repo's docstrings. A
        bounded ring behind a stream that looks resumable is exactly how someone
        concludes an event never happened."""
        operation = create_app(_StreamingFakeService()).openapi()["paths"]["/events/stream"]["get"]
        text = f"{operation['summary']} {operation['description']}".lower()

        assert "not the event store" in text
        assert "bounded" in text
        assert "rabbitmq" in text
        assert "backlog" in text
        assert "last-event-id" in text, "and that resuming from an offset is not offered"


async def _first(body: AsyncGenerator[str, None]) -> tuple[str, object]:
    return parse(await next_frame(body))
