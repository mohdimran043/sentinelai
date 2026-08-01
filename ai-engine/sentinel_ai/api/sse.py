"""The `text/event-stream` half of `GET /events/stream` (T2).

Split from `routes.py` because it is the only part of the endpoint with a shape worth
testing on its own: framing bytes correctly, and turning one `EventSubscription` into
a sequence of frames that starts with a backlog, continues with live writes, and ends
— rather than hanging — when the client, the subscription or the engine goes away.

Why SSE and not WebSocket
-------------------------
Spec §6.2 gives the WebSocket hub to the Phase 1C Go backend. That does not exist, so
today nothing can push an event to a browser and the console polls. SSE is the bridge
that fits in the gap: one-way, which is all a notification feed needs; plain HTTP, so
it needs no new dependency, no new port and no new protocol in the reverse proxy; and
it reconnects on its own. When the Go hub arrives this endpoint is the thing it
replaces, and the console's parser is already written against this shape.

The wire shape, and why it is the recorder's
--------------------------------------------
The recorder next door serves `GET /api/alerts/stream`: an `event: backlog` frame
carrying a JSON array, then individual items, with `: ` comment lines as keepalives.
The console already has a parser for that (`web/src/recorder/sse.ts`), so matching it
means the UI reuses working code instead of writing a second parser for a second
shape. Frames here are:

* `event: backlog` — a JSON **array** of `RecentEventEntry`, oldest first, sent once,
  immediately, before anything live. Possibly empty.
* `event: anomaly` — one `RecentEventEntry` object, as it is written.
* `event: overflow` — this client fell too far behind and the stream is ending; see
  `EVENT_STREAM_QUEUE_MAXSIZE`. Reconnecting is the recovery, and it is a good one:
  the new backlog is more current than the queue that was dropped.
* `: keepalive` — a comment, carries nothing, exists so that an idle stream still
  writes bytes and neither a proxy nor a load balancer treats it as dead.

No `id:` field, deliberately. An `id:` makes a browser's `EventSource` resume with
`Last-Event-ID`, and this engine cannot honour that honestly: the ring is bounded and
volatile, so what a client missed while disconnected may simply not exist any more.
Advertising resumability that silently degrades to "here is a fresher window" would be
worse than the recorder's plain reconnect, which is what this does instead.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from sentinel_ai.api.schemas import RecentEventEntry
from sentinel_ai.orchestrator.event_history import EventSubscription, RecentEvent

__all__ = [
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "STREAM_KEEPALIVE_SECONDS",
    "event_stream_body",
    "format_comment",
    "format_frame",
]

SSE_MEDIA_TYPE = "text/event-stream"

SSE_HEADERS = {
    # A stream must not be cached or transformed anywhere on the way out: a caching
    # proxy would serve a stale backlog forever, and a compressing one buffers, which
    # turns a live feed into a feed that arrives in lumps.
    "cache-control": "no-cache, no-transform",
    "connection": "keep-alive",
    # nginx-specific and harmless elsewhere: without it nginx buffers the response
    # body and the console sees nothing until the buffer fills.
    "x-accel-buffering": "no",
}

STREAM_KEEPALIVE_SECONDS = 20.0
"""How long an idle stream may go without writing anything.

Below the 30-60s idle timeout that proxies, load balancers and browsers commonly
apply, with room to spare, so a quiet site does not look like a dead connection. It
costs one comment line per interval per client. It is also the interval at which the
generator wakes up and can notice that its subscription was closed, so it doubles as
the upper bound on how long a torn-down stream can linger.
"""


def format_frame(event: str, data: str) -> str:
    """One SSE frame: an event name, its payload, and the blank line that ends it.

    `data` is split across `data:` lines because a newline inside a payload would
    otherwise terminate the field and corrupt the frame. JSON from `json.dumps`
    escapes its newlines and so never needs this, which is exactly why it is done
    here rather than trusted to every caller forever.
    """
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"


def format_comment(text: str) -> str:
    """A comment line — a frame that carries no data. The keepalive."""
    return f": {text}\n\n"


def _payload(event: RecentEvent) -> dict[str, object]:
    """The same projection `GET /cameras/{camera_id}/events` serves, so one console
    parser handles both and the two can never describe the same event differently."""
    return RecentEventEntry.from_recent_event(event).model_dump(mode="json")


async def event_stream_body(
    subscription: EventSubscription,
    *,
    keepalive_seconds: float | None = STREAM_KEEPALIVE_SECONDS,
) -> AsyncGenerator[str, None]:
    """Backlog, then live, then a clean end. Never an unbounded buffer, never a hang.

    The subscription is registered by the *caller* (the route handler) before this
    generator is ever started, which is what makes the handover gapless — see
    `EventSubscription`. This function takes the backlog as its first act, and every
    write from the registration onwards is either in that backlog or arrives after it,
    exactly once.

    Three ways it ends, all of them promptly:

    * the client goes away — Starlette cancels the response task, this generator is
      thrown into, and the `finally` unregisters the subscription so nothing keeps
      queueing for a socket nobody is reading;
    * the engine shuts down — `close_event_streams()` releases the reader, and the
      generator returns instead of waiting for an event that will never come;
    * the client falls further behind than the queue bound — it is told so and the
      stream ends, rather than the engine growing a buffer for it.

    `keepalive_seconds=None` disables the keepalive and waits indefinitely, which is
    what a test that has already arranged every event it cares about wants.
    """
    try:
        yield format_frame(
            "backlog", json.dumps([_payload(event) for event in subscription.open()])
        )
        while True:
            try:
                event = await subscription.next_event(timeout=keepalive_seconds)
            except TimeoutError:
                yield format_comment("keepalive")
                continue
            if event is None:
                if subscription.overflowed:
                    yield format_frame(
                        "overflow",
                        json.dumps(
                            {
                                "reason": "the client fell too far behind this stream",
                                "action": "reconnect; the new backlog supersedes what was dropped",
                            }
                        ),
                    )
                return
            yield format_frame("anomaly", json.dumps(_payload(event)))
    finally:
        # Reached on a normal return, on a client disconnect (Starlette closes the
        # generator) and on cancellation. Without it, a browser tab that vanished
        # would leave a subscription registered and being fed for the life of the
        # process — the leak the queue bound alone does not prevent, because the bound
        # caps one client and this caps how many there are.
        subscription.close()
