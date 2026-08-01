"""In-memory EventPublisher — the real adapter for CPU-only runs (spec §5.6).

Not a test fake: it lives in `adapters/`, is what the end-to-end CPU trace
asserts against, and validates every event through `encode_event` exactly
like `RabbitMQPublisher` does, so a schema violation is caught the same way
in every environment the AI Engine runs in.
"""

from __future__ import annotations

from sentinel_ai.adapters.serialization.event_codec import encode_event
from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.event_publisher import EventPublisher


class InMemoryPublisher(EventPublisher):
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._closed = False

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    async def publish(self, event: Event) -> None:
        encode_event(event)  # validates; raises before the event is ever accepted
        self._events.append(event)

    async def close(self) -> None:
        self._closed = True
