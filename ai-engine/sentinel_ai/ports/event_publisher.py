"""Outbound event port — the AI Engine's only channel to the Web Platform."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Event


class EventPublisher(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Deliver an event.

        Implementations must not lose events when the transport is down
        (spec §9); the RabbitMQ adapter spools to disk and replays.

        Every other failure must **raise**, never be swallowed. `VlmScheduler`
        treats a raise as "the publisher did not take this event" and hands it
        to a `FailedEventSink`; a publisher that logged and returned would
        drop the event past the last component able to save it.
        """

    @abstractmethod
    async def close(self) -> None: ...


class FailedEventSink(ABC):
    """Last resort for an event no `EventPublisher` would take (spec §9).

    Separate from the publisher's own disk spool, and deliberately so. The spool
    holds *validated* payloads waiting for a broker that is merely down; replay
    will eventually deliver every one of them. This sink holds events that failed
    for a reason replay cannot fix — a schema violation, a spool directory that
    is not writable, a bug in the transport. Mixing the two would put a payload
    that can never be accepted at the front of the replay queue.
    """

    @abstractmethod
    async def store(self, event: Event, error: BaseException) -> None:
        """Persist `event` and why it could not be published.

        Must not raise: this is the end of the line, and an exception here would
        propagate back into the escalation worker with nothing left to catch it.
        """
