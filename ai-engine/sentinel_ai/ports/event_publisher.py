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
        """

    @abstractmethod
    async def close(self) -> None: ...
