"""Multi-object tracking port (pipeline stage 3).

Synchronous by design: trackers are cheap CPU association algorithms, so an
async signature would add overhead and buy nothing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Detection, Track


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Track, ...]:
        """Associate this frame's detections with existing tracks."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all track state — used when a stream reconnects."""
