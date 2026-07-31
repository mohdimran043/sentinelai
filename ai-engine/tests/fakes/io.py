"""I/O fakes: frame sources, publishers, clip writers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameData, FrameSource


class FakeSource(FrameSource):
    def __init__(self, frames: Sequence[FrameData]) -> None:
        self._frames = list(frames)
        self.closed = False

    @staticmethod
    def make_frame(camera_id: str, frame_index: int, timestamp: float, value: int = 0) -> FrameData:
        return FrameData(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp=timestamp,
            width=4,
            height=4,
            pixels=[[value] * 4 for _ in range(4)],
        )

    @classmethod
    def constant(cls, camera_id: str, count: int, fps: float = 10.0, value: int = 0) -> FakeSource:
        return cls([cls.make_frame(camera_id, index, index / fps, value) for index in range(count)])

    def __aiter__(self) -> AsyncIterator[FrameData]:
        """Sync, matching the port and the async-iterator protocol.

        `async for` calls `__aiter__()` without awaiting it, so the method must
        return the iterator directly. Writing it as `async def` happened to work
        only because an `async def` containing `yield` is an async *generator*
        function — remove the yield and it breaks. The port's shape is the
        correct one, so the fake follows it.
        """

        async def frames() -> AsyncIterator[FrameData]:
            for frame in self._frames:
                yield frame

        return frames()

    async def close(self) -> None:
        self.closed = True


class FakePublisher(EventPublisher):
    def __init__(self, error: Exception | None = None) -> None:
        self.events: list[Event] = []
        self.closed = False
        self._error = error

    async def publish(self, event: Event) -> None:
        if self._error is not None:
            raise self._error
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class FakeClipWriter(ClipWriter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, UUID, int]] = []

    async def write(
        self, camera_id: str, event_id: UUID, frames: Sequence[FrameData], fps: float
    ) -> str:
        self.calls.append((camera_id, event_id, len(frames)))
        return f"s3://sentinel-clips/{camera_id}/{event_id}.mp4"
