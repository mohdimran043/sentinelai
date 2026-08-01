"""I/O fakes: frame sources, publishers, clip writers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import numpy as np

from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource


class FakeSource(FrameSource):
    def __init__(
        self,
        frames: Sequence[FrameData],
        packets: Sequence[EncodedPacket] | None = None,
    ) -> None:
        self._frames = list(frames)
        self._packets = (
            list(packets)
            if packets is not None
            else [self._synthetic_packet(frame) for frame in self._frames]
        )
        self.closed = False

    @staticmethod
    def make_frame(camera_id: str, frame_index: int, timestamp: float, value: int = 0) -> FrameData:
        """A 4x4 BGR frame filled with `value`.

        `pixels` is a real `numpy` array, not a list of lists: `MotionAnalyzer`
        (the only stage that reads pixels) rejects anything else, so a fake that
        handed out lists could never be fed through the real pipeline — which is
        precisely what `CameraRunner`'s end-to-end tests do. Shaped (H, W, 3) to
        match `av`'s `to_ndarray(format="bgr24")`.
        """
        return FrameData(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp=timestamp,
            width=4,
            height=4,
            pixels=np.full((4, 4, 3), value, dtype=np.uint8),
        )

    @staticmethod
    def _synthetic_packet(frame: FrameData) -> EncodedPacket:
        """A deterministic stand-in for an encoded packet, paired to `frame` by camera id
        and pts. Always a keyframe: this fake exercises the clip *plumbing* (open/append/
        finish wiring through the runner and scheduler), not GOP-quantised flush semantics
        — that is Task 4's job, against real PyAV output.
        """
        return EncodedPacket(
            camera_id=frame.camera_id,
            data=f"packet-{frame.frame_index}".encode(),
            pts=frame.timestamp,
            is_keyframe=True,
            codec="h264",
        )

    @classmethod
    def constant(cls, camera_id: str, count: int, fps: float = 10.0, value: int = 0) -> FakeSource:
        return cls([cls.make_frame(camera_id, index, index / fps, value) for index in range(count)])

    def __aiter__(self) -> AsyncIterator[FrameData]:
        """Sync, matching the port and the async-iterator protocol.

        `async for` calls `__aiter__()` without awaiting it, so the method must return the
        iterator directly. Writing it as `async def` happened to work only because an
        `async def` containing `yield` is an async *generator* function — remove the yield
        and it breaks. The port's shape is the correct one, so the fake follows it.
        """

        async def frames() -> AsyncIterator[FrameData]:
            for frame in self._frames:
                yield frame

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        """Sync for the same reason `__aiter__` is — see above."""

        async def stream() -> AsyncIterator[EncodedPacket]:
            for packet in self._packets:
                yield packet

        return stream()

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


class FakeClipHandle(ClipHandle):
    """Records every appended packet; `abort` never raises.

    `finish_error`, when set, makes `finish()` raise — spec §9 requires an event to
    survive a clip-finalisation failure, and that path needs a handle that fails.
    """

    def __init__(
        self,
        camera_id: str,
        event_id: UUID,
        *,
        finish_error: Exception | None = None,
        append_error: Exception | None = None,
        append_error_after: int = 0,
    ) -> None:
        self.camera_id = camera_id
        self.event_id = event_id
        self.packets: list[EncodedPacket] = []
        self.finished = False
        self.aborted = False
        self._finish_error = finish_error
        self._append_error = append_error
        self._append_error_after = append_error_after

    async def append(self, packet: EncodedPacket) -> None:
        """`append_error`, when set, raises from the `append_error_after`-th call on.

        A remux `append` genuinely can fail mid-clip -- PyAV raises on a packet its
        parser cannot make sense of -- and the packet loop must survive it.
        """
        if self._append_error is not None and len(self.packets) >= self._append_error_after:
            raise self._append_error
        self.packets.append(packet)

    async def finish(self) -> str:
        if self._finish_error is not None:
            raise self._finish_error
        self.finished = True
        return f"s3://sentinel-clips/{self.camera_id}/{self.event_id}.mp4"

    async def abort(self) -> None:
        self.aborted = True


class FakeClipWriter(ClipWriter):
    """`finish_error`, when set, is attached to every handle this writer opens."""

    def __init__(
        self,
        finish_error: Exception | None = None,
        append_error: Exception | None = None,
        append_error_after: int = 0,
    ) -> None:
        self.opened: list[tuple[str, UUID, float]] = []
        self.handles: list[FakeClipHandle] = []
        self._finish_error = finish_error
        self._append_error = append_error
        self._append_error_after = append_error_after

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> FakeClipHandle:
        """Returns the concrete `FakeClipHandle`, not the abstract `ClipHandle`.

        A covariant, Liskov-valid narrowing of the port's return type: callers coded against
        `ClipWriter` still see a `ClipHandle`, but the tests in this suite that assert on
        `.packets`/`.finished`/`.aborted` — attributes the port itself does not promise —
        typecheck without an `isinstance` narrowing at every call site.
        """
        self.opened.append((camera_id, event_id, fps))
        handle = FakeClipHandle(
            camera_id,
            event_id,
            finish_error=self._finish_error,
            append_error=self._append_error,
            append_error_after=self._append_error_after,
        )
        self.handles.append(handle)
        return handle
