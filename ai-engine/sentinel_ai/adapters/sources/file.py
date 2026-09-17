"""FileSource: one PyAV demux pass fanned out to decoded frames and encoded
packets (spec §4.2, §5.1)."""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
import time
from collections.abc import AsyncIterator
from functools import partial

import av

from sentinel_ai.adapters.sources.hwaccel import build_hwaccel
from sentinel_ai.ports.frame_source import (
    DeferredPixels,
    EncodedPacket,
    FrameData,
    FrameSource,
)

_FRAME_QUEUE_MAXSIZE = 8
_PACKET_QUEUE_MAXSIZE = 256
_STOP_POLL_SECONDS = 0.1


class _QueueEnd:
    """Sentinel pushed onto both queues when the demux loop finishes or fails."""


_QUEUE_END = _QueueEnd()


def _drop_oldest_put[T](q: queue.Queue[T | _QueueEnd], item: T | _QueueEnd) -> None:
    """Non-blocking put that discards the oldest entry instead of blocking.

    Used only for the packet queue: see the design note above `FileSource`.
    """
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                q.get_nowait()


class FileSource(FrameSource):
    """Demuxes `path` once and fans out to decoded frames and encoded packets.

    Frame delivery uses real (blocking) backpressure; the packet queue never
    blocks the pump. See the design note in the plan for why they differ.
    """

    def __init__(
        self,
        path: str,
        camera_id: str,
        realtime: bool,
        packet_queue_maxsize: int = _PACKET_QUEUE_MAXSIZE,
        hwaccel_device: str | None = None,
    ) -> None:
        self._path = path
        self._camera_id = camera_id
        self._realtime = realtime
        # Resolved here rather than in `_pump`, so an unavailable device type fails
        # while a human is still looking at the startup, not on a worker thread eight
        # frames into a replay.
        self._hwaccel = build_hwaccel(hwaccel_device)
        self._frame_queue: queue.Queue[FrameData | _QueueEnd] = queue.Queue(
            maxsize=_FRAME_QUEUE_MAXSIZE
        )
        self._packet_queue: queue.Queue[EncodedPacket | _QueueEnd] = queue.Queue(
            maxsize=packet_queue_maxsize
        )
        self._stop = threading.Event()
        self._pump_started = False
        self._pump_error: Exception | None = None
        self._thread: threading.Thread | None = None

    def _ensure_pump(self) -> None:
        if self._pump_started:
            return
        self._pump_started = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _put_frame_blocking(self, item: FrameData | _QueueEnd) -> bool:
        """Blocking put, polling `_stop` every 100ms so `close()` always wins.

        A plain blocking `put()` would ignore a `close()` requested while the
        queue is full and the consumer has stopped pulling.
        """
        while not self._stop.is_set():
            try:
                self._frame_queue.put(item, timeout=_STOP_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _decode_and_emit(
        self,
        packet: av.Packet[av.VideoStream],
        frame_index: int,
        pts_seconds: float,
        rate: float,
        start_wall: float,
    ) -> int:
        """Decode `packet` and push each resulting frame to the frame queue.

        Returns the next `frame_index`, or `-1` if `close()` won the race and the pump
        should stop.
        """
        for frame in packet.decode():
            if self._realtime and rate > 0:
                due = start_wall + frame_index / rate
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            # Deferred, not converted here. The mailbox this is about to push into
            # overwrites, so most of these frames are dropped before anybody looks at
            # them, and a BGR conversion is a third of the cost of decoding. See
            # `DeferredPixels`. Dimensions come from the decoder rather than from the
            # array's shape, because there is no array yet.
            frame_data = FrameData(
                camera_id=self._camera_id,
                frame_index=frame_index,
                timestamp=pts_seconds,
                width=frame.width,
                height=frame.height,
                pixels=DeferredPixels(partial(frame.to_ndarray, format="bgr24")),
            )
            if not self._put_frame_blocking(frame_data):
                return -1
            frame_index += 1
        return frame_index

    def _pump(self) -> None:
        try:
            container = (
                av.open(self._path, hwaccel=self._hwaccel)
                if self._hwaccel is not None
                else av.open(self._path)
            )
            try:
                stream = container.streams.video[0]
                time_base = stream.time_base
                if time_base is None:
                    raise RuntimeError(f"video stream has no time_base: {self._path}")
                rate = float(stream.average_rate) if stream.average_rate else 0.0
                first_pts: int | None = None
                frame_index = 0
                pts_seconds = 0.0
                start_wall = time.monotonic()
                for packet in container.demux(stream):
                    if self._stop.is_set():
                        return
                    if packet.pts is None:
                        # PyAV's synthetic end-of-stream flush packet. It carries no real
                        # payload for the clip path, so it is never pushed onto the packet
                        # queue, but `.decode()` must still be called on it: on any stream
                        # with B-frames or reference reordering, this is what flushes the
                        # frames the decoder is still holding for reorder. Skipping decode
                        # here (as opposed to just skipping the packet-queue push) silently
                        # drops the tail of the stream.
                        frame_index = self._decode_and_emit(
                            packet, frame_index, pts_seconds, rate, start_wall
                        )
                        if frame_index < 0:
                            return
                        continue
                    if first_pts is None:
                        first_pts = packet.pts
                    pts_seconds = float((packet.pts - first_pts) * time_base)

                    _drop_oldest_put(
                        self._packet_queue,
                        EncodedPacket(
                            camera_id=self._camera_id,
                            data=bytes(packet),
                            pts=pts_seconds,
                            is_keyframe=bool(packet.is_keyframe),
                            codec=stream.codec_context.name,
                        ),
                    )

                    frame_index = self._decode_and_emit(
                        packet, frame_index, pts_seconds, rate, start_wall
                    )
                    if frame_index < 0:
                        return
            finally:
                container.close()
        except Exception as exc:
            # Surfaced to whichever consumer notices the sentinel next, not
            # swallowed: a decode failure must not look like a clean end of stream.
            self._pump_error = exc
        finally:
            self._put_frame_blocking(_QUEUE_END)
            _drop_oldest_put(self._packet_queue, _QUEUE_END)

    def __aiter__(self) -> AsyncIterator[FrameData]:
        self._ensure_pump()

        async def frames() -> AsyncIterator[FrameData]:
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(None, self._frame_queue.get)
                if isinstance(item, _QueueEnd):
                    if self._pump_error is not None:
                        raise self._pump_error
                    return
                yield item

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        self._ensure_pump()

        async def packet_stream() -> AsyncIterator[EncodedPacket]:
            loop = asyncio.get_running_loop()
            while True:
                item = await loop.run_in_executor(None, self._packet_queue.get)
                if isinstance(item, _QueueEnd):
                    if self._pump_error is not None:
                        raise self._pump_error
                    return
                yield item

        return packet_stream()

    async def close(self) -> None:
        self._stop.set()
        # Wake any consumer parked in `queue.get`. The pump's own end-of-stream
        # sentinel is not enough on a mid-stream close: `_put_frame_blocking` refuses
        # to put anything once `_stop` is set, so the pump exits without ever
        # enqueuing `_QUEUE_END` and `__aiter__`/`packets()` stay blocked on a `get()`
        # that can never return. Those `get()` calls run on the event loop's *default
        # executor*, so a blocked one is a thread the loop joins forever when it
        # closes — an engine shutdown that hangs, not merely a leaked task. Found by
        # the composition root (`sentinel_ai/main.py`): every camera-run-to-EOS test
        # drains the stream, and only a real `EngineService.stop()` closes one mid-way.
        # Non-blocking, drop-oldest, mirroring `RtspSource.close()`.
        _drop_oldest_put(self._frame_queue, _QUEUE_END)
        _drop_oldest_put(self._packet_queue, _QUEUE_END)
        if self._thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, self._thread.join)
