"""RTSP frame source with exponential-backoff reconnect (spec §5.1, §6).

Implements `FrameSource` (S3) exactly as `FileSource` does: one PyAV demux
loop fans out to a decoded `FrameData` stream (`__aiter__`) and an encoded
`EncodedPacket` stream (`packets()`), so the clip path never decodes and the
inference path never re-demuxes. It also repeats `FileSource`'s hard-won
end-of-stream fix: PyAV's synthetic flush packet at end-of-stream carries no
`dts`, but `.decode()` must still be called on it, because that is what
flushes frames the decoder is still holding for B-frame/reference reorder.
Skipping decode on that packet (as opposed to skipping only the
packet-queue push) silently drops the tail of the stream — measured 48/50
vs. 50/50 on a B-frame-heavy clip in Task 4's own testing. Real IP cameras
use B-frames heavily, so this matters more here than it did for `FileSource`.

The only genuinely new logic is the reconnect loop, split out into
`_ReconnectLoop` — a class with no PyAV or asyncio-queue state of its own,
so its retry/backoff sequencing is unit-testable with fakes: no network, no
real sleep, no GPU.

Discontinuity signal (spec §5.2/§6): on every reconnect, `_on_reconnect`
resets `_frame_index` to 0 while leaving the injected monotonic `clock`
untouched. `FrameData.timestamp` keeps advancing across the reconnect (so
downstream cooldown/rate-limit logic, which assumes non-decreasing time,
never sees time run backward), while `frame_index` restarting at 0 is what
`CameraRunner._is_timeline_regression` (pipeline/runner.py) actually checks:
`frame.frame_index < self._last_frame_index`. Verified against that method
directly, not assumed — it is this task's obligation, not a new callback
into `CameraRunner`.

Timestamps are wall-clock, not the RTSP stream's own `pts`: a file has one
continuous timeline, so `FileSource` can trust its embedded `pts`, but a live
RTSP session's `pts` restarts (or is otherwise renegotiated) on every
reconnect and cannot be trusted as the pipeline's monotonic clock. Both
`FrameData.timestamp` and `EncodedPacket.pts` are stamped from the injected
`clock()` at the moment of arrival instead — the one thing guaranteed to
keep moving forward across a reconnect.

Threading: one worker thread per connection attempt runs the blocking
`av.open`/`demux`/`decode` loop (via `run_in_executor`); each decoded
`FrameData`/`EncodedPacket` crosses back to the event loop with
`asyncio.run_coroutine_threadsafe(queue.put(...), loop).result()`, which
keeps both `asyncio.Queue`s single-writer-safe from the event loop's
perspective while still blocking the worker thread on backpressure (the
queues are bounded).

Shutdown cannot hang: `packets()` and `__aiter__` only ever terminate when
`close()` pushes the close sentinel, and `close()` is exactly what stops the
reconnect loop (`should_stop` reads `self._closed`) — the frame stream and
the packet stream always end together, which is `CameraRunner.run()`'s other
obligation on this task (it `await`s the packet loop rather than cancelling
it, so a `packets()` that never terminated would hang shutdown).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import av

from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource

logger = logging.getLogger(__name__)

_FRAME_QUEUE_MAXSIZE = 8
_PACKET_QUEUE_MAXSIZE = 64

# ffmpeg/PyAV RTSP socket timeout, microseconds: bounds how long a stalled TCP
# connection can block the worker thread inside `container.demux()` before it
# raises and the `_ReconnectLoop` gets a chance to back off. Without this, a
# camera that goes silent (as opposed to actively refusing the connection)
# could block the worker thread indefinitely, since PyAV/ffmpeg blocking calls
# cannot be interrupted from another thread once entered.
_RTSP_SOCKET_TIMEOUT_MICROSECONDS = "20000000"  # 20s


class _CloseSentinel:
    """A dedicated sentinel type, not `None` or `object()`, so `isinstance`
    narrows the queue's union type for mypy strict."""

    __slots__ = ()


_CLOSE = _CloseSentinel()


def _annexb_keyframe_bytes(raw: bytes, extradata: bytes) -> bytes:
    """Prepend SDP-derived SPS/PPS to a keyframe packet's raw bytes.

    RTP H.264 (RFC 6184) carries SPS/PPS out-of-band, in the RTSP SDP's
    `sprop-parameter-sets`, negotiated once at session setup — PyAV surfaces this as
    the input stream's `codec_context.extradata`, never repeated in-band before every
    IDR the way `h264_mp4toannexb` inserts it for an MP4 source (compare
    `tests/adapters/storage/test_minio_clips.py`'s `_annexb_packets_from_fixture`,
    which relies on exactly that MP4-side insertion). Without this,
    `MinioClipWriter`'s "extract_extradata" bitstream filter — built and tested only
    against that MP4-fixture-converted-to-Annex-B stream — never finds SPS/PPS in a
    genuine RTSP packet stream, and every clip it opens comes back an empty file.
    Verified against a real mediamtx session during this task's manual demo, not a
    hypothetical: see the task report.
    """
    return extradata + raw if extradata else raw


def _put_dropping_oldest[T](q: asyncio.Queue[T], item: T) -> None:
    """Non-blocking put that discards the oldest queued entry instead of blocking.

    Used only to deliver the close sentinel: see the design note in `close()`.
    """
    while True:
        try:
            q.put_nowait(item)
            return
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()


class _ReconnectLoop:
    """Retries `connect_once` with exponential backoff; owns no I/O itself."""

    def __init__(
        self,
        connect_once: Callable[[], Awaitable[None]],
        *,
        initial_seconds: float,
        max_seconds: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        self._connect_once = connect_once
        self._initial = initial_seconds
        self._max = max_seconds
        self._sleep = sleep
        self._on_reconnect = on_reconnect
        self.backoff_log: list[float] = []

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        backoff = self._initial
        while not should_stop():
            try:
                await self._connect_once()
                backoff = self._initial
            except Exception as exc:
                if should_stop():
                    return
                logger.warning("stream error (%s); reconnecting in %.1fs", exc, backoff)
                self.backoff_log.append(backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2.0, self._max)
                if self._on_reconnect is not None:
                    self._on_reconnect()


class RtspSource(FrameSource):
    """One PyAV demux pass over an RTSP URL, reconnecting with backoff on failure.

    Not an S13 model-runtime lifecycle object — sources are not model
    runtimes — so this constructor is this task's own to define.
    """

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        reconnect_initial_seconds: float,
        reconnect_max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._camera_id = camera_id
        self._url = url
        self._clock = clock
        self._frame_index = 0
        self._closed = False
        self._frame_queue: asyncio.Queue[FrameData | _CloseSentinel] = asyncio.Queue(
            maxsize=_FRAME_QUEUE_MAXSIZE
        )
        self._packet_queue: asyncio.Queue[EncodedPacket | _CloseSentinel] = asyncio.Queue(
            maxsize=_PACKET_QUEUE_MAXSIZE
        )
        self._reconnect = _ReconnectLoop(
            self._connect_once,
            initial_seconds=reconnect_initial_seconds,
            max_seconds=reconnect_max_seconds,
            on_reconnect=self._on_reconnect,
        )
        self._task = asyncio.create_task(
            self._reconnect.run_forever(should_stop=lambda: self._closed)
        )

    def _on_reconnect(self) -> None:
        """The discontinuity signal: a new connection restarts frame numbering
        from 0 while `clock()` keeps advancing. `CameraRunner` treats a
        `frame_index` that regresses below the previous value as a stream
        discontinuity and resets the tracker, motion analyzer, gate state and
        pre-roll (see `CameraRunner._is_timeline_regression`/`_on_discontinuity`
        in pipeline/runner.py).
        """
        self._frame_index = 0

    async def _connect_once(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pump_stream, loop)

    def _pump_stream(self, loop: asyncio.AbstractEventLoop) -> None:
        """Runs on a worker thread: blocking PyAV open/demux/decode.

        Raises on any failure, including a clean end-of-stream (RTSP is a live
        feed — the stream "ending" is itself a connectivity failure the
        `_ReconnectLoop` must retry, not a terminal condition).
        """
        options = {
            "rtsp_transport": "tcp",
            "stimeout": _RTSP_SOCKET_TIMEOUT_MICROSECONDS,
        }
        container = av.open(self._url, options=options)
        try:
            stream = container.streams.video[0]
            # SDP-negotiated SPS/PPS (spec: see `_annexb_keyframe_bytes`'s docstring for
            # why this must be stitched back onto every keyframe packet by hand for a
            # real RTSP source).
            extradata = bytes(stream.codec_context.extradata or b"")
            for packet in container.demux(stream):
                if self._closed:
                    return
                arrival = self._clock()
                if packet.dts is not None:
                    # PyAV's synthetic end-of-stream flush packet has no dts; it is
                    # never pushed to the packet queue (nothing to remux), but
                    # `.decode()` below still runs on it unconditionally — see the
                    # module docstring for why skipping that silently drops frames.
                    is_keyframe = bool(packet.is_keyframe)
                    data = bytes(packet)
                    if is_keyframe:
                        data = _annexb_keyframe_bytes(data, extradata)
                    encoded = EncodedPacket(
                        camera_id=self._camera_id,
                        data=data,
                        pts=arrival,
                        is_keyframe=is_keyframe,
                        codec=stream.codec_context.name,
                    )
                    asyncio.run_coroutine_threadsafe(self._packet_queue.put(encoded), loop).result()
                for decoded in packet.decode():
                    pixels = decoded.to_ndarray(format="bgr24")
                    frame = FrameData(
                        camera_id=self._camera_id,
                        frame_index=self._frame_index,
                        timestamp=arrival,
                        width=pixels.shape[1],
                        height=pixels.shape[0],
                        pixels=pixels,
                    )
                    asyncio.run_coroutine_threadsafe(self._frame_queue.put(frame), loop).result()
                    self._frame_index += 1
            raise ConnectionError(f"RTSP stream ended: {self._url}")
        finally:
            container.close()

    def __aiter__(self) -> AsyncIterator[FrameData]:
        async def frames() -> AsyncIterator[FrameData]:
            while True:
                item = await self._frame_queue.get()
                if isinstance(item, _CloseSentinel):
                    return
                yield item

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        async def pkts() -> AsyncIterator[EncodedPacket]:
            while True:
                item = await self._packet_queue.get()
                if isinstance(item, _CloseSentinel):
                    return
                yield item

        return pkts()

    async def close(self) -> None:
        self._closed = True
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        # `CameraRunner.run()`'s `finally` block cancels the frame/packet consumer
        # tasks *before* calling `close()`, so nothing may be left draining either
        # queue. A plain blocking `put()` on a full queue would hang shutdown right
        # here — exactly the "packets() must end when frames end" obligation this
        # source owes `CameraRunner` — so this drops the oldest entry instead of
        # blocking, same as `FileSource`'s non-blocking packet-queue put.
        _put_dropping_oldest(self._frame_queue, _CLOSE)
        _put_dropping_oldest(self._packet_queue, _CLOSE)
