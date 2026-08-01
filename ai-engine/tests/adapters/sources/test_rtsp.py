"""Tests for RtspSource (Task 14).

`_ReconnectLoop`'s backoff sequencing is fully decoupled from PyAV and
networking, so it is CPU/CI-safe: no real sleep, no real connection, no GPU.
The PyAV decode path itself needs a real RTSP source (mediamtx) and is
exercised manually — see ai-engine/README.md's end-to-end demo section —
there is no CI-safe way to test a live RTSP reconnect without a running
mediamtx.
"""

from __future__ import annotations

import asyncio

import pytest

from sentinel_ai.adapters.sources.rtsp import RtspSource, _annexb_keyframe_bytes, _ReconnectLoop
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource


def test_rtsp_source_satisfies_its_port() -> None:
    assert issubclass(RtspSource, FrameSource)


async def test_reconnect_loop_backs_off_exponentially_and_caps_at_max() -> None:
    sleeps: list[float] = []
    attempts = 0

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise ConnectionError("simulated failure")
        # 4th attempt "succeeds" — the loop stops on the next should_stop check.

    loop = _ReconnectLoop(connect_once, initial_seconds=1.0, max_seconds=3.0, sleep=fake_sleep)
    await loop.run_forever(should_stop=lambda: attempts >= 4)

    assert sleeps == pytest.approx([1.0, 2.0, 3.0])  # doubling, capped at max_seconds=3.0
    assert loop.backoff_log == pytest.approx([1.0, 2.0, 3.0])


async def test_reconnect_loop_does_not_back_off_immediately_or_at_a_fixed_delay() -> None:
    """Discriminating against two specific wrong implementations: reconnecting
    immediately (no sleep at all, or always sleeping `initial_seconds`) rather
    than genuinely doubling each time.
    """
    sleeps: list[float] = []
    attempts = 0

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            raise ConnectionError("simulated failure")

    loop = _ReconnectLoop(connect_once, initial_seconds=0.5, max_seconds=100.0, sleep=fake_sleep)
    await loop.run_forever(should_stop=lambda: attempts >= 5)

    assert len(sleeps) == 4
    assert sleeps != [0.5, 0.5, 0.5, 0.5]  # not a fixed delay
    assert sleeps == pytest.approx([0.5, 1.0, 2.0, 4.0])  # genuine doubling, well under max


async def test_reconnect_loop_calls_on_reconnect_once_per_retry() -> None:
    calls = 0

    def on_reconnect() -> None:
        nonlocal calls
        calls += 1

    attempts = 0

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ConnectionError("simulated failure")

    async def fake_sleep(_seconds: float) -> None:
        return None

    loop = _ReconnectLoop(
        connect_once,
        initial_seconds=1.0,
        max_seconds=30.0,
        sleep=fake_sleep,
        on_reconnect=on_reconnect,
    )
    await loop.run_forever(should_stop=lambda: attempts >= 3)
    assert calls == 2


async def test_reconnect_loop_never_sleeps_on_first_successful_attempt() -> None:
    """A stream that connects cleanly on the first try must never pay the backoff
    delay at all — discriminates against a loop that unconditionally sleeps once
    before checking `should_stop`."""
    sleeps: list[float] = []
    attempts = 0

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    loop = _ReconnectLoop(connect_once, initial_seconds=1.0, max_seconds=30.0, sleep=fake_sleep)
    await loop.run_forever(should_stop=lambda: attempts >= 1)
    assert sleeps == []


def test_on_reconnect_resets_the_frame_index_counter() -> None:
    """Constructing RtspSource for real starts a background task that opens a
    real (here, nonexistent) RTSP URL — not CI-safe. `__new__` bypasses
    `__init__` to unit-test this one state transition in isolation.
    """
    source = RtspSource.__new__(RtspSource)
    source._frame_index = 42
    source._on_reconnect()
    assert source._frame_index == 0


async def test_close_does_not_hang_when_the_queues_are_full() -> None:
    """`CameraRunner.run()` cancels the frame/packet consumer tasks *before*
    calling `source.close()` (pipeline/runner.py's `finally` block), so by the
    time `close()` runs, nothing is left draining either queue. A `close()`
    that does a plain blocking `await queue.put(_CLOSE)` on a full queue would
    hang forever right here — this is exactly the "packets() must end when
    frames end" shutdown obligation. Bounded with `wait_for` only as a safety
    net: the correct implementation returns immediately, so this never
    actually waits out the timeout.
    """
    source = RtspSource.__new__(RtspSource)
    source._closed = False
    source._frame_queue = asyncio.Queue(maxsize=1)
    source._frame_queue.put_nowait(
        FrameData(camera_id="cam", frame_index=0, timestamp=0.0, width=1, height=1, pixels=None)
    )
    source._packet_queue = asyncio.Queue(maxsize=1)
    source._packet_queue.put_nowait(
        EncodedPacket(camera_id="cam", data=b"", pts=0.0, is_keyframe=True, codec="h264")
    )

    async def never_resolves() -> None:
        await asyncio.Event().wait()

    source._task = asyncio.create_task(never_resolves())

    await asyncio.wait_for(source.close(), timeout=2.0)


class TestAnnexbKeyframeBytes:
    """RTP H.264 (RFC 6184) carries SPS/PPS out-of-band, in the RTSP SDP's
    `sprop-parameter-sets` negotiated once at session setup -- PyAV surfaces this as
    `codec_context.extradata`, never repeated in-band before every IDR the way
    `h264_mp4toannexb` inserts it for an MP4 source. Verified live against a real
    mediamtx session during this task's manual demo: without this, every packet
    `MinioClipWriter`'s "extract_extradata" bitstream filter (Task 9, tested only
    against an MP4-fixture-converted-to-Annex-B stream) ever sees is missing SPS/PPS,
    so it can never resolve extradata and every clip it opens against a real RTSP
    source comes back an empty file."""

    def test_prepends_extradata_to_a_keyframes_bytes(self) -> None:
        raw = b"\x00\x00\x00\x01\x65abc"
        extradata = b"\x00\x00\x00\x01\x67sps\x00\x00\x00\x01\x68pps"
        assert _annexb_keyframe_bytes(raw, extradata) == extradata + raw

    def test_leaves_bytes_unchanged_when_extradata_is_empty(self) -> None:
        assert _annexb_keyframe_bytes(b"raw-bytes", b"") == b"raw-bytes"
