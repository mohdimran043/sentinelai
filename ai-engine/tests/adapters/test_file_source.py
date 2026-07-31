from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.ports.frame_source import EncodedPacket

ASSET = str(Path(__file__).resolve().parents[1] / "assets" / "synthetic_clip.mp4")


async def _drain(source: FileSource) -> None:
    async for _ in source:
        pass


async def _collect_packets(source: FileSource) -> list[EncodedPacket]:
    return [p async for p in source.packets()]


async def test_decodes_every_frame_in_order() -> None:
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)
    frames = [f async for f in source]
    assert [f.frame_index for f in frames] == list(range(50))
    assert frames[0].width == 320
    assert frames[0].height == 240
    assert isinstance(frames[0].pixels, np.ndarray)
    await source.close()


async def test_timestamps_start_at_zero_and_are_spaced_by_the_frame_interval() -> None:
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)
    stamps = [f.timestamp async for f in source]
    assert stamps[0] == 0.0
    assert stamps == pytest.approx([i * 0.1 for i in range(50)], abs=1e-6)
    await source.close()


async def test_packets_emit_every_packet_with_correct_keyframes() -> None:
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)
    # packets() must be drained alongside __aiter__: frame delivery holds real
    # backpressure, so a frame consumer that never runs at all would eventually
    # stall the shared pump. Every real caller (CameraRunner, Task 6) drains
    # both from the start, so this test does too.
    drain_frames = asyncio.ensure_future(_drain(source))
    packets = await asyncio.wait_for(_collect_packets(source), timeout=5.0)
    await drain_frames
    assert len(packets) == 50
    keyframe_indices = [i for i, p in enumerate(packets) if p.is_keyframe]
    assert keyframe_indices == [0, 10, 20, 30, 40]
    assert packets[0].pts == 0.0
    assert packets[-1].pts == pytest.approx(4.9, abs=1e-6)
    assert all(p.codec == "h264" for p in packets)
    await source.close()


async def test_concurrent_drain_of_both_streams_agrees_on_counts() -> None:
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)

    async def collect_frames() -> int:
        return len([f async for f in source])

    frames_count, packets = await asyncio.wait_for(
        asyncio.gather(collect_frames(), _collect_packets(source)), timeout=5.0
    )
    assert frames_count == 50
    assert len(packets) == 50
    await source.close()


async def test_only_iterating_frames_never_touching_packets_does_not_deadlock() -> None:
    """The literal crux case: packets() is never even called."""
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)

    async def collect_frames() -> int:
        return len([f async for f in source])

    frames_count = await asyncio.wait_for(collect_frames(), timeout=5.0)
    assert frames_count == 50
    await source.close()


async def test_realtime_false_yields_as_fast_as_possible() -> None:
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)
    start = time.monotonic()
    _ = [f async for f in source]
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, "realtime=False must not pace to the file's own framerate"
    await source.close()


async def test_preroll_buffer_against_the_real_synthetic_clip() -> None:
    """Real PyAV demuxing feeding the buffer proves GOP quantisation end to end."""
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)
    buffer = PreRollBuffer(preroll_seconds=0.25)
    drain_frames = asyncio.ensure_future(_drain(source))

    async def fill_buffer() -> None:
        async for packet in source.packets():
            buffer.append(packet)

    await asyncio.wait_for(fill_buffer(), timeout=5.0)
    await drain_frames
    flushed = buffer.flush()
    assert flushed[0].is_keyframe
    assert flushed[0].pts == 4.0  # last GOP boundary <= (4.9 - 0.25)
    await source.close()
