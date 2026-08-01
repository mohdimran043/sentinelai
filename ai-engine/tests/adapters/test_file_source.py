from __future__ import annotations

import asyncio
import time
from pathlib import Path

import av
import numpy as np
import pytest

from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.ports.frame_source import EncodedPacket

ASSET = str(Path(__file__).resolve().parents[1] / "assets" / "synthetic_clip.mp4")
ASSET_BFRAMES = str(Path(__file__).resolve().parents[1] / "assets" / "synthetic_clip_bframes.mp4")


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


def _asset_shifted_by(tmp_path: Path, seconds: float) -> str:
    """Remux the committed fixture onto a non-zero pts origin, packet-for-packet.

    A stream copy, so the bytes are identical and only the timestamps move — which is
    exactly the difference between the committed fixture and a recording whose
    container start time is not zero. Written here rather than committed as a second
    asset because the shift has to be visible in the test that depends on it.
    """
    out_path = tmp_path / "shifted.mp4"
    source = av.open(ASSET)
    try:
        in_stream = source.streams.video[0]
        assert in_stream.time_base is not None
        offset = round(seconds / float(in_stream.time_base))
        destination = av.open(str(out_path), mode="w")
        try:
            out_stream = destination.add_stream_from_template(in_stream)
            for packet in source.demux(in_stream):
                if packet.pts is None:
                    continue
                packet.pts += offset
                if packet.dts is not None:
                    packet.dts += offset
                packet.stream = out_stream
                destination.mux(packet)
        finally:
            destination.close()
    finally:
        source.close()
    return str(out_path)


async def test_timestamps_are_rebased_when_the_file_does_not_start_at_pts_zero(
    tmp_path: Path,
) -> None:
    """`test_timestamps_start_at_zero_...` above cannot prove its own name.

    `synthetic_clip.mp4`'s first packet has `pts == 0`, so `packet.pts - first_pts` is
    the identity function in every test in this suite: a mutation sweep confirmed that
    both dropping the `- first_pts` term and flipping it to `+` survive the full
    353-test run. Rebasing matters because everything downstream — the pre-roll
    buffer's horizon, `_ActiveClip.deadline`, and the clip writer's own pts origin —
    treats `FrameData.timestamp` and `EncodedPacket.pts` as one shared timeline that
    starts where the stream starts.

    Frames and packets are drained together because they share one pump, and both
    must land on the same rebased timeline, not just one of them.
    """
    source = FileSource(_asset_shifted_by(tmp_path, 30.0), camera_id="cam-1", realtime=False)

    async def collect_frames() -> list[float]:
        return [f.timestamp async for f in source]

    stamps, packets = await asyncio.wait_for(
        asyncio.gather(collect_frames(), _collect_packets(source)), timeout=5.0
    )

    assert stamps[0] == 0.0, "a 30s container origin must not become a 30s frame timestamp"
    assert stamps == pytest.approx([i * 0.1 for i in range(50)], abs=1e-6)
    assert packets[0].pts == 0.0, "packets share the timeline frames are on"
    assert [p.pts for p in packets] == pytest.approx([i * 0.1 for i in range(50)], abs=1e-6)
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


async def test_decodes_every_frame_from_a_stream_with_b_frames() -> None:
    """A pts-less flush packet at end-of-stream still holds buffered frames.

    `synthetic_clip_bframes.mp4` is encoded with `-profile:v main -bf 3`, so the decoder
    reorders and holds frames for reference; PyAV's synthetic end-of-stream packet (pts is
    None) is what flushes them. A loop that skips `.decode()` on that packet drops the tail
    of the clip -- 48 of 50 frames here -- even though it passes against the committed
    baseline-profile fixture, which forbids B-frames by construction and so never triggers
    this path.
    """
    source = FileSource(ASSET_BFRAMES, camera_id="cam-1", realtime=False)
    frames = [f async for f in source]
    assert [f.frame_index for f in frames] == list(range(50))
    await source.close()


async def test_only_iterating_frames_never_touching_packets_does_not_deadlock() -> None:
    """The literal crux case: packets() is never even called."""
    source = FileSource(ASSET, camera_id="cam-1", realtime=False)

    async def collect_frames() -> int:
        return len([f async for f in source])

    frames_count = await asyncio.wait_for(collect_frames(), timeout=5.0)
    assert frames_count == 50
    await source.close()


async def test_never_touching_packets_overflows_a_small_packet_queue_without_stalling() -> None:
    """The crux case, made to actually bite: a packet queue small enough that the

    fixture's 50 packets overflow it well before end of stream. `packets()` is never
    called, so the only way the pump can still deliver every frame is if the packet
    queue drops its oldest entry non-blockingly rather than blocking the shared pump.

    `test_only_iterating_frames_never_touching_packets_does_not_deadlock` exercises the
    same shape against the default maxsize of 256, but the fixture only ever produces 50
    packets, so that queue never fills -- it would pass identically against a queue that
    blocks instead of dropping. This test drives the queue past capacity for real.
    """
    source = FileSource(ASSET, camera_id="cam-1", realtime=False, packet_queue_maxsize=4)

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
