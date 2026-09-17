"""The short clip a notification carries, cut from the evidence clip.

The two want opposite things and this is where that is enforced: evidence keeps the
pre-roll and the post-roll for somebody who sits down with it later, and a notification
keeps however long it takes to see what happened, for somebody reading it on a phone.
"""

from __future__ import annotations

from pathlib import Path

import av
import pytest

from sentinel_ai.adapters.storage.minio_clips import _trim

ASSET = Path(__file__).resolve().parents[2] / "assets" / "synthetic_clip.mp4"


def duration_of(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if time_base is None:
            return 0.0
        last = 0.0
        for packet in container.demux(stream):
            if packet.pts is not None:
                last = max(last, float(packet.pts * time_base))
        return last


class TestTrim:
    def test_it_keeps_only_the_opening_seconds(self, tmp_path: Path) -> None:
        target = tmp_path / "short.mp4"
        _trim(ASSET, target, seconds=1.0)

        assert target.exists()
        assert duration_of(target) <= duration_of(ASSET)
        assert duration_of(target) < 1.5

    def test_the_result_still_plays(self, tmp_path: Path) -> None:
        """A trim that produced an unopenable file would be worse than no trim: the
        notification would carry a link to something that does not play."""
        target = tmp_path / "short.mp4"
        _trim(ASSET, target, seconds=1.0)

        with av.open(str(target)) as container:
            frames = sum(len(packet.decode()) for packet in container.demux(video=0))
        assert frames > 0

    def test_it_starts_at_the_beginning_not_at_the_event(self, tmp_path: Path) -> None:
        """The opening seconds are the pre-roll — somebody walking in, then falling. A
        clip that began at the keyframe would open on a person already on the floor."""
        target = tmp_path / "short.mp4"
        _trim(ASSET, target, seconds=1.0)

        with av.open(str(target)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            assert time_base is not None
            first = next(p.pts for p in container.demux(stream) if p.pts is not None)
            assert float(first * time_base) < 0.2

    def test_it_measures_from_the_first_packet_not_from_zero(self, tmp_path: Path) -> None:
        """The bug a live run found. A remuxed evidence clip carries the camera's own
        arrival timeline, so its first pts is whatever the clock said when recording
        started. Comparing that against the window directly is true on the very first
        packet, and a 3-second setting produced 0.47-second clips.

        Simulated by shifting every timestamp forward, which is exactly what the real
        clip's timeline looks like.
        """
        shifted = tmp_path / "shifted.mp4"
        _shift_timestamps(ASSET, shifted, by_seconds=10_000.0)

        target = tmp_path / "short.mp4"
        _trim(shifted, target, seconds=1.0)

        # Against the old absolute comparison this is a fraction of a second.
        assert duration_of(target) > 0.5

    def test_a_window_longer_than_the_clip_keeps_all_of_it(self, tmp_path: Path) -> None:
        """A three-second window on a two-second clip is not an error, and truncating to
        nothing would be the worst possible reading of it."""
        target = tmp_path / "short.mp4"
        _trim(ASSET, target, seconds=3600.0)

        assert duration_of(target) == pytest.approx(duration_of(ASSET), abs=0.2)

    def test_it_is_a_remux_not_a_re_encode(self, tmp_path: Path) -> None:
        """ADR 3's rule, and sharper here: this copy is the one a person actually
        watches, so re-encoding would degrade the only version most notifications are
        ever seen as."""
        target = tmp_path / "short.mp4"
        _trim(ASSET, target, seconds=1.0)

        with av.open(str(ASSET)) as source, av.open(str(target)) as short:
            assert short.streams.video[0].codec_context.name == (
                source.streams.video[0].codec_context.name
            )


def _shift_timestamps(source: Path, target: Path, *, by_seconds: float) -> None:
    """Copy a clip with every timestamp moved forward, the way a live camera's is."""
    with av.open(str(source)) as inbound, av.open(str(target), mode="w") as outbound:
        stream = inbound.streams.video[0]
        out_stream = outbound.add_stream_from_template(stream)
        time_base = stream.time_base
        assert time_base is not None
        offset = int(by_seconds / float(time_base))
        for packet in inbound.demux(stream):
            if packet.dts is None:
                continue
            packet.pts = (packet.pts or 0) + offset
            packet.dts = packet.dts + offset
            packet.stream = out_stream
            outbound.mux(packet)
