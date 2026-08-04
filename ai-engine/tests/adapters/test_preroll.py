from __future__ import annotations

import pytest

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.ports.frame_source import EncodedPacket


def packet(pts: float, is_keyframe: bool) -> EncodedPacket:
    return EncodedPacket(
        camera_id="cam-1", data=b"x", pts=pts, is_keyframe=is_keyframe, codec="h264"
    )


class TestPreRollBuffer:
    def test_rejects_a_negative_horizon(self) -> None:
        with pytest.raises(ValueError, match="preroll_seconds"):
            PreRollBuffer(preroll_seconds=-0.1)

    def test_a_zero_horizon_keeps_only_the_gop_the_trigger_lands_in(self) -> None:
        """Zero means "no lead-in", and it has to work: `Settings.clip_preroll_seconds`
        is declared `ge=0` and `PATCH /cameras/{id}` accepts `0` for the per-camera
        override, so a rejected zero is a crash at camera construction for a value both
        edges already call valid.

        "No lead-in" is still keyframe-aligned, because a clip cannot start mid-GOP:
        the flush begins at the newest keyframe at or before the trigger, which is the
        least history a decodable clip can start from.
        """
        buffer = PreRollBuffer(preroll_seconds=0.0)
        for index in range(6):
            buffer.append(packet(round(index * 0.1, 4), is_keyframe=(index % 3 == 0)))
        flushed = buffer.flush()
        assert [p.pts for p in flushed] == [0.3, 0.4, 0.5]
        assert buffer.span_seconds == pytest.approx(0.2), "no history beyond the open GOP"

    def test_the_horizon_can_be_retargeted_and_takes_effect_going_forward(self) -> None:
        """A per-camera `clip_preroll_seconds` edit re-sizes the live ring rather than
        replacing it — replacing it would throw away the history the very next
        escalation is going to want.

        Going forward, not retroactively: widening cannot resurrect packets already
        evicted under the narrower horizon, so the new length is honoured as the ring
        refills.
        """
        buffer = PreRollBuffer(preroll_seconds=0.05)
        for index in range(6):
            buffer.append(packet(round(index * 0.1, 4), is_keyframe=True))
        assert [p.pts for p in buffer.flush()] == [0.4, 0.5]

        buffer.preroll_seconds = 10.0
        assert buffer.preroll_seconds == 10.0
        assert [p.pts for p in buffer.flush()] == [0.4, 0.5], "evicted history is gone for good"
        for index in range(6, 9):
            buffer.append(packet(round(index * 0.1, 4), is_keyframe=True))
        assert [p.pts for p in buffer.flush()] == [0.4, 0.5, 0.6, 0.7, 0.8]

    def test_retargeting_rejects_a_negative_horizon(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=1.0)
        with pytest.raises(ValueError, match="preroll_seconds"):
            buffer.preroll_seconds = -1.0
        assert buffer.preroll_seconds == 1.0

    def test_returns_empty_before_any_keyframe_has_been_seen(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=1.0)
        buffer.append(packet(0.0, is_keyframe=False))
        buffer.append(packet(0.1, is_keyframe=False))
        assert buffer.flush() == ()

    def test_flush_starts_from_a_keyframe(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=0.15)
        buffer.append(packet(0.0, is_keyframe=True))
        buffer.append(packet(0.1, is_keyframe=False))
        buffer.append(packet(0.2, is_keyframe=False))
        flushed = buffer.flush()
        assert flushed[0].is_keyframe
        assert [p.pts for p in flushed] == [0.0, 0.1, 0.2]

    def test_flush_advances_to_the_newest_keyframe_at_or_before_the_horizon(self) -> None:
        """Two keyframes exist; only the closer one anchors a minimal clip."""
        buffer = PreRollBuffer(preroll_seconds=0.05)
        buffer.append(packet(0.0, is_keyframe=True))
        buffer.append(packet(0.1, is_keyframe=True))
        buffer.append(packet(0.2, is_keyframe=False))
        flushed = buffer.flush()
        # horizon = 0.2 - 0.05 = 0.15; the keyframe at 0.1 qualifies, so the
        # one at 0.0 no longer anchors anything once a closer one exists.
        assert [p.pts for p in flushed] == [0.1, 0.2]

    def test_eviction_is_bounded_and_keeps_the_keyframe_anchor(self) -> None:
        """A working buffer stays small; a no-op implementation keeps growing."""
        buffer = PreRollBuffer(preroll_seconds=0.25)
        # A 1s GOP (10 packets @ 0.1s apart) fed for 5s -- far more than the horizon.
        for i in range(50):
            buffer.append(packet(round(i * 0.1, 4), is_keyframe=(i % 10 == 0)))
        flushed = buffer.flush()
        assert flushed[0].is_keyframe
        assert buffer.span_seconds < 1.5, "retained span must stay bounded, not grow forever"
        assert flushed[0].pts == pytest.approx(4.0)
        assert flushed[-1].pts == pytest.approx(4.9)

    def test_append_never_evicts_a_keyframe_later_packets_still_depend_on(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=10.0)  # horizon far in the past
        for i in range(20):
            buffer.append(packet(round(i * 0.1, 4), is_keyframe=(i % 10 == 0)))
        # Horizon never reached: everything since the first keyframe is kept.
        assert buffer.flush()[0].pts == 0.0
        assert len(buffer.flush()) == 20

    def test_clear_genuinely_empties_the_buffer(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=0.15)
        buffer.append(packet(0.0, is_keyframe=True))
        buffer.append(packet(0.1, is_keyframe=False))
        buffer.clear()
        assert buffer.flush() == ()
        assert buffer.span_seconds == 0.0

    def test_span_seconds_is_zero_when_empty(self) -> None:
        assert PreRollBuffer(preroll_seconds=1.0).span_seconds == 0.0

    def test_span_seconds_reports_the_retained_pts_range(self) -> None:
        buffer = PreRollBuffer(preroll_seconds=10.0)
        buffer.append(packet(0.0, is_keyframe=True))
        buffer.append(packet(0.3, is_keyframe=False))
        assert buffer.span_seconds == pytest.approx(0.3)
