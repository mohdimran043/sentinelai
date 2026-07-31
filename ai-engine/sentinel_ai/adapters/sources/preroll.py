"""Keyframe-aligned encoded packet ring (spec §5.1).

Holds `EncodedPacket`s for `preroll_seconds` of the live stream so an escalation's
evidence clip can start *before* the trigger. Packets, not decoded frames: a few
seconds of H.264 costs ~2 MB where the equivalent raw RGB frames cost ~340 MB.
"""

from __future__ import annotations

from collections import deque

from sentinel_ai.ports.frame_source import EncodedPacket


class PreRollBuffer:
    """A clip cannot start mid-GOP (spec §5.1): `flush()` always begins at a
    keyframe, which quantises the actual pre-roll up to the GOP boundary --
    more context than requested, never less.
    """

    def __init__(self, preroll_seconds: float) -> None:
        if preroll_seconds <= 0:
            raise ValueError(f"preroll_seconds must be positive, got {preroll_seconds}")
        self._preroll_seconds = preroll_seconds
        self._packets: deque[EncodedPacket] = deque()

    def _anchor_index(self) -> int | None:
        """Index of the keyframe `flush()` would start from, or None.

        The target is the *newest* keyframe at or before the horizon -- the
        smallest clip that still satisfies the pre-roll target. Before the
        buffer has accumulated `preroll_seconds` of history, no keyframe is
        old enough yet; in that ramp-up window the earliest keyframe held is
        the best available anchor. `None` only when no keyframe has been
        buffered at all, per `flush()`'s contract.
        """
        if not self._packets:
            return None
        horizon = self._packets[-1].pts - self._preroll_seconds
        first_keyframe: int | None = None
        anchor: int | None = None
        for index, candidate in enumerate(self._packets):
            if not candidate.is_keyframe:
                continue
            if first_keyframe is None:
                first_keyframe = index
            if candidate.pts <= horizon:
                anchor = index
        return anchor if anchor is not None else first_keyframe

    def append(self, packet: EncodedPacket) -> None:
        """Add a packet and evict everything older than the keyframe-aligned horizon."""
        self._packets.append(packet)
        anchor = self._anchor_index()
        if anchor is None:
            return
        # Everything before `anchor` is provably unreachable from any future
        # flush(): once a newer keyframe already satisfies the horizon, an
        # older one can never become the flush point again (pts only increases).
        for _ in range(anchor):
            self._packets.popleft()

    def flush(self) -> tuple[EncodedPacket, ...]:
        """Return buffered packets from the oldest keyframe at or before the horizon.

        Returns () when no keyframe has been seen yet.
        """
        anchor = self._anchor_index()
        if anchor is None:
            return ()
        return tuple(self._packets)[anchor:]

    def clear(self) -> None:
        self._packets.clear()

    @property
    def span_seconds(self) -> float:
        """pts span currently held; 0.0 when empty."""
        if not self._packets:
            return 0.0
        return self._packets[-1].pts - self._packets[0].pts
