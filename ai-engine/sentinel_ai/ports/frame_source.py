"""Video input port. Every source (RTSP, file, USB) reduces to this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameData:
    """A decoded frame.

    `pixels` is typed `object` deliberately: it carries a numpy array at
    runtime, but typing it as such would drag numpy into the ports layer and
    the architecture fitness test would (correctly) fail the build.
    """

    camera_id: str
    frame_index: int
    timestamp: float
    width: int
    height: int
    pixels: object


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    """A demuxed, still-encoded packet. `data` is bytes so ports stay codec-agnostic.

    `pts` is float seconds on the same monotonic timeline as `FrameData.timestamp` — the
    pre-roll buffer and the clip writer line packets and frames up on it.
    """

    camera_id: str
    data: bytes
    pts: float
    is_keyframe: bool
    codec: str
    """Container short codec name, lowercase — "h264", "hevc".

    A plain `str`, deliberately: the clip writer needs to know what it is remuxing, but a
    library enum here would drag a codec dependency into the pure ports layer and the
    architecture fitness test would reject it. Without this field the clip writer has to
    *assume* H.264, and a non-H.264 source produces an empty clip instead of an error.
    """


class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...

    @abstractmethod
    def packets(self) -> AsyncIterator[EncodedPacket]:
        """The same demux pass as `__aiter__`, fanned out as still-encoded packets.

        Feeds the pre-roll buffer and the clip writer (spec §4.2). Every demuxed packet is
        offered here even though only the sampled ones are decoded into frames.
        """

    @abstractmethod
    async def close(self) -> None: ...
