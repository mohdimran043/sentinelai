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


class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...

    @abstractmethod
    async def close(self) -> None: ...
