"""Evidence clip port.

Clips must begin 3 s before the anomaly (spec §23, §26), which is why the decode stage keeps
a pre-roll ring buffer rather than starting to record on escalation.

`open` returns a handle so a clip is written as packets arrive instead of assembled in memory
first: a 3 s pre-roll plus 5 s post-roll at 1080p30 is ~180 raw frames, roughly 1.1 GB of
host RAM per concurrent clip on a machine with ~7 GB free (Phase 1B spec §4.1). Streaming
encoded packets through open/append/finish keeps peak memory at the pre-roll ring plus one
packet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from sentinel_ai.ports.frame_source import EncodedPacket


class ClipHandle(ABC):
    @abstractmethod
    async def append(self, packet: EncodedPacket) -> None: ...

    @abstractmethod
    async def finish(self) -> str:
        """Finalise the clip and return its URI."""

    @abstractmethod
    async def abort(self) -> None:
        """Discard a partial clip. MUST NOT raise.

        A pipeline shutdown mid-clip must not turn cleanup into a second failure.
        """


class ClipWriter(ABC):
    @abstractmethod
    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle: ...
