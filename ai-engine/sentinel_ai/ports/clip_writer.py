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

    @property
    def notify_uri(self) -> str | None:
        """A short clip of the same moment, for a notification, or `None`.

        Defaulted rather than abstract, because it is genuinely optional: a writer that
        produces one clip is a complete writer, and every existing one keeps working
        without knowing this exists.

        Why a second, shorter clip rather than the evidence clip's URL. The two want
        opposite things. Evidence wants context — the seconds before, the seconds after,
        enough to see what led up to it — and is watched later by somebody who chose to
        sit down with it. A notification is read on a phone at 3am by somebody deciding
        whether to walk down a corridor, and the useful length is however long it takes
        to see what happened. Sending the long one there does not fail loudly; it just
        gets watched less.

        Only read after `finish()`, and only for events at or above
        `SENTINEL_NOTIFY_CLIP_MIN_SEVERITY`.
        """
        return None

    @abstractmethod
    async def abort(self) -> None:
        """Discard a partial clip. MUST NOT raise.

        A pipeline shutdown mid-clip must not turn cleanup into a second failure.
        """


class ClipWriter(ABC):
    @abstractmethod
    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle: ...
