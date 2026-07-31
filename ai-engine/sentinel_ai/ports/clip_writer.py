"""Evidence clip port.

Clips must begin 3 s before the anomaly (spec §23, §26), which is why the
decode stage keeps a pre-roll ring buffer rather than starting to record on
escalation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from sentinel_ai.ports.frame_source import FrameData


class ClipWriter(ABC):
    @abstractmethod
    async def write(
        self, camera_id: str, event_id: UUID, frames: Sequence[FrameData], fps: float
    ) -> str:
        """Persist an evidence clip and return its URI."""
