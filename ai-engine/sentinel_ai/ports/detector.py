"""Object detection port (pipeline stage 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Detection
from sentinel_ai.ports.frame_source import FrameData


class ObjectDetector(ABC):
    @abstractmethod
    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        """Detections for a single frame.

        Results are reused by tracking, the gate, the VLM, and reporting —
        detection is never re-run per downstream consumer (spec §7).
        """
