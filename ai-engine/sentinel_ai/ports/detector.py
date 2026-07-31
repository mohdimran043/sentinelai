"""Object detection port (pipeline stage 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Detection
from sentinel_ai.ports.frame_source import FrameData


class ObjectDetector(ABC):
    @abstractmethod
    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        """Detections for a single frame.

        Results are reused by tracking, the gate, the VLM, and reporting — detection is
        never re-run per downstream consumer (spec §7).

        An implementation's `ModelRuntime.capabilities().batch_max` may advertise a batch
        size greater than 1, but this method stays per-frame in Phase 1B: the slice is
        single-camera, so every batch would have size 1 and a batch method would be
        untested, unused code. `batch_max` is advertised for Phase 2 multi-camera use only
        (spec §4.3); no Phase 1B adapter honours it.
        """
