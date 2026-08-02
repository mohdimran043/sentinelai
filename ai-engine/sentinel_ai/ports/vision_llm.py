"""Vision LLM port.

The provider abstraction is mandatory, not a convenience: spec §10 and §16.1
require Qwen, Gemini, GPT, Gemma Vision and LLaVA to be interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from sentinel_ai.domain.entities import SceneState
from sentinel_ai.domain.welfare import WelfareAssessment
from sentinel_ai.ports.frame_source import FrameData


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """Everything spec §22 requires the VLM to receive."""

    keyframe: FrameData
    scene: SceneState
    history: tuple[str, ...]
    camera_label: str
    reason_detail: str


@dataclass(frozen=True, slots=True)
class SceneDescription:
    """What one keyframe made the vision-language model say.

    `welfare` defaults to `WelfareAssessment.none()` via `field(default_factory=...)`
    — not a bare default, which would be evaluated once at class-definition time and
    shared by every response that never sets its own — so every existing adapter that
    predates this field keeps constructing `SceneDescription` untouched.
    """

    description: str
    threat_value: float
    suggested_action: str
    welfare: WelfareAssessment = field(default_factory=WelfareAssessment.none)


class VisionLanguageModel(ABC):
    @abstractmethod
    async def describe(self, request: VisionRequest) -> SceneDescription: ...
