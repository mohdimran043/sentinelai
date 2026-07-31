"""Core domain entities. Pure — no I/O, no clock reads, no third-party runtime deps."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: BBox


@dataclass(frozen=True, slots=True)
class Track:
    """A detection associated across frames by the tracker."""

    track_id: int
    label: str
    box: BBox
    age_frames: int
    speed_px_s: float


@dataclass(frozen=True, slots=True)
class SceneState:
    """Everything the escalation gate is allowed to look at.

    Deliberately contains no pixel data: the gate must be decidable from cheap
    signals alone (spec §4.1). Keyframe bytes are fetched only after escalation.
    """

    camera_id: str
    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    motion_energy: float
    scene_signature: tuple[float, ...]

    def signature_delta(self, previous: tuple[float, ...] | None) -> float:
        """Total-variation distance between two normalised histograms, in [0, 1].

        A missing previous signature yields 0.0 so the first frame of a stream
        never fires SceneChange.
        """
        if previous is None:
            return 0.0
        if len(previous) != len(self.scene_signature):
            raise ValueError(
                f"signature length mismatch: {len(self.scene_signature)} vs {len(previous)}"
            )
        return sum(abs(a - b) for a, b in zip(self.scene_signature, previous, strict=True)) / 2.0

    def tracks_of(self, labels: Collection[str]) -> Iterator[Track]:
        return (track for track in self.tracks if track.label in labels)


class EscalationReason(StrEnum):
    """The seven triggers from spec §4.1."""

    NEW_SALIENT_TRACK = "new_salient_track"
    SCENE_CHANGE = "scene_change"
    DWELL_EXCEEDED = "dwell_exceeded"
    SPEED_ANOMALY = "speed_anomaly"
    TRACK_COUNT_SPIKE = "track_count_spike"
    PERIODIC_SUMMARY = "periodic_summary"
    USER_REQUESTED = "user_requested"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_BANDS: tuple[tuple[float, Severity], ...] = (
    (0.8, Severity.CRITICAL),
    (0.6, Severity.HIGH),
    (0.4, Severity.MEDIUM),
    (0.2, Severity.LOW),
)


@dataclass(frozen=True, slots=True)
class ThreatScore:
    value: float
    severity: Severity

    @classmethod
    def from_value(cls, value: float) -> ThreatScore:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threat score must be between 0.0 and 1.0, got {value}")
        for threshold, severity in _SEVERITY_BANDS:
            if value >= threshold:
                return cls(value=value, severity=severity)
        return cls(value=value, severity=Severity.INFO)


@dataclass(frozen=True, slots=True)
class Event:
    """The unit published to the Web Platform. Carries no model identity (spec §3.3)."""

    event_id: UUID
    camera_id: str
    occurred_at: float
    reason: EscalationReason
    threat: ThreatScore
    description: str
    suggested_action: str
    labels: tuple[str, ...] = ()
    track_ids: tuple[int, ...] = ()
    keyframe_uri: str | None = None
    clip_uri: str | None = None
    description_unavailable: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
