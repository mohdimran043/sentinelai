"""Core domain entities. Pure — no I/O, no clock reads, no third-party runtime deps."""

from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID

from sentinel_ai.domain.welfare import WelfareAssessment


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

    def signature_delta_or_none(self, previous: tuple[float, ...] | None) -> float | None:
        """`signature_delta`, returning None instead of raising when incomparable.

        A mid-stream resolution change (an RTSP reconnect renegotiating), or any
        change to the signature extractor's bin count, leaves two histograms
        with different lengths. That is *not* the same as "no change": each
        caller decides what an incomparable pair means for it, and none of them
        may take the camera's pipeline down. `signature_delta` still raises, for
        callers that genuinely want strictness.
        """
        if previous is not None and len(previous) != len(self.scene_signature):
            return None
        return self.signature_delta(previous)

    def tracks_of(self, labels: Collection[str]) -> Iterator[Track]:
        return (track for track in self.tracks if track.label in labels)


class EscalationReason(StrEnum):
    """Why a frame was escalated to the vision-language model.

    The seven from spec §4.1, plus `FALL_SUSPECTED` (§7), which is a different kind of
    thing from the rest and is grouped last for that reason. The first six are
    *predicates on one frame* asking "is this scene worth a look"; `USER_REQUESTED` is
    a person asking directly; `FALL_SUSPECTED` is a **temporal state machine** over
    several seconds reporting that a specific signature completed
    (`domain/behaviour/fall.py`).
    """

    NEW_SALIENT_TRACK = "new_salient_track"
    SCENE_CHANGE = "scene_change"
    DWELL_EXCEEDED = "dwell_exceeded"
    SPEED_ANOMALY = "speed_anomaly"
    TRACK_COUNT_SPIKE = "track_count_spike"
    PERIODIC_SUMMARY = "periodic_summary"
    USER_REQUESTED = "user_requested"

    FALL_SUSPECTED = "fall_suspected"
    """A person appears to have fallen and has remained down (spec §7).

    **Bypasses the gate's three governors**, exactly as `USER_REQUESTED` does, and for
    a reason the others cannot claim: the fall state machine has already deduplicated
    this to one report per episode and will not raise again until the person stands
    up. The governors exist to stop uninteresting scenes spending GPU
    (`domain/policy/escalation.py`); a completed fall signature is neither
    uninteresting nor repeated, and a cooldown window swallowing the one escalation
    that mattered is the failure this whole subsystem exists to avoid.

    Named for what was observed, not what it means. `fall_suspected`, never
    `fall_detected`: ADR 10's argument is that the second reads to every downstream
    consumer as a trained classifier's verdict, and what is behind this is a geometry
    state machine awaiting vision-language confirmation.
    """

    ABANDONED_OBJECT = "abandoned_object"
    """An object a person was with has been left alone and still (spec §6).

    Like every behaviour reason here, and unlike the six automatic triggers, this is a
    temporal state machine's report rather than a predicate on one frame — see
    `domain/behaviour/abandonment.py`.
    """

    CAMERA_TAMPER = "camera_tamper"
    """This camera appears to have stopped seeing (spec §6).

    The one reason on this list that is about the *camera* rather than about what is in
    front of it, and the only one raised with no track ids attached. It matters
    disproportionately because every other detector goes silent when it fires, and
    silence is indistinguishable from a quiet site.
    """

    ZONE_INTRUSION = "zone_intrusion"
    """A person's feet entered a restricted area drawn on this camera (spec §6)."""

    LINE_CROSSING = "line_crossing"
    """A person crossed a virtual boundary drawn on this camera (spec §6)."""

    UNAUTHORIZED_PERSON = "unauthorized_person"
    """A tracked person did not match any face authorised on this camera (spec §8-§12).

    Named for what was measured, like `FALL_SUSPECTED` and for a sharper version of the
    same reason: this is a claim about a specific identifiable individual. What the
    system knows is that several clear views of one person matched nobody on this
    camera's roster — not that the person does not belong, which depends on whether the
    roster is complete and current. The description says so; nothing downstream may
    render it as a verdict.
    """


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
    """The unit published to the Web Platform. Carries no model identity (spec §3.3).

    Two timestamps, deliberately, because one field cannot be both things at once:

    * `occurred_at` is **Unix epoch seconds** (UTC, fractional) — the field a consumer
      sorts and displays on. It survives a restart and it means the same thing for an
      RTSP camera and a replay camera in the same process.
    * `source_timestamp` is the raw **source timeline** the scene was observed on:
      `time.monotonic()` for a live RTSP camera, seconds-from-start-of-file for a
      replayed one. It is the timeline a clip's pts and `CameraTelemetry` are on, so it
      is what correlates an event with its evidence — and it is comparable *only*
      within one process run for one camera.

    Neither is derived here: `domain/` reads no clock, so both arrive already computed
    (see `VlmScheduler._assemble`, the one place an `Event` is constructed).

    `welfare` is a vision-language model's opinion about this keyframe (spec §5),
    never a detector's verdict — see `domain/welfare.py`. It defaults to
    `WelfareAssessment.none()` via `field(default_factory=...)`, not a bare default:
    a bare `WelfareAssessment.none()` is evaluated once at class-definition time and
    would be shared by every `Event` that never passes its own. `WelfareAssessment`
    is frozen, so aliasing it would not corrupt state today, but the factory is
    still the correct mechanism to reach for, not a shortcut that happens to work.
    """

    event_id: UUID
    camera_id: str
    occurred_at: float
    reason: EscalationReason
    threat: ThreatScore
    description: str
    suggested_action: str
    source_timestamp: float | None = None
    labels: tuple[str, ...] = ()
    track_ids: tuple[int, ...] = ()
    """Every salient track in view when this was captured — the *scene*, not the
    subject. Useful as context and for search; useless for identifying who an event is
    about, because on a busy camera it is most of the frame and it changes every
    frame."""

    subject_track_ids: tuple[int, ...] = ()
    """Who this event is **about**, when that is knowable.

    Distinct from `track_ids`, and the distinction is load-bearing rather than
    pedantic. A behaviour detector knows precisely whose behaviour it reported — track
    7 fell, track 12 crossed the line — while `track_ids` is everybody who happened to
    be in shot. Keying an alert on the latter was tried and produced exactly the
    failure it deserved: subjects like `3,4,43,120,130,133,146,160`, different on every
    frame, so eleven intrusions by the same handful of people became eleven separate
    alerts and §17's deduplication requirement was silently unmet.

    Empty when nothing can attribute the event to a person: the six automatic triggers
    describe a scene rather than a subject, and camera tampering has nobody to
    attribute. Empty is a real answer — `AlertKey` treats it as a camera-level episode
    — and it deliberately does **not** fall back to `track_ids`, because that fallback
    is the bug.
    """
    keyframe_uri: str | None = None
    clip_uri: str | None = None
    description_unavailable: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
    welfare: WelfareAssessment = field(default_factory=WelfareAssessment.none)
