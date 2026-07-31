"""The escalation gate (spec §4.1) — the core of SentinelAI's cost/accuracy tradeoff.

`decide` is a pure function of (scene, profile, state). It returns both a
decision and the next state, so a caller threads state through frames without
the gate ever holding mutable state or reading a clock.

Order of evaluation matters and is deliberate:
  1. vlm_enabled           — a disabled camera short-circuits everything
  2. triggers              — is anything worth looking at?
  3. duplicate suppression — have we already described this exact scene?
  4. rate budget           — can we afford it?

Suppression at stages 3 and 4 still reports the trigger reason, so telemetry can
show what the gate declined and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.domain.policy.rate_budget import TokenBucket
from sentinel_ai.domain.policy.triggers import (
    ALL_TRIGGERS,
    DwellAnchor,
    TriggerContext,
    advance_dwell_anchors,
)

DEDUP_EPSILON = 0.05
"""Scene-signature distance below which two frames count as the same scene."""


@dataclass(frozen=True, slots=True)
class GateState:
    """Everything the gate must remember between frames."""

    bucket: TokenBucket
    previous_signature: tuple[float, ...] | None = None
    scene_delta_streak: int = 0
    dwell_anchors: dict[int, DwellAnchor] = field(default_factory=dict)
    last_summary_at: float | None = None
    last_escalation_at: float | None = None
    last_escalated_signature: tuple[float, ...] | None = None

    @classmethod
    def initial(cls, profile: CameraProfile, now: float) -> GateState:
        return cls(
            bucket=TokenBucket.full(
                capacity=profile.bucket_capacity,
                refill_seconds=profile.bucket_refill_seconds,
                now=now,
            )
        )


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    should_escalate: bool
    reason: EscalationReason | None = None
    detail: str = ""
    suppressed_by: str | None = None


@dataclass(frozen=True, slots=True)
class GateOutcome:
    decision: EscalationDecision
    state: GateState


def _speed_threshold(profile: CameraProfile) -> float:
    """Slice 1 uses the fixed fallback. Phase 4 supplies a learned percentile."""
    return profile.speed_fallback_px_s


def _next_streak(scene: SceneState, profile: CameraProfile, state: GateState) -> int:
    delta = scene.signature_delta_or_none(state.previous_signature)
    # An incomparable delta (the bin count changed) makes a sustained-change
    # streak meaningless, so the streak resets rather than extends.
    if delta is None or delta < profile.scene_delta_threshold:
        return 0
    return state.scene_delta_streak + 1


def _is_duplicate(scene: SceneState, state: GateState) -> bool:
    if state.last_escalated_signature is None:
        return False
    delta = scene.signature_delta_or_none(state.last_escalated_signature)
    # Incomparable is not "unchanged": a resolution change must not be mistaken
    # for a repeat of the scene we have already described.
    return delta is not None and delta < DEDUP_EPSILON


def _context(scene: SceneState, profile: CameraProfile, state: GateState) -> TriggerContext:
    return TriggerContext(
        scene=scene,
        profile=profile,
        previous_signature=state.previous_signature,
        scene_delta_streak=state.scene_delta_streak,
        dwell_anchors=state.dwell_anchors,
        last_summary_at=state.last_summary_at,
        speed_threshold_px_s=_speed_threshold(profile),
    )


def decide(scene: SceneState, profile: CameraProfile, state: GateState) -> GateOutcome:
    now = scene.timestamp
    ctx = _context(scene, profile, state)

    carried = replace(
        state,
        previous_signature=scene.scene_signature,
        scene_delta_streak=_next_streak(scene, profile, state),
        dwell_anchors=advance_dwell_anchors(ctx),
    )

    if not profile.vlm_enabled:
        return GateOutcome(
            decision=EscalationDecision(should_escalate=False, suppressed_by="vlm_disabled"),
            state=carried,
        )

    fired = next((outcome for trigger in ALL_TRIGGERS if (outcome := trigger(ctx)).fired), None)
    if fired is None or fired.reason is None:
        return GateOutcome(decision=EscalationDecision(should_escalate=False), state=carried)

    if _is_duplicate(scene, state):
        return GateOutcome(
            decision=EscalationDecision(
                should_escalate=False,
                reason=fired.reason,
                detail=fired.detail,
                suppressed_by="duplicate_scene",
            ),
            state=replace(carried, bucket=carried.bucket.refilled(now)),
        )

    allowed, bucket = carried.bucket.try_consume(now)
    if not allowed:
        return GateOutcome(
            decision=EscalationDecision(
                should_escalate=False,
                reason=fired.reason,
                detail=fired.detail,
                suppressed_by="rate_budget",
            ),
            state=replace(carried, bucket=bucket),
        )

    summary_at = (
        now if fired.reason is EscalationReason.PERIODIC_SUMMARY else carried.last_summary_at
    )
    return GateOutcome(
        decision=EscalationDecision(should_escalate=True, reason=fired.reason, detail=fired.detail),
        state=replace(
            carried,
            bucket=bucket,
            last_summary_at=summary_at,
            last_escalation_at=now,
            last_escalated_signature=scene.scene_signature,
        ),
    )


def force(reason: EscalationReason, state: GateState, now: float) -> GateOutcome:
    """Escalate unconditionally — used for USER_REQUESTED (spec §6)."""
    return GateOutcome(
        decision=EscalationDecision(
            should_escalate=True, reason=reason, detail="forced escalation"
        ),
        state=replace(state, last_escalation_at=now),
    )
