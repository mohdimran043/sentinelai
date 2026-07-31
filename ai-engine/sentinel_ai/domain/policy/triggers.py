"""The escalation predicates from spec §4.1.

Each is a pure function of a TriggerContext so it can be tested in isolation at
any simulated clock value. USER_REQUESTED is not here: it bypasses predicates
entirely and is handled by the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import hypot

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState


@dataclass(frozen=True, slots=True)
class DwellAnchor:
    """Where a track settled, and when. Dwell is measured from `since`."""

    cx: float
    cy: float
    since: float


@dataclass(frozen=True, slots=True)
class TriggerContext:
    scene: SceneState
    profile: CameraProfile
    previous_signature: tuple[float, ...] | None
    scene_delta_streak: int
    dwell_anchors: dict[int, DwellAnchor]
    last_summary_at: float | None
    speed_threshold_px_s: float


@dataclass(frozen=True, slots=True)
class TriggerOutcome:
    fired: bool
    reason: EscalationReason | None = None
    detail: str = ""


_NOT_FIRED = TriggerOutcome(fired=False)


def new_salient_track(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        if track.age_frames >= ctx.profile.min_track_frames:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.NEW_SALIENT_TRACK,
                detail=f"{track.label} track {track.track_id} age={track.age_frames}",
            )
    return _NOT_FIRED


def scene_change(ctx: TriggerContext) -> TriggerOutcome:
    delta = ctx.scene.signature_delta_or_none(ctx.previous_signature)
    # None means the histograms have different bin counts, so they are
    # incomparable — which is not evidence of a changed scene.
    if delta is None or delta < ctx.profile.scene_delta_threshold:
        return _NOT_FIRED
    if ctx.scene_delta_streak + 1 < ctx.profile.scene_delta_frames:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.SCENE_CHANGE,
        detail=f"signature delta={delta:.3f} sustained {ctx.scene_delta_streak + 1} frames",
    )


def dwell_exceeded(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        anchor = ctx.dwell_anchors.get(track.track_id)
        if anchor is None:
            continue
        drifted = hypot(track.box.cx - anchor.cx, track.box.cy - anchor.cy)
        if drifted > ctx.profile.dwell_radius_px:
            continue
        dwelled = ctx.scene.timestamp - anchor.since
        if dwelled >= ctx.profile.dwell_seconds:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.DWELL_EXCEEDED,
                detail=f"{track.label} track {track.track_id} dwelled {dwelled:.1f}s",
            )
    return _NOT_FIRED


def speed_anomaly(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        if track.speed_px_s > ctx.speed_threshold_px_s:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.SPEED_ANOMALY,
                detail=(
                    f"{track.label} track {track.track_id} at {track.speed_px_s:.0f}px/s "
                    f"(threshold {ctx.speed_threshold_px_s:.0f})"
                ),
            )
    return _NOT_FIRED


def track_count_spike(ctx: TriggerContext) -> TriggerOutcome:
    count = sum(1 for _ in ctx.scene.tracks_of(ctx.profile.salient_classes))
    if count <= ctx.profile.track_count_baseline:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.TRACK_COUNT_SPIKE,
        detail=f"{count} concurrent tracks (baseline {ctx.profile.track_count_baseline})",
    )


def periodic_summary(ctx: TriggerContext) -> TriggerOutcome:
    if ctx.last_summary_at is None:
        return TriggerOutcome(
            fired=True,
            reason=EscalationReason.PERIODIC_SUMMARY,
            detail="initial scene summary",
        )
    elapsed = ctx.scene.timestamp - ctx.last_summary_at
    if elapsed < ctx.profile.summary_interval_seconds:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.PERIODIC_SUMMARY,
        detail=f"{elapsed:.0f}s since last summary",
    )


def advance_dwell_anchors(ctx: TriggerContext) -> dict[int, DwellAnchor]:
    """Recompute anchors for the current frame.

    A track staying inside `dwell_radius_px` of its anchor keeps the original
    `since`; a track that left gets a fresh anchor. Departed tracks are dropped
    so the mapping cannot grow without bound.
    """
    anchors: dict[int, DwellAnchor] = {}
    for track in ctx.scene.tracks:
        cx, cy = track.box.cx, track.box.cy
        previous = ctx.dwell_anchors.get(track.track_id)
        if previous is not None and (
            hypot(cx - previous.cx, cy - previous.cy) <= ctx.profile.dwell_radius_px
        ):
            anchors[track.track_id] = previous
        else:
            anchors[track.track_id] = DwellAnchor(cx=cx, cy=cy, since=ctx.scene.timestamp)
    return anchors


# Priority order: when several triggers fire, the first one names the reason.
ALL_TRIGGERS: tuple[Callable[[TriggerContext], TriggerOutcome], ...] = (
    speed_anomaly,
    dwell_exceeded,
    track_count_spike,
    scene_change,
    new_salient_track,
    periodic_summary,
)
