"""A camera that has stopped seeing (spec §6, environmental).

The failure this catches is the one a surveillance system is least able to notice by
itself: a camera that is still *up* — decoding, publishing telemetry, reporting healthy
— and is looking at the inside of a carrier bag. Every other detector in this package
goes quiet, and quiet is exactly what a working camera watching an empty corridor looks
like. An operator glancing at a wall of tiles sees no alerts and concludes nothing is
happening.

What it reads
--------------
`MotionAnalyzer` already produces a 16-bin normalised luma histogram per frame, for the
escalation gate's scene-change trigger. Nothing new is computed here; this reads the
signal that already exists, which is why the detector costs essentially nothing.

A covered lens has almost no tonal range — everything is the same near-black, or the
same flat white where a light is pointed at it — so the histogram **collapses into one
bin**. A real scene, however dull, spreads across many. `max(signature)` is therefore a
direct measure of "is there anything to see", and it needs no reference image, no
background model and no per-site calibration.

What it deliberately does not claim
------------------------------------
* **Not a redirection detector.** A camera turned to face a different direction still
  sees a varied scene, so the histogram stays spread and this stays silent. That case
  is already covered, by the gate's `scene_change` trigger — a large sustained
  signature delta — and duplicating it here would be the §42 mistake.
* **Not a defocus or spray detector.** A smeared lens loses *detail* while keeping its
  tonal range, and a luma histogram cannot see detail. Catching that needs a frequency
  or gradient measure this pipeline does not compute.
* **Not a distinction between malice and accident.** A bag over a lens, a lorry parked
  in front of it and a failed IR illuminator at night all look identical from here. The
  candidate says the camera cannot see, which is the operationally useful claim, and
  says nothing about why.

Night-time is the honest caveat
--------------------------------
A camera in an unlit room at 03:00 legitimately sees a near-uniform dark frame, and
this detector cannot tell that from a covered lens — they are the same picture. That is
why `min_clear_seconds` exists: the view must have been *varied* for a while before a
collapse counts as a change. A camera that is dark when the engine starts never fires,
and a site that goes dark gradually at dusk crosses the threshold slowly rather than at
a stroke. On a genuinely unlit camera, turn this capability off.

Pure: no clock, no I/O, standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.observation import BehaviourObservation

__all__ = ["TamperPolicy", "TamperState", "observe_tamper"]


class _Phase(StrEnum):
    UNKNOWN = "unknown"
    """Nothing believed yet — the state a camera starts in and stays in until it has
    been demonstrably clear. A camera that was obstructed before the engine started has
    no transition to report."""

    CLEAR = "clear"
    COLLAPSING = "collapsing"
    REPORTED = "reported"


@dataclass(frozen=True, slots=True)
class TamperPolicy:
    """Thresholds, named and configurable (spec §12)."""

    concentration_threshold: float = 0.85
    """Share of the frame that must fall in a single luma bin to count as blank.

    Sixteen bins, so one bin is a sixteenth of the tonal range. Eighty-five per cent of
    a frame inside one such band is a view with essentially nothing in it. Deliberately
    high: the cost of a false tamper alarm is an operator sent to check a camera that
    is fine, and doing that twice teaches them to ignore the third.
    """

    obstructed_seconds: float = 10.0
    """How long the collapse must persist before it is reported.

    Long enough to ride out the things that briefly blank a camera and are not
    tampering: a lorry pulling across the view, headlights sweeping the lens, an
    auto-exposure hunt after a light is switched on.
    """

    min_clear_seconds: float = 30.0
    """How long the view must have been varied before a collapse counts as a change.

    This is what stops a permanently dark camera alarming once a minute forever, and
    what makes this a detector of *becoming* blind rather than of *being* blind. See
    the module docstring on night-time.
    """

    def __post_init__(self) -> None:
        self._require(
            0.0 < self.concentration_threshold <= 1.0,
            "concentration_threshold must be in (0, 1]",
        )
        self._require(self.obstructed_seconds > 0.0, "obstructed_seconds must be > 0")
        self._require(self.min_clear_seconds >= 0.0, "min_clear_seconds must be >= 0")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class TamperState:
    """Whole-camera state — there is one view, so unlike every other detector here this
    is not keyed by track."""

    phase: _Phase = _Phase.UNKNOWN
    since: float | None = None
    """When the current phase began."""

    peak_concentration: float = 0.0
    """The worst reading seen during this collapse, carried so the candidate can say how
    blank the view actually went rather than what it happened to be on the last frame."""

    @property
    def phase_name(self) -> str:
        return self.phase.value


def concentration(signature: tuple[float, ...]) -> float:
    """How much of the frame sits in its single most populated luma bin.

    1.0 is a perfectly flat image; a varied scene is well under half. An empty
    signature answers 0.0 — "no evidence of blankness" — rather than 1.0, because a
    missing measurement must not be able to raise an alarm.
    """
    return max(signature) if signature else 0.0


def observe_tamper(
    observation: BehaviourObservation,
    policy: TamperPolicy,
    state: TamperState,
) -> tuple[TamperState, tuple[BehaviourCandidate, ...]]:
    """Advance the machine by one frame. Pure; the caller threads the state.

    Raised once per episode: the camera moves to `REPORTED` and stays there until the
    view clears again. A lens covered for an hour is one alert.
    """
    now = observation.timestamp
    blank = concentration(observation.scene.scene_signature) >= policy.concentration_threshold

    if not blank:
        if state.phase in (_Phase.CLEAR, _Phase.UNKNOWN) and state.since is not None:
            # Already accumulating clear time; leave `since` alone so the clock keeps
            # running rather than restarting on every good frame.
            return replace(state, phase=_Phase.CLEAR), ()
        # Entering clear from a collapse, a report, or the very first frame.
        return TamperState(phase=_Phase.CLEAR, since=now), ()

    # Blank from here on.
    if state.phase is _Phase.REPORTED:
        return state, ()

    if state.phase in (_Phase.UNKNOWN, _Phase.CLEAR):
        if (
            state.phase is _Phase.UNKNOWN
            or state.since is None
            or now - state.since < policy.min_clear_seconds
        ):
            # Not yet believed to have been seeing anything, so there is no transition
            # here to report.
            #
            # Reset to `UNKNOWN` rather than staying `CLEAR`, and this is not
            # bookkeeping — an earlier version returned the state untouched, which left
            # `since` pointing at the start of a clear run that had *already ended*.
            # The clock then went on accruing while the view was blank, `min_clear_
            # seconds` elapsed with nothing to see, and a camera blanked five seconds
            # after startup reported tampering half a minute later. The clear run is
            # over; the view has to earn it again.
            return TamperState(phase=_Phase.UNKNOWN, since=None), ()
        return (
            replace(
                state,
                phase=_Phase.COLLAPSING,
                since=now,
                peak_concentration=concentration(observation.scene.scene_signature),
            ),
            (),
        )

    # COLLAPSING: the obstruction clock is running.
    peak = max(state.peak_concentration, concentration(observation.scene.scene_signature))
    if state.since is None or now - state.since < policy.obstructed_seconds:
        return replace(state, peak_concentration=peak), ()

    blanked_for = now - state.since
    return (
        replace(state, phase=_Phase.REPORTED, peak_concentration=peak),
        (
            BehaviourCandidate(
                kind=BehaviourKind.CAMERA_TAMPER,
                summary=(
                    f"this camera appears to have stopped seeing: {peak:.0%} of the "
                    f"frame has been a single flat tone for {blanked_for:.0f}s. The "
                    f"view may be obstructed, the lens covered, or the scene "
                    f"unlit — the camera is still delivering frames either way"
                ),
                # Nobody to attribute this to. Inventing a track id would be worse than
                # an empty tuple — see `BehaviourCandidate.track_ids`.
                track_ids=(),
                started_at=state.since,
            ),
        ),
    )
