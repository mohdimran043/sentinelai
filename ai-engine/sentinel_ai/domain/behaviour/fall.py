"""Possible fall / collapse, as a temporal signature rather than a posture (spec §7).

The thing this module exists to *not* be
----------------------------------------
A person lying on the ground is not a fall. It is a person lying on the ground, and
there are many innocent reasons for it — a resident napping on a dayroom sofa, someone
doing floor exercises, a cleaner reaching under furniture. A detector that fires on
posture alone produces exactly the alert an operator learns to dismiss, and an alert
that gets dismissed is worse than no alert because it costs attention that a real
collapse then does not get.

So what is detected here is the **transition**, in the order §7 lays out:

    upright, for long enough to be believed
        -> rapid downward movement
            -> body becomes horizontal
                -> and stays down, still, for long enough
                    -> candidate

Every one of those four is required, and the candidate is raised once per episode.
Someone already on the floor when they enter frame raises nothing — the machine never
saw them upright, so it has no transition to report and says so by staying silent.
That is a real limitation and it is the honest one: this detects falling, not lying.

Scale invariance, which is why nothing here is in pixels per second
-------------------------------------------------------------------
A person three metres from the camera and the same person twenty metres away fall at
wildly different pixel rates, so any `px/s` threshold is really a threshold on
distance-from-camera wearing a speed's clothes — tuned on one camera and wrong on the
next. Every rate and distance below is therefore expressed in **body heights**: the
descent rate is body-heights per second, the stillness radius is a fraction of a body
height. The person's own bounding box supplies the unit, so the same `FallPolicy`
means the same thing on a corridor camera and a car-park camera.

Two ways to read the body, and pose is the better one
------------------------------------------------------
Bounding-box aspect ratio (width/height) is the geometry-only signal and it is
available on every camera: an upright adult is tall and narrow, a fallen one is short
and wide. It is also confounded by a great deal — a crouch, a carried object, a
tracker box that grew to include the chair someone sat in.

Torso angle from pose keypoints is confounded by much less, so when a camera enabled a
capability that loads a pose model, the machine prefers it and falls back to aspect
ratio per frame when the skeleton cannot answer (see
`observation.PersonPose.torso_angle_degrees`). **Pose is never required.** A camera
without it runs the same state machine on the weaker signal and says which signal it
used, in `FallEvidence.used_pose`, so nothing downstream has to guess how much to
trust a given candidate.

What this is worth, and what it is not
---------------------------------------
A completed signature is evidence that something happened worth a human's attention.
It is **not** a medical finding and nothing here may be rendered as one: ADR 10's whole
argument is that a `fall_detected: true` boolean reads to every downstream consumer as
a trained classifier's verdict. This module therefore emits no probability and no
boolean verdict — it emits `FallEvidence`, the measurements that made it fire, and a
vision-language model is asked to confirm before anything reaches an operator. The
welfare concern that results carries `basis="temporal_pose_vlm"`, which is ADR 10's own
prescription for a second judgement source: a new basis value, never a widening of
`single_frame_vlm`.

Pure: no clock, no I/O, standard library only. Time arrives as
`BehaviourObservation.timestamp`, on the camera's source timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import hypot

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.observation import BehaviourObservation

__all__ = [
    "PERSON_LABEL",
    "FallEvidence",
    "FallPolicy",
    "FallTracker",
    "Posture",
    "observe_falls",
]

PERSON_LABEL = "person"
"""The only detector class this machine reads. A falling car is not this feature."""


class Posture(StrEnum):
    """How the body is currently oriented, as this machine judges it.

    `UNKNOWN` is a real member and not a failure: a track whose box is degenerate
    (zero height, which a tracker can emit for one frame at the edge of frame) has no
    aspect ratio, and a pose that could not answer has no angle. Guessing would let a
    single bad frame either start or complete a fall signature.
    """

    UPRIGHT = "upright"
    HORIZONTAL = "horizontal"
    UNKNOWN = "unknown"


class _Phase(StrEnum):
    """Where one tracked person is in the signature. Private: nothing outside needs to
    branch on it, and exposing it would invite a caller to treat `DOWN` as a finding
    when `DOWN` is precisely the state that has not yet earned one."""

    UPRIGHT = "upright"
    DESCENDING = "descending"
    DOWN = "down"
    REPORTED = "reported"


@dataclass(frozen=True, slots=True)
class FallPolicy:
    """Every threshold, named and configurable (spec §12: none of this is hard-coded).

    The defaults are a starting point for an indoor camera watching people at a
    conversational distance, not a calibration. §32 requires them to be measured
    rather than asserted, and `docs/` records what measurement has actually been done.
    """

    min_upright_seconds: float = 0.4
    """How long a person must have been upright before a descent counts.

    Guards the two cheapest false positives there are: a person who is *already* on
    the floor when the track begins, and a track that flickers into existence on a
    partially-occluded body whose first box happens to be wide. Neither is a fall, and
    neither has a transition to report.
    """

    upright_aspect_max: float = 0.75
    """Bounding-box width/height at or below which a person reads as upright.

    An adult standing is nearer 0.3-0.5; 0.75 leaves room for outstretched arms and a
    carried bag without admitting a crouch.
    """

    horizontal_aspect_min: float = 1.1
    """Width/height at or above which a person reads as horizontal. Above 1.0 means
    genuinely wider than tall, which a standing person is not."""

    upright_torso_degrees_max: float = 35.0
    """Torso angle from vertical below which pose says upright. Leaning to pick
    something up should not read as the start of a fall."""

    horizontal_torso_degrees_min: float = 60.0
    """Torso angle from vertical above which pose says horizontal.

    Deliberately short of 90: a person collapsed against a wall or slumped over a
    table is not flat, and requiring flatness would miss exactly the cases where
    someone has come to rest against something on the way down.
    """

    min_keypoint_confidence: float = 0.4
    """Per-joint score below which a keypoint is treated as absent.

    A pose model returns a full skeleton whatever it can see, so this is what stops a
    guessed hip from deciding a posture. Below this bar the frame falls back to aspect
    ratio rather than trusting the skeleton.
    """

    min_descent_rate: float = 0.7
    """Body heights per second of downward centroid movement that counts as a fall
    rather than a sit.

    The measurement that separates the two: sitting deliberately moves the centroid
    roughly a third of a body height over a second or more; an uncontrolled drop
    covers more, faster. In body heights per second so it means the same thing at
    every distance from the camera — see this module's docstring.
    """

    descent_window_seconds: float = 2.5
    """How long the body has to reach horizontal after the rapid descent began.

    A fall completes in well under a second; the window is generous because tracking
    and sampling are not. Past it the machine gives up on the episode and returns to
    watching — a descent that never became horizontal was a stumble, a crouch, or a
    tracking artefact, and none of those is worth an operator's time.
    """

    settle_seconds: float = 3.0
    """How long the person must remain down and still before a candidate is raised.

    This is the clause that makes the difference between "fell" and "fell and is not
    getting up", and the second is what a welfare system exists to notice. It is also
    the dominant term in time-to-alert, so it trades directly against §33's latency
    target: every second here is a second before anyone is told.
    """

    still_radius: float = 0.35
    """How far the centroid may drift, in body heights, and still count as settled.

    Not zero: a person breathing, shifting, or being partially occluded moves the box.
    Movement beyond this restarts the settle timer rather than cancelling the episode —
    someone struggling to get up is still down.
    """

    def __post_init__(self) -> None:
        self._require(self.min_upright_seconds >= 0.0, "min_upright_seconds must be >= 0")
        self._require(self.upright_aspect_max > 0.0, "upright_aspect_max must be > 0")
        self._require(
            self.horizontal_aspect_min > self.upright_aspect_max,
            "horizontal_aspect_min must exceed upright_aspect_max, or a body would read "
            "as both upright and horizontal at once",
        )
        self._require(
            0.0 <= self.upright_torso_degrees_max < self.horizontal_torso_degrees_min <= 90.0,
            "torso angles must satisfy 0 <= upright_max < horizontal_min <= 90, leaving a "
            "band in between that is neither",
        )
        self._require(
            0.0 <= self.min_keypoint_confidence <= 1.0,
            "min_keypoint_confidence must be in [0, 1]",
        )
        self._require(self.min_descent_rate > 0.0, "min_descent_rate must be > 0")
        self._require(self.descent_window_seconds > 0.0, "descent_window_seconds must be > 0")
        self._require(self.settle_seconds > 0.0, "settle_seconds must be > 0")
        self._require(self.still_radius >= 0.0, "still_radius must be >= 0")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class FallEvidence:
    """The measurements that completed one signature.

    Measurements, deliberately, and **no score**. A single float would be read as a
    probability by everything downstream, and there is no calibration behind this that
    could earn one — ADR 10 makes the same argument about the VLM's welfare opinion.
    What a consumer gets instead is what actually happened, in units it can weigh: how
    fast the body went down, how long it has stayed down, and whether a pose model or
    only a bounding box was doing the reading.
    """

    track_id: int
    descent_rate: float
    """Peak downward centroid rate during the descent, in body heights per second."""

    settled_seconds: float
    """How long the body had been down and still when the candidate was raised. At
    least `FallPolicy.settle_seconds` by construction."""

    used_pose: bool
    """Whether the posture that completed the signature came from a torso angle rather
    than a bounding-box aspect ratio. A consumer that wants to weight pose-backed
    candidates more heavily can; nothing here does it for them."""

    started_at: float
    """When the descent began, on the camera's source timeline — the instant a clip
    should be centred on, which is earlier than when the candidate was raised."""

    def summary(self) -> str:
        """One line for the escalation detail and the VLM prompt.

        Hedged on purpose, and this wording is load-bearing: §7 forbids claiming
        medical certainty, and "appears to have fallen" is a description of what the
        geometry showed, where "has collapsed" would be a diagnosis this system is in
        no position to make.
        """
        reading = "pose" if self.used_pose else "body geometry"
        return (
            f"person track {self.track_id} appears to have fallen "
            f"({self.descent_rate:.1f} body heights/s descent by {reading}) and has "
            f"remained down for {self.settled_seconds:.0f}s"
        )

    def to_candidate(self) -> BehaviourCandidate:
        """This evidence in the currency the pipeline compares detectors in.

        Deliberately lossy, and this record is the reason the conversion exists rather
        than `observe_falls` returning a `BehaviourCandidate` directly: the descent
        rate and the settle duration are measurements worth keeping in their own
        types, where a consumer can read them as numbers. What crosses into the
        pipeline is the claim, in prose two very different readers — an operator and a
        vision-language model — both have to act on.
        """
        return BehaviourCandidate(
            kind=BehaviourKind.FALL,
            summary=self.summary(),
            track_ids=(self.track_id,),
            started_at=self.started_at,
        )


@dataclass(frozen=True, slots=True)
class _TrackState:
    """One tracked person's progress through the signature."""

    phase: _Phase
    since: float
    """When the current phase began."""

    centroid: tuple[float, float]
    height: float
    last_seen: float
    descent_rate: float = 0.0
    descent_started_at: float = 0.0
    settled_since: float | None = None
    used_pose: bool = False


@dataclass(frozen=True, slots=True)
class FallTracker:
    """The machine's whole memory, across every person on one camera.

    Frozen and returned anew by `observe_falls`, exactly as `GateState` is: a caller
    threads it frame to frame and the detector holds no mutable state of its own,
    which is what makes "what happens 7.2 seconds later" a one-line test.
    """

    tracks: dict[int, _TrackState] = field(default_factory=dict)

    def phase_of(self, track_id: int) -> str | None:
        """Which phase a track is in, as a plain string, or `None` if unknown.

        Exposed for telemetry and tests only. Returning `str` rather than the private
        `_Phase` keeps callers from branching on a member — `DOWN` in particular is
        not a finding, and a console rendering it as one would be reporting people who
        sat down.
        """
        state = self.tracks.get(track_id)
        return None if state is None else state.phase.value


def _posture(
    observation: BehaviourObservation,
    track_id: int,
    box_aspect: float | None,
    policy: FallPolicy,
) -> tuple[Posture, bool]:
    """This frame's posture, and whether pose supplied it.

    Pose first, geometry second, `UNKNOWN` rather than a guess — see the module
    docstring on why the fallback is per frame rather than per camera: a skeleton can
    answer on one frame and not the next as someone passes behind furniture, and
    freezing the choice for the whole episode would either discard good angles or
    trust stale ones.
    """
    pose = observation.pose_for(track_id)
    if pose is not None:
        angle = pose.torso_angle_degrees(min_keypoint_confidence=policy.min_keypoint_confidence)
        if angle is not None:
            if angle <= policy.upright_torso_degrees_max:
                return Posture.UPRIGHT, True
            if angle >= policy.horizontal_torso_degrees_min:
                return Posture.HORIZONTAL, True
            # Between the two bands: genuinely neither, and saying so is the point of
            # having a band. A leaning person is not upright and has not fallen.
            return Posture.UNKNOWN, True

    if box_aspect is None:
        return Posture.UNKNOWN, False
    if box_aspect <= policy.upright_aspect_max:
        return Posture.UPRIGHT, False
    if box_aspect >= policy.horizontal_aspect_min:
        return Posture.HORIZONTAL, False
    return Posture.UNKNOWN, False


def observe_falls(
    observation: BehaviourObservation,
    policy: FallPolicy,
    tracker: FallTracker,
) -> tuple[FallTracker, tuple[FallEvidence, ...]]:
    """Advance the machine by one frame.

    Returns the next tracker and any signatures completed *on this frame*. Pure: the
    caller threads the tracker, and `observation.timestamp` is the only clock.

    A candidate is raised **once** per episode — the track moves to `REPORTED` and
    stays there until the person stands up again, which is what stops a person lying
    still for two minutes from producing an alert on every frame for two minutes.
    Suppressing repeats here rather than downstream is deliberate: an alert engine can
    only deduplicate what it is told about, and telling it four thousand times would
    make the deduplication the thing under load.
    """
    now = observation.timestamp
    next_tracks: dict[int, _TrackState] = {}
    completed: list[FallEvidence] = []

    for track in observation.scene.tracks:
        if track.label != PERSON_LABEL:
            continue

        box = track.box
        height = box.y2 - box.y1
        width = box.x2 - box.x1
        centroid = (box.cx, box.cy)
        previous = tracker.tracks.get(track.track_id)

        if height <= 0.0:
            # A degenerate box — zero height, which a tracker emits for a frame or two
            # at the edge of frame or on a badly occluded body. It has no aspect ratio
            # and no body-height unit, so it can answer nothing.
            #
            # The state is carried forward **completely untouched**, including
            # `last_seen`, rather than updated with this frame's geometry. That is the
            # whole point: the next good frame then measures against the last good one
            # over the true elapsed interval. Updating `last_seen` here would leave the
            # centroid reference one frame older than the time reference and inflate
            # the next rate by exactly the gap; adopting this frame's centroid would
            # invent a descent out of a tracking artefact. An early version did the
            # latter and reported 13.8 body heights per second for a 2.8 body-height
            # fall — a number no body produces, on evidence meant to be measurements.
            #
            # A fall containing one bad frame is therefore still detected, over the
            # longer interval, with an honest rate. Skipping the episode instead would
            # trade a physically impossible number for a missed collapse, and this
            # system is built to prefer the first (see `domain/policy/notification.py`
            # on the same trade).
            if previous is not None:
                next_tracks[track.track_id] = previous
            continue

        aspect = width / height
        posture, used_pose = _posture(observation, track.track_id, aspect, policy)

        if previous is None:
            # A track first seen is assumed to be *whatever it looks like*, but the
            # `min_upright_seconds` clause means a track born upright still cannot
            # descend into a candidate until it has held that for long enough. A track
            # born horizontal enters `DOWN` with no descent behind it and can never
            # complete: `_settle` requires a started timer, so someone already on the
            # floor is watched and never reported. See the module docstring.
            next_tracks[track.track_id] = _TrackState(
                phase=_Phase.UPRIGHT if posture is Posture.UPRIGHT else _Phase.DOWN,
                since=now,
                centroid=centroid,
                height=height,
                last_seen=now,
                used_pose=used_pose,
            )
            continue

        elapsed = now - previous.last_seen
        # A non-advancing or regressing timeline means the frame is not usable for
        # rate arithmetic (a reconnect resets the source clock; see
        # `pipeline/runner.py`). Carry the state forward on the new geometry rather
        # than computing a rate against a timestamp from another timeline.
        if elapsed <= 0.0:
            next_tracks[track.track_id] = replace(
                previous, centroid=centroid, height=height, last_seen=now
            )
            continue

        # The body-height unit, taken from the *previous* frame's height: during a
        # fall the current frame's height is collapsing, so normalising by it would
        # divide by a shrinking number and inflate the rate exactly when the machine
        # is deciding whether the rate is high.
        # `previous.height` is always a real height: degenerate frames never reach
        # here, so this needs no fallback and cannot be zero.
        unit = previous.height
        descent_rate = (box.cy - previous.centroid[1]) / unit / elapsed
        drift = hypot(centroid[0] - previous.centroid[0], centroid[1] - previous.centroid[1])
        drift_in_heights = drift / unit

        state = replace(previous, centroid=centroid, height=height, last_seen=now)
        state, evidence = _advance(
            state=state,
            now=now,
            posture=posture,
            used_pose=used_pose,
            descent_rate=descent_rate,
            drift_in_heights=drift_in_heights,
            policy=policy,
            track_id=track.track_id,
        )
        next_tracks[track.track_id] = state
        if evidence is not None:
            completed.append(evidence)

    # Tracks absent from this frame are dropped rather than aged out. A person who
    # left the frame takes their episode with them: re-identifying them later is the
    # tracker's job, not this machine's, and keeping stale state would let a track id
    # reused for a different person inherit someone else's descent.
    return FallTracker(tracks=next_tracks), tuple(completed)


def _advance(
    *,
    state: _TrackState,
    now: float,
    posture: Posture,
    used_pose: bool,
    descent_rate: float,
    drift_in_heights: float,
    policy: FallPolicy,
    track_id: int,
) -> tuple[_TrackState, FallEvidence | None]:
    """One track, one frame. Split out so each phase reads as its own rule."""

    # Standing up ends any episode, from any phase, including `REPORTED`. That is what
    # re-arms the machine: a person who fell, was reported, got up and fell again is
    # two episodes and deserves two candidates.
    if posture is Posture.UPRIGHT:
        if state.phase is _Phase.UPRIGHT:
            return replace(state, used_pose=used_pose), None
        return (
            replace(
                state,
                phase=_Phase.UPRIGHT,
                since=now,
                descent_rate=0.0,
                descent_started_at=0.0,
                settled_since=None,
                used_pose=used_pose,
            ),
            None,
        )

    if state.phase is _Phase.UPRIGHT:
        if now - state.since < policy.min_upright_seconds:
            # Not yet believed to have been standing. Nothing that happens now can be
            # a transition from standing.
            return state, None
        if descent_rate >= policy.min_descent_rate:
            return (
                replace(
                    state,
                    phase=_Phase.DESCENDING,
                    since=now,
                    descent_rate=descent_rate,
                    descent_started_at=now,
                    used_pose=used_pose,
                ),
                None,
            )
        # Became horizontal without ever moving fast: lying down deliberately. Watched,
        # never reported — `DOWN` with no started timer cannot complete in `_settle`.
        if posture is Posture.HORIZONTAL:
            return (
                replace(state, phase=_Phase.DOWN, since=now, settled_since=None),
                None,
            )
        return state, None

    if state.phase is _Phase.DESCENDING:
        if now - state.descent_started_at > policy.descent_window_seconds:
            # The descent never resolved into a horizontal body. Give up on the
            # episode and watch from wherever the body now is.
            return (
                replace(
                    state,
                    phase=_Phase.DOWN if posture is Posture.HORIZONTAL else _Phase.UPRIGHT,
                    since=now,
                    descent_rate=0.0,
                    descent_started_at=0.0,
                    settled_since=None,
                ),
                None,
            )
        if posture is Posture.HORIZONTAL:
            return (
                replace(
                    state,
                    phase=_Phase.DOWN,
                    since=now,
                    # Keep the fastest rate seen across the descent, not the last one:
                    # by the time the body is horizontal it has stopped moving, and
                    # the last frame's rate would understate what happened.
                    descent_rate=max(state.descent_rate, descent_rate),
                    settled_since=now,
                    used_pose=used_pose,
                ),
                None,
            )
        return (
            replace(state, descent_rate=max(state.descent_rate, descent_rate)),
            None,
        )

    if state.phase is _Phase.DOWN:
        return _settle(
            state=state,
            now=now,
            drift_in_heights=drift_in_heights,
            policy=policy,
            track_id=track_id,
        )

    # REPORTED: already raised, and not upright yet. Hold until they stand.
    return state, None


def _settle(
    *,
    state: _TrackState,
    now: float,
    drift_in_heights: float,
    policy: FallPolicy,
    track_id: int,
) -> tuple[_TrackState, FallEvidence | None]:
    """The final clause: down, still, and long enough.

    Movement restarts the timer rather than ending the episode — someone struggling to
    get up is still down, and cancelling on the first twitch would lose precisely the
    case worth reporting.
    """
    if state.settled_since is None:
        # Reached `DOWN` without a descent behind it (already horizontal when first
        # seen, or lay down slowly). There is no transition to report, so the timer is
        # never started and this track can never complete. See the module docstring on
        # why that is deliberate rather than a gap.
        return state, None

    if drift_in_heights > policy.still_radius:
        return replace(state, settled_since=now), None

    settled_seconds = now - state.settled_since
    if settled_seconds < policy.settle_seconds:
        return state, None

    return (
        replace(state, phase=_Phase.REPORTED, since=now),
        FallEvidence(
            track_id=track_id,
            descent_rate=state.descent_rate,
            settled_seconds=settled_seconds,
            used_pose=state.used_pose,
            started_at=state.descent_started_at,
        ),
    )
