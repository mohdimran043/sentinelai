"""When an unrecognised face becomes an unauthorised-person alert (spec §11).

§11 is a whole section of the specification and it says one thing: **do not alert
because a single frame produced a low-confidence match.** This module is that
requirement, as a pure state machine per tracked person.

    frame 1 -> unknown
    frame 2 -> unknown
    frame 3 -> unknown
    frame 4 -> unknown
        same tracked person
        + every face above the quality floor
        + every match below the threshold
        + observed for at least the minimum duration
            -> unauthorised person

Why each clause is there, and what removing it would cost
-----------------------------------------------------------
* **Same tracked person.** Four unknown faces belonging to four people walking past is
  not evidence about any of them. Without this the rule degenerates into "four
  unrecognised faces in a row", which a busy lobby satisfies constantly.
* **Every face above the quality floor.** A blurred or side-on face embeds near the
  middle of the space and is mildly similar to everybody, so it reads as a weak match.
  Counting those as evidence of *not* matching is counting noise as signal — which is
  why unusable faces are discarded rather than treated as unknown
  (`domain/identity.py` makes the same argument for quality being a precondition).
* **Every match below the threshold.** One good frame that recognised somebody ends
  the episode outright. A person who is recognised on frame 3 was there all along; the
  first two frames were bad angles, not a different person.
* **Minimum duration as well as a count.** Frame count alone is a frame-rate
  measurement: four frames is a sixth of a second at 25 fps and four seconds at
  1 fps. Both are needed, and they mean different things — the count is about evidence,
  the duration is about somebody actually being present.

The bias this encodes
----------------------
Every clause makes the detector *slower to alarm*. That is deliberate and it is the
opposite of the bias `domain/policy/notification.py` takes for welfare, where a missed
collapse is the worst outcome. Here the worst outcome is accusing somebody who belongs
there: a false unauthorised-person alert is an accusation about a specific identifiable
individual, and a site that produces them stops being trusted by the people it watches.
A missed one is a person who walked through unremarked, which is what happens today
without the feature at all.

Pure: no clock, no I/O. Time arrives on the observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from sentinel_ai.domain.identity import FaceObservation

__all__ = [
    "AuthorizationPolicy",
    "AuthorizationTracker",
    "UnauthorizedFinding",
    "observe_authorization",
]


class _Phase(StrEnum):
    WATCHING = "watching"
    """Accumulating unknown sightings. Not a finding."""

    RECOGNISED = "recognised"
    """This track was confidently matched to somebody authorised here, and is therefore
    **immune for the life of the track**.

    Stickiness matters more than it looks. A person walking down a corridor is face-on
    for perhaps one frame in three, so an authorised employee recognised on frame 3
    will fail to match on frames 4 through 20. Letting those re-accumulate means the
    system alerts on the people it just recognised, which is the single worst failure
    this feature can have — and it is the one an early version of this module had,
    caught by a test rather than by a site.

    The cost is a tracker that swaps two people's ids mid-episode could carry one
    person's recognition onto another. That is a real risk and a much rarer one than
    the false accusation, and §11's whole bias is toward being slow to accuse.
    """

    REPORTED = "reported"


@dataclass(frozen=True, slots=True)
class AuthorizationPolicy:
    """Every threshold §10 and §11 ask to be configurable. None of it is hard-coded."""

    match_threshold: float = 0.42
    """Cosine similarity at or above which a face is considered the same person.

    **The most consequential number in this system**, and the reason `domain/identity.py`
    refuses to expose a `matches()` boolean: raising it makes the system fail to
    recognise people who belong, lowering it makes it recognise the wrong person, and
    neither failure is visible without measurement against the actual enrolled faces
    and the actual cameras.

    The default is a conservative starting point for an ArcFace-family embedding, not a
    calibration. §32's obligation applies: measure it on the site before trusting it.
    """

    min_observations: int = 4
    """How many usable, unmatched sightings of one tracked person are needed.

    §11's worked example uses four. Together with `min_duration_seconds` rather than
    instead of it — see the module docstring on why a count alone is a frame-rate
    measurement.
    """

    min_duration_seconds: float = 2.0
    """How long the person must have been observed, independent of frame count."""

    min_box_pixels: int = 48
    """Smaller side of the face box below which the face is discarded unexamined (§10).

    Not "matched with low confidence" — discarded. A 20-pixel face cannot support any
    conclusion, and letting it contribute an unknown sighting is how a camera mounted
    too high accuses everybody who walks under it.
    """

    min_detector_confidence: float = 0.7
    min_frontality: float = 0.35
    """A face turned this far from the camera is discarded rather than judged."""

    cooldown_seconds: float = 300.0
    """How long after reporting a track before the same track could be reported again.

    In practice a track that has been reported stays `REPORTED` until it leaves frame,
    so this matters for the case a tracker re-acquires the same person under the same
    id after a gap. Five minutes, because the second alert about the same person in
    the same place adds nothing an operator did not already have.
    """

    def __post_init__(self) -> None:
        self._require(-1.0 <= self.match_threshold <= 1.0, "match_threshold must be in [-1, 1]")
        self._require(self.min_observations >= 1, "min_observations must be >= 1")
        self._require(self.min_duration_seconds >= 0.0, "min_duration_seconds must be >= 0")
        self._require(self.min_box_pixels >= 1, "min_box_pixels must be >= 1")
        self._require(
            0.0 <= self.min_detector_confidence <= 1.0,
            "min_detector_confidence must be in [0, 1]",
        )
        self._require(0.0 <= self.min_frontality <= 1.0, "min_frontality must be in [0, 1]")
        self._require(self.cooldown_seconds >= 0.0, "cooldown_seconds must be >= 0")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class UnauthorizedFinding:
    """A tracked person who has been observed long enough, well enough, and matched
    nobody who is allowed to be here.

    Carries the near-match rather than only the verdict, because §10 requires an
    operator to be shown the reference and the confidence. "Nobody matched" and
    "matched Employee B at 0.38 against a 0.45 threshold" are very different things to
    hand somebody, and the second frequently means the threshold is wrong rather than
    the person is.
    """

    track_id: int
    observations: int
    duration_seconds: float
    best_similarity: float
    best_person_name: str | None
    best_person_authorized_elsewhere: bool
    """True when the closest match is somebody enrolled but not permitted on this
    camera. A meaningfully different situation from an unknown face — it is a known
    person somewhere they should not be — and an operator reads it differently."""

    first_seen: float

    def summary(self) -> str:
        """One line for the escalation detail and the vision-language prompt.

        Hedged for `FallEvidence.summary`'s reason and more carefully: this sentence is
        about a specific identifiable person. "Did not match" is a statement about this
        system's enrolled faces, which is all it can honestly claim — not a statement
        about whether the person belongs.
        """
        if self.best_person_name is not None and self.best_person_authorized_elsewhere:
            return (
                f"person track {self.track_id} most closely resembles "
                f"{self.best_person_name} ({self.best_similarity:.2f}), who is enrolled "
                f"but not authorised on this camera; seen for "
                f"{self.duration_seconds:.0f}s across {self.observations} clear views"
            )
        near = (
            f" (closest enrolled face: {self.best_person_name} at {self.best_similarity:.2f})"
            if self.best_person_name is not None
            else ""
        )
        return (
            f"person track {self.track_id} did not match any face authorised on this "
            f"camera across {self.observations} clear views over "
            f"{self.duration_seconds:.0f}s{near}"
        )


@dataclass(frozen=True, slots=True)
class _TrackState:
    phase: _Phase
    first_seen: float
    last_seen: float
    observations: int
    best_similarity: float
    best_person_name: str | None
    best_person_authorized_elsewhere: bool
    reported_at: float | None = None


@dataclass(frozen=True, slots=True)
class AuthorizationTracker:
    """Per-track evidence for one camera. Threaded by the caller, as every machine in
    `domain/behaviour/` is."""

    tracks: dict[int, _TrackState] = field(default_factory=dict)

    def observations_for(self, track_id: int) -> int:
        state = self.tracks.get(track_id)
        return 0 if state is None else state.observations


def observe_authorization(
    faces: tuple[FaceObservation, ...],
    policy: AuthorizationPolicy,
    tracker: AuthorizationTracker,
    *,
    now: float,
) -> tuple[AuthorizationTracker, tuple[UnauthorizedFinding, ...]]:
    """Advance the machine by one frame's worth of faces.

    Takes the faces rather than a whole observation because, unlike the behaviour
    detectors, this stage runs only where a face pipeline ran — and on a frame where
    nobody's face was readable it has nothing to say rather than something to reset.

    **Tracks absent from `faces` keep their state**, which is the one place this
    differs from `domain/behaviour/`. A person walking down a corridor is face-on for
    perhaps one frame in three, and dropping the evidence every time they turn their
    head would mean `min_observations` is never reached on any real camera. State is
    dropped when the *track* is gone, which is the caller's business and happens on a
    discontinuity.
    """
    next_tracks = dict(tracker.tracks)
    findings: list[UnauthorizedFinding] = []

    for face in faces:
        if not face.quality.is_usable(
            min_box_pixels=policy.min_box_pixels,
            min_confidence=policy.min_detector_confidence,
            min_frontality=policy.min_frontality,
        ):
            # Discarded, not counted as unknown. See the module docstring: a face this
            # poor is mildly similar to everybody, and counting it as evidence of not
            # matching is counting noise as signal.
            continue

        best = face.best
        recognised_here = (
            best is not None and best.similarity >= policy.match_threshold and best.authorized_here
        )

        previous = next_tracks.get(face.track_id)

        if recognised_here:
            # One good frame that recognised somebody ends the episode — and keeps it
            # ended. A person recognised on frame 3 was there all along; the frames
            # before and after were bad angles, not a different person. See
            # `_Phase.RECOGNISED` for why this is sticky rather than a reset.
            next_tracks[face.track_id] = _TrackState(
                phase=_Phase.RECOGNISED,
                first_seen=previous.first_seen if previous is not None else now,
                last_seen=now,
                observations=(previous.observations + 1) if previous is not None else 1,
                best_similarity=best.similarity if best is not None else 0.0,
                best_person_name=best.display_name if best is not None else None,
                best_person_authorized_elsewhere=False,
            )
            continue

        if previous is not None and previous.phase is _Phase.RECOGNISED:
            # Already established as somebody who belongs here. Later frames where the
            # face is unreadable or ambiguous change nothing.
            next_tracks[face.track_id] = replace(previous, last_seen=now)
            continue

        if previous is not None and previous.phase is _Phase.REPORTED:
            if (
                previous.reported_at is not None
                and now - previous.reported_at < policy.cooldown_seconds
            ):
                next_tracks[face.track_id] = replace(previous, last_seen=now)
                continue
            # Past the cooldown; start a fresh episode for this track.
            previous = None

        similarity = best.similarity if best is not None else 0.0
        name = best.display_name if best is not None else None
        # "Enrolled but not permitted here" is only a claim worth making when the match
        # is actually confident. Below the threshold the closest face is just the
        # closest face, and naming them as an unauthorised *person* would be an
        # accusation the evidence does not support.
        elsewhere = (
            best is not None
            and best.similarity >= policy.match_threshold
            and not best.authorized_here
        )

        if previous is None:
            next_tracks[face.track_id] = _TrackState(
                phase=_Phase.WATCHING,
                first_seen=now,
                last_seen=now,
                observations=1,
                best_similarity=similarity,
                best_person_name=name,
                best_person_authorized_elsewhere=elsewhere,
            )
            continue

        # Keep the *strongest* near-match seen across the episode, not the latest: the
        # best look this camera ever got at them is the most informative thing to hand
        # an operator, and the last frame is often the worst.
        keep_new = similarity > previous.best_similarity
        state = replace(
            previous,
            last_seen=now,
            observations=previous.observations + 1,
            best_similarity=max(previous.best_similarity, similarity),
            best_person_name=name if keep_new else previous.best_person_name,
            best_person_authorized_elsewhere=(
                elsewhere if keep_new else previous.best_person_authorized_elsewhere
            ),
        )

        duration = state.last_seen - state.first_seen
        if (
            state.observations >= policy.min_observations
            and duration >= policy.min_duration_seconds
        ):
            next_tracks[face.track_id] = replace(state, phase=_Phase.REPORTED, reported_at=now)
            findings.append(
                UnauthorizedFinding(
                    track_id=face.track_id,
                    observations=state.observations,
                    duration_seconds=duration,
                    best_similarity=state.best_similarity,
                    best_person_name=state.best_person_name,
                    best_person_authorized_elsewhere=state.best_person_authorized_elsewhere,
                    first_seen=state.first_seen,
                )
            )
        else:
            next_tracks[face.track_id] = state

    return AuthorizationTracker(tracks=next_tracks), tuple(findings)
