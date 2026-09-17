"""An object left behind, as a transition rather than a state (spec §6).

The same discipline `fall.py` establishes, applied to luggage: a bag sitting on the
floor is not an abandoned bag. It is a bag. Waiting rooms, corridors and dayrooms are
full of objects that have been somewhere for hours, and a detector that fires on
"unattended object present" fires constantly on furniture the detector mislabelled and
on a suitcase someone is standing beside.

So what is detected is the **hand-off**:

    an object a person was with
        -> the person leaves
            -> the object stays put
                -> for long enough
                    -> candidate

An object that was never attended raises nothing, ever. It has no owner to have been
abandoned by, and the machine says so by staying silent. That cost is deliberate and it
is the same one falls pay: a bag dropped entirely outside the camera's view and only
tracked afterwards is invisible to this.

Distances are in person-heights, not pixels
--------------------------------------------
"Is anyone with this bag" is a question about the real world, and pixels do not answer
it — a person two metres from a bag is 400 px away near the camera and 40 px away at
the far end of a corridor. An adult is about 1.7 m tall and the detector can see how
many pixels tall each person is, so the person's **own bounding-box height** is the
unit. One person-height is roughly one large step. See `fall.py` on why every threshold
in this package is normalised by something the scene supplies.

Positions are compared at the **ground point** — the bottom-centre of a box — rather
than the centroid. Two objects at the same place on the floor have the same ground
point regardless of how tall they are, where their centroids differ by half the height
of a standing person. Using centroids would make every bag look distant from the person
carrying it.

Pure: no clock, no I/O, standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import hypot

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.observation import BehaviourObservation
from sentinel_ai.domain.entities import BBox, Track

__all__ = [
    "ABANDONABLE_LABELS",
    "PERSON_LABEL",
    "AbandonmentPolicy",
    "AbandonmentTracker",
    "observe_abandonment",
]

PERSON_LABEL = "person"

ABANDONABLE_LABELS: frozenset[str] = frozenset({"backpack", "handbag", "suitcase"})
"""The COCO classes that can be abandoned, and the only ones this reads.

All three are in `DEFAULT_SALIENT_CLASSES`, so a camera running the standard profile is
already tracking them. Deliberately narrow: `bottle` and `cup` are also COCO classes
and also get left on tables, and a detector that alerted on a forgotten coffee cup is a
detector an operator mutes within a day.
"""


def _ground(box: BBox) -> tuple[float, float]:
    """Where the thing is standing, in the image. See the module docstring."""
    return (box.cx, box.y2)


class _Phase(StrEnum):
    """Private, like `fall.py`'s: `UNATTENDED` is not a finding, it is the state that
    has not yet earned one, and exposing it would invite a console to render every
    momentarily-unwatched bag as an alert."""

    UNSEEN = "unseen"
    ATTENDED = "attended"
    UNATTENDED = "unattended"
    REPORTED = "reported"


@dataclass(frozen=True, slots=True)
class AbandonmentPolicy:
    """Thresholds, named and configurable (spec §12)."""

    attend_radius: float = 1.5
    """How close a person must be, in that person's own heights, to count as with the
    object. About two and a half metres for an adult — close enough to be carrying or
    guarding it, far enough that standing back from your own suitcase does not
    instantly orphan it."""

    unattended_seconds: float = 30.0
    """How long the object must stand alone before it is worth telling anyone.

    The dominant term in time-to-alert for this detector, and the one most worth tuning
    per site: a railway platform wants this short, a hospital dayroom where people
    routinely put a bag down and cross the room does not.
    """

    still_radius: float = 0.75
    """How far the object may drift, in **object diagonals**, and still count as left
    behind. Its own size rather than a person's, because this is a question about the
    object's own motion — has it been picked up — and the nearest person may be gone.

    Movement restarts the timer rather than cancelling: a bag nudged by a passer-by is
    still an abandoned bag, just a differently-placed one.
    """

    min_object_diagonal_px: float = 24.0
    """Below this the box is too small to reason about. A handful of pixels of "bag"
    at the far end of a corridor is as likely to be a shadow, and its ground point
    moves by more than its own size on tracker noise alone — which would make
    `still_radius` meaningless in exactly the cases it is least reliable."""

    def __post_init__(self) -> None:
        self._require(self.attend_radius > 0.0, "attend_radius must be > 0")
        self._require(self.unattended_seconds > 0.0, "unattended_seconds must be > 0")
        self._require(self.still_radius >= 0.0, "still_radius must be >= 0")
        self._require(self.min_object_diagonal_px >= 0.0, "min_object_diagonal_px must be >= 0")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _ObjectState:
    phase: _Phase
    ground: tuple[float, float]
    diagonal: float
    unattended_since: float | None = None
    owner_track_id: int | None = None
    """Whose it was when it was last attended. Carried so the candidate can name both
    the object and the person who left it — an investigator's first question is who,
    and the track id is what makes the clip searchable for them."""


@dataclass(frozen=True, slots=True)
class AbandonmentTracker:
    """Per-object state for one camera. Threaded by the caller, as `FallTracker` is."""

    objects: dict[int, _ObjectState] = field(default_factory=dict)

    def phase_of(self, track_id: int) -> str | None:
        state = self.objects.get(track_id)
        return None if state is None else state.phase.value


def _nearest_person_in_heights(object_box: BBox, people: list[Track]) -> tuple[float, int] | None:
    """Distance to the closest person in that person's own heights, and who they are.

    `None` when nobody is in frame — which is not the same as "everybody is far away",
    and both callers treat it as unattended. The distinction matters only for the
    owner id, which is why it is returned rather than inferred.
    """
    object_ground = _ground(object_box)
    best: tuple[float, int] | None = None
    for person in people:
        height = person.box.y2 - person.box.y1
        if height <= 0.0:
            # A degenerate person box has no unit to measure against. Skipping is
            # right: it cannot establish attendance, and it must not be able to
            # accidentally establish abandonment either.
            continue
        person_ground = _ground(person.box)
        distance = hypot(object_ground[0] - person_ground[0], object_ground[1] - person_ground[1])
        in_heights = distance / height
        if best is None or in_heights < best[0]:
            best = (in_heights, person.track_id)
    return best


def observe_abandonment(
    observation: BehaviourObservation,
    policy: AbandonmentPolicy,
    tracker: AbandonmentTracker,
) -> tuple[AbandonmentTracker, tuple[BehaviourCandidate, ...]]:
    """Advance the machine by one frame. Pure; the caller threads the tracker.

    Raised **once** per episode, like every detector in this package: the object moves
    to `REPORTED` and stays there until somebody collects it. A bag left for an hour is
    one alert, not one alert per frame for an hour.
    """
    now = observation.timestamp
    people = [track for track in observation.scene.tracks if track.label == PERSON_LABEL]
    next_objects: dict[int, _ObjectState] = {}
    found: list[BehaviourCandidate] = []

    for track in observation.scene.tracks:
        if track.label not in ABANDONABLE_LABELS:
            continue

        box = track.box
        diagonal = hypot(box.x2 - box.x1, box.y2 - box.y1)
        if diagonal < policy.min_object_diagonal_px:
            # Too small to reason about. Deliberately drops any state this track had:
            # an object that shrank below the floor is one the tracker is no longer
            # describing reliably, and resuming its timer later would be resuming a
            # measurement taken on a different thing.
            continue

        ground = _ground(box)
        previous = tracker.objects.get(track.track_id)
        nearest = _nearest_person_in_heights(box, people)
        attended = nearest is not None and nearest[0] <= policy.attend_radius

        if previous is None:
            # First sight. An object that arrives already attended can later be
            # abandoned; one that arrives alone starts `UNSEEN` and — because only
            # `ATTENDED` can transition into a running timer — never reports. See the
            # module docstring on why that silence is the honest answer.
            next_objects[track.track_id] = _ObjectState(
                phase=_Phase.ATTENDED if attended else _Phase.UNSEEN,
                ground=ground,
                diagonal=diagonal,
                owner_track_id=nearest[1] if nearest is not None and attended else None,
            )
            continue

        state, candidate = _advance(
            state=replace(previous, ground=ground, diagonal=diagonal),
            previous_ground=previous.ground,
            previous_diagonal=previous.diagonal,
            now=now,
            attended=attended,
            owner=nearest[1] if nearest is not None and attended else None,
            policy=policy,
            track_id=track.track_id,
            label=track.label,
        )
        next_objects[track.track_id] = state
        if candidate is not None:
            found.append(candidate)

    # Objects absent from this frame are dropped, for `fall.py`'s reason: a track id the
    # tracker reuses for a different object must not inherit this one's timer.
    return AbandonmentTracker(objects=next_objects), tuple(found)


def _advance(
    *,
    state: _ObjectState,
    previous_ground: tuple[float, float],
    previous_diagonal: float,
    now: float,
    attended: bool,
    owner: int | None,
    policy: AbandonmentPolicy,
    track_id: int,
    label: str,
) -> tuple[_ObjectState, BehaviourCandidate | None]:
    """One object, one frame."""

    if attended:
        # Somebody is with it. That ends any episode, including a reported one — which
        # is what re-arms the machine, so a bag collected and left again is two
        # abandonments and deserves two alerts.
        return (
            replace(
                state,
                phase=_Phase.ATTENDED,
                unattended_since=None,
                owner_track_id=owner if owner is not None else state.owner_track_id,
            ),
            None,
        )

    if state.phase is _Phase.UNSEEN:
        # Never had an owner. Watched, never reported.
        return state, None

    if state.phase is _Phase.ATTENDED:
        return replace(state, phase=_Phase.UNATTENDED, unattended_since=now), None

    if state.phase is _Phase.REPORTED:
        return state, None

    # UNATTENDED: the timer is running.
    unit = previous_diagonal if previous_diagonal > 0.0 else state.diagonal
    drift = hypot(state.ground[0] - previous_ground[0], state.ground[1] - previous_ground[1])
    if unit > 0.0 and drift / unit > policy.still_radius:
        # It moved. Not collected — nobody is near it — so the episode continues, but
        # from here: an object being pushed along a corridor has not been abandoned
        # *there* yet.
        return replace(state, unattended_since=now), None

    if state.unattended_since is None:
        return state, None

    alone_for = now - state.unattended_since
    if alone_for < policy.unattended_seconds:
        return state, None

    owner_phrase = (
        f" (last with person track {state.owner_track_id})"
        if state.owner_track_id is not None
        else ""
    )
    return (
        replace(state, phase=_Phase.REPORTED),
        BehaviourCandidate(
            kind=BehaviourKind.ABANDONED_OBJECT,
            summary=(
                f"a {label} (track {track_id}) appears to have been left unattended"
                f"{owner_phrase} and has stayed in place for {alone_for:.0f}s"
            ),
            track_ids=(track_id,),
            started_at=state.unattended_since,
        ),
    )
