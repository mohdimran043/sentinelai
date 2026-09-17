"""Restricted areas and virtual tripwires (spec §6).

Two detectors sharing one file because they share all their geometry and half their
failure modes: both ask where a tracked person is standing, and both are only as good
as the answer to that question.

Where a person "is"
--------------------
The **ground point** — bottom-centre of the bounding box — not the centroid. A person
standing at the edge of a restricted area has a centroid roughly half their height
*above* the ground, which in a camera looking down at any angle projects somewhere
quite different from where their feet are. Zones are drawn on floors, so feet are what
must be tested.

This is still a monocular approximation and it fails in a knowable way: a person
standing outside a zone but leaning over it, or occluded from the waist down so the box
bottom sits at their knees, lands in the wrong place. Perspective-correct answers need
a camera calibration this system does not have.

Coordinates are normalised, and that is not cosmetic
-----------------------------------------------------
A zone is stored as fractions of frame width and height, not pixels. An RTSP camera can
renegotiate resolution mid-stream (`pipeline/runner.py` treats it as a discontinuity),
and a zone drawn in pixels against 1920x1080 silently becomes a quarter of the intended
area when the stream drops to 960x540. Fractions survive it. They also mean an operator
who draws a zone on a console preview gets the same zone the engine tests against,
whatever size that preview was rendered at.

`frame_width`/`frame_height` of 0 means the caller could not supply dimensions, and
every zone test then answers "outside". Refusing to place a zone is the safe failure:
the alternative collapses every zone onto a single pixel, which would either alarm
constantly or never, and an operator would have no way to tell which.

Intrusion is a transition; so is a crossing
--------------------------------------------
Both are reported **once per episode**, and re-arm on leaving: a person who enters a
restricted area, is reported, leaves and comes back has trespassed twice. A person
standing inside for an hour is one alert, not one per frame — the same rule every
detector in this package follows, and for the reason `fall.py` states: an alert engine
can only deduplicate what it is told about.

Pure: no clock, no I/O, standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.observation import BehaviourObservation
from sentinel_ai.domain.entities import BBox

__all__ = [
    "CrossingDirection",
    "CrossingLine",
    "RestrictedZone",
    "ZonePolicy",
    "ZoneTracker",
    "observe_zones",
    "point_in_polygon",
]

Point = tuple[float, float]

PERSON_LABEL = "person"


class CrossingDirection(StrEnum):
    """Which way across a line is worth reporting.

    Named by the line's own endpoints rather than by compass or screen direction,
    because "northbound" and "left to right" both stop meaning anything the moment
    somebody re-points the camera. `A_TO_B` is the side the cross product calls
    negative crossing to the side it calls positive; which of those is "into the car
    park" is a question the operator answers by drawing the line the right way round,
    and the console shows them an arrow so they can see which they drew.
    """

    BOTH = "both"
    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"


@dataclass(frozen=True, slots=True)
class RestrictedZone:
    """A polygon on the floor, in normalised frame coordinates."""

    name: str
    polygon: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a zone needs a name; it is what an operator reads in the alert")
        if len(self.polygon) < 3:
            raise ValueError(
                f"zone {self.name!r} needs at least 3 points to enclose anything, "
                f"got {len(self.polygon)}"
            )
        for x, y in self.polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"zone {self.name!r} has a point outside the frame: ({x}, {y}). "
                    f"Coordinates are fractions of frame width and height, in [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class CrossingLine:
    """A virtual tripwire between two points, in normalised frame coordinates."""

    name: str
    start: Point
    end: Point
    direction: CrossingDirection = CrossingDirection.BOTH

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a line needs a name; it is what an operator reads in the alert")
        if self.start == self.end:
            raise ValueError(f"line {self.name!r} has zero length; it cannot be crossed")
        for x, y in (self.start, self.end):
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"line {self.name!r} has an endpoint outside the frame: ({x}, {y}). "
                    f"Coordinates are fractions of frame width and height, in [0, 1]."
                )


@dataclass(frozen=True, slots=True)
class ZonePolicy:
    """What this camera watches for, and how sure it has to be (spec §12)."""

    zones: tuple[RestrictedZone, ...] = ()
    lines: tuple[CrossingLine, ...] = ()

    min_frames_inside: int = 3
    """Consecutive frames a person's ground point must be inside a zone before it counts.

    Debounce, not certainty. A tracker box jitters by a few pixels frame to frame, and
    a person walking along a zone boundary would otherwise produce an entry and an exit
    on alternating frames — and because leaving re-arms the detector, that is an alarm
    every other frame rather than one.
    """

    def __post_init__(self) -> None:
        if self.min_frames_inside < 1:
            raise ValueError("min_frames_inside must be >= 1")
        names = [zone.name for zone in self.zones] + [line.name for line in self.lines]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Names are how an operator tells two alerts apart, and how this tracker
            # keys its per-episode state. Two zones called "entrance" would share one
            # episode and silently suppress each other's alerts.
            raise ValueError(f"zone and line names must be unique on a camera; got {duplicates}")

    @property
    def is_empty(self) -> bool:
        """Nothing configured. The composition root uses this to avoid attaching a
        detector that could never fire — a capability switched on with no geometry
        drawn is a checkbox that does nothing, and saying so early is better than
        letting an operator believe a boundary exists."""
        return not self.zones and not self.lines


def _ground(box: BBox) -> Point:
    return (box.cx, box.y2)


def _to_pixels(point: Point, width: int, height: int) -> Point:
    return (point[0] * width, point[1] * height)


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Ray casting, counting crossings of a ray cast in +x from `point`.

    Public because it is the part most worth testing directly, and because a console
    drawing the same zones needs to agree with the engine about what is inside.

    The `(y1 > y) != (y2 > y)` form is the standard half-open edge test: it counts an
    edge exactly once even when the ray passes through a vertex, which the naive
    `y1 <= y <= y2` form double-counts — turning a point that is plainly inside into a
    point with an even crossing count, and therefore outside.
    """
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing_x > x:
                inside = not inside
    return inside


def _side(point: Point, start: Point, end: Point) -> float:
    """Signed area of the triangle (start, end, point) — which side of the line it is on.

    Sign only; the magnitude is meaningless here because it scales with how far along
    the line the point projects.
    """
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Whether segment a1-a2 properly crosses segment b1-b2.

    Strict on both sides: a movement that merely *touches* the line and retreats has
    not crossed it, and a person walking exactly along a tripwire would otherwise trip
    it on every frame.
    """
    d1 = _side(a1, b1, b2)
    d2 = _side(a2, b1, b2)
    d3 = _side(b1, a1, a2)
    d4 = _side(b2, a1, a2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass(frozen=True, slots=True)
class _TrackZoneState:
    frames_inside: int = 0
    reported: bool = False


@dataclass(frozen=True, slots=True)
class ZoneTracker:
    """Per-camera state: where each track was, and which episodes are already reported."""

    previous_ground: dict[int, Point] = field(default_factory=dict)
    zone_state: dict[tuple[int, str], _TrackZoneState] = field(default_factory=dict)

    def is_inside(self, track_id: int, zone_name: str) -> bool:
        state = self.zone_state.get((track_id, zone_name))
        return state is not None and state.frames_inside > 0


def observe_zones(
    observation: BehaviourObservation,
    policy: ZonePolicy,
    tracker: ZoneTracker,
) -> tuple[ZoneTracker, tuple[BehaviourCandidate, ...]]:
    """Advance both machines by one frame. Pure; the caller threads the tracker."""
    width, height = observation.frame_width, observation.frame_height
    if width <= 0 or height <= 0 or policy.is_empty:
        # No dimensions means no zone can be placed — see the module docstring on why
        # refusing is the safe failure. State is carried unchanged rather than reset:
        # a single frame with missing dimensions must not re-arm every episode.
        return tracker, ()

    now = observation.timestamp
    people = [track for track in observation.scene.tracks if track.label == PERSON_LABEL]
    found: list[BehaviourCandidate] = []
    next_ground: dict[int, Point] = {}
    next_zone_state: dict[tuple[int, str], _TrackZoneState] = {}

    for person in people:
        ground = _ground(person.box)
        next_ground[person.track_id] = ground
        previous = tracker.previous_ground.get(person.track_id)

        for zone in policy.zones:
            polygon = tuple(_to_pixels(vertex, width, height) for vertex in zone.polygon)
            inside = point_in_polygon(ground, polygon)
            key = (person.track_id, zone.name)
            state = tracker.zone_state.get(key, _TrackZoneState())

            if not inside:
                # Left. The episode ends and the detector re-arms — a person who comes
                # back has trespassed again.
                continue

            frames_inside = state.frames_inside + 1
            should_report = not state.reported and frames_inside >= policy.min_frames_inside
            next_zone_state[key] = _TrackZoneState(
                frames_inside=frames_inside, reported=state.reported or should_report
            )
            if should_report:
                found.append(
                    BehaviourCandidate(
                        kind=BehaviourKind.ZONE_INTRUSION,
                        summary=(
                            f"person track {person.track_id} has entered the "
                            f"restricted area {zone.name!r} and stayed there for "
                            f"{frames_inside} frames"
                        ),
                        track_ids=(person.track_id,),
                        started_at=now,
                    )
                )

        if previous is None:
            # A track first seen has no previous position, so it cannot have crossed
            # anything yet. A person who appears already past a tripwire — walking in
            # from outside the frame — is not a crossing this camera observed.
            continue

        for line in policy.lines:
            start = _to_pixels(line.start, width, height)
            end = _to_pixels(line.end, width, height)
            if not _segments_intersect(previous, ground, start, end):
                continue
            if not _crossing_matches(previous, ground, start, end, line.direction):
                continue
            found.append(
                BehaviourCandidate(
                    kind=BehaviourKind.LINE_CROSSING,
                    summary=(
                        f"person track {person.track_id} has crossed the boundary {line.name!r}"
                    ),
                    track_ids=(person.track_id,),
                    started_at=now,
                )
            )

    # Tracks absent from this frame are dropped, for `fall.py`'s reason: a reused track
    # id must not inherit another person's episode or position.
    return ZoneTracker(previous_ground=next_ground, zone_state=next_zone_state), tuple(found)


def _crossing_matches(
    previous: Point,
    current: Point,
    start: Point,
    end: Point,
    direction: CrossingDirection,
) -> bool:
    """Whether a confirmed crossing went the way this line cares about."""
    if direction is CrossingDirection.BOTH:
        return True
    was_negative = _side(previous, start, end) < 0
    return was_negative if direction is CrossingDirection.A_TO_B else not was_negative
