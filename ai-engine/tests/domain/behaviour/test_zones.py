"""Restricted areas and tripwires: the geometry, and the debounce around it."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.zones import (
    CrossingDirection,
    CrossingLine,
    RestrictedZone,
    ZonePolicy,
    ZoneTracker,
    observe_zones,
    point_in_polygon,
)
from sentinel_ai.domain.entities import Track
from tests.domain.behaviour.conftest import observation, person_box, track

# A square covering the left quarter of the frame, floor level.
LEFT_QUARTER = RestrictedZone(
    name="loading bay", polygon=((0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0))
)
# A vertical tripwire down the middle of the frame.
MIDLINE = CrossingLine(name="perimeter", start=(0.5, 0.0), end=(0.5, 1.0))


def run(
    script: list[tuple[list[Track], float]], policy: ZonePolicy
) -> tuple[ZoneTracker, list[BehaviourCandidate]]:
    tracker = ZoneTracker()
    raised: list[BehaviourCandidate] = []
    for tracks, timestamp in script:
        tracker, found = observe_zones(observation(tracks, timestamp), policy, tracker)
        raised.extend(found)
    return tracker, raised


def walk(
    xs: Sequence[float], *, ground: float = 900.0, track_id: int = 1
) -> list[tuple[list[Track], float]]:
    """One person walking through a sequence of x positions, one per half second."""
    return [
        ([track(track_id, "person", person_box(x, ground))], round(0.5 * index, 3))
        for index, x in enumerate(xs)
    ]


class TestPointInPolygon:
    def test_a_point_inside_is_inside(self) -> None:
        square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        assert point_in_polygon((5.0, 5.0), square)

    def test_a_point_outside_is_outside(self) -> None:
        square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
        assert not point_in_polygon((15.0, 5.0), square)

    def test_a_ray_through_a_vertex_is_counted_once(self) -> None:
        """The classic ray-casting bug. A naive `y1 <= y <= y2` edge test counts the
        shared vertex twice, making a point that is plainly inside come out even —
        and therefore outside."""
        triangle = ((0.0, 0.0), (10.0, 5.0), (0.0, 10.0))
        assert point_in_polygon((2.0, 5.0), triangle)

    def test_a_concave_polygon_is_handled(self) -> None:
        """An L-shape. The notch must be outside, which a convex-hull test would get
        wrong and an operator drawing round a pillar would immediately notice."""
        el = ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0))
        assert point_in_polygon((2.0, 8.0), el)
        assert not point_in_polygon((8.0, 8.0), el)


class TestZoneIntrusion:
    def test_walking_into_a_restricted_area_is_reported_once(self) -> None:
        policy = ZonePolicy(zones=(LEFT_QUARTER,))
        # 1920 wide; the zone is x < 480. Start outside, walk in, stay.
        _, raised = run(walk([900, 700, 500, 400, 350, 300, 280, 260]), policy)
        intrusions = [c for c in raised if c.kind is BehaviourKind.ZONE_INTRUSION]
        assert len(intrusions) == 1
        assert "loading bay" in intrusions[0].summary

    def test_standing_inside_does_not_re_report_every_frame(self) -> None:
        policy = ZonePolicy(zones=(LEFT_QUARTER,))
        script = walk([900, 400] + [300] * 60)
        _, raised = run(script, policy)
        assert len(raised) == 1

    def test_a_brief_clip_of_the_boundary_is_debounced(self) -> None:
        """One frame inside, then back out. Tracker jitter along a boundary would
        otherwise alarm on alternating frames — and because leaving re-arms the
        detector, that is an alarm every other frame rather than one."""
        policy = ZonePolicy(zones=(LEFT_QUARTER,), min_frames_inside=3)
        _, raised = run(walk([900, 470, 900, 470, 900]), policy)
        assert raised == []

    def test_leaving_and_returning_is_two_intrusions(self) -> None:
        policy = ZonePolicy(zones=(LEFT_QUARTER,), min_frames_inside=1)
        _, raised = run(walk([900, 300, 300, 900, 900, 300, 300]), policy)
        assert len(raised) == 2

    def test_a_person_outside_never_reports(self) -> None:
        policy = ZonePolicy(zones=(LEFT_QUARTER,))
        _, raised = run(walk([900, 1000, 1100, 1200] * 10), policy)
        assert raised == []

    def test_feet_decide_not_the_centroid(self) -> None:
        """A person whose feet are outside the zone but whose body leans over it must
        not be inside. Zones are drawn on floors."""
        policy = ZonePolicy(zones=(LEFT_QUARTER,), min_frames_inside=1)
        # Feet at x=600 (outside the x<480 zone); the centroid is the same x, but a
        # very tall box would put a naive centre-of-box test in a different place.
        _, raised = run(walk([600] * 10), policy)
        assert raised == []


class TestLineCrossing:
    def test_crossing_a_tripwire_is_reported(self) -> None:
        policy = ZonePolicy(lines=(MIDLINE,))
        # 1920 wide; the line is x = 960. Walk left to right across it.
        _, raised = run(walk([700, 800, 900, 1000, 1100]), policy)
        crossings = [c for c in raised if c.kind is BehaviourKind.LINE_CROSSING]
        assert len(crossings) == 1
        assert "perimeter" in crossings[0].summary

    def test_walking_alongside_a_line_does_not_cross_it(self) -> None:
        policy = ZonePolicy(lines=(MIDLINE,))
        _, raised = run(walk([700, 750, 800, 850, 900] * 5), policy)
        assert raised == []

    def test_a_directional_line_ignores_the_wrong_way(self) -> None:
        """A tripwire on an exit should not alarm on people coming in."""
        one_way = CrossingLine(
            name="exit", start=(0.5, 0.0), end=(0.5, 1.0), direction=CrossingDirection.A_TO_B
        )
        policy = ZonePolicy(lines=(one_way,))
        _, right_to_left = run(walk([1100, 1000, 900, 800]), policy)
        _, left_to_right = run(walk([800, 900, 1000, 1100]), policy)
        assert len(right_to_left) + len(left_to_right) == 1, "exactly one direction fires"

    def test_a_track_first_seen_past_the_line_has_not_crossed_it(self) -> None:
        """Somebody walking in from outside the frame. There is no previous position,
        so no crossing was observed — and claiming one would be inventing it."""
        policy = ZonePolicy(lines=(MIDLINE,))
        _, raised = run(walk([1100, 1200, 1300]), policy)
        assert raised == []


class TestDegenerateInput:
    def test_missing_frame_dimensions_place_no_zones(self) -> None:
        """Refusing is the safe failure: the alternative collapses every zone onto one
        pixel, which would either alarm constantly or never, with no way to tell."""
        policy = ZonePolicy(zones=(LEFT_QUARTER,), min_frames_inside=1)
        tracker = ZoneTracker()
        obs = observation([track(1, "person", person_box(100, 900))], 0.0, width=0, height=0)
        _, raised = observe_zones(obs, policy, tracker)
        assert raised == ()

    def test_an_empty_policy_raises_nothing(self) -> None:
        assert ZonePolicy().is_empty
        _, raised = run(walk([100, 200, 300]), ZonePolicy())
        assert raised == []


class TestPolicyValidation:
    def test_a_polygon_with_two_points_encloses_nothing(self) -> None:
        with pytest.raises(ValueError, match="at least 3 points"):
            RestrictedZone(name="bad", polygon=((0.0, 0.0), (1.0, 1.0)))

    def test_coordinates_outside_the_frame_are_rejected(self) -> None:
        """Pixels mistaken for fractions is the mistake this catches — a zone drawn at
        1920 rather than 1.0 would silently sit entirely off-screen."""
        with pytest.raises(ValueError, match="fractions of frame"):
            RestrictedZone(name="bad", polygon=((0.0, 0.0), (1920.0, 0.0), (1920.0, 1080.0)))

    def test_a_zero_length_line_cannot_be_crossed(self) -> None:
        with pytest.raises(ValueError, match="zero length"):
            CrossingLine(name="bad", start=(0.5, 0.5), end=(0.5, 0.5))

    def test_an_unnamed_zone_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            RestrictedZone(name="  ", polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)))

    def test_duplicate_names_on_one_camera_are_rejected(self) -> None:
        """Names key the per-episode state, so two zones called the same thing would
        share one episode and silently suppress each other."""
        other = RestrictedZone(name="loading bay", polygon=((0.5, 0.5), (0.9, 0.5), (0.9, 0.9)))
        with pytest.raises(ValueError, match="must be unique"):
            ZonePolicy(zones=(LEFT_QUARTER, other))

    def test_min_frames_inside_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="min_frames_inside"):
            ZonePolicy(zones=(LEFT_QUARTER,), min_frames_inside=0)
