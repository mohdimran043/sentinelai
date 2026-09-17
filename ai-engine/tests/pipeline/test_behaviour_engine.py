"""The engine that holds every behaviour machine's state for one camera.

What it owns is small and load-bearing: which detectors run, which candidate wins when
two finish on the same frame, and that a stream discontinuity wipes state without
wiping configuration.
"""

from __future__ import annotations

from sentinel_ai.domain.behaviour.abandonment import AbandonmentPolicy
from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.fall import FallPolicy
from sentinel_ai.domain.behaviour.tamper import TamperPolicy
from sentinel_ai.domain.behaviour.zones import RestrictedZone, ZonePolicy
from sentinel_ai.pipeline.behaviour import _PRIORITY, BehaviourEngine
from tests.domain.behaviour.conftest import (
    bag_box,
    flat_signature,
    observation,
    person_box,
    ticks,
    track,
)

ZONE = RestrictedZone(name="bay", polygon=((0.0, 0.0), (0.25, 0.0), (0.25, 1.0), (0.0, 1.0)))


class TestWhatRuns:
    def test_an_engine_with_no_policies_is_disabled(self) -> None:
        """The normal case, and what lets the frame loop skip the whole subsystem —
        including the pose forward pass — for a camera that enabled none of it."""
        engine = BehaviourEngine()
        assert not engine.enabled
        assert not engine.needs_pose

    def test_only_the_fall_machine_asks_for_pose(self) -> None:
        """Running a keypoint pass for a camera that enabled only abandonment would be
        a second forward pass per frame whose output nothing reads — the cost §13
        exists to avoid, reappearing one layer down."""
        assert BehaviourEngine(fall=FallPolicy()).needs_pose
        assert not BehaviourEngine(abandonment=AbandonmentPolicy()).needs_pose
        assert not BehaviourEngine(tamper=TamperPolicy()).needs_pose
        assert not BehaviourEngine(zones=ZonePolicy(zones=(ZONE,))).needs_pose

    def test_a_zone_policy_with_no_geometry_is_dropped(self) -> None:
        """A capability switched on with nothing drawn is a checkbox that does nothing.
        Dropping it here means `enabled` tells the truth and the frame loop does not
        pay to feed a detector with no boundaries to test against."""
        assert not BehaviourEngine(zones=ZonePolicy()).enabled

    def test_a_tamper_only_engine_is_enabled(self) -> None:
        """Tamper needs no model, so this is the configuration that lets camera-health
        monitoring run across a whole site for nothing."""
        assert BehaviourEngine(tamper=TamperPolicy()).enabled


class TestPriority:
    def test_every_kind_is_ranked(self) -> None:
        """An unranked kind sorts arbitrarily against the ranked ones, which would make
        "which alert did the operator get" depend on dict ordering. `observe` indexes
        rather than `.get`s so that adding a detector without ranking it is a build
        failure — this is the local version of that."""
        assert set(_PRIORITY) == set(BehaviourKind)

    def test_the_ranking_is_strict_so_ties_cannot_happen(self) -> None:
        assert len(set(_PRIORITY.values())) == len(_PRIORITY)

    def test_a_fall_outranks_everything_else(self) -> None:
        """Somebody may be hurt. Nothing else on the list is time-critical the same
        way."""
        assert _PRIORITY[BehaviourKind.FALL] == max(_PRIORITY.values())

    def test_tampering_outranks_what_it_makes_untrustworthy(self) -> None:
        """An intrusion alert from a camera that has just gone blind is an alert about
        a scene nobody can see."""
        assert _PRIORITY[BehaviourKind.CAMERA_TAMPER] > _PRIORITY[BehaviourKind.ZONE_INTRUSION]
        assert _PRIORITY[BehaviourKind.CAMERA_TAMPER] > _PRIORITY[BehaviourKind.ABANDONED_OBJECT]

    def test_an_abandoned_object_is_last(self) -> None:
        """It has by construction been sitting there for at least `unattended_seconds`,
        so it is the one finding that loses nothing by waiting a frame."""
        assert _PRIORITY[BehaviourKind.ABANDONED_OBJECT] == min(_PRIORITY.values())

    def test_two_candidates_on_one_frame_come_back_worst_first(self) -> None:
        """The property the frame loop relies on: it escalates `candidates[0]`."""
        engine = BehaviourEngine(tamper=TamperPolicy(), abandonment=AbandonmentPolicy())

        # Drive an abandonment to completion while the view is varied.
        person = track(1, "person", person_box(500, 800))
        bag = track(2, "backpack", bag_box(520, 800))
        far = track(1, "person", person_box(1800, 800))
        for t in ticks(0.0, 2.0, 0.5):
            engine.observe(observation([person, bag], t))
        for t in ticks(2.5, 40.0, 0.5):
            engine.observe(observation([far, bag], t))

        # Now blank the view long enough to complete a tamper episode too, and drive
        # a fresh abandonment alongside it.
        for t in ticks(40.5, 42.0, 0.5):
            engine.observe(observation([person, bag], t, signature=flat_signature()))
        found: list[BehaviourCandidate] = []
        for t in ticks(42.5, 120.0, 0.5):
            found.extend(engine.observe(observation([far, bag], t, signature=flat_signature())))

        kinds = [candidate.kind for candidate in found]
        assert BehaviourKind.CAMERA_TAMPER in kinds
        assert BehaviourKind.ABANDONED_OBJECT in kinds


class TestReset:
    def test_reset_drops_state_but_keeps_configuration(self) -> None:
        """A discontinuity invalidates every track id and every timestamp these
        machines hold, but not what the operator configured."""
        engine = BehaviourEngine(fall=FallPolicy(), tamper=TamperPolicy())
        for t in ticks(0.0, 40.0, 1.0):
            engine.observe(observation([], t))

        engine.reset()

        assert engine.enabled, "the policies survive"
        assert engine.needs_pose
        # The tamper machine had accumulated 40s of clear view; after a reset it has to
        # earn that again, so a blank frame immediately afterwards cannot fire.
        found: list[BehaviourCandidate] = []
        for t in ticks(41.0, 80.0, 1.0):
            found.extend(engine.observe(observation([], t, signature=flat_signature())))
        assert found == []
