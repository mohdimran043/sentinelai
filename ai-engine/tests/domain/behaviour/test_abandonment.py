"""Abandoned objects: the hand-off, and everything that is not one.

As with falls, the negative cases carry the weight. A detector that fires on "bag on
floor" fires all day in a waiting room, and an operator who dismisses it twice ignores
it the third time — which is when it matters.
"""

from __future__ import annotations

import pytest

from sentinel_ai.domain.behaviour.abandonment import (
    ABANDONABLE_LABELS,
    AbandonmentPolicy,
    AbandonmentTracker,
    observe_abandonment,
)
from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.entities import Track
from tests.domain.behaviour.conftest import bag_box, observation, person_box, ticks, track

POLICY = AbandonmentPolicy()


def run(
    script: list[tuple[list[Track], float]], policy: AbandonmentPolicy = POLICY
) -> tuple[AbandonmentTracker, list[BehaviourCandidate]]:
    tracker = AbandonmentTracker()
    raised: list[BehaviourCandidate] = []
    for tracks, timestamp in script:
        tracker, found = observe_abandonment(observation(tracks, timestamp), policy, tracker)
        raised.extend(found)
    return tracker, raised


def owner_then_leaves(
    *, leave_at: float = 2.0, until: float = 40.0
) -> list[tuple[list[Track], float]]:
    """A person stands with a bag, then walks away and stays away."""
    script: list[tuple[list[Track], float]] = []
    for t in ticks(0.0, leave_at, 0.5):
        script.append(
            ([track(1, "person", person_box(500, 800)), track(2, "backpack", bag_box(520, 800))], t)
        )
    for t in ticks(leave_at + 0.5, until, 0.5):
        # 1500 px away — many person-heights from the bag.
        script.append(
            (
                [
                    track(1, "person", person_box(1800, 800)),
                    track(2, "backpack", bag_box(520, 800)),
                ],
                t,
            )
        )
    return script


class TestTheHandOffCompletes:
    def test_a_bag_left_behind_raises_exactly_one_candidate(self) -> None:
        _, raised = run(owner_then_leaves())
        assert len(raised) == 1
        assert raised[0].kind is BehaviourKind.ABANDONED_OBJECT

    def test_nothing_is_raised_before_the_timer_elapses(self) -> None:
        _, raised = run(owner_then_leaves(leave_at=2.0, until=2.0 + POLICY.unattended_seconds - 2))
        assert raised == []

    def test_the_candidate_names_the_object_and_its_last_owner(self) -> None:
        """An investigator's first question is who left it, and the track id is what
        makes the clip searchable for them."""
        _, raised = run(owner_then_leaves())
        summary = raised[0].summary
        assert "backpack" in summary
        assert "person track 1" in summary

    def test_the_candidate_points_at_when_it_was_left_not_when_it_was_reported(self) -> None:
        """Thirty seconds apart. A clip centred on the report opens on a bag that has
        been sitting there for half a minute, missing the hand-off entirely."""
        _, raised = run(owner_then_leaves(leave_at=2.0))
        assert raised[0].started_at == pytest.approx(2.5, abs=0.6)

    def test_a_bag_collected_and_left_again_is_two_episodes(self) -> None:
        script = owner_then_leaves(leave_at=2.0, until=40.0)
        # Owner returns, picks it up, then leaves again.
        for t in ticks(40.5, 43.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(520, 800)),
                        track(2, "backpack", bag_box(520, 800)),
                    ],
                    t,
                )
            )
        for t in ticks(43.5, 85.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "backpack", bag_box(520, 800)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert len(raised) == 2


class TestItStaysSilent:
    def test_a_bag_that_was_never_attended_is_never_reported(self) -> None:
        """Furniture the detector mislabelled, or a bag that has been there since
        before the engine started. It has no owner to have been abandoned by."""
        script = [([track(2, "backpack", bag_box(520, 800))], t) for t in ticks(0.0, 120.0, 0.5)]
        _, raised = run(script)
        assert raised == []

    def test_a_bag_its_owner_stays_beside_is_not_abandoned(self) -> None:
        script = [
            ([track(1, "person", person_box(540, 800)), track(2, "backpack", bag_box(520, 800))], t)
            for t in ticks(0.0, 120.0, 0.5)
        ]
        _, raised = run(script)
        assert raised == []

    def test_a_bag_carried_away_is_not_abandoned(self) -> None:
        """Person and bag move together across the frame. The person is never far from
        it, so the timer never starts."""
        script: list[tuple[list[Track], float]] = []
        for index, t in enumerate(ticks(0.0, 60.0, 0.5)):
            x = 200 + index * 12
            script.append(
                (
                    [
                        track(1, "person", person_box(x, 800)),
                        track(2, "backpack", bag_box(x + 20, 800)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert raised == []

    def test_an_object_of_an_uninteresting_class_is_ignored(self) -> None:
        """`bottle` and `cup` are COCO classes too, and get left on tables constantly.
        A detector that alarmed on a forgotten coffee cup is one an operator mutes."""
        assert "bottle" not in ABANDONABLE_LABELS
        script = []
        for t in ticks(0.0, 2.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(500, 800)),
                        track(2, "bottle", bag_box(520, 800)),
                    ],
                    t,
                )
            )
        for t in ticks(2.5, 60.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "bottle", bag_box(520, 800)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert raised == []

    def test_a_tiny_box_is_not_reasoned_about(self) -> None:
        """A handful of pixels at the far end of a corridor is as likely a shadow, and
        its ground point moves by more than its own size on tracker noise alone."""
        script = []
        for t in ticks(0.0, 2.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(500, 800)),
                        track(2, "backpack", bag_box(520, 800, size=8)),
                    ],
                    t,
                )
            )
        for t in ticks(2.5, 60.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "backpack", bag_box(520, 800, size=8)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert raised == []

    def test_an_object_that_leaves_frame_takes_its_timer_with_it(self) -> None:
        script = owner_then_leaves(leave_at=2.0, until=10.0)
        # Bag disappears, then a *different* object reuses track id 2.
        for t in ticks(10.5, 12.0, 0.5):
            script.append(([track(1, "person", person_box(1800, 800))], t))
        for t in ticks(12.5, 30.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "backpack", bag_box(900, 800)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert raised == [], "a reused track id must not inherit the old object's timer"


class TestMovementRestartsRatherThanCancels:
    def test_a_nudged_bag_is_still_abandoned_just_later(self) -> None:
        """Kicked by a passer-by. It is still an abandoned bag, in a new place."""
        script = owner_then_leaves(leave_at=2.0, until=20.0)
        for t in ticks(20.5, 21.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "backpack", bag_box(700, 800)),
                    ],
                    t,
                )
            )
        for t in ticks(21.5, 60.0, 0.5):
            script.append(
                (
                    [
                        track(1, "person", person_box(1800, 800)),
                        track(2, "backpack", bag_box(700, 800)),
                    ],
                    t,
                )
            )
        _, raised = run(script)
        assert len(raised) == 1
        assert raised[0].started_at >= 20.0, "the timer restarted where it came to rest"


class TestPolicyValidation:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("attend_radius", 0.0),
            ("unattended_seconds", 0.0),
            ("still_radius", -1.0),
            ("min_object_diagonal_px", -1.0),
        ],
    )
    def test_out_of_range_thresholds_are_rejected(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            AbandonmentPolicy(**{field_name: value})
