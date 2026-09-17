"""A camera going blind — and the many ways of being dark that are not that."""

from __future__ import annotations

import pytest

from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.tamper import (
    TamperPolicy,
    TamperState,
    concentration,
    observe_tamper,
)
from tests.domain.behaviour.conftest import flat_signature, observation, ticks

POLICY = TamperPolicy()
VARIED = tuple([1.0 / 16] * 16)


def run(
    script: list[tuple[tuple[float, ...], float]], policy: TamperPolicy = POLICY
) -> tuple[TamperState, list[BehaviourCandidate]]:
    state = TamperState()
    raised: list[BehaviourCandidate] = []
    for signature, timestamp in script:
        state, found = observe_tamper(
            observation([], timestamp, signature=signature), policy, state
        )
        raised.extend(found)
    return state, raised


def clear_then_covered(
    *, clear_until: float = 40.0, until: float = 60.0
) -> list[tuple[tuple[float, ...], float]]:
    script: list[tuple[tuple[float, ...], float]] = [
        (VARIED, t) for t in ticks(0.0, clear_until, 1.0)
    ]
    script.extend((flat_signature(), t) for t in ticks(clear_until + 1.0, until, 1.0))
    return script


class TestConcentration:
    def test_a_varied_scene_is_spread_across_bins(self) -> None:
        assert concentration(VARIED) == pytest.approx(1.0 / 16)

    def test_a_blank_view_concentrates(self) -> None:
        assert concentration(flat_signature(0.95)) == pytest.approx(0.95)

    def test_an_empty_signature_reads_as_not_blank(self) -> None:
        """A missing measurement must not be able to raise an alarm."""
        assert concentration(()) == 0.0


class TestItFires:
    def test_a_lens_covered_after_a_clear_period_is_reported_once(self) -> None:
        _, raised = run(clear_then_covered())
        assert len(raised) == 1
        assert raised[0].kind is BehaviourKind.CAMERA_TAMPER

    def test_the_candidate_has_no_track_ids(self) -> None:
        """There is nobody to attribute a covered lens to, and inventing a track id
        would be worse than an empty tuple."""
        _, raised = run(clear_then_covered())
        assert raised[0].track_ids == ()

    def test_the_summary_says_what_was_observed_not_why(self) -> None:
        """A bag over a lens, a lorry parked in front and a failed IR illuminator are
        the same picture from here. The claim is that the camera cannot see."""
        _, raised = run(clear_then_covered())
        summary = raised[0].summary.lower()
        assert "stopped seeing" in summary
        assert "still delivering frames" in summary
        assert "sabotage" not in summary

    def test_nothing_is_raised_before_the_obstruction_window_elapses(self) -> None:
        _, raised = run(
            clear_then_covered(clear_until=40.0, until=40.0 + POLICY.obstructed_seconds - 3)
        )
        assert raised == []

    def test_a_view_that_clears_and_goes_blank_again_is_two_episodes(self) -> None:
        script = clear_then_covered(clear_until=40.0, until=60.0)
        script.extend((VARIED, t) for t in ticks(61.0, 100.0, 1.0))
        script.extend((flat_signature(), t) for t in ticks(101.0, 120.0, 1.0))
        _, raised = run(script)
        assert len(raised) == 2


class TestItStaysSilent:
    def test_a_camera_that_was_blind_from_the_start_never_fires(self) -> None:
        """The case that would otherwise make an unlit camera alarm forever. There is
        no transition here to report."""
        script = [(flat_signature(), t) for t in ticks(0.0, 300.0, 1.0)]
        _, raised = run(script)
        assert raised == []

    def test_a_brief_blank_does_not_fire(self) -> None:
        """A lorry pulling across the view, headlights sweeping the lens, an
        auto-exposure hunt. All blank a camera for a second or two."""
        script = [(VARIED, t) for t in ticks(0.0, 40.0, 1.0)]
        script.extend((flat_signature(), t) for t in ticks(41.0, 45.0, 1.0))
        script.extend((VARIED, t) for t in ticks(46.0, 80.0, 1.0))
        _, raised = run(script)
        assert raised == []

    def test_a_collapse_too_soon_after_startup_does_not_fire(self) -> None:
        """`min_clear_seconds` has not elapsed, so the view is not yet believed to
        have been seeing anything."""
        script = [(VARIED, t) for t in ticks(0.0, 5.0, 1.0)]
        script.extend((flat_signature(), t) for t in ticks(6.0, 60.0, 1.0))
        _, raised = run(script)
        assert raised == []

    def test_a_merely_dim_scene_is_not_blank(self) -> None:
        """Dusk concentrates the histogram without collapsing it. 70% in one bin is a
        dark room, not a covered lens."""
        script = [(VARIED, t) for t in ticks(0.0, 40.0, 1.0)]
        script.extend((flat_signature(0.70), t) for t in ticks(41.0, 120.0, 1.0))
        _, raised = run(script)
        assert raised == []


class TestPolicyValidation:
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("concentration_threshold", 0.0),
            ("concentration_threshold", 1.5),
            ("obstructed_seconds", 0.0),
            ("min_clear_seconds", -1.0),
        ],
    )
    def test_out_of_range_thresholds_are_rejected(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError):
            TamperPolicy(**{field_name: value})
