from __future__ import annotations

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, EscalationReason, SceneState, Track
from sentinel_ai.domain.policy.escalation import GateState, decide, force

PROFILE = CameraProfile(camera_id="cam-1")
QUIET_SIGNATURE = (1.0, 0.0)


def running_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        label="person",
        box=BBox(90.0, 90.0, 110.0, 110.0),
        age_frames=20,
        speed_px_s=500.0,
    )


def scene(
    *,
    timestamp: float,
    tracks: tuple[Track, ...] = (),
    signature: tuple[float, ...] = QUIET_SIGNATURE,
    frame_index: int = 0,
) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=timestamp,
        detections=(),
        tracks=tracks,
        motion_energy=0.0,
        scene_signature=signature,
    )


def initial_state(now: float = 0.0) -> GateState:
    return GateState.initial(PROFILE, now=now)


def settled_state() -> GateState:
    """Past the initial summary, with a known signature and a full bucket."""
    return decide(scene(timestamp=0.0), PROFILE, initial_state()).state


def alternating(tick: int) -> tuple[float, ...]:
    """A signature that changes every frame.

    Necessary for any test targeting the rate budget: with a constant signature
    duplicate-suppression fires first and the budget is never reached, so the
    test would pass for the wrong reason.
    """
    return (1.0, 0.0) if tick % 2 == 0 else (0.0, 1.0)


class TestFirstFrame:
    def test_the_first_frame_escalates_to_establish_a_baseline(self) -> None:
        outcome = decide(scene(timestamp=0.0), PROFILE, initial_state())
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.PERIODIC_SUMMARY

    def test_the_first_frame_consumes_a_token(self) -> None:
        outcome = decide(scene(timestamp=0.0), PROFILE, initial_state())
        assert outcome.state.bucket.available == 1.0


class TestQuietScene:
    def test_a_quiet_scene_does_not_escalate(self) -> None:
        outcome = decide(scene(timestamp=1.0), PROFILE, settled_state())
        assert outcome.decision.should_escalate is False
        assert outcome.decision.reason is None

    def test_a_quiet_scene_spends_no_tokens(self) -> None:
        state = settled_state()
        outcome = decide(scene(timestamp=1.0), PROFILE, state)
        assert outcome.state.bucket.available == state.bucket.available


class TestPriority:
    def test_speed_anomaly_outranks_periodic_summary(self) -> None:
        state = settled_state()
        outcome = decide(
            scene(timestamp=100.0, tracks=(running_track(),), signature=(0.0, 1.0)),
            PROFILE,
            state,
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.SPEED_ANOMALY


class TestBudgetGovernor:
    def test_the_bucket_caps_escalations_under_sustained_pressure(self) -> None:
        """A chaotic scene must not be able to saturate the GPU (spec §4.1).

        60 s at 10 fps where every single frame trips speed_anomaly. Signatures
        alternate so dedup never suppresses — the budget is the only limiter.
        Expect 7 VLM calls, not 600.
        """
        state = initial_state()
        escalations = 0
        for tick in range(600):
            outcome = decide(
                scene(
                    timestamp=tick * 0.1,
                    tracks=(running_track(),),
                    signature=alternating(tick),
                    frame_index=tick,
                ),
                PROFILE,
                state,
            )
            state = outcome.state
            escalations += outcome.decision.should_escalate
        assert escalations == 7

    def test_a_denied_decision_records_the_budget_as_the_suppressor(self) -> None:
        state = initial_state()
        for tick in range(3):
            outcome = decide(
                scene(
                    timestamp=float(tick),
                    tracks=(running_track(),),
                    signature=alternating(tick),
                ),
                PROFILE,
                state,
            )
            state = outcome.state
        assert outcome.decision.should_escalate is False
        assert outcome.decision.suppressed_by == "rate_budget"
        assert outcome.decision.reason is EscalationReason.SPEED_ANOMALY, (
            "the reason is still reported so telemetry can show what was suppressed"
        )


class TestDeduplication:
    # settled_state() has already escalated once on QUIET_SIGNATURE, so the first
    # escalation in these tests must use a different signature — otherwise it is
    # itself a duplicate and the test would prove nothing.
    OTHER_SIGNATURE = (0.0, 1.0)

    def test_an_unchanged_scene_is_suppressed_even_with_budget_available(self) -> None:
        first = decide(
            scene(timestamp=20.0, tracks=(running_track(),), signature=self.OTHER_SIGNATURE),
            PROFILE,
            settled_state(),
        )
        assert first.decision.should_escalate is True
        assert first.state.bucket.available >= 1.0, "budget is not the limiter here"

        second = decide(
            scene(timestamp=40.0, tracks=(running_track(),), signature=self.OTHER_SIGNATURE),
            PROFILE,
            first.state,
        )
        assert second.decision.should_escalate is False
        assert second.decision.suppressed_by == "duplicate_scene"

    def test_a_materially_changed_scene_is_not_deduplicated(self) -> None:
        first = decide(
            scene(timestamp=20.0, tracks=(running_track(),), signature=self.OTHER_SIGNATURE),
            PROFILE,
            settled_state(),
        )
        assert first.decision.should_escalate is True

        second = decide(
            scene(timestamp=40.0, tracks=(running_track(),), signature=QUIET_SIGNATURE),
            PROFILE,
            first.state,
        )
        assert second.decision.should_escalate is True

    def test_dedup_does_not_refund_a_suppressed_token(self) -> None:
        state = settled_state()
        first = decide(scene(timestamp=20.0, tracks=(running_track(),)), PROFILE, state)
        second = decide(scene(timestamp=40.0, tracks=(running_track(),)), PROFILE, first.state)
        assert second.state.bucket.available == first.state.bucket.refilled(40.0).available


class TestVlmDisabled:
    def test_a_camera_with_vlm_disabled_never_escalates(self) -> None:
        profile = CameraProfile(camera_id="cam-1", vlm_enabled=False)
        outcome = decide(
            scene(timestamp=0.0, tracks=(running_track(),)),
            profile,
            GateState.initial(profile, now=0.0),
        )
        assert outcome.decision.should_escalate is False
        assert outcome.decision.suppressed_by == "vlm_disabled"


class TestSceneDeltaStreak:
    def test_the_streak_accumulates_across_frames_of_sustained_change(self) -> None:
        state = settled_state()
        signatures = [(0.0, 1.0), (1.0, 0.0)] * 3
        for index, signature in enumerate(signatures):
            outcome = decide(scene(timestamp=1.0 + index, signature=signature), PROFILE, state)
            state = outcome.state
        assert state.scene_delta_streak >= PROFILE.scene_delta_frames

    def test_the_streak_resets_when_the_scene_settles(self) -> None:
        state = settled_state()
        changed = decide(scene(timestamp=1.0, signature=(0.0, 1.0)), PROFILE, state)
        assert changed.state.scene_delta_streak == 1
        settled = decide(scene(timestamp=2.0, signature=(0.0, 1.0)), PROFILE, changed.state)
        assert settled.state.scene_delta_streak == 0


class TestForce:
    def test_a_user_request_escalates_regardless_of_budget(self) -> None:
        state = initial_state()
        for tick in range(3):
            state = decide(
                scene(timestamp=float(tick), tracks=(running_track(),)), PROFILE, state
            ).state
        outcome = force(EscalationReason.USER_REQUESTED, state, now=3.0)
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.USER_REQUESTED

    def test_a_forced_call_still_records_the_escalation_time(self) -> None:
        outcome = force(EscalationReason.USER_REQUESTED, initial_state(), now=7.0)
        assert outcome.state.last_escalation_at == 7.0
