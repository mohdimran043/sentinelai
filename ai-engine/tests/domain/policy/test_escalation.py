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


def standing_track(track_id: int = 1, cx: float = 100.0, cy: float = 100.0) -> Track:
    """Old enough to be salient, slow enough not to trip speed_anomaly."""
    return Track(
        track_id=track_id,
        label="person",
        box=BBox(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0),
        age_frames=20,
        speed_px_s=0.0,
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


class TestEveryReasonIsReachableThroughTheGate:
    """The composition, not the predicates.

    `test_triggers.py` proves each predicate in isolation. What only a
    gate-level test can prove is that `decide` threads `dwell_anchors` and
    `scene_delta_streak` out of `GateState` and into `TriggerContext` correctly
    across frames — so every reason is actually reachable end to end.

    Every frame here is spaced clear of the post-call cooldown, and every scene
    that is expected to escalate carries a signature dedup cannot match against
    the last escalated one. Otherwise these tests would pass for the wrong
    reason: a suppressed decision still reports its trigger.
    """

    CHANGED_SIGNATURE = (0.0, 1.0)

    def test_an_aged_salient_track_escalates_as_new_salient_track(self) -> None:
        outcome = decide(
            scene(
                timestamp=10.0,
                tracks=(standing_track(),),
                signature=self.CHANGED_SIGNATURE,
            ),
            PROFILE,
            settled_state(),
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.NEW_SALIENT_TRACK

    def test_a_crowd_escalates_as_track_count_spike(self) -> None:
        crowd = tuple(
            standing_track(track_id=index, cx=200.0 * index, cy=200.0 * index)
            for index in range(1, PROFILE.track_count_baseline + 2)
        )
        assert len(crowd) > PROFILE.track_count_baseline

        outcome = decide(
            scene(timestamp=10.0, tracks=crowd, signature=self.CHANGED_SIGNATURE),
            PROFILE,
            settled_state(),
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.TRACK_COUNT_SPIKE

    def test_sustained_signature_change_escalates_as_scene_change(self) -> None:
        """The streak must survive the round trip through GateState.

        The gate fires on the frame whose incoming streak reaches
        `scene_delta_frames - 1`, i.e. the fifth consecutive changing frame.
        Signatures alternate, so that frame must be one whose signature differs
        from the last escalated one — otherwise dedup suppresses it.
        """
        state = settled_state()
        reasons = []
        for index in range(5):
            outcome = decide(
                scene(timestamp=2.0 * (index + 1), tracks=(), signature=alternating(index + 1)),
                PROFILE,
                state,
            )
            state = outcome.state
            reasons.append(outcome.decision.reason)

        assert reasons[:4] == [None, None, None, None], "four frames is not yet sustained"
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.SCENE_CHANGE
        assert state.scene_delta_streak == 5

    def test_a_stationary_track_escalates_as_dwell_exceeded(self) -> None:
        """Anchors come from the previous frame while the current one advances them.

        During the wait the signature is held constant and equal to the one
        already escalated, so dedup suppresses the NEW_SALIENT_TRACK that fires
        every frame and the budget survives intact. The final frame changes the
        signature so the dwell escalation is actually observable.
        """
        state = settled_state()
        parked = standing_track()
        for second in range(1, int(PROFILE.dwell_seconds) + 1):
            outcome = decide(
                scene(timestamp=float(second), tracks=(parked,), signature=QUIET_SIGNATURE),
                PROFILE,
                state,
            )
            assert outcome.decision.reason is not EscalationReason.DWELL_EXCEEDED, (
                f"dwell must not fire {second}s in — the window is "
                f"{PROFILE.dwell_seconds}s from the anchor"
            )
            state = outcome.state

        # The anchor was laid on the first frame of the walk (t=1.0), so the
        # window closes at t = 1.0 + dwell_seconds.
        outcome = decide(
            scene(
                timestamp=1.0 + PROFILE.dwell_seconds,
                tracks=(parked,),
                signature=self.CHANGED_SIGNATURE,
            ),
            PROFILE,
            state,
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.DWELL_EXCEEDED


class TestSignatureLengthChange:
    """An RTSP reconnect can renegotiate resolution mid-stream, and any change to
    the signature extractor's bin count does the same. The histograms are then
    incomparable — but that must degrade the gate, not kill the camera.
    """

    RESCALED_SIGNATURE = (0.5, 0.25, 0.25)

    def test_a_longer_signature_does_not_crash_the_gate(self) -> None:
        state = settled_state()
        outcome = decide(scene(timestamp=1.0, signature=self.RESCALED_SIGNATURE), PROFILE, state)
        assert outcome.state.previous_signature == self.RESCALED_SIGNATURE

    def test_an_incomparable_signature_resets_the_scene_delta_streak(self) -> None:
        """A changed bin count makes a sustained-change streak meaningless."""
        state = settled_state()
        changed = decide(scene(timestamp=1.0, signature=(0.0, 1.0)), PROFILE, state)
        assert changed.state.scene_delta_streak == 1, "streak established"

        rescaled = decide(
            scene(timestamp=2.0, signature=self.RESCALED_SIGNATURE), PROFILE, changed.state
        )
        assert rescaled.state.scene_delta_streak == 0

    def test_an_incomparable_signature_is_not_a_duplicate(self) -> None:
        """Incomparable is not 'unchanged' — dedup must not swallow the escalation."""
        outcome = decide(
            scene(
                timestamp=20.0,
                tracks=(running_track(),),
                signature=self.RESCALED_SIGNATURE,
            ),
            PROFILE,
            settled_state(),
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.suppressed_by is None


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
