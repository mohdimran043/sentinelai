from __future__ import annotations

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, EscalationReason, SceneState, Track
from sentinel_ai.domain.policy.triggers import (
    ALL_TRIGGERS,
    DwellAnchor,
    TriggerContext,
    advance_dwell_anchors,
    dwell_exceeded,
    new_salient_track,
    periodic_summary,
    scene_change,
    speed_anomaly,
    track_count_spike,
)

PROFILE = CameraProfile(camera_id="cam-1")


def track(
    track_id: int = 1,
    label: str = "person",
    age_frames: int = 10,
    speed_px_s: float = 0.0,
    cx: float = 100.0,
    cy: float = 100.0,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        box=BBox(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0),
        age_frames=age_frames,
        speed_px_s=speed_px_s,
    )


def scene(
    *,
    tracks: tuple[Track, ...] = (),
    timestamp: float = 0.0,
    signature: tuple[float, ...] = (1.0, 0.0),
) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=timestamp,
        detections=(),
        tracks=tracks,
        motion_energy=0.0,
        scene_signature=signature,
    )


def context(
    *,
    scene_state: SceneState,
    previous_signature: tuple[float, ...] | None = None,
    scene_delta_streak: int = 0,
    dwell_anchors: dict[int, DwellAnchor] | None = None,
    last_summary_at: float | None = None,
    speed_threshold_px_s: float | None = None,
) -> TriggerContext:
    return TriggerContext(
        scene=scene_state,
        profile=PROFILE,
        previous_signature=previous_signature,
        scene_delta_streak=scene_delta_streak,
        dwell_anchors=dwell_anchors or {},
        last_summary_at=last_summary_at,
        speed_threshold_px_s=speed_threshold_px_s or PROFILE.speed_fallback_px_s,
    )


class TestNewSalientTrack:
    def test_fires_when_a_salient_track_reaches_the_debounce_age(self) -> None:
        outcome = new_salient_track(context(scene_state=scene(tracks=(track(age_frames=8),))))
        assert outcome.fired is True
        assert outcome.reason is EscalationReason.NEW_SALIENT_TRACK

    def test_does_not_fire_below_the_debounce_age(self) -> None:
        """Flicker must not fire the gate — this is the whole point of debouncing."""
        assert (
            new_salient_track(context(scene_state=scene(tracks=(track(age_frames=7),)))).fired
            is False
        )

    def test_ignores_non_salient_classes(self) -> None:
        assert (
            new_salient_track(
                context(scene_state=scene(tracks=(track(label="bird", age_frames=100),)))
            ).fired
            is False
        )

    def test_does_not_fire_on_an_empty_scene(self) -> None:
        assert new_salient_track(context(scene_state=scene())).fired is False


class TestSceneChange:
    def test_fires_only_after_the_delta_is_sustained(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.0, 1.0)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=PROFILE.scene_delta_frames - 1,
        )
        assert scene_change(ctx).fired is True

    def test_does_not_fire_on_a_single_frame_spike(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.0, 1.0)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=0,
        )
        assert scene_change(ctx).fired is False

    def test_does_not_fire_below_the_delta_threshold(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.95, 0.05)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=100,
        )
        assert scene_change(ctx).fired is False

    def test_first_frame_never_fires(self) -> None:
        ctx = context(scene_state=scene(), previous_signature=None, scene_delta_streak=100)
        assert scene_change(ctx).fired is False

    def test_a_signature_of_a_different_length_never_fires(self) -> None:
        """A resolution change makes the histograms incomparable, not different."""
        ctx = context(
            scene_state=scene(signature=(0.5, 0.25, 0.25)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=100,
        )
        assert scene_change(ctx).fired is False


class TestDwellExceeded:
    def test_fires_once_the_anchor_is_older_than_the_dwell_window(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(cx=100.0, cy=100.0),), timestamp=31.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is True

    def test_does_not_fire_before_the_window_elapses(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(),), timestamp=29.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is False

    def test_a_track_that_moved_away_has_no_stale_anchor(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(cx=1000.0, cy=1000.0),), timestamp=31.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is False


class TestAdvanceDwellAnchors:
    def test_a_new_track_gets_an_anchor_at_the_current_time(self) -> None:
        anchors = advance_dwell_anchors(
            context(scene_state=scene(tracks=(track(),), timestamp=12.0))
        )
        assert anchors[1] == DwellAnchor(cx=100.0, cy=100.0, since=12.0)

    def test_a_stationary_track_keeps_its_original_anchor_time(self) -> None:
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(track(cx=110.0, cy=100.0),), timestamp=20.0),
                dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=5.0)},
            )
        )
        assert anchors[1].since == 5.0

    def test_a_track_that_left_the_radius_gets_a_fresh_anchor(self) -> None:
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(track(cx=200.0, cy=100.0),), timestamp=20.0),
                dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=5.0)},
            )
        )
        assert anchors[1] == DwellAnchor(cx=200.0, cy=100.0, since=20.0)

    def test_anchors_for_departed_tracks_are_dropped(self) -> None:
        """Otherwise the dict grows without bound on a busy camera."""
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(), timestamp=20.0),
                dwell_anchors={99: DwellAnchor(cx=1.0, cy=1.0, since=0.0)},
            )
        )
        assert anchors == {}


class TestSpeedAnomaly:
    def test_fires_above_the_threshold(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(speed_px_s=200.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is True

    def test_does_not_fire_at_or_below_the_threshold(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(speed_px_s=180.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is False

    def test_ignores_non_salient_classes(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(label="bird", speed_px_s=999.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is False


class TestTrackCountSpike:
    def test_fires_above_the_baseline(self) -> None:
        tracks = tuple(track(track_id=i) for i in range(7))
        assert track_count_spike(context(scene_state=scene(tracks=tracks))).fired is True

    def test_does_not_fire_at_the_baseline(self) -> None:
        tracks = tuple(track(track_id=i) for i in range(6))
        assert track_count_spike(context(scene_state=scene(tracks=tracks))).fired is False


class TestPeriodicSummary:
    def test_fires_on_the_very_first_frame_to_establish_a_baseline(self) -> None:
        ctx = context(scene_state=scene(timestamp=0.0), last_summary_at=None)
        assert periodic_summary(ctx).fired is True

    def test_fires_once_the_interval_has_elapsed(self) -> None:
        ctx = context(scene_state=scene(timestamp=45.0), last_summary_at=0.0)
        assert periodic_summary(ctx).fired is True

    def test_does_not_fire_before_the_interval(self) -> None:
        ctx = context(scene_state=scene(timestamp=44.9), last_summary_at=0.0)
        assert periodic_summary(ctx).fired is False


def test_triggers_are_ordered_most_urgent_first() -> None:
    """Priority order decides which reason is reported when several fire."""
    assert [fn.__name__ for fn in ALL_TRIGGERS] == [
        "speed_anomaly",
        "dwell_exceeded",
        "track_count_spike",
        "scene_change",
        "new_salient_track",
        "periodic_summary",
    ]
