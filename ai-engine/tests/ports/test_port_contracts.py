from __future__ import annotations

import inspect
from uuid import uuid4

import pytest

from sentinel_ai.domain.entities import (
    BBox,
    Detection,
    EscalationReason,
    Event,
    SceneState,
    ThreatScore,
)
from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameSource
from sentinel_ai.ports.model_runtime import LifecycleState, ModelRuntime
from sentinel_ai.ports.tracker import Tracker
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeTracker, FakeVisionLLM

ALL_PORTS = [
    ModelRuntime,
    ObjectDetector,
    Tracker,
    VisionLanguageModel,
    FrameSource,
    EventPublisher,
    ClipWriter,
]

BOX = BBox(0.0, 0.0, 10.0, 10.0)


def a_frame(frame_index: int = 0, timestamp: float = 0.0):
    return FakeSource.make_frame("cam-1", frame_index, timestamp)


def a_scene() -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=(),
        motion_energy=0.0,
        scene_signature=(1.0,),
    )


def a_vision_request() -> VisionRequest:
    return VisionRequest(
        keyframe=a_frame(),
        scene=a_scene(),
        history=(),
        camera_label="Front Door",
        reason_detail="test",
    )


@pytest.mark.parametrize("port", ALL_PORTS, ids=lambda p: p.__name__)
def test_every_port_is_abstract_and_cannot_be_instantiated(port: type) -> None:
    assert inspect.isabstract(port), f"{port.__name__} has no abstract methods"
    with pytest.raises(TypeError):
        port()  # type: ignore[call-arg,abstract]


def test_model_runtime_exposes_the_seven_spec_section_10_methods() -> None:
    required = {
        "initialize",
        "health",
        "predict",
        "warmup",
        "shutdown",
        "version",
        "capabilities",
    }
    # Equality, not a subset: an eighth abstract method added to the §10
    # interface must be a deliberate, visible change to this test.
    assert required == set(ModelRuntime.__abstractmethods__)


def test_lifecycle_states_cover_all_eight_from_spec_section_5() -> None:
    assert {s.value for s in LifecycleState} == {
        "loaded",
        "unloaded",
        "sleeping",
        "downloading",
        "updating",
        "offline",
        "healthy",
        "unhealthy",
    }


@pytest.mark.parametrize(
    ("fake", "port"),
    [
        (FakeDetector, ObjectDetector),
        (FakeTracker, Tracker),
        (FakeVisionLLM, VisionLanguageModel),
        (FakeSource, FrameSource),
        (FakePublisher, EventPublisher),
        (FakeClipWriter, ClipWriter),
    ],
    ids=lambda x: x.__name__,
)
def test_fakes_satisfy_their_ports(fake: type, port: type) -> None:
    assert issubclass(fake, port)


class TestFakeDetector:
    async def test_it_replays_scripted_detections_in_order(self) -> None:
        first = (Detection("person", 0.9, BOX),)
        second = (Detection("car", 0.8, BBox(5.0, 5.0, 25.0, 25.0)),)
        detector = FakeDetector(script=[first, second])

        assert await detector.detect(a_frame()) == first
        assert await detector.detect(a_frame()) == second

    async def test_it_repeats_the_final_entry_once_the_script_runs_out(self) -> None:
        only = (Detection("person", 0.9, BOX),)
        detector = FakeDetector(script=[only])

        assert await detector.detect(a_frame()) == only
        assert await detector.detect(a_frame()) == only

    async def test_it_counts_calls_so_tests_can_assert_reuse(self) -> None:
        detector = FakeDetector(script=[()])
        await detector.detect(a_frame())
        await detector.detect(a_frame())
        assert detector.call_count == 2

    def test_an_empty_script_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one entry"):
            FakeDetector(script=[])


class TestFakeVisionLLM:
    async def test_it_records_every_request_for_assertion(self) -> None:
        vlm = FakeVisionLLM(
            response=SceneDescription(
                description="Two people talking near the entrance.",
                threat_value=0.1,
                suggested_action="No action required.",
            )
        )
        request = a_vision_request()
        result = await vlm.describe(request)

        assert result.description == "Two people talking near the entrance."
        assert vlm.requests == [request]
        assert vlm.call_count == 1

    async def test_it_can_be_configured_to_raise_for_failure_path_tests(self) -> None:
        """Needed for spec §9: a VLM timeout must still produce an event."""
        vlm = FakeVisionLLM(error=TimeoutError("vlm timed out"))
        with pytest.raises(TimeoutError):
            await vlm.describe(a_vision_request())
        assert vlm.call_count == 1, "the request is recorded even when it fails"


class TestFakeTracker:
    def test_it_assigns_stable_ids_and_ages_tracks(self) -> None:
        tracker = FakeTracker()
        detection = Detection("person", 0.9, BOX)

        first = tracker.update((detection,), timestamp=0.0)
        second = tracker.update((detection,), timestamp=0.1)

        assert first[0].track_id == second[0].track_id
        assert second[0].age_frames == first[0].age_frames + 1

    def test_it_computes_speed_from_centroid_displacement(self) -> None:
        tracker = FakeTracker()
        tracker.update((Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),), timestamp=0.0)
        moved = tracker.update(
            (Detection("person", 0.9, BBox(20.0, 0.0, 30.0, 10.0)),), timestamp=1.0
        )
        assert moved[0].speed_px_s == pytest.approx(20.0)

    def test_a_distant_detection_starts_a_new_track(self) -> None:
        tracker = FakeTracker(match_radius_px=10.0)
        first = tracker.update((Detection("person", 0.9, BOX),), timestamp=0.0)
        far = tracker.update(
            (Detection("person", 0.9, BBox(500.0, 500.0, 510.0, 510.0)),), timestamp=0.1
        )
        assert far[0].track_id != first[0].track_id

    def test_reset_clears_all_state(self) -> None:
        """Re-sight the object after the reset — an empty update proves nothing.

        `tracker.update((), ...) == ()` holds for any implementation, including
        one whose `reset` does nothing at all, because an empty detection tuple
        always yields an empty result.
        """
        tracker = FakeTracker(match_radius_px=10.0)
        near = Detection("person", 0.9, BOX)
        far = Detection("person", 0.9, BBox(500.0, 500.0, 510.0, 510.0))

        tracker.update((near,), timestamp=0.0)
        before = tracker.update((near, far), timestamp=0.1)
        far_id = before[1].track_id
        assert far_id != 1, "the id counter has advanced past its starting value"

        tracker.reset()
        after = tracker.update((far,), timestamp=1.0)

        assert after[0].track_id != far_id, "the association must be forgotten"
        assert after[0].track_id == 1, "the id counter must be rewound"
        assert after[0].age_frames == 1, "a re-sighted object is new, not aged"


class TestFakeSource:
    async def test_it_yields_every_frame_in_order(self) -> None:
        source = FakeSource.constant("cam-1", count=3, fps=10.0)
        indices = [frame.frame_index async for frame in source]
        assert indices == [0, 1, 2]

    async def test_constant_spaces_timestamps_by_the_frame_interval(self) -> None:
        source = FakeSource.constant("cam-1", count=3, fps=10.0)
        stamps = [frame.timestamp async for frame in source]
        assert stamps == pytest.approx([0.0, 0.1, 0.2])

    async def test_close_is_recorded(self) -> None:
        source = FakeSource.constant("cam-1", count=1)
        await source.close()
        assert source.closed is True


class TestFakePublisher:
    async def test_it_collects_published_events(self) -> None:
        publisher = FakePublisher()
        event = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=1.0,
            reason=EscalationReason.SPEED_ANOMALY,
            threat=ThreatScore.from_value(0.7),
            description="A person is running.",
            suggested_action="Review the clip.",
        )
        await publisher.publish(event)
        assert publisher.events == [event]

    async def test_it_can_be_configured_to_fail(self) -> None:
        publisher = FakePublisher(error=ConnectionError("broker down"))
        with pytest.raises(ConnectionError):
            await publisher.publish(
                Event(
                    event_id=uuid4(),
                    camera_id="cam-1",
                    occurred_at=1.0,
                    reason=EscalationReason.PERIODIC_SUMMARY,
                    threat=ThreatScore.from_value(0.1),
                    description="",
                    suggested_action="",
                )
            )
        assert publisher.events == []


class TestFakeClipWriter:
    async def test_it_returns_a_uri_and_records_the_frame_count(self) -> None:
        writer = FakeClipWriter()
        event_id = uuid4()
        frames = [a_frame(i, i / 10.0) for i in range(5)]

        uri = await writer.write("cam-1", event_id, frames, fps=10.0)

        assert uri == f"s3://sentinel-clips/cam-1/{event_id}.mp4"
        assert writer.calls == [("cam-1", event_id, 5)]
