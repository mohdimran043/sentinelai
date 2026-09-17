"""The measurement code, measured.

A benchmark nobody has tested is a source of confident wrong numbers, and wrong
performance numbers are worse than none: they get quoted in documentation and then
defended. These pin the two things most able to be quietly wrong — the percentile
definition, and which calls get counted at all.
"""

from __future__ import annotations

import pytest

from sentinel_ai.benchmark.timing import (
    Recorder,
    TimedDetector,
    TimedPoseEstimator,
    TimedVisionLanguageModel,
    percentile,
    summarise,
)
from sentinel_ai.domain.behaviour.observation import PersonPose
from sentinel_ai.domain.entities import BBox, Detection, SceneState, Track
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.pose import PoseEstimator
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest


def a_frame() -> FrameData:
    return FrameData(
        camera_id="cam-1", frame_index=0, timestamp=0.0, width=4, height=4, pixels=object()
    )


def a_track() -> Track:
    return Track(
        track_id=1,
        label="person",
        box=BBox(0.0, 0.0, 10.0, 20.0),
        age_frames=3,
        speed_px_s=0.0,
    )


class TestPercentile:
    def test_nearest_rank_returns_a_value_that_actually_occurred(self) -> None:
        """The whole reason it is not interpolated: an interpolated p95 is a number no
        call took, and a latency budget compared against it is being compared against
        fiction."""
        samples = [1.0, 2.0, 3.0, 4.0, 100.0]
        assert percentile(samples, 0.95) in samples

    def test_p50_of_a_known_distribution(self) -> None:
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0

    def test_p95_picks_the_tail_not_the_body(self) -> None:
        samples = [1.0] * 95 + [50.0] * 5
        assert percentile(samples, 0.95) == 1.0
        assert percentile(samples, 0.96) == 50.0

    def test_unsorted_input_is_handled(self) -> None:
        assert percentile([9.0, 1.0, 5.0], 0.5) == 5.0

    def test_empty_samples_are_zero_rather_than_an_error(self) -> None:
        """A stage that never ran is a real benchmark outcome — no camera enabled it,
        or nothing escalated — and must print an empty row instead of taking the run
        down at the reporting step, after all the expensive work is done."""
        assert percentile([], 0.5) == 0.0

    def test_a_single_sample_is_every_percentile(self) -> None:
        assert percentile([7.0], 0.5) == 7.0
        assert percentile([7.0], 0.95) == 7.0


class TestSummarise:
    def test_no_mean_is_reported(self) -> None:
        """Deliberate. A mean hides the tail, and the tail is the only part that
        decides whether a camera is dropping frames."""
        summary = summarise("detector", [1.0, 2.0, 3.0])
        assert not hasattr(summary, "mean_ms")
        assert not hasattr(summary, "average_ms")

    def test_count_is_carried_so_a_thin_sample_is_visible(self) -> None:
        """A p95 over four calls is not a p95. A reader comparing the VLM's handful of
        describes against the detector's thousands needs to see which is which."""
        assert summarise("vlm", [1.0, 2.0]).count == 2

    def test_max_is_the_real_maximum(self) -> None:
        assert summarise("detector", [1.0, 400.0, 2.0]).max_ms == 400.0

    def test_an_empty_stage_summarises_as_zeros(self) -> None:
        summary = summarise("pose", [])
        assert summary.count == 0
        assert summary.p50_ms == 0.0


class TestRecorder:
    def test_seconds_are_recorded_as_milliseconds(self) -> None:
        recorder = Recorder()
        recorder.record("detector", 0.005)
        assert recorder.samples["detector"] == [pytest.approx(5.0)]

    def test_stages_appear_in_the_order_they_first_ran(self) -> None:
        recorder = Recorder()
        recorder.record("detector", 0.001)
        recorder.record("vlm", 2.0)
        recorder.record("detector", 0.001)
        assert [s.name for s in recorder.summaries()] == ["detector", "vlm"]

    def test_the_table_has_a_row_per_stage(self) -> None:
        recorder = Recorder()
        recorder.record("detector", 0.004)
        recorder.record("pose", 0.005)
        table = recorder.table()
        assert "detector" in table
        assert "pose" in table


class TestWrappersDelegate:
    """Each wrapper must be invisible to the pipeline — it implements the same port and
    returns the same value. A wrapper that changed behaviour would make the benchmark
    measure something the engine does not do."""

    async def test_the_detector_wrapper_returns_the_inner_result(self) -> None:
        expected = (Detection("person", 0.9, BBox(0.0, 0.0, 1.0, 1.0)),)

        class Inner(ObjectDetector):
            async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
                return expected

        recorder = Recorder()
        assert await TimedDetector(Inner(), recorder).detect(a_frame()) == expected
        assert recorder.samples["detector"]

    async def test_a_failing_call_is_still_measured(self) -> None:
        """In a `finally`. A benchmark that dropped the slow failures would report the
        healthy subset as though it were the whole distribution — which is exactly
        backwards, because failures are usually the slow ones."""

        class Exploding(ObjectDetector):
            async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
                raise RuntimeError("cuda is on fire")

        recorder = Recorder()
        with pytest.raises(RuntimeError):
            await TimedDetector(Exploding(), recorder).detect(a_frame())
        assert len(recorder.samples["detector"]) == 1

    async def test_pose_calls_with_no_tracks_are_not_counted(self) -> None:
        """`Yolo11PoseEstimator` short-circuits an empty frame without a forward pass.
        Counting those would drag the reported p50 toward zero on a mostly-empty
        camera and make pose look free precisely where it is not being used."""

        class Inner(PoseEstimator):
            async def estimate(
                self, frame: FrameData, tracks: tuple[Track, ...]
            ) -> dict[int, PersonPose]:
                return {}

        recorder = Recorder()
        wrapper = TimedPoseEstimator(Inner(), recorder)
        await wrapper.estimate(a_frame(), ())
        assert "pose" not in recorder.samples

        await wrapper.estimate(a_frame(), (a_track(),))
        assert len(recorder.samples["pose"]) == 1

    async def test_the_vlm_wrapper_returns_the_inner_description(self) -> None:
        expected = SceneDescription(
            description="Two people are talking.", threat_value=0.1, suggested_action="None."
        )

        class Inner(VisionLanguageModel):
            async def describe(self, request: VisionRequest) -> SceneDescription:
                return expected

        recorder = Recorder()
        request = VisionRequest(
            keyframe=a_frame(),
            scene=SceneState(
                camera_id="cam-1",
                frame_index=0,
                timestamp=0.0,
                detections=(),
                tracks=(),
                motion_energy=0.0,
                scene_signature=(1.0,),
            ),
            history=(),
            camera_label="Cam",
            reason_detail="test",
        )
        assert await TimedVisionLanguageModel(Inner(), recorder).describe(request) == expected
        assert recorder.samples["vlm"]
