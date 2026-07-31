from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sentinel_ai.pipeline.stages.motion import SIGNATURE_BINS, MotionAnalyzer
from sentinel_ai.ports.frame_source import FrameData


def a_frame(pixels: npt.NDArray[np.uint8]) -> FrameData:
    height, width = pixels.shape[0], pixels.shape[1]
    return FrameData(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        width=width,
        height=height,
        pixels=pixels,
    )


def solid(value: int, height: int = 8, width: int = 8) -> npt.NDArray[np.uint8]:
    return np.full((height, width, 3), value, dtype=np.uint8)


def signature_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Mirrors `SceneState.signature_delta`'s total-variation formula, so
    this test pins the exact property the escalation gate's SceneChange
    trigger depends on without importing the domain layer into a pipeline test."""
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / 2.0


class TestMotionAnalyzer:
    def test_first_frame_yields_exactly_zero_energy(self) -> None:
        analyzer = MotionAnalyzer()
        signals = analyzer.analyze(a_frame(solid(128)))
        assert signals.motion_energy == 0.0

    def test_identical_frames_yield_zero_energy(self) -> None:
        analyzer = MotionAnalyzer()
        analyzer.analyze(a_frame(solid(128)))
        signals = analyzer.analyze(a_frame(solid(128)))
        assert signals.motion_energy == 0.0

    def test_a_fully_inverted_frame_yields_energy_near_one(self) -> None:
        analyzer = MotionAnalyzer()
        analyzer.analyze(a_frame(solid(0)))
        signals = analyzer.analyze(a_frame(solid(255)))
        assert signals.motion_energy == pytest.approx(1.0)

    def test_signature_sums_to_one_and_has_the_configured_length(self) -> None:
        analyzer = MotionAnalyzer(bins=SIGNATURE_BINS)
        signals = analyzer.analyze(a_frame(solid(128)))
        assert len(signals.scene_signature) == SIGNATURE_BINS
        assert sum(signals.scene_signature) == pytest.approx(1.0)

    def test_black_and_white_frames_have_disjoint_signatures_with_delta_one(self) -> None:
        analyzer = MotionAnalyzer()
        black = analyzer.analyze(a_frame(solid(0))).scene_signature
        white = analyzer.analyze(a_frame(solid(255))).scene_signature
        assert signature_delta(black, white) == pytest.approx(1.0)

    def test_reset_drops_the_previous_frame(self) -> None:
        """A no-op reset would still compare against the black frame below,
        giving energy near 1.0 instead of exactly 0.0."""
        analyzer = MotionAnalyzer()
        analyzer.analyze(a_frame(solid(0)))
        analyzer.reset()
        signals = analyzer.analyze(a_frame(solid(255)))
        assert signals.motion_energy == 0.0

    def test_a_resolution_change_is_treated_as_a_discontinuity_not_motion(self) -> None:
        analyzer = MotionAnalyzer()
        analyzer.analyze(a_frame(solid(0, height=8, width=8)))
        signals = analyzer.analyze(a_frame(solid(255, height=16, width=16)))
        assert signals.motion_energy == 0.0

    def test_rejects_non_positive_bin_count(self) -> None:
        with pytest.raises(ValueError, match="bins"):
            MotionAnalyzer(bins=0)

    def test_rejects_non_ndarray_pixels(self) -> None:
        analyzer = MotionAnalyzer()
        bad_frame = FrameData(
            camera_id="cam-1", frame_index=0, timestamp=0.0, width=1, height=1, pixels=[[0]]
        )
        with pytest.raises(TypeError, match="numpy array"):
            analyzer.analyze(bad_frame)
