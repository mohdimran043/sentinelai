"""`DeferredPixels`: convert once, remember, and let the decoder's frame go.

The measured reason this exists is in its own docstring. These tests pin the three
properties the pipeline depends on, and the one it must not accidentally acquire.
"""

from __future__ import annotations

import numpy as np

from sentinel_ai.ports.frame_source import DeferredPixels, FrameData


def a_frame(pixels: object) -> FrameData:
    return FrameData(
        camera_id="cam-1", frame_index=0, timestamp=0.0, width=4, height=3, pixels=pixels
    )


class TestLaziness:
    def test_nothing_is_converted_until_it_is_asked_for(self) -> None:
        calls = 0

        def convert() -> object:
            nonlocal calls
            calls += 1
            return np.zeros((3, 4, 3), dtype=np.uint8)

        deferred = DeferredPixels(convert)
        a_frame(deferred)
        assert calls == 0
        assert not deferred.resolved

    def test_the_conversion_happens_once_however_often_it_is_read(self) -> None:
        """A keyframe is read by the detector and again by the VLM an escalation later.
        Paying twice would make an escalation cost more than the frames around it."""
        calls = 0

        def convert() -> object:
            nonlocal calls
            calls += 1
            return np.zeros((3, 4, 3), dtype=np.uint8)

        frame = a_frame(DeferredPixels(convert))
        first = frame.pixel_array()
        assert frame.pixel_array() is first
        assert frame.pixel_array() is first
        assert calls == 1

    def test_the_converter_is_released_so_the_decoder_buffer_can_be(self) -> None:
        """It closes over the decoder's `VideoFrame`. A retained keyframe holding one
        would pin a decode buffer for as long as the escalation takes — on every camera,
        for every escalation."""
        sentinel = object()
        deferred = DeferredPixels(lambda: sentinel)
        deferred.get()
        assert deferred.resolved
        assert deferred._convert is None


class TestEagerPixelsStillWork:
    def test_a_plain_array_passes_straight_through(self) -> None:
        """Every test in this repository that builds a frame by hand does it this way,
        and none of them should have to know `DeferredPixels` exists."""
        pixels = np.zeros((3, 4, 3), dtype=np.uint8)
        assert a_frame(pixels).pixel_array() is pixels

    def test_a_non_array_is_returned_unchanged_rather_than_rejected_here(self) -> None:
        """The port stays free of numpy, so the type check belongs to the adapters that
        need an array — and each of them does one. Rejecting here would put numpy in
        `ports/` and fail the architecture test."""
        assert a_frame("not pixels").pixel_array() == "not pixels"
