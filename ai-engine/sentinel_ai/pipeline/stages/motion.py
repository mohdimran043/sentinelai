"""Motion stage (spec §5.2): motion energy and scene signature.

Outermost layer, so numpy is allowed here -- and nowhere in `domain/` or
`ports/` (the architecture fitness test enforces that; this module must never
be imported from either).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sentinel_ai.ports.frame_source import FrameData

SIGNATURE_BINS: int = 16


@dataclass(frozen=True, slots=True)
class MotionSignals:
    motion_energy: float
    scene_signature: tuple[float, ...]


def _as_pixels(value: object) -> npt.NDArray[np.uint8]:
    """Narrow `FrameData.pixels` (typed `object` to keep numpy out of ports)."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"FrameData.pixels must be a numpy array, got {type(value).__name__}")
    return value


MOTION_SAMPLE_STRIDE = 4
"""Take every Nth pixel in each axis before measuring motion.

Sixteen times less data for a statistic that is a *mean over the whole frame* and a
16-bin histogram — neither of which changes meaningfully when computed on a regular
quarter-resolution sample. What does change is the cost: this runs on the event loop for
every processed frame, and at 1080p the full-frame version was the single largest thing
on it.

A stride rather than a resize: slicing is a view, so it allocates nothing and touches
only the pixels it reads, while any real downscale would read all of them first — which
is the cost being avoided.

The same stride is applied to the previous frame by construction, because what is stored
in `_previous` is already sampled. Comparing a sampled frame against a full one would
compare different things and report it as motion.
"""


def _sampled(pixels: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    stride = MOTION_SAMPLE_STRIDE
    if pixels.ndim < 2 or min(pixels.shape[0], pixels.shape[1]) <= stride:
        # Too small to sample without changing the answer. A test fixture, usually.
        return pixels
    sampled: npt.NDArray[np.uint8] = pixels[::stride, ::stride]
    return sampled


def _luma(pixels: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """BT.601 luma. `pixels` is `to_ndarray(format='bgr24')` shaped (H, W, 3),
    or already single-channel (H, W)."""
    if pixels.ndim == 2:
        return pixels.astype(np.float64)
    blue = pixels[..., 0].astype(np.float64)
    green = pixels[..., 1].astype(np.float64)
    red = pixels[..., 2].astype(np.float64)
    result: npt.NDArray[np.float64] = 0.114 * blue + 0.587 * green + 0.299 * red
    return result


def _signature(luma: npt.NDArray[np.float64], bins: int) -> tuple[float, ...]:
    counts, _ = np.histogram(luma, bins=bins, range=(0.0, 255.0))
    total = counts.sum()
    if total == 0:
        return tuple(1.0 / bins for _ in range(bins))
    normalized = counts / total
    return tuple(float(value) for value in normalized)


class MotionAnalyzer:
    def __init__(self, bins: int = SIGNATURE_BINS) -> None:
        if bins <= 0:
            raise ValueError(f"bins must be positive, got {bins}")
        self._bins = bins
        self._previous: npt.NDArray[np.uint8] | None = None

    def analyze(self, frame: FrameData) -> MotionSignals:
        """Motion energy vs the previous frame, plus a normalised luma histogram.

        The first frame yields motion_energy == 0.0.
        """
        pixels = _sampled(_as_pixels(frame.pixel_array()))

        if self._previous is None or self._previous.shape != pixels.shape:
            # No previous frame, or a resolution change (a discontinuity, not
            # motion) -- treated exactly like the very first frame of a stream.
            energy = 0.0
        else:
            # `int16`, not `float64`, and the difference is not a micro-optimisation.
            # This runs inline on the event loop for every processed frame, and a
            # 1920x1080 frame promoted to float64 is a 50 MB allocation — twice, for
            # current and previous — before the subtraction even starts. Measured on two
            # 1080p cameras that was most of the loop's time, and the loop is the
            # resource this engine is short of (ADR 17).
            #
            # `int16` is exactly wide enough: both operands are `uint8`, so the
            # difference fits in [-255, 255] with no overflow and no rounding. The
            # result is identical to the float version, to the bit.
            current = pixels.astype(np.int16)
            previous = self._previous.astype(np.int16)
            energy = float(np.mean(np.abs(current - previous))) / 255.0

        self._previous = pixels
        signature = _signature(_luma(pixels), self._bins)
        return MotionSignals(motion_energy=energy, scene_signature=signature)

    def reset(self) -> None:
        """Drop the previous frame -- call on stream discontinuity."""
        self._previous = None
