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
        pixels = _as_pixels(frame.pixels)

        if self._previous is None or self._previous.shape != pixels.shape:
            # No previous frame, or a resolution change (a discontinuity, not
            # motion) -- treated exactly like the very first frame of a stream.
            energy = 0.0
        else:
            current = pixels.astype(np.float64)
            previous = self._previous.astype(np.float64)
            energy = float(np.mean(np.abs(current - previous))) / 255.0

        self._previous = pixels
        signature = _signature(_luma(pixels), self._bins)
        return MotionSignals(motion_energy=energy, scene_signature=signature)

    def reset(self) -> None:
        """Drop the previous frame -- call on stream discontinuity."""
        self._previous = None
