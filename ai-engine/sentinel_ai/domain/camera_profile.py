"""Per-camera tuning.

Slice 1 uses fixed defaults; Phase 4 replaces the percentile and baseline
fields with learned per-camera values.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SALIENT_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "backpack",
        "handbag",
        "suitcase",
    }
)


@dataclass(frozen=True, slots=True)
class CameraProfile:
    camera_id: str
    salient_classes: frozenset[str] = DEFAULT_SALIENT_CLASSES

    min_track_frames: int = 8
    scene_delta_threshold: float = 0.35
    scene_delta_frames: int = 5
    dwell_radius_px: float = 48.0
    dwell_seconds: float = 30.0
    speed_percentile: float = 0.95
    speed_fallback_px_s: float = 180.0
    track_count_baseline: int = 6
    summary_interval_seconds: float = 45.0

    bucket_capacity: int = 2
    bucket_refill_seconds: float = 10.0

    cooldown_seconds: float = 5.0
    """Minimum spacing between VLM invocations (spec §4.1's second gate).

    Deliberately shorter than `bucket_refill_seconds`: at or above it the
    cooldown would dominate the bucket entirely and `bucket_capacity` would
    become dead configuration, since the burst allowance that lets a genuinely
    novel scene get a second look could never be spent. At half the refill
    interval it caps the worst case at 2 calls in 5 s rather than 2 calls in
    100 ms, while leaving the steady-state rate the bucket sets untouched. It
    also comfortably exceeds a single VLM turnaround, so the next call is never
    admitted before the previous description exists.
    """

    vlm_enabled: bool = True

    def __post_init__(self) -> None:
        self._require(self.min_track_frames >= 1, "min_track_frames must be >= 1")
        self._require(
            0.0 < self.scene_delta_threshold <= 1.0,
            "scene_delta_threshold must be in (0.0, 1.0]",
        )
        self._require(self.scene_delta_frames >= 1, "scene_delta_frames must be >= 1")
        self._require(self.dwell_radius_px >= 0.0, "dwell_radius_px must be >= 0")
        self._require(self.dwell_seconds > 0.0, "dwell_seconds must be > 0")
        self._require(0.0 < self.speed_percentile < 1.0, "speed_percentile must be in (0.0, 1.0)")
        self._require(self.speed_fallback_px_s > 0.0, "speed_fallback_px_s must be > 0")
        self._require(self.track_count_baseline >= 1, "track_count_baseline must be >= 1")
        self._require(self.summary_interval_seconds > 0.0, "summary_interval_seconds must be > 0")
        self._require(self.bucket_capacity >= 1, "bucket_capacity must be >= 1")
        self._require(self.bucket_refill_seconds > 0.0, "bucket_refill_seconds must be > 0")
        self._require(self.cooldown_seconds > 0.0, "cooldown_seconds must be > 0")

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
