"""Scene construction shared by the behaviour detector tests.

Every detector here is a pure function of a `BehaviourObservation`, so a test is a
scripted sequence of boxes at chosen timestamps. Building those by hand in four files
would be four subtly different ideas of what a scene is.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from sentinel_ai.domain.behaviour.observation import BehaviourObservation
from sentinel_ai.domain.entities import BBox, SceneState, Track


def track(track_id: int, label: str, box: BBox) -> Track:
    return Track(track_id=track_id, label=label, box=box, age_frames=10, speed_px_s=0.0)


def person_box(cx: float, ground_y: float, height: float = 100.0) -> BBox:
    """A standing person, positioned by where their *feet* are.

    Ground-anchored because that is how every zone and proximity test in this package
    measures — see `domain/behaviour/zones.py` on why centroids are the wrong point.
    """
    return BBox(x1=cx - height * 0.2, y1=ground_y - height, x2=cx + height * 0.2, y2=ground_y)


def bag_box(cx: float, ground_y: float, size: float = 40.0) -> BBox:
    return BBox(x1=cx - size / 2, y1=ground_y - size, x2=cx + size / 2, y2=ground_y)


def observation(
    tracks: Sequence[Track],
    timestamp: float,
    *,
    signature: tuple[float, ...] | None = None,
    width: int = 1920,
    height: int = 1080,
) -> BehaviourObservation:
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=timestamp,
        detections=(),
        tracks=tuple(tracks),
        motion_energy=0.0,
        # A plausibly varied 16-bin luma histogram unless a test is about tampering.
        scene_signature=signature if signature is not None else tuple([1.0 / 16] * 16),
    )
    return BehaviourObservation(scene=scene, frame_width=width, frame_height=height)


def flat_signature(concentration: float = 0.95, bins: int = 16) -> tuple[float, ...]:
    """A histogram with `concentration` of the frame in one bin — a blank view."""
    rest = (1.0 - concentration) / (bins - 1)
    return tuple([concentration] + [rest] * (bins - 1))


def ticks(start: float, stop: float, step: float) -> list[float]:
    out: list[float] = []
    current = start
    while current <= stop + 1e-9:
        out.append(round(current, 3))
        current += step
    return out


@pytest.fixture
def varied() -> tuple[float, ...]:
    return tuple([1.0 / 16] * 16)
