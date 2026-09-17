"""Per-stage costs the sweep cannot isolate: the behaviour machines and the face models.

`harness.py` composes the whole engine and reports what each stage costs *inside* it,
which is the right measurement for the question "how many cameras fit". It cannot answer
two others:

* **What do the behaviour detectors cost?** They are microseconds. Inside a pipeline
  whose detector pass is 5 ms they round to nothing, and "rounds to nothing" is a claim
  worth being able to prove rather than assert.
* **What does the face pipeline cost?** It only runs on frames that contain a person
  *and* a readable face, so a sweep over arbitrary footage measures how often that
  happened, not what it costs when it does.

Both are therefore measured directly, on synthetic input with a stated shape, and both
print the same p50/p95 columns the sweep does.

Run it with `python -m sentinel_ai.benchmark.micro`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sentinel_ai.benchmark.timing import summarise
from sentinel_ai.domain.behaviour.abandonment import AbandonmentPolicy, observe_abandonment
from sentinel_ai.domain.behaviour.fall import FallPolicy, observe_falls
from sentinel_ai.domain.behaviour.observation import BehaviourObservation
from sentinel_ai.domain.behaviour.tamper import TamperPolicy, observe_tamper
from sentinel_ai.domain.behaviour.zones import (
    CrossingLine,
    RestrictedZone,
    ZonePolicy,
    observe_zones,
)
from sentinel_ai.domain.entities import BBox, Detection, SceneState, Track
from sentinel_ai.ports.frame_source import FrameData

__all__ = ["run_behaviour_micro", "run_face_micro"]

_FPS = 25.0
_FRAMES = 500
"""Enough that the p95 is not one unlucky sample, and short enough to run in a second."""

_WARM_FRAMES = 50
"""Discarded. The first frames pay for branch prediction and for the dict growth every
tracker does as it meets its first track."""


_Observer = Callable[..., tuple[Any, Any]]
"""`(observation, policy, state) -> (state, candidates)`, with the types erased.

Four detectors, four concrete policy types and four concrete state types, and nothing
in `domain/` makes them one family — deliberately, per
[ADR 13](../../../docs/decisions.md). A benchmark that only ever calls them and times
the call is the one place that erasure costs nothing, so it is declared here rather than
imposed on the detectors.
"""


def _scene(timestamp: float, people: int) -> SceneState:
    """A synthetic crowd walking at a constant speed.

    Constant, because these detectors are being measured rather than exercised: a scene
    that made one of them complete would time the escalation path instead of the
    per-frame cost, and the escalation path is what the sweep already measures.
    """
    detections: list[Detection] = []
    tracks: list[Track] = []
    for index in range(people):
        box = BBox(x1=100.0 + 40 * index, y1=200.0, x2=160.0 + 40 * index, y2=380.0)
        detections.append(Detection(label="person", confidence=0.9, box=box))
        tracks.append(
            Track(
                track_id=index,
                label="person",
                box=box,
                age_frames=int(timestamp * _FPS),
                speed_px_s=12.0,
            )
        )
    return SceneState(
        camera_id="micro",
        frame_index=int(timestamp * _FPS),
        timestamp=timestamp,
        detections=tuple(detections),
        tracks=tuple(tracks),
        motion_energy=0.2,
        # Flat: sixteen equal bins is a maximally varied view, which keeps the tamper
        # machine in its CLEAR phase rather than short-circuiting through a blank one.
        scene_signature=(1.0 / 16,) * 16,
    )


def _detectors() -> dict[str, tuple[Any, _Observer, Callable[[], Any]]]:
    from sentinel_ai.domain.behaviour.abandonment import AbandonmentTracker
    from sentinel_ai.domain.behaviour.fall import FallTracker
    from sentinel_ai.domain.behaviour.tamper import TamperState
    from sentinel_ai.domain.behaviour.zones import ZoneTracker

    return {
        "fall": (FallPolicy(), observe_falls, FallTracker),
        "abandonment": (AbandonmentPolicy(), observe_abandonment, AbandonmentTracker),
        "tamper": (TamperPolicy(), observe_tamper, TamperState),
        "zones": (
            ZonePolicy(
                zones=(
                    RestrictedZone(
                        name="micro-zone",
                        polygon=((0.0, 0.3), (0.4, 0.3), (0.4, 1.0), (0.0, 1.0)),
                    ),
                ),
                lines=(CrossingLine(name="micro-line", start=(0.5, 0.0), end=(0.5, 1.0)),),
            ),
            observe_zones,
            ZoneTracker,
        ),
    }


def run_behaviour_micro(crowd_sizes: Sequence[int] = (1, 10)) -> None:
    """Time each pure state machine over `_FRAMES` frames, at each crowd size."""
    print("\nBehaviour state machines — pure Python, no model, no GPU")
    for people in crowd_sizes:
        observations = [
            BehaviourObservation(
                scene=_scene(index / _FPS, people), frame_width=1920, frame_height=1080
            )
            for index in range(_FRAMES)
        ]
        print(f"\n  {people} tracked {'person' if people == 1 else 'people'}:")
        for name, (policy, observe, initial) in _detectors().items():
            state = initial()
            for observation in observations[:_WARM_FRAMES]:
                state, _ = observe(observation, policy, state)
            samples: list[float] = []  # milliseconds, as `summarise` expects
            for observation in observations:
                started = time.perf_counter()
                state, _ = observe(observation, policy, state)
                samples.append((time.perf_counter() - started) * 1000.0)
            summary = summarise(name, samples)
            print(
                f"    {name:12s} p50 {summary.p50_ms * 1000:8.1f} us"
                f"   p95 {summary.p95_ms * 1000:8.1f} us"
            )


async def run_face_micro(device: str = "cuda", faces: int = 6) -> None:
    """Time one `detect()` — which is detection *and* embedding, see the adapter.

    Needs `insightface` and a face image. The image comes from insightface's own bundled
    sample rather than from this repository, for §12's reason: a face image checked in
    to make a benchmark run is a face image stored for no operational purpose.
    """
    try:
        import cv2
        from insightface.data import get_image

        from sentinel_ai.adapters.face.insightface_pipeline import InsightFacePipeline
    except ImportError as error:
        print(f"\nFace pipeline not measured: {error}. Install the 'face-gpu' extra.")
        return

    frame_pixels = cv2.resize(get_image("t1"), (1920, 1080))
    blank: Any = np.random.randint(40, 90, (1080, 1920, 3), dtype=np.uint8)

    pipeline = InsightFacePipeline(device=device)
    started = time.perf_counter()
    await pipeline.initialize()
    await pipeline.warmup()
    load_seconds = time.perf_counter() - started

    print(f"\nFace pipeline — {pipeline.version()}")
    print(f"  loaded in {load_seconds:.1f}s, reports {pipeline.capabilities().vram_mib} MiB")
    for label, pixels in (("with faces", frame_pixels), ("no faces", blank)):
        samples = []
        found = 0
        for index in range(25):
            frame = FrameData(
                camera_id="micro",
                frame_index=index,
                timestamp=index / _FPS,
                width=int(pixels.shape[1]),
                height=int(pixels.shape[0]),
                pixels=pixels,
            )
            if index < 3:  # warm
                await pipeline.detect(frame)
                continue
            started = time.perf_counter()
            found = len(await pipeline.detect(frame))
            samples.append((time.perf_counter() - started) * 1000.0)
        summary = summarise(label, samples)
        print(
            f"  1080p, {found} faces   p50 {summary.p50_ms:7.1f} ms   p95 {summary.p95_ms:7.1f} ms"
        )
    await pipeline.shutdown()


def main() -> int:
    run_behaviour_micro()
    asyncio.run(run_face_micro())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
