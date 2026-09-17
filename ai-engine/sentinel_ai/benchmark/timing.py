"""Latency recording, and the port wrappers that collect it (spec §32).

**Wrappers, not instrumentation.** Every timer here is a decorator implementing the
same port as the thing it wraps, installed by the benchmark harness and by nothing
else. The alternative — timing calls inside `Yolo11Detector` and friends — would put
measurement code on the hot path of the shipped engine, where it is a permanent small
cost paid by every deployment to answer a question only a benchmark asks. It would
also measure a pipeline subtly different from the one that ships.

Because the wrappers implement the ports, `main.compose` cannot tell the difference,
which is the property that makes the measurement worth having: what is timed is the
real graph, assembled the real way.

Percentiles, and why the mean is not reported
----------------------------------------------
`summarise` reports p50, p95 and max, and deliberately no mean. A mean latency hides
exactly the thing a surveillance system cares about — the tail — because one 400 ms
stall averaged across two hundred 4 ms frames disappears into the third decimal place.
The frames that matter are the slow ones: a camera whose p95 exceeds its frame
interval is a camera dropping frames, whatever its mean says.

`max` is reported rather than p99 because these runs are short enough that p99 would
often be interpolating between two samples, and calling that a percentile would dress
one unlucky frame up as a distribution.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from sentinel_ai.domain.behaviour.observation import PersonPose
from sentinel_ai.domain.entities import Detection, Track
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.face import DetectedFace, FaceDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.pose import PoseEstimator
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest

__all__ = [
    "LatencySummary",
    "Recorder",
    "TimedDetector",
    "TimedFaceDetector",
    "TimedPoseEstimator",
    "TimedVisionLanguageModel",
    "percentile",
    "summarise",
]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """What one stage cost, in milliseconds.

    `count` is carried because a percentile over four samples is not a percentile, and
    a reader comparing two stages needs to know when one of them barely ran — the VLM
    on a quiet clip typically fires a handful of times against the detector's
    thousands.
    """

    name: str
    count: int
    p50_ms: float
    p95_ms: float
    max_ms: float

    def row(self) -> str:
        return (
            f"{self.name:<22} {self.count:>7}  {self.p50_ms:>8.1f}  "
            f"{self.p95_ms:>8.1f}  {self.max_ms:>8.1f}"
        )


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than linear interpolation on purpose: an interpolated value is
    a number no frame actually took, and for a latency budget the honest answer to
    "what is the 95th percentile" is "the slowest of the fastest 95%", which is a
    measurement rather than an estimate.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if fraction <= 0.0:
        return ordered[0]
    rank = max(1, min(len(ordered), -(-int(fraction * len(ordered) * 1000) // 1000)))
    return ordered[rank - 1]


def summarise(name: str, samples: Sequence[float]) -> LatencySummary:
    """Milliseconds in, `LatencySummary` out.

    Empty samples summarise as zeros with a `count` of 0 rather than raising — a stage
    that never ran is a real outcome of a benchmark (no camera enabled it, or nothing
    escalated), and it should print as an empty row instead of taking the run down.
    """
    return LatencySummary(
        name=name,
        count=len(samples),
        p50_ms=percentile(samples, 0.50),
        p95_ms=percentile(samples, 0.95),
        max_ms=max(samples) if samples else 0.0,
    )


@dataclass
class Recorder:
    """Collected samples, by stage name.

    A plain list per stage, kept in full rather than reduced online. These runs are
    bounded (minutes, not days), so the memory is trivial, and keeping the raw samples
    means the percentile definition can change without re-running the benchmark.

    Not thread-safe, and does not need to be: every port method it wraps is `async`
    and appends on the event loop thread, after its offloaded work has returned.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)

    def record(self, name: str, seconds: float) -> None:
        self.samples.setdefault(name, []).append(seconds * 1000.0)

    def summaries(self) -> list[LatencySummary]:
        """Every stage, in insertion order — which is pipeline order, because that is
        the order the stages first run in."""
        return [summarise(name, values) for name, values in self.samples.items()]

    def table(self) -> str:
        header = f"{'stage':<22} {'calls':>7}  {'p50 ms':>8}  {'p95 ms':>8}  {'max ms':>8}"
        rows = [summary.row() for summary in self.summaries()]
        return "\n".join([header, "-" * len(header), *rows])


class TimedDetector(ObjectDetector):
    """`ObjectDetector`, timed. Delegates everything; adds one `perf_counter` pair."""

    def __init__(self, inner: ObjectDetector, recorder: Recorder, name: str = "detector") -> None:
        self._inner = inner
        self._recorder = recorder
        self._name = name

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        started = time.perf_counter()
        try:
            return await self._inner.detect(frame)
        finally:
            # In a `finally` so a failed call is still measured. A benchmark that
            # silently dropped the slow failures would report the healthy subset as
            # though it were the whole distribution.
            self._recorder.record(self._name, time.perf_counter() - started)


class TimedPoseEstimator(PoseEstimator):
    """`PoseEstimator`, timed.

    Records only calls that actually ran a forward pass. `Yolo11PoseEstimator` returns
    immediately when a frame has no tracks, and counting those as sub-microsecond
    samples would drag the reported p50 toward zero on any camera that is mostly
    empty — making pose look free precisely where it is not being used.
    """

    def __init__(self, inner: PoseEstimator, recorder: Recorder, name: str = "pose") -> None:
        self._inner = inner
        self._recorder = recorder
        self._name = name

    async def estimate(self, frame: FrameData, tracks: tuple[Track, ...]) -> dict[int, PersonPose]:
        if not tracks:
            return await self._inner.estimate(frame, tracks)
        started = time.perf_counter()
        try:
            return await self._inner.estimate(frame, tracks)
        finally:
            self._recorder.record(self._name, time.perf_counter() - started)


class TimedVisionLanguageModel(VisionLanguageModel):
    """`VisionLanguageModel`, timed. The stage whose tail matters most: a describe is
    two to three orders of magnitude slower than a detect, and §33's alert-latency
    budget is mostly this number plus the clip's post-roll."""

    def __init__(self, inner: VisionLanguageModel, recorder: Recorder, name: str = "vlm") -> None:
        self._inner = inner
        self._recorder = recorder
        self._name = name

    async def describe(self, request: VisionRequest) -> SceneDescription:
        started = time.perf_counter()
        try:
            return await self._inner.describe(request)
        finally:
            self._recorder.record(self._name, time.perf_counter() - started)


class TimedFaceDetector(FaceDetector):
    """`FaceDetector`, timed.

    Like `TimedPoseEstimator`, it records only the calls that did work. The runner skips
    the face pipeline entirely on a frame with no person tracks, and counting those
    would report the cost of face recognition on an empty corridor rather than on a busy
    one — which is the number that decides how many cameras fit.
    """

    def __init__(self, inner: FaceDetector, recorder: Recorder, name: str = "face") -> None:
        self._inner = inner
        self._recorder = recorder
        self._name = name

    async def detect(self, frame: FrameData) -> tuple[DetectedFace, ...]:
        started = time.perf_counter()
        try:
            return await self._inner.detect(frame)
        finally:
            self._recorder.record(self._name, time.perf_counter() - started)
