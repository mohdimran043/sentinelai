"""Model fakes. These are what let the whole pipeline run on CPU in CI."""

from __future__ import annotations

from collections.abc import Sequence
from math import hypot

from sentinel_ai.domain.entities import BBox, Detection, Track
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime
from sentinel_ai.ports.tracker import Tracker
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest


class FakeDetector(ObjectDetector):
    """Replays a script of detections; repeats the last entry once exhausted."""

    def __init__(self, script: Sequence[tuple[Detection, ...]]) -> None:
        if not script:
            raise ValueError("script must contain at least one entry")
        self._script = list(script)
        self.call_count = 0

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        index = min(self.call_count, len(self._script) - 1)
        self.call_count += 1
        return self._script[index]


class FakeTracker(Tracker):
    """Nearest-centroid association — enough to exercise ageing and speed."""

    def __init__(self, match_radius_px: float = 80.0) -> None:
        self._match_radius = match_radius_px
        self._next_id = 1
        self._state: dict[int, tuple[str, BBox, int, float]] = {}

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Track, ...]:
        unmatched = dict(self._state)
        results: list[Track] = []
        new_state: dict[int, tuple[str, BBox, int, float]] = {}

        for detection in detections:
            best_id: int | None = None
            best_distance = self._match_radius
            for track_id, (label, box, _age, _ts) in unmatched.items():
                if label != detection.label:
                    continue
                distance = hypot(detection.box.cx - box.cx, detection.box.cy - box.cy)
                if distance <= best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None:
                track_id = self._next_id
                self._next_id += 1
                age, speed = 1, 0.0
            else:
                track_id = best_id
                _label, previous_box, previous_age, previous_ts = unmatched.pop(best_id)
                age = previous_age + 1
                elapsed = timestamp - previous_ts
                speed = (
                    hypot(
                        detection.box.cx - previous_box.cx,
                        detection.box.cy - previous_box.cy,
                    )
                    / elapsed
                    if elapsed > 0
                    else 0.0
                )

            new_state[track_id] = (detection.label, detection.box, age, timestamp)
            results.append(
                Track(
                    track_id=track_id,
                    label=detection.label,
                    box=detection.box,
                    age_frames=age,
                    speed_px_s=speed,
                )
            )

        self._state = new_state
        return tuple(results)

    def reset(self) -> None:
        self._state.clear()
        self._next_id = 1


class FakeVisionLLM(VisionLanguageModel):
    """Records every request so tests can assert what the gate escalated."""

    def __init__(
        self,
        response: SceneDescription | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or SceneDescription(
            description="Nothing notable in view.",
            threat_value=0.05,
            suggested_action="No action required.",
        )
        self._error = error
        self.requests: list[VisionRequest] = []

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response

    @property
    def call_count(self) -> int:
        return len(self.requests)


class FakeModelRuntime(ModelRuntime):
    """A controllable `ModelRuntime`: tests drive its lifecycle state directly rather
    than simulating a real load/warmup/shutdown sequence."""

    def __init__(
        self,
        model_key: str,
        kind: str = "vision",
        vram_mib: int = 100,
        initialize_error: Exception | None = None,
    ) -> None:
        self._model_key = model_key
        self._kind = kind
        self._vram_mib = vram_mib
        self._initialize_error = initialize_error
        self._state = LifecycleState.UNLOADED
        self.initialize_calls = 0
        self.warmup_calls = 0
        self.shutdown_calls = 0
        self.unhealthy_details: list[str] = []
        self.predict_calls: list[object] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self._initialize_error is not None:
            self._state = LifecycleState.UNHEALTHY
            raise self._initialize_error
        self._state = LifecycleState.LOADED

    async def warmup(self) -> None:
        self.warmup_calls += 1
        self._state = LifecycleState.HEALTHY

    async def predict(self, request: object) -> object:
        self.predict_calls.append(request)
        return request

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._state = LifecycleState.UNLOADED

    def mark_unhealthy(self, detail: str) -> None:
        self._state = LifecycleState.UNHEALTHY
        self.unhealthy_details.append(detail)

    def health(self) -> HealthReport:
        vram = self._vram_mib if self._state != LifecycleState.UNLOADED else 0
        detail = self.unhealthy_details[-1] if self.unhealthy_details else ""
        return HealthReport(state=self._state, detail=detail, vram_mib=vram)

    def version(self) -> str:
        return "fake-1"

    def capabilities(self) -> Capabilities:
        return Capabilities(model_key=self._model_key, kind=self._kind, vram_mib=self._vram_mib)
