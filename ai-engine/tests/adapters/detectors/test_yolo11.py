"""Tests for Yolo11Detector (Task 11).

Only the parts that need no torch, no ultralytics weights, and no GPU run
without a marker — see yolo11.py's module docstring for why the lazy-import
split makes that possible. The salient-class guard test below also runs
unmarked: it injects a fake `ultralytics` module via `sys.modules` instead of
needing the real package importable, so the safety-critical guard it proves
gets checked in plain CI too. The facts that genuinely need the real
`ultralytics` package (its bundled COCO names) or real inference are
`@pytest.mark.gpu` further down; CI does not install the `gpu` extra, so
those are skipped there, not merely deselected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pytest

from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, _to_detections
from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import ModelRuntime


def test_yolo11_detector_satisfies_both_ports() -> None:
    assert issubclass(Yolo11Detector, ObjectDetector)
    assert issubclass(Yolo11Detector, ModelRuntime)


def test_to_detections_maps_class_ids_through_the_injected_names() -> None:
    names = {0: "person", 2: "car"}
    result = _to_detections(
        boxes_xyxy=[(0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 25.0, 25.0)],
        confidences=[0.91, 0.77],
        class_ids=[0, 2],
        names=names,
    )
    assert result == (
        Detection(label="person", confidence=0.91, box=BBox(0.0, 0.0, 10.0, 10.0)),
        Detection(label="car", confidence=0.77, box=BBox(5.0, 5.0, 25.0, 25.0)),
    )


def test_to_detections_on_no_boxes_is_empty() -> None:
    assert _to_detections([], [], [], {}) == ()


def test_to_detections_with_wrong_names_mapping_produces_wrong_labels() -> None:
    """A mapping that swaps two class ids must produce swapped labels, not raise.

    This is the guard against a vacuous test: if `_to_detections` ignored
    `names` (e.g. hardcoded "person"), this assertion would fail rather than
    merely "not raise" — proving the function actually uses the injected
    mapping to resolve each class id.
    """
    wrong_names = {0: "car", 2: "person"}  # deliberately swapped vs. real COCO
    result = _to_detections(
        boxes_xyxy=[(0.0, 0.0, 10.0, 10.0)],
        confidences=[0.5],
        class_ids=[0],
        names=wrong_names,
    )
    assert result[0].label == "car"
    assert result[0].label != "person"


async def test_real_detector_raises_and_marks_unhealthy_on_missing_salient_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-time guard against a mismatched checkpoint: this test would
    fail (no exception, state stays healthy) if `initialize()` did not
    actually check `model.names` against `DEFAULT_SALIENT_CLASSES`.

    Injects a fake `ultralytics` module via `sys.modules` rather than
    monkeypatching an attribute on the real, installed package — the guard
    this proves is safety-critical (its absence means the system silently
    never alerts on a person), so it must run in plain CI, which does not
    install the `gpu` extra `ultralytics` ships under.
    """
    import sys
    import types

    class _FakeYolo:
        def __init__(self, _weights_path: str) -> None:
            self.names = {0: "person", 1: "car"}  # missing most salient classes

    fake_ultralytics = types.ModuleType("ultralytics")
    # `types.ModuleType` has no static `YOLO` attribute to assign directly
    # under mypy strict; `setattr` is the standard escape for building a fake
    # module object like this.
    setattr(fake_ultralytics, "YOLO", _FakeYolo)  # noqa: B010
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    detector = Yolo11Detector(model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device="cpu")
    with pytest.raises(ValueError, match="salient"):
        await detector.initialize()
    assert detector.health().state.value == "unhealthy"


_P = ParamSpec("_P")
_T = TypeVar("_T")


class _ManualExecutor(ThreadPoolExecutor):
    """Queues submitted work instead of running it, so a test can count exactly how
    many `_detect_sync` bodies the detector has released at once.

    A `ThreadPoolExecutor` subclass because `loop.set_default_executor()` type-checks
    for one, and `Yolo11Detector.detect` offloads through `run_in_executor(None, ...)`
    — so this intercepts the real production code path unchanged. No thread is ever
    started, nothing sleeps, and the interleaving under test is decided entirely by
    when the test calls `run_next()`.
    """

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.pending: list[tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]] = []

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        future: Future[_T] = Future()
        self.pending.append((future, fn, args))
        return future

    def run_next(self) -> None:
        future, fn, args = self.pending.pop(0)
        future.set_result(fn(*args))


class _Boxes:
    def __init__(self, class_id: int) -> None:
        self.xyxy = np.array([[0.0, 0.0, 10.0, 10.0]])
        self.conf = np.array([0.9])
        self.cls = np.array([float(class_id)])

    def __len__(self) -> int:
        return 1


class _Result:
    def __init__(self, class_id: int) -> None:
        self.boxes = _Boxes(class_id)


class _SharedStateYolo:
    """Stand-in for `ultralytics.YOLO` that keeps per-call state on `self`, exactly as
    the real one keeps `batch`/`results`/`source` on `self.predictor`.

    `predict()` writes the incoming frame to `self.batch` and then reads it back to
    build the result. That is the whole defect in miniature: if two cameras' calls are
    ever in flight together, the second overwrites `batch` before the first reads it
    and camera A is handed camera B's detections — a well-formed, silently wrong
    answer, not a crash.
    """

    def __init__(self) -> None:
        self.batch: Any = None
        self.calls = 0

    def predict(self, *, source: Any, verbose: bool, **_kwargs: Any) -> list[_Result]:
        self.calls += 1
        self.batch = source
        return [_Result(class_id=int(self.batch[0][0][0]) // 10 - 1)]


def _frame(camera_id: str, value: int) -> FrameData:
    return FrameData(
        camera_id=camera_id,
        frame_index=0,
        timestamp=0.0,
        width=4,
        height=4,
        pixels=np.full((4, 4, 3), value, dtype=np.uint8),
    )


async def _settle() -> None:
    """Let every ready coroutine reach its next suspension point. Zero wall clock."""
    for _ in range(10):
        await asyncio.sleep(0)


async def test_two_cameras_detecting_concurrently_never_share_a_model_call() -> None:
    """B1: one `Yolo11Detector` is shared by every `CameraRunner` (`main.py`'s
    `compose()`), and `cameras.example.json` ships two cameras — so two `detect()`
    calls are genuinely in flight at once on one `ultralytics.YOLO`, whose `predict()`
    keeps per-call state on `self.predictor`.

    Fails against the unsynchronised version with `2 == 1`: both `_detect_sync` bodies
    are handed to the executor in the same event-loop pass, which on the real default
    thread pool means two threads inside `predict()` on the same model object at the
    same time. The reviewer's probe measured exactly that — "max concurrent detect()
    bodies in flight: 2, distinct executor threads used: 16".

    Deterministic by construction: the executor never runs anything until this test
    says so, so "how many bodies were released" is a fact about the detector's own
    synchronisation and nothing else.
    """
    executor = _ManualExecutor()
    asyncio.get_running_loop().set_default_executor(executor)

    model = _SharedStateYolo()
    detector = Yolo11Detector(model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device="cpu")
    detector._model = model
    detector._names = {0: "person", 1: "car"}

    task_a = asyncio.create_task(detector.detect(_frame("cam-a", 10)))
    task_b = asyncio.create_task(detector.detect(_frame("cam-b", 20)))
    await _settle()

    assert len(executor.pending) == 1, (
        "two cameras must never have a model call in flight at the same time"
    )
    executor.run_next()
    await _settle()

    assert len(executor.pending) == 1, "the second camera's call is released only after the first"
    executor.run_next()
    await _settle()

    detections_a = await task_a
    detections_b = await task_b
    assert model.calls == 2
    assert [d.label for d in detections_a] == ["person"], "cam-a must get cam-a's detections"
    assert [d.label for d in detections_b] == ["car"], "cam-b must get cam-b's detections"


@pytest.mark.gpu
async def test_real_shared_detector_stays_correct_under_concurrent_cameras() -> None:
    """A smoke check on real hardware, **not** a regression pin for B1.

    Stated plainly because this project has already shipped twenty-two tests that
    discriminated nothing: this test passes against the unlocked detector too. It was
    run both ways on an RTX 4060 — 60 rounds of four concurrent `detect()` calls on one
    shared model, locked and unlocked — and neither produced a single mis-attributed
    result. The Python-level interleaving that corrupts `self.predictor` is real by
    construction (ultralytics documents one model instance per thread) but did not
    reproduce here; CUDA synchronisation inside `predict()` appears to serialise the
    window in practice, which is a property of this driver and this model, not a
    guarantee.

    What it does earn its place for: proving the lock does not deadlock or change the
    answers under genuine concurrent load on a real GPU. The discriminating pin is
    `test_two_cameras_detecting_concurrently_never_share_a_model_call` above.
    """
    import cv2
    import ultralytics

    from sentinel_ai.adapters.detectors.yolo11 import select_device

    detector = Yolo11Detector(
        model_id="yolo11s.pt", conf=0.25, iou=0.45, imgsz=640, device=select_device()
    )
    await detector.initialize()
    await detector.warmup()

    pixels = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    assert pixels is not None
    height, width = pixels.shape[:2]
    busy = FrameData(
        camera_id="cam-a",
        frame_index=0,
        timestamp=0.0,
        width=width,
        height=height,
        pixels=pixels,
    )
    blank = FrameData(
        camera_id="cam-b",
        frame_index=0,
        timestamp=0.0,
        width=width,
        height=height,
        pixels=np.zeros((height, width, 3), dtype=np.uint8),
    )

    for _ in range(10):
        busy_result, blank_result = await asyncio.gather(
            detector.detect(busy), detector.detect(blank)
        )
        assert "person" in {d.label for d in busy_result}, "the busy camera lost its detections"
        assert blank_result == (), "the blank camera was handed another camera's detections"

    await detector.shutdown()


@pytest.mark.gpu
def test_real_coco_names_cover_every_default_salient_class() -> None:
    """The bundled coco.yaml ships inside the installed `ultralytics` package
    — no download, no CUDA — but the package itself is `gpu`-extra only, so
    this still cannot run in plain CI.
    """
    import ultralytics
    import yaml

    coco_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    names = set(yaml.safe_load(coco_yaml.read_text())["names"].values())
    assert names >= DEFAULT_SALIENT_CLASSES


@pytest.mark.gpu
def test_real_coco_names_would_catch_a_mismatch() -> None:
    """Proves the `<=` check in `initialize()` is not vacuous: a names set
    missing a salient class must fail the same assertion `initialize()` makes.
    """
    incomplete_names = {"person", "car"}  # missing truck/bus/motorcycle/etc.
    assert not (incomplete_names >= DEFAULT_SALIENT_CLASSES)


@pytest.mark.gpu
async def test_real_detector_loads_warms_up_and_reports_health() -> None:
    from sentinel_ai.adapters.detectors.yolo11 import select_device

    detector = Yolo11Detector(
        model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device=select_device()
    )
    await detector.initialize()
    await detector.warmup()
    assert detector.health().state.value == "healthy"
    assert detector.capabilities().vram_mib > 0
    assert detector.capabilities().labels >= DEFAULT_SALIENT_CLASSES
    await detector.shutdown()
    assert detector.health().state.value == "unloaded"
    # An unloaded model must not keep reporting VRAM it no longer holds.
    assert detector.capabilities().vram_mib == 0


@pytest.mark.gpu
async def test_real_detector_detects_a_person_in_a_real_image() -> None:
    """Runs actual inference end-to-end and checks the *content* of the
    result on a real photo of a person, not just that it returned without
    raising: a detector that always returned `()` fails this test (it would
    only have passed the old, weaker version of it, which asserted merely
    that a blank frame does not hallucinate a person — a fact `()` also
    satisfies vacuously).
    """
    import cv2
    import numpy as np
    import ultralytics

    from sentinel_ai.adapters.detectors.yolo11 import select_device
    from sentinel_ai.ports.frame_source import FrameData

    detector = Yolo11Detector(
        model_id="yolo11s.pt", conf=0.25, iou=0.45, imgsz=640, device=select_device()
    )
    await detector.initialize()
    await detector.warmup()

    # A synthetic frame with no real content should not hallucinate a person.
    blank = FrameData(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        width=640,
        height=640,
        pixels=np.zeros((640, 640, 3), dtype=np.uint8),
    )
    blank_detections = await detector.detect(blank)
    assert "person" not in {d.label for d in blank_detections}

    # `bus.jpg` ships inside the installed `ultralytics` package (its
    # long-standing quickstart demo image: a bus with several people at a
    # stop) — real content, no download, deterministic across runs, and no
    # new binary fixture added to this repo. `cv2.imread` decodes straight to
    # BGR `uint8`, matching `FrameData.pixels`' documented convention.
    image_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    pixels = cv2.imread(str(image_path))
    assert pixels is not None, f"failed to decode fixture image at {image_path}"
    height, width = pixels.shape[:2]
    real_photo = FrameData(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        width=width,
        height=height,
        pixels=pixels,
    )
    detections = await detector.detect(real_photo)
    labels = {d.label for d in detections}
    assert detections, "expected at least one detection in a real photo of people at a bus stop"
    assert "person" in labels

    await detector.shutdown()


@pytest.mark.gpu
async def test_real_detector_rejects_non_ndarray_pixels() -> None:
    from sentinel_ai.adapters.detectors.yolo11 import select_device
    from sentinel_ai.ports.frame_source import FrameData

    detector = Yolo11Detector(
        model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device=select_device()
    )
    await detector.initialize()
    bad_frame = FrameData(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        width=4,
        height=4,
        pixels=[[0, 0, 0]],
    )
    with pytest.raises(TypeError, match="pixels"):
        await detector.detect(bad_frame)
    await detector.shutdown()
