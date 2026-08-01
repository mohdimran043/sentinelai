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

from pathlib import Path

import pytest

from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, _to_detections
from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.detector import ObjectDetector
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
