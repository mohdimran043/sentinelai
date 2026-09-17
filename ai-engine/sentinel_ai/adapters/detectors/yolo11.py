"""Ultralytics YOLO11 object detector (spec §5; ports/detector.py, ports/model_runtime.py).

Implements both `ObjectDetector` (the typed interface the pipeline calls) and
`ModelRuntime` (the lifecycle interface the registry manages) per S13 —
`ModelRuntime.predict()` delegates to `detect()`.

Heavy ML dependencies (`torch`, `ultralytics`) are imported lazily, inside the
methods that need them, never at module scope. This is not a style
preference: it is what lets this module — and every pure helper in it — be
imported and unit-tested on CPU in CI without the `gpu` extra installed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sentinel_ai.adapters.vram import measure_model_vram_mib, sample_pools
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ultralytics import YOLO

_MODEL_CACHE_DIR = Path(os.environ.get("SENTINEL_MODEL_CACHE_DIR", "./var/models"))
"""Where a bare weights filename (e.g. "yolo11s.pt") resolves to on disk.

Ultralytics downloads a bare filename into the current working directory,
which is unpredictable across dev/container/systemd invocations. Resolving
through one fixed cache directory keeps the download in one place regardless
of launch context. The env var mirrors the `SENTINEL_` settings prefix
without adding a field to the frozen `Settings` class (S1 is closed here).
"""

_CUDA_CONTEXT_OVERHEAD_MIB = 300
"""Fixed tax added on top of `torch.cuda.memory_reserved()` in `warmup()`.

`torch.cuda.memory_allocated()` — the naive choice — only counts currently
live tensors; it excludes the caching allocator's reserved pool entirely.
`memory_reserved()` is materially closer to reality (it *is* memory the
driver has actually handed to this process), but it still misses the CUDA
context itself — created on first kernel launch, ~150-300 MiB, held for the
life of the process — and cuDNN/cuBLAS workspace buffers. Both of those are
real and visible to `nvidia-smi`, but no torch-level API surfaces the
context's true size.

This constant is therefore a deliberate, documented over-estimate rather
than a measured value. `capabilities().vram_mib` feeds `plan_residency()`
(`sentinel_ai/domain/policy/vram_budget.py`), which does hard
admission-control arithmetic against the total VRAM budget: the planner
evicting one model too early because this estimate ran high is recoverable;
an OOM mid-escalation because it ran low is not. Measured on an RTX 4060 with
YOLO11s at imgsz=640: `memory_allocated()` 68 MiB, `memory_reserved()` 132
MiB, real `nvidia-smi` delta ~281 MiB — i.e. even `memory_reserved()` alone
undershoots by ~150 MiB, which is what this constant is covering.
"""


def select_device() -> str:
    """Return "cuda" if a CUDA device is visible, else "cpu".

    Ultralytics does not probe availability itself: passing device="cuda" on
    a machine with none raises deep inside torch, not at construction. The
    caller decides *before* construction — hence `device` is a required
    constructor argument (S13), not a default computed inside the adapter.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_detections(
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    names: Mapping[int, str],
) -> tuple[Detection, ...]:
    """Pure conversion from Ultralytics' box arrays to domain `Detection`s.

    Takes plain Python values, not ultralytics/torch tensors, so it is
    unit-testable with an injected `names` mapping and no model at all.
    """
    return tuple(
        Detection(
            label=names[class_id],
            confidence=float(confidence),
            box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        )
        for (x1, y1, x2, y2), confidence, class_id in zip(
            boxes_xyxy, confidences, class_ids, strict=True
        )
    )


class Yolo11Detector(ObjectDetector, ModelRuntime):
    """One instance serves every camera, so it must serialise its own model access.

    `compose()` builds exactly one detector and hands it to every `CameraRunner`
    (unlike trackers, motion analyzers, pre-roll buffers and sources, which are all
    per-camera). That sharing is deliberate — see `_lock` below for why the
    alternative is worse — but it means N camera tasks call `detect()` concurrently
    on one `ultralytics.YOLO`, and `YOLO.predict` keeps per-call state on
    `self.predictor` (`batch`, `results`, `source`). Ultralytics documents one model
    instance per thread for exactly this reason. Unsynchronised, two cameras'
    `predict` calls interleave on that shared state and the detections of one camera
    are attributed to the other — silently, because both calls still return a
    well-formed result.
    """

    def __init__(self, model_id: str, conf: float, iou: float, imgsz: int, device: str) -> None:
        self._model_id = model_id
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._model: YOLO | None = None
        self._names: dict[int, str] = {}
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0
        self._vram_baseline_mib = 0
        self._allocated_baseline_mib = 0
        self._lock = asyncio.Lock()
        """Serialises `predict()` across every camera sharing this detector.

        An `asyncio.Lock` rather than one detector per camera. Note what does *not*
        justify that: a second detector would comfortably fit. `plan_residency()`
        admits against `total_mib - reserved_mib` = 6144 usable, and the shipped
        figures are 432 MiB for the detector and 2766 MiB for the VLM (`config.py`) —
        the values Tasks 11 and 13 actually measured on the RTX 4060, not the 900 and
        4400 of spec §4.3's design table, which both models came in well under. So
        2766 + 2*432 = 3630 leaves 2514 MiB free, and `InsufficientVram` would not
        appear until eight cameras.

        The argument that does carry: there is one GPU, so the kernels serialise
        whatever we do here. A lock makes that explicit at zero VRAM cost, while
        per-camera weights buy no parallelism and scale VRAM linearly in cameras on a
        box with a single card. A dedicated single-thread
        executor was the other candidate; it buys the same mutual exclusion but adds
        an executor to create and tear down across every initialize/shutdown cycle,
        and with this lock held the model is already only ever touched by one thread
        at a time. Costs nothing but latency the pipeline already absorbs: a camera
        waiting here simply drops staler frames in its `_LatestSlot`, which is what
        that mailbox is for.
        """

    async def initialize(self) -> None:
        self._state = LifecycleState.DOWNLOADING
        # Sampled before anything is placed, so `warmup()` can report *this* model's
        # footprint rather than the process-wide pool — see `adapters/vram.py` for the
        # bug that made every model after the first claim its predecessors' memory.
        self._vram_baseline_mib, self._allocated_baseline_mib = sample_pools(self._device)
        try:
            from ultralytics import YOLO

            from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES

            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            weights_path = (
                self._model_id
                if Path(self._model_id).is_absolute()
                else str(_MODEL_CACHE_DIR / self._model_id)
            )
            model = YOLO(weights_path)
            names: dict[int, str] = dict(model.names)

            missing = DEFAULT_SALIENT_CLASSES - set(names.values())
            if missing:
                raise ValueError(
                    f"model_id={self._model_id!r} names are missing salient classes "
                    f"{missing!r} — every salient-class escalation trigger would "
                    "silently never fire"
                )

            self._model = model
            self._names = names
            self._state = LifecycleState.LOADED
        except Exception as exc:
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Yolo11Detector.warmup called before initialize()")
        dummy = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=self._imgsz,
            height=self._imgsz,
            pixels=np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8),
        )
        await self.detect(dummy)
        self._vram_mib = measure_model_vram_mib(
            self._vram_baseline_mib, self._device, self._allocated_baseline_mib
        )
        self._state = LifecycleState.HEALTHY

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        if self._model is None:
            raise RuntimeError("Yolo11Detector.detect called before initialize()")
        loop = asyncio.get_running_loop()
        # Held across the offload, not merely around the submission: the shared state
        # this protects lives inside `predict()`, on the executor thread.
        async with self._lock:
            model = self._model
            if model is None:  # pragma: no cover - a shutdown() that raced the acquire
                raise RuntimeError("Yolo11Detector.detect called before initialize()")
            return await loop.run_in_executor(None, self._detect_sync, model, frame)

    def _detect_sync(self, model: YOLO, frame: FrameData) -> tuple[Detection, ...]:
        """Resolve the frame's pixels and run the forward pass, both on this thread.

        Resolving *here* rather than in `detect()` is deliberate and was measured. The
        detector is the first consumer of a frame in `CameraRunner._process_frame`, so
        it is the one that pays `DeferredPixels`' colour-space conversion — and doing
        that in `detect()` would pay it on the event loop. At twenty cameras this
        process uses 113% of a *single* core on a 24-core box, which makes the loop the
        saturated resource: a millisecond of numpy on it is a millisecond no other
        camera gets, while 23 cores sit idle. Here it lands on one of them.

        Every later consumer — motion, pose, faces, the VLM keyframe — reads the value
        this call cached, so none of them pays it again or pays it on the loop.
        """
        resolved = frame.pixel_array()
        if not isinstance(resolved, np.ndarray):
            raise TypeError(f"FrameData.pixels must be a numpy array, got {type(resolved)!r}")
        pixels: npt.NDArray[np.uint8] = resolved
        results = model.predict(
            source=pixels,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = [tuple(row) for row in boxes.xyxy.tolist()]
        confidences = boxes.conf.tolist()
        class_ids = [int(c) for c in boxes.cls.tolist()]
        return _to_detections(xyxy, confidences, class_ids, self._names)

    async def predict(self, request: object) -> object:
        if not isinstance(request, FrameData):
            raise TypeError(f"Yolo11Detector.predict expects FrameData, got {type(request)!r}")
        return await self.detect(request)

    async def shutdown(self) -> None:
        self._model = None
        self._vram_mib = 0
        import gc

        import torch

        # gc.collect() before empty_cache() is load-bearing, not tidiness: an
        # nn.Module graph is full of reference cycles, so dropping the last
        # reference does not free it under CPython refcounting alone — it needs
        # a collection pass before the caching allocator has anything to hand
        # back to the driver. Measured on Qwen2.5-VL in Task 13: empty_cache()
        # alone left ~2.4 GB reserved after shutdown, and gc.collect() first
        # dropped it to ~54 MiB. The 600 s idle-unload in ResidentSet only
        # reclaims VRAM if this genuinely releases it.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state = LifecycleState.UNLOADED

    def mark_unhealthy(self, detail: str) -> None:
        self._state = LifecycleState.UNHEALTHY
        self._health_detail = detail

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        import ultralytics

        return f"ultralytics=={ultralytics.__version__} model={self._model_id}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_key=self._model_id,
            kind="detector",
            labels=frozenset(self._names.values()),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
