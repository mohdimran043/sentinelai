"""Ultralytics YOLO11-pose keypoint estimator (ports/pose.py, ports/model_runtime.py).

Implements both `PoseEstimator` (what the pipeline calls) and `ModelRuntime` (what the
registry owns the VRAM through), exactly as `Yolo11Detector` does — see that module for
the argument, which applies here unchanged including the shared-instance lock.

Heavy ML dependencies are imported lazily inside the methods that need them, never at
module scope, so this file and its pure helpers import and unit-test on a CPU box with
no `gpu` extra installed.

Why `n` rather than `s` by default
-----------------------------------
`Settings.pose_model_id` defaults to `yolo11n-pose.pt`, the smallest of the family,
where the object detector defaults to `yolo11s.pt`. The two are doing different jobs.
The detector's misses are unrecoverable — a person it never boxes is a person no
downstream stage can reason about at all. Pose runs *on people the detector already
found*, and `domain/behaviour/fall.py` degrades to bounding-box geometry whenever a
skeleton is missing or low-confidence, so a weaker pose model costs accuracy on a
signal that has a working fallback. Paying for that with a second per-person forward
pass on every camera is the wrong trade; §32 is where this default gets revisited
against measurement rather than argument.

Top-down, and what that costs
------------------------------
Ultralytics' pose models are single-stage: they detect people and place keypoints in
one pass over the whole frame, so this adapter does **not** crop per track. The
consequence is that its people and the pipeline's tracked people are two independent
detections that have to be reconciled, which `attribute_poses` does by IoU. A pose the
reconciliation cannot confidently attribute is dropped rather than guessed — the port
documents partial results as normal, and a skeleton attributed to the wrong person is
worse than no skeleton, because `fall.py` would then read one person's posture as
another's across a sequence of frames.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from sentinel_ai.adapters.vram import measure_model_vram_mib, sample_pools
from sentinel_ai.domain.behaviour.observation import Keypoint, KeypointName, PersonPose
from sentinel_ai.domain.entities import BBox, Track
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime
from sentinel_ai.ports.pose import PoseEstimator

if TYPE_CHECKING:
    from ultralytics import YOLO

_MODEL_CACHE_DIR = Path(os.environ.get("SENTINEL_MODEL_CACHE_DIR", "./var/models"))
"""Where a bare weights filename resolves to. Same directory and same reasoning as
`adapters/detectors/yolo11.py`; duplicated as a constant rather than imported so
neither adapter has to import the other to share a path."""

COCO_KEYPOINT_ORDER: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
"""The COCO-17 output order every Ultralytics pose model emits.

**This is the mapping the port exists to contain.** Positional indices live here, at
the boundary, and nowhere else: `domain/` names joints (`KeypointName`) and never
indexes them, so substituting a model with a different skeleton order is a change to
this tuple and touches no detector. A `domain` module reading `keypoints[5]` would
break silently on that substitution, which is the failure this shape prevents.
"""

_WANTED: dict[str, KeypointName] = {name.value: name for name in KeypointName}
"""Only the joints anything actually reads. A skeleton carrying all seventeen would be
sixteen floats of noise per person per frame that no consumer has a test for."""

MIN_ATTRIBUTION_IOU = 0.45
"""How well a pose's own person box must overlap a tracked box to be attributed to it.

Not a tuning knob so much as a correctness floor. Below a confident overlap the
skeleton is dropped: `fall.py` reasons about one person across seconds, so a pose
attributed to the wrong track does not merely add noise — it makes one person's
posture read as another's for as long as the confusion lasts, which is exactly how a
fall detector reports a fall that did not happen.
"""


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over union. Pure, so the attribution rule is testable without a GPU."""
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = a.area + b.area - intersection
    return intersection / union if union > 0.0 else 0.0


def attribute_poses(
    tracks: tuple[Track, ...],
    detected: tuple[tuple[BBox, dict[KeypointName, Keypoint]], ...],
    *,
    min_iou: float = MIN_ATTRIBUTION_IOU,
) -> dict[int, PersonPose]:
    """Match single-stage pose output onto the pipeline's tracks, best-overlap first.

    Pure and separately tested: this is the part of the adapter most able to be
    quietly wrong, and it needs no model weights to exercise.

    **One pose per track and one track per pose.** Pairs are taken in descending IoU
    order and both sides are consumed, so two overlapping people cannot both be
    attributed the same skeleton — in a crowd that is precisely the mistake that would
    let one person's fall be read off another's body. Anything left unmatched, on
    either side, is simply absent from the result.
    """
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for pose_index, (box, _) in enumerate(detected):
            overlap = _iou(track.box, box)
            if overlap >= min_iou:
                pairs.append((overlap, track_index, pose_index))
    # Sorted by overlap descending; ties broken by index so the result is deterministic
    # rather than dependent on detection order, which a model may vary run to run.
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    attributed: dict[int, PersonPose] = {}
    used_tracks: set[int] = set()
    used_poses: set[int] = set()
    for _, track_index, pose_index in pairs:
        if track_index in used_tracks or pose_index in used_poses:
            continue
        used_tracks.add(track_index)
        used_poses.add(pose_index)
        track = tracks[track_index]
        attributed[track.track_id] = PersonPose(
            track_id=track.track_id, keypoints=detected[pose_index][1]
        )
    return attributed


def keypoints_from_row(
    xy_row: list[list[float]], conf_row: list[float]
) -> dict[KeypointName, Keypoint]:
    """One person's raw keypoint arrays, mapped onto the names anything reads.

    Tolerant of a short row on purpose: a model variant emitting fewer than seventeen
    joints should yield the ones it does have rather than raising in the middle of a
    frame loop that has a working fallback.
    """
    keypoints: dict[KeypointName, Keypoint] = {}
    for index, name in enumerate(COCO_KEYPOINT_ORDER):
        wanted = _WANTED.get(name)
        if wanted is None or index >= len(xy_row):
            continue
        x, y = xy_row[index][0], xy_row[index][1]
        confidence = conf_row[index] if index < len(conf_row) else 0.0
        keypoints[wanted] = Keypoint(x=float(x), y=float(y), confidence=float(confidence))
    return keypoints


class Yolo11PoseEstimator(PoseEstimator, ModelRuntime):
    """One instance serves every camera that enabled a pose-needing capability.

    Shared and lock-serialised for `Yolo11Detector`'s reasons exactly: one GPU means
    the kernels serialise regardless, and `ultralytics.YOLO` keeps per-call state on
    `self.predictor`, so two cameras calling `predict` concurrently can silently
    attribute one camera's skeletons to the other.
    """

    def __init__(self, model_id: str, conf: float, imgsz: int, device: str) -> None:
        self._model_id = model_id
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._model: YOLO | None = None
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0
        self._vram_baseline_mib = 0
        self._allocated_baseline_mib = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        self._state = LifecycleState.DOWNLOADING
        # Before anything is placed — see `adapters/vram.py`. This adapter is the one
        # that exposed the process-wide-measurement bug: warming up last, it reported
        # the VLM's 2816 MiB for a model that costs about 460.
        self._vram_baseline_mib, self._allocated_baseline_mib = sample_pools(self._device)
        try:
            from ultralytics import YOLO

            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            weights_path = (
                self._model_id
                if Path(self._model_id).is_absolute()
                else str(_MODEL_CACHE_DIR / self._model_id)
            )
            self._model = YOLO(weights_path)
            self._state = LifecycleState.LOADED
        except Exception as exc:
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Yolo11PoseEstimator.warmup called before initialize()")
        dummy = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=self._imgsz,
            height=self._imgsz,
            pixels=np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8),
        )
        # One track, so `estimate` does not take its empty-tracks short circuit and the
        # forward pass genuinely runs — a warmup that skipped the kernel would leave
        # the first real frame paying the compile cost this method exists to absorb.
        warm_track = Track(
            track_id=0,
            label="person",
            box=BBox(x1=0.0, y1=0.0, x2=float(self._imgsz), y2=float(self._imgsz)),
            age_frames=1,
            speed_px_s=0.0,
        )
        await self.estimate(dummy, (warm_track,))
        self._vram_mib = measure_model_vram_mib(
            self._vram_baseline_mib, self._device, self._allocated_baseline_mib
        )
        self._state = LifecycleState.HEALTHY

    async def estimate(self, frame: FrameData, tracks: tuple[Track, ...]) -> dict[int, PersonPose]:
        if self._model is None:
            raise RuntimeError("Yolo11PoseEstimator.estimate called before initialize()")
        if not tracks:
            # Nothing to attribute a skeleton to, so the forward pass would be pure
            # cost. The common case on a quiet camera, and skipping it is most of why
            # pose is affordable at all.
            #
            # Before `pixel_array()`, deliberately: on a camera with nobody in it this
            # is the difference between a colour-space conversion per frame and none at
            # all, which is the whole point of the pixels being deferred.
            return {}
        resolved = frame.pixel_array()
        if not isinstance(resolved, np.ndarray):
            raise TypeError(f"FrameData.pixels must be a numpy array, got {type(resolved)!r}")
        pixels: npt.NDArray[np.uint8] = resolved
        loop = asyncio.get_running_loop()
        # Held across the offload, not merely around the submission — the shared
        # predictor state this protects lives on the executor thread.
        async with self._lock:
            model = self._model
            if model is None:  # pragma: no cover - a shutdown() that raced the acquire
                raise RuntimeError("Yolo11PoseEstimator.estimate called before initialize()")
            detected = await loop.run_in_executor(None, self._estimate_sync, model, pixels)
        return attribute_poses(tracks, detected)

    def _estimate_sync(
        self, model: YOLO, pixels: npt.NDArray[np.uint8]
    ) -> tuple[tuple[BBox, dict[KeypointName, Keypoint]], ...]:
        results = model.predict(
            source=pixels,
            conf=self._conf,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        result: Any = results[0]
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or boxes is None or len(boxes) == 0:
            return ()

        xy = keypoints.xy.tolist()
        if not xy:
            return ()
        # `.conf` is None when the checkpoint was exported without keypoint scores.
        # Substituting 1.0 would be the dangerous default — every joint would then
        # clear `min_keypoint_confidence` and a guessed hip would decide a posture —
        # so 0.0 is used instead, which makes `fall.py` fall back to geometry. A pose
        # model that cannot say how sure it is contributes nothing, which is correct.
        conf = (
            keypoints.conf.tolist()
            if keypoints.conf is not None
            else [[0.0] * len(xy[0]) for _ in xy]
        )
        person_boxes = boxes.xyxy.tolist()

        out: list[tuple[BBox, dict[KeypointName, Keypoint]]] = []
        for index, box_row in enumerate(person_boxes):
            if index >= len(xy):
                continue
            out.append(
                (
                    BBox(
                        x1=float(box_row[0]),
                        y1=float(box_row[1]),
                        x2=float(box_row[2]),
                        y2=float(box_row[3]),
                    ),
                    keypoints_from_row(xy[index], conf[index] if index < len(conf) else []),
                )
            )
        return tuple(out)

    async def predict(self, request: object) -> object:
        if not isinstance(request, FrameData):
            raise TypeError(f"Yolo11PoseEstimator.predict expects FrameData, got {type(request)!r}")
        return await self.estimate(request, ())

    async def shutdown(self) -> None:
        self._model = None
        self._vram_mib = 0
        import gc

        import torch

        # `gc.collect()` before `empty_cache()` is load-bearing, not tidiness — see
        # `adapters/detectors/yolo11.py` for the measurement behind it.
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
            kind="pose",
            labels=frozenset({"person"}),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
