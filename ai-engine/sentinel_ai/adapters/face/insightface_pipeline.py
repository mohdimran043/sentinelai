"""Face detection and embedding via InsightFace (ports/face.py, ports/model_runtime.py).

SCRFD for detection and ArcFace (`buffalo_l`) for embedding, which is the pairing most
face-recognition work has converged on. Both run under ONNX Runtime, **not** torch, and
that isolation is deliberate: an earlier attempt at this used `facenet-pytorch`, which
pins `torch<2.3` and duly downgraded the engine's torch 2.14 install, breaking the
detector, the pose model and the VLM in one command. A face pipeline is an optional
capability and must not be able to do that.

One model, two ports
---------------------
`ports/face.py` splits detection from embedding because §9's enrolment flow rejects
between the two, and because they are independently replaceable. InsightFace's
`FaceAnalysis` does not split — one `get()` call detects, aligns and embeds together,
and asking it for detection alone then embedding separately would run the detector
twice per frame.

So this class implements both ports over one pass: `detect()` performs the whole
analysis and carries each face's **already-computed embedding** in `DetectedFace.aligned`,
which the port types as an opaque `object` for exactly this kind of implementation
freedom. `embed()` then unwraps rather than recomputing.

The consequence is worth stating rather than hiding: a face that fails its quality
check has already cost its embedding. That is wasted work, and it is cheaper than a
second detection pass over the frame. What the quality check still buys — and the
reason it is applied at all — is that the *policy* never sees an embedding it should
not trust (`domain/policy/authorization.py`).

Frontality is an estimate, and a coarse one
---------------------------------------------
`FaceQuality.frontality` is derived from the five landmarks SCRFD emits, as how centred
the nose sits between the eyes. That is a yaw proxy: it detects a head turned left or
right, which is the rotation that actually destroys a face embedding. It is close to
blind to pitch — somebody looking at the floor scores well — and it cannot see roll at
all. A real pose estimate needs a model this pipeline does not load, and the honest
handling is to say so here rather than to present the number as more than it is.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from sentinel_ai.adapters.vram import device_used_mib, measure_foreign_vram_mib
from sentinel_ai.domain.entities import BBox
from sentinel_ai.domain.identity import FaceEmbedding, FaceQuality
from sentinel_ai.ports.face import DetectedFace, FaceDetector, FaceEmbedder
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

__all__ = ["ARCFACE_DIMENSION", "InsightFacePipeline", "estimate_frontality"]

ARCFACE_DIMENSION = 512
"""What `buffalo_l`'s recognition model emits. Named because the store refuses to
compare vectors of different dimensions, and a mismatch there is the symptom of an
embedding model swapped without re-enrolling anybody."""

_MODEL_CACHE_DIR = Path(os.environ.get("SENTINEL_MODEL_CACHE_DIR", "./var/models"))


def estimate_frontality(landmarks: npt.NDArray[np.float32] | None) -> float:
    """How face-on the subject is, in [0, 1], from SCRFD's five landmarks.

    The landmark order is (left eye, right eye, nose, left mouth, right mouth). The
    measure is how far the nose sits from the midpoint of the eyes, as a fraction of
    the eye separation: face-on puts it in the middle, a turned head pushes it toward
    one eye.

    Pure and separately tested, because it is the part of this adapter most able to be
    quietly wrong and it needs no model to exercise. Its limits are in the module
    docstring — this sees yaw, and is close to blind to pitch.
    """
    if landmarks is None or len(landmarks) < 3:
        # No landmarks, no estimate. 0.0 rather than 1.0: an unknown pose must fail a
        # frontality floor rather than sail through it.
        return 0.0
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_separation = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_separation <= 0.0:
        # Eyes at the same x means a profile view or a degenerate detection. Either way
        # there is no usable face here.
        return 0.0
    midpoint = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    offset = abs(float(nose[0]) - midpoint) / eye_separation
    # Doubled so that a nose halfway to one eye scores 0 rather than 0.5 — at that
    # point the head is turned far enough that the embedding is not worth trusting.
    return max(0.0, min(1.0, 1.0 - 2.0 * offset))


class InsightFacePipeline(FaceDetector, FaceEmbedder, ModelRuntime):
    """One shared instance across every camera that enabled person authorisation.

    Lock-serialised for `Yolo11Detector`'s reasons: the ONNX session holds per-call
    state and there is one accelerator underneath whatever this does.
    """

    def __init__(self, model_name: str = "buffalo_l", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._app: FaceAnalysis | None = None
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0
        self._vram_baseline_mib = 0
        self._on_gpu = False
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return ARCFACE_DIMENSION

    async def initialize(self) -> None:
        self._state = LifecycleState.DOWNLOADING
        # Driver-level, not torch's allocator: ONNX Runtime allocates outside it, so
        # `sample_pools` reports this model as costing 0 MiB however much it holds.
        # See `adapters/vram.device_used_mib`.
        self._vram_baseline_mib = device_used_mib(self._device)
        try:
            from insightface.app import FaceAnalysis

            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # CUDA when the runtime offers it, CPU otherwise. The fallback keeps the
            # capability working rather than making it fast: measured on an RTX 4090,
            # one 1080p frame holding six faces takes 15 ms under
            # `CUDAExecutionProvider` and 125 ms under `CPUExecutionProvider`. A site
            # that falls back is a site whose face cameras are capped near 8 fps, so
            # the provider actually chosen is reported through `version()` rather than
            # left for somebody to infer from a latency graph.
            providers = ["CPUExecutionProvider"]
            if self._device.startswith("cuda"):
                import onnxruntime

                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            app = FaceAnalysis(
                name=self._model_name,
                root=str(_MODEL_CACHE_DIR),
                providers=providers,
                # Detection and recognition only. `genderage` and `landmark_3d_68` ship
                # in the same bundle and are not loaded: this system has no use for an
                # inferred age or gender, and holding a model that produces them
                # invites somebody to start recording them.
                allowed_modules=["detection", "recognition"],
            )
            self._on_gpu = providers[0].startswith("CUDA")
            app.prepare(ctx_id=0 if self._on_gpu else -1, det_size=(640, 640))
            self._app = app
            self._state = LifecycleState.LOADED
        except Exception as exc:
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._app is None:
            raise RuntimeError("InsightFacePipeline.warmup called before initialize()")
        dummy = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=640,
            height=640,
            pixels=np.zeros((640, 640, 3), dtype=np.uint8),
        )
        await self.detect(dummy)
        await self._warm_recognition()
        # Measured after both warmup passes, so the figure includes the detection *and*
        # recognition workspaces ONNX Runtime allocates lazily on first use. That is
        # about a third of the total, and all of it memory the planner has to have
        # counted before it admits the next model.
        self._vram_mib = measure_foreign_vram_mib(self._vram_baseline_mib, self._device)
        self._state = LifecycleState.HEALTHY

    async def _warm_recognition(self) -> None:
        """Force one ArcFace pass, because the warmup frame has no face in it.

        `detect()` on a blank frame exercises SCRFD and stops there: no face is found,
        so the recognition model is never called and ONNX Runtime never allocates its
        workspace. Measured on an RTX 4090 that left **148 MiB** — a third of this
        model's real footprint — invisible to `warmup()`, to be allocated later by the
        first frame with a person in it, at which point `plan_residency()` has already
        admitted a model set on the strength of the smaller figure.

        The alternative was to warm up on a real photograph, which would mean shipping
        a face in the repository. §12 says not to store face images that are not
        needed, and this one is not: ArcFace takes an aligned 112x112 crop and does not
        care whether it depicts anybody.

        Best-effort. A failure here costs an accurate VRAM figure, which is worth
        reporting; it must not cost the capability, which would be a much worse trade.
        """
        app = self._app
        recognition = None if app is None else app.models.get("recognition")
        if recognition is None:  # pragma: no cover - a bundle without recognition
            return
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, recognition.get_feat, [crop])
        except Exception as error:  # pragma: no cover - warmup is advisory
            logger.warning(
                "face recognition warmup failed; the reported VRAM figure will "
                "undercount this model: %s",
                error,
            )

    async def detect(self, frame: FrameData) -> tuple[DetectedFace, ...]:
        if self._app is None:
            raise RuntimeError("InsightFacePipeline.detect called before initialize()")
        resolved = frame.pixel_array()
        if not isinstance(resolved, np.ndarray):
            raise TypeError(f"FrameData.pixels must be a numpy array, got {type(resolved)!r}")
        pixels: npt.NDArray[np.uint8] = resolved
        loop = asyncio.get_running_loop()
        async with self._lock:
            app = self._app
            if app is None:  # pragma: no cover - a shutdown that raced the acquire
                raise RuntimeError("InsightFacePipeline.detect called before initialize()")
            return await loop.run_in_executor(None, self._detect_sync, app, pixels)

    def _detect_sync(
        self, app: FaceAnalysis, pixels: npt.NDArray[np.uint8]
    ) -> tuple[DetectedFace, ...]:
        found: list[DetectedFace] = []
        for face in app.get(pixels):
            raw: Any = face
            x1, y1, x2, y2 = (float(value) for value in raw.bbox)
            box = BBox(x1=x1, y1=y1, x2=x2, y2=y2)
            quality = FaceQuality(
                # The *smaller* side: a 200x20 sliver is a 20-pixel face, and it is the
                # limiting dimension that decides whether an embedder can read it.
                box_pixels=int(min(x2 - x1, y2 - y1)),
                detector_confidence=float(raw.det_score),
                frontality=estimate_frontality(getattr(raw, "kps", None)),
            )
            embedding = FaceEmbedding.of(tuple(float(value) for value in raw.normed_embedding))
            # The embedding rides in `aligned`, which the port types as opaque for
            # exactly this — see the module docstring on why this pipeline cannot
            # usefully separate the two stages.
            found.append(DetectedFace(box=box, quality=quality, aligned=embedding))
        return tuple(found)

    async def embed(self, faces: tuple[DetectedFace, ...]) -> tuple[FaceEmbedding, ...]:
        """Unwrap the embeddings `detect` already produced.

        No forward pass happens here. Order-preserving by construction, which is the
        contract callers rely on to pair embeddings back to their detections.
        """
        embeddings: list[FaceEmbedding] = []
        for face in faces:
            if not isinstance(face.aligned, FaceEmbedding):
                raise TypeError(
                    "this embedder only accepts faces produced by its own detect(); "
                    f"got {type(face.aligned)!r} in DetectedFace.aligned"
                )
            embeddings.append(face.aligned)
        return tuple(embeddings)

    async def predict(self, request: object) -> object:
        if not isinstance(request, FrameData):
            raise TypeError(f"InsightFacePipeline.predict expects FrameData, got {type(request)!r}")
        return await self.detect(request)

    async def shutdown(self) -> None:
        self._app = None
        self._vram_mib = 0
        self._on_gpu = False
        import gc

        gc.collect()
        self._state = LifecycleState.UNLOADED

    def mark_unhealthy(self, detail: str) -> None:
        self._state = LifecycleState.UNHEALTHY
        self._health_detail = detail

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        import insightface

        provider = "CUDAExecutionProvider" if self._on_gpu else "CPUExecutionProvider"
        return f"insightface=={insightface.__version__} model={self._model_name} {provider}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_key=self._model_name,
            kind="face",
            labels=frozenset({"face"}),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
