"""Warmup must exercise *both* face models, not just the one that runs on a blank frame.

The blind spot this pins, measured on an RTX 4090: `warmup()` ran `detect()` against a
blank 640x640 frame, SCRFD correctly found no faces in it, and so ArcFace was never
called and ONNX Runtime never allocated its workspace. The adapter reported 480 MiB for
a model whose steady-state footprint is 608 — a third of it invisible until the first
frame with a person in it, by which time `plan_residency()` has already admitted a
model set on the strength of the smaller number.

Fakes rather than the real bundle: this is about the call the adapter makes, and
`buffalo_l` is a 300 MB download that needs a GPU to be interesting.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from sentinel_ai.adapters.face.insightface_pipeline import InsightFacePipeline
from sentinel_ai.ports.model_runtime import LifecycleState


class _FakeRecognition:
    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def get_feat(self, imgs: list[Any]) -> Any:
        self.batches.append(list(imgs))
        return np.zeros((len(imgs), 512), dtype=np.float32)


class _RaisingRecognition:
    def get_feat(self, imgs: list[Any]) -> Any:
        raise RuntimeError("no CUDA kernel for this shape")


class _FakeAnalysis:
    """Stands in for `insightface.app.FaceAnalysis`. Finds nothing, like a blank frame."""

    def __init__(self, recognition: Any, **kwargs: Any) -> None:
        self.models = {"recognition": recognition}
        self.prepared: dict[str, Any] | None = None

    def prepare(self, **kwargs: Any) -> None:
        self.prepared = kwargs

    def get(self, pixels: Any) -> list[Any]:
        return []


def _pipeline_with(recognition: Any) -> tuple[InsightFacePipeline, Any]:
    analysis = _FakeAnalysis(recognition)
    pipeline = InsightFacePipeline(device="cpu")
    return pipeline, analysis


async def _start(pipeline: InsightFacePipeline, analysis: Any) -> None:
    with patch("insightface.app.FaceAnalysis", return_value=analysis):
        await pipeline.initialize()
    await pipeline.warmup()


class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_runs_the_recognition_model_too(self) -> None:
        """The assertion the 148 MiB blind spot needed. A warmup that only proves SCRFD
        loads is a warmup that leaves a third of the footprint to be discovered in
        production."""
        recognition = _FakeRecognition()
        pipeline, analysis = _pipeline_with(recognition)
        await _start(pipeline, analysis)
        assert recognition.batches, "recognition model was never called during warmup"

    @pytest.mark.asyncio
    async def test_the_warmup_face_is_synthetic_and_correctly_shaped(self) -> None:
        """ArcFace takes an aligned 112x112 crop and does not care whether it depicts
        anybody, which is the whole reason this repository ships no face image (§12)."""
        recognition = _FakeRecognition()
        pipeline, analysis = _pipeline_with(recognition)
        await _start(pipeline, analysis)
        (crop,) = recognition.batches[0]
        assert crop.shape == (112, 112, 3)
        assert not crop.any(), "the warmup crop must be synthetic, not a photograph"

    @pytest.mark.asyncio
    async def test_a_failed_recognition_warmup_does_not_fail_the_model(self) -> None:
        """Costing an accurate VRAM figure is worth reporting. Costing the capability
        would be a much worse trade — this is the one optional model whose absence
        turns an enabled capability into a silent no-op, which §40 forbids."""
        pipeline, analysis = _pipeline_with(_RaisingRecognition())
        await _start(pipeline, analysis)
        assert pipeline.health().state is LifecycleState.HEALTHY


class TestProviderIsReported:
    @pytest.mark.asyncio
    async def test_version_names_the_provider_actually_in_use(self) -> None:
        """A CPU fallback is 8x slower than the GPU path (125 ms against 15 ms for one
        1080p frame holding six faces). An operator must be able to read which one they
        got, rather than infer it from a latency graph."""
        pipeline, analysis = _pipeline_with(_FakeRecognition())
        await _start(pipeline, analysis)
        assert "CPUExecutionProvider" in pipeline.version()
