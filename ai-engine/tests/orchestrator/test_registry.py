from __future__ import annotations

import pytest

from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.ports.model_runtime import LifecycleState
from tests.fakes.models import FakeModelRuntime

DETECTOR_SPEC = ModelSpec(model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None)
VLM_SPEC = ModelSpec(model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0)


def test_register_and_get_round_trip() -> None:
    registry = ModelRegistry()
    runtime = FakeModelRuntime("yolo11s")
    registry.register(DETECTOR_SPEC, runtime)
    assert registry.get("yolo11s") is runtime
    assert registry.specs() == (DETECTOR_SPEC,)


def test_get_of_an_unknown_key_raises() -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="qwen25vl3b"):
        registry.get("qwen25vl3b")


async def test_state_reflects_the_runtimes_own_health() -> None:
    registry = ModelRegistry()
    runtime = FakeModelRuntime("yolo11s")
    registry.register(DETECTOR_SPEC, runtime)
    assert registry.state("yolo11s") == LifecycleState.UNLOADED
    await runtime.initialize()
    assert registry.state("yolo11s") == LifecycleState.LOADED


async def test_health_aggregates_every_registered_runtime() -> None:
    registry = ModelRegistry()
    detector = FakeModelRuntime("yolo11s")
    vlm = FakeModelRuntime("qwen25vl3b")
    registry.register(DETECTOR_SPEC, detector)
    registry.register(VLM_SPEC, vlm)
    await detector.initialize()

    health = registry.health()

    assert set(health) == {"yolo11s", "qwen25vl3b"}
    assert health["yolo11s"].state == LifecycleState.LOADED
    assert health["qwen25vl3b"].state == LifecycleState.UNLOADED


def test_kind_reads_from_the_runtimes_capabilities() -> None:
    """There is no `kind` field on `ModelSpec` (Reconciliation Log R2): the registry
    must read it from `ModelRuntime.capabilities().kind` instead."""
    registry = ModelRegistry()
    registry.register(VLM_SPEC, FakeModelRuntime("qwen25vl3b", kind="vision-language"))
    assert registry.kind("qwen25vl3b") == "vision-language"
