from __future__ import annotations

from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from tests.fakes.models import FakeModelRuntime

TOTAL_MIB = 8192
RESERVED_MIB = 2048

DETECTOR_SPEC = ModelSpec(model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None)
VLM_SPEC = ModelSpec(model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0)


def new_resident_set() -> tuple[ResidentSet, FakeModelRuntime, FakeModelRuntime]:
    registry = ModelRegistry()
    detector = FakeModelRuntime("yolo11s", vram_mib=900)
    vlm = FakeModelRuntime("qwen25vl3b", vram_mib=4400)
    registry.register(DETECTOR_SPEC, detector)
    registry.register(VLM_SPEC, vlm)
    return ResidentSet(registry, total_mib=TOTAL_MIB, reserved_mib=RESERVED_MIB), detector, vlm


class TestEnsure:
    async def test_a_required_model_is_loaded_and_warmed_up(self) -> None:
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        assert detector.initialize_calls == 1
        assert detector.warmup_calls == 1
        assert resident_set.resident() == frozenset({"yolo11s"})

    async def test_an_already_resident_model_is_not_reloaded(self) -> None:
        """Fails against a version that reloads every call — `initialize_calls` would
        be 2, not 1, after the second `ensure` finds the model already resident."""
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        await resident_set.ensure(("yolo11s",), now=1.0)
        assert detector.initialize_calls == 1

    async def test_both_models_fit_the_budget_together(self) -> None:
        resident_set, _detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s", "qwen25vl3b"), now=0.0)
        assert resident_set.resident() == frozenset({"yolo11s", "qwen25vl3b"})


class TestSweepIdle:
    async def test_the_vlm_is_unloaded_600s_after_its_last_ensure_call(self) -> None:
        """Fails against a `sweep_idle` that unloads nothing: at now=600.0 the VLM
        would still be resident and `shutdown_calls` would be 0, not 1."""
        resident_set, _detector, vlm = new_resident_set()
        await resident_set.ensure(("yolo11s", "qwen25vl3b"), now=0.0)
        await resident_set.sweep_idle(now=599.0)
        assert resident_set.resident() == frozenset({"yolo11s", "qwen25vl3b"})

        await resident_set.sweep_idle(now=600.0)
        assert resident_set.resident() == frozenset({"yolo11s"})
        assert vlm.shutdown_calls == 1

    async def test_the_detector_never_idle_evicts(self) -> None:
        """Fails against a `sweep_idle` that unloads everything regardless of
        `idle_unload_seconds`: the always-on detector (`idle_unload_seconds=None`)
        would be evicted here even after a 10,000s idle gap."""
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        await resident_set.sweep_idle(now=10_000.0)
        assert resident_set.resident() == frozenset({"yolo11s"})
        assert detector.shutdown_calls == 0

    async def test_re_ensuring_the_vlm_refreshes_its_idle_clock(self) -> None:
        resident_set, _detector, vlm = new_resident_set()
        await resident_set.ensure(("qwen25vl3b",), now=0.0)
        await resident_set.ensure(("qwen25vl3b",), now=500.0)  # a fresh "use" before 600s
        await resident_set.sweep_idle(now=1000.0)  # only 500s since the refresh
        assert resident_set.resident() == frozenset({"qwen25vl3b"})
        assert vlm.shutdown_calls == 0
