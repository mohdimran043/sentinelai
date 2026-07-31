from __future__ import annotations

from typing import Any

import pytest

from sentinel_ai.domain.policy.vram_budget import (
    InsufficientVram,
    ModelSpec,
    ResidencyPlan,
    plan_residency,
)

DETECTOR = ModelSpec("yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None)
VLM = ModelSpec("qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0)
ALT_VLM = ModelSpec("llava7b_awq", vram_mib=4400, priority=50, idle_unload_seconds=600.0)
BIG_VLM = ModelSpec("qwen25vl7b", vram_mib=6200, priority=50, idle_unload_seconds=600.0)
SPECS = {spec.model_key: spec for spec in (DETECTOR, VLM, ALT_VLM, BIG_VLM)}

TOTAL_MIB = 8192
RESERVED_MIB = 2048
USABLE_MIB = TOTAL_MIB - RESERVED_MIB  # 6144


def plan(**kwargs: Any) -> ResidencyPlan:
    defaults: dict[str, Any] = {
        "specs": SPECS,
        "currently_resident": (),
        "required": (),
        "last_used_at": {},
        "now": 0.0,
        "total_mib": TOTAL_MIB,
        "reserved_mib": RESERVED_MIB,
    }
    return plan_residency(**{**defaults, **kwargs})


class TestLoading:
    def test_a_required_model_is_scheduled_for_load(self) -> None:
        result = plan(required=("yolo11s",))
        assert result.load == ("yolo11s",)
        assert result.resident == ("yolo11s",)

    def test_an_already_resident_model_is_not_reloaded(self) -> None:
        result = plan(currently_resident=("yolo11s",), required=("yolo11s",))
        assert result.load == ()
        assert result.resident == ("yolo11s",)

    def test_detector_and_vlm_both_fit_the_8gb_budget(self) -> None:
        """Spec §4.3: 900 + 4400 = 5300 MiB against a 6144 MiB usable budget."""
        result = plan(required=("yolo11s", "qwen25vl3b"))
        assert set(result.resident) == {"yolo11s", "qwen25vl3b"}
        assert result.free_mib == TOTAL_MIB - RESERVED_MIB - 900 - 4400

    def test_reserved_headroom_is_never_allocated(self) -> None:
        result = plan(required=("yolo11s",))
        assert result.free_mib == TOTAL_MIB - RESERVED_MIB - 900


class TestEviction:
    def test_a_resident_model_is_evicted_to_make_room_for_a_peer(self) -> None:
        """Two 4400 MiB VLMs cannot co-reside in 6144 MiB, so one must go."""
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=("llava7b_awq",),
            last_used_at={"qwen25vl3b": 0.0},
            now=10.0,
        )
        assert result.unload == ("qwen25vl3b",)
        assert result.resident == ("llava7b_awq",)

    def test_eviction_prefers_the_least_recently_used(self) -> None:
        specs = {
            "a": ModelSpec("a", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
            "b": ModelSpec("b", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
            "c": ModelSpec("c", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
        }
        result = plan_residency(
            specs=specs,
            currently_resident=("a", "b"),
            required=("c",),
            last_used_at={"a": 5.0, "b": 1.0},
            now=10.0,
            total_mib=TOTAL_MIB,
            reserved_mib=RESERVED_MIB,
        )
        assert result.unload == ("b",), "b was used longest ago"

    def test_a_higher_priority_resident_model_is_never_evicted(self) -> None:
        """The detector must stay resident — it is the always-on stage.

        Loading a second VLM needs space. The lower-priority VLM is evicted and
        the detector survives, even though it is the least recently used.
        """
        result = plan(
            currently_resident=("yolo11s", "qwen25vl3b"),
            required=("llava7b_awq",),
            last_used_at={"yolo11s": 0.0, "qwen25vl3b": 9_999.0},
            now=10_000.0,
        )
        assert result.unload == ("qwen25vl3b",)
        assert set(result.resident) == {"yolo11s", "llava7b_awq"}

    def test_idle_models_are_unloaded_even_when_no_load_is_needed(self) -> None:
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=(),
            last_used_at={"qwen25vl3b": 0.0},
            now=601.0,
        )
        assert result.unload == ("qwen25vl3b",)

    def test_a_model_within_its_idle_window_stays_resident(self) -> None:
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=(),
            last_used_at={"qwen25vl3b": 0.0},
            now=599.0,
        )
        assert result.unload == ()

    def test_a_model_with_no_idle_timeout_is_never_idle_evicted(self) -> None:
        result = plan(
            currently_resident=("yolo11s",),
            required=(),
            last_used_at={"yolo11s": 0.0},
            now=1_000_000.0,
        )
        assert result.unload == ()


class TestFailure:
    def test_a_model_larger_than_the_budget_raises(self) -> None:
        specs = {"huge": ModelSpec("huge", vram_mib=99_000, priority=1, idle_unload_seconds=None)}
        with pytest.raises(InsufficientVram, match="huge"):
            plan_residency(
                specs=specs,
                currently_resident=(),
                required=("huge",),
                last_used_at={},
                now=0.0,
                total_mib=TOTAL_MIB,
                reserved_mib=RESERVED_MIB,
            )

    def test_an_unknown_model_key_raises(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            plan(required=("nope",))

    def test_it_raises_when_only_higher_priority_models_could_be_evicted(self) -> None:
        """The detector outranks the VLM, so a VLM that needs its space cannot load."""
        specs = {
            "detector": ModelSpec(
                "detector", vram_mib=5000, priority=100, idle_unload_seconds=None
            ),
            "vlm": ModelSpec("vlm", vram_mib=5000, priority=50, idle_unload_seconds=600.0),
        }
        with pytest.raises(InsufficientVram, match="nothing evictable"):
            plan_residency(
                specs=specs,
                currently_resident=("detector",),
                required=("vlm",),
                last_used_at={"detector": 0.0},
                now=0.0,
                total_mib=TOTAL_MIB,
                reserved_mib=RESERVED_MIB,
            )


def test_the_7b_vlm_cannot_fit_the_8gb_dev_budget_at_all() -> None:
    """Validates the spec's central hardware finding: 6200 MiB > 6144 MiB usable,
    so Qwen2.5-VL-7B does not fit even on an otherwise empty 8 GB GPU."""
    assert BIG_VLM.vram_mib > USABLE_MIB
    with pytest.raises(InsufficientVram, match="qwen25vl7b"):
        plan(required=("qwen25vl7b",))


def test_a_24gb_budget_accommodates_the_7b_vlm_alongside_the_detector() -> None:
    """The planner is a function of the budget, not a hardcoded layout (spec §4.3)."""
    result = plan_residency(
        specs=SPECS,
        currently_resident=(),
        required=("yolo11s", "qwen25vl7b"),
        last_used_at={},
        now=0.0,
        total_mib=24_564,
        reserved_mib=2048,
    )
    assert set(result.resident) == {"yolo11s", "qwen25vl7b"}
    assert result.unload == ()
