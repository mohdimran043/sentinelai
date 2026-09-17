"""Per-model VRAM accounting — the arithmetic, without a GPU.

This exists because the bug it guards against shipped. Every adapter measured itself
with `torch.cuda.memory_reserved()`, which is **process-wide**, so each model reported
the cumulative total of everything loaded before it. A live `/health` reported the pose
model at 2816 MiB — the VLM's footprint — for a model that costs about 460.

Two models made it look like the documented conservatism. Three made it obvious. These
tests pin the shape of the fix so a fourth model cannot quietly reintroduce it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from sentinel_ai.adapters.vram import (
    CUDA_CONTEXT_OVERHEAD_MIB,
    device_used_mib,
    measure_foreign_vram_mib,
    measure_model_vram_mib,
    reserved_mib,
)


@contextmanager
def at_pools(
    reserved: int, allocated: int | None = None, peak: int | None = None
) -> Iterator[None]:
    """Pretend the process-wide pools currently hold these totals.

    `allocated` defaults to `reserved` so the tests that only care about the reserved
    delta read cleanly; the ones about allocator slack set it explicitly.
    """
    value = (reserved, reserved if allocated is None else allocated)
    with (
        patch("sentinel_ai.adapters.vram.sample_pools", return_value=value),
        # Peaked at whatever is currently live unless a test says otherwise, so the
        # third term never silently dominates a case about the other two.
        patch("sentinel_ai.adapters.vram.peak_allocated_mib", return_value=peak or 0),
    ):
        yield


class TestMeasurement:
    def test_a_model_is_charged_its_own_growth_not_the_whole_pool(self) -> None:
        """The bug, in one assertion. Loading second into a pool already holding 3000
        MiB and growing it to 3460 costs 460, not 3460."""
        with at_pools(reserved=3460, allocated=3000):
            assert measure_model_vram_mib(3000, allocated_baseline_mib=3000) == 460

    def test_the_first_model_is_also_charged_the_cuda_context(self) -> None:
        """A baseline of zero means this model found an empty pool and therefore
        created the context. Charged once, here, rather than to every model — which
        would be a smaller version of the same over-count."""
        with at_pools(reserved=152, allocated=68):
            assert measure_model_vram_mib(0) == 152 + CUDA_CONTEXT_OVERHEAD_MIB

    def test_later_models_are_not_charged_the_context_again(self) -> None:
        """Three models must not report three CUDA contexts when the process has one."""
        with at_pools(reserved=3460, allocated=3000):
            second = measure_model_vram_mib(3000, allocated_baseline_mib=3000)
        with at_pools(reserved=3920, allocated=3460):
            third = measure_model_vram_mib(3460, allocated_baseline_mib=3460)
        assert second == 460
        assert third == 460

    def test_the_parts_sum_to_the_whole(self) -> None:
        """The property `plan_residency()` depends on: every reserved byte accounted to
        exactly one model, plus one context. Over-counting evicts models that fitted;
        under-counting admits a set that does not."""
        with at_pools(reserved=452, allocated=200):
            detector = measure_model_vram_mib(0)
        with at_pools(reserved=3268, allocated=3000):
            vlm = measure_model_vram_mib(452, allocated_baseline_mib=200)
        with at_pools(reserved=3730, allocated=3400):
            pose = measure_model_vram_mib(3268, allocated_baseline_mib=3000)
        assert detector + vlm + pose == 3730 + CUDA_CONTEXT_OVERHEAD_MIB

    def test_a_shrinking_pool_never_reports_negative_vram(self) -> None:
        """Another model can be evicted between the two samples and return its memory
        to the allocator. A negative figure would make the planner believe a model
        frees memory by existing."""
        with at_pools(reserved=1000, allocated=800):
            assert measure_model_vram_mib(3000, allocated_baseline_mib=2500) == 0

    def test_no_cuda_reports_zero_rather_than_the_context_overhead(self) -> None:
        """On a CPU box there is no VRAM to cost. Reporting 300 MiB would give
        `plan_residency()` a budget to spend against hardware that has none."""
        with at_pools(reserved=0, allocated=0):
            assert measure_model_vram_mib(0) == 0


class TestAllocatorSlack:
    """The second bug, found by the live engine rather than by a test.

    Loading third, the pose model reported 0 MiB: the caching allocator had enough slack
    freed by the VLM's warmup to place it without ever growing the reserved pool.
    """

    def test_a_model_placed_entirely_inside_slack_is_still_charged_its_weights(
        self,
    ) -> None:
        # Reserved unchanged at 3460 — the allocator asked the driver for nothing — but
        # 162 MiB of new live tensors appeared, which are this model's weights.
        with at_pools(reserved=3460, allocated=1200):
            assert measure_model_vram_mib(3460, allocated_baseline_mib=1038) == 162

    def test_zero_is_never_reported_for_a_model_that_loaded(self) -> None:
        """Under-reporting is the dangerous direction: an over-reporting model makes
        the planner evict something it did not need to, an under-reporting one makes it
        admit a set that does not fit."""
        with at_pools(reserved=3460, allocated=1200):
            assert measure_model_vram_mib(3460, allocated_baseline_mib=1038) > 0

    def test_the_peak_during_warmup_wins_when_it_is_the_largest(self) -> None:
        """Weights alone are not a model's cost. With only live-tensor and reserved
        deltas, the pose model reported 11 MiB against a 462 MiB standalone
        measurement — its weights exactly, and nothing for the workspace a forward pass
        needs."""
        with at_pools(reserved=3460, allocated=1049, peak=1500):
            assert measure_model_vram_mib(3460, allocated_baseline_mib=1038) == 462

    def test_reserved_growth_still_wins_when_it_is_the_larger(self) -> None:
        """Allocated is the floor, not the answer. A model that forced the driver to
        hand over more memory is charged that, because the workspace is real."""
        with at_pools(reserved=4000, allocated=1100):
            assert measure_model_vram_mib(3460, allocated_baseline_mib=1038) == 540


class TestDeviceGuard:
    """`Settings.device="cpu"` is a supported configuration on a box that *has* a GPU —
    forcing CPU is a real thing an operator does. These call the real functions, not a
    patched `sample_pools`, because the bug they pin was in the CUDA guard itself."""

    def test_an_explicitly_non_cuda_device_reports_zero_rather_than_raising(self) -> None:
        """`torch.cuda.memory_reserved("cpu")` raises `ValueError`. Guarding only on
        `torch.cuda.is_available()` is not enough when the GPU exists but this model
        was pinned to the CPU."""
        assert reserved_mib("cpu") == 0
        assert measure_model_vram_mib(0, "cpu") == 0

    def test_a_cpu_pinned_model_does_not_report_the_gpu_s_memory(self) -> None:
        """The failure the sentinel exists to prevent: collapsing "cpu" to `None` would
        make a CPU-pinned model report the default GPU's pool as its own footprint."""
        assert measure_model_vram_mib(0, "cpu") == 0


@contextmanager
def at_device_use(used_mib: int) -> Iterator[None]:
    """Pretend the *driver* reports this much VRAM in use across the whole device."""
    with patch("sentinel_ai.adapters.vram.device_used_mib", return_value=used_mib):
        yield


class TestForeignAllocators:
    """A second instrument, for the models torch's allocator cannot see.

    The face pipeline runs under ONNX Runtime, which allocates through CUDA directly.
    Measured with `sample_pools` it reported **0 MiB** while holding 608 — the same
    class of failure as the bug at the top of this file, and the worse direction of it:
    an over-report makes `plan_residency()` evict something it did not need to, an
    under-report makes it admit a set that does not fit.
    """

    def test_a_foreign_model_is_charged_the_device_wide_growth(self) -> None:
        """The measured shape: 2871 MiB on the card before, 3353 after."""
        with at_device_use(3353):
            assert measure_foreign_vram_mib(2871) == 482

    def test_no_cuda_reports_nothing_rather_than_the_whole_baseline(self) -> None:
        """A device reading of 0 means there is no CUDA to read, not that the card
        emptied. Subtracting a baseline from it would charge the model a negative
        footprint, and clamping that to 0 would be right for the wrong reason."""
        with at_device_use(0):
            assert measure_foreign_vram_mib(2871) == 0

    def test_a_model_cannot_free_memory_by_existing(self) -> None:
        """Another process releasing memory between the two samples must not make this
        model's footprint negative — `plan_residency()` would treat it as headroom."""
        with at_device_use(2400):
            assert measure_foreign_vram_mib(2871) == 0

    def test_the_context_is_not_charged_twice(self) -> None:
        """Unlike `measure_model_vram_mib`, this one never adds
        `CUDA_CONTEXT_OVERHEAD_MIB`. A device-wide reading already contains every
        context on the card, so adding one would double-count it — and on the engine's
        own startup path the torch models have created it already."""
        with at_device_use(312):
            assert measure_foreign_vram_mib(0) == 312

    def test_a_cpu_device_is_not_asked_about_vram(self) -> None:
        """`device="cpu"` on a box that has a GPU is a supported configuration, and the
        answer is 0 rather than the card's total — which is what the whole device
        reading would otherwise return."""
        assert device_used_mib("cpu") == 0
