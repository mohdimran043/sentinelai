"""Executes the Phase 1A `plan_residency()` output (spec §5.4): loads, unloads, and
applies the 600s VLM idle-unload via `sweep_idle()`.

`plan_residency()` is pure and takes the orchestrator's own bookkeeping (currently
resident, last-used) as arguments — this class is that bookkeeping plus the I/O
(`ModelRuntime.initialize`/`shutdown`) the plan calls for.

Freshening a model's idle clock is just calling `ensure()` again with that key in
`required`: `plan_residency` stamps `last_used_at` for every required key on every
call, so a caller invoking a model (e.g. wrapping a `VisionLanguageModel.describe`
call with `await resident_set.ensure((vlm_key,), now)` first) keeps it alive for as
long as it is genuinely being used, and `sweep_idle` — `ensure` with nothing
required — evicts it exactly `idle_unload_seconds` after the last such call.
"""

from __future__ import annotations

from sentinel_ai.domain.policy.vram_budget import plan_residency
from sentinel_ai.orchestrator.registry import ModelRegistry

__all__ = ["ResidentSet"]


class ResidentSet:
    def __init__(self, registry: ModelRegistry, total_mib: int, reserved_mib: int) -> None:
        self._registry = registry
        self._total_mib = total_mib
        self._reserved_mib = reserved_mib
        self._resident: set[str] = set()
        self._last_used_at: dict[str, float] = {}

    async def ensure(self, required: tuple[str, ...], now: float) -> None:
        """Apply plan_residency(): load required, evict what must go."""
        specs = {spec.model_key: spec for spec in self._registry.specs()}
        plan = plan_residency(
            specs=specs,
            currently_resident=tuple(self._resident),
            required=required,
            last_used_at=self._last_used_at,
            now=now,
            total_mib=self._total_mib,
            reserved_mib=self._reserved_mib,
        )
        for key in plan.unload:
            await self._registry.get(key).shutdown()
            self._resident.discard(key)
        for key in plan.load:
            runtime = self._registry.get(key)
            await runtime.initialize()
            await runtime.warmup()
            self._resident.add(key)
        for key in required:
            self._last_used_at[key] = now

    async def evict(self, key: str) -> bool:
        """Unload one model now, whatever the plan says. Returns whether it was resident.

        Spec §9's VLM-OOM row calls for an eviction before the retry, and
        `plan_residency` cannot express it: the planner only evicts to make room for a
        *load*, and after an OOM the model is already resident — from the planner's
        point of view nothing needs to change. Reloading it is nevertheless the one
        thing that helps, because `ModelRuntime.shutdown()` runs `gc.collect()` and
        `torch.cuda.empty_cache()`, which is what actually hands a fragmented
        allocator pool back to the driver.

        Bookkeeping stays here rather than in the caller so `_resident` and
        `_last_used_at` cannot drift from what is really on the card.
        """
        if key not in self._resident:
            return False
        await self._registry.get(key).shutdown()
        self._resident.discard(key)
        self._last_used_at.pop(key, None)
        return True

    def mark_unhealthy(self, key: str, detail: str) -> None:
        """Record an orchestrator-observed fault on a model's own health report.

        Routed through here rather than by handing the scheduler a `ModelRegistry`:
        the resident set is already the scheduler's one handle on model lifecycle,
        and a second one would be a second place that could disagree about which
        models exist.
        """
        self._registry.get(key).mark_unhealthy(detail)

    async def sweep_idle(self, now: float) -> None:
        """Idle-evict only: `ensure` with nothing required (spec §5.4's 600s VLM
        idle-unload)."""
        await self.ensure((), now)

    def resident(self) -> frozenset[str]:
        return frozenset(self._resident)
