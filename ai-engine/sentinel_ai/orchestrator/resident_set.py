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

    async def sweep_idle(self, now: float) -> None:
        """Idle-evict only: `ensure` with nothing required (spec §5.4's 600s VLM
        idle-unload)."""
        await self.ensure((), now)

    def resident(self) -> frozenset[str]:
        return frozenset(self._resident)
