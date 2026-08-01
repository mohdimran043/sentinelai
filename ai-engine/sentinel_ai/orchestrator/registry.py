"""Model discovery and the eight §5 lifecycle states (spec §5.4).

A thin lookup layer, deliberately: lifecycle truth lives in each runtime's own
`health()` (spec §10), so the registry never keeps a second, driftable copy of it.
"""

from __future__ import annotations

from sentinel_ai.domain.policy.vram_budget import ModelSpec
from sentinel_ai.ports.model_runtime import HealthReport, LifecycleState, ModelRuntime

__all__ = ["ModelRegistry", "ModelSpec"]
# ModelSpec is re-exported so callers import one name from one place; it is the
# domain type, not a copy (Reconciliation Log R2).


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._runtimes: dict[str, ModelRuntime] = {}

    def register(self, spec: ModelSpec, runtime: ModelRuntime) -> None:
        self._specs[spec.model_key] = spec
        self._runtimes[spec.model_key] = runtime

    def get(self, key: str) -> ModelRuntime:
        try:
            return self._runtimes[key]
        except KeyError:
            raise KeyError(f"unknown model key: {key}") from None

    def state(self, key: str) -> LifecycleState:
        return self.get(key).health().state

    def kind(self, key: str) -> str:
        """A model's kind lives on its runtime's `Capabilities`, not on `ModelSpec`
        (Reconciliation Log R2): `ModelSpec` has no `kind` field to keep the type
        `plan_residency()` already consumes untouched."""
        return self.get(key).capabilities().kind

    def specs(self) -> tuple[ModelSpec, ...]:
        return tuple(self._specs.values())

    def health(self) -> dict[str, HealthReport]:
        return {key: runtime.health() for key, runtime in self._runtimes.items()}
