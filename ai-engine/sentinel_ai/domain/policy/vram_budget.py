"""VRAM residency planning (spec §4.3, §34).

Pure: it decides what to load and unload. The orchestrator's resident set
executes the plan. Residency is a function of the budget, so the same planner
serves an 8 GB laptop GPU and a 24 GB 4090 without change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class InsufficientVram(RuntimeError):
    """A required model cannot fit even with everything evictable unloaded."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_key: str
    vram_mib: int
    priority: int
    """Higher survives eviction. The always-on detector should outrank the VLM."""
    idle_unload_seconds: float | None
    """None means never idle-evict."""


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    load: tuple[str, ...]
    unload: tuple[str, ...]
    resident: tuple[str, ...]
    free_mib: int


def plan_residency(
    *,
    specs: Mapping[str, ModelSpec],
    currently_resident: Sequence[str],
    required: Sequence[str],
    last_used_at: Mapping[str, float],
    now: float,
    total_mib: int,
    reserved_mib: int,
) -> ResidencyPlan:
    for key in (*required, *currently_resident):
        if key not in specs:
            raise KeyError(f"unknown model key: {key}")

    usable = total_mib - reserved_mib
    required_set = set(required)

    resident = list(currently_resident)
    unload: list[str] = []

    # Idle eviction first: free memory nobody is asking for.
    for key in list(resident):
        if key in required_set:
            continue
        spec = specs[key]
        if spec.idle_unload_seconds is None:
            continue
        if now - last_used_at.get(key, now) >= spec.idle_unload_seconds:
            resident.remove(key)
            unload.append(key)

    def used() -> int:
        return sum(specs[key].vram_mib for key in resident)

    for key in required:
        if key in resident:
            continue
        spec = specs[key]
        if spec.vram_mib > usable:
            raise InsufficientVram(
                f"model {key} needs {spec.vram_mib} MiB but only {usable} MiB is usable"
            )
        # Evict LRU among lower-or-equal-priority, non-required models until it fits.
        while used() + spec.vram_mib > usable:
            evictable = [
                candidate
                for candidate in resident
                if candidate not in required_set and specs[candidate].priority <= spec.priority
            ]
            if not evictable:
                raise InsufficientVram(
                    f"cannot fit {key} ({spec.vram_mib} MiB): "
                    f"{used()} MiB resident, {usable} MiB usable, nothing evictable"
                )
            victim = min(evictable, key=lambda k: (last_used_at.get(k, 0.0), k))
            resident.remove(victim)
            unload.append(victim)
        resident.append(key)

    return ResidencyPlan(
        load=tuple(key for key in required if key not in currently_resident),
        unload=tuple(unload),
        resident=tuple(resident),
        free_mib=usable - used(),
    )
