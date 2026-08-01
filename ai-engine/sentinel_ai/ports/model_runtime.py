"""The §10 common model interface.

Every model implements exactly this, which is what makes new models
plug-and-play: the orchestrator never learns a concrete model type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class LifecycleState(StrEnum):
    """The eight states from spec §5."""

    LOADED = "loaded"
    UNLOADED = "unloaded"
    SLEEPING = "sleeping"
    DOWNLOADING = "downloading"
    UPDATING = "updating"
    OFFLINE = "offline"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthReport:
    state: LifecycleState
    detail: str = ""
    vram_mib: int = 0


@dataclass(frozen=True, slots=True)
class Capabilities:
    model_key: str
    kind: str
    labels: frozenset[str] = frozenset()
    vram_mib: int = 0
    batch_max: int = 1


class ModelRuntime(ABC):
    """Lifecycle contract for anything the orchestrator can load onto a GPU."""

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def predict(self, request: object) -> object: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def mark_unhealthy(self, detail: str) -> None:
        """Record a fault the *orchestrator* observed, so `health()` reflects it.

        A runtime sets `UNHEALTHY` itself for failures it can see from the inside —
        a load that raised, a checkpoint that would not download. It cannot see the
        ones that only make sense a layer out: spec §9's VLM-OOM row requires the
        model be marked unhealthy after a describe has run out of VRAM *twice*, and
        "twice" is knowledge the scheduler holds, not the runtime.

        Abstract rather than a defaulted no-op on purpose: a silently unimplemented
        `mark_unhealthy` would leave `/health` reporting a model as fine while the
        orchestrator had already given up on it, which is the state this exists to
        prevent.
        """

    @abstractmethod
    def health(self) -> HealthReport: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...
