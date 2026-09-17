"""Where alerts live across a restart (spec §17).

The register in `orchestrator/alerts.py` is process memory and always will be — it is
the working set, sorted and evicted for an operator looking at a screen right now. This
port is the other half: the thing that makes an acknowledgement mean something after
the engine is restarted.

Why a whole-set `save` rather than per-alert writes
----------------------------------------------------
The register is bounded at `SENTINEL_ALERT_REGISTER_CAPACITY` (500 by default), so the
entire set is a few hundred kilobytes. Writing all of it makes the store a pure
function of the register, which removes the failure mode that per-row persistence
invites: a store that has some of the updates, in an order nobody chose, after a
crash between two writes. There is no incremental state to get out of step.

The cost is real and bounded: a save is O(alerts), so it is done on a coalescing timer
for machine-driven changes and immediately for operator ones. `orchestrator/
alert_persistence.py` owns that policy; this port just does what it is told.

What a store may assume
------------------------
`save` receives the complete set and replaces whatever was there. It must be atomic
against a reader — a process killed mid-save must leave either the old set or the new
one, never a truncated file — because the reader is this same engine on its next start,
and an alert list that fails to parse would take the engine down over bookkeeping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from sentinel_ai.domain.alert import Alert

__all__ = ["AlertStore"]


class AlertStore(ABC):
    @abstractmethod
    async def load(self) -> tuple[Alert, ...]:
        """Everything saved by the last successful `save`, or `()` for a fresh site.

        **Must not raise on a missing or unreadable store.** A site starting for the
        first time has no file, and that is not an error; neither is a file written by
        an older build whose shape has since changed. Both mean "no alerts to restore",
        which loses triage state and keeps the engine running — the right trade, since
        the durable record of what happened is the published event stream either way.
        """

    @abstractmethod
    async def save(self, alerts: Sequence[Alert]) -> None:
        """Replace the stored set. Atomic against a concurrent reader.

        May raise. The caller logs and carries on: losing durability is bad, and taking
        surveillance down because a disk filled up is worse.
        """

    @abstractmethod
    async def close(self) -> None: ...
