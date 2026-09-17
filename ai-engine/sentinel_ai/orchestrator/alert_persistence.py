"""When the alert register is written to disk, and when it is not (spec §17).

The register changes for two very different reasons, and conflating them is what makes
a naive "save on every change" either lose data or hammer the disk.

* **A machine changed it.** An event opened an alert or bumped an occurrence count.
  This happens as often as the site is busy — a corridor at going-home time produced
  eleven alerts in forty-five seconds under measurement, and each recurrence is another
  change. Losing the last few seconds of occurrence counts to a hard kill costs
  approximately nothing: the events themselves are already published, and a count is
  recoverable context rather than a human decision.
* **A person changed it.** They acknowledged or resolved something. This happens rarely
  — a handful of times a minute at worst, because there is a human in the loop — and it
  is the one thing in the register that exists nowhere else. An acknowledgement lost to
  a restart is triage work done twice.

So the policy is asymmetric, and deliberately so: operator changes are flushed
**immediately and awaited**, machine changes are marked and flushed by a coalescing
timer. The write is O(alerts) either way, which is why the busy path is the one that
gets batched.

Why a revision counter rather than a dirty flag
-------------------------------------------------
`AlertRegister.revision` increments on every change. This coordinator records the
revision it last *successfully saved*, and flushes when the two differ. A boolean flag
would have to be cleared either before the write (losing a change that arrives during
it) or after (re-writing a set that has not changed). The counter has neither problem:
it is read before the snapshot is taken, so a change that lands during the write simply
leaves the numbers unequal and the next tick picks it up.

Failure is logged, never raised
---------------------------------
A full disk, a read-only mount, a permissions mistake. Every one of them means triage
state stops being durable, and not one of them is a reason to stop watching cameras.
The flush task logs once per transition into failure rather than once per tick, because
a message every two seconds is how an operator learns to ignore the log.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from sentinel_ai.orchestrator.alerts import AlertRegister
from sentinel_ai.ports.alert_store import AlertStore

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_FLUSH_INTERVAL_SECONDS", "AlertPersistence"]

DEFAULT_FLUSH_INTERVAL_SECONDS = 5.0
"""How long a machine-driven change may sit unwritten.

Five seconds bounds what a `kill -9` costs to five seconds of occurrence counts, and
keeps a busy site to twelve whole-set writes a minute instead of one per event. It does
not apply to operator actions, which do not wait.
"""


class AlertPersistence:
    """Restores the register at startup and keeps the store in step with it."""

    def __init__(
        self,
        register: AlertRegister,
        store: AlertStore,
        *,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    ) -> None:
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        self._register = register
        self._store = store
        self._interval = flush_interval_seconds
        self._saved_revision = -1
        self._task: asyncio.Task[None] | None = None
        self._failing = False

    async def restore(self) -> int:
        """Load the saved set into the register. Returns how many alerts came back.

        Called once, before any camera starts. `AlertStore.load` does not raise, so a
        first start, an unreadable file and a file from an incompatible build all arrive
        here as an empty tuple and all mean the same thing: begin with nothing.
        """
        alerts = await self._store.load()
        if alerts:
            self._register.restore(alerts)
            logger.info(
                "restored %d alert(s) from the store; %d still open",
                len(alerts),
                sum(1 for alert in alerts if alert.is_open),
            )
        self._saved_revision = self._register.revision
        return len(alerts)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="alert-persistence")

    async def stop(self) -> None:
        """Cancel the timer, then write once more.

        The final flush is the point of stopping cleanly: an operator who acknowledged
        something a second before the shutdown signal has their work written down,
        rather than losing it to the interval they happened to land in.
        """
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.flush()

    async def flush(self) -> bool:
        """Write the register if it has changed. Returns whether anything was written.

        Awaited by the API handlers for acknowledge and resolve, so a 200 on those
        endpoints means the decision is on disk rather than merely in memory.
        """
        revision = self._register.revision
        if revision == self._saved_revision:
            return False
        try:
            await self._store.save(self._register.snapshot())
        except Exception as error:
            if not self._failing:
                self._failing = True
                logger.error(
                    "alert triage state is no longer being persisted; acknowledgements "
                    "will not survive a restart until this is fixed: %s",
                    error,
                )
            return False
        if self._failing:
            self._failing = False
            logger.info("alert triage state is being persisted again")
        self._saved_revision = revision
        return True

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.flush()
