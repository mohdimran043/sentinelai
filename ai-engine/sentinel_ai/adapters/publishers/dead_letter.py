"""Disk-backed `FailedEventSink` (spec §9).

The publisher's spool is for events the broker will accept once it is back.
This is for the ones it never will: a payload that fails schema validation, a
spool directory that turns out not to be writable, an unexpected error inside
the transport. Those used to vanish — `VlmScheduler` released the admission
slot and moved on, and the event existed nowhere afterwards.

Serialisation is deliberately not `encode_event`. `encode_event` validates,
and a validation failure is one of the exact cases that lands here, so using
it would throw the evidence away for the same reason it was lost before.
This writes a best-effort record instead: every field it can, `repr` for
anything `json` cannot represent, plus the error that caused it. The result is
for an operator (or a repair script), not for the wire — nothing replays it
automatically, because nothing can know whether the underlying problem is
fixed.

Write-then-rename, same as the spool: a `.json` file is either absent or
complete.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.event_publisher import FailedEventSink

logger = logging.getLogger(__name__)

__all__ = ["DeadLetterSpool"]


class DeadLetterSpool(FailedEventSink):
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    async def store(self, event: Event, error: BaseException) -> None:
        self._seq += 1
        base = f"{time.time_ns():020d}-{self._seq:08d}-{event.event_id.hex}"
        record = {
            "recorded_at_ns": time.time_ns(),
            "error_type": type(error).__name__,
            "error": str(error),
            "event": _best_effort(event),
        }
        try:
            tmp_path = self._directory / f"{base}.json.tmp"
            tmp_path.write_text(json.dumps(record, default=_fallback), encoding="utf-8")
            tmp_path.replace(self._directory / f"{base}.json")
        except Exception:
            # Nothing is left to fall back to, so the log line is the last copy.
            # Never re-raised: this runs inside the escalation worker's own error
            # handling, and raising here would replace a lost event with a lost
            # event *and* a poisoned worker iteration.
            logger.error(
                "dead-letter write failed for event %s; the event is lost. record=%r",
                event.event_id,
                record,
                exc_info=True,
            )
            return
        logger.error(
            "event %s could not be published (%s); written to the dead-letter spool at %s",
            event.event_id,
            error,
            self._directory / f"{base}.json",
        )


def _fallback(obj: object) -> str:
    """How anything `json` cannot represent is written.

    `UUID` and the domain enums get their plain string form, so a dead-lettered record
    reads the same way the wire payload would and an operator can grep an event id
    across the spool, the logs and the broker. Everything else falls back to `repr`,
    which is lossy but never fails — this is the path that exists because the tidy one
    already did.
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return str(obj.value)
    return repr(obj)


def _best_effort(event: Event) -> dict[str, Any]:
    """`asdict` with a guaranteed fallback.

    `Event` is a plain frozen dataclass today, but this is the failure path: if a
    future field ever makes `asdict` raise, losing the record to that would repeat
    the very defect this module exists to close.
    """
    try:
        return asdict(event)
    except Exception:  # pragma: no cover - defensive, see docstring
        return {"repr": repr(event)}
