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

`store`'s parameter is typed `Event | WelfareNote`, wider than the
`FailedEventSink` port it implements (`event: Event`). Task 6's webhook
notifier needs exactly this disk-backed, write-then-rename spool for a
`WelfareNote` that a webhook endpoint would not take, and the brief for that
task is explicit: reuse this writer rather than build a second one — the
mechanics below (`_best_effort`'s generic `asdict`, the timestamped filename
keyed off `event_id`) already have nothing `Event`-specific about them. The
port itself stays `Event`-only on purpose (it is `VlmScheduler`'s contract,
and `Notifier`/`WelfareNote` are deliberately not folded into `EventPublisher`
for a second purpose — see `ports/notifier.py`'s module docstring); widening
only the concrete override is a contravariant, LSP-legal change that every
caller going through the narrower port interface never observes.

Because the spool now holds two record shapes with nothing else forcing a
reader to tell them apart, `store` writes a `record_type` field (`"Event"` or
`"WelfareNote"`, from `type(event).__name__`) into every record: this file's
own docstring calls the output "for an operator (or a repair script)," and a
repair script written against one shape would `KeyError` on a field the
other does not have.
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
from sentinel_ai.ports.notifier import WelfareNote

logger = logging.getLogger(__name__)

__all__ = ["DeadLetterSpool"]


class DeadLetterSpool(FailedEventSink):
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    async def store(self, event: Event | WelfareNote, error: BaseException) -> None:
        self._seq += 1
        base = f"{time.time_ns():020d}-{self._seq:08d}-{event.event_id.hex}"
        record = {
            "recorded_at_ns": time.time_ns(),
            # The spool holds two record shapes now — `Event` from the
            # publisher path, `WelfareNote` from Task 6's webhook notifier —
            # distinguished only by which fields happen to be present
            # otherwise. This module's own docstring calls the output "for
            # an operator (or a repair script)"; a repair script written
            # against the `Event` shape would `KeyError` on `reason` for a
            # welfare note it did not know to expect. `record_type` makes
            # the shape explicit instead of something a reader has to infer.
            "record_type": type(event).__name__,
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


def _best_effort(event: Event | WelfareNote) -> dict[str, Any]:
    """`asdict` with a guaranteed fallback.

    `Event` and `WelfareNote` are both plain frozen dataclasses, but this is the
    failure path: if a future field ever makes `asdict` raise, losing the record to
    that would repeat the very defect this module exists to close.
    """
    try:
        return asdict(event)
    except Exception:  # pragma: no cover - defensive, see docstring
        return {"repr": repr(event)}
