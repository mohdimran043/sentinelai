"""`LoggingNotifier` — the default `Notifier` (Task 5).

A deployment that configures no webhook still needs an observable trail, and
every later test of notification routing needs an adapter that touches no
network. This is that adapter: it does no I/O beyond the standard logging
module, so it can be the default `Notifier` wired up when nothing else is
configured, and the one every notifier-routing test builds on without
worrying about a real endpoint.

Fields are logged by name, one line per note, never the note as a whole:
`logger.info("%s", note)` (or `repr(note)`) would pass a naive "does it log"
test today but would silently start including anything added to `WelfareNote`
later, without anyone deciding that new field belongs in a log. Naming the
fields explicitly means a new field requires a deliberate edit here before it
appears anywhere.

`_summarize_concern` marks `evidence_stated=False` explicitly (see
`domain/welfare.py`'s `WelfareConcern.evidence_stated` docstring) rather than
folding it in silently: presenting a placeholder ("reported, not described")
as though the model had actually described what it saw would turn a muted
alarm into a confident-looking one, exactly the failure that field exists to
prevent.

`notify` never raises (the one exception to the port's own docstring, which
asks implementations to raise on failure): a notifier that throws would take
down the pipeline it exists to observe, and unlike a webhook, there is no
plausible failure mode here that a caller could usefully react to — the
failure would be in the note itself, not in delivery. A malformed note still
produces a log line, built from whatever fields can be read, rather than an
exception.
"""

from __future__ import annotations

import logging as _logging

from sentinel_ai.domain.welfare import WelfareConcern
from sentinel_ai.ports.notifier import Notifier, WelfareNote

logger = _logging.getLogger(__name__)

__all__ = ["LoggingNotifier"]


def _summarize_concern(concern: WelfareConcern) -> str:
    marker = "" if concern.evidence_stated else " (unevidenced)"
    return f"{concern.kind}:{concern.confidence}{marker}"


class LoggingNotifier(Notifier):
    async def notify(self, note: WelfareNote) -> None:
        try:
            concerns = ", ".join(_summarize_concern(c) for c in note.concerns)
            logger.info(
                "welfare notification: event_id=%s camera_id=%s severity=%s zone=%s "
                "concerns=[%s] clip_uri=%s",
                note.event_id,
                note.camera_id,
                note.severity,
                note.zone,
                concerns,
                note.clip_uri,
            )
        except Exception:
            # Never raise: this notifier exists to observe the pipeline, not to
            # gate it. A note malformed enough to break formatting above still
            # gets a line, built from whatever can be read off it directly,
            # rather than an exception the caller would have to guard against.
            logger.error(
                "welfare notification for event_id=%s could not be formatted",
                getattr(note, "event_id", "<unknown>"),
                exc_info=True,
            )
