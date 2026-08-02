"""Welfare notification port — the AI Engine's channel to a human, not a console.

Everything upstream of this port (the escalation gate, the VLM's welfare
prompt in `adapters/vision/qwen25vl.py`, `domain/welfare.py`) exists to decide
*whether* something is worth a person's attention. Nothing before this task
could actually reach one: RabbitMQ, the in-memory publisher and the dead-letter
spool are all still just an `Event` sitting somewhere for a console to poll.
`Notifier` is the seam a later adapter (a webhook, a log line someone tails)
implements to close that gap, and the one Task 10's dispatch logic calls
against.

Deliberately not `EventPublisher` reused for a second purpose: a `Notifier`
carries a `WelfareNote`, a narrower, human-facing view of one welfare concern,
not the full `Event` the Web Platform persists. Conflating the two would force
every future notification channel to understand `Event`'s operational fields
(`metadata`, `description_unavailable`, `track_ids`) that have nothing to do
with alerting a person.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from sentinel_ai.domain.welfare import WelfareConcern


@dataclass(frozen=True, slots=True)
class WelfareNote:
    """What a human needs to see to act on one welfare concern.

    `clip_uri` is `str | None`, not `str`: a notification can legitimately
    precede a clip that failed to write — the clip pipeline and the
    notification pipeline fail independently (spec §9's degraded-event case
    for one, a full disk for the other). A note with no clip is still worth
    sending; for a welfare system, losing the alert itself because the
    evidence upload failed would be exactly backwards — the alert is what
    gets a person looked at, the clip is only corroboration for whoever
    responds. Every `Notifier` implementation must treat `clip_uri=None` as a
    normal, deliverable note, not a reason to withhold or delay it.

    `severity` is `str`, not `domain.entities.Severity`: it is the band this
    note was routed at, read by adapters that serialise it verbatim (a
    webhook payload, a log line) rather than reasoned about as domain policy.
    """

    event_id: UUID
    camera_id: str
    label: str
    zone: str | None
    occurred_at: float
    severity: str
    description: str
    concerns: tuple[WelfareConcern, ...]
    clip_uri: str | None


class Notifier(ABC):
    """Delivers a `WelfareNote` to a human. Every adapter does I/O — a
    webhook call, a log write — so the port is `async`; a synchronous one
    would force blocking calls onto the event loop the camera pipelines run
    on.
    """

    @abstractmethod
    async def notify(self, note: WelfareNote) -> None:
        """Deliver `note`.

        Implementations should raise on failure rather than swallow it: the
        caller (Task 10's dispatch) is responsible for deciding what a failed
        notification means for the pipeline, and it can only do that if the
        failure actually reaches it. A notifier that logged and returned
        would hide the failure from the one place that could still act on it.
        """
