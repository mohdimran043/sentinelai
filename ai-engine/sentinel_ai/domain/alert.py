"""What an operator is actually shown (spec §17).

An `Event` is what happened. An `Alert` is what somebody is asked to *do something
about*, and the two are not one-to-one — which is the entire reason this type exists.

Measured on real footage: one camera watching a corridor for forty-five seconds
produced eleven events, ten of them `zone_intrusion` as successive pedestrians crossed
into a restricted area. Eleven rows in an operator's list, arriving faster than they
can be read, is not eleven times as useful as one row saying "ten people have entered
the stairwell". §17 puts it as a requirement: twenty seconds of the same unauthorised
person must produce **one** alert with `occurrences: 17`, not a hundred alerts.

So an alert is an **episode**: a first sighting, a last sighting, a count, and a state
an operator moves it through. Events flow into it; it does not flow back.

The state machine, and why `RESOLVED` is not a synonym for "over"
------------------------------------------------------------------
    ACTIVE ──acknowledge──▶ ACKNOWLEDGED ──resolve──▶ RESOLVED
       └──────────────────resolve───────────────────────▶

`ACKNOWLEDGED` means *a person has seen this*. It is the state that matters most,
because it is the only one that distinguishes "nobody has looked" from "somebody is
handling it" — and on a wall of alerts that distinction is the difference between an
operator triaging and an operator re-reading the same row every thirty seconds.

`RESOLVED` means a person has said it is finished. Nothing resolves an alert
automatically, and that is deliberate: a fall alert that aged out on its own would
leave no trace that nobody ever went to look. An alert that stops recurring simply
stops accruing occurrences and drifts down the list by `last_seen`.

Pure: no clock, no I/O, no UUID generation — `alert_id` arrives from the caller, for
`domain/`'s usual reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from uuid import UUID

from sentinel_ai.domain.entities import EscalationReason, Event, Severity
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.zone import Zone

__all__ = ["MAX_EVENT_IDS", "Alert", "AlertKey", "AlertState"]

MAX_EVENT_IDS = 50
"""How many contributing event ids one alert keeps.

Bounded because an alert that recurs for an hour would otherwise grow a list nobody
reads and every serialisation carries. The **first** ones are kept rather than the
last: an investigator works backwards from the start of an episode, and the oldest
event is the one whose clip shows how it began. `occurrences` stays exact regardless,
so the count never lies even once the ids stop being complete.
"""


class AlertState(StrEnum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class AlertKey:
    """What makes two events the same ongoing situation.

    Three parts, and `subject` is the one that took thought. Keying on camera and
    reason alone would merge ten different people entering a restricted area into one
    alert saying "somebody is in the stairwell", which is true and useless — an
    operator needs to know it is ten people. Keying on the event id would merge
    nothing.

    `subject` is therefore the tracked identity the event is about: whoever the
    behaviour belongs to. §17's worked example — one unauthorised person seen
    seventeen times in twenty seconds — is exactly one subject recurring, and it
    collapses to one alert. Ten people each seen once is ten subjects and ten alerts,
    which is the honest answer to a different question.

    Camera-level findings have no subject at all (`""`): there is nobody to attribute a
    covered lens to, so every tamper report from one camera is one episode.
    """

    camera_id: str
    reason: EscalationReason
    subject: str

    @classmethod
    def from_event(cls, event: Event) -> AlertKey:
        """Derive the key. Track ids are sorted so the same two people in either order
        are the same subject, and joined rather than kept as a tuple so the key stays
        a flat, hashable, printable thing a console can pass back in a URL."""
        return cls(
            camera_id=event.camera_id,
            reason=event.reason,
            # `subject_track_ids`, never `track_ids` — see that field on `Event` for
            # the measured failure that distinction exists to prevent.
            subject=",".join(str(track_id) for track_id in sorted(event.subject_track_ids)),
        )


@dataclass(frozen=True, slots=True)
class Alert:
    """One ongoing situation, and everything an operator needs to triage it."""

    alert_id: UUID
    key: AlertKey
    state: AlertState
    severity: Severity
    priority: EventPriority

    first_seen: float
    last_seen: float
    """Unix epoch seconds, as `Event.occurred_at` is — sortable across cameras and
    across restarts, which `source_timestamp` is not."""

    occurrences: int
    description: str
    """The most recent contributing event's description. Most recent rather than first
    because an episode develops: "a person is lying on the floor" is more use to
    somebody deciding whether to go than "a person appears to have fallen" was thirty
    seconds ago."""

    camera_label: str
    zone: Zone | None
    event_ids: tuple[UUID, ...]
    clip_uri: str | None = None
    """The first contributing clip that finished. First rather than latest for
    `MAX_EVENT_IDS`' reason: an investigator wants to see how it started."""

    notify_clip_uri: str | None = None
    """The same moment, trimmed to `SENTINEL_NOTIFY_CLIP_SECONDS`, when the writer made
    one — see `ClipHandle.notify_uri`.

    Carried beside `clip_uri` rather than replacing it because they answer different
    questions and an operator asks both. A console triaging a wall of rows wants the
    short one, which plays inline and is over before attention moves on; somebody who
    has chosen a row wants the long one, with the seconds before and after.

    Paired with `clip_uri`, never independent of it: it is the *same* recording trimmed,
    so it tracks whichever clip `clip_uri` settled on. `None` is ordinary — the writer
    may be configured not to make one, or the trim may have failed, and in both cases
    the full clip is still there.
    """

    acknowledged_by: str | None = None
    acknowledged_at: float | None = None

    def __post_init__(self) -> None:
        if self.occurrences < 1:
            raise ValueError("an alert exists because something happened; occurrences must be >= 1")
        if self.last_seen < self.first_seen:
            raise ValueError(
                f"last_seen ({self.last_seen}) precedes first_seen ({self.first_seen})"
            )

    @property
    def is_open(self) -> bool:
        """Whether this alert can still absorb new events.

        A resolved alert cannot: a person said it was finished, and quietly reopening
        it would erase that judgement. A recurrence after resolution is a new episode,
        which is the truthful reading — it happened again.
        """
        return self.state is not AlertState.RESOLVED

    def acknowledged(self, *, by: str, at: float) -> Alert:
        """A person has seen this.

        Acknowledging a resolved alert is refused rather than ignored: it means the
        operator is looking at a stale list, and silently accepting would tell them
        they had done something they had not.

        **The first acknowledgement wins, and a second one changes nothing.** Two
        consoles watching the same wall will both acknowledge the same row, and
        last-write-wins would make `acknowledged_at` drift later every time somebody
        looked — turning "when did this stop being unseen", which is the fact the wall
        is built on, into "when did somebody last click". Idempotent rather than
        refused, because the second operator has not made a mistake: they are looking
        at a live list and the alert really is acknowledged.
        """
        if self.state is AlertState.RESOLVED:
            raise ValueError(
                "this alert is already resolved; acknowledging it would change nothing"
            )
        if self.state is AlertState.ACKNOWLEDGED:
            return self
        return replace(self, state=AlertState.ACKNOWLEDGED, acknowledged_by=by, acknowledged_at=at)

    def resolved(self) -> Alert:
        """A person has said this is finished. Idempotent — resolving twice is not an
        error, because two operators closing the same row is an ordinary race rather
        than a mistake either of them made."""
        return replace(self, state=AlertState.RESOLVED)
