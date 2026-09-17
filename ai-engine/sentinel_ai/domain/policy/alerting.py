"""When an event joins an existing alert, and when it starts a new one (spec §17).

The whole of §17's deduplication requirement lives here, as three pure functions over
(alert, event, now). Pure for `domain/policy/escalation.py`'s reason: "the same person
seen seventeen times across twenty seconds, then again four minutes later" is a test
that runs in microseconds, and it is the only way anyone would ever write it.

The rule
---------
An event joins an open alert with the same `AlertKey` whose `last_seen` is within
`merge_window_seconds`. Otherwise it opens a new one.

A window rather than "any open alert with this key", because an alert has no natural
end. Nothing auto-resolves (see `domain/alert.py` on why), so without a window the
first `zone_intrusion` for a given person would absorb the one that happens the next
morning, and the console would show a single alert with `occurrences: 340` spanning a
week. The window is what makes an alert an *episode* rather than a category.

Not everything becomes an alert
--------------------------------
`is_alertable` is the other half. The gate escalates for seven automatic reasons, and
most of what comes back is a periodic summary of a corridor with nobody in it. Those
are events — worth recording, worth searching — and they are emphatically not alerts. A
list that includes them is a list an operator scrolls past, and §27's whole argument is
that the critical thing must be visible immediately.
"""

from __future__ import annotations

from uuid import UUID

from sentinel_ai.domain.alert import MAX_EVENT_IDS, Alert, AlertKey, AlertState
from sentinel_ai.domain.entities import Event, Severity
from sentinel_ai.domain.policy.priority import priority_of, severity_floor
from sentinel_ai.domain.zone import Zone

__all__ = [
    "DEFAULT_MERGE_WINDOW_SECONDS",
    "DEFAULT_MIN_ALERT_SEVERITY",
    "alert_severity",
    "is_alertable",
    "merged",
    "opened",
    "severity_rank",
    "should_merge",
]

DEFAULT_MERGE_WINDOW_SECONDS = 120.0
"""How long after its last sighting an alert still absorbs a recurrence.

Two minutes is long enough that a person loitering in and out of a restricted area is
one episode, and short enough that the same person tomorrow is a new one. It is the
knob that decides whether an operator sees an incident or a category.
"""

DEFAULT_MIN_ALERT_SEVERITY = Severity.LOW
"""The lowest severity worth putting in front of a person.

`LOW` rather than `INFO`, so a periodic summary of an empty corridor — which is what
most escalations are — stays an event and does not become a row. `severity_floor`
guarantees every structurally serious reason clears this bar regardless of what the
model said, so nothing important can be filtered out by a bad description.
"""


def alert_severity(event: Event) -> Severity:
    """The worse of what the model saw and what the reason structurally implies.

    Both, because each is blind to something the other sees. The model's threat score
    is the only thing that can notice that a routine periodic summary happens to show
    a fire. The reason's floor is the only thing that stops a mis-described keyframe
    demoting a suspected fall to `info` — several seconds of geometry watched a person
    go down and stay down, and one frame's opinion does not overturn that.
    """
    floor = severity_floor(event.reason)
    return max(event.threat.severity, floor, key=_SEVERITY_RANK.__getitem__)


def severity_rank(severity: Severity) -> int:
    """Severity as a number, for comparing two of them.

    `Severity` is a `StrEnum`, so `>` on it compares alphabetically — which puts
    `critical` below `low` and `high` below `medium`. Every comparison in this system
    goes through here instead, because that bug is silent and points the wrong way: it
    makes the most serious band look like the least.
    """
    return _SEVERITY_RANK[severity]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}
"""`Severity` is a `StrEnum`, so comparing members directly compares their *values*
alphabetically — `"critical" < "info"` is True, which would make `max` pick exactly
wrong. This table is the only ordering this type has here."""


def is_alertable(event: Event, *, minimum: Severity = DEFAULT_MIN_ALERT_SEVERITY) -> bool:
    """Whether this event is worth an operator's attention at all."""
    return _SEVERITY_RANK[alert_severity(event)] >= _SEVERITY_RANK[minimum]


def should_merge(
    alert: Alert,
    event: Event,
    *,
    now: float,
    window_seconds: float = DEFAULT_MERGE_WINDOW_SECONDS,
) -> bool:
    """Whether `event` continues `alert` rather than starting a new episode.

    `now` rather than `event.occurred_at`: the caller decides which clock the window is
    measured on, and passing the event's own timestamp is the usual choice. Keeping it
    a parameter is what lets a test ask "and four minutes later?" without constructing
    an event four minutes in the future.
    """
    return (
        alert.is_open
        and alert.key == AlertKey.from_event(event)
        and now - alert.last_seen <= window_seconds
    )


def opened(
    event: Event,
    *,
    alert_id: UUID,
    camera_label: str,
    zone: Zone | None,
    now: float,
    notify_clip_uri: str | None = None,
) -> Alert:
    """A new episode, from its first event.

    `alert_id` is supplied rather than generated: `domain/` may not import `uuid4` any
    more than it may read a clock, and an id chosen by the caller is also an id a test
    can assert on.

    `notify_clip_uri` arrives as an argument rather than off the event, because it is
    not on the event: the short clip is made by the writer and never published, so the
    only thing that can hand it over is the caller holding the clip handle.
    """
    return Alert(
        alert_id=alert_id,
        key=AlertKey.from_event(event),
        state=AlertState.ACTIVE,
        severity=alert_severity(event),
        priority=priority_of(event.reason),
        first_seen=now,
        last_seen=now,
        occurrences=1,
        description=event.description,
        camera_label=camera_label,
        zone=zone,
        event_ids=(event.event_id,),
        clip_uri=event.clip_uri,
        # Only alongside a clip. A short clip whose full recording never landed would
        # be an alert offering evidence that `clip_uri` says does not exist.
        notify_clip_uri=notify_clip_uri if event.clip_uri is not None else None,
    )


def merged(alert: Alert, event: Event, *, now: float, notify_clip_uri: str | None = None) -> Alert:
    """`alert`, having absorbed `event`.

    Severity **rises and never falls**. An episode that started as a `medium` speed
    anomaly and has since produced a `critical` reading is a critical episode, and
    letting the latest event set the severity outright would let one calm frame hide
    an escalating situation — the row would drop down an operator's list at the moment
    it most needed to be at the top.

    `state` is untouched. An acknowledged alert that recurs stays acknowledged: a person
    already knows, and re-raising it to `ACTIVE` would put it back in front of them for
    something they are in the middle of dealing with. The occurrence count and
    `last_seen` still move, so the console can show that it is ongoing without
    pretending it is unseen.
    """
    from dataclasses import replace

    return replace(
        alert,
        severity=max(alert.severity, alert_severity(event), key=_SEVERITY_RANK.__getitem__),
        last_seen=max(alert.last_seen, now),
        occurrences=alert.occurrences + 1,
        description=event.description,
        # First-N rather than last-N — see `MAX_EVENT_IDS`. An investigator works
        # backwards from the beginning of an episode.
        event_ids=(
            alert.event_ids
            if len(alert.event_ids) >= MAX_EVENT_IDS
            else (*alert.event_ids, event.event_id)
        ),
        # The first clip that finished, kept. A later clip shows the middle of an
        # episode; the first shows how it started.
        clip_uri=alert.clip_uri if alert.clip_uri is not None else event.clip_uri,
        # Moves with `clip_uri` and never on its own: the two are one recording at two
        # lengths, so a rule that let them come from different events would offer a
        # three-second preview of a moment other than the one the full clip shows.
        notify_clip_uri=(
            alert.notify_clip_uri
            if alert.clip_uri is not None
            else (notify_clip_uri if event.clip_uri is not None else None)
        ),
    )
