"""The alert register: where events become things an operator acts on (spec §17).

`domain/policy/alerting.py` decides *whether* an event joins an existing alert. This
holds the alerts while that decision is being made, and it is deliberately the only
mutable part of the subsystem — the rule stays pure and exhaustively tested, and what
lives here is bookkeeping.

In memory, but no longer only in memory
-----------------------------------------
This class is still the working set: sorted, bounded, evicted, and rebuilt from nothing
on every start. What changed is where "nothing" comes from. Given an `AlertStore`
(`ports/alert_store.py`), `orchestrator/alert_persistence.py` restores the saved set
into this register at startup and writes it back as it changes, so **an acknowledgement
now survives a restart**. An operator who acknowledged twenty alerts and then saw the
engine restart sees twenty acknowledged alerts.

Without a store it behaves exactly as it did before, which is the supported
configuration for a test and for a deployment that does not want the file. The register
itself does no I/O and knows nothing about the store — it exposes `restore()` and a
`revision` counter, and the coordinator does the rest. That is what keeps this class
synchronous and exhaustively testable without a filesystem.

The bargain that has *not* changed: this is not the record of what happened. That is
the anomaly event published to RabbitMQ. This is the record of what a human did about
it.

Bounded, and the eviction order is chosen rather than incidental
-----------------------------------------------------------------
A register that grew without limit is a memory leak on a camera with a stuck detector.
When it is full, the **oldest resolved** alert goes first, then the oldest
acknowledged, and an unacknowledged `ACTIVE` alert is evicted only when there is
nothing else left. Dropping by age alone would throw away the one alert nobody has
looked at in favour of one somebody already closed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from uuid import UUID, uuid4

from sentinel_ai.domain.alert import Alert, AlertKey, AlertState
from sentinel_ai.domain.entities import Event, Severity
from sentinel_ai.domain.policy.alerting import (
    DEFAULT_MERGE_WINDOW_SECONDS,
    DEFAULT_MIN_ALERT_SEVERITY,
    is_alertable,
    merged,
    opened,
    should_merge,
)
from sentinel_ai.domain.policy.priority import rank
from sentinel_ai.domain.zone import Zone

logger = logging.getLogger(__name__)

__all__ = ["ALERT_REGISTER_CAPACITY", "AlertRegister", "UnknownAlertError"]

ALERT_REGISTER_CAPACITY = 500
"""How many alerts are held before the oldest closed one is evicted.

Five hundred is far more than an operator will ever scroll and far less than a stuck
camera can produce in an afternoon. It exists to bound memory, not to curate.
"""

_EVICTION_ORDER: dict[AlertState, int] = {
    AlertState.RESOLVED: 0,
    AlertState.ACKNOWLEDGED: 1,
    AlertState.ACTIVE: 2,
}
"""Lower goes first. An unacknowledged alert is the last thing to be thrown away,
because it is the only one nobody has looked at."""


class UnknownAlertError(KeyError):
    """No alert with this id is in the register.

    A `KeyError` subclass so it reads naturally at the lookup site, with `alert_id`
    carried separately: `KeyError.__str__` is `repr(args[0])`, which would put literal
    quotes on the wire if an API handler passed `str(exc)` through — the same trap
    `UnknownCameraError` documents.
    """

    def __init__(self, alert_id: UUID) -> None:
        self.alert_id = alert_id
        super().__init__(f"unknown alert: {alert_id}")


class AlertRegister:
    """Every open and recently-closed alert this process knows about."""

    def __init__(
        self,
        *,
        capacity: int = ALERT_REGISTER_CAPACITY,
        merge_window_seconds: float = DEFAULT_MERGE_WINDOW_SECONDS,
        minimum_severity: Severity = DEFAULT_MIN_ALERT_SEVERITY,
        new_alert_id: Callable[[], UUID] = uuid4,
    ) -> None:
        if capacity < 1:
            raise ValueError("an alert register with no room for an alert cannot hold one")
        self._capacity = capacity
        self._merge_window_seconds = merge_window_seconds
        self._minimum_severity = minimum_severity
        # Injected so a test can assert on an id rather than discovering it. The
        # domain refuses to generate one at all (`domain/` may not import `uuid4`),
        # which is what pushed the decision out to here.
        self._new_alert_id = new_alert_id
        # Insertion-ordered: `dict` preserves it, and eviction relies on it for the
        # age half of the ordering.
        self._alerts: dict[UUID, Alert] = {}
        self._by_key: dict[AlertKey, UUID] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        """Bumped by every change to the register's contents. Never reset.

        A counter rather than a boolean `dirty` flag, so the persistence coordinator
        can record the revision it *saved* and compare. A flag would have to be cleared
        before the write completes — and a change arriving during that write would then
        be lost until the next unrelated one, which is precisely an acknowledgement that
        silently fails to persist.
        """
        return self._revision

    def restore(self, alerts: Sequence[Alert]) -> None:
        """Seed an empty register from a store. Not a merge.

        Refuses a non-empty register rather than reconciling: restoring happens once, at
        startup, before any camera runs. A merge would mean deciding whether a stored
        alert or a live one wins, a question with no good answer and no caller.

        Eviction still applies, so a stored set larger than this register's capacity is
        trimmed by the same rule as a live one — closed before open, oldest first.
        Insertion order is the caller's, and `snapshot()` sorts anyway.
        """
        if self._alerts:
            raise RuntimeError("restore() is for an empty register; this one already has alerts")
        for alert in alerts:
            self._alerts[alert.alert_id] = alert
            # Only an open alert may keep absorbing. A restored resolved alert is
            # readable and inert, exactly as it was before the restart — and rebinding
            # the key to it would let a recurrence merge into something a person had
            # already closed.
            if alert.is_open:
                self._by_key[alert.key] = alert.alert_id
        self._evict_if_full()
        self._revision += 1

    def absorb(
        self,
        event: Event,
        *,
        camera_label: str,
        zone: Zone | None,
        now: float,
        notify_clip_uri: str | None = None,
    ) -> Alert | None:
        """Fold `event` into the register. Returns the alert it landed on, or `None`.

        `None` means the event was not worth an operator's attention — most escalations
        are periodic summaries of empty corridors, and a list that includes them is a
        list nobody reads (`is_alertable`). The event is still published and still
        searchable; it simply does not become a row.

        `notify_clip_uri` is the short copy of this event's clip, when the writer made
        one. It is not on the event and cannot be — see `opened`.
        """
        if not is_alertable(event, minimum=self._minimum_severity):
            return None

        key = AlertKey.from_event(event)
        existing_id = self._by_key.get(key)
        existing = self._alerts.get(existing_id) if existing_id is not None else None

        if existing is not None and should_merge(
            existing, event, now=now, window_seconds=self._merge_window_seconds
        ):
            updated = merged(existing, event, now=now, notify_clip_uri=notify_clip_uri)
            self._alerts[existing.alert_id] = updated
            self._revision += 1
            return updated

        alert = opened(
            event,
            alert_id=self._new_alert_id(),
            camera_label=camera_label,
            zone=zone,
            now=now,
            notify_clip_uri=notify_clip_uri,
        )
        self._alerts[alert.alert_id] = alert
        # The key now points at the new episode. The previous alert stays in the
        # register and stays readable; it simply stops absorbing.
        self._by_key[key] = alert.alert_id
        self._evict_if_full()
        self._revision += 1
        return alert

    def get(self, alert_id: UUID) -> Alert:
        try:
            return self._alerts[alert_id]
        except KeyError:
            raise UnknownAlertError(alert_id) from None

    def acknowledge(self, alert_id: UUID, *, by: str, at: float) -> Alert:
        """Record that a person has seen this.

        A `ValueError` from the domain (acknowledging something already resolved) is
        allowed to propagate: it means the operator is acting on a stale list, and the
        API turns it into a 409 rather than a silent success.
        """
        updated = self.get(alert_id).acknowledged(by=by, at=at)
        self._alerts[alert_id] = updated
        self._revision += 1
        return updated

    def resolve(self, alert_id: UUID) -> Alert:
        updated = self.get(alert_id).resolved()
        self._alerts[alert_id] = updated
        self._revision += 1
        return updated

    def clear(self) -> int:
        """Drop every alert. Returns how many went.

        A blunt instrument, and deliberately the only one: there is no "clear the
        resolved ones", because an operator who wants a clean list wants a clean list,
        and a partial clear leaves them doing it twice.

        It throws away triage state, not evidence. Every event behind these alerts is
        already published and durable; what is lost is which of them a human had looked
        at — so this is a decision, and the console asks twice before making it.
        """
        count = len(self._alerts)
        self._alerts.clear()
        self._by_key.clear()
        self._revision += 1
        logger.info("cleared %d alert(s) from the register", count)
        return count

    def snapshot(self) -> tuple[Alert, ...]:
        """Every alert, worst first, then most recent first.

        Sorted here rather than by the caller so that every reader — the API, a future
        console, a test — sees the same order, and so that "what is at the top of the
        list" is a property of this class rather than of whoever asked.
        """
        return tuple(
            sorted(
                self._alerts.values(),
                key=lambda alert: (-rank(alert.priority), -alert.last_seen),
            )
        )

    def open_alerts(self) -> tuple[Alert, ...]:
        return tuple(alert for alert in self.snapshot() if alert.is_open)

    def _evict_if_full(self) -> None:
        while len(self._alerts) > self._capacity:
            victim = min(
                self._alerts.values(),
                # Closed before open; within a state, oldest first. `first_seen` rather
                # than `last_seen` so a long-running episode is not protected by its own
                # recurrences.
                key=lambda alert: (_EVICTION_ORDER[alert.state], alert.first_seen),
            )
            del self._alerts[victim.alert_id]
            self._revision += 1
            if self._by_key.get(victim.key) == victim.alert_id:
                del self._by_key[victim.key]
            logger.debug(
                "alert register full (%d); evicted %s alert %s",
                self._capacity,
                victim.state.value,
                victim.alert_id,
            )
