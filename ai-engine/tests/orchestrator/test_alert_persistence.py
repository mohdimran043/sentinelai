"""The flush policy: what gets written immediately, what waits, and what never fails.

The asymmetry is the design (see `orchestrator/alert_persistence.py`). An occurrence
count is recoverable from the event stream; an acknowledgement exists nowhere else.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import uuid4

import pytest

from sentinel_ai.domain.alert import Alert, AlertState
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore
from sentinel_ai.domain.entities import Severity as Sev
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.alert_persistence import AlertPersistence
from sentinel_ai.orchestrator.alerts import AlertRegister
from sentinel_ai.ports.alert_store import AlertStore


class RecordingStore(AlertStore):
    def __init__(self, initial: Sequence[Alert] = (), fail: bool = False) -> None:
        self.saved: list[tuple[Alert, ...]] = []
        self.loads = 0
        self.fail = fail
        self._initial = tuple(initial)

    async def load(self) -> tuple[Alert, ...]:
        self.loads += 1
        return self._initial

    async def save(self, alerts: Sequence[Alert]) -> None:
        if self.fail:
            raise OSError("no space left on device")
        self.saved.append(tuple(alerts))

    async def close(self) -> None: ...


def an_event(reason: EscalationReason = EscalationReason.FALL_SUSPECTED, track: int = 1) -> Event:
    return Event(
        event_id=uuid4(),
        camera_id="cam-1",
        occurred_at=1_700_000_000.0,
        source_timestamp=1.0,
        reason=reason,
        threat=ThreatScore(value=0.9, severity=Sev.CRITICAL),
        description="a person appears to have fallen",
        suggested_action="check on them",
        labels=("person",),
        track_ids=(track,),
        subject_track_ids=(track,),
    )


def absorb(register: AlertRegister, event: Event) -> Alert | None:
    return register.absorb(event, camera_label="Cam 1", zone=Zone.CORRIDOR, now=event.occurred_at)


class TestRestore:
    @pytest.mark.asyncio
    async def test_a_stored_acknowledgement_comes_back(self) -> None:
        register = AlertRegister()
        seed = AlertRegister()
        alert = absorb(seed, an_event())
        assert alert is not None
        acknowledged = seed.acknowledge(alert.alert_id, by="night shift", at=1.0)

        persistence = AlertPersistence(register, RecordingStore([acknowledged]))
        assert await persistence.restore() == 1

        (restored,) = register.snapshot()
        assert restored.state is AlertState.ACKNOWLEDGED
        assert restored.acknowledged_by == "night shift"

    @pytest.mark.asyncio
    async def test_an_empty_store_is_a_first_start(self) -> None:
        register = AlertRegister()
        assert await AlertPersistence(register, RecordingStore()).restore() == 0
        assert register.snapshot() == ()

    @pytest.mark.asyncio
    async def test_a_restored_open_alert_keeps_absorbing_recurrences(self) -> None:
        """Otherwise a restart silently splits one ongoing episode into two, which is
        the failure the merge window exists to prevent."""
        seed = AlertRegister()
        first = absorb(seed, an_event())
        assert first is not None

        register = AlertRegister()
        await AlertPersistence(register, RecordingStore([first])).restore()
        again = absorb(register, an_event())

        assert again is not None
        assert again.alert_id == first.alert_id
        assert again.occurrences == 2

    @pytest.mark.asyncio
    async def test_a_restored_resolved_alert_does_not_reopen(self) -> None:
        """A person said it was finished. A recurrence is a new episode, and merging
        into the closed one would erase their judgement — across a restart too."""
        seed = AlertRegister()
        first = absorb(seed, an_event())
        assert first is not None
        resolved = seed.resolve(first.alert_id)

        register = AlertRegister()
        await AlertPersistence(register, RecordingStore([resolved])).restore()
        again = absorb(register, an_event())

        assert again is not None
        assert again.alert_id != resolved.alert_id
        assert register.get(resolved.alert_id).state is AlertState.RESOLVED

    @pytest.mark.asyncio
    async def test_restoring_into_a_used_register_is_refused(self) -> None:
        """Ordering bug insurance. `restore` is not a merge, and a composition root that
        started cameras first must fail loudly rather than half-restore."""
        register = AlertRegister()
        absorb(register, an_event())
        with pytest.raises(RuntimeError, match="already has alerts"):
            await AlertPersistence(register, RecordingStore([])).restore()
            register.restore(())


class TestFlush:
    @pytest.mark.asyncio
    async def test_an_unchanged_register_is_not_rewritten(self) -> None:
        """A whole-set write is O(alerts); doing it on a timer regardless would be a
        disk write every five seconds on an idle site forever."""
        register = AlertRegister()
        store = RecordingStore()
        persistence = AlertPersistence(register, store)
        await persistence.restore()

        assert await persistence.flush() is False
        assert store.saved == []

    @pytest.mark.asyncio
    async def test_a_change_is_written_once(self) -> None:
        register = AlertRegister()
        store = RecordingStore()
        persistence = AlertPersistence(register, store)
        await persistence.restore()
        absorb(register, an_event())

        assert await persistence.flush() is True
        assert await persistence.flush() is False
        assert len(store.saved) == 1

    @pytest.mark.asyncio
    async def test_a_store_failure_does_not_propagate(self) -> None:
        """A full disk must not take surveillance down over bookkeeping."""
        register = AlertRegister()
        persistence = AlertPersistence(register, RecordingStore(fail=True))
        await persistence.restore()
        absorb(register, an_event())

        assert await persistence.flush() is False

    @pytest.mark.asyncio
    async def test_a_failed_flush_is_retried_rather_than_marked_saved(self) -> None:
        """The bug a dirty *flag* would have: clearing it on a failed write means the
        change is never written again until an unrelated one arrives."""
        register = AlertRegister()
        store = RecordingStore(fail=True)
        persistence = AlertPersistence(register, store)
        await persistence.restore()
        absorb(register, an_event())
        await persistence.flush()

        store.fail = False
        assert await persistence.flush() is True
        assert len(store.saved) == 1

    @pytest.mark.asyncio
    async def test_a_change_during_a_write_is_not_lost(self) -> None:
        """The other half of why a revision counter, not a flag. The revision is read
        *before* the snapshot, so anything landing mid-write leaves the numbers unequal
        and the next flush picks it up."""
        register = AlertRegister()

        class SlowStore(RecordingStore):
            async def save(self, alerts: Sequence[Alert]) -> None:
                await asyncio.sleep(0)
                absorb(register, an_event(track=2))
                await super().save(alerts)

        store = SlowStore()
        persistence = AlertPersistence(register, store)
        await persistence.restore()
        absorb(register, an_event(track=1))

        await persistence.flush()
        assert await persistence.flush() is True


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_stopping_writes_what_the_timer_had_not(self) -> None:
        """An operator who acknowledged something a second before SIGTERM keeps it."""
        register = AlertRegister()
        store = RecordingStore()
        persistence = AlertPersistence(register, store, flush_interval_seconds=3600)
        await persistence.restore()
        persistence.start()
        absorb(register, an_event())

        await persistence.stop()

        assert len(store.saved) == 1

    @pytest.mark.asyncio
    async def test_the_timer_writes_machine_driven_changes(self) -> None:
        register = AlertRegister()
        store = RecordingStore()
        persistence = AlertPersistence(register, store, flush_interval_seconds=0.01)
        await persistence.restore()
        persistence.start()
        absorb(register, an_event())

        for _ in range(200):
            if store.saved:
                break
            await asyncio.sleep(0.005)
        await persistence.stop()

        assert store.saved, "the coalescing timer never flushed"

    @pytest.mark.asyncio
    async def test_stopping_twice_is_harmless(self) -> None:
        persistence = AlertPersistence(AlertRegister(), RecordingStore())
        await persistence.restore()
        persistence.start()
        await persistence.stop()
        await persistence.stop()

    def test_a_non_positive_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            AlertPersistence(AlertRegister(), RecordingStore(), flush_interval_seconds=0)
