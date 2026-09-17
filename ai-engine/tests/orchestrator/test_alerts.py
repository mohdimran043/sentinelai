"""The alert register: bookkeeping, eviction order, and the state transitions.

The aggregation *rule* is pinned exhaustively in `tests/domain/policy/test_alerting.py`
and is not re-tested here. What this owns is what the register does around it — which
alert a key points at, what gets thrown away when it is full, and that the API's error
cases are reachable.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from sentinel_ai.domain.alert import Alert, AlertState
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.alerts import AlertRegister, UnknownAlertError


def event(
    *,
    reason: EscalationReason = EscalationReason.ZONE_INTRUSION,
    threat: float = 0.5,
    track_ids: tuple[int, ...] = (7,),
    camera_id: str = "cam-1",
    occurred_at: float = 0.0,
) -> Event:
    return Event(
        event_id=uuid4(),
        camera_id=camera_id,
        occurred_at=occurred_at,
        reason=reason,
        threat=ThreatScore.from_value(threat),
        description="a person is in the stairwell",
        suggested_action="Go and look.",
        track_ids=track_ids,
        subject_track_ids=track_ids,
    )


def absorb(register: AlertRegister, item: Event, *, now: float = 0.0) -> Alert | None:
    return register.absorb(item, camera_label="Corridor 1", zone=Zone.CORRIDOR, now=now)


class TestAbsorbing:
    def test_a_recurrence_lands_on_the_same_alert(self) -> None:
        register = AlertRegister()
        first = absorb(register, event(), now=0.0)
        second = absorb(register, event(), now=5.0)
        assert first is not None and second is not None
        assert first.alert_id == second.alert_id
        assert second.occurrences == 2
        assert len(register.snapshot()) == 1

    def test_a_recurrence_past_the_window_opens_a_second_alert(self) -> None:
        register = AlertRegister(merge_window_seconds=10.0)
        absorb(register, event(), now=0.0)
        absorb(register, event(), now=100.0)
        assert len(register.snapshot()) == 2

    def test_the_old_episode_stays_readable_after_a_new_one_opens(self) -> None:
        """A key points at the current episode; the previous one does not vanish, it
        just stops absorbing."""
        register = AlertRegister(merge_window_seconds=10.0)
        first = absorb(register, event(), now=0.0)
        absorb(register, event(), now=100.0)
        assert first is not None
        assert register.get(first.alert_id).occurrences == 1

    def test_an_unalertable_event_produces_nothing(self) -> None:
        """Most escalations are periodic summaries of empty corridors. They are still
        published and still searchable; they just do not become a row."""
        register = AlertRegister()
        assert (
            absorb(register, event(reason=EscalationReason.PERIODIC_SUMMARY, threat=0.01)) is None
        )
        assert register.snapshot() == ()


class TestOrdering:
    def test_the_worst_alert_is_first_whatever_arrived_last(self) -> None:
        """§27: the critical thing has to be visible immediately, not wherever it
        happens to fall in arrival order."""
        register = AlertRegister()
        absorb(register, event(reason=EscalationReason.LINE_CROSSING, track_ids=(1,)), now=0.0)
        absorb(register, event(reason=EscalationReason.FALL_SUSPECTED, track_ids=(2,)), now=1.0)
        absorb(register, event(reason=EscalationReason.LINE_CROSSING, track_ids=(3,)), now=2.0)

        top = register.snapshot()[0]
        assert top.priority is EventPriority.CRITICAL

    def test_within_a_priority_the_most_recent_is_first(self) -> None:
        register = AlertRegister()
        absorb(register, event(track_ids=(1,)), now=0.0)
        absorb(register, event(track_ids=(2,)), now=50.0)
        assert register.snapshot()[0].last_seen == 50.0

    def test_open_alerts_excludes_resolved_ones(self) -> None:
        register = AlertRegister()
        alert = absorb(register, event())
        assert alert is not None
        register.resolve(alert.alert_id)
        assert register.open_alerts() == ()
        assert len(register.snapshot()) == 1


class TestStateTransitions:
    def test_acknowledging_records_who(self) -> None:
        register = AlertRegister()
        alert = absorb(register, event())
        assert alert is not None
        updated = register.acknowledge(alert.alert_id, by="operator-1", at=9.0)
        assert updated.state is AlertState.ACKNOWLEDGED
        assert register.get(alert.alert_id).acknowledged_by == "operator-1"

    def test_acknowledging_a_resolved_alert_raises(self) -> None:
        """The operator is acting on a stale list. The API turns this into a 409
        rather than a silent success."""
        register = AlertRegister()
        alert = absorb(register, event())
        assert alert is not None
        register.resolve(alert.alert_id)
        with pytest.raises(ValueError, match="already resolved"):
            register.acknowledge(alert.alert_id, by="operator-1", at=9.0)

    def test_an_unknown_alert_id_raises_a_typed_error(self) -> None:
        register = AlertRegister()
        missing = uuid4()
        with pytest.raises(UnknownAlertError) as excinfo:
            register.get(missing)
        assert excinfo.value.alert_id == missing


class TestEviction:
    def test_the_register_is_bounded(self) -> None:
        register = AlertRegister(capacity=5, merge_window_seconds=0.0)
        for index in range(20):
            absorb(register, event(track_ids=(index,)), now=float(index))
        assert len(register.snapshot()) == 5

    def test_a_resolved_alert_is_evicted_before_an_unacknowledged_one(self) -> None:
        """Dropping by age alone would throw away the one alert nobody has looked at
        in favour of one somebody already closed."""
        register = AlertRegister(capacity=2)
        oldest = absorb(register, event(track_ids=(1,)), now=0.0)
        assert oldest is not None
        register.resolve(oldest.alert_id)
        unacknowledged = absorb(register, event(track_ids=(2,)), now=1.0)
        assert unacknowledged is not None

        absorb(register, event(track_ids=(3,)), now=2.0)

        remaining = {alert.alert_id for alert in register.snapshot()}
        assert oldest.alert_id not in remaining, "the resolved one went first"
        assert unacknowledged.alert_id in remaining

    def test_an_evicted_alert_frees_its_key_so_a_recurrence_opens_cleanly(self) -> None:
        register = AlertRegister(capacity=1)
        first = absorb(register, event(track_ids=(1,)), now=0.0)
        assert first is not None
        absorb(register, event(track_ids=(2,)), now=1.0)
        # Track 1 recurs after its alert was evicted. It must open a new one rather
        # than merging into a record that no longer exists.
        third = absorb(register, event(track_ids=(1,)), now=2.0)
        assert third is not None
        assert third.alert_id != first.alert_id
        assert third.occurrences == 1

    def test_a_capacity_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot hold one"):
            AlertRegister(capacity=0)


def test_alert_ids_are_injectable_so_a_test_can_assert_on_them() -> None:
    fixed = UUID("00000000-0000-4000-8000-000000000001")
    register = AlertRegister(new_alert_id=lambda: fixed)
    alert = absorb(register, event())
    assert alert is not None
    assert alert.alert_id == fixed
