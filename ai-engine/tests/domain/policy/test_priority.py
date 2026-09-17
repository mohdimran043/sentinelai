"""Event priority: the ordering that decides what gets the GPU first (spec §16)."""

from __future__ import annotations

import pytest

from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.domain.policy.priority import (
    EventPriority,
    outranks,
    priority_of,
    rank,
    severity_floor,
)


def test_every_reason_declares_a_priority() -> None:
    """A reason with no entry would default to something plausible, and the first
    anyone would know is a suspected fall queued behind four periodic summaries."""
    for reason in EscalationReason:
        assert isinstance(priority_of(reason), EventPriority)


def test_a_suspected_fall_is_the_only_critical_reason() -> None:
    """Critical is what preempts. Spending it on anything that is not "somebody may be
    hurt" is how preemption stops meaning anything."""
    critical = [r for r in EscalationReason if priority_of(r) is EventPriority.CRITICAL]
    assert critical == [EscalationReason.FALL_SUSPECTED]


def test_a_forced_look_cannot_displace_a_fall() -> None:
    """An operator clicking "describe now" on a quiet camera must not push a suspected
    collapse down the queue."""
    assert outranks(
        priority_of(EscalationReason.FALL_SUSPECTED),
        priority_of(EscalationReason.USER_REQUESTED),
    )


def test_a_periodic_summary_is_the_lowest_of_the_low() -> None:
    """Not an anomaly at all — the forced look that happens whether or not anything is
    wrong. It must never displace something that is."""
    assert priority_of(EscalationReason.PERIODIC_SUMMARY) is EventPriority.LOW


def test_tampering_outranks_what_it_makes_untrustworthy() -> None:
    assert outranks(
        priority_of(EscalationReason.CAMERA_TAMPER),
        priority_of(EscalationReason.LINE_CROSSING),
    )


def test_the_ordering_is_not_alphabetical() -> None:
    """`EventPriority` is a `StrEnum`, so `LOW > CRITICAL` is True by alphabet — the
    exact inversion of what anyone writing that comparison means. `outranks` is the
    only ordering this type has."""
    assert EventPriority.LOW > EventPriority.CRITICAL
    assert not outranks(EventPriority.LOW, EventPriority.CRITICAL)


def test_rank_sorts_worst_first_when_negated() -> None:
    reasons = [EscalationReason.PERIODIC_SUMMARY, EscalationReason.FALL_SUSPECTED]
    reasons.sort(key=lambda reason: -rank(priority_of(reason)))
    assert reasons[0] is EscalationReason.FALL_SUSPECTED


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (EscalationReason.FALL_SUSPECTED, Severity.CRITICAL),
        (EscalationReason.ZONE_INTRUSION, Severity.HIGH),
        (EscalationReason.DWELL_EXCEEDED, Severity.MEDIUM),
        (EscalationReason.PERIODIC_SUMMARY, Severity.INFO),
    ],
)
def test_the_severity_floor_matches_the_priority(
    reason: EscalationReason, expected: Severity
) -> None:
    assert severity_floor(reason) is expected


def test_a_routine_look_floors_at_info_not_low() -> None:
    """Giving every routine look a non-`info` severity is how an alert list stops
    being scannable."""
    assert severity_floor(EscalationReason.PERIODIC_SUMMARY) is Severity.INFO
