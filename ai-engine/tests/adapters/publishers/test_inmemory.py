"""InMemoryPublisher — the real (not fake) EventPublisher used for CPU-only
runs and the keystone end-to-end test when no broker is configured (spec §5.6).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from jsonschema import ValidationError

from sentinel_ai.adapters.publishers.inmemory import InMemoryPublisher
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore


def _event(**overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.7),
        "description": "A person is running.",
        "suggested_action": "Review the clip.",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


async def test_publish_collects_events_in_order() -> None:
    publisher = InMemoryPublisher()
    first, second = _event(), _event()

    await publisher.publish(first)
    await publisher.publish(second)

    assert publisher.events == (first, second)


async def test_close_is_idempotent_and_does_not_clear_events() -> None:
    publisher = InMemoryPublisher()
    await publisher.publish(_event())
    await publisher.close()
    await publisher.close()
    assert len(publisher.events) == 1


async def test_publish_validates_before_accepting_the_event() -> None:
    """Every payload goes through `encode_event`, so a schema-invalid event
    (here, a threat score that bypassed `ThreatScore.from_value`'s range
    check by direct construction) fails loudly at publish time rather than
    reaching a consumer."""
    publisher = InMemoryPublisher()
    bypassed = _event(threat=ThreatScore(value=5.0, severity=Severity.CRITICAL))

    with pytest.raises(ValidationError):
        await publisher.publish(bypassed)

    assert publisher.events == ()
