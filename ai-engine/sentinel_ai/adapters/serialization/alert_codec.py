"""Alert ⇄ JSON, for the durable register (spec §17).

Sibling of `event_codec.py` and deliberately *not* the same shape, because the two
serve different readers. An event crosses a process boundary to a consumer nobody here
controls, so it is validated against a committed JSON Schema on the way out. An alert
is written by this engine and read back by this engine on its next start; the only
reader is the writer, one version later.

That makes the interesting risk a different one. It is not "will a consumer cope" but
"will *this* engine cope with a file its predecessor wrote". So:

* `SCHEMA_VERSION` is written, and a file whose version this build does not know is
  discarded rather than guessed at.
* `decode_alert` rebuilds field by field and raises on anything it cannot place, and
  the store turns that into "no alerts to restore" — see `AlertStore.load`.
* Enum members are stored by value, never by name or ordinal, so reordering
  `AlertState` or adding a `Severity` cannot silently reinterpret a stored alert.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from sentinel_ai.domain.alert import Alert, AlertKey, AlertState
from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.zone import Zone

__all__ = ["SCHEMA_VERSION", "decode_alert", "decode_register", "encode_alert", "encode_register"]

SCHEMA_VERSION = 1


def encode_alert(alert: Alert) -> dict[str, Any]:
    return {
        "alert_id": str(alert.alert_id),
        "key": {
            "camera_id": alert.key.camera_id,
            "reason": alert.key.reason.value,
            "subject": alert.key.subject,
        },
        "state": alert.state.value,
        "severity": alert.severity.value,
        "priority": alert.priority.value,
        "first_seen": alert.first_seen,
        "last_seen": alert.last_seen,
        "occurrences": alert.occurrences,
        "description": alert.description,
        "camera_label": alert.camera_label,
        "zone": None if alert.zone is None else alert.zone.value,
        "event_ids": [str(event_id) for event_id in alert.event_ids],
        "clip_uri": alert.clip_uri,
        "notify_clip_uri": alert.notify_clip_uri,
        "acknowledged_by": alert.acknowledged_by,
        "acknowledged_at": alert.acknowledged_at,
    }


def decode_alert(payload: Mapping[str, Any]) -> Alert:
    """Rebuild one alert. Raises on anything unrecognised — see the module docstring."""
    raw_zone = payload["zone"]
    key = cast(Mapping[str, Any], payload["key"])
    return Alert(
        alert_id=UUID(payload["alert_id"]),
        key=AlertKey(
            camera_id=key["camera_id"],
            reason=EscalationReason(key["reason"]),
            subject=key["subject"],
        ),
        state=AlertState(payload["state"]),
        severity=Severity(payload["severity"]),
        priority=EventPriority(payload["priority"]),
        first_seen=float(payload["first_seen"]),
        last_seen=float(payload["last_seen"]),
        occurrences=int(payload["occurrences"]),
        description=payload["description"],
        camera_label=payload["camera_label"],
        zone=None if raw_zone is None else Zone(raw_zone),
        event_ids=tuple(UUID(event_id) for event_id in payload["event_ids"]),
        clip_uri=payload["clip_uri"],
        # The one tolerant read in this function, and deliberately so. Every other
        # field is subscripted, because a missing one means the file is not what this
        # build thinks it is. This field is different: it was added after
        # `SCHEMA_VERSION` 1 was already being written, and its absence is not a
        # damaged file — it is a file from a build that never made a short clip, for
        # which `None` is the correct value rather than a guess at one. Bumping the
        # version instead would discard the whole register on upgrade, throwing away
        # every acknowledgement to learn something the default already tells us.
        notify_clip_uri=payload.get("notify_clip_uri"),
        acknowledged_by=payload["acknowledged_by"],
        acknowledged_at=(
            None if payload["acknowledged_at"] is None else float(payload["acknowledged_at"])
        ),
    )


def encode_register(alerts: Sequence[Alert]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "_comment": (
            "Operator triage state for the SentinelAI engine: which alerts a human has "
            "seen and which they have closed. Rewritten whole on every save. This is "
            "not the record of what happened — that is the published event stream."
        ),
        "alerts": [encode_alert(alert) for alert in alerts],
    }


def decode_register(payload: Mapping[str, Any]) -> tuple[Alert, ...]:
    """Every alert in a stored document, oldest first as written.

    A `schema_version` this build does not know raises rather than being read
    optimistically: the fields it would be guessing at are the ones that decide whether
    an operator is shown an alert as unseen.
    """
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"alert store schema_version {version!r} is not {SCHEMA_VERSION}; refusing to guess"
        )
    return tuple(decode_alert(item) for item in cast(list[Any], payload["alerts"]))
