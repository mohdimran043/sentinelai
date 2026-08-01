"""Event ⇄ wire-payload translation, validated against the committed JSON Schema.

Lives in adapters, not domain: it depends on jsonschema and on file layout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from jsonschema import Draft202012Validator

from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore

SCHEMA_VERSION = 1

EVENT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3].parent
    / "contracts"
    / "events"
    / "anomaly_event.schema.json"
)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


_FINITE_FIELDS = ("occurred_at", "threat_score")
"""The two numeric fields on the wire. Both must be finite; see `_require_finite`."""


def _require_finite(payload: Mapping[str, object]) -> None:
    """Reject NaN and ±Infinity, which the JSON Schema cannot.

    Draft 2020-12's `type: number` admits them — jsonschema is validating Python
    floats, and `float("nan")` is a number — but they are not JSON. `json.dumps`
    happily emits the bare tokens `NaN`, `Infinity` and `-Infinity`, which are a
    Python extension: Go's `encoding/json` rejects all three outright, so a single
    such event would break the Phase 1C consumer's decode loop rather than just
    itself. Worse for `occurred_at` specifically, because every comparison against a
    NaN is false, so it also defeats the "sort by occurred_at" ordering the consumer
    is told to rely on.

    Checked here rather than in `encode_event` so it holds in both directions: the
    same guard covers a payload read back off the disk spool, where `json.loads`
    would otherwise accept the tokens it should never have written.
    """
    for field in _FINITE_FIELDS:
        value = payload.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool) and not isfinite(value):
            raise ValueError(f"{field} must be a finite number, got {value!r}")


def validate_payload(payload: Mapping[str, object]) -> None:
    _validator().validate(dict(payload))
    _require_finite(payload)


def encode_event(event: Event) -> dict[str, object]:
    """Build the wire payload and validate it before it can leave the process.

    Validating on the way *out* is the point: `Event` cannot enforce the schema
    on its own (`camera_id=""` and a directly constructed out-of-range
    `ThreatScore` both slip past it), and the Go consumer breaks on whatever we
    publish. A 15-field Draft 2020-12 validate costs microseconds against at
    most a few events per minute per camera.
    """
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(event.event_id),
        "camera_id": event.camera_id,
        "occurred_at": event.occurred_at,
        "reason": event.reason.value,
        "threat_score": event.threat.value,
        "severity": event.threat.severity.value,
        "description": event.description,
        "suggested_action": event.suggested_action,
        "labels": list(event.labels),
        "track_ids": list(event.track_ids),
        "keyframe_uri": event.keyframe_uri,
        "clip_uri": event.clip_uri,
        "description_unavailable": event.description_unavailable,
        "metadata": dict(event.metadata),
    }
    validate_payload(payload)
    return payload


def decode_event(payload: Mapping[str, object]) -> Event:
    validate_payload(payload)
    data = cast(dict[str, Any], dict(payload))
    return Event(
        event_id=UUID(data["event_id"]),
        camera_id=data["camera_id"],
        occurred_at=float(data["occurred_at"]),
        reason=EscalationReason(data["reason"]),
        threat=ThreatScore(value=float(data["threat_score"]), severity=Severity(data["severity"])),
        description=data["description"],
        suggested_action=data["suggested_action"],
        labels=tuple(data["labels"]),
        track_ids=tuple(int(i) for i in data["track_ids"]),
        keyframe_uri=data.get("keyframe_uri"),
        clip_uri=data.get("clip_uri"),
        description_unavailable=bool(data["description_unavailable"]),
        metadata=dict(data["metadata"]),
    )
