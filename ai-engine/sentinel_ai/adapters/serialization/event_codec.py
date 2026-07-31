# ai-engine/sentinel_ai/adapters/serialization/event_codec.py
"""Event ⇄ wire-payload translation, validated against the committed JSON Schema.

Lives in adapters, not domain: it depends on jsonschema and on file layout.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
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


def validate_payload(payload: Mapping[str, object]) -> None:
    _validator().validate(dict(payload))


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
