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
from sentinel_ai.domain.welfare import (
    ConcernKind,
    Confidence,
    WelfareAssessment,
    WelfareConcern,
)

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


_FINITE_FIELDS = ("occurred_at", "source_timestamp", "threat_score")
"""Every numeric field on the wire. All must be finite; see `_require_finite`.

`source_timestamp` is nullable, and `None` is not a number, so it is skipped by the
`isinstance` guard below rather than needing a case of its own."""


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
        # Both timelines, never one: `occurred_at` is Unix epoch seconds and is what a
        # consumer sorts and displays on; `source_timestamp` is the camera's own
        # timeline, which is what correlates the event with a clip's pts. See
        # `Event`'s docstring and the descriptions in the committed schema.
        "source_timestamp": event.source_timestamp,
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
    # Present only when there is something to report. `WelfareAssessment.none()`
    # (the field's default, and today the only value anything in this codebase
    # produces) must not appear as `{"concerns": [], "basis": ...}` — that would
    # read as "assessed, nothing found", a different fact from "not assessed",
    # which is what an event predating a describe step (or this field) actually
    # means. Omission is the only wire representation that does not conflate them.
    if event.welfare.concerns:
        payload["welfare"] = {
            "concerns": [
                {
                    "kind": concern.kind.value,
                    "confidence": concern.confidence.value,
                    "evidence": concern.evidence,
                }
                for concern in event.welfare.concerns
            ],
            "basis": event.welfare.basis,
        }
    validate_payload(payload)
    return payload


def _decode_welfare(payload: Mapping[str, object]) -> WelfareAssessment:
    """Rebuild `WelfareAssessment` field by field — never `WelfareAssessment(**raw)`.

    `basis` is `Literal["single_frame_vlm"]`, enforced by mypy only; the schema's
    `const` already rejects any other value before this function runs, but this
    function does not even read the payload's `basis` key, let alone assign it.
    The dataclass's own default supplies it, so there is no code path here through
    which a wire payload could ever set what a consumer trusts as provenance.
    Absent `welfare` (a payload from before this field existed, or an event with
    nothing to report — the two are indistinguishable on the wire, by design; see
    `encode_event`) decodes to `WelfareAssessment.none()`.
    """
    raw = payload.get("welfare")
    if raw is None:
        return WelfareAssessment.none()
    raw_welfare = cast(dict[str, Any], raw)
    concerns = tuple(
        WelfareConcern(
            kind=ConcernKind(item["kind"]),
            confidence=Confidence(item["confidence"]),
            evidence=item["evidence"],
        )
        for item in cast(list[dict[str, Any]], raw_welfare["concerns"])
    )
    return WelfareAssessment(concerns=concerns)


def decode_event(payload: Mapping[str, object]) -> Event:
    validate_payload(payload)
    data = cast(dict[str, Any], dict(payload))
    # Absent and explicit-null both decode to None: the field is optional on the wire,
    # so a payload written before it existed (one already sitting on the disk spool,
    # say) must still replay rather than raising on a missing key.
    source_timestamp = data.get("source_timestamp")
    return Event(
        event_id=UUID(data["event_id"]),
        camera_id=data["camera_id"],
        occurred_at=float(data["occurred_at"]),
        source_timestamp=None if source_timestamp is None else float(source_timestamp),
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
        welfare=_decode_welfare(data),
    )
