# ai-engine/tests/adapters/test_event_codec.py
from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from jsonschema import ValidationError

from sentinel_ai.adapters.serialization.event_codec import (
    EVENT_SCHEMA_PATH,
    decode_event,
    encode_event,
    validate_payload,
)
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore


def make_event(**overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.72),
        "description": "A person is running toward the gate.",
        "suggested_action": "Review the clip and confirm identity.",
        "labels": ("person",),
        "track_ids": (7,),
        "keyframe_uri": "s3://sentinel-clips/cam-1/key.jpg",
        "clip_uri": "s3://sentinel-clips/cam-1/clip.mp4",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_the_schema_file_is_valid_json() -> None:
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")


def test_an_encoded_event_validates_against_the_committed_schema() -> None:
    validate_payload(encode_event(make_event()))


def test_encoding_flattens_the_threat_score() -> None:
    payload = encode_event(make_event())
    assert payload["threat_score"] == 0.72
    assert payload["severity"] == "high"


def test_encoding_never_leaks_model_identity() -> None:
    """Spec §3.3: the Web Platform must not learn which model produced a result."""
    payload = encode_event(make_event())
    serialized = json.dumps(payload).lower()
    for forbidden in ("yolo", "qwen", "whisper", "bytetrack", "awq"):
        assert forbidden not in serialized


def test_round_trip_preserves_every_field() -> None:
    original = make_event()
    restored = decode_event(encode_event(original))
    assert restored == original


def test_round_trip_preserves_a_minimal_event() -> None:
    original = make_event(
        labels=(),
        track_ids=(),
        keyframe_uri=None,
        clip_uri=None,
        description_unavailable=True,
        description="",
    )
    assert decode_event(encode_event(original)) == original


def test_event_id_is_serialized_as_a_uuid_string() -> None:
    payload = encode_event(make_event())
    assert isinstance(payload["event_id"], str)
    UUID(str(payload["event_id"]))


@pytest.mark.parametrize(
    "mutation",
    [
        {"threat_score": 1.5},
        {"severity": "catastrophic"},
        {"reason": "aliens"},
        {"camera_id": ""},
        {"event_id": "not-a-uuid"},
    ],
    ids=["score-out-of-range", "bad-severity", "bad-reason", "empty-camera", "bad-uuid"],
)
def test_invalid_payloads_are_rejected(mutation: dict[str, object]) -> None:
    payload = encode_event(make_event()) | mutation
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_encoding_an_event_with_a_blank_camera_id_is_rejected() -> None:
    """Nothing upstream enforces minLength: 1, so the codec is the last line."""
    with pytest.raises(ValidationError, match="should be non-empty"):
        encode_event(make_event(camera_id=""))


def test_encoding_an_out_of_range_threat_score_is_rejected() -> None:
    """ThreatScore constructed directly bypasses from_value's range check."""
    bypassed = ThreatScore(value=5.0, severity=Severity.CRITICAL)
    with pytest.raises(ValidationError, match="greater than the maximum"):
        encode_event(make_event(threat=bypassed))


def test_a_payload_missing_a_required_field_is_rejected() -> None:
    payload = encode_event(make_event())
    del payload["threat_score"]
    with pytest.raises(ValidationError, match="threat_score"):
        validate_payload(payload)
