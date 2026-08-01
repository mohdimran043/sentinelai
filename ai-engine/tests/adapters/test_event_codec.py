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
        # The two timelines an event carries. The source value is the one a live run
        # actually published as `occurred_at` before D1 — a `time.monotonic()` reading.
        "source_timestamp": 51_181.128868795,
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


class TestBothTimelinesAreOnTheWire:
    """D1. `occurred_at` used to be the *source* timeline — a live run published
    `"occurred_at": 51181.128868795`, a `time.monotonic()` reading — while this
    module's own docstring tells the Phase 1C Go consumer to sort by it. The two
    timelines now travel as two fields, and the schema says which is which.
    """

    def test_the_payload_carries_both_and_they_are_distinct(self) -> None:
        payload = encode_event(make_event())
        assert payload["occurred_at"] == 1_700_000_000.0
        assert payload["source_timestamp"] == 51_181.128868795

    def test_a_round_trip_keeps_the_source_timeline(self) -> None:
        """Fails against a codec that encodes the new field but drops it on decode:
        the spool round-trips through exactly this path, so a replayed event would
        silently lose the value that ties it to its clip."""
        original = make_event()
        assert decode_event(encode_event(original)).source_timestamp == 51_181.128868795

    def test_a_payload_written_before_the_field_existed_still_decodes(self) -> None:
        """Optional on the wire, and it has to be: the disk spool holds payloads
        written by an older build, and `replay_spool` must not choke on them."""
        payload = encode_event(make_event())
        del payload["source_timestamp"]
        validate_payload(payload)
        assert decode_event(payload).source_timestamp is None

    def test_the_schema_documents_what_each_timeline_means(self) -> None:
        """The schema is the artefact Phase 1C generates its Go client from, so the
        semantics have to live there and not only in a Python docstring. Fails against
        the bare `{"type": "number"}` this field shipped with, which is precisely how
        the two readings of `occurred_at` came to disagree in the first place."""
        properties = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))["properties"]
        assert "epoch" in properties["occurred_at"]["description"].lower()
        assert "monotonic" in properties["source_timestamp"]["description"].lower()

    def test_the_schema_rejects_a_source_timestamp_that_is_not_a_number(self) -> None:
        payload = encode_event(make_event()) | {"source_timestamp": "51181.13"}
        with pytest.raises(ValidationError):
            validate_payload(payload)


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


class TestNonFiniteNumbersNeverReachTheWire:
    """`json.dumps` emits the bare tokens `NaN`, `Infinity` and `-Infinity`, which are
    a Python extension and not JSON. Go's `encoding/json` rejects all three, so one
    such event would break the Phase 1C consumer's decode loop, not merely itself."""

    def test_json_dumps_really_does_emit_invalid_json_for_these(self) -> None:
        """The premise, pinned. If a future Python ever stopped emitting bare tokens
        the guard below would still be correct but its justification would not be."""
        assert json.dumps({"x": float("nan")}) == '{"x": NaN}'
        assert json.dumps({"x": float("inf")}) == '{"x": Infinity}'
        with pytest.raises(ValueError, match="Out of range float"):
            json.dumps({"x": float("nan")}, allow_nan=False)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_occurred_at_is_rejected_on_encode(self, value: float) -> None:
        """jsonschema's `type: number` admits it — `float("nan")` is a number — so the
        schema alone cannot catch this."""
        with pytest.raises(ValueError, match="occurred_at must be a finite number"):
            encode_event(make_event(occurred_at=value))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_source_timestamp_is_rejected_on_encode(self, value: float) -> None:
        """The second numeric timeline needs the same guard as the first: a bare `NaN`
        token anywhere in the payload breaks Go's `encoding/json` for the whole event,
        not merely for the field that carries it.

        Fails against a `_FINITE_FIELDS` that was never widened when the field was
        added — the payload encodes cleanly and the consumer's decode loop is what
        finds out."""
        with pytest.raises(ValueError, match="source_timestamp must be a finite number"):
            encode_event(make_event(source_timestamp=value))

    def test_a_null_source_timestamp_is_not_mistaken_for_a_non_finite_one(self) -> None:
        """The field is nullable, so the finite check must skip `None` rather than
        trip over it."""
        assert encode_event(make_event(source_timestamp=None))["source_timestamp"] is None

    def test_a_nan_threat_score_is_rejected_on_encode(self) -> None:
        """NaN is the one that gets through the schema: `maximum: 1.0` already stops
        `inf`, but every comparison against NaN is false, so `minimum`/`maximum` both
        pass it. `ThreatScore.from_value` rejects it for the same reason it must —
        a directly constructed `ThreatScore` bypasses that check entirely."""
        bypassed = ThreatScore(value=float("nan"), severity=Severity.INFO)
        with pytest.raises(ValueError, match="threat_score must be a finite number"):
            encode_event(make_event(threat=bypassed))

    def test_an_infinite_threat_score_is_stopped_by_the_schema_bound(self) -> None:
        bypassed = ThreatScore(value=float("inf"), severity=Severity.INFO)
        with pytest.raises(ValidationError, match="greater than the maximum"):
            encode_event(make_event(threat=bypassed))

    def test_a_non_finite_value_read_back_off_the_spool_is_rejected_too(self) -> None:
        """`json.loads` accepts the tokens by default, so a payload written before this
        guard existed must not be replayed onto the broker now."""
        payload = json.loads('{"occurred_at": NaN}')
        with pytest.raises(ValueError, match="occurred_at must be a finite number"):
            validate_payload({**encode_event(make_event()), **payload})

    def test_a_finite_extreme_is_still_allowed(self) -> None:
        """The guard is about representability, not about range."""
        encode_event(make_event(occurred_at=1e308))

    def test_the_boolean_true_is_not_mistaken_for_a_number(self) -> None:
        """`bool` is a subclass of `int`, and `isfinite(True)` is True, so a naive
        check would pass it through to the schema — which is where a non-number
        belongs, with its own error message."""
        payload = {**encode_event(make_event()), "occurred_at": True}
        with pytest.raises(ValidationError):
            validate_payload(payload)
