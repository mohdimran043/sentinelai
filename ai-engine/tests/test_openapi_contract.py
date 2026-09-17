"""Spec §3.2: the committed OpenAPI document must not drift from the running app.

The spec asks for CI to export the document and fail the build if the committed copy
has drifted. That is a pytest assertion here rather than a workflow step, for two
reasons: CI already runs the suite, so there is nothing new to remember to add; and
it fails on the developer's machine at the moment they change a route, not twenty
minutes later on a push.
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.config.camera_file import EDITABLE_FIELDS
from sentinel_ai.openapi_export import OPENAPI_PATH, openapi_document, render

REGENERATE = "python -m sentinel_ai.openapi_export"


def test_the_document_covers_the_endpoints_phase_1c_consumes() -> None:
    paths = openapi_document()["paths"]
    assert set(paths) == {
        "/health",
        "/settings",
        "/storage",
        "/cameras",
        "/cameras/probe",
        "/cameras/{camera_id}",
        "/cameras/{camera_id}/enabled",
        "/cameras/{camera_id}/clips",
        "/cameras/{camera_id}/clips/{event_id}",
        "/cameras/{camera_id}/snapshot",
        "/cameras/{camera_id}/telemetry",
        "/cameras/{camera_id}/events",
        "/cameras/{camera_id}/describe",
        "/events/stream",
        "/alerts",
        "/alerts/{alert_id}/acknowledge",
        "/alerts/{alert_id}/resolve",
        "/alerts/{alert_id}/clip",
        "/authorized-persons",
        "/authorized-persons/{person_id}",
        "/authorized-persons/{person_id}/faces",
        "/authorized-persons/{person_id}/faces/{face_id}",
        "/authorized-persons/{person_id}/faces/{face_id}/image",
    }
    # Enrol, list, and fetch or remove one. The image has its own path rather than an
    # `include=` on the list, so a roster listing never carries biometric data it was
    # not asked for — see `EnrolledFaceEntry`.
    assert set(paths["/authorized-persons/{person_id}/faces"]) == {"get", "post"}
    assert set(paths["/authorized-persons/{person_id}/faces/{face_id}"]) == {"delete"}
    assert set(paths["/authorized-persons/{person_id}/faces/{face_id}/image"]) == {"get"}
    # The write surface: add, edit, remove. `POST /cameras` is what makes the console
    # able to stand up a camera without an operator editing the file and restarting.
    assert set(paths["/cameras"]) == {"get", "post"}
    assert set(paths["/cameras/probe"]) == {"post"}
    assert set(paths["/cameras/{camera_id}"]) == {"patch", "delete"}
    # Lifecycle, deliberately not a field on PATCH: everything PATCH writes is
    # metadata a running camera absorbs between frames, and this starts or stops it.
    assert set(paths["/cameras/{camera_id}/enabled"]) == {"put"}
    # Read-only, both of them. Settings are overwhelmingly restart-bound, and storage
    # is an observation of a bucket this engine does not manage the lifetime of.
    assert set(paths["/settings"]) == {"get"}
    assert set(paths["/storage"]) == {"get"}
    assert set(paths["/cameras/{camera_id}/clips"]) == {"get"}
    assert set(paths["/cameras/{camera_id}/clips/{event_id}"]) == {"get"}
    assert set(paths["/cameras/{camera_id}/describe"]) == {"post"}
    assert set(paths["/cameras/{camera_id}/telemetry"]) == {"get"}
    # The fallback for a camera with no mediamtx playlist — an EarthCam page, a file.
    assert set(paths["/cameras/{camera_id}/snapshot"]) == {"get"}
    assert set(paths["/cameras/{camera_id}/events"]) == {"get"}
    assert set(paths["/events/stream"]) == {"get"}
    # DELETE clears the whole list — triage state, never evidence.
    assert set(paths["/alerts"]) == {"get", "delete"}
    assert set(paths["/alerts/{alert_id}/acknowledge"]) == {"post"}
    # The only route that serves a recording. GET only: a clip is written by the
    # pipeline and is never editable through the API.
    assert set(paths["/alerts/{alert_id}/clip"]) == {"get"}
    assert set(paths["/alerts/{alert_id}/resolve"]) == {"post"}
    assert set(paths["/authorized-persons"]) == {"get"}
    assert set(paths["/authorized-persons/{person_id}"]) == {"put", "delete"}


class TestTheWriteEndpointIsDocumentedAsUnauthenticated:
    """The one write endpoint in the engine, in a build with no authentication.

    Everything asserted here is prose in the generated contract rather than
    behaviour, and that is the point: the person most likely to expose this port
    to a network is reading the contract, not this repository. A Phase 1C client
    author who cannot see from the document alone that the endpoint is
    unauthenticated, off by default, and persistent has not been warned.
    """

    def test_the_operation_says_it_is_unauthenticated_and_off_by_default(self) -> None:
        operation = openapi_document()["paths"]["/cameras/{camera_id}"]["patch"]
        text = f"{operation['summary']} {operation['description']}".lower()
        assert "no authentication" in text or "not authenticated" in text
        assert "sentinel_enable_camera_writes" in text
        assert "403" in str(operation["responses"]) or "403" in operation["responses"]

    def test_the_operation_names_what_it_will_not_change_and_why(self) -> None:
        """A field the endpoint refuses is only honest if the refusal is in the
        contract. Otherwise a client author builds a URL editor, discovers the 422
        in production, and reasonably concludes the engine is broken."""
        operation = openapi_document()["paths"]["/cameras/{camera_id}"]["patch"]
        text = operation["description"].lower()
        assert "url" in text
        assert "profile" in text
        assert "restart" in text

    def test_the_operation_says_the_edit_is_persisted(self) -> None:
        """ "Applied" and "written down" are different promises, and an operator who
        assumes the second when only the first is true loses the change at the next
        restart with nothing to tell them it happened."""
        operation = openapi_document()["paths"]["/cameras/{camera_id}"]["patch"]
        assert "cameras.json" in operation["description"]

    def test_the_camera_list_publishes_whether_writes_are_possible(self) -> None:
        """A console must be able to render the record read-only rather than
        offering a control that 403s."""
        schema = openapi_document()["components"]["schemas"]["CamerasResponse"]
        assert "config_writable" in schema["properties"]
        assert "config_writable" in schema["required"]

    def test_the_edit_request_admits_only_the_editable_fields(self) -> None:
        """`additionalProperties: false` is what makes a generated client's `url`
        field a compile-time impossibility rather than a runtime surprise.

        Checked against `EDITABLE_FIELDS` rather than a list written out here,
        because a field added to one and not the other is precisely the drift that
        tuple is published to prevent."""
        schema = openapi_document()["components"]["schemas"]["CameraEditRequest"]
        assert set(schema["properties"]) == set(EDITABLE_FIELDS)
        assert schema["additionalProperties"] is False

    def test_the_stored_record_that_comes_back_covers_every_editable_field(self) -> None:
        """A console that has just written renders the response, so a field it can
        edit and cannot read back is a control whose effect it has to guess at."""
        schema = openapi_document()["components"]["schemas"]["CameraEditResponse"]
        assert set(EDITABLE_FIELDS) <= set(schema["properties"])

    def test_the_camera_list_covers_every_editable_field_too(self) -> None:
        """`GET /cameras` is what a console renders when it has *not* just written —
        on first load, after a restart, and on the read-only deployments where the
        PATCH endpoint answers 403. A field readable only in an edit response is one a
        console can never show until someone changes it."""
        schema = openapi_document()["components"]["schemas"]["CameraStatus"]
        assert set(EDITABLE_FIELDS) <= set(schema["properties"])
        assert set(EDITABLE_FIELDS) - {"zone"} <= set(schema["required"]), (
            "the engine always knows this camera's policy, so a client must not have "
            "to treat it as optional"
        )


def test_the_stream_is_declared_as_an_event_stream_not_as_json() -> None:
    """A generated client that believes `/events/stream` returns
    `application/json` will try to decode the whole body as one document and hang
    until the stream ends — which, for a stream, is the point at which it is no longer
    useful."""
    responses = openapi_document()["paths"]["/events/stream"]["get"]["responses"]
    assert set(responses["200"]["content"]) == {"text/event-stream"}
    assert "503" in responses, "the engine-not-ready answer a client will actually meet"


def test_the_stream_contract_says_what_a_long_disconnect_costs() -> None:
    """The ring behind the stream is bounded and volatile. An integrator who reads
    only the contract must still learn that reconnecting gives them the current
    window and not everything they missed."""
    operation = openapi_document()["paths"]["/events/stream"]["get"]
    text = f"{operation['summary']} {operation['description']}".lower()
    assert "not the event store" in text
    assert "rabbitmq" in text
    assert "backlog" in text
    assert "last-event-id" in text


def test_the_events_endpoint_disclaims_being_the_event_store_in_the_contract() -> None:
    """The Phase 1C consumer author reads the generated contract, not this repo's
    docstrings. `/cameras/{camera_id}/events` is a volatile, silently-lossy console
    cache; the durable record is the anomaly event on RabbitMQ. Someone building
    against the contract must not be able to miss that.
    """
    operation = openapi_document()["paths"]["/cameras/{camera_id}/events"]["get"]
    text = f"{operation['summary']} {operation['description']}".lower()
    assert "not the event store" in text
    assert "audit trail" in text
    assert "rabbitmq" in text


def test_the_404_a_client_will_actually_meet_is_in_the_contract() -> None:
    """`UnknownCameraError` -> 404 is the one error the routes translate by hand, and
    a generated Go client that does not know about it will treat it as a transport
    failure."""
    responses = openapi_document()["paths"]["/cameras/{camera_id}/telemetry"]["get"]["responses"]
    assert "422" in responses, "FastAPI's own validation error shape"
    assert "200" in responses


def test_the_event_ring_always_carries_a_welfare_list_never_an_absent_one() -> None:
    """`welfare_concerns` is required, so a consumer reads a list unconditionally.

    Optional-with-a-default would generate `welfare_concerns?: Concern[]` in every
    downstream client, forcing each one to branch on presence before it can branch
    on emptiness. Two states that mean the same thing ("the model reported
    nothing") is one state too many, and the branch that conflates them with
    "assessed and found something" is the bug this system cannot afford.
    """
    entry = openapi_document()["components"]["schemas"]["RecentEventEntry"]
    assert "welfare_concerns" in entry["required"]


def test_the_event_ring_says_it_is_not_filtered_by_the_camera_notify_policy() -> None:
    """A Phase 1C consumer reading this endpoint must not assume it mirrors what was
    notified. `notify_on` narrows the note that leaves the building; it does not
    narrow this."""
    entry = openapi_document()["components"]["schemas"]["RecentEventEntry"]
    text = entry["properties"]["welfare_concerns"]["description"].lower()
    assert "notify_on" in text
    assert "not filtered" in text


def test_the_committed_document_matches_the_running_app() -> None:
    if not OPENAPI_PATH.exists():
        pytest.fail(f"{OPENAPI_PATH} is missing — run `{REGENERATE}`")
    committed = OPENAPI_PATH.read_text(encoding="utf-8")
    current = render(openapi_document())
    assert committed == current, (
        f"the committed OpenAPI document has drifted from the app — run `{REGENERATE}` "
        f"and commit {OPENAPI_PATH.name}"
    )
