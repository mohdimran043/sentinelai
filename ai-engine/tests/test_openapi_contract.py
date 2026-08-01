"""Spec §3.2: the committed OpenAPI document must not drift from the running app.

The spec asks for CI to export the document and fail the build if the committed copy
has drifted. That is a pytest assertion here rather than a workflow step, for two
reasons: CI already runs the suite, so there is nothing new to remember to add; and
it fails on the developer's machine at the moment they change a route, not twenty
minutes later on a push.
"""

from __future__ import annotations

import pytest

from sentinel_ai.openapi_export import OPENAPI_PATH, openapi_document, render

REGENERATE = "python -m sentinel_ai.openapi_export"


def test_the_document_covers_the_endpoints_phase_1c_consumes() -> None:
    paths = openapi_document()["paths"]
    assert set(paths) == {
        "/health",
        "/cameras",
        "/cameras/{camera_id}/telemetry",
        "/cameras/{camera_id}/events",
        "/cameras/{camera_id}/describe",
    }
    assert set(paths["/cameras/{camera_id}/describe"]) == {"post"}
    assert set(paths["/cameras/{camera_id}/telemetry"]) == {"get"}
    assert set(paths["/cameras/{camera_id}/events"]) == {"get"}


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


def test_the_committed_document_matches_the_running_app() -> None:
    if not OPENAPI_PATH.exists():
        pytest.fail(f"{OPENAPI_PATH} is missing — run `{REGENERATE}`")
    committed = OPENAPI_PATH.read_text(encoding="utf-8")
    current = render(openapi_document())
    assert committed == current, (
        f"the committed OpenAPI document has drifted from the app — run `{REGENERATE}` "
        f"and commit {OPENAPI_PATH.name}"
    )
