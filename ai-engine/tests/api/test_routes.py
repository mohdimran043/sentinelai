"""FastAPI surface tests (spec §5.7): thin delegation to EngineService, so
every test runs against a fake service and asserts status codes and response
shapes only — no business logic lives here to test.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from sentinel_ai.adapters.config.camera_file import (
    UNSET,
    CameraConfig,
    CameraConfigError,
    CameraEdit,
)
from sentinel_ai.api.app import create_app
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.domain.welfare import ConcernKind, Confidence
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.event_history import (
    CameraEventHistory,
    EventSubscription,
    RecentEvent,
    RecentEventLog,
)
from sentinel_ai.orchestrator.service import UnknownCameraError
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport, LifecycleState

_WELFARE_POLICY_FIELDS = (
    "notify_on",
    "notify_min_confidence",
    "clip_preroll_seconds",
    "clip_postroll_seconds",
    "summary_interval_seconds",
)


class _FakeEngineService:
    def __init__(
        self,
        *,
        cameras: tuple[CameraTelemetry, ...] = (),
        health: dict[str, HealthReport] | None = None,
        describe_result: UUID | None = None,
        events: tuple[RecentEvent, ...] = (),
        capacity: int = 200,
        update_error: Exception | None = None,
    ) -> None:
        self._cameras = {t.camera_id: t for t in cameras}
        self._health = health or {}
        self._describe_result = describe_result or uuid4()
        self._events = events
        self._capacity = capacity
        self._update_error = update_error
        self.edits: list[tuple[str, CameraEdit]] = []
        self.started = False
        self.stopped = False
        # A real log, so the two stream methods below are not a second implementation
        # of the thing they stand in for. `tests/api/test_event_stream.py` is where the
        # stream itself is exercised.
        self._log = RecentEventLog(capacity=capacity)

    def subscribe_events(self) -> EventSubscription:
        return self._log.subscribe()

    def close_event_streams(self) -> int:
        return self._log.close_all()

    def event_history(self, camera_id: str) -> CameraEventHistory:
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        return CameraEventHistory(
            camera_id=camera_id,
            capacity=self._capacity,
            events=tuple(e for e in self._events if e.camera_id == camera_id),
        )

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return tuple(self._cameras.values())

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        return self._cameras[camera_id]

    def health(self) -> dict[str, HealthReport]:
        return self._health

    async def describe_now(self, camera_id: str) -> UUID:
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        return self._describe_result

    async def update_camera(self, camera_id: str, edit: CameraEdit) -> CameraConfig:
        """Records the edit and answers with the record it implies.

        Deliberately shallow: what the store does with an edit is
        `tests/adapters/test_camera_file.py`'s subject and the composed path's is
        `tests/test_main.py`'s. What these tests own is the translation between HTTP
        and `CameraEdit` — which is exactly what `self.edits` captures.
        """
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        if self._update_error is not None:
            raise self._update_error
        self.edits.append((camera_id, edit))
        current = self._cameras[camera_id]
        return CameraConfig(
            camera_id=camera_id,
            label=current.label if edit.label is UNSET else edit.label,
            url="rtsp://host/stream",
            profile=CameraProfile(camera_id=camera_id),
            zone=current.zone if edit.zone is UNSET else edit.zone,
            # The welfare policy has no "current" here to fall back to: it is not on
            # `CameraTelemetry` (Task 9 puts it there), so an unmentioned field lands
            # on `CameraConfig`'s own default, which is what a camera the file says
            # nothing about would load as anyway.
            **{
                name: getattr(edit, name)
                for name in _WELFARE_POLICY_FIELDS
                if getattr(edit, name) is not UNSET
            },
        )


def _telemetry(
    camera_id: str = "cam-1", *, zone: Zone | None = None, label: str | None = None
) -> CameraTelemetry:
    return CameraTelemetry(
        camera_id=camera_id,
        label=label if label is not None else camera_id,
        frames_seen=100,
        frames_dropped=2,
        detections_run=98,
        escalations=3,
        escalations_dropped=0,
        discontinuities=1,
        last_frame_at=12.5,
        last_escalation_at=10.0,
        zone=zone,
    )


def test_health_returns_per_model_lifecycle_state() -> None:
    service = _FakeEngineService(
        health={"yolo11": HealthReport(state=LifecycleState.LOADED, detail="", vram_mib=1400)}
    )
    with TestClient(create_app(service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models"] == [{"key": "yolo11", "state": "loaded", "detail": "", "vram_mib": 1400}]


def test_list_cameras_returns_configured_cameras() -> None:
    service = _FakeEngineService(cameras=(_telemetry("cam-1"), _telemetry("cam-2")))
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras")

    assert response.status_code == 200
    ids = [camera["camera_id"] for camera in response.json()["cameras"]]
    assert ids == ["cam-1", "cam-2"]


def test_the_camera_list_carries_the_label_apart_from_the_id() -> None:
    """The console navigates by label and writes back by id, so the two must be
    separately readable. A `label` quietly serialised from `camera_id` would look
    right on every camera whose label was never changed — including every camera in
    the shipped example — and would make an edit look like it never landed."""
    service = _FakeEngineService(cameras=(_telemetry("cam-1", label="Front door"),))
    with TestClient(create_app(service)) as client:
        (camera,) = client.get("/cameras").json()["cameras"]

    assert (camera["camera_id"], camera["label"]) == ("cam-1", "Front door")


def test_camera_telemetry_returns_the_expected_shape() -> None:
    service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras/cam-1/telemetry")

    assert response.status_code == 200
    assert response.json() == {
        "camera_id": "cam-1",
        "label": "cam-1",
        "frames_seen": 100,
        "frames_dropped": 2,
        "detections_run": 98,
        "escalations": 3,
        "escalations_dropped": 0,
        "discontinuities": 1,
        "last_frame_at": 12.5,
        "last_escalation_at": 10.0,
        "zone": None,
        "zone_kind": None,
    }


class TestCameraZone:
    """T1 on the wire: `GET /cameras` is what a console groups by."""

    def test_cameras_are_grouped_by_zone_with_the_kind_derived(self) -> None:
        service = _FakeEngineService(
            cameras=(
                _telemetry("room-2a", zone=Zone.ROOM),
                _telemetry("corridor-1", zone=Zone.CORRIDOR),
                _telemetry("dayroom-1", zone=Zone.DAYROOM),
            )
        )
        with TestClient(create_app(service)) as client:
            cameras = client.get("/cameras").json()["cameras"]

        assert [(c["camera_id"], c["zone"], c["zone_kind"]) for c in cameras] == [
            ("room-2a", "room", "room"),
            ("corridor-1", "corridor", "common_area"),
            ("dayroom-1", "dayroom", "common_area"),
        ]

    def test_an_ungrouped_camera_is_null_rather_than_a_group_called_unknown(self) -> None:
        """Null says "nobody has grouped this camera". A sentinel string would make
        every ungrouped camera a member of one made-up zone, which a console then draws
        as a real group."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as client:
            (camera,) = client.get("/cameras").json()["cameras"]

        assert camera["zone"] is None
        assert camera["zone_kind"] is None

    def test_the_contract_constrains_the_zone_to_the_known_vocabulary(self) -> None:
        """A generated Go client should get an enum, not a free string: the whole
        reason the field is constrained is that two operators typing the same idea have
        to produce the same group."""
        schema = create_app(_FakeEngineService()).openapi()["components"]["schemas"]
        zone = schema["Zone"]
        assert set(zone["enum"]) == {"room", "corridor", "dayroom"}
        assert set(schema["ZoneKind"]["enum"]) == {"room", "common_area"}


def test_camera_telemetry_for_an_unknown_camera_is_404() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras/does-not-exist/telemetry")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown camera: does-not-exist"}, (
        "UnknownCameraError subclasses KeyError, whose __str__ is repr(args[0]); "
        "str(exc) would leak literal quotes onto the wire"
    )


def test_describe_now_returns_the_event_id() -> None:
    event_id = uuid4()
    service = _FakeEngineService(cameras=(_telemetry("cam-1"),), describe_result=event_id)
    with TestClient(create_app(service)) as client:
        response = client.post("/cameras/cam-1/describe")

    assert response.status_code == 200
    assert response.json() == {"event_id": str(event_id)}


def test_describe_now_for_an_unknown_camera_is_404() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)) as client:
        response = client.post("/cameras/does-not-exist/describe")

    assert response.status_code == 404
    assert response.json() == {"detail": "unknown camera: does-not-exist"}, (
        "UnknownCameraError subclasses KeyError, whose __str__ is repr(args[0]); "
        "str(exc) would leak literal quotes onto the wire"
    )


def _recent_event(
    camera_id: str = "cam-1",
    *,
    occurred_at: float = 100.0,
    threat_score: float = 0.3,
    severity: Severity = Severity.LOW,
    description: str = "A person walks past the door.",
    description_unavailable: bool = False,
    clip_uri: str | None = None,
    sequence: int = 1,
) -> RecentEvent:
    return RecentEvent(
        event_id=uuid4(),
        camera_id=camera_id,
        sequence=sequence,
        clip_uri=clip_uri,
        occurred_at=occurred_at,
        source_timestamp=occurred_at - 90.0,
        reason=EscalationReason.NEW_SALIENT_TRACK,
        threat_score=threat_score,
        severity=severity,
        description=description,
        suggested_action="Monitor.",
        description_unavailable=description_unavailable,
        labels=("person",),
        track_ids=(7,),
    )


class TestCameraEvents:
    def test_the_events_endpoint_returns_the_ring_oldest_first(self) -> None:
        """Chronological, because the console plots threat against time directly off
        this list."""
        events = (
            _recent_event(occurred_at=10.0, threat_score=0.1),
            _recent_event(occurred_at=20.0, threat_score=0.9, severity=Severity.CRITICAL),
        )
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=events)
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        assert response.status_code == 200
        body = response.json()
        assert body["camera_id"] == "cam-1"
        assert [e["occurred_at"] for e in body["events"]] == [10.0, 20.0]
        assert [e["threat_score"] for e in body["events"]] == [0.1, 0.9]
        assert body["returned"] == 2
        assert body["capacity"] == 200

    def test_the_response_says_it_is_volatile_and_bounded(self) -> None:
        """The distinction the endpoint exists to protect: this is a console cache,
        not the event store. A consumer must be able to see the bound (`capacity`) and
        the volatility (`volatile`) in the payload itself, not only in prose it may
        never read."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), capacity=200)
        with TestClient(create_app(service)) as client:
            body = client.get("/cameras/cam-1/events").json()

        assert body["volatile"] is True
        assert body["capacity"] == 200

    def test_a_camera_that_has_not_escalated_yet_says_so_plainly(self) -> None:
        """Nothing has happened — a 200 with an empty list and an explicit `none`
        state, never a 404 and never a bare null that could mean anything."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        assert response.status_code == 200
        body = response.json()
        assert body["events"] == []
        assert body["returned"] == 0
        assert body["latest"] is None
        assert body["latest_description_state"] == "none"

    def test_the_latest_description_is_lifted_out_for_the_live_panel(self) -> None:
        events = (
            _recent_event(occurred_at=10.0, description="Earlier."),
            _recent_event(occurred_at=20.0, description="A person walks past the door."),
        )
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=events)
        with TestClient(create_app(service)) as client:
            body = client.get("/cameras/cam-1/events").json()

        assert body["latest_description_state"] == "available"
        assert body["latest"]["description"] == "A person walks past the door."
        assert body["latest"]["occurred_at"] == 20.0
        assert body["latest"] == body["events"][-1], (
            "the live panel and the chart must be reading the same snapshot"
        )

    def test_a_vlm_that_could_not_answer_is_not_reported_as_nothing_happening(self) -> None:
        """The distinction T2 exists for. A console that cannot tell these apart shows
        a metadata-derived stand-in as though the model had described the scene, or —
        worse — shows nothing at all and tells an operator the camera is quiet.

        Fails against any design that only offers a nullable `description`: the two
        cases would be indistinguishable once the description is present but is not a
        description.
        """
        events = (
            _recent_event(
                occurred_at=20.0,
                description="new_salient_track: person (dwell 31s)",
                description_unavailable=True,
            ),
        )
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=events)
        with TestClient(create_app(service)) as client:
            body = client.get("/cameras/cam-1/events").json()

        assert body["latest_description_state"] == "unavailable"
        assert body["latest_description_state"] != "none", "the model failing is not silence"
        assert body["latest"] is not None
        assert body["latest"]["description_unavailable"] is True

    def test_an_event_carries_the_clip_uri_so_a_notification_can_link_to_it(self) -> None:
        """T3. Without this the camera page can show that something happened and can
        show what the model said about it, but cannot offer the footage."""
        events = (_recent_event(clip_uri="s3://sentinel-clips/cam-1/abc.mp4"),)
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=events)
        with TestClient(create_app(service)) as client:
            body = client.get("/cameras/cam-1/events").json()

        assert body["events"][0]["clip_uri"] == "s3://sentinel-clips/cam-1/abc.mp4"
        assert body["latest"]["clip_uri"] == "s3://sentinel-clips/cam-1/abc.mp4"

    def test_an_event_whose_clip_failed_says_null_rather_than_omitting_the_field(self) -> None:
        """`clip_uri: null` is a real answer — the event happened and there is no
        footage — and a console must be able to tell it from a field it forgot to
        read."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=(_recent_event(),))
        with TestClient(create_app(service)) as client:
            body = client.get("/cameras/cam-1/events").json()

        assert body["events"][0]["clip_uri"] is None

    def test_events_for_an_unknown_camera_is_404(self) -> None:
        service = _FakeEngineService()
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/does-not-exist/events")

        assert response.status_code == 404
        assert response.json() == {"detail": "unknown camera: does-not-exist"}, (
            "UnknownCameraError subclasses KeyError, whose __str__ is repr(args[0]); "
            "str(exc) would leak literal quotes onto the wire"
        )

    def test_the_openapi_description_warns_that_this_is_not_the_event_store(self) -> None:
        """The warning has to live where an integrator reads it — the generated
        contract — not only in a Python docstring. This is a custodial system: an
        operator who mistakes a ring that silently drops old entries for an audit
        trail will conclude an event never happened when it merely aged out.
        """
        service = _FakeEngineService()
        app = create_app(service)
        operation = app.openapi()["paths"]["/cameras/{camera_id}/events"]["get"]
        text = f"{operation['summary']} {operation['description']}".lower()

        assert "not the event store" in text
        assert "volatile" in text
        assert "restart" in text
        assert "oldest" in text or "evict" in text or "discard" in text


class TestCameraWritesAreOffByDefault:
    """The gate, and the reason it exists.

    This build ships no authentication of any kind, so a write endpoint is
    reachable by anything that can open a socket to the port. The endpoint is
    therefore opt-in per deployment, and the default posture is the one a
    deployment gets when nobody has thought about it.
    """

    def test_the_default_app_refuses_to_write(self) -> None:
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as client:
            response = client.patch("/cameras/cam-1", json={"label": "Renamed"})

        assert response.status_code == 403
        assert service.edits == [], "the service must not be reached at all"

    def test_the_refusal_names_the_setting_and_the_reason(self) -> None:
        """A 403 with no explanation sends an operator hunting for a login page
        that does not exist."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as client:
            detail = client.patch("/cameras/cam-1", json={"label": "R"}).json()["detail"]

        assert "SENTINEL_ENABLE_CAMERA_WRITES" in detail
        assert "no authentication" in detail
        assert "cameras.json" in detail

    def test_the_gate_is_checked_before_the_camera_exists(self) -> None:
        """A disabled deployment must not answer 404 for one id and 403 for another:
        that is an enumeration oracle for the camera list, handed out for free to
        exactly the caller the gate is there to keep out."""
        service = _FakeEngineService()
        with TestClient(create_app(service)) as client:
            response = client.patch("/cameras/does-not-exist", json={"label": "R"})

        assert response.status_code == 403

    def test_the_camera_list_reports_the_posture(self) -> None:
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as closed:
            assert closed.get("/cameras").json()["config_writable"] is False
        with TestClient(create_app(service, camera_writes_enabled=True)) as open_:
            assert open_.get("/cameras").json()["config_writable"] is True

    def test_the_route_exists_in_the_contract_either_way(self) -> None:
        """A contract that changes shape with a runtime flag is worse than a
        documented 403: a client generated against a writable engine would not
        compile against a read-only one, and the same binary serves both."""
        closed = create_app(_FakeEngineService()).openapi()["paths"]
        open_ = create_app(_FakeEngineService(), camera_writes_enabled=True).openapi()["paths"]

        assert "/cameras/{camera_id}" in closed
        assert closed == open_


class TestCameraEdit:
    def app(self) -> tuple[_FakeEngineService, TestClient]:
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1", label="Front door", zone=Zone.CORRIDOR),)
        )
        return service, TestClient(create_app(service, camera_writes_enabled=True))

    def test_a_label_edit_returns_the_stored_record(self) -> None:
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"label": "Back door"})

        assert response.status_code == 200
        assert response.json() == {
            "camera_id": "cam-1",
            "label": "Back door",
            "zone": "corridor",
            "zone_kind": "common_area",
            # Echoed even though this edit did not name them: the response is the
            # record as it now stands, not a diff. A camera whose file says nothing
            # about `notify_on` notifies on every kind, and saying so explicitly is
            # what stops a console rendering "no kinds" for it.
            "notify_on": sorted(ConcernKind),
            "notify_min_confidence": "likely",
            "clip_preroll_seconds": None,
            "clip_postroll_seconds": None,
            "summary_interval_seconds": None,
            "persisted": True,
            "restart_required_fields": ["url", "profile"],
        }
        assert service.edits == [("cam-1", CameraEdit(label="Back door", zone=UNSET))]

    def test_a_zone_edit_does_not_touch_the_label(self) -> None:
        """The partial-edit contract, on the wire. A console changing a zone sends
        no label, and the engine must not read that as "clear the label"."""
        service, client = self.app()
        with client:
            body = client.patch("/cameras/cam-1", json={"zone": "room"}).json()

        assert body["label"] == "Front door"
        assert service.edits == [("cam-1", CameraEdit(label=UNSET, zone=Zone.ROOM))]

    def test_an_explicit_null_zone_ungroups_and_an_omitted_one_does_not(self) -> None:
        """The single most important translation this layer performs: pydantic gives
        both cases the same attribute value, and only `model_fields_set` tells them
        apart. Getting it wrong silently ungroups every camera anyone renames."""
        service, client = self.app()
        with client:
            ungrouped = client.patch("/cameras/cam-1", json={"zone": None}).json()
            client.patch("/cameras/cam-1", json={"label": "Renamed"})

        assert ungrouped["zone"] is None
        assert ungrouped["zone_kind"] is None
        assert service.edits == [
            ("cam-1", CameraEdit(label=UNSET, zone=None)),
            ("cam-1", CameraEdit(label="Renamed", zone=UNSET)),
        ]

    def test_a_label_is_trimmed_before_it_is_stored(self) -> None:
        service, client = self.app()
        with client:
            client.patch("/cameras/cam-1", json={"label": "  Back door  "})

        assert service.edits == [("cam-1", CameraEdit(label="Back door", zone=UNSET))]

    @pytest.mark.parametrize("label", ["", "   ", None])
    def test_a_label_that_is_not_a_label_is_rejected(self, label: object) -> None:
        """Whitespace is the interesting one: `load_cameras` would accept `"   "` as
        a non-empty string and the console would render a nameless camera."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"label": label})

        assert response.status_code == 422
        assert service.edits == []

    def test_an_unknown_zone_is_rejected_rather_than_dropped(self) -> None:
        """The same fail-loud `load_cameras` applies at startup: an operator who
        typed `hallway` meant to group that camera and did not."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"zone": "hallway"})

        assert response.status_code == 422
        assert service.edits == []

    @pytest.mark.parametrize("field", ["url", "profile", "camera_id", "id", "zone_kind", "labe1"])
    def test_a_field_this_endpoint_will_not_change_is_rejected_not_ignored(
        self, field: str
    ) -> None:
        """The defect this endpoint must never have: an operator re-points a camera
        at a new stream, is told it worked, and watches the old stream for a week.
        A 422 naming the field is the only honest answer — and a typo'd `labe1`
        accepted as a no-op is the same failure wearing a smaller hat."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={field: "anything"})

        assert response.status_code == 422
        assert field in response.text
        assert service.edits == []

    def test_a_url_edit_is_rejected_even_alongside_a_valid_label(self) -> None:
        """Half-applying is worse than refusing: the operator would see the rename
        land and reasonably assume the URL did too."""
        service, client = self.app()
        with client:
            response = client.patch(
                "/cameras/cam-1", json={"label": "Renamed", "url": "rtsp://elsewhere/one"}
            )

        assert response.status_code == 422
        assert service.edits == []

    def test_an_edit_naming_nothing_is_rejected(self) -> None:
        """A 200 here would report a successful save for a request that changed
        nothing, and the operator would believe their edit landed."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={})

        assert response.status_code == 422
        assert service.edits == []

    def test_an_unknown_camera_is_404(self) -> None:
        _, client = self.app()
        with client:
            response = client.patch("/cameras/does-not-exist", json={"label": "R"})

        assert response.status_code == 404
        assert response.json() == {"detail": "unknown camera: does-not-exist"}

    def test_a_file_that_cannot_take_the_edit_is_409_not_500(self) -> None:
        """The request was fine and the engine accepted it; the file it must be
        written into is not in a state that admits it. A client can retry a 409
        after reconciling, which is not true of a 500."""
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),),
            update_error=CameraConfigError("cameras.json has no camera 'cam-1'"),
        )
        with TestClient(create_app(service, camera_writes_enabled=True)) as client:
            response = client.patch("/cameras/cam-1", json={"label": "R"})

        assert response.status_code == 409
        assert "no camera" in response.json()["detail"]

    def test_a_write_failure_says_nothing_was_changed(self) -> None:
        """An operator who sees a bare 500 does not know whether to retry or to go
        and check the file."""
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),), update_error=OSError("no space left on device")
        )
        with TestClient(create_app(service, camera_writes_enabled=True)) as client:
            response = client.patch("/cameras/cam-1", json={"label": "R"})

        assert response.status_code == 500
        assert "nothing was changed" in response.json()["detail"]


class TestCameraWelfarePolicyEdit:
    """T8. The per-camera welfare policy over `PATCH /cameras/{camera_id}`.

    Nothing in the engine reads these fields yet — Task 9 makes the runner honour
    the clip/summary overrides, Task 10 routes notifications on the other two — so
    what these tests own is exactly what this layer owns: that the HTTP body
    becomes the right `CameraEdit`, that a value the file would refuse is a 422
    here rather than a 409 from the store, and that the response says what was
    stored.
    """

    def app(self) -> tuple[_FakeEngineService, TestClient]:
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1", label="Front door", zone=Zone.CORRIDOR),)
        )
        return service, TestClient(create_app(service, camera_writes_enabled=True))

    def test_the_routing_is_editable_and_the_stored_record_comes_back(self) -> None:
        service, client = self.app()
        with client:
            response = client.patch(
                "/cameras/cam-1",
                json={"notify_on": ["self_harm", "collapse"], "notify_min_confidence": "possible"},
            )

        body = response.json()
        assert response.status_code == 200
        assert body["notify_on"] == ["collapse", "self_harm"]
        assert body["notify_min_confidence"] == "possible"
        assert service.edits == [
            (
                "cam-1",
                CameraEdit(
                    notify_on=frozenset({ConcernKind.COLLAPSE, ConcernKind.SELF_HARM}),
                    notify_min_confidence=Confidence.POSSIBLE,
                ),
            )
        ]

    def test_muting_a_camera_is_an_edit_and_not_an_empty_request(self) -> None:
        """`notify_on: []` is the one instruction this endpoint could most easily
        mistake for "asked for nothing" — the body names one field and that field
        is empty. Rejecting it as empty would leave an operator with no way to
        silence a camera short of editing the file."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"notify_on": []})

        assert response.status_code == 200
        assert response.json()["notify_on"] == []
        assert service.edits == [("cam-1", CameraEdit(notify_on=frozenset()))]

    @pytest.mark.parametrize("field", ["notify_on", "notify_min_confidence"])
    def test_a_null_routing_field_is_rejected_and_says_what_to_send_instead(
        self, field: str
    ) -> None:
        """Neither field has a "no value" state: `[]` already means "never notify",
        and a camera always has some confidence threshold. A null accepted as
        "unchanged" would report a save for a request that changed nothing."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={field: None})

        assert response.status_code == 422
        assert field in response.text
        assert "omit" in response.text
        assert service.edits == []

    def test_an_unknown_concern_kind_is_rejected_rather_than_dropped(self) -> None:
        """Dropping it would silently narrow the routing to the kinds that happened
        to be spelled right — a camera an operator believes is watched for
        `self_harm` and is not."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"notify_on": ["fainting"]})

        assert response.status_code == 422
        assert service.edits == []

    def test_a_confidence_tier_that_does_not_exist_is_rejected(self) -> None:
        """There is no `certain` tier and there never will be: a single still frame
        cannot earn one (see `domain/welfare.py`). A caller asking for it is asking
        for a threshold nothing can clear."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"notify_min_confidence": "certain"})

        assert response.status_code == 422
        assert service.edits == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("clip_preroll_seconds", 2.5),
            ("clip_postroll_seconds", 4.0),
            ("summary_interval_seconds", 30.0),
        ],
    )
    def test_a_duration_override_is_editable_and_comes_back(self, field: str, value: float) -> None:
        service, client = self.app()
        expected: dict[str, Any] = {field: value}
        with client:
            response = client.patch("/cameras/cam-1", json={field: value})

        assert response.status_code == 200
        assert response.json()[field] == value
        assert service.edits == [("cam-1", CameraEdit(**expected))]

    @pytest.mark.parametrize(
        "field",
        ["clip_preroll_seconds", "clip_postroll_seconds", "summary_interval_seconds"],
    )
    def test_an_omitted_duration_and_an_explicit_null_are_different_instructions(
        self, field: str
    ) -> None:
        """The translation this layer exists to get right, for the three fields
        where both instructions are legal. `null` reverts the camera to the global
        default; omitting it leaves whatever the operator tuned. Pydantic gives both
        the same attribute value and only `model_fields_set` tells them apart, so a
        conflation here is invisible until someone's tuned pre-roll silently
        resets."""
        service, client = self.app()
        reverting: dict[str, Any] = {field: None}
        with client:
            reverted = client.patch("/cameras/cam-1", json={field: None})
            client.patch("/cameras/cam-1", json={"label": "Renamed"})

        assert reverted.json()[field] is None
        assert service.edits == [
            # The second edit leaves this field `UNSET` — `CameraEdit`'s default —
            # which is what makes the two lines below different objects. If omitted
            # and null were conflated, both edits would carry the same value here
            # and this assertion would pass whichever way round the conflation went.
            ("cam-1", CameraEdit(**reverting)),
            ("cam-1", CameraEdit(label="Renamed")),
        ]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("clip_preroll_seconds", -0.5),
            ("clip_postroll_seconds", 0),
            ("summary_interval_seconds", 0),
            ("summary_interval_seconds", -1),
        ],
    )
    def test_a_duration_outside_its_bounds_is_a_422_not_a_conflict(
        self, field: str, value: float
    ) -> None:
        """The store's re-parse would also refuse these, but as a `CameraConfigError`
        — which this endpoint answers 409, meaning "the file is in a state that will
        not take your edit". That is a lie about a request that was simply wrong, and
        it points the operator at the file instead of at their own input."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={field: value})

        assert response.status_code == 422
        assert service.edits == []

    @pytest.mark.parametrize(
        "field",
        ["clip_preroll_seconds", "clip_postroll_seconds", "summary_interval_seconds"],
    )
    @pytest.mark.parametrize("literal", ["-1e999", "1e999", "NaN"])
    def test_a_non_finite_duration_is_a_422_and_not_a_crash(self, field: str, literal: str) -> None:
        """`1e999` and `NaN` are things `json.loads` accepts and `math.isfinite`
        does not, so they are the one out-of-range family that can arrive without
        looking out of range. They have to be answered here for the same reason
        every other bound is — but they also have to be answered *renderably*: the
        422 body echoes the input that failed, and a bare `Infinity` in it is not
        JSON that Starlette will serialise. An unhandled serialisation failure turns
        this into a 500, which tells an operator the engine is broken when their
        request was.

        Sent as a raw body because `json=` cannot express these: Python's
        `json.dumps` will emit them (its `allow_nan` defaults to true) but
        `requests`/`httpx` build the body themselves, so the literal has to be
        written out. A Python client doing exactly that is how this arrives in
        practice.
        """
        service, client = self.app()
        with client:
            response = client.patch(
                "/cameras/cam-1",
                content=f'{{"{field}": {literal}}}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 422
        assert service.edits == []

    def test_a_non_finite_duration_beside_another_bad_field_is_still_a_422(self) -> None:
        """The same value, reached through the model-level validator rather than the
        field's own bound — `label: null` is what trips it. That path echoes the
        *whole* body as the failing input, so the infinity lands in the 422 even
        though nothing about the infinity is what failed. Worth its own case: a fix
        that only sanitises the field that was out of range leaves this one
        crashing."""
        service, client = self.app()
        with client:
            response = client.patch(
                "/cameras/cam-1",
                content='{"label": null, "clip_postroll_seconds": 1e999}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 422
        assert service.edits == []

    def test_no_pre_roll_at_all_is_a_real_choice(self) -> None:
        """Pre-roll's bound is `>= 0` where the other two are `> 0`, exactly as at
        load time: a clip with no lead-in is legitimate, a clip of no length is
        not."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"clip_preroll_seconds": 0})

        assert response.status_code == 200
        assert response.json()["clip_preroll_seconds"] == 0.0
        assert service.edits == [("cam-1", CameraEdit(clip_preroll_seconds=0.0))]

    def test_a_policy_edit_alongside_a_field_that_needs_a_restart_is_still_refused(self) -> None:
        """Adding editable fields must not have widened what the body may carry: a
        `url` smuggled in beside a legitimate policy change is still a 422, and
        still changes nothing."""
        service, client = self.app()
        with client:
            response = client.patch(
                "/cameras/cam-1", json={"notify_on": [], "url": "rtsp://elsewhere/one"}
            )

        assert response.status_code == 422
        assert "url" in response.text
        assert service.edits == []


class TestValidationErrorRendering:
    """`create_app`'s `RequestValidationError` handler, which is app-wide and
    therefore owns the 422 body of *every* endpoint, not just the one that needed
    it. The properties below are the reason it can be there at all."""

    def app(self) -> TestClient:
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        return TestClient(create_app(service, camera_writes_enabled=True))

    def test_an_ordinary_422_keeps_fastapis_shape(self) -> None:
        """A console parses these. Replacing the default handler must be invisible
        to it, so the body is asserted key by key rather than "is a 422": the
        failure this guards against is a handler that answers the right status with
        a shape nothing downstream can read."""
        with self.app() as client:
            response = client.patch("/cameras/cam-1", json={"clip_postroll_seconds": 0})

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "greater_than",
                    "loc": ["body", "clip_postroll_seconds"],
                    "msg": "Input should be greater than 0",
                    "input": 0,
                    "ctx": {"gt": 0.0},
                }
            ]
        }

    def test_a_non_finite_input_is_echoed_back_as_null(self) -> None:
        """Null rather than dropped or stringified: the key stays where a client
        expects it, and `null` is what `JSON.stringify` and pydantic's `to_json`
        would have made of the same value. Only the unserialisable float changes —
        the type, location and message are the ones pydantic produced."""
        with self.app() as client:
            response = client.patch(
                "/cameras/cam-1",
                content='{"clip_preroll_seconds": 1e999}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "finite_number",
                    "loc": ["body", "clip_preroll_seconds"],
                    "msg": "Input should be a finite number",
                    "input": None,
                }
            ]
        }

    def test_a_non_finite_inside_the_echoed_input_is_nulled_too(self) -> None:
        """The property the whole handler turns on, and the one a plausible
        simplification loses: the offending float is not always *at* `input`, it can
        be anywhere inside it. A list where a float belongs makes pydantic echo the
        list back, so the infinity is one level down and a sanitiser that only
        checks whether `input` is itself a non-finite float leaves it there — a 500,
        with every other test in this file still green. Hence the recursive walk in
        `_with_non_finite_floats_nulled`."""
        with self.app() as client:
            response = client.patch(
                "/cameras/cam-1",
                content='{"clip_preroll_seconds": [1e999]}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "float_type",
                    "loc": ["body", "clip_preroll_seconds"],
                    "msg": "Input should be a valid number",
                    "input": [None],
                }
            ]
        }

    def test_a_non_finite_reaches_the_body_through_a_field_that_is_not_a_float(self) -> None:
        """Nothing about this is specific to the duration fields that motivated the
        handler. `notify_on` takes an enum, and an infinity in the array is rejected
        by the enum check — but it is still echoed back, so the 422 is still
        unrenderable without the handler. This is the general shape of the bug: any
        field of any type, on any endpoint, because JSON can carry a non-finite
        float into any of them."""
        with self.app() as client:
            response = client.patch(
                "/cameras/cam-1",
                content='{"notify_on": [1e999]}',
                headers={"content-type": "application/json"},
            )

        assert response.status_code == 422
        assert response.json() == {
            "detail": [
                {
                    "type": "enum",
                    "loc": ["body", "notify_on", 0],
                    "msg": (
                        "Input should be 'collapse', 'altercation', 'self_harm', "
                        "'medication', 'distress' or 'other'"
                    ),
                    "input": None,
                    "ctx": {
                        "expected": (
                            "'collapse', 'altercation', 'self_harm', 'medication', "
                            "'distress' or 'other'"
                        )
                    },
                }
            ]
        }


def test_lifespan_starts_and_stops_the_service() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)):
        assert service.started is True
    assert service.stopped is True
