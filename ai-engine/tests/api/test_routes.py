"""FastAPI surface tests (spec §5.7): thin delegation to EngineService, so
every test runs against a fake service and asserts status codes and response
shapes only — no business logic lives here to test.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from sentinel_ai.api.app import create_app
from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.orchestrator.event_history import CameraEventHistory, RecentEvent
from sentinel_ai.orchestrator.service import UnknownCameraError
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport, LifecycleState


class _FakeEngineService:
    def __init__(
        self,
        *,
        cameras: tuple[CameraTelemetry, ...] = (),
        health: dict[str, HealthReport] | None = None,
        describe_result: UUID | None = None,
        events: tuple[RecentEvent, ...] = (),
        capacity: int = 200,
    ) -> None:
        self._cameras = {t.camera_id: t for t in cameras}
        self._health = health or {}
        self._describe_result = describe_result or uuid4()
        self._events = events
        self._capacity = capacity
        self.started = False
        self.stopped = False

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


def _telemetry(camera_id: str = "cam-1") -> CameraTelemetry:
    return CameraTelemetry(
        camera_id=camera_id,
        frames_seen=100,
        frames_dropped=2,
        detections_run=98,
        escalations=3,
        escalations_dropped=0,
        discontinuities=1,
        last_frame_at=12.5,
        last_escalation_at=10.0,
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


def test_camera_telemetry_returns_the_expected_shape() -> None:
    service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras/cam-1/telemetry")

    assert response.status_code == 200
    assert response.json() == {
        "camera_id": "cam-1",
        "frames_seen": 100,
        "frames_dropped": 2,
        "detections_run": 98,
        "escalations": 3,
        "escalations_dropped": 0,
        "discontinuities": 1,
        "last_frame_at": 12.5,
        "last_escalation_at": 10.0,
    }


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
) -> RecentEvent:
    return RecentEvent(
        event_id=uuid4(),
        camera_id=camera_id,
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


def test_lifespan_starts_and_stops_the_service() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)):
        assert service.started is True
    assert service.stopped is True
