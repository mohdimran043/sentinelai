"""FastAPI surface tests (spec §5.7): thin delegation to EngineService, so
every test runs against a fake service and asserts status codes and response
shapes only — no business logic lives here to test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from sentinel_ai.adapters.config.camera_file import (
    UNSET,
    CameraConfig,
    CameraConfigError,
    CameraCreate,
    CameraEdit,
)
from sentinel_ai.adapters.face.encrypted_store import EncryptedFaceStore, generate_key
from sentinel_ai.adapters.sources.probe import ProbeResult
from sentinel_ai.api.app import create_app
from sentinel_ai.domain.alert import Alert
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.capabilities import (
    DEFAULT_CAPABILITIES,
    CameraCapabilities,
    Capability,
    ModelRole,
)
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore
from sentinel_ai.domain.identity import AuthorizedPerson, EnrolledFace, FaceEmbedding
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.domain.zone import Zone
from sentinel_ai.orchestrator.alerts import AlertRegister
from sentinel_ai.orchestrator.event_history import (
    CameraEventHistory,
    EventSubscription,
    RecentEvent,
    RecentEventLog,
)
from sentinel_ai.orchestrator.service import (
    CapabilityUnavailableError,
    FaceCapabilityUnavailableError,
    UnknownCameraError,
    UnknownPersonError,
)
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.face import FaceStore
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
        alert_register: AlertRegister | None = None,
        face_store: FaceStore | None = None,
        enroll_error: Exception | None = None,
    ) -> None:
        self._cameras = {t.camera_id: t for t in cameras}
        self._health = health or {}
        self._describe_result = describe_result or uuid4()
        self._events = events
        self._capacity = capacity
        self._update_error = update_error
        # A real register rather than a stub list: what these tests own is the HTTP
        # translation, and the register's own behaviour is `test_alerts.py`'s subject.
        # Using the real one means a route cannot pass here against semantics the
        # register does not actually have.
        self._register = alert_register if alert_register is not None else AlertRegister()
        self.alert_flushes = 0
        # What `read_alert_clip` will answer with, and what it was asked. `None` is the
        # ordinary "nothing recorded" case, which is most alerts most of the time.
        self.clip_bytes: bytes | None = None
        self.clip_requests: list[tuple[UUID, bool]] = []
        self._face_store = face_store
        self._enroll_error = enroll_error
        self._alerts = self._register.snapshot()
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

    async def list_people(self) -> tuple[tuple[AuthorizedPerson, int], ...]:
        if self._face_store is None:
            return ()
        people = await self._face_store.list_people()
        out: list[tuple[AuthorizedPerson, int]] = []
        for person in people:
            out.append((person, await self._face_store.reference_count(person.person_id)))
        return tuple(out)

    async def upsert_person(self, person: AuthorizedPerson) -> AuthorizedPerson:
        if self._face_store is None:
            raise FaceCapabilityUnavailableError()
        await self._face_store.add_person(person)
        return person

    async def delete_person(self, person_id: UUID) -> bool:
        if self._face_store is None:
            raise FaceCapabilityUnavailableError()
        return await self._face_store.delete_person(person_id)

    async def enroll_face(
        self, person_id: UUID, image: object, *, original: bytes | None = None
    ) -> int:
        """A stand-in for the detect-embed-store chain.

        These tests own the HTTP translation; the chain itself needs a real face model
        and is exercised in a `gpu`-marked test. `self._enroll_error` lets a test choose
        which of the documented failures to provoke.
        """
        if self._face_store is None:
            raise FaceCapabilityUnavailableError()
        if self._enroll_error is not None:
            raise self._enroll_error
        if await self._face_store.get_person(person_id) is None:
            raise UnknownPersonError(person_id)
        await self._face_store.add_embedding(person_id, FaceEmbedding.of((1.0, 0.0, 0.0)))
        return await self._face_store.reference_count(person_id)

    def alerts(self) -> tuple[Alert, ...]:
        return self._register.snapshot()

    def acknowledge_alert(self, alert_id: UUID, *, by: str, at: float) -> Alert:
        return self._register.acknowledge(alert_id, by=by, at=at)

    def resolve_alert(self, alert_id: UUID) -> Alert:
        return self._register.resolve(alert_id)

    async def read_alert_clip(self, alert_id: UUID, *, short: bool) -> bytes | None:
        """Records which variant was asked for and hands back a recognisable stand-in.

        Real MP4 bytes would prove nothing these tests are about: what the route owes
        is the right object, the right media type and the right status, and a fake that
        returns distinguishable bytes per variant is what makes "it served the short
        one" assertable at all.

        The register lookup is not decoration. `EngineService.read_alert_clip` resolves
        the URI *from the alert* — that is the whole of the authorisation — so a fake
        that skipped it would hand out bytes for an id the engine has never heard of,
        and the route's 404 path would be tested against a service that cannot produce
        it.
        """
        self._register.get(alert_id)
        self.clip_requests.append((alert_id, short))
        if self.clip_bytes is None:
            return None
        return self.clip_bytes if short else self.clip_bytes + b"-full"

    async def flush_alerts(self) -> None:
        """Counted rather than performed, so a test can assert that the handler asked
        for a write before answering — which is the whole promise of a 200 on the two
        operator endpoints."""
        self.alert_flushes += 1

    def clear_alerts(self) -> int:
        return 0

    async def snapshot(self, camera_id: str) -> bytes | None:
        return None

    async def create_camera(self, create: CameraCreate) -> CameraConfig:
        raise NotImplementedError("this fake does not exercise the create path")

    async def delete_camera(self, camera_id: str) -> None:
        raise UnknownCameraError(camera_id)

    async def probe_source(self, url: str) -> ProbeResult:
        return ProbeResult(ok=False, detail="probing is not faked here", source_kind="file")

    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        raise FaceCapabilityUnavailableError()

    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        return None

    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        raise FaceCapabilityUnavailableError()

    def last_frame_epoch(self, camera_id: str) -> float | None:
        """A fixed, plainly-not-now epoch so a test asserting on the shape sees a
        stable value rather than a moving one."""
        return 1_700_000_000.0 if camera_id in self._cameras else None

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
            # The welfare policy rides on `CameraTelemetry` too, so an unmentioned
            # field falls back to what the camera already has, exactly as the real
            # store's re-parse of the edited document does.
            **{
                name: (
                    getattr(current, name) if getattr(edit, name) is UNSET else getattr(edit, name)
                )
                for name in _WELFARE_POLICY_FIELDS
            },
            capabilities=(
                current.capabilities if edit.capabilities is UNSET else edit.capabilities
            ),
        )


def _telemetry(
    camera_id: str = "cam-1",
    *,
    zone: Zone | None = None,
    label: str | None = None,
    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES,
    notify_on: frozenset[ConcernKind] = frozenset(ConcernKind),
    notify_min_confidence: Confidence = Confidence.LIKELY,
    clip_preroll_seconds: float | None = None,
    clip_postroll_seconds: float | None = None,
    summary_interval_seconds: float | None = None,
) -> CameraTelemetry:
    return CameraTelemetry(
        camera_id=camera_id,
        label=label if label is not None else camera_id,
        capabilities=capabilities,
        frames_seen=100,
        frames_dropped=2,
        detections_run=98,
        escalations=3,
        escalations_dropped=0,
        discontinuities=1,
        last_frame_at=12.5,
        last_escalation_at=10.0,
        zone=zone,
        notify_on=notify_on,
        notify_min_confidence=notify_min_confidence,
        clip_preroll_seconds=clip_preroll_seconds,
        clip_postroll_seconds=clip_postroll_seconds,
        summary_interval_seconds=summary_interval_seconds,
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
        # True unless an operator switched this camera off. Asserted here rather than
        # left to the lifecycle tests because this is the shape contract: a console
        # reads liveness from the same object it reads the counters from, and a missing
        # `enabled` would make every camera look like it is running.
        "enabled": True,
        "frames_seen": 100,
        "frames_dropped": 2,
        "detections_run": 98,
        "escalations": 3,
        "escalations_dropped": 0,
        "discontinuities": 1,
        # Zero, and zero is a claim: the camera has this capability's counter and
        # nothing has completed. A console must not render it as "not monitored".
        "falls_suspected": 0,
        # The camera's own source timeline, unchanged — never an age.
        "last_frame_at": 12.5,
        # And the wall-clock observation a liveness indicator actually reads. The
        # two are different numbers on purpose: reading the first as an epoch is
        # the bug this field exists to prevent.
        "last_frame_epoch": 1_700_000_000.0,
        "last_escalation_at": 10.0,
        "zone": None,
        "zone_kind": None,
        # The default set, sorted — a camera whose file says nothing about
        # `capabilities` describes scenes and runs the automatic triggers, which is
        # what every camera did before the field existed.
        "capabilities": ["anomaly_detection", "scene_description"],
        "notify_on": ["altercation", "collapse", "distress", "medication", "other", "self_harm"],
        "notify_min_confidence": "likely",
        "clip_preroll_seconds": None,
        "clip_postroll_seconds": None,
        "summary_interval_seconds": None,
    }


class TestTheWelfarePolicyOnTheCameraList:
    """The read side of `PATCH /cameras/{id}`'s welfare fields.

    Without it, Task 11's read-only camera record has nothing to render and every
    edit is invisible until the engine restarts — the endpoint would accept a change
    the console could never confirm.
    """

    def test_the_camera_list_carries_the_stored_policy(self) -> None:
        service = _FakeEngineService(
            cameras=(
                _telemetry(
                    "cam-1",
                    notify_on=frozenset({ConcernKind.COLLAPSE, ConcernKind.DISTRESS}),
                    notify_min_confidence=Confidence.POSSIBLE,
                    clip_preroll_seconds=0.0,
                    clip_postroll_seconds=2.5,
                    summary_interval_seconds=90.0,
                ),
            )
        )
        with TestClient(create_app(service)) as client:
            (camera,) = client.get("/cameras").json()["cameras"]

        # Sorted, exactly as `CameraEditResponse` sorts it: a console diffing what it
        # wrote against what it reads back must not see a change that is not one.
        assert camera["notify_on"] == ["collapse", "distress"]
        assert camera["notify_min_confidence"] == "possible"
        assert camera["clip_preroll_seconds"] == 0.0
        assert camera["clip_postroll_seconds"] == 2.5
        assert camera["summary_interval_seconds"] == 90.0

    def test_an_empty_notify_on_is_a_muted_camera_not_a_missing_field(self) -> None:
        """`[]` is the "never notify" instruction `PATCH` accepts, so it has to come
        back as `[]` rather than as the every-kind default — a console that could not
        tell the two apart would show a muted camera as fully armed."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1", notify_on=frozenset()),))
        with TestClient(create_app(service)) as client:
            (camera,) = client.get("/cameras").json()["cameras"]

        assert camera["notify_on"] == []

    def test_the_durations_report_what_is_stored_not_what_is_in_force(self) -> None:
        """Null means "this camera follows the engine-wide default", the same answer
        `CameraEditResponse` gives. Resolving it here would make a console that
        re-submits what it read pin the camera to a value nobody chose."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
        with TestClient(create_app(service)) as client:
            (camera,) = client.get("/cameras").json()["cameras"]

        assert camera["clip_preroll_seconds"] is None
        assert camera["clip_postroll_seconds"] is None
        assert camera["summary_interval_seconds"] is None


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
    welfare_concerns: tuple[WelfareConcern, ...] = (),
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
        welfare_concerns=welfare_concerns,
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

    def test_a_capability_edit_reaches_the_store_as_a_whole_set(self) -> None:
        """A replacement, never an addition. A console sending one capability means
        "this camera now runs exactly this", and merging instead would make turning a
        capability *off* impossible through this endpoint."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"capabilities": ["anomaly_detection"]})

        assert response.status_code == 200
        _, edit = service.edits[-1]
        assert edit.capabilities == CameraCapabilities.of(Capability.ANOMALY_DETECTION)
        assert response.json()["capabilities"] == ["anomaly_detection"]

    def test_an_empty_capability_list_is_an_instruction_not_an_empty_edit(self) -> None:
        """`[]` means "run nothing on this camera" and must reach the store as such.

        The absent-versus-empty distinction `notify_on` already carries, with the
        polarity that matters more: collapsing `[]` into "unchanged" would leave
        models running on a camera an operator had just switched off.
        """
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"capabilities": []})

        assert response.status_code == 200
        _, edit = service.edits[-1]
        assert edit.capabilities == CameraCapabilities.none()

    def test_a_null_capability_list_is_rejected_rather_than_guessed(self) -> None:
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"capabilities": None})

        assert response.status_code == 422
        assert "capabilities" in response.text
        assert service.edits == []

    def test_an_unknown_capability_name_is_a_422_not_a_silent_drop(self) -> None:
        """Fail-loud, as an unknown `zone` is. A capability quietly dropped leaves an
        operator believing monitoring is running that was never switched on."""
        service, client = self.app()
        with client:
            response = client.patch("/cameras/cam-1", json={"capabilities": ["xray_vision"]})

        assert response.status_code == 422
        assert service.edits == []

    def test_a_capability_whose_model_was_never_loaded_is_a_409_naming_the_restart(
        self,
    ) -> None:
        """§13's honest refusal. Placing a 3B vision model is a multi-second
        download-and-place against a GPU every camera shares, so it cannot happen
        under an HTTP request — and a 200 that wrote the file and left the camera
        unable to honour it would be worse than saying so."""
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),),
            update_error=CapabilityUnavailableError("cam-1", frozenset({ModelRole.VLM})),
        )
        with TestClient(create_app(service, camera_writes_enabled=True)) as client:
            response = client.patch("/cameras/cam-1", json={"capabilities": ["scene_description"]})

        assert response.status_code == 409
        assert "restart" in response.json()["detail"]
        assert "Nothing was changed" in response.json()["detail"]

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
            # what stops a console rendering "no kinds" for it. `capabilities` is
            # echoed for the same reason, and its default is the pre-capabilities
            # behaviour: describe scenes, run the automatic triggers.
            "capabilities": ["anomaly_detection", "scene_description"],
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


class TestWelfareConcernsOnTheEventRing:
    """The console's own view of what the model said about a person's wellbeing.

    Distinct from a notification, which goes to somebody who is *not* looking at
    the console and is narrowed to the kinds that camera routes. What lands here
    is the whole assessment.
    """

    def test_an_events_response_carries_each_concern_the_model_reported(self) -> None:
        concerns = (
            WelfareConcern(
                kind=ConcernKind.COLLAPSE,
                confidence=Confidence.LIKELY,
                evidence="A person is lying motionless by the door.",
            ),
        )
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),),
            events=(_recent_event(welfare_concerns=concerns),),
        )
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        assert response.status_code == 200
        assert response.json()["events"][0]["welfare_concerns"] == [
            {
                "kind": "collapse",
                "confidence": "likely",
                "evidence": "A person is lying motionless by the door.",
                "evidence_stated": True,
            }
        ]

    def test_an_event_with_nothing_reported_carries_an_empty_list(self) -> None:
        """An empty list, not a missing key. The console renders nothing either way,
        but a consumer that has to branch on presence to read a list is a consumer
        that will eventually get the branch wrong."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), events=(_recent_event(),))
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        assert response.json()["events"][0]["welfare_concerns"] == []

    def test_a_concern_the_model_never_evidenced_says_so_on_the_wire(self) -> None:
        """Otherwise the console renders a fixed placeholder as though the model had
        said it, which is the one thing `evidence_stated` exists to prevent."""
        concerns = (
            WelfareConcern(
                kind=ConcernKind.DISTRESS,
                confidence=Confidence.POSSIBLE,
                evidence="not stated",
                evidence_stated=False,
            ),
        )
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),),
            events=(_recent_event(welfare_concerns=concerns),),
        )
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        entry = response.json()["events"][0]["welfare_concerns"][0]
        assert entry["evidence_stated"] is False
        assert entry["confidence"] == "possible"

    def test_the_ring_is_not_filtered_by_the_camera_notify_on_policy(self) -> None:
        """A camera muted for a kind still shows that kind here. `notify_on` decides
        who gets *paged*; it must not decide what an operator looking at the camera
        page is allowed to see, or muting a camera would quietly blind the console
        as well as the pager."""
        muted = _telemetry("cam-1", notify_on=frozenset({ConcernKind.COLLAPSE}))
        concerns = (
            WelfareConcern(
                kind=ConcernKind.MEDICATION,
                confidence=Confidence.LIKELY,
                evidence="Tipping an unlabelled bottle towards the mouth.",
            ),
        )
        service = _FakeEngineService(
            cameras=(muted,), events=(_recent_event(welfare_concerns=concerns),)
        )
        with TestClient(create_app(service)) as client:
            response = client.get("/cameras/cam-1/events")

        kinds = [c["kind"] for c in response.json()["events"][0]["welfare_concerns"]]
        assert kinds == ["medication"]


class TestAlertRoutes:
    """The HTTP translation. Register semantics are `tests/orchestrator/test_alerts.py`'s
    subject and are not re-tested here."""

    @staticmethod
    def _an_event(
        *,
        reason: EscalationReason = EscalationReason.ZONE_INTRUSION,
        track_ids: tuple[int, ...] = (7,),
    ) -> Event:
        return Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=0.0,
            reason=reason,
            threat=ThreatScore.from_value(0.7),
            description="a person is in the stairwell",
            suggested_action="Go and look.",
            track_ids=track_ids,
            subject_track_ids=track_ids,
        )

    def app(self) -> tuple[AlertRegister, TestClient]:
        register = AlertRegister()
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), alert_register=register)
        return register, TestClient(create_app(service))

    def test_an_empty_register_answers_an_empty_list(self) -> None:
        _, client = self.app()
        with client:
            response = client.get("/alerts")
        assert response.status_code == 200
        assert response.json() == {"alerts": [], "open_count": 0}

    def test_repeated_events_are_one_row_with_a_count(self) -> None:
        """§17's requirement, at the API boundary: an operator sees one row saying it
        happened three times, not three rows."""
        register, client = self.app()
        for index in range(3):
            register.absorb(
                self._an_event(), camera_label="Corridor 1", zone=Zone.CORRIDOR, now=float(index)
            )
        with client:
            body = client.get("/alerts").json()
        assert len(body["alerts"]) == 1
        assert body["alerts"][0]["occurrences"] == 3
        assert body["open_count"] == 1

    def test_the_worst_alert_is_first(self) -> None:
        register, client = self.app()
        register.absorb(
            self._an_event(reason=EscalationReason.LINE_CROSSING, track_ids=(1,)),
            camera_label="Corridor 1",
            zone=None,
            now=0.0,
        )
        register.absorb(
            self._an_event(reason=EscalationReason.FALL_SUSPECTED, track_ids=(2,)),
            camera_label="Corridor 1",
            zone=None,
            now=1.0,
        )
        with client:
            body = client.get("/alerts").json()
        assert body["alerts"][0]["reason"] == "fall_suspected"
        assert body["alerts"][0]["priority"] == "critical"

    def test_acknowledging_records_the_label_and_the_state(self) -> None:
        register, client = self.app()
        alert = register.absorb(self._an_event(), camera_label="Corridor 1", zone=None, now=0.0)
        assert alert is not None
        with client:
            response = client.post(
                f"/alerts/{alert.alert_id}/acknowledge", json={"by": "operator-1"}
            )
        assert response.status_code == 200
        assert response.json()["state"] == "acknowledged"
        assert response.json()["acknowledged_by"] == "operator-1"

    def test_acknowledging_an_unknown_alert_is_a_404(self) -> None:
        _, client = self.app()
        with client:
            response = client.post(f"/alerts/{uuid4()}/acknowledge", json={"by": "operator-1"})
        assert response.status_code == 404
        assert "'" not in response.json()["detail"], "KeyError.__str__ must not reach the wire"

    def test_acknowledging_a_resolved_alert_is_a_409(self) -> None:
        """The operator is acting on a stale list. Silently accepting would tell them
        they had done something they had not."""
        register, client = self.app()
        alert = register.absorb(self._an_event(), camera_label="Corridor 1", zone=None, now=0.0)
        assert alert is not None
        register.resolve(alert.alert_id)
        with client:
            response = client.post(
                f"/alerts/{alert.alert_id}/acknowledge", json={"by": "operator-1"}
            )
        assert response.status_code == 409

    def test_an_empty_acknowledger_is_rejected(self) -> None:
        register, client = self.app()
        alert = register.absorb(self._an_event(), camera_label="Corridor 1", zone=None, now=0.0)
        assert alert is not None
        with client:
            response = client.post(f"/alerts/{alert.alert_id}/acknowledge", json={"by": ""})
        assert response.status_code == 422

    def test_resolving_is_idempotent(self) -> None:
        """Two operators closing the same row is an ordinary race."""
        register, client = self.app()
        alert = register.absorb(self._an_event(), camera_label="Corridor 1", zone=None, now=0.0)
        assert alert is not None
        with client:
            first = client.post(f"/alerts/{alert.alert_id}/resolve")
            second = client.post(f"/alerts/{alert.alert_id}/resolve")
        assert first.status_code == second.status_code == 200
        assert second.json()["state"] == "resolved"

    def test_a_resolved_alert_stays_listed_but_is_not_open(self) -> None:
        register, client = self.app()
        alert = register.absorb(self._an_event(), camera_label="Corridor 1", zone=None, now=0.0)
        assert alert is not None
        register.resolve(alert.alert_id)
        with client:
            body = client.get("/alerts").json()
        assert len(body["alerts"]) == 1
        assert body["open_count"] == 0


class TestAuthorizedPersonRoutes:
    """§23's enrolment surface, and §12's constraint on what it may return."""

    def app(self, tmp_path: Path, **kwargs: object) -> tuple[EncryptedFaceStore, TestClient]:
        store = EncryptedFaceStore(tmp_path / "faces.json", encryption_key=generate_key())
        service = _FakeEngineService(
            cameras=(_telemetry("cam-1"),),
            face_store=store,
            **kwargs,  # type: ignore[arg-type]
        )
        return store, TestClient(create_app(service))

    @staticmethod
    def _body(**overrides: object) -> dict[str, object]:
        body: dict[str, object] = {"display_name": "Employee A", "camera_ids": ["cam-1"]}
        body.update(overrides)
        return body

    @staticmethod
    def _png() -> bytes:
        """A real, tiny PNG.

        The route decodes the upload before it looks the person up, so a placeholder
        byte string would 422 on every enrolment test and hide whatever they were
        actually checking.
        """
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), color=(120, 120, 120)).save(buffer, format="PNG")
        return buffer.getvalue()

    def test_an_empty_roster_lists_nothing(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        with client:
            assert client.get("/authorized-persons").json() == {"people": []}

    def test_a_person_can_be_enrolled_and_read_back(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        person_id = uuid4()
        with client:
            created = client.put(f"/authorized-persons/{person_id}", json=self._body())
            listed = client.get("/authorized-persons").json()

        assert created.status_code == 200
        assert created.json()["display_name"] == "Employee A"
        assert created.json()["camera_ids"] == ["cam-1"]
        assert len(listed["people"]) == 1

    def test_no_biometric_data_ever_appears_on_the_wire(self, tmp_path: Path) -> None:
        """§12. The roster carries names and permissions; embeddings stay in the
        encrypted store. `reference_faces` is a count, never the faces."""
        _, client = self.app(tmp_path)
        person_id = uuid4()
        with client:
            client.put(f"/authorized-persons/{person_id}", json=self._body())
            client.post(
                f"/authorized-persons/{person_id}/faces",
                files={"image": ("face.png", self._png(), "image/png")},
            )
            body = client.get("/authorized-persons").json()

        serialised = json.dumps(body).lower()
        for forbidden in ("embedding", "vector", "image", "ciphertext", "nonce"):
            assert forbidden not in serialised
        assert body["people"][0]["reference_faces"] == 1

    def test_enrolling_a_face_increments_the_reference_count(self, tmp_path: Path) -> None:
        """§10 asks for multiple references per person, and one is usually why somebody
        is not recognised from an angle."""
        _, client = self.app(tmp_path)
        person_id = uuid4()
        with client:
            client.put(f"/authorized-persons/{person_id}", json=self._body())
            first = client.post(
                f"/authorized-persons/{person_id}/faces",
                files={"image": ("a.png", self._png(), "image/png")},
            )
            second = client.post(
                f"/authorized-persons/{person_id}/faces",
                files={"image": ("b.png", self._png(), "image/png")},
            )
        assert first.json()["reference_faces"] == 1
        assert second.json()["reference_faces"] == 2

    def test_enrolling_against_an_unknown_person_is_a_404(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        with client:
            response = client.post(
                f"/authorized-persons/{uuid4()}/faces",
                files={"image": ("a.png", self._png(), "image/png")},
            )
        assert response.status_code == 404

    def test_an_image_with_no_face_is_a_422_that_says_so(self, tmp_path: Path) -> None:
        """An enrolment that silently stored nothing is how somebody becomes
        unrecognisable with nobody able to say why."""
        _, client = self.app(tmp_path, enroll_error=ValueError("no face was found in that image"))
        person_id = uuid4()
        with client:
            client.put(f"/authorized-persons/{person_id}", json=self._body())
            response = client.post(
                f"/authorized-persons/{person_id}/faces",
                files={"image": ("a.png", self._png(), "image/png")},
            )
        assert response.status_code == 422
        assert "no face" in response.json()["detail"]

    def test_deleting_removes_the_person_and_reports_204(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        person_id = uuid4()
        with client:
            client.put(f"/authorized-persons/{person_id}", json=self._body())
            client.post(
                f"/authorized-persons/{person_id}/faces",
                files={"image": ("a.png", self._png(), "image/png")},
            )
            response = client.delete(f"/authorized-persons/{person_id}")
            remaining = client.get("/authorized-persons").json()

        assert response.status_code == 204
        assert remaining["people"] == []
        assert str(person_id) not in (tmp_path / "faces.json").read_text(encoding="utf-8")

    def test_deleting_someone_who_was_never_there_is_a_404(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        with client:
            assert client.delete(f"/authorized-persons/{uuid4()}").status_code == 404

    def test_an_empty_camera_list_authorises_nowhere(self, tmp_path: Path) -> None:
        """The opposite default would make forgetting to set it a silent grant
        everywhere, which for an access rule is the failure worth designing against."""
        _, client = self.app(tmp_path)
        person_id = uuid4()
        with client:
            response = client.put(
                f"/authorized-persons/{person_id}", json=self._body(camera_ids=[])
            )
        assert response.json()["camera_ids"] == []

    def test_an_empty_display_name_is_rejected(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        with client:
            response = client.put(
                f"/authorized-persons/{uuid4()}", json=self._body(display_name="  ")
            )
        assert response.status_code == 422

    def test_an_unknown_field_is_rejected_rather_than_ignored(self, tmp_path: Path) -> None:
        _, client = self.app(tmp_path)
        with client:
            response = client.put(f"/authorized-persons/{uuid4()}", json=self._body(is_admin=True))
        assert response.status_code == 422

    def test_an_engine_without_the_capability_answers_503_not_404(self) -> None:
        """A 404 would tell an operator the person does not exist, when the truth is
        that nothing on this deployment does face recognition at all."""
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), face_store=None)
        with TestClient(create_app(service)) as client:
            assert client.get("/authorized-persons").json() == {"people": []}
            assert (
                client.put(f"/authorized-persons/{uuid4()}", json=self._body()).status_code == 503
            )
            assert client.delete(f"/authorized-persons/{uuid4()}").status_code == 503


class TestAlertDurabilityContract:
    """A 200 from acknowledge or resolve has to mean the decision reached the disk.

    Telling an operator "acknowledged" and then showing the row as unseen after a
    restart is worse than not having persistence at all: they stop trusting the list.
    """

    def app(self) -> tuple[_FakeEngineService, TestClient]:
        register = AlertRegister()
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), alert_register=register)
        register.absorb(
            TestAlertRoutes._an_event(),
            camera_label="Cam 1",
            zone=None,
            now=0.0,
        )
        return service, TestClient(create_app(service))

    def _alert_id(self, client: TestClient) -> str:
        return str(client.get("/alerts").json()["alerts"][0]["alert_id"])

    def test_acknowledging_flushes_before_answering(self) -> None:
        service, client = self.app()
        with client:
            response = client.post(
                f"/alerts/{self._alert_id(client)}/acknowledge", json={"by": "night shift"}
            )
        assert response.status_code == 200
        assert service.alert_flushes == 1

    def test_resolving_flushes_before_answering(self) -> None:
        service, client = self.app()
        with client:
            response = client.post(f"/alerts/{self._alert_id(client)}/resolve")
        assert response.status_code == 200
        assert service.alert_flushes == 1

    def test_a_refused_acknowledgement_does_not_flush(self) -> None:
        """Nothing changed, so there is nothing to write — and a write per rejected
        request is a disk write per operator working from a stale list."""
        service, client = self.app()
        with client:
            alert_id = self._alert_id(client)
            client.post(f"/alerts/{alert_id}/resolve")
            before = service.alert_flushes
            refused = client.post(f"/alerts/{alert_id}/acknowledge", json={"by": "x"})
        assert refused.status_code == 409
        assert service.alert_flushes == before

    def test_a_second_acknowledgement_keeps_the_first_operator_and_time(self) -> None:
        """Two consoles watching one wall both acknowledge the same row. Last-write-wins
        would push `acknowledged_at` later every time somebody looked, turning "when did
        this stop being unseen" into "when did somebody last click"."""
        _, client = self.app()
        with client:
            alert_id = self._alert_id(client)
            first = client.post(f"/alerts/{alert_id}/acknowledge", json={"by": "first"}).json()
            second = client.post(f"/alerts/{alert_id}/acknowledge", json={"by": "second"}).json()
        assert second["acknowledged_by"] == "first"
        assert second["acknowledged_at"] == first["acknowledged_at"]
        assert second["state"] == "acknowledged"


class TestAlertClip:
    """`GET /alerts/{alert_id}/clip` — the only route that serves a recording."""

    def app(self, clip: bytes | None) -> tuple[_FakeEngineService, str, TestClient]:
        register = AlertRegister()
        service = _FakeEngineService(cameras=(_telemetry("cam-1"),), alert_register=register)
        service.clip_bytes = clip
        alert = register.absorb(
            TestAlertRoutes._an_event(), camera_label="Corridor 1", zone=Zone.CORRIDOR, now=0.0
        )
        assert alert is not None
        return service, str(alert.alert_id), TestClient(create_app(service))

    def test_serves_the_short_clip_by_default(self) -> None:
        service, alert_id, client = self.app(b"short-clip-bytes")
        with client:
            response = client.get(f"/alerts/{alert_id}/clip")

        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert response.content == b"short-clip-bytes"
        # The default is the length somebody actually watches while triaging a wall of
        # rows. An operator who wants the full recording has to ask for it.
        assert service.clip_requests == [(UUID(alert_id), True)]

    def test_serves_the_full_clip_when_asked(self) -> None:
        service, alert_id, client = self.app(b"clip")
        with client:
            response = client.get(f"/alerts/{alert_id}/clip", params={"short": "false"})

        assert response.status_code == 200
        assert response.content == b"clip-full"
        assert service.clip_requests == [(UUID(alert_id), False)]

    def test_a_clip_that_is_gone_is_404_rather_than_500(self) -> None:
        """Retention deletes clips on a schedule, so an alert outliving its recording is
        ordinary. An operator must be told the footage has expired, not that the engine
        is broken — those lead to completely different next actions."""
        _, alert_id, client = self.app(None)
        with client:
            response = client.get(f"/alerts/{alert_id}/clip")

        assert response.status_code == 404
        assert "no clip" in response.json()["detail"]

    def test_an_unknown_alert_is_404(self) -> None:
        _, _, client = self.app(b"clip")
        with client:
            response = client.get(f"/alerts/{uuid4()}/clip")

        assert response.status_code == 404

    def test_the_clip_never_enters_a_shared_cache(self) -> None:
        """Footage of people. It may sit in the viewer's own browser briefly, because a
        clip never changes once written and re-fetching it on every render is waste —
        but `private` keeps it out of every proxy between here and there."""
        _, alert_id, client = self.app(b"clip")
        with client:
            response = client.get(f"/alerts/{alert_id}/clip")

        cache_control = response.headers["cache-control"]
        assert "private" in cache_control
        assert "public" not in cache_control
