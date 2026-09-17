"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

import base64
import time
from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse

from sentinel_ai.adapters.config.camera_file import (
    CameraConfig,
    CameraConfigError,
    CameraCreate,
    CameraEdit,
)
from sentinel_ai.adapters.sources.probe import ProbeResult
from sentinel_ai.api.schemas import (
    AcknowledgeRequest,
    AlertEntry,
    AlertsResponse,
    CameraCreateRequest,
    CameraEditRequest,
    CameraEditResponse,
    CameraEnabledRequest,
    CameraStorageModel,
    ClipRecordModel,
    ClipsResponse,
    SettingEntryModel,
    SettingsResponse,
    StorageResponse,
    CameraEventsResponse,
    CamerasResponse,
    CameraStatus,
    DescribeResponse,
    EnrolledFaceEntry,
    FacesResponse,
    HealthResponse,
    ModelHealth,
    PeopleResponse,
    PersonEntry,
    PersonRequest,
    ProbeRequest,
    ProbeResponse,
)
from sentinel_ai.api.sse import SSE_HEADERS, SSE_MEDIA_TYPE, event_stream_body
from sentinel_ai.domain.alert import Alert
from sentinel_ai.domain.capabilities import CameraCapabilities
from sentinel_ai.domain.identity import AuthorizedPerson, EnrolledFace
from sentinel_ai.config import get_settings
from sentinel_ai.config_report import describe_settings
from sentinel_ai.orchestrator.alerts import UnknownAlertError
from sentinel_ai.ports.clip_index import ClipRecord, StorageUsage
from sentinel_ai.orchestrator.event_history import CameraEventHistory, EventSubscription
from sentinel_ai.orchestrator.service import (
    CapabilityUnavailableError,
    DuplicateCameraError,
    EngineNotComposedError,
    FaceCapabilityUnavailableError,
    UnknownCameraError,
    UnknownPersonError,
)
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport


class EngineServiceProtocol(Protocol):
    """The exact EngineService (S12) surface this API uses, expressed
    structurally so tests can inject a fake without constructing the real
    orchestrator stack. `orchestrator.service.EngineService` satisfies this
    by shape; nothing here changes S12."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def cameras(self) -> tuple[CameraTelemetry, ...]: ...
    def telemetry(self, camera_id: str) -> CameraTelemetry: ...
    def event_history(self, camera_id: str) -> CameraEventHistory: ...
    def subscribe_events(self) -> EventSubscription: ...
    def close_event_streams(self) -> int: ...
    def health(self) -> dict[str, HealthReport]: ...
    def last_frame_epoch(self, camera_id: str) -> float | None: ...
    async def list_people(self) -> tuple[tuple[AuthorizedPerson, int], ...]: ...
    async def upsert_person(self, person: AuthorizedPerson) -> AuthorizedPerson: ...
    async def delete_person(self, person_id: UUID) -> bool: ...
    async def enroll_face(
        self, person_id: UUID, image: object, *, original: bytes | None = None
    ) -> int: ...
    def alerts(self) -> tuple[Alert, ...]: ...
    def acknowledge_alert(self, alert_id: UUID, *, by: str, at: float) -> Alert: ...
    def resolve_alert(self, alert_id: UUID) -> Alert: ...
    async def read_alert_clip(self, alert_id: UUID, *, short: bool) -> bytes | None: ...
    def clear_alerts(self) -> int: ...
    async def flush_alerts(self) -> None: ...
    async def create_camera(self, create: CameraCreate) -> CameraConfig: ...
    async def delete_camera(self, camera_id: str) -> None: ...
    async def probe_source(self, url: str) -> ProbeResult: ...
    async def snapshot(self, camera_id: str) -> bytes | None: ...
    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]: ...
    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None: ...
    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool: ...
    async def describe_now(self, camera_id: str) -> UUID: ...
    async def update_camera(self, camera_id: str, edit: CameraEdit) -> CameraConfig: ...
    async def set_camera_enabled(self, camera_id: str, enabled: bool) -> CameraConfig: ...
    async def storage_usage(self) -> StorageUsage: ...
    async def list_clips(self, camera_id: str, *, limit: int) -> tuple[ClipRecord, ...]: ...
    async def read_camera_clip(
        self, camera_id: str, event_id: UUID, *, short: bool
    ) -> bytes | None: ...


def get_service(request: Request) -> EngineServiceProtocol:
    service: EngineServiceProtocol = request.app.state.service
    return service


def camera_writes_enabled(request: Request) -> bool:
    """Whether this deployment permits the write endpoint to do anything.

    Read off `app.state` rather than `Settings` so that the flag is a property of
    the app a caller built (see `create_app`), and so that a test can construct a
    writable and a non-writable app in the same process without touching the
    environment or clearing `get_settings`'s cache.
    """
    enabled: bool = getattr(request.app.state, "camera_writes_enabled", False)
    return enabled


# `Annotated[..., Depends(...)]` rather than a `Depends(...)` default value: the
# latter is a function call in an argument default, which this project's ruff
# config (B008) rightly flags everywhere else, so the FastAPI-idiomatic
# workaround is used instead of a per-line suppression.
ServiceDep = Annotated[EngineServiceProtocol, Depends(get_service)]

UploadedImage = Annotated[UploadFile, File(...)]
"""The enrolment photograph, as an `Annotated` alias rather than a `File(...)` default.

Both spell the same thing to FastAPI, but a call in a default argument is evaluated once
at import and shared by every request — which is harmless for `File`, and is exactly the
bug the rule against it exists to catch elsewhere. Matching `ServiceDep`'s shape keeps
one idiom in this module instead of one idiom and one exemption."""
WritesEnabledDep = Annotated[bool, Depends(camera_writes_enabled)]

router = APIRouter()


def _unknown_camera(exc: UnknownCameraError) -> HTTPException:
    # Not str(exc): UnknownCameraError subclasses KeyError, whose __str__ is
    # repr(args[0]), so str() would put literal quotes on the wire —
    # {"detail": "'unknown camera: cam-x'"}.
    return HTTPException(status_code=404, detail=f"unknown camera: {exc.camera_id}")


@router.get(
    "/settings",
    response_model=SettingsResponse,
    summary="What this engine is configured to do",
)
async def get_settings_report() -> SettingsResponse:
    """Every setting, its effective value, and whether this deployment moved it.

    **Read-only.** Almost nothing here can change without a restart — which models to
    load, how much VRAM to budget, where the broker is — so a writable version would
    have to either lie about taking effect or restart the engine under whoever asked.
    The settings that *are* live are per-camera and editable where they belong.

    **Credentials are never reported by value**, only as `(set)` or `(unset)`, which is
    the fact an operator actually needs: "this engine has no encryption key" is a
    diagnosis and the key itself is never one. Which fields those are is decided in one
    place, by name rather than by inspecting the value — see `config_report`.
    """
    entries = describe_settings(get_settings())
    return SettingsResponse(
        settings=[
            SettingEntryModel(
                name=entry.name,
                value=entry.value,
                is_default=entry.is_default,
                secret=entry.secret,
                group=entry.group,
            )
            for entry in entries
        ],
        changed=sum(1 for entry in entries if not entry.is_default),
    )


@router.get(
    "/storage",
    response_model=StorageResponse,
    summary="What the clip bucket is holding",
)
async def get_storage(service: ServiceDep) -> StorageResponse:
    """Clip storage: how much, how many, per camera, and how long any of it survives.

    **`reachable: false` is the answer that matters.** When the object store cannot be
    listed every count here is zero, and zero counts drawn as an empty bucket is how an
    operator concludes their evidence has been deleted. This never fails for an
    unreachable store; it says so instead.

    Retention is a bucket lifecycle rule the object store applies itself, not a sweeper
    in this process — so clips expire on schedule whether or not the engine is running.
    """
    usage = await service.storage_usage()
    return StorageResponse(
        bucket=usage.bucket,
        reachable=usage.reachable,
        clips=usage.clips,
        objects=usage.objects,
        bytes_used=usage.bytes_used,
        retention_days=usage.retention_days,
        per_camera=[
            CameraStorageModel(
                camera_id=camera.camera_id, clips=camera.clips, bytes_used=camera.bytes_used
            )
            for camera in usage.per_camera
        ],
    )


@router.get("/health", response_model=HealthResponse)
async def get_health(service: ServiceDep) -> HealthResponse:
    reports = service.health()
    return HealthResponse(
        status="ok",
        models=[
            ModelHealth(
                key=key, state=report.state.value, detail=report.detail, vram_mib=report.vram_mib
            )
            for key, report in reports.items()
        ],
    )


@router.get("/cameras", response_model=CamerasResponse)
async def list_cameras(service: ServiceDep, writes_enabled: WritesEnabledDep) -> CamerasResponse:
    return CamerasResponse(
        cameras=[
            CameraStatus.from_telemetry(t, last_frame_epoch=service.last_frame_epoch(t.camera_id))
            for t in service.cameras()
        ],
        config_writable=writes_enabled,
    )


_WRITES_DISABLED = (
    "camera writes are disabled on this engine. This build has no authentication, so "
    "the write endpoints are opt-in: set SENTINEL_ENABLE_CAMERA_WRITES=true only on a "
    "deployment where everything that can reach this port is permitted to reconfigure "
    "cameras. Until then, edit cameras.json and restart."
)


def _require_writes(writes_enabled: bool) -> None:
    """The one gate in front of every endpoint that changes or probes a camera."""
    if not writes_enabled:
        raise HTTPException(status_code=403, detail=_WRITES_DISABLED)


@router.post(
    "/cameras",
    response_model=CameraEditResponse,
    status_code=201,
    summary="Add a camera and start watching it — no restart",
    responses={
        403: {"description": "Camera writes are disabled on this engine."},
        409: {
            "description": (
                "A camera with this id already exists, or a capability was asked for "
                "whose model this process never loaded."
            )
        },
        422: {"description": "The camera could not be built — the detail says why."},
        503: {"description": "The engine is still starting."},
    },
)
async def create_camera(
    body: CameraCreateRequest, service: ServiceDep, writes_enabled: WritesEnabledDep
) -> CameraEditResponse:
    """Add a camera to `cameras.json` and start it immediately.

    **The point of the write surface.** Every other endpoint here edits a camera
    somebody already put in the file by hand; this is what lets a console stand one up
    without an operator opening an editor and restarting the engine.

    Adding takes a `url` where `PATCH` refuses one, and the difference is not arbitrary:
    changing a running camera's source means tearing down its runner, its pre-roll ring
    and any clip mid-recording. A camera that does not exist yet has none of those.
    """
    _require_writes(writes_enabled)
    try:
        record = await service.create_camera(
            CameraCreate(
                camera_id=body.camera_id,
                url=body.url,
                label=body.label,
                zone=body.zone,
                capabilities=(
                    None if body.capabilities is None else CameraCapabilities.of(*body.capabilities)
                ),
            )
        )
    except DuplicateCameraError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CapabilityUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EngineNotComposedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except CameraConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CameraEditResponse.from_config(record)


@router.delete(
    "/cameras/{camera_id}",
    status_code=204,
    summary="Stop a camera and remove it from the file",
    responses={
        403: {"description": "Camera writes are disabled on this engine."},
        404: {"description": "No camera with this id."},
    },
)
async def delete_camera(
    camera_id: str, service: ServiceDep, writes_enabled: WritesEnabledDep
) -> Response:
    """Stop watching, then remove the record — in that order.

    The opposite order to `create_camera`, for the same underlying rule: never leave
    something running that the file does not describe. Writing first and then failing to
    stop would leave a camera publishing events under an id nothing can look up.
    """
    _require_writes(writes_enabled)
    try:
        await service.delete_camera(camera_id)
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    except CameraConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(status_code=204)


@router.put(
    "/cameras/{camera_id}/enabled",
    response_model=CameraEditResponse,
    summary="Start or stop watching a camera, without deleting it",
    responses={
        404: {"description": "No camera with this id."},
        503: {"description": "Camera writes are disabled on this engine."},
    },
)
async def set_camera_enabled(
    camera_id: str,
    body: CameraEnabledRequest,
    service: ServiceDep,
    writes_enabled: WritesEnabledDep,
) -> CameraEditResponse:
    """Switch a camera off without losing how it was set up.

    **The alternative to deleting it.** A camera taken down for maintenance, a feed
    that has gone permanently dark, a lens being repositioned — deleting is the wrong
    tool for all three, because it throws away the zone, the capability set, the
    notification routing and the clip overrides somebody tuned, and getting them back
    means re-entering them from memory.

    Disabling closes the source connection and destroys the runner, so a stopped camera
    costs no socket, no decode and no CPU. What it keeps is the whole record, still
    listed in `GET /cameras` with `enabled: false`, still editable, one request from
    running again with everything as it was.

    Its models stay loaded. Which model roles this process holds is fixed at startup
    from every configured camera including the stopped ones, so switching one back on
    can never fail for want of a model that was never loaded — the cost is VRAM held
    for a camera that is not using it, which is the right trade for making the switch
    reliable.

    Idempotent in both directions.
    """
    _require_writes(writes_enabled)
    try:
        record = await service.set_camera_enabled(camera_id, body.enabled)
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    return CameraEditResponse.from_config(record)


@router.post(
    "/cameras/probe",
    response_model=ProbeResponse,
    summary="Look at a stream before adding it",
    responses={403: {"description": "Camera writes are disabled on this engine."}},
)
async def probe_camera(
    body: ProbeRequest, service: ServiceDep, writes_enabled: WritesEnabledDep
) -> ProbeResponse:
    """Open a URL once, decode one frame, and report what it found — with a picture.

    A camera added with a wrong URL fails quietly: it appears in the list and its runner
    retries forever behind exponential backoff, with nothing to see but a `frames_seen`
    that never moves. This is how an operator learns that *before* saving.

    `ok: false` comes back as a 200. The caller renders the outcome either way, and a
    4xx would conflate "this URL does not play" with "your request was malformed".

    Behind the same flag as the writes it precedes, because it makes the engine fetch a
    URL the caller chose — exactly the capability that flag exists to gate on an
    unauthenticated port.
    """
    _require_writes(writes_enabled)
    result = await service.probe_source(body.url)
    thumbnail = (
        None
        if result.thumbnail_jpeg is None
        else "data:image/jpeg;base64," + base64.b64encode(result.thumbnail_jpeg).decode("ascii")
    )
    return ProbeResponse(
        ok=result.ok,
        detail=result.detail,
        source_kind=result.source_kind,
        title=result.title,
        width=result.width,
        height=result.height,
        codec=result.codec,
        fps=result.fps,
        thumbnail=thumbnail,
    )


@router.get(
    "/cameras/{camera_id}/snapshot",
    summary="The most recent frame from one camera, as a JPEG",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "The latest decoded frame."},
        404: {"description": "No such camera, or no frame has arrived yet."},
    },
)
async def camera_snapshot(camera_id: str, service: ServiceDep) -> Response:
    """A picture of what this camera is seeing right now.

    **Not video, and not a replacement for it.** Live video is mediamtx's job and the
    console plays it straight from there (`src/live/HlsPlayer.tsx`). But only cameras
    *published to* mediamtx have a playlist, and an EarthCam page or a local file does
    not — before this, those cameras showed an empty panel while the engine was
    demonstrably decoding them.

    So this is the fallback the console uses when there is no playlist: one frame, on
    request, which the engine already had. `Cache-Control: no-store`, because the whole
    value of the thing is that it is current.
    """
    try:
        image = await service.snapshot(camera_id)
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    if image is None:
        raise HTTPException(
            status_code=404, detail=f"camera {camera_id} has not delivered a frame yet"
        )
    return Response(content=image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get(
    "/cameras/{camera_id}/clips",
    response_model=ClipsResponse,
    summary="One camera's recorded clips, newest first",
    responses={404: {"description": "No camera with this id."}},
)
async def list_camera_clips(camera_id: str, service: ServiceDep, limit: int = 50) -> ClipsResponse:
    """What footage exists for this camera, read from the object store as it is now.

    **Not an index and not a history.** Retention has already removed whatever it has
    removed, so this is what survives — which is the honest answer for a browser, and
    is why nothing here has a "deleted" state. A camera that has been recording for
    longer than `SENTINEL_CLIP_RETENTION_DAYS` shows the window, not the run.

    An unknown camera is a 404 rather than an empty list: "this camera was deleted" and
    "this camera has recorded nothing" are different, and a browser that rendered them
    identically would let somebody conclude a camera had been silent all week.
    """
    try:
        clips = await service.list_clips(camera_id, limit=max(1, min(limit, 500)))
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    return ClipsResponse(
        camera_id=camera_id,
        clips=[
            ClipRecordModel(
                event_id=clip.event_id,
                size_bytes=clip.size_bytes,
                modified_at=clip.modified_at,
                has_short_copy=clip.has_short_copy,
            )
            for clip in clips
        ],
    )


@router.get(
    "/cameras/{camera_id}/clips/{event_id}",
    summary="Watch one of a camera's clips",
    response_class=Response,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "The clip, as MP4."},
        404: {"description": "No such camera, or no clip for that event."},
    },
)
async def read_camera_clip(
    camera_id: str, event_id: UUID, service: ServiceDep, short: bool = False
) -> Response:
    """The recording for one event on one camera.

    The object path is built inside the adapter from two validated components — a
    camera this engine has, and a `UUID` — never from a caller's string, and the reader
    re-checks the bucket on top of that.

    `short=false` here, unlike the alert route: somebody who has opened a footage
    browser and picked a clip is investigating, and the seconds before and after are
    the reason they came. `short=true` returns the notification trim where one exists.
    """
    try:
        clip = await service.read_camera_clip(camera_id, event_id, short=short)
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    if clip is None:
        raise HTTPException(status_code=404, detail="no clip is stored for that event")
    return Response(
        content=clip, media_type="video/mp4", headers={"Cache-Control": "private, max-age=60"}
    )


@router.get("/cameras/{camera_id}/telemetry", response_model=CameraStatus)
async def get_camera_telemetry(camera_id: str, service: ServiceDep) -> CameraStatus:
    try:
        telemetry = service.telemetry(camera_id)
    except UnknownCameraError as exc:
        # Not str(exc): UnknownCameraError subclasses KeyError, whose __str__ is
        # repr(args[0]), so str() would put literal quotes on the wire —
        # {"detail": "'unknown camera: cam-x'"}.
        raise HTTPException(status_code=404, detail=f"unknown camera: {exc.camera_id}") from exc
    return CameraStatus.from_telemetry(
        telemetry, last_frame_epoch=service.last_frame_epoch(camera_id)
    )


@router.get(
    "/cameras/{camera_id}/events",
    response_model=CameraEventsResponse,
    summary="Recent events for one camera (volatile console cache — NOT the event store)",
)
async def get_camera_events(camera_id: str, service: ServiceDep) -> CameraEventsResponse:
    """Recent anomaly events for one camera, plus the latest scene description.

    **This is a volatile, bounded, in-memory view for the operator console. It is not
    the event store and it is not an audit trail.** The durable record is the anomaly
    event published to RabbitMQ and written down by the Phase 1C consumer. What this
    endpoint reads is a per-camera ring inside the running engine that is emptied by
    any restart and that silently discards the oldest entry once the camera has
    produced more than `capacity` events. The absence of an event here means "not in
    the last `capacity` events of this process run" — never "did not happen". Anyone
    reconciling a custodial incident must go to the store, not here.

    The ring is per camera rather than global on purpose, so that a busy camera can
    only evict its own history and never a quiet camera's.

    `latest` is the last element of `events`, lifted out for the live scene panel and
    read from the same snapshot, so the panel and the chart cannot disagree about
    which event is newest. `latest_description_state` distinguishes the three cases a
    console must not confuse: `none` (nothing assembled yet in this process),
    `available` (a real vision-model description), and `unavailable` (the vision model
    could not answer, and `latest.description` is a metadata-derived stand-in rather
    than a description of the scene).

    An unknown camera is a 404, the same mapping every other camera route uses. A
    configured camera that has not escalated yet is a 200 with an empty `events` list.
    """
    try:
        history = service.event_history(camera_id)
    except UnknownCameraError as exc:
        # Not str(exc): UnknownCameraError subclasses KeyError, whose __str__ is
        # repr(args[0]), so str() would put literal quotes on the wire —
        # {"detail": "'unknown camera: cam-x'"}.
        raise HTTPException(status_code=404, detail=f"unknown camera: {exc.camera_id}") from exc
    return CameraEventsResponse.from_history(history)


@router.get(
    "/events/stream",
    summary="Live event stream (SSE) over the same volatile ring — NOT the event store",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "An open `text/event-stream`. Frames: `event: backlog` once, carrying a "
                "JSON array of RecentEventEntry (oldest first, possibly empty); then "
                "`event: anomaly`, one RecentEventEntry each, as they are assembled; "
                "`event: overflow` if this client fell too far behind, after which the "
                "stream ends and reconnecting is the recovery; and `: keepalive` comment "
                "lines on an idle stream. No `id:` field is sent and `Last-Event-ID` is "
                "not honoured — see the description."
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        503: {"description": "The engine has not finished starting; retry shortly."},
    },
)
async def stream_events(service: ServiceDep) -> StreamingResponse:
    """Every camera's anomaly events, pushed as they happen (spec §6.2, bridged).

    **The same volatile, bounded, in-memory ring `GET /cameras/{camera_id}/events`
    serves — not the event store and not an audit trail.** The durable record is the
    anomaly event published to RabbitMQ and written down by the Phase 1C consumer.
    Everything this stream can send is held in engine memory, is lost on restart, and
    is silently evicted once a camera has produced more than `capacity` events.

    **What a reconnect can and cannot give you.** Every connection opens with a
    `backlog` frame holding what the ring currently has, then continues live. There is
    no gap between the two and no duplicate across them. But the ring is bounded, so a
    client that was disconnected long enough for a camera to produce more than its
    ring holds **has permanently missed those events on this endpoint** — they are in
    RabbitMQ, and this endpoint will never show them. That is why no `id:` field is
    sent and `Last-Event-ID` is not honoured: resuming from an offset the engine may
    no longer hold would promise a continuity it cannot keep. Reconnect, take the new
    backlog as the current window, and go to the store for anything older.

    **Duplicates and updates.** An `event_id` may arrive more than once: an event is
    re-sent when something about it changes, currently when its clip finishes and
    `clip_uri` appears. Merge by `event_id`, keeping the copy with the higher
    `sequence`. Two copies with the same `sequence` are the same copy.

    **Falling behind.** A client that stops reading is buffered up to a fixed bound
    and then cut off with an `overflow` frame rather than being allowed to grow the
    engine's memory. Reconnect; the fresh backlog is more current than the queue that
    was dropped.

    **Shutdown.** The engine closes every stream as it shuts down, so a client sees a
    clean end of response rather than a connection that hangs until a proxy times it
    out.
    """
    try:
        subscription = service.subscribe_events()
    except EngineNotComposedError as exc:
        # Only reachable if a request is served before lifespan startup finished,
        # which uvicorn does not do. A 503 rather than an empty stream: a stream that
        # can never carry anything looks exactly like a quiet site.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(
        # Subscribed above, in the handler; the body below runs later, once Starlette
        # starts the response. That gap is real, and `EventSubscription` is built for
        # it — registration happens now so nothing written in between is lost, and the
        # backlog's watermark discards the copies that would otherwise be duplicated.
        event_stream_body(subscription),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )


def _person_entry(person: AuthorizedPerson, reference_faces: int) -> PersonEntry:
    return PersonEntry(
        person_id=person.person_id,
        display_name=person.display_name,
        status=person.status,
        external_reference=person.external_reference,
        camera_ids=sorted(person.camera_ids),
        zones=sorted(person.zones),
        expires_at=person.expires_at,
        reference_faces=reference_faces,
        notes=person.notes,
    )


def _face_unavailable(exc: FaceCapabilityUnavailableError) -> HTTPException:
    # 503 rather than 404: a 404 would tell an operator the person does not exist, when
    # the truth is that nothing on this deployment does face recognition at all.
    return HTTPException(status_code=503, detail=str(exc))


@router.get(
    "/authorized-persons",
    response_model=PeopleResponse,
    summary="Everyone enrolled — names and permissions, never biometric data",
)
async def list_people(service: ServiceDep) -> PeopleResponse:
    """The enrolled roster (spec §23).

    **Carries no biometric data.** No embeddings, no vectors, no reference images —
    those live in the encrypted store and never travel on this API. §12 asks that
    biometric information is not exposed unnecessarily, and for a roster listing the
    necessary amount is none. `reference_faces` is a count, because §10 asks for
    multiple references per person and "1" is usually the reason somebody is not being
    recognised from an angle.

    An engine with no camera enabling `person_authorization` answers an empty list
    rather than an error: nobody is enrolled, which is true.
    """
    return PeopleResponse(
        people=[_person_entry(person, count) for person, count in await service.list_people()]
    )


@router.put(
    "/authorized-persons/{person_id}",
    response_model=PersonEntry,
    summary="Create or replace an authorised person",
    responses={503: {"description": "No camera on this engine enables person authorization."}},
)
async def upsert_person(person_id: UUID, body: PersonRequest, service: ServiceDep) -> PersonEntry:
    """Enrol somebody, or change what they are authorised for.

    Carries no faces — those are enrolled separately, against an existing person, so
    that creating a record and handing over biometric data are two deliberate acts
    rather than one.

    **`camera_ids: []` authorises nowhere.** The opposite default would make forgetting
    to set it a silent grant everywhere, which for an access rule is the failure worth
    designing against.

    `PUT` rather than `POST` because the id is the caller's to choose and the operation
    is idempotent: replaying it produces the same record rather than a second person.
    """
    try:
        person = await service.upsert_person(
            AuthorizedPerson(
                person_id=person_id,
                display_name=body.display_name,
                status=body.status,
                external_reference=body.external_reference,
                camera_ids=frozenset(body.camera_ids),
                zones=frozenset(body.zones),
                expires_at=body.expires_at,
                notes=body.notes,
            )
        )
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    return _person_entry(person, await _reference_count(service, person_id))


@router.delete(
    "/authorized-persons/{person_id}",
    status_code=204,
    summary="Delete a person and every face enrolled for them",
    responses={
        404: {"description": "No enrolled person with this id."},
        503: {"description": "No camera on this engine enables person authorization."},
    },
)
async def delete_person(person_id: UUID, service: ServiceDep) -> Response:
    """Remove somebody and **all** of their biometric data (spec §12).

    Not a soft delete. A record that removed the name while leaving the vectors would
    keep exactly the part that identifies somebody, which is the opposite of what a
    deletion request means.

    Distinct from disabling: `PUT` with `status: disabled` revokes access while keeping
    the record, which is what an investigation of a past incident needs. This is what a
    person exercising a data right needs. Conflating them means one of those two
    obligations cannot be met.
    """
    try:
        deleted = await service.delete_person(person_id)
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"unknown person: {person_id}")
    return Response(status_code=204)


@router.post(
    "/authorized-persons/{person_id}/faces",
    response_model=PersonEntry,
    summary="Enrol one reference face from a photograph",
    responses={
        404: {"description": "No enrolled person with this id."},
        422: {
            "description": (
                "No usable face was found in the image. The reason is in the detail — "
                "an enrolment that silently stored nothing is how somebody becomes "
                "unrecognisable with nobody able to say why."
            )
        },
        503: {"description": "No camera on this engine enables person authorization."},
    },
)
async def enroll_face(person_id: UUID, service: ServiceDep, image: UploadedImage) -> PersonEntry:
    """Turn a photograph into a stored face embedding (spec §9).

    The flow is detect → quality check → align → embed → encrypt → store, and **the
    photograph is not kept**. What persists is the embedding, sealed with AES-256-GCM.
    §12 asks for embeddings over raw biometric images where possible, and here it is
    entirely possible.

    The largest face in the image is used, on the assumption that an enrolment
    photograph is *of* somebody rather than a crowd scene. Choosing by detector
    confidence instead would sometimes pick a sharp bystander over a slightly soft
    subject.

    **Enrol more than one.** §10 asks for multiple references per person, and a single
    face-on office photograph matches a corridor camera at an angle poorly. The usual
    fix for somebody not being recognised is a second reference, not a lower threshold.
    """
    payload = await image.read()
    try:
        pixels = _decode_image(payload)
        count = await service.enroll_face(person_id, pixels, original=payload)
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    except UnknownPersonError as exc:
        raise HTTPException(status_code=404, detail=f"unknown person: {exc.person_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    person = await _person_or_404(service, person_id)
    return _person_entry(person, count)


def _decode_image(payload: bytes) -> object:
    """Decode an uploaded photograph to the BGR array the face pipeline expects.

    Raises `ValueError` on anything undecodable, which the caller maps to 422 — an
    operator who uploaded a PDF should be told that, not handed a 500.
    """
    import io

    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload)) as handle:
            rgb = handle.convert("RGB")
            array = np.array(rgb)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"that file is not an image this engine can read ({exc})") from exc
    # BGR, matching `av`'s `to_ndarray(format="bgr24")` — the format every other frame
    # in this pipeline arrives in, and therefore what the face models were prepared for.
    return array[:, :, ::-1].copy()


async def _reference_count(service: ServiceDep, person_id: UUID) -> int:
    for person, count in await service.list_people():
        if person.person_id == person_id:
            return count
    return 0


async def _person_or_404(service: ServiceDep, person_id: UUID) -> AuthorizedPerson:
    for person, _ in await service.list_people():
        if person.person_id == person_id:
            return person
    raise HTTPException(status_code=404, detail=f"unknown person: {person_id}")


@router.get(
    "/authorized-persons/{person_id}/faces",
    response_model=FacesResponse,
    summary="The reference faces on one person's record — metadata, not pictures",
    responses={
        404: {"description": "No enrolled person with this id."},
        503: {"description": "No camera on this engine enables person authorization."},
    },
)
async def list_faces(person_id: UUID, service: ServiceDep) -> FacesResponse:
    """Which references exist, how old each is, and whether a photograph was kept.

    Enough to draw the roster and to decide which reference to remove when somebody has
    stopped being recognised — usually the oldest, taken under different lighting.
    """
    try:
        faces = await service.list_faces(person_id)
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    except UnknownPersonError as exc:
        raise HTTPException(status_code=404, detail=f"unknown person: {exc.person_id}") from exc
    return FacesResponse(
        faces=[
            EnrolledFaceEntry(
                face_id=face.face_id,
                # 0.0 is this store's "written before enrolment time was recorded".
                # Sent as null so a console shows "unknown" rather than 1970.
                enrolled_at=face.enrolled_at or None,
                has_image=face.has_image,
            )
            for face in faces
        ]
    )


@router.get(
    "/authorized-persons/{person_id}/faces/{face_id}/image",
    summary="One reference photograph",
    response_class=Response,
    responses={
        200: {"content": {"image/jpeg": {}}, "description": "The stored face crop."},
        404: {"description": "No such face, or no photograph was kept for it."},
        503: {"description": "No camera on this engine enables person authorization."},
    },
)
async def read_face_image(person_id: UUID, face_id: UUID, service: ServiceDep) -> Response:
    """The face crop stored at enrolment, decrypted for this one request.

    **Not the photograph that was uploaded.** What is kept is the detector's own face
    box with a margin, at most 320 px on its longest side — enough for a person to
    recognise a person, and deliberately not a copy of whatever else was in the frame
    (§12). See `orchestrator/face_crop.py`.

    `Cache-Control: no-store`, because a browser cache is a copy of biometric data on
    disk that nothing in this system knows about or can delete when the person is
    deleted.
    """
    try:
        image = await service.read_face_image(person_id, face_id)
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    if image is None:
        raise HTTPException(status_code=404, detail="no photograph is stored for that face")
    return Response(
        content=image,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.delete(
    "/authorized-persons/{person_id}/faces/{face_id}",
    status_code=204,
    summary="Remove one reference face and its photograph",
    responses={
        404: {"description": "No such face on that person's record."},
        503: {"description": "No camera on this engine enables person authorization."},
    },
)
async def delete_face(person_id: UUID, face_id: UUID, service: ServiceDep) -> Response:
    """Drop one reference — its embedding and its picture together.

    Finer-grained than deleting the person, and the difference matters: a reference
    taken in bad light makes somebody *harder* to recognise, and the fix is to remove
    that reference, not to un-enrol them and start again.
    """
    try:
        removed = await service.delete_face(person_id, face_id)
    except FaceCapabilityUnavailableError as exc:
        raise _face_unavailable(exc) from exc
    if not removed:
        raise HTTPException(status_code=404, detail=f"unknown face: {face_id}")
    return Response(status_code=204)


@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="The operator's alert list — episodes, not events (volatile)",
)
async def list_alerts(service: ServiceDep) -> AlertsResponse:
    """Every alert this process is holding, worst first then most recent first.

    **An alert is not an event.** An event is what happened; an alert is an *episode*
    that events accumulate into, with an occurrence count. Measured on real footage,
    forty-five seconds of one corridor produced eleven events — and would produce
    eleven rows without this. §17 requires that twenty seconds of the same person is
    one alert saying it happened seventeen times.

    **Volatile, and more sharply so than the event ring.** This register lives in
    engine memory and a restart empties it. The durable record is the anomaly event
    published to RabbitMQ; alert persistence belongs to the Phase 1C store, which is
    not built. The consequence worth stating plainly: **an acknowledgement does not
    survive a restart.** An operator who acknowledged twenty alerts and then saw the
    engine restart is looking at twenty unacknowledged alerts again.

    Ordering is the engine's, not the client's, so that every reader agrees about what
    is at the top — §27's "the critical thing must be visible immediately" is a
    property of that ordering rather than of whoever asked.
    """
    alerts = service.alerts()
    return AlertsResponse(
        alerts=[AlertEntry.from_alert(alert) for alert in alerts],
        open_count=sum(1 for alert in alerts if alert.is_open),
    )


@router.get(
    "/alerts/{alert_id}/clip",
    summary="Watch the recording behind one alert",
    response_class=Response,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "The clip, as MP4."},
        404: {
            "description": (
                "No alert with this id, or no clip to play — nothing was recorded, the "
                "clip has passed its retention window, or this engine has no object "
                "store configured."
            )
        },
    },
)
async def read_alert_clip(alert_id: UUID, service: ServiceDep, short: bool = True) -> Response:
    """The evidence clip, as bytes a browser can play.

    **This is the only route that serves a recording**, and the alert id is the whole of
    the authorisation. The object it reads is looked up from the alert, never taken from
    the request, so no caller can name an object the engine did not itself attach to an
    alert — and `MinioClipReader` re-checks the bucket rather than trusting that.

    `short=true`, the default, returns the notification-length copy
    (`SENTINEL_NOTIFY_CLIP_SECONDS`, three by default) — the length that is watched
    rather than scrolled past when an operator is triaging a wall of rows. It falls back
    to the full recording when no short copy was made, because "here is a longer answer"
    beats "there is nothing here" for somebody asking what happened. `short=false`
    always returns the full clip: pre-roll, the event, post-roll.

    `Cache-Control: private, max-age=60` rather than `no-store`. A clip is fixed once
    written, so re-fetching it on every render is waste — but it is footage of people,
    so it is never a shared-cache entry and never long-lived.
    """
    try:
        clip = await service.read_alert_clip(alert_id, short=short)
    except UnknownAlertError as exc:
        raise HTTPException(status_code=404, detail=f"unknown alert: {alert_id}") from exc
    if clip is None:
        raise HTTPException(
            status_code=404,
            detail="no clip is available for that alert",
        )
    return Response(
        content=clip,
        media_type="video/mp4",
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.delete(
    "/alerts",
    summary="Clear the whole alert list",
    responses={200: {"description": "How many alerts were dropped."}},
)
async def clear_alerts(service: ServiceDep) -> dict[str, int]:
    """Empty the register, and write that emptiness down.

    **Throws away triage state, not evidence.** Every event behind these alerts is
    already published to the broker and is still there; what goes is the record of which
    ones a human had looked at. That is a real loss and the console asks twice before
    calling this.

    Flushed before answering, like acknowledging and resolving: a 200 that did not
    survive a restart would be the same lie in the other direction.
    """
    dropped = service.clear_alerts()
    await service.flush_alerts()
    return {"cleared": dropped}


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertEntry,
    summary="Record that a person has seen this alert",
    responses={
        404: {"description": "No alert with this id is held by the running engine."},
        409: {
            "description": (
                "This alert is already resolved, so acknowledging it would change "
                "nothing — which usually means the operator is acting on a stale list."
            )
        },
    },
)
async def acknowledge_alert(
    alert_id: UUID, body: AcknowledgeRequest, service: ServiceDep
) -> AlertEntry:
    """Move an alert to `acknowledged`.

    The state that matters most on a wall of alerts, because it is the only one that
    distinguishes "nobody has looked" from "somebody is handling it".

    **`by` is a self-declared label, not an identity.** This engine has no
    authentication, so an acknowledgement records what somebody typed. That is worth
    having and it is not an audit trail; see `docs/operations.md`.

    A subsequent recurrence keeps the alert acknowledged rather than re-raising it —
    a person already knows, and putting it back in front of them for something they
    are in the middle of dealing with is how an operator learns to ignore the list.
    """
    try:
        alert = service.acknowledge_alert(alert_id, by=body.by, at=time.time())
    except UnknownAlertError as exc:
        # Not str(exc): `UnknownAlertError` subclasses KeyError, whose __str__ is
        # repr(args[0]), so str() would put literal quotes on the wire.
        raise HTTPException(status_code=404, detail=f"unknown alert: {exc.alert_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # Before answering, not after. A 200 here has to mean the decision is on disk —
    # an operator told "acknowledged" by an engine that then restarts and shows the
    # row as unseen has been lied to about the one piece of state they created.
    await service.flush_alerts()
    return AlertEntry.from_alert(alert)


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertEntry,
    summary="Record that a person has finished with this alert",
    responses={404: {"description": "No alert with this id is held by the running engine."}},
)
async def resolve_alert(alert_id: UUID, service: ServiceDep) -> AlertEntry:
    """Move an alert to `resolved`.

    Idempotent: two operators closing the same row is an ordinary race, not a mistake
    either of them made.

    A resolved alert stops absorbing recurrences — a person said it was finished, and
    quietly reopening it would erase that judgement. The same situation happening again
    opens a new alert, which is the truthful reading.

    **Nothing resolves itself.** There is no timeout and no auto-close anywhere in this
    subsystem, because an alert that expired quietly would leave no trace that nobody
    ever went to look.
    """
    try:
        alert = service.resolve_alert(alert_id)
    except UnknownAlertError as exc:
        raise HTTPException(status_code=404, detail=f"unknown alert: {exc.alert_id}") from exc
    # Awaited for `acknowledge`'s reason. Resolving is the more consequential of the
    # two: it stops the alert absorbing recurrences, so a resolve that did not survive
    # a restart would put a closed episode back in front of an operator *and* let it
    # start growing again.
    await service.flush_alerts()
    return AlertEntry.from_alert(alert)


@router.post("/cameras/{camera_id}/describe", response_model=DescribeResponse)
async def describe_camera_now(camera_id: str, service: ServiceDep) -> DescribeResponse:
    try:
        event_id = await service.describe_now(camera_id)
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    return DescribeResponse(event_id=event_id)


@router.patch(
    "/cameras/{camera_id}",
    response_model=CameraEditResponse,
    summary=(
        "Edit a camera's label, zone and welfare notification policy — "
        "UNAUTHENTICATED, and off by default"
    ),
    responses={
        403: {
            "description": (
                "Camera writes are disabled on this deployment, which is the default. "
                "Set `SENTINEL_ENABLE_CAMERA_WRITES=true` on the engine to enable "
                "them, having read the warning in this operation's description first."
            )
        },
        404: {"description": "No camera with this id is configured in the running engine."},
        409: {
            "description": (
                "The edit cannot be taken as things currently stand, and **nothing "
                "was written**. Either `cameras.json` no longer admits it — most "
                "often because the file has been changed by hand since the engine "
                "started and no longer contains this camera — or `capabilities` "
                "named one whose model this process did not load, which needs a "
                "restart rather than a retry."
            )
        },
        500: {"description": "The camera file could not be written. Nothing was changed."},
    },
)
async def update_camera(
    camera_id: str,
    edit: CameraEditRequest,
    service: ServiceDep,
    writes_enabled: WritesEnabledDep,
) -> CameraEditResponse:
    """Change what a camera is called, which zone it is grouped into, and what its
    welfare concerns notify a human about.

    ### This endpoint is not authenticated

    **Phase 1B ships no authentication of any kind.** The console's login screen
    is a shell, JWT arrives in Phase 1C, and until then anything that can open a
    TCP connection to this port can call this. Every other endpoint is a read, so
    reaching the port has so far cost an attacker information; this one is a
    write, and a persisted one. An anonymous caller can rename a camera to another
    camera's name and re-zone it into another wing — which is to say, make the
    console's account of *where an incident happened* wrong, in a custodial
    setting, permanently, because the change is written to `cameras.json` and
    survives the restart that would otherwise undo it.

    That is why it is **off unless a deployment turns it on**
    (`SENTINEL_ENABLE_CAMERA_WRITES=true`) and answers 403 otherwise, and why a
    deployment that turns it on should also bind the engine to localhost or put an
    authenticating reverse proxy in front of it. `GET /cameras` reports the current
    posture as `config_writable`. See `docs/operations.md`.

    ### What can be edited, and what cannot

    Editable at runtime, applied to the running camera and written to
    `cameras.json` before this returns:

    * `label` — the display name. Trimmed, non-empty, at most 120 characters.
    * `zone` — `room`, `corridor`, `dayroom`, or `null` to ungroup. Omitting the
      field and sending `null` are different instructions.
    * `notify_on` — which welfare concern kinds notify a human, as a whole
      replacement list. `[]` means never notify from this camera and is a real
      edit, not an empty one; `null` is rejected because `[]` already says it.
    * `notify_min_confidence` — `possible` or `likely`, the lowest tier that may
      notify. There is no `certain`: one still frame cannot earn it.
    * `clip_preroll_seconds`, `clip_postroll_seconds`,
      `summary_interval_seconds` — per-camera overrides of the engine-wide clip
      bounds and the profile's summary interval. **Omitting one and sending
      `null` are different instructions**, as with `zone`: omit to leave the
      override alone, send `null` to drop it and go back to the default. Pre-roll
      may be `0` (no lead-in is a real choice); the other two must be above zero.

    Every one of them comes back in the response as it was stored, so a console
    renders what the file now says rather than what it hoped it would say.

    **Not editable, and rejected with 422 rather than ignored** — the response's
    `restart_required_fields` names them:

    * `url` — changing the source means tearing down the running `CameraRunner`,
      its buffered pre-roll and any clip mid-recording, and building a new source
      in their place. It is a camera restart, not an edit. Separately, an RTSP URL
      routinely carries credentials, so an unauthenticated API neither accepts nor
      returns it.
    * `profile` — the escalation policy, which the gate is part-way through
      applying (cooldowns, a token bucket with state). Swapping it mid-flight has
      no defensible semantics.

    Both are edited by changing `cameras.json` and restarting the engine.

    ### Atomicity

    The edit is validated against the whole document, written to a temporary file,
    `fsync`ed, and renamed into place. It is never half-applied: a request that
    would produce a file the engine could not load at its next startup is refused
    outright, and everything the file holds that this engine has no model of —
    comment keys, other cameras, fields added later — is preserved byte-for-byte
    in meaning. The in-memory camera is only updated *after* the file is on disk,
    and from what the file now says, so the two cannot disagree.
    """
    _require_writes(writes_enabled)
    try:
        record = await service.update_camera(camera_id, edit.to_edit())
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
    except CapabilityUnavailableError as exc:
        # 409, the same code a file that cannot admit the edit gets, and for the same
        # reason: the request is well-formed and the engine understood it, but the
        # process as it is currently running cannot take it. Nothing was written —
        # this is raised before the store is touched (see
        # `EngineService.ensure_capabilities_available`), so the record and the
        # running camera still agree.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CameraConfigError as exc:
        # 409 rather than 400: the request is well-formed and the engine accepted
        # it, but the file it must be written into is not in a state that admits
        # the edit. Nothing was written — see `CameraFileStore.apply`.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        # The rename is the last step, so a failure here leaves the previous
        # document intact and the running camera unchanged. Say that, rather than
        # letting a bare 500 leave an operator unsure whether it half-landed.
        raise HTTPException(
            status_code=500,
            detail=f"could not write the camera file ({exc}); nothing was changed",
        ) from exc
    return CameraEditResponse.from_config(record)
