"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from sentinel_ai.adapters.config.camera_file import CameraConfig, CameraConfigError, CameraEdit
from sentinel_ai.api.schemas import (
    CameraEditRequest,
    CameraEditResponse,
    CameraEventsResponse,
    CamerasResponse,
    CameraStatus,
    DescribeResponse,
    HealthResponse,
    ModelHealth,
)
from sentinel_ai.api.sse import SSE_HEADERS, SSE_MEDIA_TYPE, event_stream_body
from sentinel_ai.orchestrator.event_history import CameraEventHistory, EventSubscription
from sentinel_ai.orchestrator.service import EngineNotComposedError, UnknownCameraError
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
    async def describe_now(self, camera_id: str) -> UUID: ...
    async def update_camera(self, camera_id: str, edit: CameraEdit) -> CameraConfig: ...


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
WritesEnabledDep = Annotated[bool, Depends(camera_writes_enabled)]

router = APIRouter()


def _unknown_camera(exc: UnknownCameraError) -> HTTPException:
    # Not str(exc): UnknownCameraError subclasses KeyError, whose __str__ is
    # repr(args[0]), so str() would put literal quotes on the wire —
    # {"detail": "'unknown camera: cam-x'"}.
    return HTTPException(status_code=404, detail=f"unknown camera: {exc.camera_id}")


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
        cameras=[CameraStatus.from_telemetry(t) for t in service.cameras()],
        config_writable=writes_enabled,
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
    return CameraStatus.from_telemetry(telemetry)


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
                "`cameras.json` cannot be edited as it currently stands — most often "
                "because the file has been changed by hand since the engine started "
                "and no longer contains this camera. Nothing was written."
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
    if not writes_enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "camera writes are disabled on this engine. This build has no "
                "authentication, so the write endpoints are opt-in: set "
                "SENTINEL_ENABLE_CAMERA_WRITES=true only on a deployment where "
                "everything that can reach this port is permitted to reconfigure "
                "cameras. Until then, edit cameras.json and restart."
            ),
        )
    try:
        record = await service.update_camera(camera_id, edit.to_edit())
    except UnknownCameraError as exc:
        raise _unknown_camera(exc) from exc
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
