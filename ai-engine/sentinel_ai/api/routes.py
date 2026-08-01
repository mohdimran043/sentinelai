"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from sentinel_ai.api.schemas import (
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


def get_service(request: Request) -> EngineServiceProtocol:
    service: EngineServiceProtocol = request.app.state.service
    return service


# `Annotated[..., Depends(...)]` rather than a `Depends(...)` default value: the
# latter is a function call in an argument default, which this project's ruff
# config (B008) rightly flags everywhere else, so the FastAPI-idiomatic
# workaround is used instead of a per-line suppression.
ServiceDep = Annotated[EngineServiceProtocol, Depends(get_service)]

router = APIRouter()


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
async def list_cameras(service: ServiceDep) -> CamerasResponse:
    return CamerasResponse(cameras=[CameraStatus.from_telemetry(t) for t in service.cameras()])


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
        # Not str(exc): UnknownCameraError subclasses KeyError, whose __str__ is
        # repr(args[0]), so str() would put literal quotes on the wire —
        # {"detail": "'unknown camera: cam-x'"}.
        raise HTTPException(status_code=404, detail=f"unknown camera: {exc.camera_id}") from exc
    return DescribeResponse(event_id=event_id)
