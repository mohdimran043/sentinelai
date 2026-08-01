"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sentinel_ai.api.schemas import (
    CameraEventsResponse,
    CamerasResponse,
    CameraStatus,
    DescribeResponse,
    HealthResponse,
    ModelHealth,
)
from sentinel_ai.orchestrator.event_history import CameraEventHistory
from sentinel_ai.orchestrator.service import UnknownCameraError
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
