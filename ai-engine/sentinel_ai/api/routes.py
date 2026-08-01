"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

from typing import Annotated, Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sentinel_ai.api.schemas import (
    CamerasResponse,
    CameraStatus,
    DescribeResponse,
    HealthResponse,
    ModelHealth,
)
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
