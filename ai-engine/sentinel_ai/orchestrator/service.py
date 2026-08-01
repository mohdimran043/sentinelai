"""The single entry point the API delegates to (spec §5.4, §5.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from uuid import UUID

from sentinel_ai.orchestrator.registry import ModelRegistry
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport

__all__ = ["EngineService", "UnknownCameraError"]


class UnknownCameraError(KeyError):
    """Raised by `telemetry()`/`describe_now()` for an unregistered camera id.

    A `KeyError` subclass so existing `except KeyError` call sites keep working, and
    a distinct type so the API layer (Task 10) can map it to HTTP 404 without
    catching every other `KeyError` in the process too.
    """

    def __init__(self, camera_id: str) -> None:
        super().__init__(f"unknown camera: {camera_id}")
        self.camera_id = camera_id


class EngineService:
    """Owns the process's cameras, models, and scheduler; the API talks to nothing else."""

    def __init__(
        self,
        cameras: Mapping[str, CameraRunner],
        registry: ModelRegistry,
        resident_set: ResidentSet,
        scheduler: VlmScheduler,
        required_model_keys: tuple[str, ...],
        clock: Callable[[], float],
    ) -> None:
        self._cameras = dict(cameras)
        self._registry = registry
        self._resident_set = resident_set
        self._scheduler = scheduler
        self._required_model_keys = required_model_keys
        self._clock = clock
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Load the required models, then start the VLM worker and every camera."""
        await self._resident_set.ensure(self._required_model_keys, self._clock())
        self._tasks.append(asyncio.create_task(self._scheduler.run()))
        for runner in self._cameras.values():
            self._tasks.append(asyncio.create_task(runner.run()))

    async def stop(self) -> None:
        """Cancel every task this service started and wait for them to unwind."""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return tuple(runner.telemetry() for runner in self._cameras.values())

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        return self._get_runner(camera_id).telemetry()

    def health(self) -> dict[str, HealthReport]:
        return self._registry.health()

    async def describe_now(self, camera_id: str) -> UUID:
        return await self._get_runner(camera_id).describe_now()

    def _get_runner(self, camera_id: str) -> CameraRunner:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise UnknownCameraError(camera_id) from None
