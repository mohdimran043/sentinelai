"""The single entry point the API delegates to (spec §5.4, §5.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from uuid import UUID

from sentinel_ai.orchestrator.registry import ModelRegistry
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport

__all__ = ["EngineService", "UnknownCameraError"]

_DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS = 60.0
"""How often `ResidentSet.sweep_idle` is polled, not the idle-unload window itself
(that is `ModelSpec.idle_unload_seconds`, spec §5.4's 600s for the VLM). Comfortably
below 600s so the unload fires within a bounded margin of its deadline rather than
being detected up to a whole interval late."""


class _IdleSweeper:
    """Periodically calls `sweep`; owns no task or model-registry state of its own.

    Split out exactly like `_ReconnectLoop` (Task 14, adapters/sources/rtsp.py): a
    plain "wait, then act" sequencer, so it is unit-testable with an injected clock
    and sleep function — no real wall-clock time, no asyncio task, no GPU. Composed
    with `ResidentSet.sweep_idle` as `sweep` here; nothing about it is specific to
    resident-set eviction.
    """

    def __init__(
        self,
        sweep: Callable[[float], Awaitable[None]],
        *,
        interval_seconds: float,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sweep = sweep
        self._interval = interval_seconds
        self._clock = clock
        self._sleep = sleep

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        while not should_stop():
            await self._sleep(self._interval)
            if should_stop():
                return
            await self._sweep(self._clock())


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
        idle_sweep_interval_seconds: float = _DEFAULT_IDLE_SWEEP_INTERVAL_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._cameras = dict(cameras)
        self._registry = registry
        self._resident_set = resident_set
        self._scheduler = scheduler
        self._required_model_keys = required_model_keys
        self._clock = clock
        self._tasks: list[asyncio.Task[None]] = []
        self._idle_sweeper = _IdleSweeper(
            resident_set.sweep_idle,
            interval_seconds=idle_sweep_interval_seconds,
            clock=clock,
            sleep=sleep,
        )

    async def start(self) -> None:
        """Load the required models, then start the VLM worker, every camera, and the
        periodic idle sweep (spec §5.4's 600s VLM idle-unload, S7-deferred to here)."""
        await self._resident_set.ensure(self._required_model_keys, self._clock())
        self._tasks.append(asyncio.create_task(self._scheduler.run()))
        for runner in self._cameras.values():
            self._tasks.append(asyncio.create_task(runner.run()))
        self._tasks.append(
            asyncio.create_task(self._idle_sweeper.run_forever(should_stop=lambda: False))
        )

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
