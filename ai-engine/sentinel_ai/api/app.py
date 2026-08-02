"""App factory + lifespan (spec §5.7). The service is injected, never
constructed here, so tests never build a real EngineService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sentinel_ai.api.routes import EngineServiceProtocol, router


def create_app(service: EngineServiceProtocol, *, camera_writes_enabled: bool = False) -> FastAPI:
    """Build the app.

    `camera_writes_enabled` is a *deployment* fact, not an engine one, which is why
    it arrives here rather than on `EngineServiceProtocol`: the engine is equally
    capable of applying an edit either way, and what the flag expresses is whether
    this port is somewhere an unauthenticated caller may be trusted to ask. Keeping
    it off the service protocol also keeps every fake in the test suite from having
    to answer a question about deployment posture.

    Defaults to False so that a caller who has not thought about it — a test, a
    script, an embedding — gets the safe posture; `main.create_default_app` is the
    only place that reads `Settings.enable_camera_writes` and passes True.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            try:
                await service.stop()
            finally:
                # After `stop()`, so the escalations its ordering exists to publish
                # still reach a console that is watching; in a `finally`, so a
                # `stop()` that is itself cancelled — uvicorn's
                # `--timeout-graceful-shutdown` does exactly that — still releases
                # every reader instead of leaving them parked on a dead engine. It is
                # synchronous and cannot be interrupted, and it does not swallow the
                # cancellation aimed at `stop()`, which stays cut short.
                service.close_event_streams()

    app = FastAPI(title="SentinelAI AI Engine", lifespan=lifespan)
    app.state.service = service
    app.state.camera_writes_enabled = camera_writes_enabled
    app.include_router(router)
    return app
