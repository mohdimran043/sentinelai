"""App factory + lifespan (spec §5.7). The service is injected, never
constructed here, so tests never build a real EngineService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sentinel_ai.api.routes import EngineServiceProtocol, router


def create_app(service: EngineServiceProtocol) -> FastAPI:
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
    app.include_router(router)
    return app
