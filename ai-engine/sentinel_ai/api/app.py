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
            await service.stop()

    app = FastAPI(title="SentinelAI AI Engine", lifespan=lifespan)
    app.state.service = service
    app.include_router(router)
    return app
