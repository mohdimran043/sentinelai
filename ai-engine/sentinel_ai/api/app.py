"""App factory + lifespan (spec §5.7). The service is injected, never
constructed here, so tests never build a real EngineService."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sentinel_ai.api.routes import EngineServiceProtocol, router


def _with_non_finite_floats_nulled(value: Any) -> Any:
    """Replace every `inf`/`-inf`/`NaN` anywhere in a validation-error structure
    with `null`, leaving everything else — including every message — untouched.

    `null` rather than a string because that is what the rest of the JSON world
    already does with these: `JSON.stringify` emits `null`, and pydantic's own
    `to_json` nulls them by default. A caller who sent an infinity and reads
    `"input": null` learns the same thing either way, which is that the value the
    server saw was not a number it can work with.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _with_non_finite_floats_nulled(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_with_non_finite_floats_nulled(item) for item in value]
    return value


async def _render_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's own 422, with any non-finite float in it nulled out first.

    **App-wide: this replaces the default handler for every endpoint, not just the
    one that made it necessary.** The body shape is deliberately identical to
    FastAPI's — `{"detail": [ ... ]}` with the same `type`/`loc`/`msg`/`input`/`ctx`
    per error — because a console parses these, and the only difference a caller can
    observe is a `null` where an unserialisable float would have been.

    Without this, *any* validation error whose echoed input contains an infinity or
    a `NaN` is a 500 rather than a 422. FastAPI puts the offending value in the
    error's `input`, and Starlette's `JSONResponse` renders with `allow_nan=False`,
    so rendering the 422 raises and the caller gets a crash instead of the answer.
    The reachable route today is `PATCH /cameras/{camera_id}`'s duration overrides —
    `1e999` and `NaN` are valid JSON that `json.loads` accepts and every `ge`/`gt`
    bound rejects — but nothing about the failure is specific to them, so the fix
    belongs where the response is rendered rather than on the fields. Any float
    field added anywhere later is covered by having been added, with nobody needing
    to know this exists.
    """
    return JSONResponse(
        # The same 422 FastAPI's own handler sends, spelled as an integer because
        # importing Starlette's constant for it is a test failure: on the pinned
        # 1.3.1 the old `HTTP_422_UNPROCESSABLE_ENTITY` raises
        # `StarletteDeprecationWarning` on import *and* on attribute access, and
        # this suite runs under `filterwarnings = ["error"]`, so a module that
        # names it does not even collect. The replacement
        # (`HTTP_422_UNPROCESSABLE_CONTENT`) is not deprecated, but pinning to it
        # buys nothing over the literal and `routes.py` spells its statuses as
        # bare ints throughout.
        status_code=422,
        content={"detail": jsonable_encoder(_with_non_finite_floats_nulled(exc.errors()))},
    )


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
    app.exception_handler(RequestValidationError)(_render_validation_error)
    app.state.service = service
    app.state.camera_writes_enabled = camera_writes_enabled
    app.include_router(router)
    return app
