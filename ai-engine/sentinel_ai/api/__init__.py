"""FastAPI surface — thin delegation to orchestrator.service.EngineService (spec §5.7).

No business logic lives here; every endpoint validates the path/body and
calls straight into EngineService. No WebSocket route in this phase — the Go
backend owns the WS hub starting in Phase 1C.
"""
