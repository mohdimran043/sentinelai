# SentinelAI

Behaviour-intelligence video surveillance. SentinelAI watches camera streams,
detects and tracks objects continuously on cheap signals, and spends GPU on a
vision-language model **only when the scene warrants it** — then publishes a
validated anomaly event with an evidence clip.

That gate is the product, not an optimisation. Describing every frame with a VLM
is unaffordable; describing nothing is useless. The
[escalation gate](docs/architecture.md#the-escalation-gate) is where SentinelAI
decides which frames are worth the money.

```
RTSP / file ─┬─► decode ──► detect ──► track ──► motion ──► escalation gate ──┐
             │   (CPU)     (YOLO11s)  (ByteTrack)  (pure)        (pure)       │
             │                                                                │
             └─► encoded packets ──► pre-roll ring buffer ──► clip (remux) ◄──┤
                                                                              ▼
                                              VLM (Qwen2.5-VL, 4-bit NF4) ────┤
                                                                              ▼
                                       anomaly event ──► RabbitMQ    clip ──► MinIO
```

## What works today

| Area | State |
|---|---|
| AI engine pipeline — decode, detect, track, motion, gate, VLM, clip, publish | **Working**, verified end to end on a real GPU box |
| FastAPI control surface | **Working** — `/health`, `/cameras`, `/cameras/{id}/telemetry`, `/cameras/{id}/events`, `/events/stream`, `POST /cameras/{id}/describe`, and `PATCH /cameras/{id}` to edit a camera's label and zone |
| Editing cameras from the console | **Working**, and **off by default.** The engine has no authentication, so the one write endpoint ships disabled behind `SENTINEL_ENABLE_CAMERA_WRITES`. `label` and `zone` are editable live and persisted to `cameras.json`; `url` and `profile` require a restart and are rejected rather than ignored. See [docs/operations.md](docs/operations.md#editing-cameras-from-the-console) |
| RabbitMQ publishing with disk spool + replay + dead-letter | **Working** |
| MinIO evidence clips (remux, 3 s pre-roll + 5 s post-roll) | **Working** |
| VRAM residency planner, LRU + priority eviction, 600 s VLM idle unload | **Working** |
| Web console — login, dashboard, camera page, recorder console sections | **Working** against live engine + recorder; event feed is MSW-mocked |
| Go backend / Postgres / Redis (Phase 1C) | **Not built.** Scaffolded in compose only |
| WebSocket event hub | **Not built.** The UI mocks the event feed with MSW |
| Continuous behaviour detection (beyond the seven triggers) | **Not built.** Phase 4 |
| Production `gRPC` transport (`SENTINEL_MODE=production`) | **Not built.** The setting exists; nothing branches on it yet |
| CORS on the engine | **Missing**, and unowned — see [Operations](docs/operations.md#known-deployment-gaps) |

Tests: **the entire CPU suite runs in seconds, on any machine, with no GPU**
(500+ tests at the time of writing; 19 more are marked `gpu`/`integration` and
deselected in CI). That is deliberate and it is the point of the architecture —
see below.

## Repository layout

```
ai-engine/          The Python inference engine. The product's core.
  sentinel_ai/
    domain/         Pure entities and policy. No I/O, no clock, no outer imports.
    ports/          Pure abstract interfaces — the seams. Same rules.
    adapters/       Everything that touches the world: YOLO, Qwen, PyAV, MinIO, RabbitMQ.
    orchestrator/   Model registry, VRAM resident set, GPU admission, VLM scheduler.
    pipeline/       One asyncio task per camera: the frame loop.
    api/            FastAPI routes over the orchestrator.
    config.py       Every SENTINEL_* setting. The only Development/Production seam.
    main.py         The composition root — the one place the real system is built.
  tests/            481 tests, incl. architecture fitness functions and port contracts.

contracts/          The ONLY coupling between the engine and the web UI.
  openapi/          ai-engine.yaml, generated from the app, drift-tested in CI.
  events/           anomaly_event.schema.json, validated on every publish.

web/                React 19 + Vite operator console. Generates its types from contracts/.
deploy/compose/     Postgres, Redis, RabbitMQ, MinIO, mediamtx.
datasets/           Sample-clip fetch script. Footage is never committed.
docs/               This documentation.
```

## Shortest path to seeing it run

No GPU needed for the tests. A GPU is needed for real inference.

```bash
# 1. Engine, CPU only — this is the whole test suite, green, in ~3s.
make install-runtime
make test

# 2. Core services (RabbitMQ, MinIO, mediamtx, Postgres, Redis).
make up

# 3. Real models on a GPU box.
make install-gpu
bash datasets/download_sample.sh
cd ai-engine && cp cameras.example.json cameras.json
uvicorn sentinel_ai.main:app --port 8000

# 4. The console, in another terminal.
cd web && npm install && npm run dev      # http://localhost:5173
```

Then `curl localhost:8000/health`, and
`curl -X POST localhost:8000/cameras/avenue_01/describe` to force one VLM call.

Full walkthrough, including streaming the sample clip through mediamtx as a live
RTSP camera: [docs/operations.md](docs/operations.md).

## Documentation

| Document | Read it when |
|---|---|
| [docs/architecture.md](docs/architecture.md) | You want to know how it fits together and why the layering is enforced |
| [docs/decisions.md](docs/decisions.md) | **Start here.** The decisions a newcomer would otherwise re-litigate or undo |
| [docs/extending.md](docs/extending.md) | You are adding a camera source, detector, VLM, publisher, trigger, or console page |
| [docs/configuration.md](docs/configuration.md) | You are tuning it — all 36 `SENTINEL_*` settings and `cameras.json` |
| [docs/operations.md](docs/operations.md) | You are running it, demoing it, or deploying it |
| [contracts/README.md](contracts/README.md) | You are changing anything that crosses the engine ↔ UI boundary |
| [ai-engine/README.md](ai-engine/README.md) | You want the engine's own deep-dive on sources, reconnect, and the demo |
| [web/README.md](web/README.md) | You are working in the console |

Design specs and phase plans live in `docs/superpowers/` — that is the process
working area, not product documentation.
