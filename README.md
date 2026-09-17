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
                                              │
                                              └─► welfare note ──► log / webhook
```

## What works today

| Area | State |
|---|---|
| AI engine pipeline — decode, detect, track, motion, gate, VLM, clip, publish | **Working**, verified end to end on a real GPU box |
| FastAPI control surface | **Working** — `/health`, `/cameras`, `/cameras/{id}/telemetry`, `/cameras/{id}/events`, `/events/stream`, `POST /cameras/{id}/describe`, and `PATCH /cameras/{id}` to edit a camera's label, zone, AI capabilities and welfare policy |
| Editing cameras from the console | **Working**, and **off by default.** The engine has no authentication, so the one write endpoint ships disabled behind `SENTINEL_ENABLE_CAMERA_WRITES`. `label`, `zone`, `capabilities` and the per-camera welfare policy are editable live and persisted to `cameras.json`; `url` and `profile` require a restart and are rejected rather than ignored. See [docs/operations.md](docs/operations.md#editing-cameras-from-the-console) |
| Resident welfare notifications | **Working** — the VLM is asked about collapse, altercation, self-harm, apparent medication intake and other distress; qualifying concerns reach a human by log line or webhook, with the clip's URL, and every concern (muted or not) shows on the camera page. **It asks a vision-language model about single frames. It is not a fall detector, it misses things, and a stretcher carry was missed entirely in measurement** — read [the limits](docs/operations.md#read-this-before-you-rely-on-it) before relying on it |
| RabbitMQ publishing with disk spool + replay + dead-letter | **Working** |
| MinIO evidence clips (remux, 3 s pre-roll + 5 s post-roll, per-camera overridable) | **Working** |
| VRAM residency planner, LRU + priority eviction, 600 s VLM idle unload | **Working** |
| Web console — command centre, alert queue, people, AI system, camera page, recorder sections | **Working** against live engine + recorder; event feed is MSW-mocked |
| Go backend / Postgres / Redis (Phase 1C) | **Not built.** Scaffolded in compose only |
| WebSocket event hub | **Not built.** The UI mocks the event feed with MSW |
| Per-camera AI capabilities | **Working** — `capabilities` in `cameras.json`, editable at runtime, and load-bearing: the process builds only the models some camera actually asks for, so a triggers-only site never places a VLM. See [configuration](docs/configuration.md#per-camera-ai-capabilities) |
| Fall / collapse detection | **Built and opt-in, and now measured — the measurement does not flatter it.** A pure temporal state machine, then vision-language confirmation, published with `basis: temporal_pose_vlm`. On URFall it saw the signature in 29 of 30 falls and **raised none of them**, while 23 of 40 ordinary daily activities produced the same signature. Do not rely on it without reading [the limits](docs/operations.md#fall-detection-read-this-before-you-rely-on-it-either) and [the numbers](docs/performance.md#fall-detection-urfall-70-clips) |
| Pose estimation (YOLO11n-pose) | **Working**, loaded only where `fall_detection` is enabled. Measured on an RTX 4090: 4.4 ms p50 at 1080p, 462 MiB |
| Benchmark harness | **Working** — `python -m sentinel_ai.benchmark` composes the real engine over N cameras and reports per-stage p50/p95/max, drop rate and peak VRAM. Every performance number in these docs comes from it |
| Continuous behaviour detection | **Working**, opt-in per camera. Abandoned object, camera tamper, zone intrusion and line crossing — pure temporal state machines, distances in body heights rather than pixels. Loitering was **already** the `dwell_exceeded` trigger and was tuned rather than duplicated. See [ADR 13](docs/decisions.md#13-behaviour-detectors-are-pure-state-machines-measured-in-body-heights) |
| Camera tamper detection | **Working**, and **free** — it reads the luma histogram the motion stage already computes, so it needs no model and can run on every camera |
| Person authorization / face recognition | **Working**, opt-in per camera and **off by default.** InsightFace SCRFD + ArcFace under ONNX Runtime; 512-d embeddings sealed individually with AES-256-GCM, **no face images stored**, no embeddings or scores logged. The engine refuses to start without `SENTINEL_FACE_ENCRYPTION_KEY`. **Measured on LFW's 2200-pair protocol at the shipped threshold: 92.7% recall, zero false matches, 96.4% accuracy.** Even-handedness partly measured on FairFace and [not uniform](docs/performance.md#person-authorization-even-handedness-fairface-1800-images). **Unauthorized is a low-severity finding, not an alarm** — read [ADR 15](docs/decisions.md#15-person-authorization-stores-sealed-embeddings-and-nothing-else) |
| Alert engine (episodes, occurrence aggregation, acknowledge/resolve) | **Working** — an alert is keyed by camera + reason + subject, so one person seen seventeen times in twenty seconds is one alert with `occurrences: 17`, not seventeen rows. See [ADR 14](docs/decisions.md#14-an-alert-is-an-episode-not-an-event) |
| Durable triage state | **Working** — acknowledgements and resolutions are flushed to disk before the API answers, and survive a `kill -9`. Verified end to end. Occurrence counts are flushed on a 5 s timer. See [ADR 16](docs/decisions.md#16-triage-state-is-durable-the-event-stream-is-still-the-record) |
| Accuracy validation on public datasets | **Working** — `python -m sentinel_ai.validation.{falls,faces}` scores the fall machine against URFall and the face pipeline against LFW and FairFace, with the real models. Every accuracy number in the docs comes from it |
| Frame-rate cap on live sources | **Working** — `SENTINEL_SOURCE_MAX_FPS` stops a 30 fps camera waking the event loop 25 times a second to discard a frame. Took three 1080p cameras from **85% dropped to 0.0%**, every delivered frame detected. A cap, not magic: it sets the temporal resolution of everything downstream |
| Adding a camera from the console | **Working** — `POST /cameras` persists it and starts watching with no restart, `DELETE` stops and removes it. The console previews the stream first, showing a real decoded frame, because a camera added with a wrong URL otherwise looks present and never delivers anything. See [configuration](docs/configuration.md#camera-sources) |
| Short notification clips, with retention | **Working** — a notification at or above `high` carries a 3 s cut of the pre-roll rather than the full evidence clip, and clips expire after a day through a bucket lifecycle rule the object store enforces. Both configurable; `0` disables either |
| Public EarthCam cameras as a source | **Working** — configure the camera's web page, not an `.m3u8`. The resolver reads the page's own configuration, picks the best variant, and re-resolves on every reconnect because the playlist URL is signed and expires. Verified live against Linkou Old Street (Taiwan) and Bourbon Street (New Orleans). Signatures never reach a log. See [configuration](docs/configuration.md#earthcam-pages) |
| Hardware decode (`SENTINEL_DECODE_HWACCEL`) | **Working**, and **off by default because it is measured to be slower end to end.** CUDA decode costs a quarter of the CPU and each delivered frame costs more. A device this FFmpeg cannot open is a startup error, never a silent fallback |
| Production `gRPC` transport (`SENTINEL_MODE=production`) | **Not built.** The setting exists; nothing branches on it yet |
| CORS on the engine | **Missing**, and unowned — see [Operations](docs/operations.md#known-deployment-gaps) |

Tests: **the entire CPU suite runs in seconds, on any machine, with no GPU**
(1017 engine tests at the time of writing, plus 488 in the console; 19 more are
marked `gpu`/`integration` and deselected in CI). That is deliberate and it is
the point of the architecture — see below.

## Repository layout

```
ai-engine/          The Python inference engine. The product's core.
  sentinel_ai/
    domain/         Pure entities and policy. No I/O, no clock, no outer imports.
      behaviour/    Temporal state machines — the fall signature lives here, pure.
    ports/          Pure abstract interfaces — the seams. Same rules.
    adapters/       Everything that touches the world: YOLO, Qwen, PyAV, MinIO, RabbitMQ.
    orchestrator/   Model registry, VRAM resident set, GPU admission, VLM scheduler.
    pipeline/       One asyncio task per camera: the frame loop.
    api/            FastAPI routes over the orchestrator.
    benchmark/      Measurement (§32). Imports the engine; nothing imports it.
    config.py       Every SENTINEL_* setting. The only Development/Production seam.
    main.py         The composition root — the one place the real system is built.
  tests/            915 tests, incl. architecture fitness functions and port contracts.

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
| [docs/configuration.md](docs/configuration.md) | You are tuning it — all 58 `SENTINEL_*` settings and `cameras.json` |
| [docs/operations.md](docs/operations.md) | You are running it, demoing it, or deploying it |
| [docs/performance.md](docs/performance.md) | You want to know what it actually costs — every figure measured on a named box, with a section on what has *not* been measured |
| [contracts/README.md](contracts/README.md) | You are changing anything that crosses the engine ↔ UI boundary |
| [ai-engine/README.md](ai-engine/README.md) | You want the engine's own deep-dive on sources, reconnect, and the demo |
| [web/README.md](web/README.md) | You are working in the console |
| [AGENTS.md](AGENTS.md) | You are a coding agent, or new here — the enforced rules, the commands, and a ledger of what is actually built |

Design specs and phase plans live in `docs/superpowers/` — that is the process
working area, not product documentation.
