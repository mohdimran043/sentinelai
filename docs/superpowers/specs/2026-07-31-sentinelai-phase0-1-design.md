# SentinelAI — Phase 0+1 Design

**Date:** 2026-07-31
**Scope:** Foundation & contracts (Phase 0) + single-camera live intelligence vertical slice (Phase 1)
**Status:** Approved

---

## 1. Context

`master-prompt.md` specifies a complete AI behaviour intelligence platform across 35 sections:
roughly 20 independently deployable services, 5 datastores, 11 model families, an MLOps
retraining pipeline, an 8-channel notification fan-out, and a ~30-screen web application.

That is not implementable as a single spec. It is decomposed into ten sub-projects, each
receiving its own spec → plan → implementation cycle. This document covers the first two,
which are specified together because Phase 0 has no observable behaviour on its own.

### Decomposition roadmap

| # | Sub-project | Delivers | Master-prompt sections |
|---|---|---|---|
| 0 | Foundation & contracts | Monorepo, API contracts, event schema, infra, CI | §2, §11, §35 |
| 1 | Vertical slice — one camera, live | decode → detect → track → gate → VLM → event → UI | §3, §6, §7, §18, §22 |
| 2 | Camera management & pipelines | CRUD, sources, per-camera model selection | §9, §16, §17 |
| 3 | Model lifecycle & orchestrator hardening | Registry, lazy load/unload, LB, failover, dashboard | §4, §5, §30 |
| 4 | Behaviour engine & anomaly detection | Per-camera baselines, anomaly scoring, 18 detectors | §20, §21 |
| 5 | Storage & retrieval | Timeline, replay, ElasticSearch, Qdrant, NL search | §19, §24, §26, §33 |
| 6 | Notifications | Multi-channel fan-out with evidence | §23 |
| 7 | Narrative layer | Daily journal, reports, AI chat | §27, §28, §29 |
| 8 | Active learning & retraining | Feedback, dataset versioning, retrain/eval/rollback | §31, §32 |
| 9 | Production topology | Multi-machine, multi-GPU, Kubernetes | §11, §34 |

Nothing from the master prompt is dropped. It is sequenced.

---

## 2. Hardware reality and its consequences

`master-prompt.md` §3.1 specifies a single RTX 4090 (24 GB). The actual development machine is:

| Resource | Available |
|---|---|
| GPU | RTX 4060 Laptop, 8 GB VRAM |
| Driver / CUDA | 580.173.02 / CUDA 12.6 |
| System RAM | 15 GB total, ~8 GB free |
| CPU | 20 cores |
| Disk | 307 GB free |
| Toolchain | Python 3.12.3, Go 1.26.1, Node 20.20.2, Docker 29.1.3, ffmpeg 6.1.1 |

Two consequences, both handled by configuration rather than by architectural compromise:

1. **Qwen2.5-VL-7B will not fit.** The Development Mode default is
   `Qwen2.5-VL-3B-Instruct-AWQ` (4-bit), ~4.4 GB including KV cache at one 448 px image.
   The vision provider is an interface (§10, §16.1 require Qwen/Gemini/GPT/Gemma/LLaVA to be
   interchangeable regardless), so raising this to 7B on 4090 hardware is a config change.
2. **Six datastores will not co-reside in 8 GB of free RAM.** Compose is tiered into profiles
   (`core`, `full`, `ai`, `web`). ElasticSearch and Qdrant are excluded from `core` and arrive
   with Phase 5, when there is something to search.

---

## 3. Architecture

### 3.1 Two independent systems

Two systems that never import each other's code. The only shared artifact is `contracts/`.

```
SentinelAI/
├─ contracts/                              ← the entire boundary between systems
│  ├─ openapi/orchestrator.v1.yaml         generated from FastAPI, committed, drift-checked
│  ├─ proto/sentinel/model/v1/model.proto  hand-authored: the §10 model interface
│  ├─ proto/sentinel/orchestrator/v1/
│  └─ events/*.schema.json                 RabbitMQ payloads, hand-authored JSON Schema
├─ ai-engine/                              ← System 1 (Python). Runs standalone.
├─ web-platform/
│  ├─ backend/                             ← System 2a (Go + Echo)
│  └─ frontend/                            ← System 2b (React 19 + Vite)
├─ deploy/
│  ├─ compose/                             profiles: core | full | ai | web
│  └─ k8s/                                 Phase 9
├─ datasets/                               fetch scripts; content gitignored
└─ docs/superpowers/specs/
```

### 3.2 Contract direction

Pydantic models in FastAPI are the source of truth for the REST control plane. CI exports the
OpenAPI document and **fails the build if the committed YAML has drifted**. The Go client
(`oapi-codegen`) and TypeScript client (`openapi-typescript`) are both generated from that
committed file. Protobuf files are hand-authored and are the source of truth for gRPC. Event
payload schemas are hand-authored JSON Schema; Pydantic models and Go structs are validated
against them in CI.

This yields contract-first guarantees without hand-writing schemas twice.

### 3.3 Enforcing §2 mechanically

In the Go backend, exactly one package — `internal/adapter/aiengine` — may import the
generated orchestrator client. `internal/domain` and `internal/usecase` cannot reference it.
A CI import-lint rule enforces this.

The consequence: no symbol anywhere in the Go backend contains "YOLO", "Qwen", or "Whisper".
The backend receives an `Event` carrying a `description` and a `threat_score`. Which model
produced them is unrepresentable in its type system.

### 3.4 The Development/Production seam

The orchestrator depends on a `ModelRuntime` ABC.

| Mode | Binding | Frame transport |
|---|---|---|
| Development (default) | `adapters/transport/inprocess.py` | Tensors stay in RAM |
| Production | `adapters/transport/grpc_model_client.py` | gRPC stream |

No pipeline, policy, or API code differs between modes. One factory reads one config value.
This is how §3.1 and §3.2 coexist without branching logic.

### 3.5 Transport map

| Hop | Mechanism | Rationale |
|---|---|---|
| React → Go | REST + WebSocket | — |
| Go → Orchestrator | REST / OpenAPI | Low call volume, debuggable, satisfies §35 Swagger |
| Orchestrator → model workers | gRPC (Production Mode only) | Frame payloads, streaming, typed |
| Python → Go | RabbitMQ topic exchange | Durable; survives a backend restart |

The last row is what makes §2 real: killing the Go backend leaves the AI Engine analysing and
buffering. Events queue rather than being lost.

---

## 4. The inference core

Two pure functions do the intellectual work. Neither touches a GPU, which is what makes them
testable and what makes TDD viable on this project.

### 4.1 The escalation gate (§6)

`ai-engine/sentinel_ai/domain/policy/escalation.py`:

```
decide(SceneState, CameraProfile, RateBudget, now) → EscalationDecision
```

It consumes only signals already produced by Stages 2–4. Nothing is recomputed (§7).

| Signal | Source | Marginal cost |
|---|---|---|
| Detections (class, bbox, confidence) | YOLO11 | already paid |
| Tracks (id, age, velocity, trajectory) | ByteTrack | CPU, negligible |
| Motion energy per zone | frame differencing | negligible |
| Scene signature (histogram / pHash) | decode stage | negligible |

**Triggers** — each an independently testable predicate:

| Trigger | Condition | Slice-1 default | Seeds anomaly type |
|---|---|---|---|
| `NewSalientTrack` | relevant-class track persists > `min_track_frames` | 8 frames | unknown visitor |
| `SceneChange` | scene-signature delta > `scene_delta_threshold`, sustained `scene_delta_frames` | 0.35 normalised, 5 frames | obstruction, lighting |
| `DwellExceeded` | track centroid within `dwell_radius_px` for > `dwell_seconds` | 48 px, 30 s | loitering |
| `SpeedAnomaly` | track speed > `speed_percentile` of that camera's history | p95, fixed fallback 180 px/s | running, fall |
| `TrackCountSpike` | concurrent tracks > `track_count_baseline` | 6 | crowding |
| `PeriodicSummary` | heartbeat every `summary_interval_seconds` | 45 s | scene summary |
| `UserRequested` | explicit API call | — | live description |

All values live on `CameraProfile` with these defaults. Slice 1 uses fixed defaults; Phase 4
replaces the percentile and baseline entries with learned per-camera values. `salient_classes`
defaults to `{person, car, truck, bus, motorcycle, bicycle, backpack, handbag, suitcase}`.

**The budget governor.** Triggers alone are insufficient — a busy street fires
`NewSalientTrack` continuously and would saturate the GPU. Every positive decision passes
through:

- a per-camera token bucket (default 1 call / 10 s, burst 2)
- a post-call cooldown
- signature-based deduplication, suppressing calls when the scene is materially unchanged

A camera therefore cannot exceed its VLM budget regardless of scene chaos. The system degrades
to metadata-only, which is precisely §6's intent.

**Output:** `EscalationDecision { should_escalate, reason, priority, context }` where `context`
carries the keyframe reference, tracked objects, behaviour history, previous summaries, and
camera profile required by §22.

### 4.2 Why this is the dissertation's measurable contribution

The gate makes the cost/accuracy tradeoff quantifiable. On the labelled benchmark datasets,
plot *VLM invocations per hour* against *anomaly recall*, versus a naive every-Nth-frame
baseline. This yields a results chapter with real numbers and a defensible claim —
"equivalent recall at a fraction of the inference budget" — rather than a feature inventory.

### 4.3 VRAM residency plan (§34)

`ai-engine/sentinel_ai/domain/policy/vram_budget.py` plans; the orchestrator's `resident_set`
executes.

| Component | VRAM | Residency |
|---|---|---|
| NVDEC decode surfaces | ~0.3 GB | always |
| YOLO11s (TensorRT FP16) | ~0.9 GB | always resident |
| ByteTrack | CPU | — |
| Qwen2.5-VL-3B-AWQ (weights + KV @ 1 image) | ~4.4 GB | resident, idle-unload after 10 min |
| Reserved headroom | ~2.0 GB | — |
| **Total** | **~7.6 GB of 8 GB** | |

The same planner given 24 GB keeps more models resident and raises the VLM to 7B. Residency is
a function of the budget, not a hardcoded layout.

---

## 5. Video sources (§17)

`mediamtx` normalises every input to RTSP, so the AI Engine has exactly one network code path.

```yaml
paths:
  avenue_01:   { source: publisher }                        # file loop, pushed by ffmpeg
  city_square: { source: https://…/index.m3u8 }             # public livestream → RTSP
  lan_cam_01:  { source: rtsp://192.168.1.x:554/stream1 }   # real LAN camera
```

| Tier | Source | Purpose |
|---|---|---|
| Deterministic | Looped video files via mediamtx | CI, regression tests, TDD |
| Labelled | CUHK Avenue, UCSD Ped2, ShanghaiTech, UCF-Crime | Precision/recall figures |
| Uncontrolled | Public HLS webcams | Robustness, soak testing, demos |
| Real | LAN RTSP/ONVIF camera | ONVIF discovery, §20 behaviour baselines |

**Ethical constraint.** Public webcam sources are limited to deliberately-public feeds —
municipal traffic cameras under open-data licences (TfL, Caltrans, NDW) and publicly broadcast
town-square/harbour livestreams. Sites that index unsecured private cameras (e.g. Insecam) are
excluded: those are private feeds exposed without owner consent, and directing a
behaviour-analysis and face-capable platform at them is untenable both ethically and for
institutional ethics approval. This warrants a paragraph in the dissertation.

USB capture remains defined as a `FrameSource` implementation but is deferred to Phase 2 rather
than shipping unused code. Laptop webcam capture is explicitly excluded (battery cost, and it
provides no ground truth).

---

## 6. Component layout

### 6.1 AI Engine

```
ai-engine/sentinel_ai/
├─ domain/                    pure, no I/O
│  ├─ entities.py             Detection, Track, Frame, SceneState, Event, ThreatScore
│  ├─ pipeline.py             PipelineSpec, StageSpec
│  └─ policy/
│     ├─ escalation.py        §6 gate
│     └─ vram_budget.py       §34 residency planner
├─ ports/                     ABCs — the seams
│  ├─ detector.py  tracker.py  vision_llm.py
│  ├─ model_runtime.py        §10: Initialize/Health/Predict/Warmup/Shutdown/Version/Capabilities
│  ├─ frame_source.py  event_publisher.py  clip_writer.py
├─ adapters/
│  ├─ detectors/yolo11.py     trackers/bytetrack.py
│  ├─ vision/qwen25vl_awq.py  vision/hosted.py
│  ├─ sources/rtsp_ffmpeg.py  sources/file.py
│  ├─ publishers/rabbitmq.py  publishers/inmemory.py
│  └─ transport/inprocess.py  transport/grpc_model_client.py
├─ orchestrator/
│  ├─ registry.py             model discovery + §5 lifecycle states
│  ├─ resident_set.py         load/unload against vram_budget
│  ├─ scheduler.py            queue, batching, retry
│  └─ service.py              the single entry point
├─ pipeline/
│  ├─ runner.py               per-camera async task
│  └─ stages/                 decode, detect, track, motion, rules
└─ api/                       FastAPI — thin, delegates to orchestrator
```

`domain/` and `ports/` have no third-party I/O dependencies. This is what allows the entire
policy layer to be unit-tested without CUDA.

### 6.2 Go backend

```
web-platform/backend/
├─ cmd/api/main.go
├─ internal/
│  ├─ domain/                 Camera, Event, User — entities + repository interfaces
│  ├─ usecase/                application services
│  └─ adapter/
│     ├─ http/                Echo handlers, JWT middleware
│     ├─ ws/                  hub, per-camera topics
│     ├─ repo/postgres/  cache/redis/  mq/rabbitmq/
│     └─ aiengine/            generated client — the ONLY package aware of the AI Engine
└─ migrations/
```

### 6.3 Frontend

React 19 + TypeScript + Vite + TailwindCSS + ShadCN UI + React Query + Zustand +
Socket.IO client + Framer Motion. Slice 1 ships three screens only: login, dashboard shell,
single camera live page.

**Visual direction** — inspired by NVIDIA VSS / Metropolis:

- Layered near-black surfaces (not flat `#000`), 1 px borders, subtle grid rules;
  instrument-cluster density rather than airy SaaS spacing
- Tabular monospace numerics for all telemetry (VRAM, FPS, inference ms, threat score) so
  digits do not jitter on update
- Panel composition anchored on the video canvas with live bounding-box overlay, telemetry
  rails surrounding it
- Motion reserved for state transitions, never decoration

**Palette constraint.** NVIDIA's identity centres on a signature green, but SentinelAI's
primary output is a threat score, and in monitoring UIs green already denotes "nominal". Using
green as both brand accent and safe-state colour muddies every escalation ramp and harms
colourblind accessibility. Resolution: green carries brand chrome and nominal state only;
escalation uses a dedicated amber → orange → red ramp; in charts the accent desaturates so data
colours do not compete with UI colours. Exact values are fixed at frontend implementation time.

---

## 7. Slice-1 data flow

```
mediamtx (RTSP)
   │
   ▼  Stage 1  decode (NVDEC) ─────────────► 3 s pre-roll ring buffer
   ▼  Stage 2  YOLO11s detect
   ▼  Stage 3  ByteTrack
   ▼  Stage 4  motion energy + scene signature
   ▼  Stage 5  escalation gate  ── no ──►  persist metadata only, WS telemetry tick
              │
             yes
              ▼
        Orchestrator.describe_scene()
              ▼
        Qwen2.5-VL-3B-AWQ
              ▼
        Event{description, threat_score, suggested_action}
              ├──► clip writer: ring buffer + tail → MinIO
              └──► RabbitMQ  ──►  Go consumer  ──►  Postgres
                                        └──►  WS  ──►  React live page
```

---

## 8. Scope fence

### IN — Phase 0+1

| Area | Delivered |
|---|---|
| Foundation | Monorepo, `contracts/`, codegen + drift check, import-lint, Dockerfiles, Swagger |
| Infra (`core`) | Postgres, Redis, RabbitMQ, MinIO, mediamtx |
| AI domain (TDD) | Entities, ports, escalation policy, VRAM policy |
| Adapters | RTSP/file source, YOLO11s, ByteTrack, Qwen2.5-VL-3B-AWQ, RabbitMQ, in-process transport |
| Orchestrator | Registry, resident set, queue + retry, FastAPI single entry point |
| Pipeline | Stages 1–5 for cameras declared in config |
| Clip writer | 3 s pre-roll ring buffer → MinIO |
| Go backend | JWT, RabbitMQ consumer → Postgres, events REST, WS hub |
| React | Login, dashboard shell, one camera live page |

The clip writer is included despite belonging to §23/§26 because the 3 s pre-roll requires a
ring buffer inside the decode stage. Retrofitting it later means surgery on the hottest code
path; it is cheap now.

### OUT — with destination

| Deferred | Phase |
|---|---|
| Camera CRUD, pipeline editor | 2 |
| Model management dashboard, load balancing, failover | 3 |
| Behaviour learning, the 18 anomaly detectors | 4 |
| ElasticSearch, Qdrant, NL search, timeline/replay UI | 5 |
| Notification channels beyond WebSocket | 6 |
| Daily journal, reports, AI chat | 7 |
| Active learning, retraining pipeline | 8 |
| Kubernetes, multi-GPU, multi-machine | 9 |
| OCR, face recognition, Whisper, pose, segmentation, action recognition | ports defined, adapters later |

Slice 1 ships only the gate's seed heuristics, not the Phase 4 behaviour engine.

---

## 9. Error handling

Governing rule: **an anomaly event is never lost to an infrastructure failure.**

| Failure | Behaviour |
|---|---|
| Camera disconnect | Exponential backoff reconnect; status → offline; event emitted |
| VLM out-of-memory | Evict LRU → retry once → mark unhealthy → pipeline continues detection-only |
| VLM timeout | Event still created from metadata, flagged `description_unavailable` |
| RabbitMQ unavailable | Disk-backed buffer, replayed on reconnect |
| Go backend down | AI Engine unaffected; events queue in RabbitMQ |
| Postgres unavailable | Consumer nacks with requeue; no acknowledgement without a durable write |

---

## 10. Testing strategy

| Layer | Approach | GPU |
|---|---|---|
| Domain policy | TDD, table-driven over synthetic `SceneState` sequences | no |
| Adapters | Contract tests against port ABCs | no |
| Pipeline integration | `FakeDetector` replaying scripted detections, full RTSP → RabbitMQ trace | no |
| Real models | `@pytest.mark.gpu`, local only, skipped in CI | yes |
| Go usecase | Fake repositories | no |
| Go handlers/repos | `httptest`, testcontainers for Postgres | no |
| Frontend | Vitest + React Testing Library + MSW | no |
| End-to-end | Playwright, one happy path | no |

The escalation gate's purity is the keystone: the most important algorithm in the system is
fully testable on CPU in CI.

---

## 11. Decision log

| Decision | Rationale |
|---|---|
| Decompose into 10 phases; spec Phases 0+1 together | 35 sections is not one implementable spec; Phase 0 alone has no observable behaviour |
| Qwen2.5-VL-3B-AWQ as Development Mode default | 7B does not fit 8 GB; provider interface is mandatory per §10/§16.1 regardless |
| Hybrid transport: REST + gRPC + RabbitMQ | Each transport used where strongest; REST satisfies §35 Swagger, RabbitMQ gives §2 independence |
| FastAPI as OpenAPI source of truth, drift-checked | Contract-first guarantees without duplicating schema authorship |
| mediamtx normalises all sources to RTSP | One network code path in the AI Engine |
| Pure-function escalation gate and VRAM planner | Makes the novel algorithms testable without CUDA |
| ElasticSearch/Qdrant excluded from `core` profile | Will not fit in 8 GB free RAM; nothing to search in slice 1 |
| Clip writer in slice 1 | Pre-roll ring buffer is expensive to retrofit into the decode path |
| Green reserved for brand + nominal; separate escalation ramp | Threat semantics would otherwise collide with brand accent |
| Public webcams limited to deliberately-public feeds | Consent and ethics approval |
