# AGENTS.md

Working notes for coding agents in this repository. Read [README.md](README.md) for what
SentinelAI *is*; this file is about how to change it without breaking the things that are
enforced mechanically.

**Snapshot: 2026-09-17.** The ledger below was verified against the working tree on that
date by running the suites, not by reading the docs. Re-verify before trusting a count —
and see §6, because this tree moves under you.

---

## 1. The one thing to understand first

SentinelAI watches camera streams, runs cheap signals continuously (decode → detect →
track → motion), and spends GPU on a vision-language model **only when an escalation gate
says the scene warrants it**. The gate is the product, not an optimisation.

Two consequences that shape almost every change:

- **Behaviour detectors never call a VLM.** They run on signals the gate already computes
  and *raise* a reason the gate then rules on. The gate stays the single place that
  decides whether GPU gets spent.
- **The engine's judgements are opinions, not verdicts.** Welfare concerns and fall
  detection are a model's reading of a scene. `Confidence` has no `certain`. Wording stays
  hedged ("possible collapse"). See ADR 10 and ADR 12.

---

## 2. Rules that will fail CI if you break them

These are not style preferences. Each one is a test or a workflow step.

| Rule | Enforced by |
|---|---|
| `domain/` and `ports/` are **pure** — no torch, numpy, cv2, av, fastapi, minio, httpx, and no `time`, `datetime`, `random`, `os`, `importlib` | `ai-engine/tests/test_architecture.py` |
| Pure layers never import `adapters`, `orchestrator`, `pipeline`, `api`, or `config` | same |
| Pure layers never read a clock — matched by call name, so aliasing cannot hide it | same |
| `web/` and `ai-engine/` never import each other; `contracts/` is the only seam | `.github/workflows/boundary-check.yml` → `scripts/check_boundaries.py` |
| `contracts/openapi/ai-engine.yaml` matches the live FastAPI app | `ai-engine/tests/test_openapi_contract.py` |
| `web/src/api/engine.types.ts` is regenerated from the contract | Web CI step — `npm run gen:api` must leave no diff |
| Every outbound event validates against `contracts/events/anomaly_event.schema.json` | `adapters/serialization/event_codec.py`, on every publish |
| **Warnings are errors** in the Python suite | `filterwarnings = ["error"]` in `pyproject.toml` — deliberately not a CLI flag, so it cannot be undone by forgetting to type it |
| mypy `strict` over both `sentinel_ai` **and** `tests` | `make typecheck` |
| Engine coverage ≥ 85% | `.github/workflows/ci.yml` |

`scripts/check_boundaries.py --self-test` plants a violation and proves the checker catches
it before it checks the real tree. Prefer that pattern: **a guard nobody has seen fail is a
hypothesis, not a guard.**

---

## 3. Commands

### Engine (from repo root)

```bash
make install-runtime   # CPU deps — enough for the whole test suite
make install-gpu       # adds torch, ultralytics, transformers, bitsandbytes
make lint              # ruff check + ruff format --check
make typecheck         # mypy strict
make test              # pytest -m "not gpu"
make check             # lint + typecheck + test
make up / down / logs  # RabbitMQ, MinIO, mediamtx, Postgres, Redis
```

The whole CPU suite runs in about five seconds on any machine, with no GPU. That is the
point of the layering, not an accident — keep it that way.

### Console (from `web/`)

```bash
npm ci
npm run gen        # regenerate API types from contracts/ — run after any contract change
npm run typecheck  # tsc -b
npm run lint       # oxlint
npm test           # vitest run
npm run build      # tsc -b && vite build
```

> **Node version trap.** The console needs **Node ≥ 20.12**. Where the default `node` is
> older, vitest and vite die at startup with
> `SyntaxError: The requested module 'node:util' does not provide an export named 'styleText'`,
> which looks like a broken dependency and is not. CI pins Node 20. To work around it
> locally: `export PATH="$HOME/.nvm/versions/node/v24.19.0/bin:$PATH"` (any ≥ 20.12 will do).

### Verified on 2026-09-17

| Check | Result |
|---|---|
| Engine, CPU (`-m "not gpu and not integration and not network"`) | **1353 passed**, 21 deselected, 5.6 s |
| Console (`npx vitest run`) | **447 passed**, 41 files, 4.1 s |
| `npm run typecheck` | clean |
| `npm run build` | succeeds |
| `scripts/check_boundaries.py` (incl. self-test) | clean |

The README's test counts are older than these; trust a run over a document.

---

## 4. Repository map

```
ai-engine/          The Python inference engine. The product's core.
  sentinel_ai/
    domain/         Pure entities and policy. No I/O, no clock, no outer imports.
      behaviour/    Temporal state machines — fall, abandonment, tamper, zones.
      policy/       Pure rules — escalation, alerting, authorization, priority, budgets.
    ports/          Pure abstract interfaces — the seams.
    adapters/       Everything touching the world: YOLO, Qwen, InsightFace, PyAV,
                    MinIO, RabbitMQ, EarthCam, the encrypted face store.
    orchestrator/   Model registry, VRAM resident set, GPU admission, VLM scheduler,
                    alert register, durable triage persistence.
    pipeline/       One asyncio task per camera: the frame loop.
    api/            FastAPI routes over the orchestrator.
    benchmark/      Measurement. Imports the engine; nothing imports it.
    validation/     Accuracy scoring against URFall / LFW / FairFace.
    config.py       All 56 SENTINEL_* settings. The only Dev/Prod seam.
    main.py         Composition root — the one place the real system is built.
  tests/            Incl. architecture fitness functions and port contracts.

contracts/          The ONLY coupling between engine and UI.
  openapi/          ai-engine.yaml, generated from the app, drift-tested in CI.
  events/           anomaly_event.schema.json, validated on every publish.

web/                React 19 + Vite operator console. Types generated from contracts/.
deploy/compose/     Postgres, Redis, RabbitMQ, MinIO, mediamtx.
datasets/           Sample-clip fetch script + URFall clips. Footage is never committed.
docs/               Product documentation. decisions.md is the one to read first.
  superpowers/      Process working area (specs, plans) — not product documentation.
scripts/            check_boundaries.py.
```

---

## 5. Implementation ledger

### Engine — working

- **Pipeline**: decode → detect (YOLO11s) → track (ByteTrack) → motion → escalation gate
  → VLM (Qwen2.5-VL, 4-bit NF4) → clip → publish. Verified end to end on a real GPU box.
- **Per-camera AI capabilities**, load-bearing: the composition root builds only the models
  some camera actually asks for, so a triggers-only site never places a VLM. Capabilities:
  `scene_description`, `anomaly_detection`, `fall_detection`, `abandoned_object`,
  `camera_tamper`, `person_authorization`, `zone_monitoring`. Model keys: `detector`,
  `vlm`, `face`, `pose`.
- **Behaviour detectors** — pure state machines, distances in body heights not pixels:
  abandoned object, camera tamper, zone intrusion, line crossing. Loitering was already the
  `dwell_exceeded` trigger and was tuned rather than duplicated.
- **Camera tamper detection** is free — it reads the luma histogram the motion stage
  already computes, so it needs no model and can run on every camera.
- **Fall detection** — a temporal state machine, then vision-language confirmation,
  published with its own `basis: temporal_pose_vlm`. Opt-in, and measured (see below).
- **Pose estimation** (YOLO11n-pose), loaded only where `fall_detection` is enabled.
- **Person authorization** — InsightFace SCRFD + ArcFace under ONNX Runtime. 512-d
  embeddings sealed individually with AES-256-GCM, **no face images stored**, no embeddings
  or scores logged. The engine refuses to start without `SENTINEL_FACE_ENCRYPTION_KEY`.
  Off by default. Unauthorized is a low-severity finding, not an alarm.
- **Resident welfare notifications** — collapse, altercation, self-harm, apparent
  medication intake, distress. Qualifying concerns reach a human by log line or webhook,
  with the clip's URL. Per-camera policy, editable at runtime.
- **Short notification clips, with retention** — a notification at or above `high` carries
  a 3 s cut of the pre-roll rather than the full evidence clip, and clips expire after a
  day via a bucket lifecycle rule the object store enforces. Both configurable; `0`
  disables either.
- **Alert engine** — an alert is an *episode*, keyed by camera + reason + subject, so one
  person seen seventeen times in twenty seconds is one alert with `occurrences: 17`.
- **Durable triage state** — acknowledgements and resolutions flush to disk before the API
  answers, and survive a `kill -9`. Occurrence counts flush on a 5 s timer.
- **VRAM residency planner**, LRU + priority eviction, 600 s VLM idle unload.
- **RabbitMQ publishing** with disk spool, replay and dead-letter.
- **MinIO evidence clips** — remuxed, never re-encoded; 3 s pre-roll + 5 s post-roll,
  overridable per camera.
- **Sources**: RTSP (with reconnect backoff), file, and public **EarthCam pages** — you
  configure the page URL, not an `.m3u8`; the resolver re-resolves on every reconnect
  because the playlist URL is signed and expires. Signatures never reach a log.
- **Hardware decode** (`SENTINEL_DECODE_HWACCEL`) — works, and **off by default because it
  is measured slower end to end**. A device FFmpeg cannot open is a startup error, never a
  silent fallback.
- **Benchmark harness** (`python -m sentinel_ai.benchmark`) and **validation harness**
  (`python -m sentinel_ai.validation.{falls,faces}`). Every performance and accuracy figure
  in the docs comes from these.

### API surface — working

`/health` · `/cameras` (GET, POST) · `/cameras/probe` · `/cameras/{id}` (PATCH, DELETE) ·
`/cameras/{id}/telemetry` · `/cameras/{id}/events` · `/cameras/{id}/snapshot` ·
`/cameras/{id}/describe` · `/events/stream` (SSE) · `/alerts` (GET, DELETE) ·
`/alerts/{id}/acknowledge` · `/alerts/{id}/resolve` · `/authorized-persons` (+ `/{id}`,
`/{id}/faces`, `/{id}/faces/{face_id}`, `/{id}/faces/{face_id}/image`).

`POST /cameras` persists a camera and starts watching with no restart; `DELETE` stops and
removes it.

**Writes are off by default.** The engine has no authentication, so camera writes ship
disabled behind `SENTINEL_ENABLE_CAMERA_WRITES`. `label`, `zone`, `capabilities` and the
welfare policy are editable live and persisted to `cameras.json`; `url` and `profile`
require a restart and are **rejected rather than ignored** — a console should render them
read-only and say why, not offer a control that does nothing.

### Console — working

Command centre, alert queue, people, site map, add-a-camera (with a real decoded preview
before saving), AI system, camera page, and six recorder sections: day report, configure,
footage, notifications, storage, masks.

All twelve rail links sit under a single **AI engine** heading, and a test puts every link
the rail renders through the real route table in `App.tsx` — a link with no route behind it
does not 404 here, it quietly redirects to the dashboard, so it needs a test rather than an
eye. The event feed is still MSW-mocked.

### Not built

| Area | State |
|---|---|
| Go backend / Postgres / Redis (Phase 1C) | Scaffolded in compose only |
| WebSocket event hub | The UI mocks the event feed with MSW |
| Production gRPC transport (`SENTINEL_MODE=production`) | The setting exists; nothing branches on it |
| CORS on the engine | Missing and unowned — see docs/operations.md |

### Measured, and honest about it

Do not describe these as better than they are; the docs deliberately do not.

- **Fall detection on URFall:** saw the signature in 29 of 30 falls and **raised none of
  them**, while 23 of 40 ordinary daily activities produced the same signature.
- **Welfare monitoring:** it asks a vision-language model about single frames. It is not a
  fall detector, it misses things, and a stretcher carry was missed entirely in measurement.
- **Person authorization on LFW's 2200-pair protocol:** 92.7% recall, zero false matches,
  96.4% accuracy. Even-handedness partly measured on FairFace and **not uniform**.

---

## 6. State of the working tree

**Most of the shipped feature set is uncommitted.** At this snapshot the tree carried ~58
untracked paths and ~82 modified files against `main`. Everything in the
situational-awareness plan — capabilities, behaviour detectors, fall detection, person
authorization, the alert engine, the benchmark and validation harnesses, EarthCam, hardware
decode, notification clips, and the newer console pages — exists **only in the working
tree**.

Two things follow:

1. `git log` is not a record of what is implemented. This file and the suites are.
2. **The tree is live.** Engine sources, engine tests and `README.md` were all edited by
   another session while this snapshot was being taken — which is why the engine test count
   moved mid-session. Re-read a file before editing it, and do not assume it is as you last
   saw it.

---

## 7. Before you change something

| You are… | Read |
|---|---|
| Doing anything non-trivial | [docs/decisions.md](docs/decisions.md) — 18 ADRs, the decisions a newcomer would otherwise re-litigate or undo |
| Learning the layering | [docs/architecture.md](docs/architecture.md) |
| Adding a source, detector, VLM, publisher, trigger or console page | [docs/extending.md](docs/extending.md) |
| Tuning it | [docs/configuration.md](docs/configuration.md) |
| Running or demoing it | [docs/operations.md](docs/operations.md) |
| Quoting a number | [docs/performance.md](docs/performance.md) — every figure measured on a named box, with a section on what has *not* been measured |
| Touching the engine ↔ UI boundary | [contracts/README.md](contracts/README.md) |

### House style

- **Immutability.** Return new objects; never mutate in place.
- **Small files.** 200–400 lines typical, 800 max. Organise by feature, not by type.
- Functions under ~50 lines; early returns rather than nesting past four levels.
- Handle errors explicitly. Never swallow one silently.
- Validate at system boundaries. Never trust external data.
- Comments explain **why**, at the density the surrounding code already uses. This codebase
  documents decisions and their rejected alternatives heavily — match that, and when a
  change makes a comment untrue, update the comment rather than leaving it to rot.
- Never hardcode secrets. `SENTINEL_FACE_ENCRYPTION_KEY` and broker credentials come from
  the environment.
