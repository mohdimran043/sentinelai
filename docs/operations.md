# Operations

## The compose stack

`deploy/compose/docker-compose.core.yml`, all under the `core` profile. Image
tags are pinned, not `latest`, so `make up` is reproducible.

```bash
make up      # docker compose --profile core up -d
make logs
make down
```

| Service | Image | Ports | Used by |
|---|---|---|---|
| `rabbitmq` | `rabbitmq:3.13-management` | 5672 (amqp), 15672 (UI) | **The engine.** Event publishing. |
| `minio` | `minio/minio:RELEASE.2024-10-13…` | 9000 (S3), 9001 (console) | **The engine.** Evidence clips. |
| `minio-init` | `minio/mc` | — | One-shot: creates the `sentinel-clips` bucket, then exits 0. |
| `mediamtx` | `bluenviron/mediamtx:1.9.3` | 8554 (RTSP), 1935, 8888, 8889 | Video-source normaliser for the demo. |
| `postgres` | `postgres:16` | 5432 | **Nothing yet.** Scaffolded for the Phase 1C Go backend. |
| `redis` | `redis:7` | 6379 | **Nothing yet.** Same. |

The engine talks only to `rabbitmq` and `minio`. Postgres and Redis share this
one compose file rather than forking a second one for Phase 1C.

`mediamtx` has no Docker healthcheck: the official image is built `FROM scratch`
— no shell, no curl — so no in-container probe can run. Reachability is
asserted from outside over the RTSP port by the integration smoke test.

> **Credentials are development-only.** `sentinel/sentinel`,
> `sentinel/sentinel123`, Redis with no password at all, and every port
> published on `0.0.0.0`. Do not put this compose file on a network you do not
> control.

## Running the engine

```bash
make install-runtime            # CPU: pipeline + fakes, no models
make install-gpu                # adds torch, ultralytics, transformers, bitsandbytes

cd ai-engine
cp cameras.example.json cameras.json     # then edit it
uvicorn sentinel_ai.main:app --host 0.0.0.0 --port 8000
```

`sentinel_ai/main.py` is the composition root and the real entry point — not
`api.app:app`, which only exposes `create_app(service)` so tests never build a
real `EngineService`.

Importing `main` reads no camera file, opens no socket and touches no GPU.
Composition happens inside the FastAPI lifespan, on the running loop, because
`RtspSource.__init__` calls `asyncio.create_task` and so cannot run before there
is a loop. That is what lets CI import the module with neither the `gpu` extra
nor a `cameras.json` present.

### Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | Per-model `state`, `detail`, `vram_mib` |
| `GET /cameras` | `CameraStatus` for every camera, plus `config_writable` — whether the PATCH below will do anything on this deployment |
| `GET /cameras/{id}/telemetry` | One camera's counters, its `label` and its `zone`. 404 if unknown |
| `PATCH /cameras/{id}` | **Write.** Edits `label` and `zone`, persisted to `cameras.json`. **Off by default** — 403 unless `SENTINEL_ENABLE_CAMERA_WRITES=true`. See [Editing cameras from the console](#editing-cameras-from-the-console) |
| `POST /cameras/{id}/describe` | Forces a `user_requested` escalation, returns `event_id` |
| `GET /cameras/{id}/events` | A bounded ring of that camera's recent events, capped at 200, plus `latest` and `latest_description_state` (`none`/`available`/`unavailable`). Volatile — this is the console's view, not the event store. RabbitMQ plus the Go consumer is the durable record. 404 if unknown |
| `GET /events/stream` | `text/event-stream`. Opens with `event: backlog` carrying a JSON array, then streams live events. A sequence watermark makes the backlog-to-live handover gapless and duplicate-free; the same `event_id` at a higher sequence is a legitimate new version, not a repeat — that is how a clip URI back-fills onto an event a client already displayed. The ring is bounded, so a long disconnect genuinely loses history |

Telemetry counters: `frames_seen`, `frames_dropped`, `detections_run`,
`escalations`, `escalations_dropped`, `discontinuities`, `last_frame_at`,
`last_escalation_at`.

Two things to know when reading `/health`:

- **`status` is the hardcoded literal `"ok"`**, regardless of model state.
  Derive health from the per-model `state` field, never from `status`.
- The VLM sitting at `UNLOADED` is usually **correct** — it is the 600 s idle
  unload having fired, not a fault. A camera watching an empty corridor is
  supposed to look like that.

### Editing cameras from the console

`PATCH /cameras/{id}` is the engine's only endpoint that changes anything
durable. It is **disabled by default** and answers 403 until a deployment sets:

```bash
SENTINEL_ENABLE_CAMERA_WRITES=true
```

> **Read this before you set it.** The engine has **no authentication** — see
> [Known deployment gaps §2](#2-no-authentication). Every other endpoint is a
> read, so an exposed port has so far cost you information. This one is a write,
> and a persisted one: anything that can open a TCP connection to `:8000` can
> rename a camera to another camera's name and move it into another wing, and
> the change is written to `cameras.json` and survives the restart that would
> otherwise undo it. In a custodial setting that is the console's account of
> *where an incident happened*, altered anonymously and permanently.
>
> Enable it only where you have already decided that everything which can reach
> the port is permitted to reconfigure cameras — which in practice means binding
> uvicorn to `127.0.0.1`, or putting an authenticating reverse proxy in front of
> it. The engine logs a warning at startup when the flag is on. Revisit this in
> Phase 1C, when JWT arrives and the flag can default to on behind real auth.

What the endpoint will and will not change:

| Field | Editable at runtime? | Why |
|---|---|---|
| `label` | **Yes** — applied live, written to `cameras.json` | Metadata. Nothing in the pipeline branches on it |
| `zone` | **Yes** — same. `null` ungroups; omitting the field leaves the grouping alone | Metadata. Changes how a console groups a camera, not how the engine watches one |
| `notify_on`, `notify_min_confidence`, `clip_preroll_seconds`, `clip_postroll_seconds`, `summary_interval_seconds` | **Yes** — live from that camera's **next escalation**, written to `cameras.json` and **survive a restart** | The per-camera welfare policy, and the sharpest edge on this endpoint. `notify_on: []` is the mute switch: an anonymous caller can stop a camera notifying anyone about anything, from the next escalation onward, while it keeps detecting, keeps recording clips and keeps publishing events — so a muted camera looks exactly like a quiet one. The startup warning and the audit line naming the old and the new policy on every edit are the only places it shows |
| `url` | **No — restart required** | Changing the source means tearing down the running `CameraRunner`, its pre-roll buffer and any clip mid-recording, and building a new source. Separately, an RTSP URL routinely carries credentials, so an unauthenticated API neither accepts nor returns it |
| `profile` | **No — restart required** | It is the escalation policy the gate is part-way through applying (cooldowns, a token bucket with live state) |

A body carrying `url` or `profile` is a **422 naming the field**, never a 200
that quietly dropped it. Edit `cameras.json` and restart for those two.

The write is atomic and never half-applied. The edited document is re-parsed in
full before anything is written, so a request that would produce a file the
engine could not load at its next startup is refused outright; the new document
is then written to a `.tmp` sibling, `fsync`ed, and renamed into place. Comment
keys, other cameras, profiles and any field this version has no model of are
preserved — the file is edited, not regenerated. The in-memory camera is updated
only *after* the file is on disk, and from what the file now says, so the two
cannot disagree about an edit that succeeded.

If `cameras.json` has been edited by hand since the engine started and no longer
holds the camera being edited, the answer is **409** and nothing is written.
Reconcile the file and restart rather than letting the console overwrite it.

In the console, this is the **Camera record** panel on a camera's own page. When
`config_writable` is false the panel still shows the stored record, read-only,
and names the environment variable — hiding it would leave a camera's label and
zone with nowhere in the console they can be read. `url` and `profile` are listed
as restart-required with no value, because no endpoint returns one.

## Running the console

```bash
cd web && npm install && npm run dev     # http://localhost:5173
```

`npm run build` · `npm run test` · `npm run typecheck` · `npm run lint` ·
`npm run gen` (regenerate types from `contracts/`).

The console reads **two separate backends** and never conflates them: the AI
engine on `:8000` and the recorder appliance (`sentinel-ingest`, a different
product) on `:8080`. The anomaly event feed is **mocked with MSW**, because
those events arrive via RabbitMQ → Go → Postgres and Go does not exist yet.
The mock worker starts unless `VITE_EVENTS_API_URL` is set. Details in
[web/README.md](../web/README.md).

## The end-to-end demo

Both levels below have actually been run on a real GPU box against real
services — not merely documented.

### Automated proxy (one command)

`ai-engine/tests/e2e/test_demo_smoke.py`, marked `gpu` + `integration`. Runs the
real detector/tracker/VLM over the downloaded Avenue clip through
`FileSource(realtime=False)` — no mediamtx, no RabbitMQ, no MinIO. Passes in
about 23 s, producing one real `Event` with a non-empty description and a threat
score in [0, 1].

```bash
bash datasets/download_sample.sh
cd ai-engine && .venv/bin/python -m pytest tests/e2e/test_demo_smoke.py -v
```

### Full stack

```bash
# 1. Fetch the clip (once). ~780 MiB archive, deleted after extraction.
bash datasets/download_sample.sh

# 2. Core services.
docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d

# 3. Loop the clip into mediamtx as the avenue_01 RTSP path.
ffmpeg -stream_loop -1 -re -i datasets/avenue/avenue_01.mp4 -t 200 \
  -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/avenue_01

# 4. Run the pipeline against it.
cd ai-engine
cat > cameras.json <<'EOF'
{"cameras": [{"id": "avenue_01", "label": "Avenue (demo)",
              "url": "rtsp://localhost:8554/avenue_01"}]}
EOF
uvicorn sentinel_ai.main:app --host 127.0.0.1 --port 8000
```

**`-rtsp_transport tcp` is not optional.** The compose file publishes only
mediamtx's TCP port (8554), not its RTP/RTCP UDP ports (8000/8001). ffmpeg's
default RTSP-publish transport is UDP, so media packets silently never reach the
container and mediamtx kills the "publishing" session as idle after its 10 s
read timeout — even though the RTSP handshake itself succeeds. This failure mode
is silent and looks like a working connection.

### Seeing the output

The `sentinel.events` exchange is a **topic exchange with no queue bound by
default**. To see events you must bind a consumer to `anomaly.#` *before* the
engine publishes; otherwise messages are routed nowhere and silently discarded.

- RabbitMQ: `http://localhost:15672` (`sentinel`/`sentinel`)
- MinIO: `http://localhost:9001`, or `mc ls local/sentinel-clips/avenue_01/`

A representative clean 120 s run: `frames_seen=2988`, `frames_dropped=56`,
`detections_run=2932`, `escalations=5`, `escalations_dropped=0`,
`discontinuities=0`. Five events published (`track_count_spike`,
`speed_anomaly` ×2, `dwell_exceeded` ×2), threat scores 0.10–0.20, five clips in
MinIO. One clip verified with `ffprobe` at 8.36 s / 209 frames / 5 keyframes,
decoding cleanly under `ffmpeg -f null -`, with real pre-roll ahead of the
trigger frame.

## Tests and CI

```bash
make check      # lint + typecheck + test
make test       # pytest -m "not gpu"
make test-gpu   # pytest -m gpu
```

CI (`.github/workflows/ci.yml`) runs on `main` and `master`: `ruff check`,
`ruff format --check`, `mypy` (strict, over `sentinel_ai` **and** `tests`), then
`pytest -m "not gpu and not integration"` with `--cov-fail-under=85`.

Two deliberate choices worth not undoing:

- **`-W error` is not a CI flag.** `filterwarnings = ["error"]` lives in
  `pyproject.toml`, so it applies to every invocation — CI's, a developer's, an
  IDE's. A flag is only enforced by whoever remembers to type it. The `httpx2`
  dependency and the `supervision>=0.24,<0.28` pin exist *solely* because a
  warning is fatal under it; as a command-line flag, either could have been
  silently undone by a passing build.
- **The OpenAPI drift check is a test, not a workflow step.** CI already runs
  the suite, and it fails on the developer's machine the moment they change a
  route rather than twenty minutes later on a push.

Web CI (`.github/workflows/web-ci.yml`) is a separate workflow so a broken
`npm install` can never block the Python job. It runs `npm ci`, then checks the
generated API types, then `typecheck`, `lint`, `test` and the production build.

The generated-types check is the one place the bullet above is deliberately
inverted — it *is* a workflow step rather than a test. `web/src/api/engine.types.ts`
is generated from `contracts/openapi/ai-engine.yaml` by `npm run gen:api` and
committed, and verifying it means regenerating the file and asking git whether
anything moved; that is a working-tree question, not one a unit test should be
answering. It runs before `typecheck` on purpose, so a contract someone forgot
to regenerate against fails as "you forgot to regenerate" rather than as a
confusing type error several steps later. This is not hypothetical: the
camera-editing contract landed with the types 197 lines behind it, and `tsc`
stayed green throughout — a stale generated file is internally consistent, it
simply describes a schema the engine no longer serves.

## Known deployment gaps

These are real, known, and **currently unowned**. Do not discover them in
production.

### 1. The engine sends no CORS headers

`ai-engine/sentinel_ai/api/app.py` adds no CORS middleware — verified, there is
no `CORSMiddleware` anywhere in the engine source. A browser calling it
cross-origin is blocked.

The console works around this with a **Vite dev-server proxy** (`/engine/*` →
`http://127.0.0.1:8000`), which makes the request same-origin as far as the
browser is concerned. **That proxy exists only under `npm run dev`.** A built
bundle served by anything else has no proxy and no CORS, and will fail.

The same applies to the recorder appliance: `/recorder/*` →
`http://127.0.0.1:8080`, also dev-only, also no CORS.

A deployment needs one of:

- CORS middleware on the engine (and on the recorder) for the console's origin;
- a reverse proxy in front of both, doing what the dev proxy does;
- the console served from the same origin as the API.

Set `VITE_ENGINE_API_URL` / `VITE_RECORDER_API_URL` to the reachable base in
that case. **Nobody owns this decision yet.** It is the single largest gap
between "works on a laptop" and "deployed".

### 2. No authentication

`web/src/store/session.ts` holds an operator display name in `sessionStorage`
and is explicitly **not** an authentication boundary; `RequireSession.tsx` is a
routing convenience, not a security guard. JWT is Phase 1C's Go backend. The
engine's API has no auth of any kind.

This is why `PATCH /cameras/{id}` — the engine's only write endpoint — ships
**disabled**, rather than being treated as one more route. Reaching an
unauthenticated read endpoint costs disclosure; reaching an unauthenticated
*write* endpoint costs control, and the two are not the same size. Gating it
behind `SENTINEL_ENABLE_CAMERA_WRITES` does not make the port safe — nothing
short of authentication does — but it does mean a deployment that has not
thought about this is not silently reconfigurable by whoever finds it, and that
turning it on is a decision someone made rather than a default they inherited.
See [Editing cameras from the console](#editing-cameras-from-the-console).

### 3. The dead-letter spool has no operator story

`SENTINEL_DEAD_LETTER_DIR` accumulates events that could not be published and
that replay cannot fix. Nothing replays it, nothing surfaces its depth on
`/health` or in telemetry, and nothing alerts on it. **Check it manually.** A
directory that silently accumulates evidence is only half a guarantee.

### 4. VRAM measurement undershoots real peaks

`capabilities().vram_mib` is measured during `warmup()` against a small dummy
frame. On a real 810×1080 frame the VLM's true peak was observed 938 MiB above
the warmup figure, and `plan_residency()`'s admission control works off the
warmup number. `SENTINEL_VRAM_RESERVED_MIB` (2048) absorbs this today. Do not
tune it toward zero.

Related: `plan_residency()` has **no notion of externally consumed VRAM** — it
assumes it owns the whole card. On a headless server that is correct. On a
desktop it survives only because the reserve is larger than the compositor's
usage, which is a coincidence, not a guarantee.

### 5. `RtspSource.close()` can take up to ~20 s

PyAV/ffmpeg blocking calls cannot be interrupted from another thread, so
shutdown is bounded by the socket timeout (`stimeout=20000000`) rather than
being immediate. Expect a slow stop on a camera whose stream has gone away.
