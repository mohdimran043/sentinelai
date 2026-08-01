# SentinelAI web UI

Slice 1: login, dashboard shell, one camera live page. React 19 + TypeScript + Vite
+ Tailwind v4 + React Query + React Router + Zustand. See
`.superpowers/sdd/ui-slice1-report.md` at the repo root for what's live vs. mocked
vs. deferred, and `.superpowers/sdd/ui-design-brief.md` for the design tokens.

## Commands

```bash
npm install
npm run dev         # http://localhost:5173; proxies /engine/* to :8000 and /recorder/* to :8080
npm run build        # tsc -b && vite build
npm run test          # vitest run
npm run typecheck     # tsc -b
npm run gen           # regenerate src/api/engine.types.ts and src/events/anomalyEvent.types.ts
```

## Two products, two API clients

This console reads **two separate backends** and never conflates them.

| | AI engine | Recorder appliance |
|---|---|---|
| What | `ai-engine/`, FastAPI on `:8000` | `sentinel-ingest`, on `:8080` |
| Client | `src/api/engineClient.ts` | `src/recorder/recorderClient.ts` |
| Types | `src/api/engine.types.ts` (generated from OpenAPI) | `src/recorder/recorder.types.ts` (hand-derived; it publishes no OpenAPI) |
| Errors | `EngineUnreachableError` / `EngineHttpError` | `RecorderUnreachableError` / `RecorderHttpError` |
| Query keys | `['engine', …]` | `['recorder', …]` |
| Routes | `/dashboard`, `/cameras/*` | `/recorder/*` |
| Dev proxy | `/engine/*` → `:8000` | `/recorder/*` → `:8080` |
| Env override | `VITE_ENGINE_API_URL` | `VITE_RECORDER_API_URL` |

Nothing under `src/recorder/` imports from `src/api/`, or the reverse.

## Data

- **Live** — `GET /health`, `GET /cameras`, `GET /cameras/{id}/telemetry`,
  `POST /cameras/{id}/describe` against the AI engine's FastAPI. The engine sets no
  CORS headers, so the dev server proxies `/engine/*` to `http://127.0.0.1:8000`
  (see `vite.config.ts`); set `VITE_ENGINE_API_URL` to point elsewhere.
- **Live** — the recorder's `GET /api/{status,cameras,models,capabilities,alerts,
  notifications,settings,storage/untracked}`. Also no CORS, so `/recorder/*` is
  proxied to `http://127.0.0.1:8080` the same way; set `VITE_RECORDER_API_URL`
  to point elsewhere. **Both proxies are development-only** — a deployment needs
  CORS on each service, a reverse proxy in front of them, or this console served
  from the same origin. Flagged for whoever owns deployment.
- **Mocked (MSW)** — the anomaly event feed and history (`src/events/`), because
  those arrive via RabbitMQ → Go → Postgres and Go doesn't exist yet (Phase 1C).
  The mock worker starts unconditionally unless `VITE_EVENTS_API_URL` is set.
  Swapping to the real Go service later is a change to that one constant.

If the engine is unreachable, the UI says so explicitly — it never renders a
fabricated zero in place of missing data.
