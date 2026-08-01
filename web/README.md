# SentinelAI web UI

Slice 1: login, dashboard shell, one camera live page. React 19 + TypeScript + Vite
+ Tailwind v4 + React Query + React Router + Zustand. See
`.superpowers/sdd/ui-slice1-report.md` at the repo root for what's live vs. mocked
vs. deferred, and `.superpowers/sdd/ui-design-brief.md` for the design tokens.

## Commands

```bash
npm install
npm run dev         # http://localhost:5173, proxies /engine/* to the AI engine on :8000
npm run build        # tsc -b && vite build
npm run test          # vitest run
npm run typecheck     # tsc -b
npm run gen           # regenerate src/api/engine.types.ts and src/events/anomalyEvent.types.ts
```

## Data

- **Live** — `GET /health`, `GET /cameras`, `GET /cameras/{id}/telemetry`,
  `POST /cameras/{id}/describe` against the AI engine's FastAPI. The engine sets no
  CORS headers, so the dev server proxies `/engine/*` to `http://127.0.0.1:8000`
  (see `vite.config.ts`); set `VITE_ENGINE_API_URL` to point elsewhere.
- **Mocked (MSW)** — the anomaly event feed and history (`src/events/`), because
  those arrive via RabbitMQ → Go → Postgres and Go doesn't exist yet (Phase 1C).
  The mock worker starts unconditionally unless `VITE_EVENTS_API_URL` is set.
  Swapping to the real Go service later is a change to that one constant.

If the engine is unreachable, the UI says so explicitly — it never renders a
fabricated zero in place of missing data.
