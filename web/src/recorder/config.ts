/**
 * Base URL for the **recorder** API (`sentinel-ingest`, the appliance that
 * serves http://127.0.0.1:8080).
 *
 * This is a DIFFERENT PRODUCT from the AI engine. The engine's base URL lives
 * in `src/api/config.ts` and its client in `src/api/engineClient.ts`; nothing
 * in `src/recorder/` may import from `src/api/`, or vice versa. Two products,
 * two clients, two type sets.
 *
 * The recorder sends no CORS headers for our dev origin (verified: it answers
 * `curl` fine but a browser at :5173 would be blocked), so — exactly as slice 1
 * already does for the engine — the Vite dev server proxies `/recorder/*` to it
 * (see `vite.config.ts`). That makes the request same-origin as far as the
 * browser is concerned.
 *
 * DEPLOYMENT: the dev proxy is a development convenience, not a fix. A real
 * deployment needs one of:
 *   - CORS on the recorder for this console's origin, or
 *   - a reverse proxy in front of both, or
 *   - this console served from the recorder itself (its own `index.html` shows
 *     it already supports being pointed at another host via
 *     `window.__SENTINEL_API__` / `VITE_SENTINEL_API`).
 * Set `VITE_RECORDER_API_URL` to the reachable base in that case.
 */
export const RECORDER_BASE_URL: string = import.meta.env.VITE_RECORDER_API_URL ?? '/recorder/api'
