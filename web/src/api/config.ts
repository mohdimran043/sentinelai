/**
 * Base URL for the AI engine's FastAPI.
 *
 * The engine (`ai-engine/sentinel_ai/api/app.py`) sets no CORS headers, so a
 * browser calling it cross-origin from the Vite dev server would be blocked.
 * Rather than touch `ai-engine/`, the dev server proxies `/engine/*` to it
 * (see `vite.config.ts`), which makes the request same-origin as far as the
 * browser is concerned. In a real deployment, set `VITE_ENGINE_API_URL` to
 * wherever the engine is reachable (which will need CORS enabled there, or a
 * reverse proxy doing the same job this dev proxy does).
 */
export const ENGINE_BASE_URL: string = import.meta.env.VITE_ENGINE_API_URL ?? '/engine'
