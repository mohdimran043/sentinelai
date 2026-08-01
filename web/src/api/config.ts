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

/**
 * Base URL mediamtx serves HLS from.
 *
 * Video delivery is mediamtx's job, not the AI engine's — spec §5's whole
 * point (`ai-engine/cameras.example.json`'s own comment: mediamtx normalises
 * every camera to one protocol). So unlike `ENGINE_BASE_URL` above, this is
 * never proxied through the engine or through this console's own server: the
 * browser fetches HLS from mediamtx directly, over its own HTTP port (8888 by
 * default). That only works because mediamtx already answers with
 * `Access-Control-Allow-Origin: *` on every HLS route — verified directly
 * (`curl -i http://localhost:8888/demo_live/index.m3u8`) — so, unlike
 * `/engine` and `/recorder` in `vite.config.ts`, no dev proxy is needed here
 * at all.
 *
 * Defaults to mediamtx's default HLS port on localhost, which is where it
 * runs in this project's own dev compose file. Configurable via
 * `VITE_MEDIAMTX_BASE_URL` because mediamtx will not be on localhost in a
 * real deployment.
 */
export const MEDIAMTX_BASE_URL: string = import.meta.env.VITE_MEDIAMTX_BASE_URL ?? 'http://localhost:8888'
