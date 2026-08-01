import { http, HttpResponse } from 'msw'
import { RECORDER_BASE_URL } from '@/recorder/config'
import {
  REAL_ALERTS,
  REAL_CAMERAS,
  REAL_CAPABILITIES,
  REAL_MODELS,
  REAL_SETTINGS,
  REAL_STATUS,
  REAL_STORAGE,
} from '@/recorder/mocks/fixtures'
import type {
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
  RecorderModelsResponse,
  RecorderSettingsResponse,
  RecorderStatusResponse,
  RecorderUntrackedResponse,
} from '@/recorder/recorder.types'

/**
 * MSW handlers for the recorder API — **tests only**.
 *
 * The running app talks to the real appliance through the Vite dev proxy; this
 * exists so the test suite can pin a payload and so the request actually goes
 * through `recorderClient` (URL building, error mapping, JSON parsing) instead
 * of being stubbed out at the module boundary.
 */

export const recorderAlertsHandler = (body: RecorderAlertsResponse) =>
  http.get(`${RECORDER_BASE_URL}/alerts`, () => HttpResponse.json(body))

export const recorderModelsHandler = (body: RecorderModelsResponse) =>
  http.get(`${RECORDER_BASE_URL}/models`, () => HttpResponse.json(body))

export const recorderSettingsHandler = (body: RecorderSettingsResponse) =>
  http.get(`${RECORDER_BASE_URL}/settings`, () => HttpResponse.json(body))

export const recorderStorageHandler = (body: RecorderUntrackedResponse) =>
  http.get(`${RECORDER_BASE_URL}/storage/untracked`, () => HttpResponse.json(body))

export const recorderCapabilitiesHandler = (body: RecorderCapabilitiesResponse) =>
  http.get(`${RECORDER_BASE_URL}/capabilities`, () => HttpResponse.json(body))

export const recorderCamerasHandler = (body: RecorderCamerasResponse) =>
  http.get(`${RECORDER_BASE_URL}/cameras`, () => HttpResponse.json(body))

export const recorderStatusHandler = (body: RecorderStatusResponse) =>
  http.get(`${RECORDER_BASE_URL}/status`, () => HttpResponse.json(body))

/**
 * A tiny valid JPEG (a 1x1 black pixel), so `CamerasPage` tests never touch
 * the network for a thumbnail. Every camera's snapshot resolves to this same
 * byte string unless a test overrides the route with `server.use(...)`.
 */
const TINY_JPEG_BASE64 =
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k='

function tinyJpegBytes(): Uint8Array {
  const binary = atob(TINY_JPEG_BASE64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

/** `GET /api/snapshot/:cameraId` — succeeds for any camera id by default. */
export const recorderSnapshotHandler = () =>
  http.get(`${RECORDER_BASE_URL}/snapshot/:cameraId`, () =>
    HttpResponse.arrayBuffer(tinyJpegBytes().buffer as ArrayBuffer, {
      headers: { 'Content-Type': 'image/jpeg' },
    }),
  )

/** Make one specific camera's snapshot fail, the way an unregistered camera id does live (404, JSON body). */
export const recorderSnapshotFailureHandler = (cameraId: string) =>
  http.get(`${RECORDER_BASE_URL}/snapshot/${cameraId}`, () =>
    HttpResponse.json({ error: `no such camera: ${cameraId}` }, { status: 404 }),
  )

/**
 * `POST /api/cameras/:cameraId/start|stop`. `onRequest` fires once per
 * matched call — tests use it to prove the request was actually sent (not
 * just that a click handler ran), exactly as `recorderClient.test.ts` does
 * for the GETs.
 */
export const recorderCameraActionHandler = (
  action: 'start' | 'stop',
  options: { status?: number; error?: string; onRequest?: (cameraId: string) => void } = {},
) =>
  http.post(`${RECORDER_BASE_URL}/cameras/:cameraId/${action}`, ({ params }) => {
    const cameraId = String(params.cameraId)
    options.onRequest?.(cameraId)
    if (options.error !== undefined) {
      return HttpResponse.json({ error: options.error }, { status: options.status ?? 500 })
    }
    return HttpResponse.json({ ok: true }, { status: options.status ?? 200 })
  })

/** Make an endpoint fail the way the recorder does: a JSON body with an `error` key. */
export const recorderErrorHandler = (path: string, status: number, error: string) =>
  http.get(`${RECORDER_BASE_URL}${path}`, () => HttpResponse.json({ error }, { status }))

/** Make an endpoint unreachable — no response at all, as if nothing were listening. */
export const recorderUnreachableHandler = (path: string) =>
  http.get(`${RECORDER_BASE_URL}${path}`, () => HttpResponse.error())

/** Defaults: the real payloads. Tests override per case with `server.use(...)`. */
export const recorderHandlers = [
  recorderAlertsHandler(REAL_ALERTS),
  recorderModelsHandler(REAL_MODELS),
  recorderSettingsHandler(REAL_SETTINGS),
  recorderStorageHandler(REAL_STORAGE),
  recorderCapabilitiesHandler(REAL_CAPABILITIES),
  recorderCamerasHandler(REAL_CAMERAS),
  recorderStatusHandler(REAL_STATUS),
  recorderSnapshotHandler(),
]
