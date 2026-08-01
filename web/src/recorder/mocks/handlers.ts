import { http, HttpResponse } from 'msw'
import { RECORDER_BASE_URL } from '@/recorder/config'
import {
  REAL_ALERTS,
  REAL_MODELS,
  REAL_SETTINGS,
} from '@/recorder/mocks/fixtures'
import type {
  RecorderAlertsResponse,
  RecorderModelsResponse,
  RecorderSettingsResponse,
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
]
