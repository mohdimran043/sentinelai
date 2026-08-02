import { RECORDER_BASE_URL } from '@/recorder/config'
import type {
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
  RecorderCoverageReport,
  RecorderJournalDayResponse,
  RecorderJournalResponse,
  RecorderMaskRegion,
  RecorderMaskResponse,
  RecorderMaskValidation,
  RecorderModelsResponse,
  RecorderNotificationsResponse,
  RecorderSettingsResponse,
  RecorderStatusResponse,
  RecorderUntrackedResponse,
} from '@/recorder/recorder.types'

/**
 * Typed client for the recorder appliance (`sentinel-ingest`, :8080).
 *
 * Deliberately a mirror-image of `src/api/engineClient.ts` rather than a shared
 * abstraction over it: the two are different products with different failure
 * vocabularies, and one console being able to say "the recorder is down but the
 * AI engine is fine" is the whole point of keeping them apart. Nothing here
 * imports from `src/api/`.
 */

/** The recorder did not answer at all — network failure, DNS, connection refused. */
export class RecorderUnreachableError extends Error {
  constructor(cause: unknown) {
    super('The recorder is unreachable.')
    this.name = 'RecorderUnreachableError'
    this.cause = cause
  }
}

/** The recorder answered with a non-2xx status. */
export class RecorderHttpError extends Error {
  readonly status: number
  constructor(status: number, detail: string) {
    super(`The recorder returned ${status}: ${detail}`)
    this.name = 'RecorderHttpError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${RECORDER_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (cause) {
    throw new RecorderUnreachableError(cause)
  }

  if (!response.ok) {
    throw await toHttpError(response)
  }

  return (await response.json()) as T
}

/**
 * The recorder's 404 body is `{"error": "no such endpoint: GET /api/x"}` — an
 * `error` key, not the engine's `detail`. Read both so neither product's
 * message is swallowed into a bare status code. Shared by `request` (which
 * always expects a JSON body back) and `postCameraAction` (which does not).
 */
async function toHttpError(response: Response): Promise<RecorderHttpError> {
  let detail = response.statusText
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object') {
      const record = body as Record<string, unknown>
      const message = record.error ?? record.detail
      if (message !== undefined) detail = String(message)
    }
  } catch {
    // Body wasn't JSON — keep statusText.
  }
  return new RecorderHttpError(response.status, detail)
}

export function getRecorderStatus(): Promise<RecorderStatusResponse> {
  return request<RecorderStatusResponse>('/status')
}

export function listRecorderCameras(): Promise<RecorderCamerasResponse> {
  return request<RecorderCamerasResponse>('/cameras')
}

/**
 * The whole rolling alert window in one response (500 alerts / ~426 KB on the
 * live instance).
 *
 * VERIFIED 2026-08-01: the endpoint accepts no query parameters. `?limit`,
 * `?offset`, `?camera_id`, `?event_type` and `?since_ns` were each probed and
 * every one returned all 500 rows unchanged, and there is no
 * `GET /api/alerts/{id}`. Filtering, ordering and paging are therefore the
 * client's job — see `AlertsPage`. Do not "optimise" this into a server query
 * without re-probing first.
 */
export function getRecorderAlerts(): Promise<RecorderAlertsResponse> {
  return request<RecorderAlertsResponse>('/alerts')
}

export function getRecorderModels(): Promise<RecorderModelsResponse> {
  return request<RecorderModelsResponse>('/models')
}

export function getRecorderCapabilities(): Promise<RecorderCapabilitiesResponse> {
  return request<RecorderCapabilitiesResponse>('/capabilities')
}

export function getRecorderNotifications(): Promise<RecorderNotificationsResponse> {
  return request<RecorderNotificationsResponse>('/notifications')
}

export function getRecorderSettings(): Promise<RecorderSettingsResponse> {
  return request<RecorderSettingsResponse>('/settings')
}

export function getRecorderUntrackedStorage(): Promise<RecorderUntrackedResponse> {
  return request<RecorderUntrackedResponse>('/storage/untracked')
}

/** `GET /api/journal/{camera}` — the day-by-day seal/revision index for one camera. */
export function getRecorderJournal(cameraId: string): Promise<RecorderJournalResponse> {
  return request<RecorderJournalResponse>(`/journal/${encodeURIComponent(cameraId)}`)
}

/**
 * `GET /api/journal/{camera}/{date}`, optionally `?history=1`.
 *
 * VERIFIED 2026-08-01: a date with no written chapter answers 200 with
 * `{camera_id, date, detail, sealed: false}` and no `revision` key — never a
 * 404. See `RecorderJournalDayResponse`.
 */
export function getRecorderJournalDay(
  cameraId: string,
  date: string,
  options: { history?: boolean } = {},
): Promise<RecorderJournalDayResponse> {
  const query = options.history ? '?history=1' : ''
  return request<RecorderJournalDayResponse>(
    `/journal/${encodeURIComponent(cameraId)}/${encodeURIComponent(date)}${query}`,
  )
}

/**
 * `GET /api/report?camera=&date=` — the live coverage/integrity/segments
 * document for one camera-day. VERIFIED: never 404s for a well-formed camera
 * id and date, even an unknown camera or a date with nothing recorded — it
 * answers 200 with zeroed coverage instead. Missing `camera` or `date`
 * answers 400. See `RecorderCoverageReport` for why this shares a type with
 * the journal's embedded `revision.report`.
 */
export function getRecorderReport(cameraId: string, date: string): Promise<RecorderCoverageReport> {
  const params = new URLSearchParams({ camera: cameraId, date })
  return request<RecorderCoverageReport>(`/report?${params.toString()}`)
}

/**
 * `GET /api/snapshot/{camera_id}?t=...` — a live JPEG frame (verified: 200,
 * `image/jpeg`, ~5.6 KB). Not fetched through `request`: this is consumed
 * directly as an `<img src>`, never parsed as JSON. `cacheBust` should change
 * on every render an operator expects a fresher frame — `CamerasPage` uses
 * `GET /api/status`'s own `server_time_ns`, so the thumbnail turns over on the
 * same cadence as the worker-status poll rather than running a second timer.
 */
export function recorderSnapshotUrl(cameraId: string, cacheBust: number): string {
  return `${RECORDER_BASE_URL}/snapshot/${encodeURIComponent(cameraId)}?t=${cacheBust}`
}

/**
 * `POST /api/cameras/{camera_id}/start` and `/stop` — worker lifecycle
 * control. Probing this live (unlike every GET above) would itself stop or
 * start a real running recorder worker, so its response body was never
 * captured and is deliberately not parsed or trusted: a non-2xx throws exactly
 * as every other recorder call does (via `toHttpError`), and a 2xx resolves
 * with nothing. The caller (`useRecorderCameraAction`) refetches
 * `GET /api/status` to learn what actually happened.
 */
async function postCameraAction(cameraId: string, action: 'start' | 'stop'): Promise<void> {
  let response: Response
  try {
    response = await fetch(
      `${RECORDER_BASE_URL}/cameras/${encodeURIComponent(cameraId)}/${action}`,
      { method: 'POST', headers: { Accept: 'application/json' } },
    )
  } catch (cause) {
    throw new RecorderUnreachableError(cause)
  }

  if (!response.ok) {
    throw await toHttpError(response)
  }
}

export function startRecorderCamera(cameraId: string): Promise<void> {
  return postCameraAction(cameraId, 'start')
}

export function stopRecorderCamera(cameraId: string): Promise<void> {
  return postCameraAction(cameraId, 'stop')
}

/** `GET /api/masks/{camera_id}` — the mask editor's whole state for one camera. */
export function getRecorderMask(cameraId: string): Promise<RecorderMaskResponse> {
  return request<RecorderMaskResponse>(`/masks/${encodeURIComponent(cameraId)}`)
}

/**
 * `GET /api/masks/{camera_id}/calibration-frame` — a real, unmasked frame to
 * draw against. Not fetched through `request`: like `recorderSnapshotUrl`,
 * this is consumed directly as an `<img src>`.
 *
 * VERIFIED 2026-08-01: this 404s with `{"error": "no such endpoint: GET
 * ..."}` whenever the owning camera's `GET /api/masks/{camera_id}` reports
 * `calibration_available: false` — which is every camera on the reference
 * instance, all of them recording. That 404 IS the refusal already carried in
 * `calibration_refusal`, not a second failure to explain; callers must gate on
 * `calibration_available` and never request this URL (nor fall back to
 * `recorderSnapshotUrl`) when it is false. See `MasksPage.tsx`.
 */
export function recorderCalibrationFrameUrl(cameraId: string): string {
  return `${RECORDER_BASE_URL}/masks/${encodeURIComponent(cameraId)}/calibration-frame`
}

/**
 * Shared by `validateRecorderMask` (always a dry run) and `saveRecorderMask`
 * (persists when valid). Both answer with the identical `RecorderMaskValidation`
 * shape, but VERIFIED 2026-08-01 they disagree on how a rejection is reported:
 * `POST .../validate` always answers HTTP 200, valid or not — the body's
 * `valid` flag is the only signal. `PUT /api/masks/{camera_id}` answers HTTP
 * 200 when it saves and HTTP 422 when it refuses to (confirmed live: a
 * rejected `PUT` left the camera's on-disk regions byte-for-byte unchanged).
 * Both status codes therefore carry the same well-formed body and are handled
 * identically here; anything else (404 unknown camera, 400 unparsable JSON,
 * 5xx) is a genuine transport failure and throws exactly as `request` does.
 */
async function submitMaskRegions(
  path: string,
  method: 'POST' | 'PUT',
  regions: RecorderMaskRegion[],
): Promise<RecorderMaskValidation> {
  let response: Response
  try {
    response = await fetch(`${RECORDER_BASE_URL}${path}`, {
      method,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions }),
    })
  } catch (cause) {
    throw new RecorderUnreachableError(cause)
  }

  if (response.status === 200 || response.status === 422) {
    return (await response.json()) as RecorderMaskValidation
  }

  throw await toHttpError(response)
}

/** `POST /api/masks/{camera_id}/validate` — checks a candidate region set against the recorder's own polygon rules. Never persists. */
export function validateRecorderMask(
  cameraId: string,
  regions: RecorderMaskRegion[],
): Promise<RecorderMaskValidation> {
  return submitMaskRegions(`/masks/${encodeURIComponent(cameraId)}/validate`, 'POST', regions)
}

/** `PUT /api/masks/{camera_id}` — validates, and persists only if the result is valid. */
export function saveRecorderMask(
  cameraId: string,
  regions: RecorderMaskRegion[],
): Promise<RecorderMaskValidation> {
  return submitMaskRegions(`/masks/${encodeURIComponent(cameraId)}`, 'PUT', regions)
}
