import { RECORDER_BASE_URL } from '@/recorder/config'
import type {
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
  RecorderCoverageReport,
  RecorderJournalDayResponse,
  RecorderJournalResponse,
  RecorderModelsResponse,
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

/**
 * The recorder did not answer at all — network failure, DNS, connection refused,
 * or a gateway in front of it reporting that nothing answered on its behalf.
 *
 * The message names the address, because the commonest cause by far is that the
 * recorder simply is not running: it is a **separate product** from the AI engine
 * (`sentinel-ingest`, not in this repository), and a console pointed at a dev proxy
 * with nothing behind it produced a bare "502 Bad Gateway" that told an operator
 * neither which of the two systems was down nor where it was being looked for.
 */
export class RecorderUnreachableError extends Error {
  constructor(cause: unknown) {
    super(
      `The recorder appliance is not reachable at ${RECORDER_BASE_URL}. ` +
        'It is a separate service from the AI engine — the rest of this console ' +
        'is unaffected.',
    )
    this.name = 'RecorderUnreachableError'
    this.cause = cause
  }
}

/**
 * Statuses only a gateway sends: the upstream was absent or timed out, so the request
 * never reached the recorder and the body is the proxy's own.
 *
 * **503 is deliberately not here.** A proxy sends it, but so does a healthy recorder
 * reporting that one of its subsystems is down — "model registry unavailable" — and
 * swallowing that into "not reachable" would discard the more specific truth. 503 is
 * resolved by looking at the body instead; see `toHttpError`.
 */
const GATEWAY_STATUSES: ReadonlySet<number> = new Set([502, 504])

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
async function toHttpError(response: Response): Promise<Error> {
  if (GATEWAY_STATUSES.has(response.status)) {
    return new RecorderUnreachableError(new Error(`gateway status ${response.status}`))
  }

  let detail: string | undefined
  try {
    const body: unknown = await response.json()
    if (body && typeof body === 'object') {
      const record = body as Record<string, unknown>
      const message = record.error ?? record.detail
      if (message !== undefined) detail = String(message)
    }
  } catch {
    // Body wasn't JSON. Left undefined so the 503 branch below can tell "the recorder
    // explained itself" from "something returned a bare status".
  }

  // A 503 with no message in it is a gateway's, not the recorder's: a recorder that is
  // refusing service says why. Without this, a console pointed at a dev proxy with
  // nothing behind it reported "Service Unavailable", which names neither the failing
  // system nor where it was looked for.
  if (detail === undefined && response.status === 503) {
    return new RecorderUnreachableError(new Error('503 with no recorder message'))
  }

  return new RecorderHttpError(response.status, detail ?? response.statusText)
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
