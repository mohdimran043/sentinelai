import { RECORDER_BASE_URL } from '@/recorder/config'
import type {
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
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
    // The recorder's 404 body is `{"error": "no such endpoint: GET /api/x"}` —
    // an `error` key, not the engine's `detail`. Read both so neither product's
    // message is swallowed into a bare status code.
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
    throw new RecorderHttpError(response.status, detail)
  }

  return (await response.json()) as T
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
