import { ENGINE_BASE_URL } from '@/api/config'
import type { components } from '@/api/engine.types'

export type HealthResponse = components['schemas']['HealthResponse']
export type CamerasResponse = components['schemas']['CamerasResponse']
export type CameraStatus = components['schemas']['CameraStatus']
export type DescribeResponse = components['schemas']['DescribeResponse']
export type CameraEventsResponse = components['schemas']['CameraEventsResponse']
export type RecentEventEntry = components['schemas']['RecentEventEntry']
export type LatestDescriptionState = CameraEventsResponse['latest_description_state']

/** The engine did not answer at all — network failure, DNS, connection refused. */
export class EngineUnreachableError extends Error {
  constructor(cause: unknown) {
    super('The AI engine is unreachable.')
    this.name = 'EngineUnreachableError'
    this.cause = cause
  }
}

/** The engine answered with a non-2xx status. */
export class EngineHttpError extends Error {
  readonly status: number
  constructor(status: number, detail: string) {
    super(`AI engine returned ${status}: ${detail}`)
    this.name = 'EngineHttpError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${ENGINE_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (cause) {
    throw new EngineUnreachableError(cause)
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body: unknown = await response.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        detail = String((body as { detail: unknown }).detail)
      }
    } catch {
      // Body wasn't JSON — keep statusText.
    }
    throw new EngineHttpError(response.status, detail)
  }

  return (await response.json()) as T
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function listCameras(): Promise<CamerasResponse> {
  return request<CamerasResponse>('/cameras')
}

export function getCameraTelemetry(cameraId: string): Promise<CameraStatus> {
  return request<CameraStatus>(`/cameras/${encodeURIComponent(cameraId)}/telemetry`)
}

export function describeCameraNow(cameraId: string): Promise<DescribeResponse> {
  return request<DescribeResponse>(`/cameras/${encodeURIComponent(cameraId)}/describe`, {
    method: 'POST',
  })
}

/**
 * A volatile, bounded, in-memory ring of this camera's recent events (capped at
 * `capacity`, currently 200 — see `CameraEventsResponse`'s own description). Not
 * the event store: an absent event here means "not in the last `capacity` events
 * of this process run", never "did not happen". `latest` is `events.at(-1)` read
 * from the same snapshot, so the live panel and the chart cannot disagree.
 */
export function getCameraEvents(cameraId: string): Promise<CameraEventsResponse> {
  return request<CameraEventsResponse>(`/cameras/${encodeURIComponent(cameraId)}/events`)
}
