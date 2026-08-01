import type { SentinelAIAnomalyEvent } from '@/events/anomalyEvent.types'

/**
 * The one module boundary for anomaly events. Today every request is intercepted
 * by MSW (see `mocks/handlers.ts`) because events arrive via RabbitMQ -> Go ->
 * Postgres, and Go is Phase 1C (spec §7-8). When that service exists, swapping to
 * it is a change to `EVENTS_BASE_URL` — nothing that calls `fetchEventFeed` or
 * `fetchEventHistory` needs to change.
 */
export const EVENTS_MOCK_PATH = '/mock/events'
export const EVENTS_BASE_URL: string = import.meta.env.VITE_EVENTS_API_URL ?? EVENTS_MOCK_PATH

export class EventServiceUnreachableError extends Error {
  constructor(cause: unknown) {
    super('The event service is unreachable.')
    this.name = 'EventServiceUnreachableError'
    this.cause = cause
  }
}

interface EventsEnvelope {
  events: SentinelAIAnomalyEvent[]
}

async function request(path: string): Promise<SentinelAIAnomalyEvent[]> {
  let response: Response
  try {
    response = await fetch(`${EVENTS_BASE_URL}${path}`, {
      headers: { Accept: 'application/json' },
    })
  } catch (cause) {
    throw new EventServiceUnreachableError(cause)
  }
  if (!response.ok) {
    throw new EventServiceUnreachableError(new Error(`HTTP ${response.status}`))
  }
  const body = (await response.json()) as EventsEnvelope
  return body.events
}

/** Recent events for one camera's live feed. */
export function fetchEventFeed(cameraId: string): Promise<SentinelAIAnomalyEvent[]> {
  return request(`/cameras/${encodeURIComponent(cameraId)}/events`)
}

/** Same underlying data, used for the ribbon/history view rather than the live
 * feed. Kept as a distinct named export so a future Go backend can serve history
 * from a different, paginated endpoint without the feed call changing shape. */
export function fetchEventHistory(cameraId: string): Promise<SentinelAIAnomalyEvent[]> {
  return request(`/cameras/${encodeURIComponent(cameraId)}/events`)
}
