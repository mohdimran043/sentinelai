import { useEffect, useRef, useState } from 'react'
import { ENGINE_BASE_URL } from '@/api/config'
import { SseStreamParser, type SseMessage } from '@/recorder/sse'
import type { RecentEventEntry } from '@/api/engineClient'

/**
 * `GET /events/stream` — every camera's anomaly events, pushed as they happen
 * (`ai-engine/sentinel_ai/api/sse.py`). Deliberately the SAME wire shape as the
 * recorder's `GET /api/alerts/stream` (backlog frame, then named frames, then
 * comment keepalives), so this reuses `@/recorder/sse`'s parser rather than
 * writing a second one — see that module's doc comment. The two clients stay
 * separate below: different base URL, different event names, different merge
 * semantics (this stream carries a `sequence` per event and versions events;
 * the recorder's alerts do not).
 *
 * Frames actually sent, per `sse.py`: `backlog` once (a JSON array, oldest
 * first, possibly empty), then `anomaly` (one `RecentEventEntry` object) per
 * live write, and `overflow` (a JSON object, not just a bare event name) if
 * this client fell too far behind — after which the engine ends the stream
 * and reconnecting is the recovery. `: keepalive` comment lines carry nothing
 * and are already dropped by the shared parser.
 */

export type EventStreamStatus = 'connecting' | 'live' | 'reconnecting'

export interface EventStreamHandlers {
  onBacklog: (events: RecentEventEntry[]) => void
  onEvent: (event: RecentEventEntry) => void
  /**
   * Fired when the engine reports this client fell too far behind. Purely
   * informational: the engine closes the stream immediately afterwards, and
   * the generic reconnect-on-drop path below already handles reopening it and
   * receiving a fresh backlog. Nothing here needs to (or should) do its own
   * recovery.
   */
  onOverflow: () => void
  onStatusChange: (status: EventStreamStatus) => void
}

const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 15_000

export interface EventStreamConnection {
  /** Stops the loop for good — no further reconnect attempts. Call on unmount. */
  close: () => void
}

/**
 * Opens the engine's event stream and keeps it open for as long as the caller
 * wants it, reconnecting with backoff whenever the connection drops for any
 * reason: a network error, the engine closing it (shutdown, or an overflow),
 * or a non-2xx response. Every parsed frame is handed to `handlers`; this
 * function does no merging or de-duplication itself — see `mergeEngineEvents`
 * and `useEngineEventStream` below for that.
 */
export function connectEventStream(handlers: EventStreamHandlers): EventStreamConnection {
  let closed = false
  let attempt = 0
  let controller: AbortController | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  function dispatch(message: SseMessage): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(message.data)
    } catch {
      return // A malformed payload is dropped, not a reason to crash the page.
    }
    switch (message.event) {
      case 'backlog':
        if (Array.isArray(parsed)) handlers.onBacklog(parsed as RecentEventEntry[])
        return
      case 'anomaly':
        if (parsed && typeof parsed === 'object') handlers.onEvent(parsed as RecentEventEntry)
        return
      case 'overflow':
        handlers.onOverflow()
        return
      default:
        return // An event name this build does not know about is dropped, not guessed at.
    }
  }

  function scheduleReconnect(): void {
    handlers.onStatusChange('reconnecting')
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** attempt, RECONNECT_MAX_MS)
    attempt += 1
    reconnectTimer = setTimeout(() => void run(), delay)
  }

  async function run(): Promise<void> {
    if (closed) return
    controller = new AbortController()
    handlers.onStatusChange(attempt === 0 ? 'connecting' : 'reconnecting')

    try {
      const response = await fetch(`${ENGINE_BASE_URL}/events/stream`, {
        headers: { Accept: 'text/event-stream' },
        signal: controller.signal,
      })
      if (!response.ok || !response.body) {
        throw new Error(`event stream responded ${response.status}`)
      }

      attempt = 0
      handlers.onStatusChange('live')

      const parser = new SseStreamParser()
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        for (const message of parser.push(decoder.decode(value, { stream: true }))) {
          dispatch(message)
        }
      }
      // The engine closed the stream cleanly — a normal shutdown, or the
      // overflow case above. Either way this is still a disconnect from this
      // console's point of view, so fall through to the reconnect path.
      throw new Error('event stream ended')
    } catch (error) {
      if (closed || (error instanceof DOMException && error.name === 'AbortError')) return
      scheduleReconnect()
    }
  }

  void run()

  return {
    close() {
      closed = true
      if (reconnectTimer !== null) clearTimeout(reconnectTimer)
      controller?.abort()
    },
  }
}

/**
 * Matches the recorder's own `ALERT_STREAM_CAP` convention: a page left open
 * for hours must not grow this list without bound. 500 is arbitrary but
 * generous next to any one camera's 200-entry ring, since this list spans
 * every camera on the site at once.
 */
export const EVENT_STREAM_CAP = 500

/**
 * Merge freshly-arrived events into the current set, honouring the stream's
 * own versioning rule (`sse.py`, `RecentEventEntry.sequence`): the same
 * `event_id` can legitimately arrive more than once — once when it is raised,
 * again later with a higher `sequence` when its clip finishes uploading and
 * `clip_uri` appears. That re-send must UPDATE the entry already held, not add
 * a second row for the same event; a same-or-lower `sequence` for an
 * `event_id` already held is a duplicate (or a stale re-delivery) and is
 * dropped rather than clobbering the newer copy already merged in.
 *
 * One function handles both an incremental live event and the whole `backlog`
 * frame a reconnect re-sends (the backlog is "here is the window again", not
 * "here is what you missed") — the same de-duplication logic is correct for
 * either case because it never assumes anything about the input's order.
 *
 * Sorted newest-first by `occurred_at` for a recent-activity reading, and
 * capped at `EVENT_STREAM_CAP`.
 */
export function mergeEngineEvents(
  current: readonly RecentEventEntry[],
  incoming: readonly RecentEventEntry[],
): RecentEventEntry[] {
  const byId = new Map<string, RecentEventEntry>()
  for (const event of current) byId.set(event.event_id, event)
  for (const event of incoming) {
    const existing = byId.get(event.event_id)
    if (!existing || event.sequence > existing.sequence) {
      byId.set(event.event_id, event)
    }
    // else: existing.sequence >= event.sequence — a duplicate or a stale
    // re-delivery, dropped so it can never displace the newer copy.
  }
  return [...byId.values()]
    .sort((a, b) => b.occurred_at - a.occurred_at)
    .slice(0, EVENT_STREAM_CAP)
}

/**
 * Wires the engine's live event stream into local component state. Unlike the
 * recorder's `useRecorderAlertStream`, there is no bulk REST endpoint for
 * "every event on every camera" to patch into — `GET /cameras/{id}/events` is
 * per camera — so this hook owns its own merged list rather than reaching
 * into a React Query cache entry that does not exist.
 */
export function useEngineEventStream(): {
  events: RecentEventEntry[]
  status: EventStreamStatus
} {
  const [events, setEvents] = useState<RecentEventEntry[]>([])
  const [status, setStatus] = useState<EventStreamStatus>('connecting')
  const eventsRef = useRef<RecentEventEntry[]>([])

  useEffect(() => {
    function merge(incoming: RecentEventEntry[]): void {
      eventsRef.current = mergeEngineEvents(eventsRef.current, incoming)
      setEvents(eventsRef.current)
    }

    const connection = connectEventStream({
      onBacklog: merge,
      onEvent: (event) => merge([event]),
      onOverflow: () => {},
      onStatusChange: setStatus,
    })

    return () => connection.close()
  }, [])

  return { events, status }
}
