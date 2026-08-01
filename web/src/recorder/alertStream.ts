import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { RECORDER_BASE_URL } from '@/recorder/config'
import { SseStreamParser, type SseMessage } from '@/recorder/sse'
import type { RecorderAlert, RecorderAlertsResponse } from '@/recorder/recorder.types'

/** The same React Query key `useRecorderAlerts` (in `queries.ts`) uses — deliberately duplicated here rather than imported, so this file never has to import from the contested `queries.ts` module. */
const ALERTS_QUERY_KEY = ['recorder', 'alerts'] as const

/**
 * `GET /api/alerts/stream` — verified live 2026-08-01: `text/event-stream`,
 * opens with `event: backlog` carrying a JSON array (~83 KB on the live
 * instance), then a `: ping` comment (a keepalive, carries nothing) at least
 * every ~25s. A live, individually-streamed alert was never observed in the
 * probing window, so the event name it arrives under is NOT verified — this
 * deliberately does not assume `event: alert`. Any event other than
 * `backlog` is treated as carrying either one alert (an object) or several (an
 * array), whichever the body actually is; the default unnamed `message` event
 * is handled identically. Re-verify the event name against a real one before
 * relying on it elsewhere.
 */

/**
 * Matches the recorder's own rolling-window cap on `GET /api/alerts` (500 on
 * the live instance — see `RecorderAlertsResponse`), so a page left open for
 * hours holds no more "the window" than the recorder itself considers current.
 */
export const ALERT_STREAM_CAP = 500

/**
 * Merge freshly-arrived alerts into the current list, by `id`.
 *
 * One function handles two situations identically: an incremental new alert
 * arriving live, and the recorder re-sending the WHOLE backlog after a
 * reconnect (its `backlog` event is not "since you last saw", it is "here is
 * the window again"). An alert already present is overwritten by its own
 * newest copy rather than duplicated, the merged set is re-sorted newest
 * first, and capped at `ALERT_STREAM_CAP` so the list cannot grow without
 * bound for as long as the page stays open.
 */
export function mergeAlerts(
  current: readonly RecorderAlert[],
  incoming: readonly RecorderAlert[],
): RecorderAlert[] {
  const byId = new Map<string, RecorderAlert>()
  for (const alert of current) byId.set(alert.id, alert)
  for (const alert of incoming) byId.set(alert.id, alert)
  return [...byId.values()].sort((a, b) => b.at_ns - a.at_ns).slice(0, ALERT_STREAM_CAP)
}

export type AlertStreamStatus = 'connecting' | 'live' | 'reconnecting'

export interface AlertStreamHandlers {
  onBacklog: (alerts: RecorderAlert[]) => void
  onAlert: (alert: RecorderAlert) => void
  onStatusChange: (status: AlertStreamStatus) => void
}

/** Delay before the first reconnect attempt; doubles on each further drop, capped. */
const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 15_000

export interface AlertStreamConnection {
  /** Stops the loop for good — no further reconnect attempts. Call on unmount. */
  close: () => void
}

/**
 * Opens the alert stream and keeps it open for as long as the caller wants it,
 * reconnecting with backoff whenever the connection drops for any reason: a
 * network error, the recorder closing it, or a non-2xx response. Every parsed
 * message is handed to `handlers`; this function does no merging or state
 * itself — see `mergeAlerts` and `useRecorderAlertStream` below for that.
 */
export function connectAlertStream(handlers: AlertStreamHandlers): AlertStreamConnection {
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
    if (message.event === 'backlog') {
      if (Array.isArray(parsed)) handlers.onBacklog(parsed as RecorderAlert[])
      return
    }
    if (Array.isArray(parsed)) {
      for (const alert of parsed) handlers.onAlert(alert as RecorderAlert)
    } else if (parsed && typeof parsed === 'object') {
      handlers.onAlert(parsed as RecorderAlert)
    }
  }

  /**
   * The only caller (`run`'s catch block, below) already checks `closed`
   * before calling this, so it is not re-checked here — a second guard that
   * can never actually trigger is worse than no guard: it reads as a safety
   * net that isn't one.
   */
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
      const response = await fetch(`${RECORDER_BASE_URL}/alerts/stream`, {
        headers: { Accept: 'text/event-stream' },
        signal: controller.signal,
      })
      if (!response.ok || !response.body) {
        throw new Error(`alert stream responded ${response.status}`)
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
      // The recorder closed the stream cleanly. Still a disconnect from this
      // console's point of view — fall through to the reconnect path.
      throw new Error('alert stream ended')
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
 * Wires the live alert stream into the SAME React Query cache entry
 * `useRecorderAlerts` (in `queries.ts`) populates from `GET /api/alerts` —
 * this hook never fetches or owns a REST response itself, it only patches the
 * `alerts` array already there. That is deliberate: `deliveries`, `dropped`
 * and `note` only exist on the wire from the REST response, so if no cached
 * response exists yet this hook has nothing truthful to merge into and skips
 * the update rather than fabricating one. Everything the page renders keeps
 * coming from `useRecorderAlerts()`; this only keeps its `alerts` fresher than
 * the 15s poll would on its own.
 *
 * A small buffer of everything received so far is kept independently of the
 * cache (`bufferedAlerts`), so that the rare race where the stream's backlog
 * arrives before the REST GET resolves self-heals on the very next stream
 * event instead of silently losing that first backlog forever.
 */
export function useRecorderAlertStream(): AlertStreamStatus {
  const queryClient = useQueryClient()
  const [status, setStatus] = useState<AlertStreamStatus>('connecting')
  const bufferedAlerts = useRef<RecorderAlert[]>([])

  useEffect(() => {
    function merge(incoming: RecorderAlert[]): void {
      bufferedAlerts.current = mergeAlerts(bufferedAlerts.current, incoming)
      queryClient.setQueryData<RecorderAlertsResponse>(ALERTS_QUERY_KEY, (old) => {
        if (!old) return old
        return { ...old, alerts: mergeAlerts(old.alerts, bufferedAlerts.current) }
      })
    }

    const connection = connectAlertStream({
      onBacklog: merge,
      onAlert: (alert) => merge([alert]),
      onStatusChange: setStatus,
    })

    return () => connection.close()
  }, [queryClient])

  return status
}
