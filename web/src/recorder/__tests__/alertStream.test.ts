import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import { server } from '@/test/mswServer'
import { RECORDER_BASE_URL } from '@/recorder/config'
import { ALERT_STREAM_CAP, connectAlertStream, mergeAlerts } from '@/recorder/alertStream'
import { makeAlert } from '@/recorder/mocks/fixtures'

describe('mergeAlerts', () => {
  it('appends a genuinely new alert', () => {
    const a = makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 1 })
    const b = makeAlert({ cameraId: 'room_4b', eventType: 'OBSTRUCTED', offsetSeconds: 2 })
    expect(mergeAlerts([a], [b]).map((x) => x.id)).toEqual([b.id, a.id])
  })

  it('does not duplicate a row for an alert that already exists — the same id overwrites, not appends', () => {
    const a = makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 1 })
    const merged = mergeAlerts([a], [a])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.id).toBe(a.id)
  })

  it('re-merging the WHOLE backlog on reconnect produces the same set, not a doubled one', () => {
    const backlog = Array.from({ length: 10 }, (_, i) =>
      makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: i }),
    )
    const afterFirstConnect = mergeAlerts([], backlog)
    // Reconnect: the recorder resends the identical backlog, not a delta.
    const afterReconnect = mergeAlerts(afterFirstConnect, backlog)
    expect(afterReconnect).toHaveLength(10)
    expect(afterReconnect.map((a) => a.id).sort()).toEqual(afterFirstConnect.map((a) => a.id).sort())
  })

  it('caps the retained list so a page left open indefinitely cannot grow it without bound', () => {
    const many = Array.from({ length: ALERT_STREAM_CAP + 50 }, (_, i) =>
      makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: i }),
    )
    const merged = mergeAlerts([], many)
    expect(merged).toHaveLength(ALERT_STREAM_CAP)
    // Keeps the NEWEST ones (highest offsetSeconds / at_ns), not an arbitrary slice.
    expect(merged[0]!.id).toBe(many[many.length - 1]!.id)
  })

  it('keeps the merged set sorted newest-first', () => {
    const early = makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 1 })
    const late = makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 99 })
    const merged = mergeAlerts([early], [late])
    expect(merged.map((a) => a.id)).toEqual([late.id, early.id])
  })
})

/** A controllable `text/event-stream` body the test can push chunks into on demand. */
function controllableStream() {
  let controllerRef!: ReadableStreamDefaultController<Uint8Array>
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller
    },
  })
  const encoder = new TextEncoder()
  return {
    stream,
    push(text: string) {
      controllerRef.enqueue(encoder.encode(text))
    },
    close() {
      controllerRef.close()
    },
    error() {
      controllerRef.error(new Error('stream error'))
    },
  }
}

describe('connectAlertStream', () => {
  it('delivers the opening backlog and then a live alert as they arrive on the wire', async () => {
    const first = controllableStream()
    server.use(
      http.get(`${RECORDER_BASE_URL}/alerts/stream`, () => {
        return new HttpResponse(first.stream, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const backlogAlert = makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 1 })
    const liveAlert = makeAlert({ cameraId: 'room_4b', eventType: 'OBSTRUCTED', offsetSeconds: 2 })
    const onBacklog = vi.fn()
    const onAlert = vi.fn()
    const onStatusChange = vi.fn()

    const connection = connectAlertStream({ onBacklog, onAlert, onStatusChange })

    first.push(`event: backlog\ndata: ${JSON.stringify([backlogAlert])}\n\n`)
    await vi.waitFor(() => expect(onBacklog).toHaveBeenCalledWith([backlogAlert]))
    expect(onStatusChange).toHaveBeenCalledWith('live')

    first.push(`data: ${JSON.stringify(liveAlert)}\n\n`)
    await vi.waitFor(() => expect(onAlert).toHaveBeenCalledWith(liveAlert))

    connection.close()
  })

  it('reconnects after the connection drops, and the reconnect is a genuinely new request', async () => {
    vi.useFakeTimers()
    let requestCount = 0
    const first = controllableStream()
    const second = controllableStream()
    server.use(
      http.get(`${RECORDER_BASE_URL}/alerts/stream`, () => {
        requestCount += 1
        const body = requestCount === 1 ? first.stream : second.stream
        return new HttpResponse(body, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const statuses: string[] = []
    const connection = connectAlertStream({
      onBacklog: () => {},
      onAlert: () => {},
      onStatusChange: (status) => statuses.push(status),
    })

    await vi.waitFor(() => expect(requestCount).toBe(1))
    first.push('event: backlog\ndata: []\n\n')
    await vi.waitFor(() => expect(statuses).toContain('live'))

    // The connection drops (recorder closes it, or the network hiccups).
    first.close()
    await vi.waitFor(() => expect(statuses.at(-1)).toBe('reconnecting'))

    // It must not reconnect instantly and hammer the recorder — nothing new
    // yet while the backoff timer is still pending, even once some (but not
    // all) of the backoff delay has elapsed.
    expect(requestCount).toBe(1)
    await vi.advanceTimersByTimeAsync(10)
    expect(requestCount).toBe(1)

    await vi.advanceTimersByTimeAsync(990)
    await vi.waitFor(() => expect(requestCount).toBe(2))

    second.push('event: backlog\ndata: []\n\n')
    await vi.waitFor(() => expect(statuses.at(-1)).toBe('live'))

    connection.close()
    vi.useRealTimers()
  })

  it('stops reconnecting once closed, so an unmounted page cannot leak a background retry loop or call back into it', async () => {
    vi.useFakeTimers()
    let requestCount = 0
    const first = controllableStream()
    server.use(
      http.get(`${RECORDER_BASE_URL}/alerts/stream`, () => {
        requestCount += 1
        return new HttpResponse(first.stream, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const statuses: string[] = []
    const connection = connectAlertStream({
      onBacklog: () => {},
      onAlert: () => {},
      onStatusChange: (status) => statuses.push(status),
    })
    await vi.waitFor(() => expect(requestCount).toBe(1))
    const statusesAtClose = statuses.length

    // The drop and the deliberate close race each other, the way an operator
    // navigating away right as the connection hiccups would in real use.
    first.close()
    connection.close()

    await vi.advanceTimersByTimeAsync(30_000)
    // No new network request — the retry loop is not still running in the background.
    expect(requestCount).toBe(1)
    // And no further callback into a component that has already unmounted —
    // an extra late `onStatusChange('reconnecting')` would be a setState
    // after unmount in the real hook, not just a wasted timer.
    expect(statuses.length).toBe(statusesAtClose)
    vi.useRealTimers()
  })
})
