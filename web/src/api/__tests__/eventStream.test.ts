import { http, HttpResponse } from 'msw'
import { describe, expect, it, vi } from 'vitest'
import { server } from '@/test/mswServer'
import { ENGINE_BASE_URL } from '@/api/config'
import { EVENT_STREAM_CAP, connectEventStream, mergeEngineEvents } from '@/api/eventStream'
import type { RecentEventEntry } from '@/api/engineClient'

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
    camera_id: 'room_4b',
    sequence: 1,
    clip_uri: null,
    occurred_at: 1_700_000_000,
    source_timestamp: null,
    reason: 'new_salient_track',
    threat_score: 0.2,
    severity: 'low',
    description: 'A person walks across the frame.',
    suggested_action: 'Continue monitoring.',
    description_unavailable: false,
    labels: ['person'],
    track_ids: [1],
    welfare_concerns: [],
    ...overrides,
  }
}

describe('mergeEngineEvents', () => {
  it('appends a genuinely new event (different event_id)', () => {
    const a = makeEvent({ event_id: 'a', occurred_at: 1 })
    const b = makeEvent({ event_id: 'b', occurred_at: 2 })
    expect(mergeEngineEvents([a], [b]).map((e) => e.event_id).sort()).toEqual(['a', 'b'])
  })

  it('a re-send at the SAME sequence is a duplicate: the row is not doubled', () => {
    const a = makeEvent({ event_id: 'a', sequence: 3 })
    const merged = mergeEngineEvents([a], [a])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.event_id).toBe('a')
  })

  it('a re-send at a HIGHER sequence for the same event_id updates the row in place rather than adding a second one', () => {
    const original = makeEvent({ event_id: 'a', sequence: 1, clip_uri: null })
    const backfilled = makeEvent({ event_id: 'a', sequence: 2, clip_uri: 's3://clips/a.mp4' })
    const merged = mergeEngineEvents([original], [backfilled])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.clip_uri).toBe('s3://clips/a.mp4')
    expect(merged[0]!.sequence).toBe(2)
  })

  it('a stale re-delivery at a LOWER sequence than what is already held is dropped, not applied', () => {
    const newer = makeEvent({ event_id: 'a', sequence: 5, clip_uri: 's3://clips/a.mp4' })
    const stale = makeEvent({ event_id: 'a', sequence: 2, clip_uri: null })
    const merged = mergeEngineEvents([newer], [stale])
    expect(merged).toHaveLength(1)
    expect(merged[0]!.sequence).toBe(5)
    expect(merged[0]!.clip_uri).toBe('s3://clips/a.mp4')
  })

  it('re-merging the WHOLE backlog on reconnect produces the same set, not a doubled one', () => {
    const backlog = Array.from({ length: 10 }, (_, i) =>
      makeEvent({ event_id: `e${i}`, occurred_at: i }),
    )
    const afterFirstConnect = mergeEngineEvents([], backlog)
    const afterReconnect = mergeEngineEvents(afterFirstConnect, backlog)
    expect(afterReconnect).toHaveLength(10)
    expect(afterReconnect.map((e) => e.event_id).sort()).toEqual(
      afterFirstConnect.map((e) => e.event_id).sort(),
    )
  })

  it('caps the retained list so a page left open indefinitely cannot grow it without bound', () => {
    const many = Array.from({ length: EVENT_STREAM_CAP + 50 }, (_, i) =>
      makeEvent({ event_id: `e${i}`, occurred_at: i }),
    )
    const merged = mergeEngineEvents([], many)
    expect(merged).toHaveLength(EVENT_STREAM_CAP)
    // Keeps the NEWEST ones (highest occurred_at), not an arbitrary slice.
    expect(merged[0]!.event_id).toBe(many[many.length - 1]!.event_id)
  })

  it('keeps the merged set sorted newest-first by occurred_at', () => {
    const early = makeEvent({ event_id: 'a', occurred_at: 1 })
    const late = makeEvent({ event_id: 'b', occurred_at: 99 })
    const merged = mergeEngineEvents([early], [late])
    expect(merged.map((e) => e.event_id)).toEqual(['b', 'a'])
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
  }
}

describe('connectEventStream', () => {
  it('delivers the opening backlog and then a live anomaly as they arrive on the wire', async () => {
    const first = controllableStream()
    server.use(
      http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
        return new HttpResponse(first.stream, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const backlogEvent = makeEvent({ event_id: 'backlog-1' })
    const liveEvent = makeEvent({ event_id: 'live-1' })
    const onBacklog = vi.fn()
    const onEvent = vi.fn()
    const onOverflow = vi.fn()
    const onStatusChange = vi.fn()

    const connection = connectEventStream({ onBacklog, onEvent, onOverflow, onStatusChange })

    first.push(`event: backlog\ndata: ${JSON.stringify([backlogEvent])}\n\n`)
    await vi.waitFor(() => expect(onBacklog).toHaveBeenCalledWith([backlogEvent]))
    expect(onStatusChange).toHaveBeenCalledWith('live')

    first.push(`event: anomaly\ndata: ${JSON.stringify(liveEvent)}\n\n`)
    await vi.waitFor(() => expect(onEvent).toHaveBeenCalledWith(liveEvent))
    expect(onOverflow).not.toHaveBeenCalled()

    connection.close()
  })

  it('treats an overflow frame as informational and reconnects once the engine ends the stream, exactly like any other drop', async () => {
    vi.useFakeTimers()
    let requestCount = 0
    const first = controllableStream()
    const second = controllableStream()
    server.use(
      http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
        requestCount += 1
        const body = requestCount === 1 ? first.stream : second.stream
        return new HttpResponse(body, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const onOverflow = vi.fn()
    const statuses: string[] = []
    connectEventStream({
      onBacklog: () => {},
      onEvent: () => {},
      onOverflow,
      onStatusChange: (status) => statuses.push(status),
    })

    await vi.waitFor(() => expect(requestCount).toBe(1))
    first.push('event: backlog\ndata: []\n\n')
    await vi.waitFor(() => expect(statuses).toContain('live'))

    first.push(
      'event: overflow\ndata: {"reason":"the client fell too far behind this stream","action":"reconnect"}\n\n',
    )
    await vi.waitFor(() => expect(onOverflow).toHaveBeenCalledTimes(1))
    // The engine closes the stream right after the overflow frame.
    first.close()
    await vi.waitFor(() => expect(statuses.at(-1)).toBe('reconnecting'))

    await vi.advanceTimersByTimeAsync(1000)
    await vi.waitFor(() => expect(requestCount).toBe(2))

    vi.useRealTimers()
  })

  it('reconnects after the connection drops, and the reconnect is a genuinely new request', async () => {
    vi.useFakeTimers()
    let requestCount = 0
    const first = controllableStream()
    const second = controllableStream()
    server.use(
      http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
        requestCount += 1
        const body = requestCount === 1 ? first.stream : second.stream
        return new HttpResponse(body, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const statuses: string[] = []
    const connection = connectEventStream({
      onBacklog: () => {},
      onEvent: () => {},
      onOverflow: () => {},
      onStatusChange: (status) => statuses.push(status),
    })

    await vi.waitFor(() => expect(requestCount).toBe(1))
    first.push('event: backlog\ndata: []\n\n')
    await vi.waitFor(() => expect(statuses).toContain('live'))

    first.close()
    await vi.waitFor(() => expect(statuses.at(-1)).toBe('reconnecting'))

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

  it('stops reconnecting once closed, so an unmounted page cannot leak a background retry loop', async () => {
    vi.useFakeTimers()
    let requestCount = 0
    const first = controllableStream()
    server.use(
      http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
        requestCount += 1
        return new HttpResponse(first.stream, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )

    const statuses: string[] = []
    const connection = connectEventStream({
      onBacklog: () => {},
      onEvent: () => {},
      onOverflow: () => {},
      onStatusChange: (status) => statuses.push(status),
    })
    await vi.waitFor(() => expect(requestCount).toBe(1))
    const statusesAtClose = statuses.length

    first.close()
    connection.close()

    await vi.advanceTimersByTimeAsync(30_000)
    expect(requestCount).toBe(1)
    expect(statuses.length).toBe(statusesAtClose)
    vi.useRealTimers()
  })

  it('drops a frame under an unknown event name rather than guessing at its meaning', async () => {
    const first = controllableStream()
    server.use(
      http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
        return new HttpResponse(first.stream, { headers: { 'Content-Type': 'text/event-stream' } })
      }),
    )
    const onBacklog = vi.fn()
    const onEvent = vi.fn()
    const connection = connectEventStream({
      onBacklog,
      onEvent,
      onOverflow: () => {},
      onStatusChange: () => {},
    })
    first.push('event: backlog\ndata: []\n\n')
    await vi.waitFor(() => expect(onBacklog).toHaveBeenCalled())

    first.push('event: mystery\ndata: {"event_id":"x"}\n\n')
    // Give the microtask queue a turn; nothing should have fired.
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(onEvent).not.toHaveBeenCalled()

    connection.close()
  })
})
