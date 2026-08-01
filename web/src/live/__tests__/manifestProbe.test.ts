import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { startManifestProbe } from '@/live/manifestProbe'

describe('startManifestProbe', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    vi.useFakeTimers()
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('probes immediately on start, with no-store so a cache cannot lie about reachability', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    startManifestProbe('http://x/index.m3u8', { onOk: vi.fn(), onFail: vi.fn() })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      'http://x/index.m3u8',
      expect.objectContaining({ cache: 'no-store' }),
    )
  })

  it('calls onOk on a 2xx response and rechecks on the slow healthy interval', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    const onOk = vi.fn()
    startManifestProbe('http://x/index.m3u8', { onOk, onFail: vi.fn() })
    await vi.advanceTimersByTimeAsync(0)
    expect(onOk).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(11_999)
    expect(fetchMock).toHaveBeenCalledTimes(1) // not yet due
    await vi.advanceTimersByTimeAsync(2)
    expect(fetchMock).toHaveBeenCalledTimes(2) // due just after 12s
  })

  it('calls onFail on a non-2xx response and retries with backoff, not the healthy interval', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    const onFail = vi.fn()
    startManifestProbe('http://x/index.m3u8', { onOk: vi.fn(), onFail })
    await vi.advanceTimersByTimeAsync(0)
    expect(onFail).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1_999)
    expect(fetchMock).toHaveBeenCalledTimes(1) // the first backoff step is 2s, not yet
    await vi.advanceTimersByTimeAsync(2)
    expect(fetchMock).toHaveBeenCalledTimes(2)

    // Second failure backs off further (4s), not the same 2s again.
    await vi.advanceTimersByTimeAsync(3_998)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(2)
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('treats a rejected fetch (network error / CORS) the same as a bad response', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    const onFail = vi.fn()
    const onOk = vi.fn()
    startManifestProbe('http://x/index.m3u8', { onOk, onFail })
    await vi.advanceTimersByTimeAsync(0)
    expect(onFail).toHaveBeenCalledTimes(1)
    expect(onOk).not.toHaveBeenCalled()
  })

  it('resets to the fast backoff after recovering, rather than remembering the old attempt count', async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 404 }))
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 404 }))
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    const onOk = vi.fn()
    const onFail = vi.fn()
    startManifestProbe('http://x/index.m3u8', { onOk, onFail })

    await vi.advanceTimersByTimeAsync(0) // fail #1 (attempt 0 -> delay 2s)
    await vi.advanceTimersByTimeAsync(2_000) // fail #2 (attempt 1 -> delay 4s)
    await vi.advanceTimersByTimeAsync(4_000) // ok — attempt resets to 0
    expect(onOk).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledTimes(3)

    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    await vi.advanceTimersByTimeAsync(12_000) // the healthy recheck fires, now fails
    // If attempt had not reset, this failure would schedule a long-backoff
    // retry; because it reset, the very next retry is the fast 2s one.
    await vi.advanceTimersByTimeAsync(1_999)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    await vi.advanceTimersByTimeAsync(2)
    expect(fetchMock).toHaveBeenCalledTimes(5)
  })

  it('caps backoff at a maximum instead of growing without bound', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    startManifestProbe('http://x/index.m3u8', { onOk: vi.fn(), onFail: vi.fn() })
    await vi.advanceTimersByTimeAsync(0)
    for (let i = 0; i < 6; i++) {
      await vi.advanceTimersByTimeAsync(20_000)
    }
    // Uncapped exponential backoff (2000 * 2^n) would have retried far less
    // often across this same 120s window; the cap is what keeps it frequent.
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(7)
  })

  it('stop() ends the loop for good — no further requests after it is called', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    const handle = startManifestProbe('http://x/index.m3u8', { onOk: vi.fn(), onFail: vi.fn() })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    handle.stop()
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
