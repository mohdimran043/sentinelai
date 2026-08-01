import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { HlsPlayer } from '@/live/HlsPlayer'

/**
 * Covers the branch `HlsPlayer.test.tsx` cannot: a browser with Media Source
 * Extensions but no native HLS (i.e. everything that is not Safari), which
 * needs `hls.js`. `hls.js` itself is mocked — this is orchestration, not
 * decoding, per this project's own instruction not to pretend jsdom can test
 * real media playback. What IS real here: the lazy `import('hls.js')` only
 * firing once the manifest is reachable, and the exact arguments this build
 * hands the library (`attachMedia`/`loadSource` with the derived URL), and
 * that a fatal `hls.js` error is what turns into `dropped` — not just any
 * error.
 */

interface FakeHlsInstance {
  attachMedia: ReturnType<typeof vi.fn>
  loadSource: ReturnType<typeof vi.fn>
  on: ReturnType<typeof vi.fn>
  destroy: ReturnType<typeof vi.fn>
}

const { instances } = vi.hoisted(() => ({ instances: [] as FakeHlsInstance[] }))

vi.mock('hls.js', () => {
  class FakeHls implements FakeHlsInstance {
    static isSupported = vi.fn(() => true)
    static Events = { ERROR: 'hlsError' } as const
    attachMedia = vi.fn()
    loadSource = vi.fn()
    on = vi.fn()
    destroy = vi.fn()
    constructor() {
      instances.push(this)
    }
  }
  return { default: FakeHls }
})

describe('HlsPlayer — MSE browsers (hls.js)', () => {
  let fetchMock: ReturnType<typeof vi.fn>

  beforeEach(() => {
    instances.length = 0
    ;(window as { MediaSource?: unknown }).MediaSource = function MediaSource() {}
    fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }))
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    delete (window as { MediaSource?: unknown }).MediaSource
    vi.unstubAllGlobals()
  })

  it('imports hls.js lazily and attaches it to the video element with the derived URL', async () => {
    render(<HlsPlayer cameraId="demo_live" />)

    await waitFor(() => expect(instances).toHaveLength(1))
    const hls = instances[0]!

    expect(hls.attachMedia).toHaveBeenCalledWith(screen.getByTestId('hls-video'))
    expect(hls.loadSource).toHaveBeenCalledWith('http://localhost:8888/demo_live/index.m3u8')
  })

  it('treats a fatal hls.js error as a drop, shown honestly rather than as still live', async () => {
    render(<HlsPlayer cameraId="demo_live" />)
    await waitFor(() => expect(instances).toHaveLength(1))
    const hls = instances[0]!

    const registration = hls.on.mock.calls.find(([event]) => event === 'hlsError')
    expect(registration).toBeDefined()
    const errorCallback = registration![1] as (event: string, data: { fatal: boolean }) => void
    errorCallback('hlsError', { fatal: true })

    expect(await screen.findByText(/stream dropped/i)).toBeInTheDocument()
  })

  it('ignores a non-fatal hls.js error — hls.js recovers those on its own, it is not a drop', async () => {
    render(<HlsPlayer cameraId="demo_live" />)
    await waitFor(() => expect(instances).toHaveLength(1))
    const hls = instances[0]!

    const registration = hls.on.mock.calls.find(([event]) => event === 'hlsError')
    expect(registration).toBeDefined()
    const errorCallback = registration![1] as (event: string, data: { fatal: boolean }) => void
    errorCallback('hlsError', { fatal: false })

    expect(screen.queryByText(/stream dropped/i)).not.toBeInTheDocument()
  })
})
