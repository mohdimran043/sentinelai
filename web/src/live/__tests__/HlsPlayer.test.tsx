import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { HlsPlayer } from '@/live/HlsPlayer'
import { MEDIAMTX_BASE_URL } from '@/api/config'

describe('HlsPlayer — a browser that can play neither natively nor via MSE', () => {
  it('says so plainly, probes nothing, and never shows a player', async () => {
    // Deliberately not mocked: this project's real jsdom test environment has
    // no native HLS and no MediaSource, so this is the honest "can't play it"
    // path, exercised for free — not a fabricated stand-in.
    const fetchSpy = vi.spyOn(globalThis, 'fetch')

    render(<HlsPlayer cameraId="demo_live" />)

    expect(await screen.findByText(/can't play live video/i)).toBeInTheDocument()
    // An unsupported browser gains nothing from polling mediamtx — it never tries.
    expect(fetchSpy).not.toHaveBeenCalled()
    // No spinner/"connecting" status masquerading as a working player.
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByTestId('hls-video')).toHaveClass('hidden')

    fetchSpy.mockRestore()
  })
})

describe('HlsPlayer — native HLS support (Safari-like)', () => {
  let fetchMock: ReturnType<typeof vi.fn>
  let canPlayTypeSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.useFakeTimers()
    canPlayTypeSpy = vi.spyOn(HTMLMediaElement.prototype, 'canPlayType').mockReturnValue('maybe')
    fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
  })

  afterEach(() => {
    canPlayTypeSpy.mockRestore()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  /**
   * Testing Library's `findBy*`/`waitFor` poll on real `setTimeout`, which
   * fake timers intercept too — mixing them hangs. Advancing fake time
   * inside `act` and then asserting synchronously is the combination that
   * actually flushes the resulting React state update.
   */
  async function advance(ms: number): Promise<void> {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })
  }

  it('checks first, without claiming a stream exists before anything answers', () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    render(<HlsPlayer cameraId="replay_01" />)
    expect(screen.getByText(/checking for a live stream/i)).toBeInTheDocument()
  })

  it('falls back to the engine snapshot — not an empty panel — and derives the URL from the base and camera id', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 404 }))
    render(<HlsPlayer cameraId="replay_01" />)

    await advance(0)

    // No playlist is a permanent state for an EarthCam page or a file, not a wait.
    // The engine holds the latest frame either way, so the panel shows that and says
    // plainly that it is a still.
    expect(screen.getByTestId('camera-snapshot')).toBeInTheDocument()
    expect(screen.getByText(/this is a still and not a stream/i)).toBeInTheDocument()
    // The checking state does not linger once an answer is in — no forever spinner.
    expect(screen.queryByText(/checking for a live stream/i)).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      `${MEDIAMTX_BASE_URL}/replay_01/index.m3u8`,
      expect.anything(),
    )
  })

  it('recovers to live the moment the manifest becomes reachable, playing via the native element', async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 404 }))
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    render(<HlsPlayer cameraId="avenue_01" />)

    await advance(0)
    expect(screen.getByTestId('camera-snapshot')).toBeInTheDocument()

    await advance(2_000) // first backoff retry, now succeeds
    // Real video wins the moment it exists: the still is a fallback, never a preference.
    expect(screen.queryByTestId('camera-snapshot')).not.toBeInTheDocument()

    const video = screen.getByTestId('hls-video') as HTMLVideoElement
    expect(video).not.toHaveClass('hidden')
    expect(video.src).toBe(`${MEDIAMTX_BASE_URL}/avenue_01/index.m3u8`)
  })

  it('shows the drop honestly, not a frozen live view, and recovers automatically once the probe succeeds again', async () => {
    fetchMock.mockResolvedValue(new Response(null, { status: 200 }))
    render(<HlsPlayer cameraId="demo_live" />)
    await advance(0)

    const video = screen.getByTestId('hls-video')
    expect(video).not.toHaveClass('hidden')

    await act(async () => {
      fireEvent.error(video)
    })

    expect(screen.getByText(/stream dropped/i)).toBeInTheDocument()
    expect(video).toHaveClass('hidden')

    // The ongoing probe (still returning 200 throughout) reaches its next
    // healthy-interval recheck and brings it back without a reload.
    await advance(12_000)
    expect(screen.queryByText(/stream dropped/i)).not.toBeInTheDocument()
    expect(screen.getByTestId('hls-video')).not.toHaveClass('hidden')
  })
})
