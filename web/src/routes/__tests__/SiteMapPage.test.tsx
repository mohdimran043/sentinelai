import { beforeEach, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { server } from '@/test/mswServer'
import { renderWithProviders } from '@/test/renderWithProviders'
import { ENGINE_BASE_URL } from '@/api/config'
import { SiteMapPage } from '@/routes/SiteMapPage'
import { useSiteMapStore } from '@/sitemap/siteMapStore'
import * as engineClient from '@/api/engineClient'
import type { CamerasResponse, CameraStatus, RecentEventEntry } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, listCameras: vi.fn() }
})

const listCameras = vi.mocked(engineClient.listCameras)

function makeCameraStatus(overrides: Partial<CameraStatus> = {}): CameraStatus {
  return {
    camera_id: 'room_4b',
    label: 'room_4b',
    frames_seen: 0,
    frames_dropped: 0,
    detections_run: 0,
    escalations: 0,
    escalations_dropped: 0,
    discontinuities: 0,
    last_frame_at: null,
    // The wall-clock observation liveness actually reads; `last_frame_at` is the
    // camera's own source timeline and means something different.
    last_frame_epoch: null,
    last_escalation_at: null,
    // What an unconfigured camera loads with: both capabilities, every concern
    // kind, the stronger tier, and null durations meaning "follow the engine
    // default".
    capabilities: ['anomaly_detection', 'scene_description'],
    falls_suspected: 0,
    notify_on: ['collapse', 'altercation', 'self_harm', 'medication', 'distress', 'other'],
    notify_min_confidence: 'likely',
    clip_preroll_seconds: null,
    clip_postroll_seconds: null,
    summary_interval_seconds: null,
    ...overrides,
  }
}

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: 'e1',
    camera_id: 'room_4b',
    sequence: 1,
    clip_uri: null,
    occurred_at: Date.now() / 1000,
    source_timestamp: null,
    reason: 'new_salient_track',
    threat_score: 0.9,
    severity: 'critical',
    description: 'test event',
    suggested_action: 'Review now.',
    description_unavailable: false,
    labels: [],
    track_ids: [],
    welfare_concerns: [],
    ...overrides,
  }
}

/** A controllable `text/event-stream` body the test can push frames into on demand. */
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
  }
}

function registerEventStream() {
  const handle = controllableStream()
  server.use(
    http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
      return new HttpResponse(handle.stream, { headers: { 'Content-Type': 'text/event-stream' } })
    }),
  )
  return handle
}

/** Default: an empty backlog and otherwise silent, for tests that do not care about the stream itself. */
function registerEmptyEventStream() {
  server.use(
    http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('event: backlog\ndata: []\n\n'))
        },
      })
      return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } })
    }),
  )
}

describe('SiteMapPage', () => {
  beforeEach(() => {
    listCameras.mockReset()
    localStorage.clear()
    useSiteMapStore.setState({ placements: {}, floorplanImage: null })
  })

  it('renders a marker for each configured camera on the zone-grouped schematic when no floorplan image is set', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({
      cameras: [
        makeCameraStatus({ camera_id: 'room_4b', zone: 'room' }),
        makeCameraStatus({ camera_id: 'corridor_1', zone: 'corridor' }),
      ],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)

    expect(await screen.findByTestId('marker-room_4b')).toBeInTheDocument()
    expect(screen.getByTestId('marker-corridor_1')).toBeInTheDocument()
    expect(screen.getByText('Rooms')).toBeInTheDocument()
    expect(screen.getByText('Corridors')).toBeInTheDocument()
  })

  it('invites the operator to act on a genuinely empty camera list, rather than an empty map', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({ cameras: [], config_writable: false })

    renderWithProviders(<SiteMapPage />)

    expect(await screen.findByText(/no cameras configured/i)).toBeInTheDocument()
  })

  it('reports cameras as unreachable rather than rendering a blank map', async () => {
    registerEmptyEventStream()
    listCameras.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))

    renderWithProviders(<SiteMapPage />)

    expect(await screen.findByText(/cameras unreachable/i, {}, { timeout: 4000 })).toBeInTheDocument()
  })

  it('states the placements are saved in this browser only, not shared between operators', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ zone: 'room' })],
      config_writable: false,
    } as CamerasResponse)

    renderWithProviders(<SiteMapPage />)

    expect(await screen.findByText(/this browser only/i)).toBeInTheDocument()
  })

  it('colours a marker by the latest live event severity — starting inert, not a default green', async () => {
    const stream = registerEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)
    const marker = await screen.findByTestId('marker-room_4b')
    expect(marker).toHaveAccessibleName(/no events yet/i)

    stream.push('event: backlog\ndata: []\n\n')
    stream.push(`event: anomaly\ndata: ${JSON.stringify(makeEvent({ severity: 'critical' }))}\n\n`)

    await waitFor(() => {
      expect(screen.getByTestId('marker-room_4b')).toHaveAccessibleName(/breach/i)
    })
  })

  it('an event that is later re-sent with a higher sequence (a clip back-filling in) updates the SAME activity row rather than adding a second one', async () => {
    const stream = registerEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)
    await screen.findByTestId('marker-room_4b')

    stream.push('event: backlog\ndata: []\n\n')
    const original = makeEvent({ event_id: 'evt-1', sequence: 1, clip_uri: null })
    stream.push(`event: anomaly\ndata: ${JSON.stringify(original)}\n\n`)

    const feed = await screen.findByTestId('live-activity-panel')
    await waitFor(() => expect(within(feed).getAllByRole('listitem')).toHaveLength(1))
    expect(within(feed).queryByText(/clip attached/i)).not.toBeInTheDocument()

    const backfilled = makeEvent({ event_id: 'evt-1', sequence: 2, clip_uri: 's3://clips/evt-1.mp4' })
    stream.push(`event: anomaly\ndata: ${JSON.stringify(backfilled)}\n\n`)

    await waitFor(() => expect(within(feed).getByText(/clip attached/i)).toBeInTheDocument())
    // Still exactly one row for this event — the back-fill updated it, it did not duplicate it.
    expect(within(feed).getAllByRole('listitem')).toHaveLength(1)
  })

  it('a duplicate re-send at the SAME sequence changes nothing in the feed', async () => {
    const stream = registerEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)
    await screen.findByTestId('marker-room_4b')

    stream.push('event: backlog\ndata: []\n\n')
    const event = makeEvent({ event_id: 'evt-1', sequence: 1 })
    stream.push(`event: anomaly\ndata: ${JSON.stringify(event)}\n\n`)

    const feed = await screen.findByTestId('live-activity-panel')
    await waitFor(() => expect(within(feed).getAllByRole('listitem')).toHaveLength(1))

    stream.push(`event: anomaly\ndata: ${JSON.stringify(event)}\n\n`)
    // Give the duplicate a turn to (not) apply.
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(within(feed).getAllByRole('listitem')).toHaveLength(1)
  })

  it('"Arrange cameras" switches markers from links into keyboard/pointer-editable buttons, and back', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })
    const user = userEvent.setup()

    renderWithProviders(<SiteMapPage />)
    await screen.findByTestId('marker-room_4b')
    expect(screen.getByRole('link', { name: /room_4b/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /arrange cameras/i }))
    expect(screen.queryByRole('link', { name: /room_4b/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /room_4b/i })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /done arranging/i }))
    expect(await screen.findByRole('link', { name: /room_4b/i })).toBeInTheDocument()
  })

  it('uploading a floorplan image replaces the zone-grouped schematic with the image', async () => {
    registerEmptyEventStream()
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<SiteMapPage />)
    await screen.findByTestId('marker-room_4b')
    expect(screen.getByText('Rooms')).toBeInTheDocument()

    const file = new File(['fake-bytes'], 'plan.png', { type: 'image/png' })
    const input = screen.getByLabelText(/upload floorplan image/i) as HTMLInputElement
    await userEvent.setup().upload(input, file)

    await waitFor(() => expect(screen.queryByText('Rooms')).not.toBeInTheDocument())
    const stage = screen.getByTestId('floorplan-stage')
    const img = stage.querySelector('img')
    expect(img?.getAttribute('src')).toMatch(/^data:image\/png;base64,/)
  })
})
