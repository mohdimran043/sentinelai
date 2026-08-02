import { beforeEach, describe, expect, it, vi } from 'vitest'
import { http, HttpResponse } from 'msw'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes, useParams } from 'react-router-dom'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { ENGINE_BASE_URL } from '@/api/config'
import { DashboardPage } from '@/routes/DashboardPage'
import * as engineClient from '@/api/engineClient'
import type { CameraStatus, CamerasResponse, HealthResponse, RecentEventEntry } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return {
    ...actual,
    getHealth: vi.fn(),
    listCameras: vi.fn(),
  }
})

const getHealth = vi.mocked(engineClient.getHealth)
const listCameras = vi.mocked(engineClient.listCameras)

function makeCameraStatus(overrides: Partial<CameraStatus> = {}): CameraStatus {
  const camera_id = overrides.camera_id ?? 'avenue_01'
  return {
    camera_id,
    // Defaults to the id, like the real engine does when `cameras.json` gives
    // no label — so tests that only override `camera_id` still find their
    // camera by the same text a tile renders. Pass `label` explicitly to test
    // the label/id divergence itself.
    label: camera_id,
    frames_seen: 1000,
    frames_dropped: 2,
    detections_run: 900,
    escalations: 3,
    escalations_dropped: 0,
    discontinuities: 0,
    last_frame_at: Date.now() / 1000,
    last_escalation_at: null,
    ...overrides,
  }
}

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: 'e1',
    camera_id: 'avenue_01',
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
    ...overrides,
  }
}

const sampleHealth: HealthResponse = {
  status: 'ok',
  models: [
    { key: 'yolo11s', state: 'loaded', detail: 'ready', vram_mib: 512 },
    { key: 'qwen2.5-vl-3b-awq', state: 'sleeping', detail: 'idle-unloaded', vram_mib: 0 },
  ],
}

const sampleCameras: CamerasResponse = {
  cameras: [makeCameraStatus()],
  config_writable: false,
}

/** Default: an empty backlog and otherwise silent, for tests that do not care about the live event stream. */
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

/** A controllable `text/event-stream` body a test can push frames into on demand. */
function registerControllableEventStream() {
  let controllerRef!: ReadableStreamDefaultController<Uint8Array>
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller
    },
  })
  server.use(
    http.get(`${ENGINE_BASE_URL}/events/stream`, () => {
      return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } })
    }),
  )
  const encoder = new TextEncoder()
  return { push: (text: string) => controllerRef.enqueue(encoder.encode(text)) }
}

/** A stand-in for the real camera page — proves the route lands on the right camera without coupling this test to `CameraPage`'s own (separately owned, in-progress) internals. */
function CameraRouteProbe() {
  const { cameraId } = useParams()
  return <div data-testid="camera-route-probe">Camera view: {cameraId}</div>
}

function renderDashboardWithCameraRoute() {
  return renderWithProviders(
    <Routes>
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/cameras/:cameraId" element={<CameraRouteProbe />} />
    </Routes>,
    { route: '/dashboard' },
  )
}

describe('DashboardPage', () => {
  beforeEach(() => {
    getHealth.mockReset()
    listCameras.mockReset()
  })

  it('renders live camera and model data when the engine answers', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue(sampleCameras)

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('avenue_01')).toBeInTheDocument()
    expect(screen.getByText('yolo11s')).toBeInTheDocument()
    // The invented Phase-1C placeholders (storage, visitor counts, etc.) are gone.
    expect(screen.queryByText('Awaiting Phase 1C')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Storage usage' })).not.toBeInTheDocument()
  })

  it('shows a camera tile with its zone, liveness and an inert "no events yet" pill absent any live event', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('room_4b')).toBeInTheDocument()
    expect(screen.getByText('Room')).toBeInTheDocument()
    expect(screen.getByText('Live')).toBeInTheDocument()
    expect(screen.getByText('no events yet')).toBeInTheDocument()
  })

  it('groups cameras by zone when more than one zone is present, ungrouped cameras included', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [
        makeCameraStatus({ camera_id: 'room_4b', zone: 'room' }),
        makeCameraStatus({ camera_id: 'corridor_1', zone: 'corridor' }),
        makeCameraStatus({ camera_id: 'loose_cam', zone: null }),
      ],
      config_writable: false,
    })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('room_4b')).toBeInTheDocument()
    expect(screen.getByText('Rooms')).toBeInTheDocument()
    expect(screen.getByText('Corridors')).toBeInTheDocument()
    expect(screen.getByText('Ungrouped')).toBeInTheDocument()
    expect(screen.getByText('loose_cam')).toBeInTheDocument()
  })

  it('shows a per-tile "Ungrouped" zone label, not a redundant band heading, when every camera lacks a zone', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue(sampleCameras)

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('avenue_01')).toBeInTheDocument()
    // Exactly one "Ungrouped" — the tile's own zone label. getByText throws on
    // more than one match, so this also proves there is no second, redundant
    // band-heading copy of the same text.
    expect(screen.getByText('Ungrouped')).toBeInTheDocument()
  })

  it('colours a tile by its latest live severity, and shows the threat score', async () => {
    const stream = registerControllableEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'avenue_01' })],
      config_writable: false,
    })

    renderWithProviders(<DashboardPage />)
    expect(await screen.findByText('avenue_01')).toBeInTheDocument()
    expect(screen.getByText('no events yet')).toBeInTheDocument()

    stream.push('event: backlog\ndata: []\n\n')
    stream.push(
      `event: anomaly\ndata: ${JSON.stringify(
        makeEvent({ camera_id: 'avenue_01', severity: 'critical', threat_score: 0.87 }),
      )}\n\n`,
    )

    await waitFor(() => {
      expect(screen.getByText('critical')).toBeInTheDocument()
    })
    expect(screen.getByText('0.87')).toBeInTheDocument()
  })

  it('degrades honestly instead of showing zeros when the engine is unreachable', async () => {
    registerEmptyEventStream()
    getHealth.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))
    listCameras.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))

    renderWithProviders(<DashboardPage />)

    await waitFor(
      () => {
        expect(screen.getByRole('alert')).toHaveTextContent(/the ai engine is unreachable/i)
      },
      { timeout: 4000 },
    )
    expect(screen.queryByText('avenue_01')).not.toBeInTheDocument()
    // No fabricated numeric tiles for cameras/health while unreachable.
    expect(screen.queryByText(/^0$/)).not.toBeInTheDocument()
  })

  it('invites the operator to act on a genuinely empty camera list', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue({ status: 'ok', models: [] })
    listCameras.mockResolvedValue({ cameras: [], config_writable: false })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText(/no cameras configured/i)).toBeInTheDocument()
  })

  it('names a tile by its label rather than its id', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', label: 'Room 4B, window side' })],
      config_writable: false,
    })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('Room 4B, window side')).toBeInTheDocument()
  })

  it("clicking a camera tile navigates to that camera's own page", async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [
        makeCameraStatus({ camera_id: 'room_4b', zone: 'room' }),
        makeCameraStatus({ camera_id: 'corridor_1', zone: 'corridor' }),
      ],
      config_writable: false,
    })
    const user = userEvent.setup()

    renderDashboardWithCameraRoute()
    await screen.findByText('room_4b')

    await user.click(screen.getByRole('link', { name: /corridor_1/i }))

    expect(await screen.findByTestId('camera-route-probe')).toHaveTextContent(
      'Camera view: corridor_1',
    )
  })

  it('reaching a camera tile by keyboard (Tab to it, then Enter) navigates to the same page a click would', async () => {
    registerEmptyEventStream()
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue({
      cameras: [makeCameraStatus({ camera_id: 'room_4b', zone: 'room' })],
      config_writable: false,
    })
    const user = userEvent.setup()

    renderDashboardWithCameraRoute()
    await screen.findByText('room_4b')

    const link = screen.getByRole('link', { name: /room_4b/i })
    link.focus()
    expect(link).toHaveFocus()
    await user.keyboard('{Enter}')

    expect(await screen.findByTestId('camera-route-probe')).toHaveTextContent(
      'Camera view: room_4b',
    )
  })
})
