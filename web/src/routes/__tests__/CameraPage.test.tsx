import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraPage } from '@/routes/CameraPage'
import * as engineClient from '@/api/engineClient'
import type {
  CameraEventsResponse,
  CameraStatus,
  DescribeResponse,
  RecentEventEntry,
} from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return {
    ...actual,
    getCameraTelemetry: vi.fn(),
    describeCameraNow: vi.fn(),
    getCameraEvents: vi.fn(),
  }
})

const getCameraTelemetry = vi.mocked(engineClient.getCameraTelemetry)
const describeCameraNow = vi.mocked(engineClient.describeCameraNow)
const getCameraEvents = vi.mocked(engineClient.getCameraEvents)

const telemetry: CameraStatus = {
  camera_id: 'avenue_01',
  frames_seen: 5000,
  frames_dropped: 10,
  detections_run: 4800,
  escalations: 7,
  escalations_dropped: 1,
  discontinuities: 0,
  last_frame_at: Date.now() / 1000,
  last_escalation_at: Date.now() / 1000 - 120,
}

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: '11111111-1111-4111-8111-111111111111',
    occurred_at: Date.now() / 1000 - 30,
    source_timestamp: 12.5,
    reason: 'new_salient_track',
    threat_score: 0.2,
    severity: 'low',
    description: 'A person walks across the frame.',
    suggested_action: 'Continue monitoring.',
    description_unavailable: false,
    labels: ['person'],
    track_ids: [1],
    ...overrides,
  }
}

function makeEventsResponse(overrides: Partial<CameraEventsResponse> = {}): CameraEventsResponse {
  return {
    camera_id: 'avenue_01',
    capacity: 200,
    returned: 0,
    volatile: true,
    latest_description_state: 'none',
    latest: null,
    events: [],
    ...overrides,
  }
}

function renderCameraPage() {
  return renderWithProviders(
    <Routes>
      <Route path="/cameras/:cameraId" element={<CameraPage />} />
    </Routes>,
    { route: '/cameras/avenue_01' },
  )
}

describe('CameraPage', () => {
  beforeEach(() => {
    getCameraTelemetry.mockReset()
    describeCameraNow.mockReset()
    getCameraEvents.mockReset()
  })

  it('shows live telemetry', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockResolvedValue(makeEventsResponse())

    renderCameraPage()

    expect(await screen.findByText('Frames seen')).toBeInTheDocument()
    expect(screen.getByText('5,000')).toBeInTheDocument()
  })

  it('calls describe-now and reports the queued event id', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockResolvedValue(makeEventsResponse())
    const response: DescribeResponse = { event_id: '22222222-2222-4222-8222-222222222222' }
    describeCameraNow.mockResolvedValue(response)
    const user = userEvent.setup()

    renderCameraPage()
    await screen.findByText('Frames seen')

    await user.click(screen.getByRole('button', { name: /describe now/i }))

    expect(await screen.findByTestId('describe-feedback')).toHaveTextContent(/queued/i)
    expect(describeCameraNow).toHaveBeenCalledWith('avenue_01')
  })

  it('reports telemetry unreachable honestly rather than blank tiles', async () => {
    getCameraTelemetry.mockRejectedValue(
      new engineClient.EngineUnreachableError(new Error('refused')),
    )
    getCameraEvents.mockResolvedValue(makeEventsResponse())

    renderCameraPage()

    expect(await screen.findByRole('alert', {}, { timeout: 4000 })).toHaveTextContent(
      /telemetry unreachable/i,
    )
    expect(screen.queryByText('Frames seen')).not.toBeInTheDocument()
  })

  describe('the three description states', () => {
    it('"none": reads as an invitation, not an error, and the chart and notifications agree nothing has happened', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({ latest_description_state: 'none', latest: null, events: [] }),
      )

      renderCameraPage()

      expect(await screen.findByText(/no description yet/i)).toBeInTheDocument()
      // Never rendered as an error banner/alert.
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
      expect(await screen.findByText(/no events yet/i)).toBeInTheDocument()
      expect(await screen.findByText(/no notifications yet/i)).toBeInTheDocument()
    })

    it('"available": shows the real description with its threat score and severity', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const event = makeEvent({
        description: 'Two people are talking calmly near the entrance.',
        threat_score: 0.35,
        severity: 'medium',
      })
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'available',
          latest: event,
          returned: 1,
          events: [event],
        }),
      )

      renderCameraPage()

      const panel = await screen.findByTestId('scene-description-panel')
      expect(
        await within(panel).findByText('Two people are talking calmly near the entrance.'),
      ).toBeInTheDocument()
      expect(within(panel).getByText(/0\.35/)).toBeInTheDocument()
      expect(within(panel).getByText('medium')).toBeInTheDocument()
      // Must not show the degraded-VLM caveat when the description is real.
      expect(within(panel).queryByText(/could not answer/i)).not.toBeInTheDocument()
    })

    it('"unavailable": says so plainly rather than presenting the stand-in as a real description', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const event = makeEvent({
        description: 'Escalation triggered (new_salient_track); the vision model could not answer.',
        description_unavailable: true,
        severity: 'low',
      })
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'unavailable',
          latest: event,
          returned: 1,
          events: [event],
        }),
      )

      renderCameraPage()

      const panel = await screen.findByTestId('scene-description-panel')
      expect(await within(panel).findByRole('status')).toHaveTextContent(/could not answer/i)
      expect(
        within(panel).getByText(
          'Escalation triggered (new_salient_track); the vision model could not answer.',
        ),
      ).toBeInTheDocument()
      // The "none" invitation copy must not also appear.
      expect(within(panel).queryByText(/no description yet/i)).not.toBeInTheDocument()
    })
  })

  describe('threat-over-time chart', () => {
    it('renders one cell per ring event, not an empty grid', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const events = [
        makeEvent({ event_id: 'a', severity: 'info', occurred_at: 1_700_000_000 }),
        makeEvent({ event_id: 'b', severity: 'medium', occurred_at: 1_700_000_100 }),
        makeEvent({ event_id: 'c', severity: 'critical', occurred_at: 1_700_000_200 }),
      ]
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'available',
          latest: events[2],
          returned: 3,
          events,
        }),
      )

      renderCameraPage()

      const ribbon = await screen.findByRole('img', { name: /threat over time/i })
      // One direct-child cell div per event — fails immediately if the chart
      // renders an empty or fixed-size grid instead of the real ring.
      expect(ribbon.children).toHaveLength(3)

      // The screen-reader table backs each visual cell with the same data —
      // asserting on it proves the chart carries real per-event content, not
      // three identical blank swatches.
      const table = screen.getByRole('table')
      const rows = within(table).getAllByRole('row').slice(1) // drop the header row
      expect(rows).toHaveLength(3)
      expect(rows[2]!.textContent).toMatch(/critical/i)
    })

    it('shows an honest empty state instead of an empty chart when nothing has escalated', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      getCameraEvents.mockResolvedValue(makeEventsResponse())

      renderCameraPage()

      expect(await screen.findByText(/no events yet/i)).toBeInTheDocument()
      expect(screen.queryByRole('img', { name: /threat over time/i })).not.toBeInTheDocument()
    })

    it('reports the ring bound honestly rather than implying full history', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const event = makeEvent()
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'available',
          latest: event,
          returned: 1,
          capacity: 200,
          events: [event],
        }),
      )

      renderCameraPage()

      expect(
        await screen.findByTestId('chart-footnote', {}, { timeout: 4000 }),
      ).toHaveTextContent(/1 of up to 200 retained events/i)
    })
  })

  describe('notifications', () => {
    it('lists recent events newest first, with when, reason, and description', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const older = makeEvent({
        event_id: 'older',
        reason: 'dwell_exceeded',
        occurred_at: 1_700_000_000,
        description: 'A figure lingers by the gate.',
      })
      const newer = makeEvent({
        event_id: 'newer',
        reason: 'scene_change',
        occurred_at: 1_700_000_500,
        description: 'The scene changed abruptly.',
      })
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'available',
          latest: newer,
          returned: 2,
          events: [older, newer],
        }),
      )

      renderCameraPage()

      const panel = await screen.findByTestId('notifications-panel')
      expect(await within(panel).findByText('The scene changed abruptly.')).toBeInTheDocument()
      expect(within(panel).getByText('A figure lingers by the gate.')).toBeInTheDocument()

      const items = within(panel).getAllByRole('listitem')
      const newerIndex = items.findIndex((item) => item.textContent?.includes('scene changed'))
      const olderIndex = items.findIndex((item) => item.textContent?.includes('lingers by the gate'))
      expect(newerIndex).toBeGreaterThanOrEqual(0)
      expect(olderIndex).toBeGreaterThan(newerIndex)
    })

    it('flags a description-unavailable event in the list rather than presenting the stand-in as real', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      const event = makeEvent({ description_unavailable: true, description: 'stand-in text' })
      getCameraEvents.mockResolvedValue(
        makeEventsResponse({
          latest_description_state: 'unavailable',
          latest: event,
          returned: 1,
          events: [event],
        }),
      )

      renderCameraPage()

      const panel = await screen.findByTestId('notifications-panel')
      expect(await within(panel).findByText(/description unavailable/i)).toBeInTheDocument()
    })

    it('shows an honest empty state when nothing has happened yet', async () => {
      getCameraTelemetry.mockResolvedValue(telemetry)
      getCameraEvents.mockResolvedValue(makeEventsResponse())

      renderCameraPage()

      expect(await screen.findByText(/no notifications yet/i)).toBeInTheDocument()
    })
  })

  it('reports the event ring unreachable independently of telemetry', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    getCameraEvents.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))

    renderCameraPage()

    expect(await screen.findByText('Frames seen')).toBeInTheDocument()
    expect(
      await screen.findByText(/live scene description unreachable/i, {}, { timeout: 4000 }),
    ).toBeInTheDocument()
    expect(await screen.findByText(/chart unavailable/i)).toBeInTheDocument()
    expect(await screen.findByText(/notifications unavailable/i)).toBeInTheDocument()
  })
})
