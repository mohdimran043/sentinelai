import { describe, expect, it } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { AlertsPage } from '@/routes/recorder/AlertsPage'
import {
  recorderAlertsHandler,
  recorderErrorHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_ALERTS, makeAlert, makeAlertsResponse } from '@/recorder/mocks/fixtures'
import type { RecorderAlert } from '@/recorder/recorder.types'

/** Rows in the feed table, excluding the header row. */
async function feedRows() {
  const table = await screen.findByRole('table', { name: /^Alerts,/ })
  return within(table).getAllByRole('row').slice(1)
}

function cellsOfRows(rows: HTMLElement[], columnIndex: number): string[] {
  return rows.map((row) => within(row).getAllByRole('cell')[columnIndex]?.textContent?.trim() ?? '')
}

const CAMERA_COLUMN = 1
const EVENT_COLUMN = 2
const WHEN_COLUMN = 0

/** Two cameras x two event types x 30 each = 120 alerts, more than one page. */
function mixedAlerts(): RecorderAlert[] {
  const cameras = ['corridor_1', 'room_4b']
  const types = ['UNDERLIT', 'OBSTRUCTED']
  const alerts: RecorderAlert[] = []
  let offset = 0
  for (let index = 0; index < 30; index += 1) {
    for (const cameraId of cameras) {
      for (const eventType of types) {
        offset += 5
        alerts.push(makeAlert({ cameraId, eventType, offsetSeconds: offset }))
      }
    }
  }
  return alerts
}

describe('AlertsPage', () => {
  it('renders the real recorder payload, formatting at_ns as human time', async () => {
    renderWithProviders(<AlertsPage />)

    const rows = await feedRows()
    expect(rows).toHaveLength(REAL_ALERTS.alerts.length)
    expect(cellsOfRows(rows, CAMERA_COLUMN)).toEqual(['room_4b', 'room_4b', 'room_4b', 'room_4b'])
    // 1785578902544883331 ns, rendered in the suite's pinned UTC zone.
    expect(cellsOfRows(rows, WHEN_COLUMN)[0]).toBe('2026-08-01 10:08:22')
    expect(cellsOfRows(rows, WHEN_COLUMN)[0]).not.toContain('1785578902')
    // The recorder's own honesty note is shown, not summarised away.
    expect(screen.getByText(/an empty feed is not confirmation that anyone is well/i)).toBeInTheDocument()
  })

  it('filters by camera, hiding the other cameras’ alerts', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    await feedRows()
    expect(new Set(cellsOfRows(await feedRows(), CAMERA_COLUMN))).toEqual(
      new Set(['corridor_1', 'room_4b']),
    )

    await user.selectOptions(screen.getByLabelText('Camera'), 'room_4b')

    await waitFor(async () => {
      expect(new Set(cellsOfRows(await feedRows(), CAMERA_COLUMN))).toEqual(new Set(['room_4b']))
    })
    expect(screen.getByTestId('alert-count')).toHaveTextContent('1–50 of 60')
  })

  it('filters by event type, hiding the other types', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    await feedRows()
    await user.selectOptions(screen.getByLabelText('Event type'), 'OBSTRUCTED')

    await waitFor(async () => {
      expect(new Set(cellsOfRows(await feedRows(), EVENT_COLUMN))).toEqual(new Set(['OBSTRUCTED']))
    })
    expect(screen.getByTestId('alert-count')).toHaveTextContent('of 60')
  })

  it('combines both filters instead of applying only one', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    await feedRows()
    await user.selectOptions(screen.getByLabelText('Camera'), 'corridor_1')
    await user.selectOptions(screen.getByLabelText('Event type'), 'UNDERLIT')

    await waitFor(() => {
      expect(screen.getByTestId('alert-count')).toHaveTextContent('1–30 of 30')
    })
    const rows = await feedRows()
    expect(new Set(cellsOfRows(rows, CAMERA_COLUMN))).toEqual(new Set(['corridor_1']))
    expect(new Set(cellsOfRows(rows, EVENT_COLUMN))).toEqual(new Set(['UNDERLIT']))
  })

  it('only offers cameras and event types that appear in the window', async () => {
    server.use(
      recorderAlertsHandler(
        makeAlertsResponse([
          makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 1 }),
        ]),
      ),
    )
    renderWithProviders(<AlertsPage />)

    await feedRows()
    const cameraSelect = screen.getByLabelText('Camera')
    expect(within(cameraSelect).getAllByRole('option').map((o) => o.textContent)).toEqual([
      'All cameras',
      'room_4b',
    ])
    expect(within(screen.getByLabelText('Event type')).getAllByRole('option')).toHaveLength(2)
  })

  it('reverses the feed when the order is flipped', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    const newestFirst = cellsOfRows(await feedRows(), WHEN_COLUMN)[0]!

    await user.selectOptions(screen.getByLabelText('Order'), 'oldest')

    await waitFor(async () => {
      expect(cellsOfRows(await feedRows(), WHEN_COLUMN)[0]).not.toBe(newestFirst)
    })
    const oldestFirst = cellsOfRows(await feedRows(), WHEN_COLUMN)[0]!
    expect(oldestFirst < newestFirst).toBe(true)
  })

  it('pages rather than rendering the whole window at once', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    let rows = await feedRows()
    expect(rows).toHaveLength(50)
    expect(rows.length).toBeLessThan(120)
    expect(screen.getByTestId('alert-count')).toHaveTextContent('1–50 of 120')

    const firstPageTop = cellsOfRows(rows, WHEN_COLUMN)[0]!
    await user.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => {
      expect(screen.getByTestId('alert-count')).toHaveTextContent('51–100 of 120')
    })
    rows = await feedRows()
    expect(rows).toHaveLength(50)
    expect(cellsOfRows(rows, WHEN_COLUMN)[0]).not.toBe(firstPageTop)

    await user.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() => {
      expect(screen.getByTestId('alert-count')).toHaveTextContent('101–120 of 120')
    })
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
  })

  it('returns to the first page when a filter changes', async () => {
    const user = userEvent.setup()
    server.use(recorderAlertsHandler(makeAlertsResponse(mixedAlerts())))
    renderWithProviders(<AlertsPage />)

    await feedRows()
    await user.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() => {
      expect(screen.getByTestId('alert-count')).toHaveTextContent('51–100')
    })

    await user.selectOptions(screen.getByLabelText('Camera'), 'room_4b')

    await waitFor(() => {
      expect(screen.getByTestId('alert-count')).toHaveTextContent('1–50 of 60')
    })
  })

  it('opens a detail view with the exact timestamp and the alert’s delivery attempts', async () => {
    const user = userEvent.setup()
    renderWithProviders(<AlertsPage />)

    const rows = await feedRows()
    await user.click(within(rows[0]!).getByRole('button', { name: /^Details for UNDERLIT/ }))

    const detail = screen.getByRole('region', { name: /UNDERLIT\s*room_4b/ })
    expect(within(detail).getByText('1785578902552363255-11105')).toBeInTheDocument()
    expect(within(detail).getByText('2026-08-01 10:08:22.544')).toBeInTheDocument()
    expect(within(detail).getByText('luma_mean 0.063 below floor 0.08')).toBeInTheDocument()
    // The real payload records exactly one dashboard delivery for this alert.
    const deliveryTable = within(detail).getByRole('table')
    expect(within(deliveryTable).getAllByRole('row')).toHaveLength(2)
    expect(within(deliveryTable).getByText('delivered')).toBeInTheDocument()

    await user.click(within(detail).getByRole('button', { name: 'Close' }))
    expect(screen.queryByRole('region', { name: /UNDERLIT/ })).not.toBeInTheDocument()
  })

  it('says so when an alert has no recorded delivery attempt', async () => {
    const user = userEvent.setup()
    server.use(
      recorderAlertsHandler(
        makeAlertsResponse([
          makeAlert({ cameraId: 'room_4b', eventType: 'LOST', offsetSeconds: 1 }),
        ]),
      ),
    )
    renderWithProviders(<AlertsPage />)

    const rows = await feedRows()
    await user.click(within(rows[0]!).getByRole('button', { name: /^Details for LOST/ }))

    expect(
      screen.getByText(/No delivery attempt is recorded against this alert/i),
    ).toBeInTheDocument()
    expect(screen.getByText(/not the same as a successful delivery/i)).toBeInTheDocument()
  })

  it('says the window itself is empty when the recorder has raised nothing', async () => {
    server.use(recorderAlertsHandler(makeAlertsResponse([])))
    renderWithProviders(<AlertsPage />)

    expect(await screen.findByText(/Nothing has been delivered/i)).toBeInTheDocument()
    expect(screen.getByText(/it is not a statement that anyone is well/i)).toBeInTheDocument()
    expect(screen.getByTestId('alert-count')).toHaveTextContent('no alerts match · 0 in the window')
  })

  it('says the FILTER matched nothing — not that the recorder is empty — when the window is not', async () => {
    const user = userEvent.setup()
    server.use(
      recorderAlertsHandler(
        makeAlertsResponse([
          makeAlert({ cameraId: 'corridor_1', eventType: 'UNDERLIT', offsetSeconds: 5 }),
          makeAlert({ cameraId: 'room_4b', eventType: 'OBSTRUCTED', offsetSeconds: 10 }),
        ]),
      ),
    )
    renderWithProviders(<AlertsPage />)

    await feedRows()
    await user.selectOptions(screen.getByLabelText('Camera'), 'corridor_1')
    await user.selectOptions(screen.getByLabelText('Event type'), 'OBSTRUCTED')

    expect(await screen.findByText(/No alert matches those filters/i)).toBeInTheDocument()
    expect(screen.getByText(/The window itself is not empty/i)).toBeInTheDocument()
    expect(screen.queryByText(/Nothing has been delivered/i)).not.toBeInTheDocument()
    expect(screen.getByTestId('alert-count')).toHaveTextContent('no alerts match · 2 in the window')
  })

  it('surfaces dropped unsuppressible alerts as a fault, not a quiet night', async () => {
    server.use(
      recorderAlertsHandler(
        makeAlertsResponse(
          [makeAlert({ cameraId: 'room_4b', eventType: 'LOST', offsetSeconds: 1 })],
          { dropped: { total: 3, welfare_locked: 2 }, failed_deliveries: 1 },
        ),
      ),
    )
    renderWithProviders(<AlertsPage />)

    await feedRows()
    expect(screen.getByText(/Treat it as a fault in the system, not as a quiet night/i)).toBeInTheDocument()
    expect(screen.getByText('of those, unsuppressible').parentElement).toHaveTextContent('2')
  })

  it('says the recorder is unreachable instead of showing an empty feed', async () => {
    server.use(recorderUnreachableHandler('/alerts'))
    renderWithProviders(<AlertsPage />)

    expect(await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 })).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(screen.queryByText(/Nothing has been delivered/i)).not.toBeInTheDocument()
  })

  it('reports a recorder error response with the recorder’s own words', async () => {
    server.use(recorderErrorHandler('/alerts', 500, 'alert store unavailable'))
    renderWithProviders(<AlertsPage />)

    expect(
      await screen.findByText(/alert store unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })
})
