import { describe, expect, it } from 'vitest'
import { http, HttpResponse } from 'msw'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { RECORDER_BASE_URL } from '@/recorder/config'
import { DayReportPage } from '@/routes/recorder/DayReportPage'
import {
  recorderReportHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_REPORT, REAL_REPORT_FULLY_LOST, REAL_REPORT_UNKNOWN_CAMERA } from '@/recorder/mocks/fixtures'

const PINNED_ROUTE = '/recorder/report?camera=room_4b&date=2026-08-01'

describe('DayReportPage — camera/date picker', () => {
  it('defaults camera to the first in the registry and fills in a date, without being asked', async () => {
    renderWithProviders(<DayReportPage />, { route: '/recorder/report' })

    // A filled-in date is the only thing that proves the effect writing both
    // defaults (one setSearchParams batching camera and date) has landed.
    // Waiting on the camera select instead settles a render too early: while
    // no `camera` is in the URL the select is controlled to "", which matches
    // none of its options, so React falls back to marking the first option
    // selected — it already reads `room_4b` in the commit where the options
    // appear, before the search-params update commits.
    await waitFor(() => {
      expect((screen.getByLabelText('Date') as HTMLInputElement).value).toMatch(
        /^\d{4}-\d{2}-\d{2}$/,
      )
    })

    expect((screen.getByLabelText('Camera') as HTMLSelectElement).value).toBe('room_4b')
  })

  it('respects a camera/date already in the URL rather than overwriting it', async () => {
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    await screen.findByRole('option', { name: 'Room 4B' })
    expect((screen.getByLabelText('Camera') as HTMLSelectElement).value).toBe('room_4b')
    expect((screen.getByLabelText('Date') as HTMLInputElement).value).toBe('2026-08-01')
  })

  it('requests /api/report with the picked camera and date', async () => {
    let seen = ''
    server.use(
      http.get(`${RECORDER_BASE_URL}/report`, ({ request }) => {
        seen = request.url
        return HttpResponse.json(REAL_REPORT)
      }),
    )
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    await screen.findByText(/recorded/i)
    const url = new URL(seen)
    expect(url.searchParams.get('camera')).toBe('room_4b')
    expect(url.searchParams.get('date')).toBe('2026-08-01')
  })

  it('re-requests the report when the camera is changed', async () => {
    const seenCameras: string[] = []
    server.use(
      http.get(`${RECORDER_BASE_URL}/report`, ({ request }) => {
        seenCameras.push(new URL(request.url).searchParams.get('camera') ?? '')
        return HttpResponse.json(REAL_REPORT)
      }),
    )
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })
    await screen.findByRole('option', { name: 'Corridor 1' })

    const user = userEvent.setup()
    await user.selectOptions(screen.getByLabelText('Camera'), 'corridor_1')

    expect(seenCameras).toContain('corridor_1')
  })
})

describe('DayReportPage — coverage KPIs', () => {
  it('renders the recorded percentage, expected/gap time and segment totals from the real payload', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    await screen.findByText('recorded')
    expect(within((screen.getByText('recorded').parentElement)!).getByText('79.9%')).toBeInTheDocument()
    expect(within(screen.getByText('expected').parentElement!).getByText('12h 13m')).toBeInTheDocument()
    // "2h 19m" also appears in the gap-causes table below (same underlying total, by construction),
    // so this value is read from inside the KPI tile specifically, not just anywhere on the page.
    expect(within(screen.getByText('gap time').parentElement!).getByText('2h 19m')).toBeInTheDocument()
    const segmentsTile = screen.getByText('segments').parentElement!
    expect(within(segmentsTile).getByText('328')).toBeInTheDocument()
    expect(within(segmentsTile).getByText('5m 26s')).toBeInTheDocument()
  })

  it('flags the recorder’s own unexplained-shortfall figure instead of hiding it', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    expect(
      await screen.findByText(/recorder's own accounting gap, not this console's/i),
    ).toBeInTheDocument()
  })
})

describe('DayReportPage — integrity and gap causes', () => {
  it('breaks integrity time down by state, worst first, with escalation tones', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    const table = await screen.findByRole('table', { name: /Time spent in each integrity state/ })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    // UNDERLIT (26383s) outranks OBSTRUCTED (17646s).
    expect(within(rows[0]!).getByText('UNDERLIT')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('7h 19m')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('OBSTRUCTED')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('4h 54m')).toBeInTheDocument()
  })

  it('groups every gap by cause instead of rendering one row per interval', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    const table = await screen.findByRole('table', { name: /Gap causes, worst first/ })
    const rows = within(table).getAllByRole('row').slice(1)
    // The fixture's every gap collapses to exactly one cause row (LOST), regardless
    // of how many intervals carried it — the live day has 594, this excerpt has 6.
    expect(rows).toHaveLength(1)
    expect(within(rows[0]!).getByText('LOST')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('2')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('2h 19m')).toBeInTheDocument()
    expect(
      screen.getByText(`${REAL_REPORT.intervals.length} recorded intervals carried 2 gaps`, {
        exact: false,
      }),
    ).toBeInTheDocument()
  })
})

describe('DayReportPage — blind spots', () => {
  it('renders the mask path, total masked share and every region', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    const table = await screen.findByRole('table', { name: /Masked regions/ })
    expect(screen.getByText('14.6%')).toBeInTheDocument()
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    expect(within(rows[0]!).getByText('toilet')).toBeInTheDocument()
    expect(within(rows[0]!).getByText('8.5%')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('washbasin')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('6.1%')).toBeInTheDocument()
  })

  it('says no mask is configured rather than rendering an empty table', async () => {
    server.use(
      recorderReportHandler({ ...REAL_REPORT, blind_spots: null }),
    )
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    expect(
      await screen.findByText(/No privacy mask is configured for this camera/),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: /Masked regions/ })).not.toBeInTheDocument()
  })
})

describe('DayReportPage — retrieval holes', () => {
  it('groups holes by cause and states the total instead of listing every one', async () => {
    server.use(recorderReportHandler(REAL_REPORT))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    expect(await screen.findByText(/3 gaps in what can be retrieved/)).toBeInTheDocument()
    const table = await screen.findByRole('table', { name: /Retrieval holes grouped by cause/ })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    const expiredRow = rows.find((row) => row.textContent?.includes('Expired by retention'))!
    expect(within(expiredRow).getByText('1')).toBeInTheDocument()
    expect(within(expiredRow).getByText('13h 34m')).toBeInTheDocument()
    const noSegmentRow = rows.find((row) => row.textContent?.includes('No segment'))!
    expect(within(noSegmentRow).getByText('2')).toBeInTheDocument()
  })

  it('says so when there are no retrieval holes at all', async () => {
    server.use(recorderReportHandler({ ...REAL_REPORT, retrieval_holes: [] }))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    expect(
      await screen.findByText(/No retrieval holes on this day/),
    ).toBeInTheDocument()
  })
})

describe('DayReportPage — empty and degraded coverage', () => {
  it('reports no coverage record for an unknown camera rather than a fabricated zero', async () => {
    server.use(recorderReportHandler(REAL_REPORT_UNKNOWN_CAMERA))
    renderWithProviders(<DayReportPage />, { route: '/recorder/report?camera=nope&date=2026-08-01' })

    expect(await screen.findByText('no record')).toBeInTheDocument()
    expect(screen.getByText(/nothing to account for/)).toBeInTheDocument()
    // The KPI/integrity/gap-cause/blind-spot/retrieval-hole tables never render for a
    // camera-day with no coverage record at all — only the Absent explanation does.
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('flags a fully-lost day as a real outage, not an empty report', async () => {
    server.use(recorderReportHandler(REAL_REPORT_FULLY_LOST))
    renderWithProviders(<DayReportPage />, {
      route: '/recorder/report?camera=corridor_1&date=2026-08-01',
    })

    expect(
      await screen.findByText(/recorded none of it\. This is a real outage, not an empty report/),
    ).toBeInTheDocument()
    expect(screen.getByText('0.0%')).toBeInTheDocument()
  })
})

describe('DayReportPage — recorder unavailable', () => {
  it('does not let a failed report request read as a clean day', async () => {
    server.use(recorderUnreachableHandler('/report'))
    renderWithProviders(<DayReportPage />, { route: PINNED_ROUTE })

    // `retry: 1`'s default backoff pushes the query past the default 1s wait.
    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })
})
