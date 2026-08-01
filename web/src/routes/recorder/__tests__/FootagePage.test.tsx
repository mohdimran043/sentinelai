import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { FootagePage } from '@/routes/recorder/FootagePage'
import {
  recorderCamerasHandler,
  recorderJournalDayHandler,
  recorderJournalHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import {
  REAL_JOURNAL_DAY_HISTORY,
  REAL_JOURNAL_DAY_NO_CHAPTER,
  REAL_JOURNAL_DAY_SEALED,
  REAL_JOURNAL_ROOM_4B,
} from '@/recorder/mocks/fixtures'

async function journalTable() {
  return screen.findByRole('table', { name: /Daily journal/ })
}

describe('FootagePage — camera and snapshot', () => {
  it('defaults to the first camera in the registry and requests a cache-busted snapshot', async () => {
    renderWithProviders(<FootagePage />)

    const picker = await screen.findByLabelText('Camera')
    expect((picker as HTMLSelectElement).value).toBe('room_4b')

    const image = await screen.findByAltText(/Most recent available frame from room_4b/)
    const src = (image as HTMLImageElement).src
    expect(src).toContain('/snapshot/room_4b')
    expect(src).toMatch(/[?&]t=\d+/)
  })

  it('says so, rather than showing a blank frame, when the recorder cannot produce a snapshot', async () => {
    renderWithProviders(<FootagePage />)
    const image = await screen.findByAltText(/Most recent available frame from room_4b/)

    // jsdom does not actually fetch <img src>, so the failure path is driven directly.
    image.dispatchEvent(new Event('error'))

    expect(
      await screen.findByText(/could not produce a frame for room_4b just now/i),
    ).toBeInTheDocument()
    expect(screen.queryByAltText(/Most recent available frame/)).not.toBeInTheDocument()
  })

  it('reports no cameras registered rather than an empty picker', async () => {
    server.use(recorderCamerasHandler({ cameras: [], context_note: '' }))
    renderWithProviders(<FootagePage />)

    expect(await screen.findByText('No cameras registered')).toBeInTheDocument()
    expect(screen.queryByLabelText('Camera')).not.toBeInTheDocument()
  })
})

describe('FootagePage — journal', () => {
  it('lists every sealed day with its status and revision count', async () => {
    renderWithProviders(<FootagePage />)

    const table = await journalTable()
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(REAL_JOURNAL_ROOM_4B.days.length)

    const first = within(rows[0]!).getAllByRole('cell')
    expect(first[0]).toHaveTextContent('2026-07-31')
    expect(within(rows[0]!).getByText('sealed')).toBeInTheDocument()
    expect(first[2]).toHaveTextContent('2')
  })

  it('says so when the journal has never been chaptered for this camera', async () => {
    server.use(recorderJournalHandler({ camera_id: 'room_4b', days: [], note: '' }))
    renderWithProviders(<FootagePage />)

    expect(
      await screen.findByText('No day has ever been sealed for this camera'),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: /Daily journal/ })).not.toBeInTheDocument()
  })

  it('does not let a failed journal request pass as an empty one', async () => {
    server.use(recorderUnreachableHandler('/journal/room_4b'))
    renderWithProviders(<FootagePage />)

    // `retry: 1`'s default backoff pushes the query past the default 1s wait.
    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('table', { name: /Daily journal/ })).not.toBeInTheDocument()
  })
})

describe('FootagePage — day detail and provenance', () => {
  it('auto-selects the most recent day and shows its seal, revision and lineage', async () => {
    renderWithProviders(<FootagePage />)

    expect(await screen.findByText('day complete; sealed from the coverage journal and segment index while the evidence behind it still existed')).toBeInTheDocument()
    expect(screen.getByText(/revision 2/)).toBeInTheDocument()
    expect(screen.getByText(/supersedes revision 1/)).toBeInTheDocument()
    expect(
      screen.getByText('b326bb2ffc3de165018964e33956d4019b122c8f0208185961e532654ceefdc3'),
    ).toBeInTheDocument()
  })

  it('says a day has not been written yet, without inventing a revision for it', async () => {
    server.use(recorderJournalDayHandler(REAL_JOURNAL_DAY_NO_CHAPTER))
    renderWithProviders(<FootagePage />)

    expect(await screen.findByText('not yet written')).toBeInTheDocument()
    expect(screen.getByText(/No chapter has been written for this day/)).toBeInTheDocument()
    expect(screen.queryByText(/revision \d/)).not.toBeInTheDocument()
  })

  it('shows the full revision history, marking which one is current, only once asked for', async () => {
    server.use(recorderJournalDayHandler(REAL_JOURNAL_DAY_SEALED, REAL_JOURNAL_DAY_HISTORY))
    renderWithProviders(<FootagePage />)

    await screen.findByText(/revision 2/)
    expect(screen.queryByText('Revision history')).not.toBeInTheDocument()

    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Show revision history/ }))

    const table = await screen.findByRole('table', {
      name: /Every revision written for 2026-07-31/,
    })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    expect(within(rows[0]!).getByText('provisional')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('sealed')).toBeInTheDocument()
    expect(within(rows[1]!).getByText('current')).toBeInTheDocument()
    expect(within(rows[0]!).queryByText('current')).not.toBeInTheDocument()
  })

  it('links to the day report for the same camera and date', async () => {
    renderWithProviders(<FootagePage />)
    await screen.findByText(/revision 2/)

    const link = screen.getByRole('link', { name: 'Day report' })
    expect(link).toHaveAttribute('href', '/recorder/report?camera=room_4b&date=2026-07-31')
  })
})
