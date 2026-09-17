import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AlertsPage } from '@/routes/AlertsPage'
import * as engineClient from '@/api/engineClient'
import type { AlertEntry, AlertsResponse } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, listAlerts: vi.fn() }
})

const listAlerts = vi.mocked(engineClient.listAlerts)

let nextId = 0

function alert(overrides: Partial<AlertEntry> = {}): AlertEntry {
  nextId += 1
  return {
    alert_id: `00000000-0000-0000-0000-${String(nextId).padStart(12, '0')}`,
    camera_id: 'linkou',
    camera_label: 'Linkou Old Street',
    zone: null,
    reason: 'zone_intrusion',
    subject: '1',
    state: 'active',
    severity: 'medium',
    priority: 'medium',
    first_seen: 1_700_000_000,
    last_seen: 1_700_000_000,
    occurrences: 1,
    description: 'a person crossed into a restricted area',
    event_ids: [],
    clip_uri: null,
    notify_clip_uri: null,
    acknowledged_by: null,
    acknowledged_at: null,
    ...overrides,
  }
}

function respondWith(alerts: AlertEntry[]): void {
  const response: AlertsResponse = {
    alerts,
    open_count: alerts.filter((a) => a.state !== 'resolved').length,
  }
  listAlerts.mockResolvedValue(response)
}

const ROSTER = [
  alert({ camera_id: 'linkou', camera_label: 'Linkou', severity: 'critical' }),
  alert({ camera_id: 'linkou', camera_label: 'Linkou', severity: 'high' }),
  alert({ camera_id: 'abbeyroad', camera_label: 'Abbey Road', severity: 'high' }),
  alert({ camera_id: 'abbeyroad', camera_label: 'Abbey Road', severity: 'medium' }),
  alert({ camera_id: 'bourbonstreet', camera_label: 'Bourbon Street', severity: 'low' }),
]

function rows(): HTMLElement[] {
  return screen.queryAllByRole('article')
}

async function renderPage(alerts: AlertEntry[] = ROSTER) {
  respondWith(alerts)
  const view = renderWithProviders(<AlertsPage />)
  await waitFor(() => expect(screen.getByLabelText('Camera')).toBeInTheDocument())
  return view
}

beforeEach(() => {
  listAlerts.mockReset()
})

describe('AlertsPage filters, narrowing the list', () => {
  it('opens showing everything, hiding nothing the operator has not hidden', async () => {
    await renderPage()

    expect(rows()).toHaveLength(5)
    expect(screen.getByTestId('alert-filter-summary')).toHaveTextContent('5 alerts')
  })

  it('narrows to one camera', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'abbeyroad')

    expect(rows()).toHaveLength(2)
    for (const row of rows()) {
      expect(within(row).getByRole('link')).toHaveTextContent('Abbey Road')
    }
  })

  it('narrows to one severity band', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Severity'), 'high')

    expect(rows()).toHaveLength(2)
  })

  it('combines camera and severity rather than replacing one with the other', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'linkou')
    await user.selectOptions(screen.getByLabelText('Severity'), 'critical')

    expect(rows()).toHaveLength(1)
  })

  it('says how much of the list it is showing once anything is narrowed', async () => {
    // A filtered page and a quiet site look identical. This is the line that keeps an
    // operator from reading one as the other.
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'bourbonstreet')

    expect(screen.getByTestId('alert-filter-summary')).toHaveTextContent('1 of 5 alerts')
  })

  it('restores the whole list from one control', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'linkou')
    await user.selectOptions(screen.getByLabelText('Severity'), 'high')
    await user.click(screen.getByRole('button', { name: /clear filters/i }))

    expect(rows()).toHaveLength(5)
    expect(screen.queryByRole('button', { name: /clear filters/i })).not.toBeInTheDocument()
  })

  it('offers no clear control on an unfiltered page', async () => {
    await renderPage()

    expect(screen.queryByRole('button', { name: /clear filters/i })).not.toBeInTheDocument()
  })
})

describe('AlertsPage filters, the counts on each option', () => {
  it('counts each camera', async () => {
    await renderPage()

    const camera = screen.getByLabelText('Camera')
    expect(within(camera).getByRole('option', { name: 'Linkou — 2' })).toBeInTheDocument()
    expect(within(camera).getByRole('option', { name: 'Abbey Road — 2' })).toBeInTheDocument()
  })

  it('shows a severity band that is present and one that is not', async () => {
    // "Critical — 0" is the answer to "is anything critical?", and an option that
    // simply was not rendered would answer it only by absence.
    await renderPage([alert({ severity: 'low' })])

    const severity = screen.getByLabelText('Severity')
    expect(within(severity).getByRole('option', { name: 'Low — 1' })).toBeInTheDocument()
    expect(within(severity).getByRole('option', { name: 'Critical — 0' })).toBeInTheDocument()
  })

  it('re-counts the severities against the chosen camera', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'linkou')

    const severity = screen.getByLabelText('Severity')
    expect(within(severity).getByRole('option', { name: 'Critical — 1' })).toBeInTheDocument()
    // Linkou has no low-severity alert, though the site as a whole does.
    expect(within(severity).getByRole('option', { name: 'Low — 0' })).toBeInTheDocument()
  })

  it('keeps every camera countable once one is chosen', async () => {
    // Counting the fully-filtered set would read 0 against every other camera and
    // make the control useless for moving between them mid-incident.
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'linkou')

    expect(
      within(screen.getByLabelText('Camera')).getByRole('option', { name: 'Abbey Road — 2' }),
    ).toBeInTheDocument()
  })
})

describe('AlertsPage filters, what an empty list is allowed to claim', () => {
  it('does not say nothing is waiting when the filters are what emptied it', async () => {
    // The failure this guards: an operator narrows to one camera, reads "Nothing is
    // waiting for you", and walks away from a site with an open critical alert.
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Camera'), 'bourbonstreet')
    await user.selectOptions(screen.getByLabelText('Severity'), 'critical')

    expect(rows()).toHaveLength(0)
    expect(screen.getByText(/no alerts match these filters/i)).toBeInTheDocument()
    expect(screen.queryByText(/nothing is waiting for you/i)).not.toBeInTheDocument()
    expect(screen.getByText(/outside the camera or severity you picked/i)).toBeInTheDocument()
  })

  it('offers a way back from an empty filtered list', async () => {
    const user = userEvent.setup()
    await renderPage()

    await user.selectOptions(screen.getByLabelText('Severity'), 'critical')
    await user.selectOptions(screen.getByLabelText('Camera'), 'bourbonstreet')
    // Two of them: one in the bar, one inside the empty state where the eye already is.
    await user.click(screen.getAllByRole('button', { name: /clear filters/i })[1]!)

    expect(rows()).toHaveLength(5)
  })

  it('does not claim everything was dealt with when nothing was ever raised', async () => {
    // The two readings of an empty list are "all handled" and "none yet", and the
    // reassuring one must not be shown for the other. An engine that has just started
    // has raised nothing; saying it has all been dealt with invents work nobody did.
    respondWith([])
    renderWithProviders(<AlertsPage />)
    await waitFor(() => expect(screen.getByText(/no alerts yet/i)).toBeInTheDocument())

    expect(screen.queryByText(/dealt with/i)).not.toBeInTheDocument()
  })

  it('still says nothing is waiting when that is the truth', async () => {
    await renderPage([alert({ state: 'resolved' })])

    expect(screen.getByText(/nothing is waiting for you/i)).toBeInTheDocument()
  })

  it('offers no filters at all when the engine has raised nothing', async () => {
    // Facet counts over an empty list are a row of zeroes that answers nothing.
    respondWith([])
    renderWithProviders(<AlertsPage />)
    await waitFor(() => expect(screen.getByText(/no alerts yet/i)).toBeInTheDocument())

    expect(screen.queryByLabelText('Camera')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Severity')).not.toBeInTheDocument()
  })
})

describe('AlertsPage filters, severity is not priority', () => {
  it('filters on severity even where the row badge disagrees', async () => {
    const user = userEvent.setup()
    await renderPage([
      alert({ severity: 'critical', priority: 'low' }),
      alert({ severity: 'low', priority: 'critical' }),
    ])

    await user.selectOptions(screen.getByLabelText('Severity'), 'critical')

    expect(rows()).toHaveLength(1)
    // The badge on the surviving row reads "low" — its priority. That is not a bug,
    // and the row prints its severity so the operator can see why it matched.
    expect(within(rows()[0]!).getByText('severity critical')).toBeInTheDocument()
  })
})
