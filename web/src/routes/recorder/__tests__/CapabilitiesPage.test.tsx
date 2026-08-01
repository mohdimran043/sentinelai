import { describe, expect, it } from 'vitest'
import { screen, within } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { CapabilitiesPage } from '@/routes/recorder/CapabilitiesPage'
import {
  recorderCapabilitiesHandler,
  recorderErrorHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_CAPABILITIES } from '@/recorder/mocks/fixtures'

describe('CapabilitiesPage — the whole matrix', () => {
  it('renders all thirteen declared capabilities, not just the available ones', async () => {
    renderWithProviders(<CapabilitiesPage />)

    for (const capability of REAL_CAPABILITIES.capabilities) {
      expect(await screen.findByText(capability.label)).toBeInTheDocument()
    }
    // Thirteen tiles, one per capability, plus the summary readings.
    const tile = (await screen.findByText('declared')).parentElement!
    expect(within(tile).getByText('13')).toBeInTheDocument()
  })

  it('counts available and unavailable to match the real 6/7 split', async () => {
    renderWithProviders(<CapabilitiesPage />)

    const availableTile = (
      await screen.findByText('available', { selector: '.eyebrow' })
    ).parentElement!
    expect(within(availableTile).getByText('6')).toBeInTheDocument()

    const unavailableTile = screen.getByText('not available', {
      selector: '.eyebrow',
    }).parentElement!
    expect(within(unavailableTile).getByText('7')).toBeInTheDocument()
  })
})

describe('CapabilitiesPage — availability is rendered, not hidden', () => {
  it('marks an available capability with a nominal pill', async () => {
    renderWithProviders(<CapabilitiesPage />)

    const heading = await screen.findByText('Coverage & gaps')
    const tile = heading.closest('.rounded-md') as HTMLElement
    expect(within(tile).getByText('available')).toBeInTheDocument()
    expect(within(tile).queryByText('not available')).not.toBeInTheDocument()
  })

  it('marks an unavailable capability with a distinct, non-nominal pill rather than dropping it', async () => {
    renderWithProviders(<CapabilitiesPage />)

    const heading = await screen.findByText('Welfare & distress detection')
    const tile = heading.closest('.rounded-md') as HTMLElement
    expect(within(tile).getByText('not available')).toBeInTheDocument()
    expect(within(tile).queryByText(/^available$/)).not.toBeInTheDocument()
  })

  it('shows the recorder’s own reason for why a capability is unavailable', async () => {
    renderWithProviders(<CapabilitiesPage />)

    expect(
      await screen.findByText(/No perception tier is connected/),
    ).toBeInTheDocument()
    expect(screen.getByText(/Not implemented \(slice 9\)\./)).toBeInTheDocument()
  })

  it('does not invent a reason for an available capability the recorder gave none for', async () => {
    renderWithProviders(<CapabilitiesPage />)

    // "coverage" has no `reason` field in the real payload.
    await screen.findByText('Coverage & gaps')
    const explainedCount = REAL_CAPABILITIES.capabilities.filter((c) => c.reason).length
    const paragraphs = document.querySelectorAll('.muted')
    // Every reason present in the fixture is rendered, and nothing extra is fabricated
    // for the ones that have none.
    expect(paragraphs.length).toBe(explainedCount)
  })
})

describe('CapabilitiesPage — recorder unavailable', () => {
  it('does not let a failed request read as an all-clear', async () => {
    server.use(recorderUnreachableHandler('/capabilities'))
    renderWithProviders(<CapabilitiesPage />)

    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByText('Coverage & gaps')).not.toBeInTheDocument()
  })

  it('reports an error response in the recorder’s own words', async () => {
    server.use(recorderErrorHandler('/capabilities', 503, 'capability registry unavailable'))
    renderWithProviders(<CapabilitiesPage />)

    expect(
      await screen.findByText(/capability registry unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })

  it('handles a hypothetical all-available response without dropping the count reading', async () => {
    server.use(
      recorderCapabilitiesHandler({
        capabilities: REAL_CAPABILITIES.capabilities.map((c) => ({ ...c, available: true })),
      }),
    )
    renderWithProviders(<CapabilitiesPage />)

    const unavailableTile = (await screen.findByText('not available')).parentElement!
    expect(within(unavailableTile).getByText('0')).toBeInTheDocument()
    expect(screen.queryByText('breach')).not.toBeInTheDocument()
  })
})
