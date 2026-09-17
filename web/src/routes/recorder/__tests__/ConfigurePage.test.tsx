import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { ConfigurePage } from '@/routes/recorder/ConfigurePage'
import {
  recorderErrorHandler,
  recorderSettingsHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_SETTINGS } from '@/recorder/mocks/fixtures'
import type { RecorderSettingsResponse } from '@/recorder/recorder.types'

function withSettings(patch: Partial<RecorderSettingsResponse>) {
  server.use(
    recorderSettingsHandler({
      ...REAL_SETTINGS,
      ...patch,
      settings: { ...REAL_SETTINGS.settings, ...patch.settings },
    }),
  )
}

describe('ConfigurePage — wired fact', () => {
  it('says plainly that the setting decides nothing when wired is false', async () => {
    renderWithProviders(<ConfigurePage />)

    expect(await screen.findByText('This setting decides nothing today')).toBeInTheDocument()
    expect(screen.getByText(/wired: false/)).toBeInTheDocument()
  })

  it('does not claim the setting is inert once the recorder reports it wired', async () => {
    withSettings({ wired: true })
    renderWithProviders(<ConfigurePage />)

    expect(await screen.findByText('This setting is wired')).toBeInTheDocument()
    expect(screen.queryByText('This setting decides nothing today')).not.toBeInTheDocument()
    expect(screen.queryByText(/wired: false/)).not.toBeInTheDocument()
  })
})

describe('ConfigurePage — the online warning', () => {
  it('surfaces the recorder’s online warning verbatim', async () => {
    renderWithProviders(<ConfigurePage />)

    expect(await screen.findByText(REAL_SETTINGS.online_warning)).toBeInTheDocument()
  })

  it('shows the warning regardless of which mode is currently selected', async () => {
    withSettings({ settings: { inference_mode: 'local', online_acknowledged: false } })
    renderWithProviders(<ConfigurePage />)
    expect(await screen.findByText(REAL_SETTINGS.online_warning)).toBeInTheDocument()
  })
})

describe('ConfigurePage — recorded setting', () => {
  it('renders local mode as "On this machine"', async () => {
    renderWithProviders(<ConfigurePage />)

    expect(await screen.findByText('On this machine')).toBeInTheDocument()
    expect(screen.queryByText('Offsite')).not.toBeInTheDocument()
  })

  it('renders online mode as "Offsite"', async () => {
    withSettings({ settings: { inference_mode: 'online', online_acknowledged: true } })
    renderWithProviders(<ConfigurePage />)

    expect(await screen.findByText('Offsite')).toBeInTheDocument()
    expect(screen.queryByText('On this machine')).not.toBeInTheDocument()
  })

  it('renders "No" when offsite use has not been acknowledged', async () => {
    renderWithProviders(<ConfigurePage />)
    await screen.findByText('On this machine')

    // REAL_SETTINGS.online_acknowledged is false.
    const row = screen.getByText('Offsite use acknowledged').closest('div')!
    expect(row).toHaveTextContent('No')
    expect(row).not.toHaveTextContent('Yes')
  })

  it('renders "Yes" once offsite use is acknowledged', async () => {
    withSettings({ settings: { inference_mode: 'local', online_acknowledged: true } })
    renderWithProviders(<ConfigurePage />)

    const row = (await screen.findByText('Offsite use acknowledged')).closest('div')!
    expect(row).toHaveTextContent('Yes')
    expect(row).not.toHaveTextContent('No')
  })

  it('renders the recorder’s own note about why the setting is recorded', async () => {
    renderWithProviders(<ConfigurePage />)
    expect(await screen.findByText(REAL_SETTINGS.note)).toBeInTheDocument()
  })
})

describe('ConfigurePage — recorder unavailable', () => {
  it('does not let a failed request read as an all-clear', async () => {
    server.use(recorderUnreachableHandler('/settings'))
    renderWithProviders(<ConfigurePage />)

    expect(
      await screen.findByText(/recorder appliance is not reachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByText('This setting decides nothing today')).not.toBeInTheDocument()
  })

  it('reports an error response in the recorder’s own words', async () => {
    server.use(recorderErrorHandler('/settings', 503, 'settings store unavailable'))
    renderWithProviders(<ConfigurePage />)

    expect(
      await screen.findByText(/settings store unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
  })
})
