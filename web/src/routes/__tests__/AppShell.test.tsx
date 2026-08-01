import { describe, expect, it } from 'vitest'
import { Route, Routes } from 'react-router-dom'
import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AppShell } from '@/routes/AppShell'
import { SectionNotBuiltPage } from '@/routes/recorder/SectionNotBuiltPage'
import { RECORDER_SECTIONS } from '@/routes/recorder/sections'
import { useSessionStore } from '@/store/session'

function renderShell(route: string) {
  useSessionStore.setState({ operatorName: 'Ada' })
  return renderWithProviders(
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<h1>Dashboard</h1>} />
        {/* Stands in for whichever built screen the real router mounts here. */}
        <Route path="/recorder/alerts" element={<h1>Alerts screen</h1>} />
        <Route path="/recorder/:section" element={<SectionNotBuiltPage />} />
      </Route>
    </Routes>,
    { route },
  )
}

describe('AppShell navigation', () => {
  it('lists every recorder section from the parity checklist, grouped away from the engine', () => {
    renderShell('/dashboard')

    const rail = screen.getByRole('navigation', { name: 'Sections' })
    expect(within(rail).getByText('AI engine')).toBeInTheDocument()
    expect(within(rail).getByText('Recorder')).toBeInTheDocument()

    for (const section of RECORDER_SECTIONS) {
      const link = within(rail).getByRole('link', { name: new RegExp(`^${section.label}`) })
      expect(link).toHaveAttribute('href', `/recorder/${section.path}`)
    }
  })

  it('marks the sections whose screens are not built yet, and only those', () => {
    renderShell('/dashboard')

    const rail = screen.getByRole('navigation', { name: 'Sections' })
    const expectedPending = RECORDER_SECTIONS.filter((section) => !section.built).length
    expect(within(rail).getAllByText('soon')).toHaveLength(expectedPending)

    for (const section of RECORDER_SECTIONS) {
      const link = within(rail).getByRole('link', { name: new RegExp(`^${section.label}`) })
      const marked = within(link).queryByText('soon') !== null
      expect(marked).toBe(!section.built)
    }
  })

  it('marks the current section for assistive tech', () => {
    renderShell('/recorder/storage')

    const current = screen.getByRole('link', { current: 'page' })
    expect(current).toHaveAttribute('href', '/recorder/storage')
  })

  it('navigates to a recorder section without leaving the shell', async () => {
    const user = userEvent.setup()
    renderShell('/dashboard')

    await user.click(screen.getByRole('link', { name: /^Capabilities/ }))

    expect(screen.getByRole('heading', { level: 1, name: 'Capabilities' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Sections' })).toBeInTheDocument()
  })
})

describe('SectionNotBuiltPage', () => {
  it('says the screen is not built here, rather than implying the recorder is empty', () => {
    renderShell('/recorder/footage')

    expect(screen.getByText('not built here')).toBeInTheDocument()
    expect(screen.getByText(/not built in this console yet/i)).toBeInTheDocument()
    expect(screen.getByText(/nothing below is a report that the recorder is empty/i)).toBeInTheDocument()
  })

  it('names the endpoint the section will read', () => {
    renderShell('/recorder/notifications')

    expect(screen.getByText('GET /api/notifications')).toBeInTheDocument()
  })

  it('redirects an unknown section to a real screen instead of rendering a blank page', () => {
    renderShell('/recorder/not-a-section')

    expect(screen.getByRole('heading', { level: 1, name: 'Alerts screen' })).toBeInTheDocument()
    expect(screen.queryByText('not built here')).not.toBeInTheDocument()
  })
})
