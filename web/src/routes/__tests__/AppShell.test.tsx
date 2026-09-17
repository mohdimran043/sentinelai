import { describe, expect, it } from 'vitest'
import { Route, Routes, createRoutesFromElements, matchRoutes } from 'react-router-dom'
import { screen, within } from '@testing-library/react'
import { appRoutes } from '@/App'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AppShell } from '@/routes/AppShell'
import { SectionNotBuiltPage } from '@/routes/recorder/SectionNotBuiltPage'
import {
  RECORDER_LANDING_PATH,
  RECORDER_SECTIONS,
  findRecorderSection,
} from '@/routes/recorder/sections'
import { useSessionStore } from '@/store/session'

function renderShell(route: string) {
  useSessionStore.setState({ operatorName: 'Ada' })
  return renderWithProviders(
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<h1>Dashboard</h1>} />
        {/* Stands in for the built screen behind the real router's `/recorder/report`.
            It must be a path `appRoutes` actually answers: a stub at a route the app
            does not have masks a redirect that loops in production — which is exactly
            what `/recorder/alerts` did here once that section was removed. */}
        <Route path="/recorder/report" element={<h1>Day report screen</h1>} />
        <Route path="/recorder/:section" element={<SectionNotBuiltPage />} />
      </Route>
    </Routes>,
    { route },
  )
}

describe('AppShell navigation', () => {
  it('lists every recorder section under the engine heading, not a group of its own', () => {
    renderShell('/dashboard')

    const rail = screen.getByRole('navigation', { name: 'Sections' })
    expect(within(rail).getByText('AI engine')).toBeInTheDocument()
    // The recorder sections that survived have no engine equivalent, so the
    // second heading was separating screens an operator moves between rather
    // than screens they choose between. One group, one heading.
    expect(within(rail).queryByText('Recorder')).not.toBeInTheDocument()

    for (const section of RECORDER_SECTIONS) {
      const link = within(rail).getByRole('link', { name: new RegExp(`^${section.label}`) })
      expect(link).toHaveAttribute('href', `/recorder/${section.path}`)
    }
  })

  it('points every link at a route the real router answers', () => {
    renderShell('/dashboard')

    const rail = screen.getByRole('navigation', { name: 'Sections' })
    const hrefs = within(rail)
      .getAllByRole('link')
      .map((link) => link.getAttribute('href'))

    // Guards against the rail, not the router: a link with no route behind it
    // does not 404 here, it quietly renders something else. `*` sends the
    // operator to the dashboard and `/recorder/:section` renders the
    // "not built" placeholder, so both read as a working link that went to the
    // wrong place. Matched against the real table in `App.tsx`.
    expect(hrefs.length).toBeGreaterThan(0)
    const table = createRoutesFromElements(appRoutes)

    for (const href of hrefs) {
      const matched = matchRoutes(table, href ?? '')
      expect(matched, `no route matches ${href}`).not.toBeNull()

      const leaf = matched![matched!.length - 1].route.path
      expect(leaf, `${href} falls through to the catch-all`).not.toBe('*')
      expect(leaf, `${href} has no screen of its own`).not.toBe('/recorder/:section')
    }
  })

  it('marks the sections whose screens are not built yet, and only those', () => {
    renderShell('/dashboard')

    const rail = screen.getByRole('navigation', { name: 'Sections' })
    const expectedPending = RECORDER_SECTIONS.filter((section) => !section.built).length
    // `getAllByText` throws on zero matches rather than returning `[]`, and the
    // parity checklist can legitimately reach "every section built" — so the
    // zero case has to be asserted as an absence, not as a call that can never
    // return an empty array.
    if (expectedPending === 0) {
      expect(within(rail).queryByText('soon')).not.toBeInTheDocument()
    } else {
      expect(within(rail).getAllByText('soon')).toHaveLength(expectedPending)
    }

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

    // "Capabilities" used to be here and is gone: the engine's per-camera capability
    // panel answers the same question and can also change the answer. Storage has no
    // engine equivalent, which is the test for whether a recorder link earns its place.
    await user.click(screen.getByRole('link', { name: /^Storage/ }))

    expect(screen.getByRole('heading', { level: 1, name: /storage/i })).toBeInTheDocument()
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


  it('sends a removed section somewhere the real router answers, not back to itself', () => {
    // The bug this exists to prevent: `SectionNotBuiltPage` redirects an unrecognised
    // section somewhere, and if that somewhere is itself a section the console no
    // longer has, the redirect lands back here and fires again. Asserted against
    // `appRoutes` rather than this file's stubs, because a stub route is exactly what
    // hid it — a page can only be a safe redirect target if the real app mounts it.
    const target = `/recorder/${RECORDER_LANDING_PATH}`
    const leaf = matchRoutes(createRoutesFromElements(appRoutes), target)?.at(-1)?.route

    expect(leaf).toBeDefined()
    // Not the `:section` placeholder — landing there is what makes the redirect loop.
    expect(leaf?.path).toBe(target)
    expect(findRecorderSection(RECORDER_LANDING_PATH)).toBeDefined()
  })

  it('redirects an unknown section to a real screen instead of rendering a blank page', () => {
    renderShell('/recorder/not-a-section')

    expect(
      screen.getByRole('heading', { level: 1, name: 'Day report screen' }),
    ).toBeInTheDocument()
    expect(screen.queryByText('not built here')).not.toBeInTheDocument()
  })
})
