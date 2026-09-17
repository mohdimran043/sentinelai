import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/routes/AppShell'
import { LoginPage } from '@/routes/LoginPage'
import { DashboardPage } from '@/routes/DashboardPage'
import { CameraPage } from '@/routes/CameraPage'
import { AddCameraPage } from '@/routes/AddCameraPage'
import { SiteMapPage } from '@/routes/SiteMapPage'
// Aliased even though the recorder's own alerts screen is gone: the import
// path (`@/routes/` vs `@/routes/recorder/`) is the only thing distinguishing
// the two, and that is a one-character difference to misread when the recorder
// screen comes back.
import { AlertsPage as EngineAlertsPage } from '@/routes/AlertsPage'
import { PeoplePage } from '@/routes/PeoplePage'
import { AiSystemPage } from '@/routes/AiSystemPage'
import { RequireSession } from '@/routes/RequireSession'
import { SectionNotBuiltPage } from '@/routes/recorder/SectionNotBuiltPage'
import { RECORDER_LANDING_PATH } from '@/routes/recorder/sections'
import { ConfigurePage } from '@/routes/recorder/ConfigurePage'
import { StoragePage } from '@/routes/recorder/StoragePage'
import { FootagePage } from '@/routes/recorder/FootagePage'
import { DayReportPage } from '@/routes/recorder/DayReportPage'

/**
 * Two products, two route namespaces — and one table that answers for both.
 *
 * `/dashboard` and `/cameras/*` read the AI engine (`src/api/`). Everything
 * under `/recorder/*` reads the recorder appliance (`src/recorder/`). The
 * prefix is not decoration — it is how an operator, and a bug report, can tell
 * which system a screen was talking to. It stayed when the rail stopped
 * grouping the recorder separately (`routes/AppShell.tsx`): the operator no
 * longer has to know, but the URL still says.
 *
 * `appRoutes` is exported separately from the `BrowserRouter` that mounts it
 * because a nav link is only "working" if something here answers it, and that
 * is a property of this table rather than of the rail pointing at it.
 * `routes/__tests__/AppShell.test.tsx` puts every link the rail renders
 * through `createRoutesFromElements(appRoutes)` and fails if one lands on the
 * `*` catch-all (a dead link, silently redirected to the dashboard) or on the
 * `/recorder/:section` placeholder (a section the rail offers with no screen
 * behind it). Neither looks broken from the outside — one renders the
 * dashboard, the other renders a page that explains itself — which is exactly
 * why they need a test rather than an eye.
 */
export const appRoutes = (
  <>
    <Route path="/login" element={<LoginPage />} />
    <Route
      element={
        <RequireSession>
          <AppShell />
        </RequireSession>
      }
    >
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/site-map" element={<SiteMapPage />} />
      <Route path="/alerts" element={<EngineAlertsPage />} />
      <Route path="/people" element={<PeoplePage />} />
      <Route path="/ai-system" element={<AiSystemPage />} />
      {/* Before the `:cameraId` route: "new" would otherwise be read as a camera
          id and 404 against an engine that has no camera called "new". */}
      <Route path="/cameras/new" element={<AddCameraPage />} />
      <Route path="/cameras/:cameraId" element={<CameraPage />} />

      {/* The recorder's own landing was its alerts screen, which the engine's
          alert queue supersedes. Day report is the first thing it still answers
          that nothing else does. */}
      <Route
        path="/recorder"
        element={<Navigate to={`/recorder/${RECORDER_LANDING_PATH}`} replace />}
      />
      <Route path="/recorder/configure" element={<ConfigurePage />} />
      <Route path="/recorder/storage" element={<StoragePage />} />
      <Route path="/recorder/footage" element={<FootagePage />} />
      <Route path="/recorder/report" element={<DayReportPage />} />
      {/* Sections with a built screen get their own route above this one.
          Everything else falls through to a page that says so out loud. */}
      <Route path="/recorder/:section" element={<SectionNotBuiltPage />} />
    </Route>
    <Route path="/" element={<Navigate to="/dashboard" replace />} />
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </>
)

export function App() {
  return (
    <BrowserRouter>
      <Routes>{appRoutes}</Routes>
    </BrowserRouter>
  )
}
