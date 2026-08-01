import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/routes/AppShell'
import { LoginPage } from '@/routes/LoginPage'
import { DashboardPage } from '@/routes/DashboardPage'
import { CameraPage } from '@/routes/CameraPage'
import { RequireSession } from '@/routes/RequireSession'
import { SectionNotBuiltPage } from '@/routes/recorder/SectionNotBuiltPage'

/**
 * Two products, two route namespaces.
 *
 * `/dashboard` and `/cameras/*` read the AI engine (`src/api/`). Everything
 * under `/recorder/*` reads the recorder appliance (`src/recorder/`). The
 * prefix is not decoration — it is how an operator, and a bug report, can tell
 * which system a screen was talking to.
 */
export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          element={
            <RequireSession>
              <AppShell />
            </RequireSession>
          }
        >
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/cameras/:cameraId" element={<CameraPage />} />

          <Route path="/recorder" element={<Navigate to="/recorder/alerts" replace />} />
          {/* Sections with a built screen get their own route above this one.
              Everything else falls through to a page that says so out loud. */}
          <Route path="/recorder/:section" element={<SectionNotBuiltPage />} />
        </Route>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
