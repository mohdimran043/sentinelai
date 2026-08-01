import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router-dom'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraPage } from '@/routes/CameraPage'
import * as engineClient from '@/api/engineClient'
import type { CameraStatus, DescribeResponse } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return {
    ...actual,
    getCameraTelemetry: vi.fn(),
    describeCameraNow: vi.fn(),
  }
})

const getCameraTelemetry = vi.mocked(engineClient.getCameraTelemetry)
const describeCameraNow = vi.mocked(engineClient.describeCameraNow)

const telemetry: CameraStatus = {
  camera_id: 'avenue_01',
  frames_seen: 5000,
  frames_dropped: 10,
  detections_run: 4800,
  escalations: 7,
  escalations_dropped: 1,
  discontinuities: 0,
  last_frame_at: Date.now() / 1000,
  last_escalation_at: Date.now() / 1000 - 120,
}

function renderCameraPage() {
  return renderWithProviders(
    <Routes>
      <Route path="/cameras/:cameraId" element={<CameraPage />} />
    </Routes>,
    { route: '/cameras/avenue_01' },
  )
}

describe('CameraPage', () => {
  beforeEach(() => {
    getCameraTelemetry.mockReset()
    describeCameraNow.mockReset()
  })

  it('shows live telemetry and the mocked event feed', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)

    renderCameraPage()

    expect(await screen.findByText('Frames seen')).toBeInTheDocument()
    expect(screen.getByText('5,000')).toBeInTheDocument()
    // Event feed is mocked via MSW and should render something for this camera.
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /event feed/i })).toBeInTheDocument()
    })
  })

  it('calls describe-now and reports the queued event id', async () => {
    getCameraTelemetry.mockResolvedValue(telemetry)
    const response: DescribeResponse = { event_id: '22222222-2222-4222-8222-222222222222' }
    describeCameraNow.mockResolvedValue(response)
    const user = userEvent.setup()

    renderCameraPage()
    await screen.findByText('Frames seen')

    await user.click(screen.getByRole('button', { name: /describe now/i }))

    expect(await screen.findByTestId('describe-feedback')).toHaveTextContent(/queued/i)
    expect(describeCameraNow).toHaveBeenCalledWith('avenue_01')
  })

  it('reports telemetry unreachable honestly rather than blank tiles', async () => {
    getCameraTelemetry.mockRejectedValue(
      new engineClient.EngineUnreachableError(new Error('refused')),
    )

    renderCameraPage()

    expect(await screen.findByRole('alert', {}, { timeout: 4000 })).toHaveTextContent(
      /telemetry unreachable/i,
    )
    expect(screen.queryByText('Frames seen')).not.toBeInTheDocument()
  })
})
