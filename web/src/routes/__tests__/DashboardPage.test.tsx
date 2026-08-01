import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { DashboardPage } from '@/routes/DashboardPage'
import * as engineClient from '@/api/engineClient'
import type { CamerasResponse, HealthResponse } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return {
    ...actual,
    getHealth: vi.fn(),
    listCameras: vi.fn(),
  }
})

const getHealth = vi.mocked(engineClient.getHealth)
const listCameras = vi.mocked(engineClient.listCameras)

const sampleHealth: HealthResponse = {
  status: 'ok',
  models: [
    { key: 'yolo11s', state: 'loaded', detail: 'ready', vram_mib: 512 },
    { key: 'qwen2.5-vl-3b-awq', state: 'sleeping', detail: 'idle-unloaded', vram_mib: 0 },
  ],
}

const sampleCameras: CamerasResponse = {
  cameras: [
    {
      camera_id: 'avenue_01',
      frames_seen: 1000,
      frames_dropped: 2,
      detections_run: 900,
      escalations: 3,
      escalations_dropped: 0,
      discontinuities: 0,
      last_frame_at: Date.now() / 1000,
      last_escalation_at: null,
    },
  ],
}

describe('DashboardPage', () => {
  beforeEach(() => {
    getHealth.mockReset()
    listCameras.mockReset()
  })

  it('renders live camera and model data when the engine answers', async () => {
    getHealth.mockResolvedValue(sampleHealth)
    listCameras.mockResolvedValue(sampleCameras)

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText('avenue_01')).toBeInTheDocument()
    expect(screen.getByText('yolo11s')).toBeInTheDocument()
    expect(screen.getByText('Awaiting Phase 1C')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Storage usage' })).toBeInTheDocument()
  })

  it('degrades honestly instead of showing zeros when the engine is unreachable', async () => {
    getHealth.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))
    listCameras.mockRejectedValue(new engineClient.EngineUnreachableError(new Error('refused')))

    renderWithProviders(<DashboardPage />)

    await waitFor(
      () => {
        expect(screen.getByRole('alert')).toHaveTextContent(/the ai engine is unreachable/i)
      },
      { timeout: 4000 },
    )
    expect(screen.queryByText('avenue_01')).not.toBeInTheDocument()
    // No fabricated numeric tiles for cameras/health while unreachable.
    expect(screen.queryByText(/^0$/)).not.toBeInTheDocument()
  })

  it('invites the operator to act on a genuinely empty camera list', async () => {
    getHealth.mockResolvedValue({ status: 'ok', models: [] })
    listCameras.mockResolvedValue({ cameras: [] })

    renderWithProviders(<DashboardPage />)

    expect(await screen.findByText(/no cameras configured/i)).toBeInTheDocument()
  })
})
