import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AddCameraPage } from '@/routes/AddCameraPage'
import * as engineClient from '@/api/engineClient'
import type { CamerasResponse, ProbeResponse } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return {
    ...actual,
    listCameras: vi.fn(),
    probeSource: vi.fn(),
    createCamera: vi.fn(),
  }
})

const listCameras = vi.mocked(engineClient.listCameras)
const probeSource = vi.mocked(engineClient.probeSource)
const createCamera = vi.mocked(engineClient.createCamera)

const A_PLAYABLE_STREAM: ProbeResponse = {
  ok: true,
  detail: 'decoded a frame',
  source_kind: 'earthcam',
  title: 'Linkou Old Street Cam',
  width: 1920,
  height: 1080,
  codec: 'h264',
  fps: 15,
  thumbnail: 'data:image/jpeg;base64,AAAA',
}

function cameraList(ids: string[] = [], writable = true): CamerasResponse {
  return {
    config_writable: writable,
    cameras: ids.map((camera_id) => ({
      camera_id,
      label: camera_id,
      frames_seen: 0,
      frames_dropped: 0,
      detections_run: 0,
      escalations: 0,
      escalations_dropped: 0,
      discontinuities: 0,
      last_frame_at: null,
      last_frame_epoch: null,
      last_escalation_at: null,
      capabilities: [],
      falls_suspected: 0,
      notify_on: [],
      notify_min_confidence: 'likely',
      clip_preroll_seconds: null,
      clip_postroll_seconds: null,
      summary_interval_seconds: null,
    })),
  }
}

beforeEach(() => {
  listCameras.mockReset()
  probeSource.mockReset()
  createCamera.mockReset()
  listCameras.mockResolvedValue(cameraList())
})

describe('AddCameraPage, before a stream has been proved', () => {
  it('asks for the stream and nothing else', async () => {
    renderWithProviders(<AddCameraPage />)

    expect(await screen.findByLabelText(/stream url/i)).toBeInTheDocument()
    // The rest of the form is gated on a decoded frame. A camera added with a wrong URL
    // fails silently — it appears in the list and never delivers a frame — so the one
    // field worth proving first is proved first.
    expect(screen.queryByText(/camera id/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /add camera/i })).not.toBeInTheDocument()
  })

  it('will not probe an empty URL', async () => {
    renderWithProviders(<AddCameraPage />)
    expect(await screen.findByRole('button', { name: /preview/i })).toBeDisabled()
  })

  it('shows why a URL did not play, and still does not open the form', async () => {
    probeSource.mockResolvedValue({
      ok: false,
      detail: 'ConnectionRefusedError: [Errno 111] Connection refused',
      source_kind: 'rtsp',
      title: '',
      width: 0,
      height: 0,
      codec: '',
      fps: 0,
      thumbnail: null,
    })
    const user = userEvent.setup()
    renderWithProviders(<AddCameraPage />)

    await user.type(await screen.findByLabelText(/stream url/i), 'rtsp://nope/1')
    await user.click(screen.getByRole('button', { name: /preview/i }))

    expect(await screen.findByText(/Connection refused/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /add camera/i })).not.toBeInTheDocument()
  })
})

describe('AddCameraPage, once a frame has been decoded', () => {
  async function probeSuccessfully() {
    probeSource.mockResolvedValue(A_PLAYABLE_STREAM)
    const user = userEvent.setup()
    renderWithProviders(<AddCameraPage />)
    await user.type(
      await screen.findByLabelText(/stream url/i),
      'https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/',
    )
    await user.click(screen.getByRole('button', { name: /preview/i }))
    await screen.findByRole('button', { name: /add camera/i })
    return user
  }

  it('shows the frame it decoded, not just the numbers', async () => {
    await probeSuccessfully()

    // The whole point of a preview: a stream that opens and decodes green is identical
    // to a working one in every other field on this panel.
    const frame = screen.getByAltText(/one frame decoded/i)
    expect(frame).toHaveAttribute('src', 'data:image/jpeg;base64,AAAA')
    expect(screen.getByText('1920×1080')).toBeInTheDocument()
    expect(screen.getByText('earthcam')).toBeInTheDocument()
  })

  it("takes the camera's own name for itself as the default label", async () => {
    await probeSuccessfully()
    expect(screen.getByDisplayValue('Linkou Old Street Cam')).toBeInTheDocument()
  })

  it('offers every group, and ungrouped', async () => {
    await probeSuccessfully()
    const group = screen.getByLabelText(/^group$/i)
    const options = [...group.querySelectorAll('option')].map((option) => option.textContent)
    expect(options).toContain('Ungrouped')
    expect(options).toContain('Corridor')
  })

  it('sends the id, url, label, group and capabilities', async () => {
    createCamera.mockResolvedValue({
      camera_id: 'linkou',
      label: 'Linkou Old Street Cam',
      zone: 'corridor',
      zone_kind: 'common_area',
      capabilities: ['anomaly_detection', 'camera_tamper'],
      notify_on: [],
      notify_min_confidence: 'likely',
      clip_preroll_seconds: null,
      clip_postroll_seconds: null,
      summary_interval_seconds: null,
      persisted: true,
      restart_required_fields: ['url', 'profile'],
    })
    const user = await probeSuccessfully()

    await user.type(screen.getByLabelText(/camera id/i), 'linkou')
    await user.selectOptions(screen.getByLabelText(/^group$/i), 'corridor')
    await user.click(screen.getByRole('button', { name: /add camera/i }))

    expect(createCamera).toHaveBeenCalledWith({
      camera_id: 'linkou',
      url: 'https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/',
      label: 'Linkou Old Street Cam',
      zone: 'corridor',
      capabilities: ['anomaly_detection', 'camera_tamper'],
    })
  })

  it('sends null for ungrouped rather than an empty string', async () => {
    createCamera.mockResolvedValue({
      camera_id: 'x',
      label: '',
      zone: null,
      zone_kind: null,
      capabilities: [],
      notify_on: [],
      notify_min_confidence: 'likely',
      clip_preroll_seconds: null,
      clip_postroll_seconds: null,
      summary_interval_seconds: null,
      persisted: true,
      restart_required_fields: [],
    })
    const user = await probeSuccessfully()

    await user.type(screen.getByLabelText(/camera id/i), 'x')
    await user.click(screen.getByRole('button', { name: /add camera/i }))

    expect(createCamera).toHaveBeenCalledWith(expect.objectContaining({ zone: null }))
  })

  it('refuses an id that is already taken, before asking the engine', async () => {
    listCameras.mockResolvedValue(cameraList(['linkou']))
    const user = await probeSuccessfully()

    await user.type(screen.getByLabelText(/camera id/i), 'linkou')

    expect(screen.getByText(/already exists/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /add camera/i })).toBeDisabled()
    expect(createCamera).not.toHaveBeenCalled()
  })

  it('cannot be saved without an id', async () => {
    await probeSuccessfully()
    expect(screen.getByRole('button', { name: /add camera/i })).toBeDisabled()
  })

  it("shows the engine's own refusal rather than rewording it", async () => {
    // The engine explains which model was not loaded and that a restart is needed.
    // Anything this page said instead would be a worse version of that.
    createCamera.mockRejectedValue(
      new engineClient.EngineHttpError(409, 'this engine did not load the face model'),
    )
    const user = await probeSuccessfully()

    await user.type(screen.getByLabelText(/camera id/i), 'linkou')
    await user.click(screen.getByRole('button', { name: /add camera/i }))

    expect(await screen.findByText(/did not load the face model/)).toBeInTheDocument()
  })
})

describe('AddCameraPage, on a read-only engine', () => {
  it('says which setting turns writes on, and cannot save', async () => {
    listCameras.mockResolvedValue(cameraList([], false))
    probeSource.mockResolvedValue(A_PLAYABLE_STREAM)
    const user = userEvent.setup()
    renderWithProviders(<AddCameraPage />)

    expect(await screen.findByText(/SENTINEL_ENABLE_CAMERA_WRITES/)).toBeInTheDocument()

    await user.type(await screen.findByLabelText(/stream url/i), 'rtsp://host/1')
    await user.click(screen.getByRole('button', { name: /preview/i }))
    await user.type(await screen.findByLabelText(/camera id/i), 'cam-1')

    expect(screen.getByRole('button', { name: /add camera/i })).toBeDisabled()
  })
})
