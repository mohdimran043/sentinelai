import { describe, expect, it } from 'vitest'
import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { RECORDER_BASE_URL } from '@/recorder/config'
import { CamerasPage } from '@/routes/recorder/CamerasPage'
import {
  recorderCameraActionHandler,
  recorderCamerasHandler,
  recorderErrorHandler,
  recorderStatusHandler,
  recorderUnreachableHandler,
} from '@/recorder/mocks/handlers'
import { REAL_CAMERAS, REAL_STATUS, makeCamera, makeStatus } from '@/recorder/mocks/fixtures'
import type { RecorderCamerasResponse, RecorderStatusResponse } from '@/recorder/recorder.types'
import { http, HttpResponse } from 'msw'

function camerasOf(...cameras: RecorderCamerasResponse['cameras']): RecorderCamerasResponse {
  return { cameras, context_note: REAL_CAMERAS.context_note }
}

function statusOf(...statuses: RecorderStatusResponse['cameras']): RecorderStatusResponse {
  return { cameras: statuses, server_time_ns: REAL_STATUS.server_time_ns }
}

describe('CamerasPage — registry', () => {
  it('renders the real registry joined with real worker status', async () => {
    renderWithProviders(<CamerasPage />)

    expect(await screen.findByRole('heading', { name: 'Room 4B' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Corridor 1' })).toBeInTheDocument()
    // room_4b is UNDERLIT in the live capture, the other three are LOST.
    expect(screen.getAllByText('LOST')).toHaveLength(3)
    expect(screen.getByText('UNDERLIT')).toBeInTheDocument()
    expect(screen.getAllByText('1280×720 @ 15fps')).toHaveLength(4)
    expect(screen.getByText('rtsp://10.0.0.99:554/nothing-here')).toBeInTheDocument()
  })

  it('renders the recorder’s own context note about the inert fields, not a paraphrase', async () => {
    renderWithProviders(<CamerasPage />)
    expect(
      await screen.findByText(/Setting elevated_watch does not cause a camera to be watched more closely/),
    ).toBeInTheDocument()
  })

  it('shows an empty mask file honestly as "none configured", not blank', async () => {
    renderWithProviders(<CamerasPage />)
    await screen.findByRole('heading', { name: 'Corridor 1' })
    expect(screen.getAllByText('none configured').length).toBeGreaterThan(0)
  })

  it('says the registry is unreachable rather than rendering an empty grid', async () => {
    server.use(recorderUnreachableHandler('/cameras'))
    renderWithProviders(<CamerasPage />)

    expect(
      await screen.findByText(/The recorder is unreachable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /Room|Corridor|Dayroom/ })).not.toBeInTheDocument()
  })

  it('shows the registry even when worker status cannot be read, without inventing state', async () => {
    server.use(recorderErrorHandler('/status', 503, 'status store unavailable'))
    renderWithProviders(<CamerasPage />)

    expect(await screen.findByRole('heading', { name: 'Room 4B' })).toBeInTheDocument()
    expect(
      await screen.findByText(/status store unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.getAllByText('no status reported').length).toBe(4)
  })
})

describe('CamerasPage — a camera the recorder reports no status for', () => {
  it('renders "no status reported" and disables start/stop rather than guessing running state', async () => {
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'ghost_cam', name: 'Ghost Cam' }))),
      recorderStatusHandler(statusOf()),
    )
    renderWithProviders(<CamerasPage />)

    await screen.findByRole('heading', { name: 'Ghost Cam' })
    expect(screen.getByText('no status reported')).toBeInTheDocument()
    expect(screen.getByText(/not reporting worker status for this camera/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Start camera|Stop camera/ })).not.toBeInTheDocument()
  })
})

describe('CamerasPage — frame age', () => {
  it('renders the -1 sentinel as "no frame yet" rather than a negative age', async () => {
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a' }))),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a', last_frame_age_ms: -1 }))),
    )
    renderWithProviders(<CamerasPage />)

    expect(await screen.findByText('no frame yet')).toBeInTheDocument()
  })
})

describe('CamerasPage — inert fields are legible, not editable', () => {
  it('shows the four stored-but-inert fields as plain values with no control to change them', async () => {
    server.use(
      recorderCamerasHandler(
        camerasOf(
          makeCamera({
            id: 'cam_a',
            elevated_watch: true,
            capacity: 12,
            audio_enabled: true,
            face_recognition_enabled: false,
          }),
        ),
      ),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a' }))),
    )
    renderWithProviders(<CamerasPage />)

    expect(await screen.findByText('stored, read by nothing')).toBeInTheDocument()
    expect(screen.getByText('Elevated watch')).toBeInTheDocument()
    expect(screen.getAllByText('true')).toHaveLength(2) // elevated_watch and audio_enabled
    expect(screen.getByText('12')).toBeInTheDocument()
    // No checkbox, switch or input anywhere on the page — this is a report, not a form.
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
    expect(screen.queryByRole('switch')).not.toBeInTheDocument()
    expect(document.querySelector('input')).not.toBeInTheDocument()
  })
})

describe('CamerasPage — snapshot thumbnail', () => {
  it('cache-busts the snapshot URL using the status poll’s own server_time_ns', async () => {
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }))),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a' })), ),
    )
    renderWithProviders(<CamerasPage />)

    const img = (await screen.findByAltText('Live snapshot from Cam A')) as HTMLImageElement
    expect(img.getAttribute('src')).toBe(
      `${RECORDER_BASE_URL}/snapshot/cam_a?t=${REAL_STATUS.server_time_ns}`,
    )
  })

  it('shows the row is unaffected when one camera’s snapshot fails to load', async () => {
    server.use(
      recorderCamerasHandler(
        camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }), makeCamera({ id: 'cam_b', name: 'Cam B' })),
      ),
      recorderStatusHandler(
        statusOf(makeStatus({ camera_id: 'cam_a' }), makeStatus({ camera_id: 'cam_b' })),
      ),
    )
    renderWithProviders(<CamerasPage />)

    const failingImg = await screen.findByAltText('Live snapshot from Cam A')
    fireEvent.error(failingImg)

    await waitFor(() => {
      expect(screen.getByText('Snapshot unavailable')).toBeInTheDocument()
    })
    // The other row's snapshot and the rest of the page are untouched.
    expect(screen.getByAltText('Live snapshot from Cam B')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Cam A' })).toBeInTheDocument()
  })
})

describe('CamerasPage — start/stop worker control', () => {
  it('does not send any start/stop request merely from rendering the page', async () => {
    let calls = 0
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }))),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a', running: true }))),
      recorderCameraActionHandler('stop', { onRequest: () => (calls += 1) }),
    )
    renderWithProviders(<CamerasPage />)

    await screen.findByRole('button', { name: 'Stop camera' })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(calls).toBe(0)
  })

  it('sends exactly one stop request when clicked, and reflects the outcome once status refetches', async () => {
    const user = userEvent.setup()
    const seenCameraIds: string[] = []
    let stopped = false
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }))),
      http.get(`${RECORDER_BASE_URL}/status`, () =>
        HttpResponse.json(
          statusOf(makeStatus({ camera_id: 'cam_a', running: !stopped, integrity_state: 'HEALTHY' })),
        ),
      ),
      recorderCameraActionHandler('stop', {
        onRequest: (cameraId) => {
          seenCameraIds.push(cameraId)
          stopped = true
        },
      }),
    )
    renderWithProviders(<CamerasPage />)

    const button = await screen.findByRole('button', { name: 'Stop camera' })
    await user.click(button)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Start camera' })).toBeInTheDocument()
    })
    expect(seenCameraIds).toEqual(['cam_a'])
    expect(screen.getByTestId('camera-action-outcome-cam_a')).toHaveTextContent(/sent/i)
  })

  it('shows the recorder’s own error and leaves the button actionable when the request fails', async () => {
    const user = userEvent.setup()
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }))),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a', running: true }))),
      recorderCameraActionHandler('stop', { status: 500, error: 'worker refused to stop' }),
    )
    renderWithProviders(<CamerasPage />)

    const button = await screen.findByRole('button', { name: 'Stop camera' })
    await user.click(button)

    expect(await screen.findByRole('alert')).toHaveTextContent(/worker refused to stop/i)
    // The button is still "Stop camera" — the recorder never actually stopped it.
    expect(screen.getByRole('button', { name: 'Stop camera' })).not.toBeDisabled()
  })

  it('offers "Start camera" — not "Stop" — for a worker that is not running', async () => {
    server.use(
      recorderCamerasHandler(camerasOf(makeCamera({ id: 'cam_a', name: 'Cam A' }))),
      recorderStatusHandler(statusOf(makeStatus({ camera_id: 'cam_a', running: false }))),
    )
    renderWithProviders(<CamerasPage />)

    expect(await screen.findByRole('button', { name: 'Start camera' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Stop camera' })).not.toBeInTheDocument()
  })

  it('sends the start request to the specific camera id clicked, not a fixed one', async () => {
    const user = userEvent.setup()
    const seenPaths: string[] = []
    server.use(
      recorderCamerasHandler(
        camerasOf(
          makeCamera({ id: 'cam_a', name: 'Cam A' }),
          makeCamera({ id: 'cam_b', name: 'Cam B' }),
        ),
      ),
      recorderStatusHandler(
        statusOf(
          makeStatus({ camera_id: 'cam_a', running: false }),
          makeStatus({ camera_id: 'cam_b', running: false }),
        ),
      ),
      http.post(`${RECORDER_BASE_URL}/cameras/:cameraId/start`, ({ params }) => {
        seenPaths.push(String(params.cameraId))
        return HttpResponse.json({ ok: true })
      }),
    )
    renderWithProviders(<CamerasPage />)

    await screen.findByRole('heading', { name: 'Cam B' })
    const cardB = screen.getByRole('heading', { name: 'Cam B' }).closest('div')!.parentElement!
    await user.click(within(cardB).getByRole('button', { name: 'Start camera' }))

    await waitFor(() => {
      expect(seenPaths).toEqual(['cam_b'])
    })
  })
})
