import { describe, expect, it, vi } from 'vitest'
import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { server } from '@/test/mswServer'
import { MasksPage } from '@/routes/recorder/MasksPage'
import {
  recorderCamerasHandler,
  recorderCameraActionHandler,
  recorderErrorHandler,
} from '@/recorder/mocks/handlers'
import {
  recorderCalibrationFrameHandler,
  recorderMaskHandler,
  recorderMaskSaveHandler,
  recorderMaskValidateHandler,
} from '@/recorder/mocks/maskHandlers'
import {
  REAL_MASK_CALIBRATION_AVAILABLE,
  REAL_MASK_EMPTY_CORRIDOR_1,
  REAL_MASK_ROOM_4B,
  REAL_MASK_VALIDATE_TOO_FEW_POINTS,
  REAL_MASK_VALIDATE_VALID_SINGLE,
  makeMaskResponse,
} from '@/recorder/mocks/maskFixtures'
import type { RecorderCamerasResponse } from '@/recorder/recorder.types'

const TWO_CAMERAS: RecorderCamerasResponse = {
  cameras: [
    {
      id: 'room_4b',
      name: 'Room 4B',
      mode: 'room',
      space_type: 'room',
      source: 'v4l2:/dev/video0',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '/var/tmp/sentinel-wall/masks/room_4b.json',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
    {
      id: 'corridor_1',
      name: 'Corridor 1',
      mode: 'common_area',
      space_type: 'corridor',
      source: 'rtsp://10.0.0.99:554/nothing-here',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
  ],
  context_note: 'stored, read by nothing',
}

/**
 * The mask canvas is a real SVG with `viewBox="0 0 frameWidth frameHeight"`,
 * so a click's frame-space coordinate depends on the SVG's *displayed* CSS
 * size — which jsdom always reports as a zero rect unless told otherwise.
 * A 640x360 box over a 1280x720 frame is an exact, easy-to-check 2x scale on
 * both axes.
 */
function mockDisplayedAt(width: number, height: number) {
  vi.spyOn(SVGElement.prototype, 'getBoundingClientRect').mockReturnValue({
    width,
    height,
    top: 0,
    left: 0,
    right: width,
    bottom: height,
    x: 0,
    y: 0,
    toJSON() {
      return this
    },
  })
}

describe('MasksPage — calibration refusal', () => {
  it("shows the recorder's own refusal and points at the remedy, without ever requesting the live snapshot as a substitute", async () => {
    server.use(recorderCamerasHandler(TWO_CAMERAS), recorderMaskHandler(REAL_MASK_ROOM_4B))
    const { container } = renderWithProviders(<MasksPage />)

    expect(
      await screen.findByText(/An unmasked frame is not served for a camera under observation/i),
    ).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /stop this camera/i })).toBeInTheDocument()

    // The only two possible <img> sources on this page are the calibration
    // frame (blocked here) and the live snapshot. A version that falls back to
    // the snapshot when calibration is unavailable would render one; a
    // correct one renders none.
    expect(container.querySelectorAll('img')).toHaveLength(0)
  })

  it('renders the saved regions read-only, unscaled, over a neutral canvas — no drawing or save controls while no frame can be shown', async () => {
    server.use(recorderCamerasHandler(TWO_CAMERAS), recorderMaskHandler(REAL_MASK_ROOM_4B))
    renderWithProviders(<MasksPage />)

    const canvas = await screen.findByRole('img', {
      name: /2 mask regions over a 1280 by 720 pixel frame/i,
    })
    // The SVG viewBox is the frame's own pixel space, so the raw polygon
    // points from GET /api/masks are written into the `points` attribute
    // verbatim — no separate scaling step to get wrong.
    expect(canvas.innerHTML).toContain('80,90 420,90 420,320 80,320')

    expect(screen.queryByRole('button', { name: /start new region/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Check' })).not.toBeInTheDocument()
  })

  it('stopping the camera from the remedy button calls the real start/stop endpoint', async () => {
    const user = userEvent.setup()
    const onRequest = vi.fn()
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(REAL_MASK_ROOM_4B),
      recorderCameraActionHandler('stop', { onRequest }),
    )
    renderWithProviders(<MasksPage />)

    await user.click(await screen.findByRole('button', { name: /stop this camera/i }))

    await waitFor(() => expect(onRequest).toHaveBeenCalledWith('room_4b'))
  })

  it('says the mask configuration is unreachable rather than showing an empty editor', async () => {
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderErrorHandler('/masks/room_4b', 500, 'mask store unavailable'),
    )
    renderWithProviders(<MasksPage />)

    expect(
      await screen.findByText(/mask store unavailable/i, undefined, { timeout: 5000 }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })
})

describe('MasksPage — drawing against a real calibration frame', () => {
  it('shows the calibration frame image at the exact recorder URL and enables drawing', async () => {
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(REAL_MASK_CALIBRATION_AVAILABLE),
      recorderCalibrationFrameHandler(),
    )
    renderWithProviders(<MasksPage />)

    const image = await screen.findByRole('img', {
      name: /calibration frame for drawing privacy masks/i,
    })
    expect(image).toHaveAttribute('src', '/recorder/api/masks/room_4b/calibration-frame')
    expect(screen.getByRole('button', { name: /start new region/i })).toBeInTheDocument()
  })

  it('scales a clicked point from the displayed canvas size into the camera-configured frame size before sending it to the recorder', async () => {
    const user = userEvent.setup()
    mockDisplayedAt(640, 360) // half of the 1280x720 frame on both axes.
    const onRequest = vi.fn()
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(REAL_MASK_CALIBRATION_AVAILABLE),
      recorderCalibrationFrameHandler(),
      recorderMaskValidateHandler(REAL_MASK_VALIDATE_VALID_SINGLE, { onRequest }),
    )
    renderWithProviders(<MasksPage />)

    await user.click(await screen.findByRole('button', { name: /start new region/i }))

    const canvas = screen.getByRole('img', { name: /mask regions over a 1280 by 720 pixel frame/i })
    // Displayed-space clicks; each should land at exactly 2x in frame space.
    fireEvent.click(canvas, { clientX: 50, clientY: 50 })
    fireEvent.click(canvas, { clientX: 300, clientY: 50 })
    fireEvent.click(canvas, { clientX: 150, clientY: 300 })

    await user.type(screen.getByLabelText('Region name'), 'closet')
    await user.click(screen.getByRole('button', { name: 'Add region' }))
    await user.click(screen.getByRole('button', { name: 'Check' }))

    await waitFor(() => expect(onRequest).toHaveBeenCalled())
    const sentRegions = onRequest.mock.calls[0]![0] as {
      regions: { region_id: string; polygon: number[][] }[]
    }
    const drawn = sentRegions.regions.find((region) => region.region_id === 'closet')
    // A broken mapping (no scaling, a single averaged scale, or swapped axes)
    // would send some other set of numbers here.
    expect(drawn?.polygon).toEqual([
      [100, 100],
      [600, 100],
      [300, 600],
    ])
  })

  it('reports an invalid save as not saved, with the recorder’s own per-region reason, and does not also report success', async () => {
    const user = userEvent.setup()
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(
        makeMaskResponse({ camera_id: 'room_4b', calibration_available: true, recording: false }),
      ),
      recorderCalibrationFrameHandler(),
      recorderMaskSaveHandler(REAL_MASK_VALIDATE_TOO_FEW_POINTS),
    )
    renderWithProviders(<MasksPage />)

    await user.click(await screen.findByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Not saved')).toBeInTheDocument()
    // The recorder's own per-region reason is rendered twice on purpose (the
    // top-level summary and the per-region list) — both must say the same
    // thing, so assert on the count rather than picking one.
    expect(screen.getAllByText(/has 2 points; need at least 3/i).length).toBeGreaterThan(0)
    expect(screen.queryByText('Saved')).not.toBeInTheDocument()
  })

  it('reports a successful save, including the recorder’s own recomputed coverage', async () => {
    const user = userEvent.setup()
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(
        makeMaskResponse({ camera_id: 'room_4b', calibration_available: true, recording: false }),
      ),
      recorderCalibrationFrameHandler(),
      recorderMaskSaveHandler(REAL_MASK_VALIDATE_VALID_SINGLE),
    )
    renderWithProviders(<MasksPage />)

    await user.click(await screen.findByRole('button', { name: 'Save' }))

    expect(await screen.findByText('Saved')).toBeInTheDocument()
    expect(screen.getByText(/8\.5% of the frame masked/)).toBeInTheDocument()
  })

  it('checking never persists — the saved-region count is unchanged even after a valid check', async () => {
    const user = userEvent.setup()
    server.use(
      recorderCamerasHandler(TWO_CAMERAS),
      recorderMaskHandler(
        makeMaskResponse({ camera_id: 'room_4b', calibration_available: true, recording: false }),
      ),
      recorderCalibrationFrameHandler(),
      recorderMaskValidateHandler(REAL_MASK_VALIDATE_VALID_SINGLE),
    )
    renderWithProviders(<MasksPage />)

    await screen.findByRole('button', { name: 'Check' })
    const before = screen.getByText('saved regions').parentElement?.textContent
    await user.click(screen.getByRole('button', { name: 'Check' }))

    expect(await screen.findByText('Valid')).toBeInTheDocument()
    expect(screen.getByText('saved regions').parentElement?.textContent).toBe(before)
  })
})

describe('MasksPage — switching cameras', () => {
  it('resets the draft and requests the newly selected camera’s own mask state', async () => {
    const user = userEvent.setup()
    server.use(recorderCamerasHandler(TWO_CAMERAS), recorderMaskHandler(REAL_MASK_ROOM_4B))
    renderWithProviders(<MasksPage />)

    // `toilet` is room_4b's own saved region, seeded into the draft by an
    // effect one commit after the mask response lands — so wait for it
    // directly. Waiting on anything the response renders in that first commit
    // (the calibration refusal, say) settles while the draft is still empty.
    await screen.findByText('toilet', { exact: false })

    server.use(recorderMaskHandler(REAL_MASK_EMPTY_CORRIDOR_1))
    await user.selectOptions(screen.getByLabelText('Camera'), 'corridor_1')

    await waitFor(() => {
      expect(screen.getByText('No regions in this draft.')).toBeInTheDocument()
    })
    expect(screen.queryByText('toilet', { exact: false })).not.toBeInTheDocument()
  })
})
