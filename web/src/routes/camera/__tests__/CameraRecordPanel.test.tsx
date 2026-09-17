import { describe, expect, it } from 'vitest'
import { beforeEach, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraRecordPanel, type StoredCameraRecord } from '@/routes/camera/CameraRecordPanel'
import * as engineClient from '@/api/engineClient'
import { EngineHttpError, type CameraEditResponse } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, updateCamera: vi.fn() }
})

const updateCamera = vi.mocked(engineClient.updateCamera)

const ALL_KINDS = [
  'altercation',
  'collapse',
  'distress',
  'medication',
  'other',
  'self_harm',
] as const

const storedResponse: CameraEditResponse = {
  camera_id: 'avenue_01',
  label: 'East door',
  zone: 'corridor',
  zone_kind: 'common_area',
  // The capabilities and welfare notification policy the engine echoes back on
  // every edit, in the sorted whole-list form it stores.
  capabilities: ['anomaly_detection', 'scene_description'],
  notify_on: [...ALL_KINDS],
  notify_min_confidence: 'likely',
  clip_preroll_seconds: null,
  clip_postroll_seconds: null,
  summary_interval_seconds: null,
  persisted: true,
  restart_required_fields: ['url', 'profile'],
}

const record: StoredCameraRecord = {
  label: 'Avenue entrance',
  zone: 'corridor',
  zone_kind: 'common_area',
  notify_on: [...ALL_KINDS],
  notify_min_confidence: 'likely',
  clip_preroll_seconds: null,
  clip_postroll_seconds: null,
  summary_interval_seconds: null,
}

describe('CameraRecordPanel, read-only', () => {
  it('shows the stored record even when writes are disabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.getByText('Avenue entrance')).toBeInTheDocument()
    expect(screen.getByText('Corridor · Common area')).toBeInTheDocument()
  })

  it('offers no editing controls at all, rather than disabled ones', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.queryByLabelText(/label/i)).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /save/i })).not.toBeInTheDocument()
  })

  it('names the flag and the way to change the record without it', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    const notice = screen.getByRole('status')
    expect(notice).toHaveTextContent('SENTINEL_ENABLE_CAMERA_WRITES')
    expect(notice).toHaveTextContent(/cameras\.json/)
    expect(notice).toHaveTextContent(/restart/i)
  })

  it('says an ungrouped camera is ungrouped rather than showing an empty row', () => {
    renderWithProviders(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, zone: null, zone_kind: null, label: 'Loose camera' }}
        writable={false}
      />,
    )

    expect(screen.getByText(/ungrouped/i)).toBeInTheDocument()
  })

  it('names url and profile as restart-required and shows no value for either', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    // The engine returns neither, because an RTSP URL carries credentials — so the
    // `url` and `profile` rows' values must be exactly the placeholder text and
    // nothing else, never a URL that leaked through some other code path.
    expect(screen.getByText(/^url$/i)).toBeInTheDocument()
    expect(screen.getByText(/^profile$/i)).toBeInTheDocument()
    expect(screen.getAllByText('restart required')).toHaveLength(2)
  })
})

describe('CameraRecordPanel, the welfare policy read-only', () => {
  it('shows every routed kind rather than a count, so the list can be checked', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    const routed = screen.getByTestId('camera-record-notify-on')
    for (const kind of ['Collapse', 'Altercation', 'Self harm', 'Medication', 'Distress', 'Other']) {
      expect(routed).toHaveTextContent(kind)
    }
  })

  it('reads a muted camera as muted, not as unconfigured', () => {
    // `notify_on: []` is a stored choice — somebody silenced this camera. An
    // empty row would read as "nobody has set this up yet", which is the
    // opposite: a muted camera keeps detecting and recording and tells no one.
    renderWithProviders(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, notify_on: [] }}
        writable={false}
      />,
    )

    expect(screen.getByTestId('camera-record-notify-on')).toHaveTextContent(/muted/i)
  })

  it('shows the threshold and says the durations follow a default when unset', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.getByTestId('camera-record-min-confidence')).toHaveTextContent('Likely')
    // Null is "what is stored", not the effective value — the panel must not
    // resolve a default it was never told.
    expect(screen.getByTestId('camera-record-clip-preroll')).toHaveTextContent(/default/i)
    expect(screen.getByTestId('camera-record-clip-postroll')).toHaveTextContent(/default/i)
    expect(screen.getByTestId('camera-record-summary-interval')).toHaveTextContent(/default/i)
  })

  it('shows a stored duration override as the number it is', () => {
    renderWithProviders(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{
          ...record,
          clip_preroll_seconds: 0,
          clip_postroll_seconds: 8.5,
          summary_interval_seconds: 30,
        }}
        writable={false}
      />,
    )

    // Zero is an override, not an absent one: it means no lead-in at all.
    expect(screen.getByTestId('camera-record-clip-preroll')).toHaveTextContent('0')
    expect(screen.getByTestId('camera-record-clip-preroll')).not.toHaveTextContent(/default/i)
    expect(screen.getByTestId('camera-record-clip-postroll')).toHaveTextContent('8.5')
    expect(screen.getByTestId('camera-record-summary-interval')).toHaveTextContent('30')
  })

  it('says what these fields actually route, so nobody reads them as a detector', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    const note = screen.getByTestId('camera-record-welfare-note')
    expect(note).toHaveTextContent(/single frame/i)
    expect(note).toHaveTextContent(/not a detector/i)
  })
})

describe('CameraRecordPanel, editing', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  it('offers a label input and a zone select when writes are enabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByLabelText(/^label$/i)).toHaveValue('Avenue entrance')
    expect(screen.getByLabelText(/^zone$/i)).toHaveValue('corridor')
    expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument()
  })

  it('disables save until something actually changes, since an empty edit is a 422', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()

    await user.type(screen.getByLabelText(/^label$/i), '!')

    expect(screen.getByRole('button', { name: /save/i })).toBeEnabled()
  })

  it('sends only the changed field', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/^label$/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { label: 'East door' })
  })

  it('sends an explicit null zone when ungrouping', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, label: 'Avenue entrance', zone: null, zone_kind: null })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.selectOptions(screen.getByLabelText(/^zone$/i), '')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { zone: null })
  })

  it('refuses an empty label without calling the engine', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.clear(screen.getByLabelText(/^label$/i))

    expect(screen.getByText(/cannot be empty/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('confirms the save by reporting what was stored and that it survives a restart', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/^label$/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(await screen.findByTestId('camera-record-feedback')).toHaveTextContent(/saved/i)
    expect(screen.getByTestId('camera-record-feedback')).toHaveTextContent(/cameras\.json/)
  })

  it('does not clobber a field the operator is editing when the record refreshes', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    const input = screen.getByLabelText(/^label$/i)
    await user.clear(input)
    await user.type(input, 'Half-typed name')

    // A 5s poll lands mid-edit carrying a change someone else made.
    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    expect(screen.getByLabelText(/^label$/i)).toHaveValue('Half-typed name')
  })

  it('does not revert a field changed elsewhere while the operator edits a different one', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, zone: 'room' as const })
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    // Operator touches only the zone.
    await user.selectOptions(screen.getByLabelText(/^zone$/i), 'room')

    // A 5s poll lands mid-edit: someone else renamed the camera. The operator
    // never touched the label, so this should not end up in the save request.
    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'East door' }}
        writable={true}
      />,
    )

    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { zone: 'room' })
    expect(updateCamera).not.toHaveBeenCalledWith(
      'avenue_01',
      expect.objectContaining({ label: expect.anything() }),
    )
  })

  it('does track the record while the form is untouched', () => {
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    expect(screen.getByLabelText(/^label$/i)).toHaveValue('Renamed by someone else')
  })
})

describe('CameraRecordPanel, editing the welfare policy', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  it('omits notify_on entirely when the operator never touches it', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/^label$/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { label: 'East door' })
    expect(updateCamera).not.toHaveBeenCalledWith(
      'avenue_01',
      expect.objectContaining({ notify_on: expect.anything() }),
    )
  })

  it('sends only the threshold when only the threshold changed', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, notify_min_confidence: 'possible' })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.selectOptions(screen.getByLabelText(/minimum confidence/i), 'possible')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { notify_min_confidence: 'possible' })
  })

  it('offers no "certain" tier, because one still frame cannot earn it', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const options = screen.getByLabelText(/minimum confidence/i).querySelectorAll('option')
    expect([...options].map((option) => option.getAttribute('value'))).toEqual([
      'possible',
      'likely',
    ])
  })

  it('sends a duration override as a number', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, clip_postroll_seconds: 9 })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.type(screen.getByLabelText(/post-roll/i), '9')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { clip_postroll_seconds: 9 })
  })

  it('sends null when a stored override is cleared, reverting to the default', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, summary_interval_seconds: 30 }}
        writable={true}
      />,
    )

    await user.clear(screen.getByLabelText(/summary interval/i))
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { summary_interval_seconds: null })
  })

  it('accepts a zero pre-roll, which means no lead-in at all', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, clip_preroll_seconds: 0 })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.type(screen.getByLabelText(/pre-roll/i), '0')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { clip_preroll_seconds: 0 })
  })

  it('refuses a zero post-roll without calling the engine', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.type(screen.getByLabelText(/post-roll/i), '0')

    expect(screen.getByText(/greater than zero/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('refuses a negative pre-roll without calling the engine', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.type(screen.getByLabelText(/pre-roll/i), '-1')

    expect(screen.getByText(/negative/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('refuses text that is not a number rather than silently discarding it', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.type(screen.getByLabelText(/summary interval/i), 'soon')

    // The operator's text survives; a number input would have eaten it and shown
    // an empty box, which reads as "cleared" and means something else entirely.
    expect(screen.getByLabelText(/summary interval/i)).toHaveValue('soon')
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('warns when the welfare policy changed elsewhere while an edit was in progress', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/^label$/i), '!')

    // Someone muted this camera from another console mid-edit.
    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, notify_on: [] }}
        writable={true}
      />,
    )

    const warning = await screen.findByTestId('camera-record-stale')
    expect(warning).toHaveTextContent(/notify_on/i)
  })

})

describe('CameraRecordPanel, failures', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  async function editAndSave() {
    const user = userEvent.setup()
    const input = screen.getByLabelText(/^label$/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))
  }

  it('shows the engine\'s own words on a 409 and does not retry', async () => {
    updateCamera.mockRejectedValue(
      new EngineHttpError(409, 'cameras.json no longer contains avenue_01'),
    )
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    const alert = await screen.findByTestId('camera-record-error')
    expect(alert).toHaveTextContent('cameras.json no longer contains avenue_01')
    expect(alert).toHaveTextContent(/nothing was written/i)
    expect(updateCamera).toHaveBeenCalledTimes(1)
  })

  it('keeps the operator\'s text after a failure', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(409, 'conflict'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    await screen.findByTestId('camera-record-error')
    expect(screen.getByLabelText(/^label$/i)).toHaveValue('East door')
  })

  it('explains a 403 as writes having been turned off', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(403, 'camera writes are disabled'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    expect(await screen.findByTestId('camera-record-error')).toHaveTextContent(
      /disabled on this engine/i,
    )
  })

  it('surfaces a 422 rather than swallowing it, since it means a contract drift', async () => {
    updateCamera.mockRejectedValue(new EngineHttpError(422, 'url is not an editable field'))
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await editAndSave()

    expect(await screen.findByTestId('camera-record-error')).toHaveTextContent(
      'url is not an editable field',
    )
  })

  it('warns when the record changed elsewhere while an edit was in progress', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/^label$/i), '!')

    rerender(
      <CameraRecordPanel
        cameraId="avenue_01"
        record={{ ...record, label: 'Renamed by someone else' }}
        writable={true}
      />,
    )

    const warning = await screen.findByTestId('camera-record-stale')
    expect(warning).toHaveTextContent(/changed elsewhere/i)
    expect(warning).toHaveTextContent(/label/i)
    // The operator's text is informed against, never replaced.
    expect(screen.getByLabelText(/^label$/i)).toHaveValue('Avenue entrance!')
  })

  it('does not warn when nothing changed underneath', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/^label$/i), '!')
    rerender(<CameraRecordPanel cameraId="avenue_01" record={{ ...record }} writable={true} />)

    expect(screen.queryByTestId('camera-record-stale')).not.toBeInTheDocument()
  })
})
