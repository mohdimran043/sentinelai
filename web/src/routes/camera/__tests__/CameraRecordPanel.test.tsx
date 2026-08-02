import { describe, expect, it } from 'vitest'
import { beforeEach, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraRecordPanel } from '@/routes/camera/CameraRecordPanel'
import * as engineClient from '@/api/engineClient'
import { EngineHttpError } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, updateCamera: vi.fn() }
})

const updateCamera = vi.mocked(engineClient.updateCamera)

const storedResponse = {
  camera_id: 'avenue_01',
  label: 'East door',
  zone: 'corridor' as const,
  zone_kind: 'common_area' as const,
  persisted: true as const,
  restart_required_fields: ['url', 'profile'],
}

const record = { label: 'Avenue entrance', zone: 'corridor' as const, zone_kind: 'common_area' as const }

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
        record={{ label: 'Loose camera', zone: null, zone_kind: null }}
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

describe('CameraRecordPanel, editing', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  it('offers a label input and a zone select when writes are enabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByLabelText(/label/i)).toHaveValue('Avenue entrance')
    expect(screen.getByLabelText(/zone/i)).toHaveValue('corridor')
    expect(screen.getByRole('button', { name: /save/i })).toBeInTheDocument()
  })

  it('disables save until something actually changes, since an empty edit is a 422', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()

    await user.type(screen.getByLabelText(/label/i), '!')

    expect(screen.getByRole('button', { name: /save/i })).toBeEnabled()
  })

  it('sends only the changed field', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/label/i)
    await user.clear(input)
    await user.type(input, 'East door')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { label: 'East door' })
  })

  it('sends an explicit null zone when ungrouping', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, label: 'Avenue entrance', zone: null, zone_kind: null })
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.selectOptions(screen.getByLabelText(/zone/i), '')
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledWith('avenue_01', { zone: null })
  })

  it('refuses an empty label without calling the engine', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    await user.clear(screen.getByLabelText(/label/i))

    expect(screen.getByText(/cannot be empty/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /save/i })).toBeDisabled()
    expect(updateCamera).not.toHaveBeenCalled()
  })

  it('confirms the save by reporting what was stored and that it survives a restart', async () => {
    updateCamera.mockResolvedValue(storedResponse)
    const user = userEvent.setup()
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />)

    const input = screen.getByLabelText(/label/i)
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

    const input = screen.getByLabelText(/label/i)
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

    expect(screen.getByLabelText(/label/i)).toHaveValue('Half-typed name')
  })

  it('does not revert a field changed elsewhere while the operator edits a different one', async () => {
    updateCamera.mockResolvedValue({ ...storedResponse, zone: 'room' as const })
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    // Operator touches only the zone.
    await user.selectOptions(screen.getByLabelText(/zone/i), 'room')

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

    expect(screen.getByLabelText(/label/i)).toHaveValue('Renamed by someone else')
  })
})

describe('CameraRecordPanel, failures', () => {
  beforeEach(() => {
    updateCamera.mockReset()
  })

  async function editAndSave() {
    const user = userEvent.setup()
    const input = screen.getByLabelText(/label/i)
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
    expect(screen.getByLabelText(/label/i)).toHaveValue('East door')
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

    await user.type(screen.getByLabelText(/label/i), '!')

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
    expect(screen.getByLabelText(/label/i)).toHaveValue('Avenue entrance!')
  })

  it('does not warn when nothing changed underneath', async () => {
    const user = userEvent.setup()
    const { rerender } = renderWithProviders(
      <CameraRecordPanel cameraId="avenue_01" record={record} writable={true} />,
    )

    await user.type(screen.getByLabelText(/label/i), '!')
    rerender(<CameraRecordPanel cameraId="avenue_01" record={{ ...record }} writable={true} />)

    expect(screen.queryByTestId('camera-record-stale')).not.toBeInTheDocument()
  })
})
