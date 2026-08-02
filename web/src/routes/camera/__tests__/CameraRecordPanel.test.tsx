import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CameraRecordPanel } from '@/routes/camera/CameraRecordPanel'

const record = { label: 'Avenue entrance', zone: 'corridor' as const, zone_kind: 'common_area' as const }

describe('CameraRecordPanel, read-only', () => {
  it('shows the stored record even when writes are disabled', () => {
    renderWithProviders(<CameraRecordPanel cameraId="avenue_01" record={record} writable={false} />)

    expect(screen.getByText('Avenue entrance')).toBeInTheDocument()
    expect(screen.getByText(/corridor/i)).toBeInTheDocument()
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

    // The engine returns neither, because an RTSP URL carries credentials.
    expect(screen.getByText(/^url$/i)).toBeInTheDocument()
    expect(screen.getByText(/^profile$/i)).toBeInTheDocument()
    expect(screen.getAllByText(/restart required/i).length).toBeGreaterThanOrEqual(2)
    expect(screen.queryByText(/rtsp:/i)).not.toBeInTheDocument()
  })
})
