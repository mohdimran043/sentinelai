import { describe, expect, it } from 'vitest'
import { screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/renderWithProviders'
import { AlertRow } from '@/alerts/AlertRow'
import type { AlertEntry } from '@/api/engineClient'

const alert: AlertEntry = {
  alert_id: '77917175-a7cc-469d-992f-1ae10edeb6e7',
  camera_id: 'linkou',
  camera_label: 'Linkou Old Street, Taiwan',
  zone: 'corridor',
  reason: 'zone_intrusion',
  subject: '285',
  state: 'active',
  severity: 'high',
  priority: 'high',
  first_seen: 1_700_000_000,
  last_seen: 1_700_000_000,
  occurrences: 1,
  description: 'A person is walking on the sidewalk.',
  event_ids: ['acd37cd0-b796-4eb1-8607-0312a2e00ac7'],
  clip_uri: 's3://sentinel-clips/linkou/acd37cd0.mp4',
  notify_clip_uri: 's3://sentinel-clips/linkou/acd37cd0-notify.mp4',
  acknowledged_by: null,
  acknowledged_at: null,
}

describe('AlertRow, the clip', () => {
  it('plays the recording on the full row', () => {
    renderWithProviders(<AlertRow alert={alert} />)

    expect(screen.getByTestId('alert-clip-video')).toHaveAttribute('data-variant', 'short')
  })

  it('leaves the dashboard rail a list, not a wall of video', () => {
    // The compact row is scanned and clicked through. A player in each one competes
    // with the twelve rows around it for the attention the rail exists to direct.
    renderWithProviders(<AlertRow alert={alert} compact />)

    expect(screen.queryByTestId('alert-clip-video')).not.toBeInTheDocument()
  })

  it('says a clip has not landed yet rather than showing an empty frame', () => {
    renderWithProviders(
      <AlertRow alert={{ ...alert, clip_uri: null, notify_clip_uri: null }} />,
    )

    expect(screen.queryByTestId('alert-clip-video')).not.toBeInTheDocument()
    expect(screen.getByText(/no clip yet/i)).toBeInTheDocument()
  })
})
