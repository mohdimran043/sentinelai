import { beforeEach, describe, expect, it, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CapabilityPanel } from '@/routes/camera/CapabilityPanel'
import * as engineClient from '@/api/engineClient'
import type { CameraEditResponse, CameraStatus } from '@/api/engineClient'

vi.mock('@/api/engineClient', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/engineClient')>()
  return { ...actual, updateCamera: vi.fn() }
})

const updateCamera = vi.mocked(engineClient.updateCamera)

const ALL_KINDS = [
  'collapse',
  'altercation',
  'self_harm',
  'medication',
  'distress',
  'other',
] as const

const camera: CameraStatus = {
  camera_id: 'avenue_01',
  label: 'avenue_01',
  frames_seen: 5000,
  frames_dropped: 10,
  detections_run: 4800,
  escalations: 7,
  escalations_dropped: 1,
  discontinuities: 0,
  last_frame_at: 1_700_000_000,
  last_frame_epoch: 1_700_000_000,
  last_escalation_at: 1_700_000_000,
  capabilities: ['anomaly_detection', 'scene_description'],
  falls_suspected: 0,
  notify_on: [...ALL_KINDS],
  notify_min_confidence: 'likely',
  clip_preroll_seconds: null,
  clip_postroll_seconds: null,
  summary_interval_seconds: null,
}

function withCamera(overrides: Partial<CameraStatus> = {}): CameraStatus {
  return { ...camera, ...overrides }
}

/**
 * What `PATCH /cameras/{id}` answers with — the whole stored record, not the
 * telemetry the panel renders from. Kept separate from `camera` above because
 * they are genuinely different shapes: the response carries `persisted` and
 * `restart_required_fields`, which no amount of spreading a `CameraStatus`
 * produces, and a test that fakes the save with the wrong one is asserting
 * against a reply the engine cannot send.
 */
const editResponse: CameraEditResponse = {
  camera_id: 'avenue_01',
  label: 'avenue_01',
  zone: null,
  zone_kind: null,
  capabilities: ['anomaly_detection', 'scene_description'],
  notify_on: [...ALL_KINDS],
  notify_min_confidence: 'likely',
  clip_preroll_seconds: null,
  clip_postroll_seconds: null,
  summary_interval_seconds: null,
  persisted: true,
  restart_required_fields: ['url', 'profile'],
}

beforeEach(() => {
  updateCamera.mockReset()
})

describe('CapabilityPanel, what the camera detects', () => {
  it('shows what enabling a capability would cost, including the free one', () => {
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    // §13's promise — a disabled capability loads no model — only means something to
    // an operator who can see the price of switching one on.
    expect(screen.getAllByText(/no model/i).length).toBeGreaterThan(0)
  })

  it('marks person authorisation as biometric where the decision is made', () => {
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    expect(screen.getByText('biometric')).toBeInTheDocument()
  })
})

describe('CapabilityPanel, what reaches a person', () => {
  it('offers a checkbox per concern kind, checked as stored', () => {
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    for (const name of ['Collapse', 'Altercation', 'Self harm', 'Medication', 'Distress', 'Other']) {
      expect(screen.getByRole('checkbox', { name })).toBeChecked()
    }
  })

  it('shows a kind the camera does not route as unchecked', () => {
    renderWithProviders(
      <CapabilityPanel camera={withCamera({ notify_on: ['collapse'] })} writable={true} />,
    )

    expect(screen.getByRole('checkbox', { name: 'Collapse' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: 'Distress' })).not.toBeChecked()
  })

  it('reads an empty routing list as muted rather than as unconfigured', () => {
    renderWithProviders(<CapabilityPanel camera={withCamera({ notify_on: [] })} writable={true} />)

    expect(screen.getByTestId('capability-panel-muted-warning')).toHaveTextContent(
      /keeps detecting/i,
    )
  })

  it('sends the whole remaining list when one kind is unchecked', async () => {
    updateCamera.mockResolvedValue({ ...editResponse, notify_on: [] })
    const user = userEvent.setup()
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    await user.click(screen.getByRole('checkbox', { name: 'Medication' }))
    await user.click(screen.getByRole('button', { name: /save/i }))

    // A whole list, never a diff: the engine stores what it is sent.
    expect(updateCamera).toHaveBeenCalledWith('avenue_01', {
      notify_on: ['collapse', 'altercation', 'self_harm', 'distress', 'other'],
    })
  })
})

describe('CapabilityPanel, the two lists together', () => {
  it('warns when a camera watches for falls and would tell nobody', () => {
    // The whole reason these two used to be separate panels and are now one. Each
    // list is individually sensible and the combination is a silent failure.
    renderWithProviders(
      <CapabilityPanel
        camera={withCamera({ capabilities: ['fall_detection'], notify_on: ['distress'] })}
        writable={true}
      />,
    )

    expect(screen.getByText(/would not tell anyone about one/i)).toBeInTheDocument()
  })

  it('does not warn when falls are watched for and routed', () => {
    renderWithProviders(
      <CapabilityPanel
        camera={withCamera({ capabilities: ['fall_detection'], notify_on: ['collapse'] })}
        writable={true}
      />,
    )

    expect(screen.queryByText(/would not tell anyone about one/i)).not.toBeInTheDocument()
  })

  it('does not warn about collapse on a camera that is not watching for falls', () => {
    renderWithProviders(
      <CapabilityPanel
        camera={withCamera({ capabilities: ['anomaly_detection'], notify_on: [] })}
        writable={true}
      />,
    )

    expect(screen.queryByText(/would not tell anyone about one/i)).not.toBeInTheDocument()
  })

  it('saves both lists in one request, because they are one decision', async () => {
    updateCamera.mockResolvedValue({ ...editResponse })
    const user = userEvent.setup()
    renderWithProviders(
      <CapabilityPanel camera={withCamera({ notify_on: ['collapse'] })} writable={true} />,
    )

    await user.click(screen.getByRole('checkbox', { name: /fall/i }))
    await user.click(screen.getByRole('checkbox', { name: 'Distress' }))
    await user.click(screen.getByRole('button', { name: /save/i }))

    expect(updateCamera).toHaveBeenCalledTimes(1)
    expect(updateCamera).toHaveBeenCalledWith('avenue_01', {
      capabilities: ['anomaly_detection', 'scene_description', 'fall_detection'],
      notify_on: ['collapse', 'distress'],
    })
  })

  it('sends only the list the operator touched', async () => {
    updateCamera.mockResolvedValue({ ...editResponse })
    const user = userEvent.setup()
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    await user.click(screen.getByRole('checkbox', { name: 'Distress' }))
    await user.click(screen.getByRole('button', { name: /save/i }))

    // No `capabilities` key: an untouched list must not be rewritten, or two operators
    // editing different halves would overwrite each other.
    expect(updateCamera).toHaveBeenCalledWith('avenue_01', {
      notify_on: ['collapse', 'altercation', 'self_harm', 'medication', 'other'],
    })
  })

  it('discards both drafts together', async () => {
    const user = userEvent.setup()
    renderWithProviders(<CapabilityPanel camera={camera} writable={true} />)

    await user.click(screen.getByRole('checkbox', { name: 'Distress' }))
    await user.click(screen.getByRole('checkbox', { name: /fall/i }))
    await user.click(screen.getByRole('button', { name: /discard/i }))

    expect(screen.getByRole('checkbox', { name: 'Distress' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: /fall/i })).not.toBeChecked()
    expect(updateCamera).not.toHaveBeenCalled()
  })
})

describe('CapabilityPanel, read-only engines', () => {
  it('disables both lists and says which setting turns editing on', () => {
    renderWithProviders(<CapabilityPanel camera={camera} writable={false} />)

    for (const box of screen.getAllByRole('checkbox')) {
      expect(box).toBeDisabled()
    }
    expect(screen.getByText(/SENTINEL_ENABLE_CAMERA_WRITES/)).toBeInTheDocument()
  })
})
