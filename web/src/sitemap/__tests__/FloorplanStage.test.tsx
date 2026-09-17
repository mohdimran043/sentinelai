import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { FloorplanStage } from '@/sitemap/FloorplanStage'
import type { CameraStatus } from '@/api/engineClient'
import type { Tone } from '@/lib/severity'
import type { NormalizedPoint } from '@/sitemap/placement'

function makeCamera(overrides: Partial<CameraStatus> = {}): CameraStatus {
  return {
    camera_id: 'cam',
    label: 'cam',
    frames_seen: 0,
    frames_dropped: 0,
    detections_run: 0,
    escalations: 0,
    escalations_dropped: 0,
    discontinuities: 0,
    last_frame_at: null,
    // The wall-clock observation liveness actually reads; `last_frame_at` is the
    // camera's own source timeline and means something different.
    last_frame_epoch: null,
    last_escalation_at: null,
    // What an unconfigured camera loads with: both capabilities, every concern
    // kind, the stronger tier, and null durations meaning "follow the engine
    // default".
    capabilities: ['anomaly_detection', 'scene_description'],
    falls_suspected: 0,
    notify_on: ['collapse', 'altercation', 'self_harm', 'medication', 'distress', 'other'],
    notify_min_confidence: 'likely',
    clip_preroll_seconds: null,
    clip_postroll_seconds: null,
    summary_interval_seconds: null,
    ...overrides,
  }
}

function renderStage(props: {
  cameras: CameraStatus[]
  placements?: Record<string, NormalizedPoint>
  toneByCameraId?: Map<string, Tone>
  floorplanImage?: string | null
  editable?: boolean
  onPlace?: (cameraId: string, point: NormalizedPoint) => void
}) {
  const onPlace = props.onPlace ?? vi.fn()
  const utils = render(
    <MemoryRouter>
      <FloorplanStage
        cameras={props.cameras}
        placements={props.placements ?? {}}
        toneByCameraId={props.toneByCameraId ?? new Map()}
        floorplanImage={props.floorplanImage ?? null}
        editable={props.editable ?? false}
        onPlace={onPlace}
      />
    </MemoryRouter>,
  )
  return { ...utils, onPlace }
}

describe('FloorplanStage — background', () => {
  it('renders zone band labels (schematic empty state) when no floorplan image is set', () => {
    renderStage({
      cameras: [
        makeCamera({ camera_id: 'room_4b', zone: 'room' }),
        makeCamera({ camera_id: 'corridor_1', zone: 'corridor' }),
      ],
    })
    expect(screen.getByText('Rooms')).toBeInTheDocument()
    expect(screen.getByText('Corridors')).toBeInTheDocument()
  })

  it('renders the operator-supplied image and NOT the schematic bands once one is set', () => {
    renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      floorplanImage: 'data:image/png;base64,AAA',
    })
    expect(screen.queryByText('Rooms')).not.toBeInTheDocument()
    const img = screen.getByRole('presentation', { hidden: true }) as HTMLImageElement
    expect(img.src).toContain('data:image/png;base64,AAA')
  })
})

describe('FloorplanStage — markers', () => {
  it('renders one marker per camera, positioned at its resolved placement', () => {
    renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.25, y: 0.75 } },
    })
    const marker = screen.getByTestId('marker-room_4b')
    expect(marker.style.left).toBe('25%')
    expect(marker.style.top).toBe('75%')
  })

  it('marks a camera with no stored placement as not yet placed', () => {
    renderStage({ cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })] })
    expect(screen.getByLabelText(/not yet placed/i)).toBeInTheDocument()
  })

  it('does not say "not yet placed" for a camera the operator has explicitly positioned', () => {
    renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
    })
    expect(screen.queryByLabelText(/not yet placed/i)).not.toBeInTheDocument()
  })

  it('a marker in view mode is a link to that camera\'s own page', () => {
    renderStage({ cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })] })
    const link = screen.getByRole('link', { name: /room_4b/i })
    expect(link).toHaveAttribute('href', '/cameras/room_4b')
  })

  it('a marker in edit mode is a button, not a link — so clicking it does not navigate away mid-arrangement', () => {
    renderStage({ cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })], editable: true })
    expect(screen.queryByRole('link', { name: /room_4b/i })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /room_4b/i })).toBeInTheDocument()
  })
})

describe('FloorplanStage — keyboard placement', () => {
  it('ArrowRight nudges the focused marker right and commits through onPlace, for a keyboard-only operator', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: true,
    })
    const marker = screen.getByTestId('marker-room_4b')
    marker.focus()
    fireEvent.keyDown(marker, { key: 'ArrowRight' })
    expect(onPlace).toHaveBeenCalledWith('room_4b', { x: 0.52, y: 0.5 })
  })

  it('Shift+ArrowLeft nudges by the larger step', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: true,
    })
    const marker = screen.getByTestId('marker-room_4b')
    fireEvent.keyDown(marker, { key: 'ArrowLeft', shiftKey: true })
    expect(onPlace).toHaveBeenCalledWith('room_4b', { x: 0.4, y: 0.5 })
  })

  it('a key that is not an arrow does nothing', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: true,
    })
    const marker = screen.getByTestId('marker-room_4b')
    fireEvent.keyDown(marker, { key: 'Enter' })
    expect(onPlace).not.toHaveBeenCalled()
  })

  it('arrow keys do nothing outside edit mode, so browsing never accidentally rearranges the map', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: false,
    })
    const marker = screen.getByTestId('marker-room_4b')
    fireEvent.keyDown(marker, { key: 'ArrowRight' })
    expect(onPlace).not.toHaveBeenCalled()
  })
})

describe('FloorplanStage — pointer placement', () => {
  it('dragging a marker over the stage commits its new position through onPlace', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: true,
    })
    const stage = screen.getByTestId('floorplan-stage')
    vi.spyOn(stage, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 400,
      height: 200,
      right: 400,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })
    const marker = screen.getByTestId('marker-room_4b')
    fireEvent.pointerDown(marker, { pointerId: 1, buttons: 1 })
    fireEvent.pointerMove(marker, { pointerId: 1, buttons: 1, clientX: 200, clientY: 50 })
    expect(onPlace).toHaveBeenCalledWith('room_4b', { x: 0.5, y: 0.25 })
  })

  it('moving the pointer with no button held does not drag the marker', () => {
    const { onPlace } = renderStage({
      cameras: [makeCamera({ camera_id: 'room_4b', zone: 'room' })],
      placements: { room_4b: { x: 0.5, y: 0.5 } },
      editable: true,
    })
    const stage = screen.getByTestId('floorplan-stage')
    vi.spyOn(stage, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 400,
      height: 200,
      right: 400,
      bottom: 200,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })
    const marker = screen.getByTestId('marker-room_4b')
    fireEvent.pointerMove(marker, { pointerId: 1, buttons: 0, clientX: 200, clientY: 50 })
    expect(onPlace).not.toHaveBeenCalled()
  })
})

describe('FloorplanStage — tone-transition motion', () => {
  it('remounts the marker (a new DOM node) when its tone changes, replaying the entrance animation on the transition', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const { rerender } = render(
      <MemoryRouter>
        <FloorplanStage
          cameras={cameras}
          placements={{ room_4b: { x: 0.5, y: 0.5 } }}
          toneByCameraId={new Map([['room_4b', 'inert']])}
          floorplanImage={null}
          editable={false}
          onPlace={vi.fn()}
        />
      </MemoryRouter>,
    )
    const before = screen.getByTestId('marker-room_4b')

    rerender(
      <MemoryRouter>
        <FloorplanStage
          cameras={cameras}
          placements={{ room_4b: { x: 0.5, y: 0.5 } }}
          toneByCameraId={new Map([['room_4b', 'breach']])}
          floorplanImage={null}
          editable={false}
          onPlace={vi.fn()}
        />
      </MemoryRouter>,
    )
    const after = screen.getByTestId('marker-room_4b')
    expect(after).not.toBe(before)
    expect(document.body.contains(before)).toBe(false)
  })

  it('does NOT remount the marker on a re-render where the tone is unchanged, so the animation does not replay for no reason', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const { rerender } = render(
      <MemoryRouter>
        <FloorplanStage
          cameras={cameras}
          placements={{ room_4b: { x: 0.5, y: 0.5 } }}
          toneByCameraId={new Map([['room_4b', 'caution']])}
          floorplanImage={null}
          editable={false}
          onPlace={vi.fn()}
        />
      </MemoryRouter>,
    )
    const before = screen.getByTestId('marker-room_4b')

    // Same tone, different unrelated prop (placements object is a new
    // reference but the same value) — a plain re-render, not a transition.
    rerender(
      <MemoryRouter>
        <FloorplanStage
          cameras={cameras}
          placements={{ room_4b: { x: 0.5, y: 0.5 } }}
          toneByCameraId={new Map([['room_4b', 'caution']])}
          floorplanImage={null}
          editable={false}
          onPlace={vi.fn()}
        />
      </MemoryRouter>,
    )
    const after = screen.getByTestId('marker-room_4b')
    expect(after).toBe(before)
  })
})
