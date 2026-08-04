import { describe, expect, it } from 'vitest'
import {
  UNGROUPED_BAND_KEY,
  buildSchematicBands,
  buildSchematicPositions,
  resolveEffectivePlacements,
} from '@/sitemap/schematicLayout'
import type { CameraStatus } from '@/api/engineClient'

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
    last_escalation_at: null,
    // The welfare policy an unconfigured camera loads with: every concern kind,
    // the stronger tier, and null durations meaning "follow the engine default".
    notify_on: ['collapse', 'altercation', 'self_harm', 'medication', 'distress', 'other'],
    notify_min_confidence: 'likely',
    clip_preroll_seconds: null,
    clip_postroll_seconds: null,
    summary_interval_seconds: null,
    ...overrides,
  }
}

describe('buildSchematicBands', () => {
  it('groups cameras by zone in the fixed room/corridor/dayroom order regardless of input order', () => {
    const cameras = [
      makeCamera({ camera_id: 'dayroom_1', zone: 'dayroom' }),
      makeCamera({ camera_id: 'room_4b', zone: 'room' }),
      makeCamera({ camera_id: 'corridor_1', zone: 'corridor' }),
    ]
    const bands = buildSchematicBands(cameras)
    expect(bands.map((band) => band.key)).toEqual(['room', 'corridor', 'dayroom'])
  })

  it('puts a camera with zone: null in its own trailing "ungrouped" band rather than a real zone', () => {
    const cameras = [
      makeCamera({ camera_id: 'room_2a', zone: 'room' }),
      makeCamera({ camera_id: 'nobody_asked', zone: null }),
    ]
    const bands = buildSchematicBands(cameras)
    expect(bands.at(-1)?.key).toBe(UNGROUPED_BAND_KEY)
    expect(bands.at(-1)?.cameraIds).toEqual(['nobody_asked'])
  })

  it('omits a zone band entirely when no camera belongs to it, rather than rendering an empty strip', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const bands = buildSchematicBands(cameras)
    expect(bands.map((band) => band.key)).toEqual(['room'])
  })

  it('groups every camera sharing a zone into the same band', () => {
    const cameras = [
      makeCamera({ camera_id: 'room_a', zone: 'room' }),
      makeCamera({ camera_id: 'room_b', zone: 'room' }),
    ]
    const bands = buildSchematicBands(cameras)
    expect(bands).toHaveLength(1)
    expect(bands[0]!.cameraIds.sort()).toEqual(['room_a', 'room_b'])
  })
})

describe('buildSchematicPositions', () => {
  it('returns a normalized point strictly inside the plan for every camera', () => {
    const cameras = [
      makeCamera({ camera_id: 'room_4b', zone: 'room' }),
      makeCamera({ camera_id: 'corridor_1', zone: 'corridor' }),
      makeCamera({ camera_id: 'dayroom_1', zone: 'dayroom' }),
      makeCamera({ camera_id: 'nobody_asked', zone: null }),
    ]
    const positions = buildSchematicPositions(cameras)
    expect(positions.size).toBe(4)
    for (const point of positions.values()) {
      expect(point.x).toBeGreaterThan(0)
      expect(point.x).toBeLessThan(1)
      expect(point.y).toBeGreaterThan(0)
      expect(point.y).toBeLessThan(1)
    }
  })

  it('stacks zone bands vertically in order: room above corridor above dayroom', () => {
    const cameras = [
      makeCamera({ camera_id: 'room_4b', zone: 'room' }),
      makeCamera({ camera_id: 'corridor_1', zone: 'corridor' }),
      makeCamera({ camera_id: 'dayroom_1', zone: 'dayroom' }),
    ]
    const positions = buildSchematicPositions(cameras)
    const roomY = positions.get('room_4b')!.y
    const corridorY = positions.get('corridor_1')!.y
    const dayroomY = positions.get('dayroom_1')!.y
    expect(roomY).toBeLessThan(corridorY)
    expect(corridorY).toBeLessThan(dayroomY)
  })

  it('spreads two cameras in the same zone apart on the x axis rather than stacking them on top of each other', () => {
    const cameras = [
      makeCamera({ camera_id: 'room_a', zone: 'room' }),
      makeCamera({ camera_id: 'room_b', zone: 'room' }),
    ]
    const positions = buildSchematicPositions(cameras)
    expect(positions.get('room_a')!.x).not.toBe(positions.get('room_b')!.x)
    expect(positions.get('room_a')!.y).toBe(positions.get('room_b')!.y)
  })

  it('returns an empty map for an empty camera list', () => {
    expect(buildSchematicPositions([]).size).toBe(0)
  })
})

describe('resolveEffectivePlacements', () => {
  it('uses the operator-stored placement when one exists, marked as not-default', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const resolved = resolveEffectivePlacements(cameras, { room_4b: { x: 0.9, y: 0.1 } })
    expect(resolved.get('room_4b')).toEqual({ point: { x: 0.9, y: 0.1 }, isDefault: false })
  })

  it('falls back to the schematic position when no placement is stored, marked as default', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const resolved = resolveEffectivePlacements(cameras, {})
    const schematic = buildSchematicPositions(cameras).get('room_4b')
    expect(resolved.get('room_4b')).toEqual({ point: schematic, isDefault: true })
  })

  it('clamps a corrupted stored placement rather than rendering a marker off the plan', () => {
    const cameras = [makeCamera({ camera_id: 'room_4b', zone: 'room' })]
    const resolved = resolveEffectivePlacements(cameras, { room_4b: { x: 5, y: -5 } })
    expect(resolved.get('room_4b')).toEqual({ point: { x: 1, y: 0 }, isDefault: false })
  })

  it('resolves every camera, including one nobody has placed and one already stored, in the same call', () => {
    const cameras = [
      makeCamera({ camera_id: 'placed', zone: 'room' }),
      makeCamera({ camera_id: 'unplaced', zone: 'room' }),
    ]
    const resolved = resolveEffectivePlacements(cameras, { placed: { x: 0.2, y: 0.8 } })
    expect(resolved.get('placed')!.isDefault).toBe(false)
    expect(resolved.get('unplaced')!.isDefault).toBe(true)
  })
})
