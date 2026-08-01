import { describe, expect, it } from 'vitest'
import {
  formatFrameAge,
  formatResolution,
  joinCamerasWithStatus,
} from '@/recorder/cameraJoin'
import type { RecorderCamera, RecorderWorkerStatus } from '@/recorder/recorder.types'

function camera(overrides: Partial<RecorderCamera> = {}): RecorderCamera {
  return {
    id: 'room_4b',
    name: 'Room 4B',
    mode: 'room',
    space_type: 'room',
    source: 'v4l2:/dev/video0',
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
    ...overrides,
  }
}

function status(overrides: Partial<RecorderWorkerStatus> = {}): RecorderWorkerStatus {
  return {
    camera_id: 'room_4b',
    mode: 'room',
    integrity_state: 'HEALTHY',
    running: true,
    failed: false,
    restarts: 0,
    last_frame_age_ms: 100,
    frames_dropped: 0,
    ...overrides,
  }
}

describe('joinCamerasWithStatus', () => {
  it('attaches each camera to the status entry with the matching id', () => {
    const joined = joinCamerasWithStatus(
      [camera({ id: 'room_4b' }), camera({ id: 'corridor_1', name: 'Corridor 1' })],
      [status({ camera_id: 'corridor_1', integrity_state: 'LOST' }), status({ camera_id: 'room_4b' })],
    )

    expect(joined).toHaveLength(2)
    expect(joined.find((row) => row.camera.id === 'room_4b')?.status?.integrity_state).toBe('HEALTHY')
    expect(joined.find((row) => row.camera.id === 'corridor_1')?.status?.integrity_state).toBe('LOST')
  })

  it('leaves status undefined, not fabricated, when the recorder reports none for a registered camera', () => {
    const joined = joinCamerasWithStatus([camera({ id: 'room_2a' })], [])

    expect(joined).toHaveLength(1)
    expect(joined[0]!.status).toBeUndefined()
  })

  it('drops a status entry for a camera absent from the registry rather than inventing a phantom row', () => {
    const joined = joinCamerasWithStatus(
      [camera({ id: 'room_4b' })],
      [status({ camera_id: 'room_4b' }), status({ camera_id: 'ghost_cam' })],
    )

    expect(joined).toHaveLength(1)
    expect(joined.map((row) => row.camera.id)).toEqual(['room_4b'])
  })

  it('preserves the registry order and count even with zero cameras', () => {
    expect(joinCamerasWithStatus([], [status()])).toEqual([])
  })
})

describe('formatFrameAge', () => {
  it('renders the -1 sentinel as "no frame yet", never as a negative age', () => {
    expect(formatFrameAge(-1)).toBe('no frame yet')
  })

  it('renders sub-second ages in milliseconds', () => {
    expect(formatFrameAge(0)).toBe('0 ms')
    expect(formatFrameAge(999)).toBe('999 ms')
  })

  it('renders ages of a second or more in seconds, one decimal place', () => {
    expect(formatFrameAge(1000)).toBe('1.0 s')
    expect(formatFrameAge(2415)).toBe('2.4 s')
  })
})

describe('formatResolution', () => {
  it('combines width, height and fps into one reading', () => {
    expect(formatResolution({ width: 1280, height: 720, fps: 15 })).toBe('1280×720 @ 15fps')
  })
})
