import type { RecorderCamera, RecorderWorkerStatus } from '@/recorder/recorder.types'

/**
 * `GET /api/cameras` (the registry) and `GET /api/status` (live worker state)
 * are two separate responses keyed the same way — `id` on one, `camera_id` on
 * the other — and nothing in either payload joins them for you. This is that
 * join, kept pure and separate from `CamerasPage` so it can be tested without
 * rendering anything, the same way `alertQuery.ts` backs `AlertsPage`.
 */
export interface JoinedCamera {
  camera: RecorderCamera
  /**
   * `undefined` when the recorder's registry knows about a camera that
   * `/api/status` says nothing about. That is a real, renderable state — not
   * an error to swallow by inventing a fake "unknown" status row.
   */
  status: RecorderWorkerStatus | undefined
}

/**
 * Driven off the registry, not off `/api/status`: the registry is the
 * recorder's own account of which cameras exist. A status entry with no
 * matching camera in the registry is dropped rather than surfaced as a phantom
 * camera — `/api/status` describes workers for registered cameras, it does not
 * define what a camera is.
 */
export function joinCamerasWithStatus(
  cameras: RecorderCamera[],
  statuses: RecorderWorkerStatus[],
): JoinedCamera[] {
  const statusById = new Map(statuses.map((status) => [status.camera_id, status]))
  return cameras.map((camera) => ({ camera, status: statusById.get(camera.id) }))
}

/**
 * `last_frame_age_ms` uses `-1` as a sentinel for "no frame has ever
 * arrived" — not an age, and must never be formatted as "-1 ms" or, worse,
 * silently clamped to zero and read as "just now".
 */
export function formatFrameAge(lastFrameAgeMs: number): string {
  if (lastFrameAgeMs < 0) return 'no frame yet'
  if (lastFrameAgeMs < 1000) return `${lastFrameAgeMs} ms`
  return `${(lastFrameAgeMs / 1000).toFixed(1)} s`
}

/** `1280, 720, 15` -> `"1280×720 @ 15fps"`. */
export function formatResolution(camera: Pick<RecorderCamera, 'width' | 'height' | 'fps'>): string {
  return `${camera.width}×${camera.height} @ ${camera.fps}fps`
}

/**
 * The four fields the recorder's own `context_note` says are stored and
 * validated but read by nothing. Centralised here so `CamerasPage` and its
 * tests share one list instead of two that can drift apart.
 */
export const INERT_CAMERA_FIELDS = [
  'elevated_watch',
  'capacity',
  'audio_enabled',
  'face_recognition_enabled',
] as const
