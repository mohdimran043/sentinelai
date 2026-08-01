import type {
  RecorderMaskRegion,
  RecorderMaskResponse,
  RecorderMaskValidation,
} from '@/recorder/recorder.types'

/**
 * Fixtures for `GET /api/masks/{camera_id}`, `PUT /api/masks/{camera_id}` and
 * `POST /api/masks/{camera_id}/validate` — captured live 2026-08-01.
 *
 * Kept in a file of their own (see `recorder/mocks/fixtures.ts` for the rest of
 * the recorder's fixtures) because several other sections were landing in that
 * shared file at the same time this was written, and a plain append is not
 * safe against a concurrent full-file rewrite. This file has exactly one
 * writer.
 *
 * Every camera on the reference instance is `recording`, so
 * `calibration_available: true` and a real calibration-frame image were never
 * observed live. `REAL_MASK_CALIBRATION_AVAILABLE` is `REAL_MASK_ROOM_4B` with
 * only that flag (and the now-inapplicable refusal) changed — not a second
 * live capture. Everything else below is verbatim.
 */

export const REAL_MASK_ROOM_4B: RecorderMaskResponse = {
  camera_id: 'room_4b',
  mode: 'room',
  frame_width: 1280,
  frame_height: 720,
  mask_path: '/var/tmp/sentinel-wall/masks/room_4b.json',
  regions: [
    { region_id: 'toilet', polygon: [[80, 90], [420, 90], [420, 320], [80, 320]] },
    { region_id: 'washbasin', polygon: [[900, 400], [1180, 400], [1180, 600], [900, 600]] },
  ],
  editable: true,
  recording: true,
  calibration_available: false,
  calibration_refusal:
    'This camera is recording. An unmasked frame is not served for a camera under observation — stop it first. Regions can still be adjusted against the recorded frame, which is already masked.',
  inventory: {
    path: '/var/tmp/sentinel-wall/masks/room_4b.json',
    regions: [
      { region_id: 'toilet', area_px: 78200, area_pct: 8.485243055555555, bbox: [80, 90, 420, 320] },
      { region_id: 'washbasin', area_px: 56000, area_pct: 6.076388888888889, bbox: [900, 400, 1180, 600] },
    ],
    total_px: 134200,
    total_pct: 14.561631944444445,
    frame_px: 921600,
    may_overlap: false,
    note: 'Masked regions are never observed and never recorded. Coverage percentages elsewhere describe only the unmasked part of the frame.',
  },
  note: "Polygon coordinates are in this camera's configured frame pixels (frame_width x frame_height), not in the pixels of the snapshot image, which is downscaled. A polygon sent in image coordinates validates cleanly and masks the wrong part of the room.",
}

/**
 * `corridor_1`: no mask configured. `inventory` is genuinely absent on the
 * wire when `regions` is empty — not `undefined` standing in for a missing
 * capture.
 */
export const REAL_MASK_EMPTY_CORRIDOR_1: RecorderMaskResponse = {
  camera_id: 'corridor_1',
  mode: 'common_area',
  frame_width: 1280,
  frame_height: 720,
  mask_path: '',
  regions: [],
  editable: true,
  recording: true,
  calibration_available: false,
  calibration_refusal:
    'This camera is recording. An unmasked frame is not served for a camera under observation — stop it first. Regions can still be adjusted against the recorded frame, which is already masked.',
  note: "Polygon coordinates are in this camera's configured frame pixels (frame_width x frame_height), not in the pixels of the snapshot image, which is downscaled. A polygon sent in image coordinates validates cleanly and masks the wrong part of the room.",
}

/**
 * NOT a second live capture — see the file-level note above. Exists only to
 * exercise the branch the reference instance cannot currently produce (every
 * camera on it is recording).
 */
export const REAL_MASK_CALIBRATION_AVAILABLE: RecorderMaskResponse = {
  ...REAL_MASK_ROOM_4B,
  recording: false,
  calibration_available: true,
  calibration_refusal: undefined,
}

export function makeMaskResponse(overrides: Partial<RecorderMaskResponse> = {}): RecorderMaskResponse {
  return { ...REAL_MASK_ROOM_4B, ...overrides }
}

/* --- POST .../validate and PUT .../{camera_id} response bodies --- */

export const REAL_MASK_VALIDATE_VALID_SINGLE: RecorderMaskValidation = {
  valid: true,
  regions: [{ region_id: 'toilet', valid: true }],
  inventory: {
    path: '',
    regions: [{ region_id: 'toilet', area_px: 78200, area_pct: 8.485243055555555, bbox: [80, 90, 420, 320] }],
    total_px: 78200,
    total_pct: 8.485243055555555,
    frame_px: 921600,
    may_overlap: false,
    note: 'Masked regions are never observed and never recorded. Coverage percentages elsewhere describe only the unmasked part of the frame.',
  },
}

export const REAL_MASK_VALIDATE_NO_REGIONS: RecorderMaskValidation = {
  valid: false,
  error: 'mask set contains no regions',
  regions: [],
}

export const REAL_MASK_VALIDATE_TOO_FEW_POINTS: RecorderMaskValidation = {
  valid: false,
  error: 'region "x": has 2 points; need at least 3',
  regions: [{ region_id: 'x', valid: false, error: 'region "x": has 2 points; need at least 3' }],
}

export const REAL_MASK_VALIDATE_OUT_OF_BOUNDS: RecorderMaskValidation = {
  valid: false,
  error: 'region "x": point (9999,9999) outside 1280x720',
  regions: [{ region_id: 'x', valid: false, error: 'region "x": point (9999,9999) outside 1280x720' }],
}

export const REAL_MASK_VALIDATE_DUPLICATE_ID: RecorderMaskValidation = {
  valid: false,
  error: 'duplicate region_id "a"',
  regions: [
    { region_id: 'a', valid: true },
    { region_id: 'a', valid: true },
  ],
}

/**
 * A three-point triangle well inside a 1280x720 frame — enough to pass the
 * recorder's minimum-point-count and bounds rules for editor tests that just
 * need *a* valid new region.
 */
export const DRAFT_TRIANGLE: RecorderMaskRegion = {
  region_id: 'drawer',
  polygon: [[500, 500], [600, 500], [550, 600]],
}
