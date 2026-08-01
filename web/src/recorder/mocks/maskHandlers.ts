import { http, HttpResponse } from 'msw'
import { RECORDER_BASE_URL } from '@/recorder/config'
import { REAL_MASK_ROOM_4B, REAL_MASK_VALIDATE_VALID_SINGLE } from '@/recorder/mocks/maskFixtures'
import type { RecorderMaskResponse, RecorderMaskValidation } from '@/recorder/recorder.types'

/**
 * MSW handlers for the mask editor endpoints — **tests only**. Kept apart from
 * `recorder/mocks/handlers.ts` for the same reason `maskFixtures.ts` is kept
 * apart from `fixtures.ts`: that file had several other sections landing in it
 * at the same time this was written. `handlers.ts` only needs one import and
 * one line spreading `maskHandlers` into `recorderHandlers`.
 */

export const recorderMaskHandler = (body: RecorderMaskResponse) =>
  http.get(`${RECORDER_BASE_URL}/masks/:cameraId`, () => HttpResponse.json(body))

/** `POST .../validate` — always HTTP 200, valid or not; see `RecorderMaskValidation`. */
export const recorderMaskValidateHandler = (
  body: RecorderMaskValidation,
  options: { onRequest?: (regions: unknown) => void } = {},
) =>
  http.post(`${RECORDER_BASE_URL}/masks/:cameraId/validate`, async ({ request }) => {
    options.onRequest?.((await request.json()) as { regions: unknown })
    return HttpResponse.json(body)
  })

/** `PUT .../{camera_id}` — HTTP 200 when it saves, HTTP 422 when it refuses to. */
export const recorderMaskSaveHandler = (
  body: RecorderMaskValidation,
  options: { onRequest?: (regions: unknown) => void } = {},
) =>
  http.put(`${RECORDER_BASE_URL}/masks/:cameraId`, async ({ request }) => {
    options.onRequest?.((await request.json()) as { regions: unknown })
    return HttpResponse.json(body, { status: body.valid ? 200 : 422 })
  })

/**
 * A tiny valid JPEG (a 1x1 black pixel) — enough for an `<img>` element to
 * have a resolvable `src`. jsdom never decodes it, so pixel content does not
 * matter; only that the byte stream and `Content-Type` are genuinely there.
 */
const TINY_JPEG_BASE64 =
  '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k='

function tinyJpegBytes(): Uint8Array {
  const binary = atob(TINY_JPEG_BASE64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

/** `calibration_available: true` — the calibration frame is genuinely served. */
export const recorderCalibrationFrameHandler = () =>
  http.get(`${RECORDER_BASE_URL}/masks/:cameraId/calibration-frame`, () =>
    HttpResponse.arrayBuffer(tinyJpegBytes().buffer as ArrayBuffer, {
      headers: { 'Content-Type': 'image/jpeg' },
    }),
  )

/**
 * `calibration_available: false` — VERIFIED 2026-08-01 live: the recorder
 * answers this exact 404 shape (its generic "no such endpoint" body, not a
 * refusal-specific one) whenever the owning camera is recording.
 */
export const recorderCalibrationFrameMissingHandler = () =>
  http.get(`${RECORDER_BASE_URL}/masks/:cameraId/calibration-frame`, ({ params }) =>
    HttpResponse.json(
      { error: `no such endpoint: GET /api/masks/${String(params.cameraId)}/calibration-frame` },
      { status: 404 },
    ),
  )

/**
 * Default handlers: the real, currently-recording `room_4b` state (matches
 * every camera on the reference instance today). Tests override per case with
 * `server.use(...)`.
 */
export const maskHandlers = [
  recorderMaskHandler(REAL_MASK_ROOM_4B),
  recorderMaskValidateHandler(REAL_MASK_VALIDATE_VALID_SINGLE),
  recorderMaskSaveHandler(REAL_MASK_VALIDATE_VALID_SINGLE),
  recorderCalibrationFrameMissingHandler(),
]
