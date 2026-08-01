/**
 * Pure coordinate mapping between the mask editor's on-screen canvas and the
 * camera's configured frame pixel space.
 *
 * `GET /api/masks/{camera_id}` is explicit (see its `note` field, verified live
 * 2026-08-01 against `room_4b`): polygon coordinates are in the camera's
 * CONFIGURED frame pixels (`frame_width` x `frame_height`), never in the
 * pixels of whatever image is actually on screen. The calibration frame — and
 * the neutral placeholder shown when no frame can be served — is laid out at
 * whatever size the page gives it, which is routinely smaller than the real
 * frame (a 1280x720 camera rendered in a ~600px-wide panel). Every point an
 * operator clicks has to be rescaled before it means anything to the
 * recorder. Getting this backwards does not fail loudly — it silently draws
 * a mask over the wrong part of the room, which in a custodial deployment is
 * the one outcome this whole feature exists to prevent.
 *
 * Kept as a pure, DOM-free function so the mapping can be tested with plain
 * numbers instead of a real layout, and so `MasksPage.tsx` has exactly one
 * place that does this arithmetic.
 */

export interface FrameSize {
  frameWidth: number
  frameHeight: number
}

export interface DisplaySize {
  width: number
  height: number
}

/**
 * A point the operator clicked, already relative to the displayed canvas's own
 * top-left corner (i.e. `clientX - rect.left`, `clientY - rect.top`), mapped
 * into the camera's configured frame pixel space.
 *
 * Scaled per axis rather than by one shared factor. In practice the canvas is
 * laid out with `aspect-ratio: frameWidth / frameHeight`, so X and Y happen to
 * scale by the same amount — but nothing here assumes that. A caller whose
 * container is not exactly locked to the frame's aspect ratio (a rounding
 * error, a resize mid-drag) still gets a correct per-axis mapping instead of a
 * silently distorted one.
 */
export function displayPointToFramePoint(
  displayPoint: readonly [number, number],
  display: DisplaySize,
  frame: FrameSize,
): [number, number] {
  if (display.width <= 0 || display.height <= 0) return [0, 0]
  const scaleX = frame.frameWidth / display.width
  const scaleY = frame.frameHeight / display.height
  const [x, y] = displayPoint
  return [roundToTenth(x * scaleX), roundToTenth(y * scaleY)]
}

function roundToTenth(value: number): number {
  return Math.round(value * 10) / 10
}

/**
 * SVG `points` attribute value for a polygon already expressed in frame-pixel
 * space. The canvas's `<svg>` uses `viewBox="0 0 frameWidth frameHeight"`, so
 * these coordinates are handed to the browser as-is — the viewBox transform,
 * not this function, is what scales them to the displayed size. That keeps
 * rendering and the click-mapping above as two independent, separately
 * checkable pieces of arithmetic rather than one that has to get the same
 * scale factor right twice.
 */
export function polygonPointsAttr(polygon: readonly (readonly [number, number])[]): string {
  return polygon.map(([x, y]) => `${x},${y}`).join(' ')
}
