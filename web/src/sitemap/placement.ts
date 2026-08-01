/**
 * Pure geometry for the site map. Nothing here touches the DOM, React or
 * storage — see `siteMapStore.ts` for persistence and `FloorplanStage.tsx`
 * for the pointer/keyboard wiring built on top of it.
 *
 * Every position is normalized to 0..1 on both axes, not pixels: the same
 * placement then re-renders correctly at any plan size (a phone screen, a
 * wall display, a resized window) without ever being re-computed or
 * migrated. `x`/`y` are always kept inside [0, 1] — `clampUnit`/`clampPoint`
 * are the single choke point that guarantees it, so a stray pointer event or
 * a corrupted `localStorage` value can never place a marker off the plan.
 */

export interface NormalizedPoint {
  x: number
  y: number
}

export function clampUnit(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(1, Math.max(0, value))
}

export function clampPoint(point: NormalizedPoint): NormalizedPoint {
  return { x: clampUnit(point.x), y: clampUnit(point.y) }
}

/** The subset of `DOMRect` this module actually needs — kept narrow so a test can supply a plain object instead of a real `DOMRect`. */
export type StageRect = Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>

/**
 * Converts a pointer's viewport position into a normalized point relative to
 * the stage's own bounding rect, clamped to stay on the plan. A
 * zero-area rect (the stage has not laid out yet, e.g. `display: none`)
 * yields the centre point rather than dividing by zero.
 */
export function pointFromClientPosition(
  clientX: number,
  clientY: number,
  stageRect: StageRect,
): NormalizedPoint {
  if (stageRect.width <= 0 || stageRect.height <= 0) return { x: 0.5, y: 0.5 }
  return clampPoint({
    x: (clientX - stageRect.left) / stageRect.width,
    y: (clientY - stageRect.top) / stageRect.height,
  })
}

export type NudgeDirection = 'up' | 'down' | 'left' | 'right'

/** The default keyboard nudge step, and the larger one used with Shift held. */
export const NUDGE_STEP = 0.02
export const NUDGE_STEP_LARGE = 0.1

/** Moves a normalized point by `step` in one of the four directions, clamped to the plan. */
export function nudgePoint(
  point: NormalizedPoint,
  direction: NudgeDirection,
  step: number = NUDGE_STEP,
): NormalizedPoint {
  switch (direction) {
    case 'up':
      return clampPoint({ x: point.x, y: point.y - step })
    case 'down':
      return clampPoint({ x: point.x, y: point.y + step })
    case 'left':
      return clampPoint({ x: point.x - step, y: point.y })
    case 'right':
      return clampPoint({ x: point.x + step, y: point.y })
  }
}

const NUDGE_KEYS: Record<string, NudgeDirection> = {
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
}

/** Maps a keyboard event's `key` to a nudge direction, or `undefined` for any other key. */
export function nudgeDirectionForKey(key: string): NudgeDirection | undefined {
  return NUDGE_KEYS[key]
}
