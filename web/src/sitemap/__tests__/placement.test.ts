import { describe, expect, it } from 'vitest'
import {
  NUDGE_STEP,
  clampPoint,
  clampUnit,
  nudgeDirectionForKey,
  nudgePoint,
  pointFromClientPosition,
} from '@/sitemap/placement'

describe('clampUnit', () => {
  it('leaves an in-range value untouched', () => {
    expect(clampUnit(0.42)).toBe(0.42)
  })

  it('clamps a value above 1 down to 1', () => {
    expect(clampUnit(1.7)).toBe(1)
  })

  it('clamps a value below 0 up to 0', () => {
    expect(clampUnit(-0.3)).toBe(0)
  })

  it('treats NaN as 0 rather than propagating it onto the plan', () => {
    expect(clampUnit(Number.NaN)).toBe(0)
  })
})

describe('clampPoint', () => {
  it('clamps both axes independently', () => {
    expect(clampPoint({ x: 1.5, y: -0.5 })).toEqual({ x: 1, y: 0 })
  })
})

describe('pointFromClientPosition', () => {
  const stageRect = { left: 100, top: 50, width: 400, height: 200 }

  it('maps a client position inside the stage to the matching normalized point', () => {
    expect(pointFromClientPosition(300, 150, stageRect)).toEqual({ x: 0.5, y: 0.5 })
  })

  it('maps the stage top-left corner to (0, 0)', () => {
    expect(pointFromClientPosition(100, 50, stageRect)).toEqual({ x: 0, y: 0 })
  })

  it('clamps a client position outside the stage rather than placing the marker off-plan', () => {
    expect(pointFromClientPosition(-1000, -1000, stageRect)).toEqual({ x: 0, y: 0 })
    expect(pointFromClientPosition(10_000, 10_000, stageRect)).toEqual({ x: 1, y: 1 })
  })

  it('falls back to the centre rather than dividing by zero when the stage has not laid out yet', () => {
    expect(pointFromClientPosition(300, 150, { left: 0, top: 0, width: 0, height: 0 })).toEqual({
      x: 0.5,
      y: 0.5,
    })
  })
})

describe('nudgePoint', () => {
  const centre = { x: 0.5, y: 0.5 }

  it('moves up by decreasing y', () => {
    expect(nudgePoint(centre, 'up', 0.1)).toEqual({ x: 0.5, y: 0.4 })
  })

  it('moves down by increasing y', () => {
    expect(nudgePoint(centre, 'down', 0.1)).toEqual({ x: 0.5, y: 0.6 })
  })

  it('moves left by decreasing x', () => {
    expect(nudgePoint(centre, 'left', 0.1)).toEqual({ x: 0.4, y: 0.5 })
  })

  it('moves right by increasing x', () => {
    expect(nudgePoint(centre, 'right', 0.1)).toEqual({ x: 0.6, y: 0.5 })
  })

  it('clamps at the edge instead of moving past it', () => {
    expect(nudgePoint({ x: 0, y: 0 }, 'left', NUDGE_STEP)).toEqual({ x: 0, y: 0 })
    expect(nudgePoint({ x: 1, y: 1 }, 'down', NUDGE_STEP)).toEqual({ x: 1, y: 1 })
  })
})

describe('nudgeDirectionForKey', () => {
  it('maps the four arrow keys to their directions', () => {
    expect(nudgeDirectionForKey('ArrowUp')).toBe('up')
    expect(nudgeDirectionForKey('ArrowDown')).toBe('down')
    expect(nudgeDirectionForKey('ArrowLeft')).toBe('left')
    expect(nudgeDirectionForKey('ArrowRight')).toBe('right')
  })

  it('returns undefined for a key that is not a nudge, so a marker does not move on every keystroke', () => {
    expect(nudgeDirectionForKey('Enter')).toBeUndefined()
    expect(nudgeDirectionForKey('a')).toBeUndefined()
  })
})
