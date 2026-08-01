import { describe, expect, it } from 'vitest'
import { displayPointToFramePoint, polygonPointsAttr } from '@/recorder/maskGeometry'

describe('displayPointToFramePoint', () => {
  it('scales a click up when the canvas is displayed smaller than the real frame', () => {
    // 1280x720 frame shown in a 640x360 box — exactly half size, both axes.
    const framePoint = displayPointToFramePoint([100, 50], { width: 640, height: 360 }, {
      frameWidth: 1280,
      frameHeight: 720,
    })
    expect(framePoint).toEqual([200, 100])
  })

  it('scales X and Y independently, not by one shared factor', () => {
    // A 1000x900 frame shown in a 500x300 box: 2x on X, 3x on Y. A broken
    // implementation that derives a single scale (e.g. from width only, or
    // averages the two axes) gets Y wrong here while X still happens to pass.
    const framePoint = displayPointToFramePoint([10, 10], { width: 500, height: 300 }, {
      frameWidth: 1000,
      frameHeight: 900,
    })
    expect(framePoint).toEqual([20, 30])
  })

  it('is the identity when the canvas is shown at native size', () => {
    const framePoint = displayPointToFramePoint([80, 90], { width: 1280, height: 720 }, {
      frameWidth: 1280,
      frameHeight: 720,
    })
    expect(framePoint).toEqual([80, 90])
  })

  it('maps the origin to the origin regardless of scale', () => {
    expect(
      displayPointToFramePoint([0, 0], { width: 320, height: 180 }, { frameWidth: 1280, frameHeight: 720 }),
    ).toEqual([0, 0])
  })

  it('does not divide by zero when the canvas has not been laid out yet', () => {
    expect(
      displayPointToFramePoint([50, 50], { width: 0, height: 0 }, { frameWidth: 1280, frameHeight: 720 }),
    ).toEqual([0, 0])
  })
})

describe('polygonPointsAttr', () => {
  it('renders frame-pixel coordinates verbatim as an SVG points string', () => {
    expect(
      polygonPointsAttr([
        [80, 90],
        [420, 90],
        [420, 320],
        [80, 320],
      ]),
    ).toBe('80,90 420,90 420,320 80,320')
  })

  it('renders an empty polygon as an empty string rather than throwing', () => {
    expect(polygonPointsAttr([])).toBe('')
  })
})
