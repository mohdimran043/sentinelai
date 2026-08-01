import { afterEach, describe, expect, it, vi } from 'vitest'
import { canPlayHls, supportsMediaSourceHls, supportsNativeHls } from '@/live/hlsSupport'

describe('supportsNativeHls', () => {
  it('is false in this project\'s real test environment (jsdom implements no media stack)', () => {
    // Deliberately not mocked: jsdom's real `canPlayType` always returns ''.
    // This is the honest "this browser can't play it" case exercised for
    // free, with no fake standing in for a real capability check.
    const video = document.createElement('video')
    expect(supportsNativeHls(video)).toBe(false)
  })

  it('reads true when the element actually reports HLS support (Safari)', () => {
    const video = document.createElement('video')
    vi.spyOn(video, 'canPlayType').mockReturnValue('maybe')
    expect(supportsNativeHls(video)).toBe(true)
  })

  it('reads false when the element explicitly says no', () => {
    const video = document.createElement('video')
    vi.spyOn(video, 'canPlayType').mockReturnValue('')
    expect(supportsNativeHls(video)).toBe(false)
  })
})

describe('supportsMediaSourceHls', () => {
  const originalMediaSource = (window as { MediaSource?: unknown }).MediaSource

  afterEach(() => {
    if (originalMediaSource === undefined) {
      delete (window as { MediaSource?: unknown }).MediaSource
    } else {
      ;(window as { MediaSource?: unknown }).MediaSource = originalMediaSource
    }
  })

  it('is false in this project\'s real test environment (jsdom has no MediaSource)', () => {
    expect(supportsMediaSourceHls()).toBe(false)
  })

  it('is true once MediaSource exists on window', () => {
    ;(window as { MediaSource?: unknown }).MediaSource = function MediaSource() {}
    expect(supportsMediaSourceHls()).toBe(true)
  })
})

describe('canPlayHls', () => {
  afterEach(() => {
    delete (window as { MediaSource?: unknown }).MediaSource
  })

  it('is false when neither native nor MSE support exists — the real jsdom case', () => {
    const video = document.createElement('video')
    expect(canPlayHls(video)).toBe(false)
  })

  it('is true from native support alone', () => {
    const video = document.createElement('video')
    vi.spyOn(video, 'canPlayType').mockReturnValue('probably')
    expect(canPlayHls(video)).toBe(true)
  })

  it('is true from MediaSource support alone', () => {
    ;(window as { MediaSource?: unknown }).MediaSource = function MediaSource() {}
    const video = document.createElement('video')
    expect(canPlayHls(video)).toBe(true)
  })
})
