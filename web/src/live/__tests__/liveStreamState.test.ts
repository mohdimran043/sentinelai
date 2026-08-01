import { describe, expect, it } from 'vitest'
import {
  initialLiveStreamState,
  nextLiveStreamState,
  type LiveStreamState,
} from '@/live/liveStreamState'

describe('initialLiveStreamState', () => {
  it('starts unsupported browsers in the terminal unsupported state, never checking', () => {
    expect(initialLiveStreamState(false)).toEqual({ kind: 'unsupported' })
  })

  it('starts a supported browser checking, never assuming live', () => {
    expect(initialLiveStreamState(true)).toEqual({ kind: 'checking' })
  })
})

describe('nextLiveStreamState', () => {
  it('goes live the first time the manifest is reachable', () => {
    const next = nextLiveStreamState({ kind: 'checking' }, { type: 'probe-ok' })
    expect(next).toEqual({ kind: 'live' })
  })

  it('becomes unavailable, not stuck checking or broken, the first time the manifest fails', () => {
    const next = nextLiveStreamState({ kind: 'checking' }, { type: 'probe-fail' })
    expect(next).toEqual({ kind: 'unavailable', attempt: 1 })
  })

  it('counts up consecutive unavailable attempts rather than resetting each time', () => {
    let state: LiveStreamState = { kind: 'checking' }
    state = nextLiveStreamState(state, { type: 'probe-fail' })
    state = nextLiveStreamState(state, { type: 'probe-fail' })
    state = nextLiveStreamState(state, { type: 'probe-fail' })
    expect(state).toEqual({ kind: 'unavailable', attempt: 3 })
  })

  it('recovers from unavailable to live the moment a publisher shows up', () => {
    const next = nextLiveStreamState({ kind: 'unavailable', attempt: 5 }, { type: 'probe-ok' })
    expect(next).toEqual({ kind: 'live' })
  })

  it('distinguishes a drop (was live) from never having been live at all', () => {
    const next = nextLiveStreamState({ kind: 'live' }, { type: 'probe-fail' })
    expect(next).toEqual({ kind: 'dropped', attempt: 1 })
    expect(next.kind).not.toBe('unavailable')
  })

  it('treats a fatal playback error while live the same as a manifest that disappeared', () => {
    const next = nextLiveStreamState({ kind: 'live' }, { type: 'play-error' })
    expect(next).toEqual({ kind: 'dropped', attempt: 1 })
  })

  it('counts up consecutive dropped attempts', () => {
    let state: LiveStreamState = { kind: 'live' }
    state = nextLiveStreamState(state, { type: 'probe-fail' })
    state = nextLiveStreamState(state, { type: 'probe-fail' })
    expect(state).toEqual({ kind: 'dropped', attempt: 2 })
  })

  it('recovers from dropped back to live', () => {
    const next = nextLiveStreamState({ kind: 'dropped', attempt: 4 }, { type: 'probe-ok' })
    expect(next).toEqual({ kind: 'live' })
  })

  it('ignores a stray play-error while not live rather than fabricating a drop', () => {
    const next = nextLiveStreamState({ kind: 'unavailable', attempt: 2 }, { type: 'play-error' })
    expect(next).toEqual({ kind: 'unavailable', attempt: 2 })
  })

  it('never leaves unsupported no matter what event arrives', () => {
    const events = [{ type: 'probe-ok' }, { type: 'probe-fail' }, { type: 'play-error' }] as const
    for (const event of events) {
      expect(nextLiveStreamState({ kind: 'unsupported' }, event)).toEqual({ kind: 'unsupported' })
    }
  })
})
