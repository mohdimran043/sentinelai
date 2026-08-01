import { describe, expect, it } from 'vitest'
import {
  cameraLiveness,
  STALE_AFTER_SECONDS,
  toneForLiveness,
  toneForModelState,
  toneForSeverity,
} from '@/lib/severity'

describe('toneForSeverity', () => {
  it('maps info/low to nominal, never to the escalation ramp', () => {
    expect(toneForSeverity('info')).toBe('nominal')
    expect(toneForSeverity('low')).toBe('nominal')
  })

  it('maps medium to caution', () => {
    expect(toneForSeverity('medium')).toBe('caution')
  })

  it('maps high/critical to breach', () => {
    expect(toneForSeverity('high')).toBe('breach')
    expect(toneForSeverity('critical')).toBe('breach')
  })

  it('falls back to inert for anything unrecognised, never to nominal green', () => {
    expect(toneForSeverity('bogus')).toBe('inert')
  })
})

describe('toneForModelState', () => {
  it('treats loaded/healthy as nominal', () => {
    expect(toneForModelState('loaded')).toBe('nominal')
    expect(toneForModelState('healthy')).toBe('nominal')
  })

  it('treats offline/unhealthy as breach', () => {
    expect(toneForModelState('offline')).toBe('breach')
    expect(toneForModelState('unhealthy')).toBe('breach')
  })

  it('treats transitional states as caution', () => {
    expect(toneForModelState('downloading')).toBe('caution')
    expect(toneForModelState('updating')).toBe('caution')
    expect(toneForModelState('sleeping')).toBe('caution')
  })

  it('treats unloaded as inert, not nominal', () => {
    expect(toneForModelState('unloaded')).toBe('inert')
  })
})

describe('cameraLiveness', () => {
  const now = 1_000_000_000 * 1000 // ms

  it('is no-data when last_frame_at is null — never invents a status', () => {
    expect(cameraLiveness(null, now)).toBe('no-data')
  })

  it('is live within the staleness window', () => {
    const lastFrameAt = now / 1000 - (STALE_AFTER_SECONDS - 1)
    expect(cameraLiveness(lastFrameAt, now)).toBe('live')
  })

  it('is stale once past the staleness window', () => {
    const lastFrameAt = now / 1000 - (STALE_AFTER_SECONDS + 1)
    expect(cameraLiveness(lastFrameAt, now)).toBe('stale')
  })
})

describe('toneForLiveness', () => {
  it('never renders stale/no-data as nominal green', () => {
    expect(toneForLiveness('live')).toBe('nominal')
    expect(toneForLiveness('stale')).toBe('caution')
    expect(toneForLiveness('no-data')).toBe('inert')
  })
})
