import { describe, expect, it } from 'vitest'
import { buildThreatRibbon } from '@/lib/ribbon'
import type { SentinelAIAnomalyEvent } from '@/events/anomalyEvent.types'

function makeEvent(overrides: Partial<SentinelAIAnomalyEvent>): SentinelAIAnomalyEvent {
  return {
    schema_version: 1,
    event_id: '11111111-1111-4111-8111-111111111111',
    camera_id: 'avenue_01',
    occurred_at: 0,
    reason: 'new_salient_track',
    threat_score: 0.1,
    severity: 'info',
    description: 'test',
    suggested_action: 'none',
    labels: [],
    track_ids: [],
    keyframe_uri: null,
    clip_uri: null,
    description_unavailable: false,
    metadata: {},
    ...overrides,
  }
}

describe('buildThreatRibbon', () => {
  it('produces bucketCount cells even with no events, all nominal (observed, nothing happened)', () => {
    const cells = buildThreatRibbon([], { bucketCount: 10, spanSeconds: 100, now: 100_000 })
    expect(cells).toHaveLength(10)
    expect(cells.every((cell) => cell.tone === 'nominal')).toBe(true)
  })

  it('escalates the bucket tone to the worst severity observed in it', () => {
    const now = 100_000
    const nowSeconds = now / 1000
    const events = [
      makeEvent({ occurred_at: nowSeconds - 5, severity: 'critical', event_id: 'a' }),
    ]
    const cells = buildThreatRibbon(events, { bucketCount: 10, spanSeconds: 100, now })
    const breachCells = cells.filter((cell) => cell.tone === 'breach')
    expect(breachCells).toHaveLength(1)
  })

  it('ignores events outside the requested span', () => {
    const now = 100_000
    const nowSeconds = now / 1000
    const events = [makeEvent({ occurred_at: nowSeconds - 10_000, severity: 'critical' })]
    const cells = buildThreatRibbon(events, { bucketCount: 10, spanSeconds: 100, now })
    expect(cells.every((cell) => cell.tone === 'nominal')).toBe(true)
  })
})
