import { describe, expect, it } from 'vitest'
import { buildThreatRibbon } from '@/lib/ribbon'
import type { RecentEventEntry } from '@/api/engineClient'

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: '11111111-1111-4111-8111-111111111111',
    camera_id: 'avenue_01',
    sequence: 1,
    clip_uri: null,
    occurred_at: 0,
    source_timestamp: null,
    reason: 'new_salient_track',
    threat_score: 0.1,
    severity: 'info',
    description: 'test',
    suggested_action: 'none',
    description_unavailable: false,
    labels: [],
    track_ids: [],
    welfare_concerns: [],
    ...overrides,
  }
}

describe('buildThreatRibbon', () => {
  it('produces no cells for an empty ring rather than fabricating placeholder buckets', () => {
    expect(buildThreatRibbon([])).toEqual([])
  })

  it('produces exactly one cell per event, not a fixed bucket count', () => {
    const events = [
      makeEvent({ event_id: 'a', severity: 'info' }),
      makeEvent({ event_id: 'b', severity: 'medium' }),
      makeEvent({ event_id: 'c', severity: 'critical' }),
    ]
    expect(buildThreatRibbon(events)).toHaveLength(3)
  })

  it('preserves the ring order (oldest first) rather than sorting or reversing', () => {
    const events = [
      makeEvent({ event_id: 'a', occurred_at: 10, severity: 'info' }),
      makeEvent({ event_id: 'b', occurred_at: 20, severity: 'critical' }),
      makeEvent({ event_id: 'c', occurred_at: 30, severity: 'medium' }),
    ]
    const cells = buildThreatRibbon(events)
    expect(cells.map((cell) => cell.tone)).toEqual(['nominal', 'breach', 'caution'])
  })

  it('colours info/low nominal (green), medium caution (amber), and high/critical breach (red)', () => {
    const severities: [RecentEventEntry['severity'], string][] = [
      ['info', 'nominal'],
      ['low', 'nominal'],
      ['medium', 'caution'],
      ['high', 'breach'],
      ['critical', 'breach'],
    ]
    for (const [severity, expectedTone] of severities) {
      const cells = buildThreatRibbon([makeEvent({ severity })])
      expect(cells[0]!.tone).toBe(expectedTone)
    }
  })

  it('flags a description-unavailable event in its own tooltip description', () => {
    const cells = buildThreatRibbon([makeEvent({ description_unavailable: true })])
    expect(cells[0]!.description).toMatch(/description unavailable/i)
  })

  it('does not flag an ordinary event as description-unavailable', () => {
    const cells = buildThreatRibbon([makeEvent({ description_unavailable: false })])
    expect(cells[0]!.description).not.toMatch(/description unavailable/i)
  })
})
