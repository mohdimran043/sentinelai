import { describe, expect, it } from 'vitest'
import { latestEventByCamera, markerTone } from '@/sitemap/markers'
import type { RecentEventEntry } from '@/api/engineClient'

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: 'e1',
    camera_id: 'room_4b',
    sequence: 1,
    clip_uri: null,
    occurred_at: 100,
    source_timestamp: null,
    reason: 'new_salient_track',
    threat_score: 0.2,
    severity: 'low',
    description: 'test',
    suggested_action: 'none',
    description_unavailable: false,
    labels: [],
    track_ids: [],
    welfare_concerns: [],
    ...overrides,
  }
}

describe('latestEventByCamera', () => {
  it('picks the most recent event per camera by occurred_at', () => {
    const older = makeEvent({ event_id: 'a', camera_id: 'room_4b', occurred_at: 100 })
    const newer = makeEvent({ event_id: 'b', camera_id: 'room_4b', occurred_at: 200 })
    const latest = latestEventByCamera([older, newer])
    expect(latest.get('room_4b')?.event_id).toBe('b')
  })

  it('keeps events from different cameras separate', () => {
    const a = makeEvent({ event_id: 'a', camera_id: 'room_4b' })
    const b = makeEvent({ event_id: 'b', camera_id: 'corridor_1' })
    const latest = latestEventByCamera([a, b])
    expect(latest.get('room_4b')?.event_id).toBe('a')
    expect(latest.get('corridor_1')?.event_id).toBe('b')
  })

  it('breaks a tie on occurred_at by the higher sequence', () => {
    const first = makeEvent({ event_id: 'a', occurred_at: 100, sequence: 1 })
    const second = makeEvent({ event_id: 'a', occurred_at: 100, sequence: 2 })
    const latest = latestEventByCamera([first, second])
    expect(latest.get('room_4b')?.sequence).toBe(2)
  })

  it('returns an empty map for no events', () => {
    expect(latestEventByCamera([]).size).toBe(0)
  })
})

describe('markerTone', () => {
  it('is inert (never nominal/green) for a camera with no event yet, so a quiet map is not a wall of green', () => {
    expect(markerTone(undefined)).toBe('inert')
  })

  it('is nominal for an info or low severity event', () => {
    expect(markerTone(makeEvent({ severity: 'info' }))).toBe('nominal')
    expect(markerTone(makeEvent({ severity: 'low' }))).toBe('nominal')
  })

  it('is caution for a medium severity event', () => {
    expect(markerTone(makeEvent({ severity: 'medium' }))).toBe('caution')
  })

  it('is breach for a high or critical severity event', () => {
    expect(markerTone(makeEvent({ severity: 'high' }))).toBe('breach')
    expect(markerTone(makeEvent({ severity: 'critical' }))).toBe('breach')
  })
})
