import { describe, expect, it } from 'vitest'
import { panelStateFor } from '@/lib/descriptionPanel'
import type { CameraEventsResponse, RecentEventEntry } from '@/api/engineClient'

function makeEvent(overrides: Partial<RecentEventEntry> = {}): RecentEventEntry {
  return {
    event_id: '11111111-1111-4111-8111-111111111111',
    camera_id: 'avenue_01',
    sequence: 1,
    clip_uri: null,
    occurred_at: 1_700_000_000,
    source_timestamp: 12.5,
    reason: 'new_salient_track',
    threat_score: 0.2,
    severity: 'low',
    description: 'A person walks across the frame.',
    suggested_action: 'none',
    description_unavailable: false,
    labels: ['person'],
    track_ids: [1],
    welfare_concerns: [],
    ...overrides,
  }
}

function makeResponse(overrides: Partial<CameraEventsResponse> = {}): CameraEventsResponse {
  return {
    camera_id: 'avenue_01',
    capacity: 200,
    returned: 0,
    volatile: true,
    latest_description_state: 'none',
    latest: null,
    events: [],
    ...overrides,
  }
}

describe('panelStateFor', () => {
  it('reports none when nothing has been assembled yet', () => {
    const response = makeResponse({ latest_description_state: 'none', latest: null })
    expect(panelStateFor(response)).toEqual({ kind: 'none' })
  })

  it('reports available with the latest event when the VLM answered', () => {
    const event = makeEvent({ description: 'Two people talking near the entrance.' })
    const response = makeResponse({
      latest_description_state: 'available',
      latest: event,
      returned: 1,
      events: [event],
    })
    expect(panelStateFor(response)).toEqual({ kind: 'available', event })
  })

  it('reports unavailable with the metadata-derived stand-in event when the VLM could not answer', () => {
    const event = makeEvent({
      description_unavailable: true,
      description: 'Escalation triggered (new_salient_track); the vision model could not answer.',
    })
    const response = makeResponse({
      latest_description_state: 'unavailable',
      latest: event,
      returned: 1,
      events: [event],
    })
    expect(panelStateFor(response)).toEqual({ kind: 'unavailable', event })
  })

  it('degrades to none rather than crashing if latest is null despite a non-none state', () => {
    // Defends against a contract violation the schema promises cannot happen
    // (`latest` is null exactly when the state is 'none') without pretending
    // to know what actually went wrong.
    const response = makeResponse({ latest_description_state: 'available', latest: null })
    expect(panelStateFor(response)).toEqual({ kind: 'none' })
  })

  it('does not conflate available and unavailable even when both carry a real-looking description', () => {
    const available = makeEvent({ description_unavailable: false })
    const unavailable = makeEvent({ description_unavailable: true })
    const availableState = panelStateFor(
      makeResponse({ latest_description_state: 'available', latest: available }),
    )
    const unavailableState = panelStateFor(
      makeResponse({ latest_description_state: 'unavailable', latest: unavailable }),
    )
    expect(availableState.kind).toBe('available')
    expect(unavailableState.kind).toBe('unavailable')
  })
})
