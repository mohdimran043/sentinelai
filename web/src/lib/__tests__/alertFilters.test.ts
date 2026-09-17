import { describe, expect, it } from 'vitest'
import type { AlertEntry } from '@/api/engineClient'
import type { EventSeverity } from '@/lib/severity'
import {
  ANY,
  type AlertFilters,
  NO_FILTERS,
  SEVERITY_ORDER,
  applyAlertFilters,
  cameraFacets,
  isNarrowed,
  severityFacets,
} from '@/lib/alertFilters'

let nextId = 0

function alert(overrides: Partial<AlertEntry> = {}): AlertEntry {
  nextId += 1
  return {
    alert_id: `00000000-0000-0000-0000-${String(nextId).padStart(12, '0')}`,
    camera_id: 'linkou',
    camera_label: 'Linkou Old Street',
    zone: null,
    reason: 'zone_intrusion',
    subject: '1',
    state: 'active',
    severity: 'medium',
    priority: 'medium',
    first_seen: 1_700_000_000,
    last_seen: 1_700_000_000,
    occurrences: 1,
    description: 'something happened',
    event_ids: [],
    clip_uri: null,
    notify_clip_uri: null,
    acknowledged_by: null,
    acknowledged_at: null,
    ...overrides,
  }
}

function withFilters(overrides: Partial<AlertFilters> = {}): AlertFilters {
  return { ...NO_FILTERS, ...overrides }
}

const ROSTER: AlertEntry[] = [
  alert({ camera_id: 'linkou', camera_label: 'Linkou', severity: 'critical' }),
  alert({ camera_id: 'linkou', camera_label: 'Linkou', severity: 'high' }),
  alert({ camera_id: 'linkou', camera_label: 'Linkou', severity: 'medium' }),
  alert({ camera_id: 'abbeyroad', camera_label: 'Abbey Road', severity: 'high' }),
  alert({ camera_id: 'abbeyroad', camera_label: 'Abbey Road', severity: 'medium' }),
  alert({ camera_id: 'bourbonstreet', camera_label: 'Bourbon Street', severity: 'low' }),
]

describe('applyAlertFilters', () => {
  it('shows everything open by default, hiding nothing an operator has not hidden', () => {
    expect(applyAlertFilters(ROSTER, NO_FILTERS)).toHaveLength(6)
  })

  it('narrows to one camera', () => {
    const shown = applyAlertFilters(ROSTER, withFilters({ camera: 'linkou' }))

    expect(shown).toHaveLength(3)
    expect(shown.every((a) => a.camera_id === 'linkou')).toBe(true)
  })

  it('narrows to one severity band', () => {
    const shown = applyAlertFilters(ROSTER, withFilters({ severity: 'high' }))

    expect(shown).toHaveLength(2)
    expect(shown.every((a) => a.severity === 'high')).toBe(true)
  })

  it('combines camera and severity rather than picking one', () => {
    const shown = applyAlertFilters(ROSTER, withFilters({ camera: 'linkou', severity: 'high' }))

    expect(shown).toHaveLength(1)
    expect(shown[0]?.camera_id).toBe('linkou')
    expect(shown[0]?.severity).toBe('high')
  })

  it('filters on severity, never on priority', () => {
    // The two are different questions and this console badges the row with priority.
    // A filter that quietly read the badge would hide alerts an operator asked for.
    const alerts = [
      alert({ severity: 'critical', priority: 'low' }),
      alert({ severity: 'low', priority: 'critical' }),
    ]
    const shown = applyAlertFilters(alerts, withFilters({ severity: 'critical' }))

    expect(shown).toHaveLength(1)
    expect(shown[0]?.priority).toBe('low')
  })

  it('keeps resolved alerts out of the default view and back in on request', () => {
    const alerts = [alert({ state: 'active' }), alert({ state: 'resolved' })]

    expect(applyAlertFilters(alerts, withFilters({ state: 'open' }))).toHaveLength(1)
    expect(applyAlertFilters(alerts, withFilters({ state: ANY }))).toHaveLength(2)
  })

  it('applies the state filter alongside the others', () => {
    const alerts = [
      alert({ camera_id: 'linkou', state: 'resolved' }),
      alert({ camera_id: 'linkou', state: 'active' }),
    ]

    expect(applyAlertFilters(alerts, withFilters({ camera: 'linkou' }))).toHaveLength(1)
  })
})

describe('cameraFacets', () => {
  it('offers one option per camera that has an alert, with its count', () => {
    const facets = cameraFacets(ROSTER, NO_FILTERS)

    expect(facets.map((f) => [f.value, f.count])).toEqual([
      ['abbeyroad', 2],
      ['bourbonstreet', 1],
      ['linkou', 3],
    ])
  })

  it('does not offer a camera with nothing to show', () => {
    // Derived from the alerts, not from the camera roster: an option that can only
    // ever empty the list is one an operator clicks once and never trusts again.
    expect(cameraFacets(ROSTER, NO_FILTERS).map((f) => f.value)).not.toContain('front_door')
  })

  it('counts against the other filters, not against its own', () => {
    // `severity: high` leaves Linkou 1 and Abbey Road 1, and Bourbon Street none. If
    // the counts ignored severity they would read 3/2/1 and promise rows the severity
    // filter will hide.
    const facets = cameraFacets(ROSTER, withFilters({ severity: 'high' }))

    expect(facets.map((f) => [f.value, f.count])).toEqual([
      ['abbeyroad', 1],
      ['bourbonstreet', 0],
      ['linkou', 1],
    ])
  })

  it('keeps a camera listed at zero rather than removing it', () => {
    // The bug this replaced: the option set was derived from the filtered pool, so
    // choosing a severity deleted every camera without one — including whichever the
    // operator was about to switch to, leaving no way to reach it.
    const facets = cameraFacets(ROSTER, withFilters({ severity: 'critical' }))

    expect(facets.map((f) => f.value)).toEqual(['abbeyroad', 'bourbonstreet', 'linkou'])
    expect(facets.find((f) => f.value === 'bourbonstreet')?.count).toBe(0)
  })

  it('keeps the same options whatever is filtered, so the control never changes shape', () => {
    const shapes = [
      NO_FILTERS,
      withFilters({ severity: 'critical' }),
      withFilters({ camera: 'linkou' }),
      withFilters({ camera: 'linkou', severity: 'low' }),
      withFilters({ state: ANY }),
    ].map((filters) => cameraFacets(ROSTER, filters).map((f) => f.value))

    for (const shape of shapes) {
      expect(shape).toEqual(['abbeyroad', 'bourbonstreet', 'linkou'])
    }
  })

  it('does not zero every other camera once one is picked', () => {
    // The failure mode of counting the fully-filtered set: every unselected option
    // reads 0 and the control becomes useless for moving between cameras.
    const facets = cameraFacets(ROSTER, withFilters({ camera: 'linkou' }))

    expect(facets.find((f) => f.value === 'abbeyroad')?.count).toBe(2)
    expect(facets.find((f) => f.value === 'linkou')?.count).toBe(3)
  })

  it('keeps the selected camera listed even when another filter empties it', () => {
    // Otherwise the option vanishes while still selected, and the only control that
    // could undo the choice is gone with it.
    const facets = cameraFacets(ROSTER, withFilters({ camera: 'bourbonstreet', severity: 'high' }))

    expect(facets.find((f) => f.value === 'bourbonstreet')).toEqual({
      value: 'bourbonstreet',
      label: 'Bourbon Street',
      count: 0,
    })
  })

  it('offers nothing at all when the register is empty', () => {
    expect(cameraFacets([], NO_FILTERS)).toEqual([])
  })

  it('orders by camera id so options do not jump around as counts move', () => {
    const facets = cameraFacets(ROSTER, NO_FILTERS)

    expect(facets.map((f) => f.value)).toEqual([...facets.map((f) => f.value)].sort())
  })
})

describe('severityFacets', () => {
  it('offers the whole vocabulary, worst first, however few are present', () => {
    // Fixed shape on purpose: an operator must be able to see "Critical — 0" rather
    // than infer it from an option that is not there.
    const facets = severityFacets([alert({ severity: 'low' })], NO_FILTERS)

    expect(facets.map((f) => f.value)).toEqual([...SEVERITY_ORDER])
    expect(facets.find((f) => f.value === 'critical')?.count).toBe(0)
  })

  it('counts each band', () => {
    const facets = severityFacets(ROSTER, NO_FILTERS)

    expect(facets.map((f) => [f.value, f.count])).toEqual([
      ['critical', 1],
      ['high', 2],
      ['medium', 2],
      ['low', 1],
    ])
  })

  it('counts against the camera filter, not against its own', () => {
    const facets = severityFacets(ROSTER, withFilters({ camera: 'linkou' }))

    expect(facets.map((f) => [f.value, f.count])).toEqual([
      ['critical', 1],
      ['high', 1],
      ['medium', 1],
      ['low', 0],
    ])
  })

  it('labels each band for reading rather than for the wire', () => {
    expect(severityFacets(ROSTER, NO_FILTERS).map((f) => f.label)).toEqual([
      'Critical',
      'High',
      'Medium',
      'Low',
    ])
  })

  it('does not offer info, which an alert can never carry', () => {
    expect(SEVERITY_ORDER).not.toContain('info' as EventSeverity)
  })
})

describe('isNarrowed', () => {
  it('is false for the default view, so the page does not claim to be hiding things', () => {
    expect(isNarrowed(NO_FILTERS)).toBe(false)
  })

  it('ignores the state filter, which is a view rather than a narrowing', () => {
    // "Everything" widens the list. Reporting it as a filter would put "6 of 6" and a
    // "Clear filters" button on the fullest view the page has.
    expect(isNarrowed(withFilters({ state: ANY }))).toBe(false)
  })

  it('is true once a camera or a severity is picked', () => {
    expect(isNarrowed(withFilters({ camera: 'linkou' }))).toBe(true)
    expect(isNarrowed(withFilters({ severity: 'high' }))).toBe(true)
  })
})
