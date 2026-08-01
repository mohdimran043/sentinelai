import { describe, expect, it } from 'vitest'
import {
  ALL,
  cameraIdsIn,
  deliveriesForAlert,
  eventTypesIn,
  paginate,
  selectAlerts,
} from '@/recorder/alertQuery'
import { makeAlert } from '@/recorder/mocks/fixtures'
import type { RecorderDelivery } from '@/recorder/recorder.types'

const alerts = [
  makeAlert({ cameraId: 'room_4b', eventType: 'UNDERLIT', offsetSeconds: 10 }),
  makeAlert({ cameraId: 'room_4b', eventType: 'OBSTRUCTED', offsetSeconds: 20 }),
  makeAlert({ cameraId: 'corridor_1', eventType: 'UNDERLIT', offsetSeconds: 30 }),
  makeAlert({ cameraId: 'corridor_1', eventType: 'LOST', offsetSeconds: 40 }),
]

describe('selectAlerts', () => {
  it('keeps everything when nothing is filtered', () => {
    const result = selectAlerts(alerts, { cameraId: ALL, eventType: ALL, order: 'newest' })
    expect(result).toHaveLength(4)
  })

  it('drops the other cameras when a camera is chosen', () => {
    const result = selectAlerts(alerts, {
      cameraId: 'corridor_1',
      eventType: ALL,
      order: 'newest',
    })

    expect(result).toHaveLength(2)
    expect(result.map((alert) => alert.camera_id)).toEqual(['corridor_1', 'corridor_1'])
    // The discriminating half: a no-op filter would still have room_4b in here.
    expect(result.some((alert) => alert.camera_id === 'room_4b')).toBe(false)
  })

  it('drops the other event types when a type is chosen', () => {
    const result = selectAlerts(alerts, { cameraId: ALL, eventType: 'UNDERLIT', order: 'newest' })

    expect(result.map((alert) => alert.event_type)).toEqual(['UNDERLIT', 'UNDERLIT'])
    expect(result.some((alert) => alert.event_type === 'OBSTRUCTED')).toBe(false)
    expect(result.some((alert) => alert.event_type === 'LOST')).toBe(false)
  })

  it('intersects the two filters rather than applying only the last one', () => {
    const result = selectAlerts(alerts, {
      cameraId: 'room_4b',
      eventType: 'UNDERLIT',
      order: 'newest',
    })

    expect(result).toHaveLength(1)
    expect(result[0]?.camera_id).toBe('room_4b')
    expect(result[0]?.event_type).toBe('UNDERLIT')
  })

  it('returns nothing — not everything — when the combination has no match', () => {
    const result = selectAlerts(alerts, {
      cameraId: 'room_4b',
      eventType: 'LOST',
      order: 'newest',
    })
    expect(result).toEqual([])
  })

  it('orders by at_ns in both directions, not by arrival order', () => {
    // Deliberately shuffled so a function that just returns the input, or
    // reverses it, cannot pass both assertions.
    const shuffled = [alerts[2]!, alerts[0]!, alerts[3]!, alerts[1]!]

    const newest = selectAlerts(shuffled, { cameraId: ALL, eventType: ALL, order: 'newest' })
    expect(newest.map((alert) => alert.at_ns)).toEqual(
      [...alerts].map((alert) => alert.at_ns).sort((a, b) => b - a),
    )

    const oldest = selectAlerts(shuffled, { cameraId: ALL, eventType: ALL, order: 'oldest' })
    expect(oldest.map((alert) => alert.at_ns)).toEqual(
      [...alerts].map((alert) => alert.at_ns).sort((a, b) => a - b),
    )
    expect(oldest[0]?.at_ns).toBeLessThan(newest[0]!.at_ns)
  })

  it('does not mutate the caller’s array', () => {
    const input = [...alerts]
    selectAlerts(input, { cameraId: ALL, eventType: ALL, order: 'oldest' })
    expect(input).toEqual(alerts)
  })
})

describe('option lists', () => {
  it('offers each camera and event type once, sorted', () => {
    expect(cameraIdsIn(alerts)).toEqual(['corridor_1', 'room_4b'])
    expect(eventTypesIn(alerts)).toEqual(['LOST', 'OBSTRUCTED', 'UNDERLIT'])
  })
})

describe('paginate', () => {
  const items = Array.from({ length: 120 }, (_, index) => index)

  it('slices a page and reports its position in one-based terms', () => {
    const first = paginate(items, 0, 50)
    expect(first.items).toHaveLength(50)
    expect(first.items[0]).toBe(0)
    expect(first.firstItemNumber).toBe(1)
    expect(first.lastItemNumber).toBe(50)
    expect(first.pageCount).toBe(3)

    const last = paginate(items, 2, 50)
    expect(last.items).toEqual([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
    expect(last.firstItemNumber).toBe(101)
    expect(last.lastItemNumber).toBe(120)
  })

  it('clamps a page index past the end instead of showing an empty table', () => {
    const page = paginate(items, 99, 50)
    expect(page.pageIndex).toBe(2)
    expect(page.items).toHaveLength(20)
  })

  it('reports an empty set honestly rather than as page 1 of 0', () => {
    const page = paginate([], 0, 50)
    expect(page.total).toBe(0)
    expect(page.pageCount).toBe(1)
    expect(page.firstItemNumber).toBe(0)
    expect(page.lastItemNumber).toBe(0)
  })
})

describe('deliveriesForAlert', () => {
  const deliveries: RecorderDelivery[] = [
    { alert_id: 'b', camera_id: 'c', event_type: 'LOST', channel: 'webhook', status: 'failed', attempts: 2, at_ns: 30 },
    { alert_id: 'a', camera_id: 'c', event_type: 'LOST', channel: 'dashboard', status: 'delivered', attempts: 1, at_ns: 20 },
    { alert_id: 'a', camera_id: 'c', event_type: 'LOST', channel: 'webhook', status: 'failed', attempts: 1, at_ns: 10 },
  ]

  it('returns only that alert’s attempts, oldest first', () => {
    const result = deliveriesForAlert(deliveries, 'a')
    expect(result.map((delivery) => delivery.at_ns)).toEqual([10, 20])
    expect(result.every((delivery) => delivery.alert_id === 'a')).toBe(true)
  })

  it('returns nothing for an alert with no recorded attempt', () => {
    expect(deliveriesForAlert(deliveries, 'zzz')).toEqual([])
  })
})
