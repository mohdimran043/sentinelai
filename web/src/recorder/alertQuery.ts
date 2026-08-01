import type { RecorderAlert, RecorderDelivery } from '@/recorder/recorder.types'

/**
 * Filtering, ordering and paging for the alert feed.
 *
 * This lives client-side because it has to. VERIFIED 2026-08-01 against the
 * live appliance: `GET /api/alerts` accepts no query parameters at all —
 * `?limit`, `?offset`, `?camera_id`, `?event_type` and `?since_ns` each
 * returned the identical full 500-row body — and there is no
 * `GET /api/alerts/{id}`. One request brings the whole rolling window (~426 KB)
 * and the console narrows it.
 *
 * Kept as pure functions, separate from the page, so the behaviour can be
 * tested without a DOM and so the page cannot quietly diverge from it.
 */

export const ALL = 'all'

export type AlertOrder = 'newest' | 'oldest'

export interface AlertQuery {
  /** A camera id, or `ALL`. */
  cameraId: string
  /** An event type, or `ALL`. */
  eventType: string
  order: AlertOrder
}

export const DEFAULT_ALERT_QUERY: AlertQuery = {
  cameraId: ALL,
  eventType: ALL,
  order: 'newest',
}

/** Camera ids present in this window, sorted. Not the camera registry — only what actually raised something. */
export function cameraIdsIn(alerts: readonly RecorderAlert[]): string[] {
  return [...new Set(alerts.map((alert) => alert.camera_id))].sort()
}

/** Event types present in this window, sorted. */
export function eventTypesIn(alerts: readonly RecorderAlert[]): string[] {
  return [...new Set(alerts.map((alert) => alert.event_type))].sort()
}

/**
 * Apply the query. Ordering is always by `at_ns` — the recorder returns newest
 * first, but this never assumes that, because "the list happened to arrive
 * sorted" is not a guarantee anyone made.
 */
export function selectAlerts(
  alerts: readonly RecorderAlert[],
  query: AlertQuery,
): RecorderAlert[] {
  const matched = alerts.filter(
    (alert) =>
      (query.cameraId === ALL || alert.camera_id === query.cameraId) &&
      (query.eventType === ALL || alert.event_type === query.eventType),
  )
  const direction = query.order === 'newest' ? -1 : 1
  return matched.sort((a, b) => direction * (a.at_ns - b.at_ns))
}

export interface Page<T> {
  items: T[]
  /** Zero-based, clamped into range. */
  pageIndex: number
  pageCount: number
  /** One-based index of the first item shown, or 0 when there are none. */
  firstItemNumber: number
  /** One-based index of the last item shown, or 0 when there are none. */
  lastItemNumber: number
  total: number
}

/**
 * Slice one page out of the matched set.
 *
 * The alert window is 500 rows and grows under the operator's feet, so the page
 * index is clamped rather than trusted: a filter change or a shrinking feed must
 * not leave the table showing nothing with no way back.
 */
export function paginate<T>(items: readonly T[], pageIndex: number, pageSize: number): Page<T> {
  const total = items.length
  const pageCount = Math.max(1, Math.ceil(total / pageSize))
  const clamped = Math.min(Math.max(pageIndex, 0), pageCount - 1)
  const start = clamped * pageSize
  const slice = items.slice(start, start + pageSize)
  return {
    items: slice,
    pageIndex: clamped,
    pageCount,
    firstItemNumber: total === 0 ? 0 : start + 1,
    lastItemNumber: total === 0 ? 0 : start + slice.length,
    total,
  }
}

/** Delivery attempts recorded for one alert, oldest attempt first. */
export function deliveriesForAlert(
  deliveries: readonly RecorderDelivery[],
  alertId: string,
): RecorderDelivery[] {
  return deliveries
    .filter((delivery) => delivery.alert_id === alertId)
    .sort((a, b) => a.at_ns - b.at_ns)
}
