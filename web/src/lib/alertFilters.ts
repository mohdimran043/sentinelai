import type { AlertEntry } from '@/api/engineClient'
import type { EventSeverity } from '@/lib/severity'

/**
 * Narrowing the alert list, as pure functions over what the engine sent.
 *
 * All client-side: `GET /alerts` returns the whole register (bounded at
 * `SENTINEL_ALERT_REGISTER_CAPACITY`, 500) in one response, so there is nothing to
 * ask the engine for. Filtering here also means the facet counts below are computed
 * from the same snapshot the rows are, and cannot disagree with them.
 */

/** Every filter can be switched off, and `'all'` is what off looks like. */
export const ANY = 'all'

export type StateFilter = 'open' | typeof ANY
export type CameraFilter = string | typeof ANY
export type SeverityFilter = EventSeverity | typeof ANY

export interface AlertFilters {
  state: StateFilter
  camera: CameraFilter
  severity: SeverityFilter
}

export const NO_FILTERS: AlertFilters = { state: 'open', camera: ANY, severity: ANY }

/**
 * Severity in ramp order, worst first — the whole vocabulary, not only what is
 * present.
 *
 * Fixed rather than derived so the control never changes shape as alerts arrive, and
 * so `Critical — 0` is a thing an operator can *see*. A dropdown that silently omits
 * the band nobody wants to find answers "is anything critical?" only by absence,
 * which is the one question this page should answer out loud.
 *
 * `info` is omitted deliberately: `is_alertable` will not raise an alert below the
 * minimum severity, so an `info` option could only ever read zero.
 */
export const SEVERITY_ORDER: readonly EventSeverity[] = ['critical', 'high', 'medium', 'low']

export interface Facet<T> {
  value: T
  label: string
  count: number
}

function matchesState(alert: AlertEntry, state: StateFilter): boolean {
  return state === ANY || alert.state !== 'resolved'
}

function matchesCamera(alert: AlertEntry, camera: CameraFilter): boolean {
  return camera === ANY || alert.camera_id === camera
}

function matchesSeverity(alert: AlertEntry, severity: SeverityFilter): boolean {
  return severity === ANY || alert.severity === severity
}

export function applyAlertFilters(
  alerts: readonly AlertEntry[],
  filters: AlertFilters,
): AlertEntry[] {
  return alerts.filter(
    (alert) =>
      matchesState(alert, filters.state) &&
      matchesCamera(alert, filters.camera) &&
      matchesSeverity(alert, filters.severity),
  )
}

/**
 * Counts for one control, measured against every filter *except that control's own*.
 *
 * The rule that makes a count worth printing. Counting the fully-filtered set would
 * make every unselected option read zero the moment anything is picked, and counting
 * the unfiltered set would promise rows that the other filters will hide — "High —
 * 12" next to a camera showing two. Excluding only this control's own value is what
 * makes the number a true answer to "what would I get if I chose this instead".
 */
function withoutOwn(
  alerts: readonly AlertEntry[],
  filters: AlertFilters,
  own: keyof AlertFilters,
): AlertEntry[] {
  return applyAlertFilters(alerts, { ...filters, [own]: ANY })
}

/**
 * One option per camera that has an alert, ordered by id so the list is stable as
 * counts move.
 *
 * **Which cameras are listed and what each one counts are two different questions**,
 * and conflating them is a bug this had. Deriving the option set from the *filtered*
 * pool makes a camera disappear the moment another filter excludes it — pick
 * `critical` and every camera without a critical alert vanishes, including the one you
 * were about to switch to. So the options come from every camera present in the
 * register, and only the counts narrow. A camera reading `— 0` says "nothing here at
 * this severity", which is an answer; a camera that is not on the list says nothing at
 * all, and reads as though it had been removed from the site.
 *
 * That also makes this agree with `severityFacets`, which has always shown its empty
 * bands. Both controls now keep a fixed shape and move only their numbers.
 *
 * Still derived from the alerts rather than from `GET /cameras`: a camera with no
 * alert at all has nothing to offer either control, and listing every camera on the
 * site would bury the three that matter.
 */
export function cameraFacets(
  alerts: readonly AlertEntry[],
  filters: AlertFilters,
): Facet<string>[] {
  const pool = withoutOwn(alerts, filters, 'camera')
  const counts = new Map<string, Facet<string>>()
  // Option set first, from everything the engine sent, so it does not change shape.
  for (const alert of alerts) {
    if (!counts.has(alert.camera_id)) {
      counts.set(alert.camera_id, {
        value: alert.camera_id,
        label: alert.camera_label,
        count: 0,
      })
    }
  }
  for (const alert of pool) {
    const facet = counts.get(alert.camera_id)
    if (facet !== undefined) facet.count += 1
  }
  return [...counts.values()].sort((a, b) => a.value.localeCompare(b.value))
}

export function severityFacets(
  alerts: readonly AlertEntry[],
  filters: AlertFilters,
): Facet<EventSeverity>[] {
  const pool = withoutOwn(alerts, filters, 'severity')
  return SEVERITY_ORDER.map((severity) => ({
    value: severity,
    label: severity.charAt(0).toUpperCase() + severity.slice(1),
    count: pool.filter((alert) => alert.severity === severity).length,
  }))
}

/** Whether anything is narrowing the list beyond the default view. */
export function isNarrowed(filters: AlertFilters): boolean {
  return filters.camera !== ANY || filters.severity !== ANY
}
