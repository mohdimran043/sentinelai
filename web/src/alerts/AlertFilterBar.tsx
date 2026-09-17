import { Select } from '@/components/ui/Select'
import type { AlertEntry } from '@/api/engineClient'
import type { EventSeverity } from '@/lib/severity'
import {
  ANY,
  type AlertFilters,
  cameraFacets,
  isNarrowed,
  severityFacets,
} from '@/lib/alertFilters'

export interface AlertFilterBarProps {
  /** Every alert the engine sent, before any filtering — the facets measure against this. */
  alerts: readonly AlertEntry[]
  filters: AlertFilters
  onChange: (filters: AlertFilters) => void
  /** How many rows survive the current filters, for the summary line. */
  shown: number
}

/**
 * Narrow the alert list to one camera, one severity band, or both.
 *
 * **Every option carries its count**, which is the thing that makes this usable
 * during an incident. An operator scanning a site does not want to click through four
 * cameras to find where the trouble is — "Abbey Road — 20" against "Bourbon Street —
 * 3" answers that from the closed dropdown. The counts are faceted (see
 * `alertFilters.ts`), so each one is a true answer to "what would I get if I chose
 * this instead" rather than a number that lies as soon as another filter is set.
 *
 * **Severity, not priority.** They are different questions and this console shows
 * priority on the row's badge — severity is how alarming the scene looked to the
 * model, priority is how costly it would be to get this one wrong, and they genuinely
 * diverge. Because the filter's criterion is not the one on the badge, `AlertRow`
 * prints the severity in its readout line; a filter whose effect you cannot see on
 * the rows it produced is one you cannot trust.
 */
export function AlertFilterBar({ alerts, filters, onChange, shown }: AlertFilterBarProps) {
  const cameras = cameraFacets(alerts, filters)
  const severities = severityFacets(alerts, filters)
  const narrowed = isNarrowed(filters)

  return (
    <div className="mb-3 flex flex-wrap items-end gap-x-3 gap-y-2">
      <fieldset className="m-0 border-0 p-0">
        <legend className="readout mb-1 block text-[10px] uppercase tracking-[0.08em] text-dimmer">
          Showing
        </legend>
        <div className="flex items-center gap-2">
          {(['open', ANY] as const).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => onChange({ ...filters, state: option })}
              aria-pressed={filters.state === option}
              className={
                filters.state === option
                  ? 'rounded-sm border border-signal-d bg-[#0f1610] px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-signal'
                  : 'rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:text-fg'
              }
            >
              {option === 'open' ? 'Needs attention' : 'Everything'}
            </button>
          ))}
        </div>
      </fieldset>

      <FilterField label="Camera" htmlFor="alert-filter-camera">
        <Select
          id="alert-filter-camera"
          value={filters.camera}
          onChange={(event) => onChange({ ...filters, camera: event.target.value })}
          className="w-auto min-w-[190px] py-[5px] text-[12.5px]"
        >
          <option value={ANY}>All cameras</option>
          {cameras.map((facet) => (
            <option key={facet.value} value={facet.value}>
              {facet.label} — {facet.count}
            </option>
          ))}
        </Select>
      </FilterField>

      <FilterField label="Severity" htmlFor="alert-filter-severity">
        <Select
          id="alert-filter-severity"
          value={filters.severity}
          onChange={(event) =>
            onChange({ ...filters, severity: event.target.value as EventSeverity | typeof ANY })
          }
          className="w-auto min-w-[150px] py-[5px] text-[12.5px]"
        >
          <option value={ANY}>All severities</option>
          {severities.map((facet) => (
            <option key={facet.value} value={facet.value}>
              {facet.label} — {facet.count}
            </option>
          ))}
        </Select>
      </FilterField>

      {narrowed ? (
        <button
          type="button"
          onClick={() => onChange({ ...filters, camera: ANY, severity: ANY })}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:border-signal-d hover:text-signal"
        >
          Clear filters
        </button>
      ) : null}

      {/* Stated out loud, because a filtered list and a quiet site look identical.
          `aria-live` so a screen reader hears the count change rather than having to
          go looking for it after every choice. */}
      <p
        className="readout m-0 ml-auto text-[11.5px] text-dimmer"
        aria-live="polite"
        data-testid="alert-filter-summary"
      >
        {narrowed ? `${shown} of ${alerts.length} alerts` : `${shown} alerts`}
      </p>
    </div>
  )
}

function FilterField({
  label,
  htmlFor,
  children,
}: {
  label: string
  htmlFor: string
  children: React.ReactNode
}) {
  return (
    <div>
      <label
        htmlFor={htmlFor}
        className="readout mb-1 block text-[10px] uppercase tracking-[0.08em] text-dimmer"
      >
        {label}
      </label>
      {children}
    </div>
  )
}
