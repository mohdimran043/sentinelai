import { useState } from 'react'
import {
  useAcknowledgeAlert,
  useAlerts,
  useClearAlerts,
  useResolveAlert,
} from '@/api/queries'
import type { AlertEntry } from '@/api/engineClient'
import { AlertRow } from '@/alerts/AlertRow'
import { AlertFilterBar } from '@/alerts/AlertFilterBar'
import {
  ANY,
  type AlertFilters,
  NO_FILTERS,
  applyAlertFilters,
  isNarrowed,
} from '@/lib/alertFilters'
import { Absent } from '@/components/ui/Absent'
import { Notice } from '@/components/ui/Notice'
import { useSessionStore } from '@/store/session'

/**
 * The alert log (spec §17, §21).
 *
 * Two things this page is careful about:
 *
 * **Open first, everything on request.** The default is what still needs
 * attention, because that is what an operator opens this page to find. Resolved
 * alerts are one click away rather than gone, because "what happened earlier"
 * is the other question this page gets asked.
 *
 * **Filters narrow, and say that they are narrowing.** Camera and severity both
 * start at "all", so the page never opens already hiding something. Once anything
 * is set, the count says "7 of 55" and the empty state changes its mind about what
 * an empty list means — a filtered page and a quiet site look identical, and only
 * one of them is good news.
 *
 * **It says what it cannot promise.** The register is in engine memory, so an
 * acknowledgement does not survive a restart. That is stated on the page rather
 * than in documentation, because the person who needs to know it is the one
 * looking at the list.
 */
export function AlertsPage() {
  const query = useAlerts()
  const acknowledge = useAcknowledgeAlert()
  const resolve = useResolveAlert()
  const operatorName = useSessionStore((state) => state.operatorName)
  const [filters, setFilters] = useState<AlertFilters>(NO_FILTERS)

  const alerts: AlertEntry[] = query.data?.alerts ?? []
  const shown = applyAlertFilters(alerts, filters)
  const narrowed = isNarrowed(filters)
  const failure = acknowledge.error ?? resolve.error

  return (
    <div>
      <h1>Alerts</h1>
      <p className="lede">
        One row per situation, not per event — a person seen seventeen times in
        twenty seconds is one alert that says so. Acknowledging tells the rest of
        the room somebody is on it.
      </p>

      <Notice tone="caution" className="mb-4">
        This list lives in the engine's memory. A restart empties it, and{' '}
        <strong className="font-semibold text-fg">acknowledgements do not survive one</strong>.
        The durable record is the published event stream.
      </Notice>

      {failure ? (
        <Notice tone="breach" className="mb-4">
          {failure.message}
        </Notice>
      ) : null}

      {/* Only once there is something to narrow. Facet counts over an empty list are
          a row of zeroes that answers nothing. */}
      {alerts.length > 0 ? (
        <AlertFilterBar
          alerts={alerts}
          filters={filters}
          onChange={setFilters}
          shown={shown.length}
        />
      ) : null}

      {/* Only with something to clear: a destructive control offered against an
          empty list is a button that can do nothing but frighten. */}
      {alerts.length > 0 ? <ClearAll /> : null}

      {query.isPending ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          {[0, 1, 2].map((key) => (
            <div
              key={key}
              className="h-[110px] animate-pulse rounded-md border border-line bg-panel motion-reduce:animate-none"
            />
          ))}
        </div>
      ) : query.isError ? (
        <Absent title="Alerts unavailable">
          {query.error.message} This console cannot currently tell you whether
          anything is waiting.
        </Absent>
      ) : shown.length === 0 ? (
        <EmptyList
          narrowed={narrowed}
          total={alerts.length}
          onClearFilters={() => setFilters({ ...filters, camera: ANY, severity: ANY })}
        />
      ) : (
        <div className="flex flex-col gap-2">
          {shown.map((alert) => (
            <AlertRow
              key={alert.alert_id}
              alert={alert}
              busy={acknowledge.isPending || resolve.isPending}
              onAcknowledge={() =>
                acknowledge.mutate({
                  alertId: alert.alert_id,
                  // The signed-in operator's name. The engine records it as a
                  // self-declared label — there is no authentication behind it —
                  // so this is a note about who was at the console, not an identity.
                  by: operatorName ?? 'unknown operator',
                })
              }
              onResolve={() => resolve.mutate(alert.alert_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * What an empty list means, which is not one thing.
 *
 * Three different situations end up here and an operator has to be able to tell them
 * apart, because two are fine and one is a page lying to them:
 *
 * * the filters excluded everything — the rows exist, they are just not shown;
 * * everything raised has been dealt with;
 * * the engine has raised nothing this process run.
 *
 * The first is why this component exists. "Nothing is waiting for you" under a camera
 * filter is how somebody walks away from a site with an open critical alert on it.
 */
function EmptyList({
  narrowed,
  total,
  onClearFilters,
}: {
  narrowed: boolean
  total: number
  onClearFilters: () => void
}) {
  if (narrowed) {
    return (
      <Absent title="No alerts match these filters.">
        <p className="mt-0 mb-2">
          {total === 1 ? 'The one alert' : `All ${total} alerts`} held by this engine{' '}
          {total === 1 ? 'is' : 'are'} outside the camera or severity you picked — not
          absent. Widen the filters to see {total === 1 ? 'it' : 'them'}.
        </p>
        <button
          type="button"
          onClick={onClearFilters}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:border-signal-d hover:text-signal"
        >
          Clear filters
        </button>
      </Absent>
    )
  }

  // An empty register, whichever view is selected. "Everything raised so far has been
  // dealt with" is false when nothing was raised, and it is the more reassuring of the
  // two readings — so the register being empty has to be checked before the view is.
  if (total === 0) {
    return (
      <Absent title="No alerts yet.">
        The engine has not raised an alert in this process run. That is not a claim that
        nothing has happened — it is what an empty, volatile register looks like.
      </Absent>
    )
  }

  return (
    <Absent title="Nothing is waiting for you.">
      Everything raised so far has been dealt with. Switch to “Everything” for the log.
    </Absent>
  )
}

/**
 * Empty the list.
 *
 * Two clicks, and the second names the consequence rather than saying "Confirm". This is
 * the one control here that destroys something: acknowledging and resolving move an
 * alert between states, and this deletes them all. What it cannot destroy is evidence —
 * every event behind these rows is already published — and it says so, because an
 * operator hesitating over it deserves to know exactly what they are losing.
 */
function ClearAll() {
  const clear = useClearAlerts()
  const [armed, setArmed] = useState(false)

  if (!armed) {
    return (
      <div className="mb-4">
        <button
          type="button"
          onClick={() => setArmed(true)}
          className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:border-breach hover:text-breach"
        >
          Clear all
        </button>
      </div>
    )
  }

  return (
    <Notice tone="caution" className="mb-4">
      <p className="mt-0 mb-2">
        This drops every alert, including ones nobody has looked at. The events behind
        them stay published — what is lost is the record of which had been seen.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => clear.mutate(undefined, { onSuccess: () => setArmed(false) })}
          disabled={clear.isPending}
          className="rounded-sm border border-breach bg-[#1a0d0d] px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-breach disabled:opacity-40"
        >
          {clear.isPending ? 'Clearing…' : 'Yes, clear every alert'}
        </button>
        <button
          type="button"
          onClick={() => setArmed(false)}
          className="font-mono text-[11px] uppercase tracking-[0.07em] text-dim underline-offset-2 hover:text-fg hover:underline"
        >
          Keep them
        </button>
        {clear.isError ? (
          <span className="text-[11.5px] text-breach">{clear.error.message}</span>
        ) : null}
      </div>
    </Notice>
  )
}
