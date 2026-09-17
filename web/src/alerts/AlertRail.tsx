import { Link } from 'react-router-dom'
import { useAlerts } from '@/api/queries'
import { AlertRow } from '@/alerts/AlertRow'
import { Absent } from '@/components/ui/Absent'
import { unseenAlerts } from '@/lib/alerts'

const RAIL_LIMIT = 6

/**
 * The standing alert rail (spec §21).
 *
 * Shows **open** alerts only, worst first — the engine has already ordered them,
 * so this does not re-sort and cannot disagree with the alerts page about what is
 * most urgent.
 *
 * Capped at six. A rail that grows without bound stops being a rail and starts
 * being the page; past six rows an operator is scrolling rather than scanning,
 * and the honest response is to send them somewhere built for it.
 *
 * The empty state says "nothing is waiting", not "no data". On a surveillance
 * console those are opposite meanings and an operator must never have to work out
 * which one they are looking at.
 */
export function AlertRail() {
  const query = useAlerts()

  if (query.isPending) {
    return (
      <div className="flex flex-col gap-2" aria-busy="true">
        {[0, 1].map((key) => (
          <div
            key={key}
            className="h-[74px] animate-pulse rounded-md border border-line bg-panel motion-reduce:animate-none"
          />
        ))}
      </div>
    )
  }

  if (query.isError) {
    return (
      <Absent title="Alerts unavailable">
        {query.error.message} Nothing here has been replaced with a zero — this
        console cannot currently tell you whether anything is waiting.
      </Absent>
    )
  }

  const open = query.data.alerts.filter((alert) => alert.state !== 'resolved')

  if (open.length === 0) {
    return (
      <div className="rounded-md border border-line bg-panel p-[15px]">
        <p className="m-0 text-[13px] text-dim">
          Nothing is waiting for you.
        </p>
        <p className="muted m-0 mt-1">
          Alerts appear here the moment the engine raises one. This is not a claim
          that nothing is happening — read{' '}
          <Link
            to="/alerts"
            className="underline decoration-line-2 underline-offset-2 hover:decoration-signal"
          >
            the alert log
          </Link>{' '}
          for what has already been dealt with.
        </p>
      </div>
    )
  }

  const shown = open.slice(0, RAIL_LIMIT)
  const overflow = open.length - shown.length

  return (
    <div className="flex flex-col gap-2">
      {shown.map((alert) => (
        <AlertRow key={alert.alert_id} alert={alert} compact />
      ))}
      {overflow > 0 ? (
        <Link
          to="/alerts"
          className="rounded-md border border-line bg-panel px-[13px] py-[9px] font-mono text-[11.5px] uppercase tracking-[0.08em] text-dim hover:border-line-2 hover:text-fg"
        >
          {overflow} more waiting →
        </Link>
      ) : null}
    </div>
  )
}

/**
 * The unseen count, for the navigation rail.
 *
 * Counts `active` only — an alert somebody has acknowledged is being dealt with,
 * and including it would keep the badge lit while the work is in hand, which is
 * how a badge stops meaning anything.
 */
export function useUnseenAlertCount(): number {
  const query = useAlerts()
  return query.data ? unseenAlerts(query.data.alerts).length : 0
}
