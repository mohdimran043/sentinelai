import { Link } from 'react-router-dom'
import type { AlertEntry } from '@/api/engineClient'
import { AlertClip } from '@/alerts/AlertClip'
import { Pill } from '@/components/ui/Pill'
import { cn } from '@/lib/cn'
import { formatSpan, reasonLabel, toneForPriority } from '@/lib/alerts'
import { formatAgo } from '@/lib/format'

export interface AlertRowProps {
  alert: AlertEntry
  /**
   * Compact form for the dashboard rail, where vertical space is the scarce
   * resource and the operator's next action is to click through. The full form
   * on the alerts page carries the controls instead.
   */
  compact?: boolean
  onAcknowledge?: () => void
  onResolve?: () => void
  busy?: boolean
}

/**
 * One alert, as an operator scans it.
 *
 * The reading order is deliberate and it is not the order the fields arrive in:
 * **what**, then **where**, then **how long**, then **how many**. An operator
 * triaging a wall of these decides from the first two and only needs the rest
 * once they have chosen a row — so the first two are the large type and the rest
 * is the mono readout line beneath.
 *
 * Occurrences appear only when there is more than one. "×1" on every row is
 * noise that makes the genuine "×17" harder to see, which is the whole point of
 * aggregating in the first place.
 */
export function AlertRow({ alert, compact, onAcknowledge, onResolve, busy }: AlertRowProps) {
  const tone = toneForPriority(alert.priority)
  const acknowledged = alert.state === 'acknowledged'
  const resolved = alert.state === 'resolved'

  return (
    <article
      className={cn(
        'relative rounded-md border bg-panel',
        compact ? 'p-[11px_13px]' : 'p-[14px_16px]',
        // A 2px left rule, not a thick colour bar: the craft floor bans anything
        // heavier, and at this density a hairline reads as well.
        'before:absolute before:inset-y-0 before:left-0 before:w-[2px] before:content-[""]',
        tone === 'breach' && 'border-[#3a1c17] before:bg-breach',
        tone === 'caution' && 'border-[#3a2c0a] before:bg-caution',
        tone === 'inert' && 'border-line before:bg-line-2',
        // A resolved row stays legible but stops competing. Opacity rather than a
        // grey palette, so the priority colour is still readable at a glance.
        resolved && 'opacity-55',
      )}
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className={cn('font-semibold', compact ? 'text-[13.5px]' : 'text-[15px]')}>
          {reasonLabel(alert.reason)}
        </h3>
        <div className="flex shrink-0 items-center gap-1.5">
          {alert.occurrences > 1 ? (
            <span className="readout text-[11.5px] text-dim" title="times this has recurred">
              ×{alert.occurrences}
            </span>
          ) : null}
          <Pill tone={resolved ? 'inert' : tone}>{alert.priority}</Pill>
        </div>
      </div>

      <div className="mt-1 flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[12.5px]">
        <Link
          to={`/cameras/${encodeURIComponent(alert.camera_id)}`}
          className="text-fg underline decoration-line-2 underline-offset-2 hover:decoration-signal"
        >
          {alert.camera_label}
        </Link>
        {alert.zone ? <span className="text-dimmer">·</span> : null}
        {alert.zone ? <span className="text-dim">{alert.zone}</span> : null}
      </div>

      {compact ? null : (
        <p className="muted mt-2 mb-0 max-w-[80ch]">{alert.description}</p>
      )}

      {/* Only on the full row. The dashboard rail is a list of what is happening, read
          at a glance and clicked through — a video player in it would compete with the
          twelve rows around it for the attention the rail exists to direct. */}
      {compact ? null : (
        <AlertClip
          alertId={alert.alert_id}
          clipUri={alert.clip_uri ?? null}
          notifyClipUri={alert.notify_clip_uri ?? null}
        />
      )}

      <div className="readout mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-dimmer">
        <span title="when this episode started">{formatAgo(alert.first_seen)}</span>
        {alert.occurrences > 1 ? (
          <span title="from first to most recent sighting">
            over {formatSpan(alert.first_seen, alert.last_seen)}
          </span>
        ) : null}
        {alert.subject ? <span title="tracked subject">track {alert.subject}</span> : null}
        {/* The badge above shows *priority* — how costly this would be to get wrong —
            and severity is the other question: how alarming the scene looked to the
            model. They diverge often enough to matter (a critical-looking scene on a
            low-priority reason is real), and the alert list filters on severity. A
            filter whose criterion is invisible on the rows it produced is one an
            operator cannot check, so it is printed here rather than left on the wire. */}
        <span title="how alarming the scene looked to the model">
          severity {alert.severity}
        </span>
        {acknowledged && alert.acknowledged_by ? (
          <span className="text-signal">seen by {alert.acknowledged_by}</span>
        ) : null}
        {resolved ? <span>resolved</span> : null}
      </div>

      {compact || (!onAcknowledge && !onResolve) ? null : (
        <div className="mt-3 flex flex-wrap gap-2">
          {onAcknowledge && !acknowledged && !resolved ? (
            <button
              type="button"
              onClick={onAcknowledge}
              disabled={busy}
              className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-fg hover:border-signal-d hover:text-signal disabled:opacity-50"
            >
              I'm on it
            </button>
          ) : null}
          {onResolve && !resolved ? (
            <button
              type="button"
              onClick={onResolve}
              disabled={busy}
              className="rounded-sm border border-line-2 bg-panel-2 px-[11px] py-[5px] font-mono text-[11px] uppercase tracking-[0.08em] text-dim hover:border-line-2 hover:text-fg disabled:opacity-50"
            >
              Close
            </button>
          ) : null}
        </div>
      )}
    </article>
  )
}
