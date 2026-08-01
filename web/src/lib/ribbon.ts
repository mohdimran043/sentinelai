import type { RecentEventEntry } from '@/api/engineClient'
import { toneForSeverity } from '@/lib/severity'
import type { RibbonCell } from '@/components/ui/Ribbon'
import { formatClockTime, formatThreatScore, humanizeEnum } from '@/lib/format'

/**
 * One ribbon cell per ring event, in the order `CameraEventsResponse.events`
 * is documented to arrive in (oldest first — "plot them in this order").
 *
 * Deliberately not time-bucketed: the ring is already bounded at `capacity`
 * (200), so there is no need to invent bucket boundaries that could merge or
 * hide events the ring chose to keep. Every retained event gets its own cell,
 * which is also why this never emits more than 200 cells regardless of how
 * bursty or quiet the camera has been.
 */
export function buildThreatRibbon(events: RecentEventEntry[]): RibbonCell[] {
  return events.map((event): RibbonCell => {
    const tone = toneForSeverity(event.severity)
    const caveat = event.description_unavailable ? ' — description unavailable' : ''
    return {
      tone,
      description:
        `${formatClockTime(event.occurred_at)} — ${humanizeEnum(event.reason)}, ` +
        `${event.severity} (threat ${formatThreatScore(event.threat_score)})${caveat}`,
    }
  })
}
