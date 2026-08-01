import type { SentinelAIAnomalyEvent } from '@/events/anomalyEvent.types'
import { toneForSeverity, type Tone } from '@/lib/severity'
import type { RibbonCell } from '@/components/ui/Ribbon'
import { formatClockTime } from '@/lib/format'

const toneRank: Record<Tone, number> = { inert: 0, nominal: 1, caution: 2, breach: 3 }

export interface BuildRibbonOptions {
  bucketCount?: number
  spanSeconds?: number
  now?: number
}

/**
 * Buckets an event history into fixed-width time slots for the ribbon. A bucket
 * with no events reads as `nominal` — the camera was observed and nothing
 * anomalous happened, which is different from having no data at all. Only
 * buckets entirely in the future (past `now`) are `inert`.
 */
export function buildThreatRibbon(
  events: SentinelAIAnomalyEvent[],
  { bucketCount = 48, spanSeconds = 2 * 60 * 60, now = Date.now() }: BuildRibbonOptions = {},
): RibbonCell[] {
  const nowSeconds = now / 1000
  const startSeconds = nowSeconds - spanSeconds
  const bucketSeconds = spanSeconds / bucketCount

  const cells: { tone: Tone; count: number; worst: SentinelAIAnomalyEvent | null }[] = Array.from(
    { length: bucketCount },
    () => ({ tone: 'nominal', count: 0, worst: null }),
  )

  for (const event of events) {
    if (event.occurred_at < startSeconds || event.occurred_at > nowSeconds) continue
    const index = Math.min(bucketCount - 1, Math.floor((event.occurred_at - startSeconds) / bucketSeconds))
    const cell = cells[index]!
    const tone = toneForSeverity(event.severity)
    cell.count += 1
    if (toneRank[tone] >= toneRank[cell.tone]) {
      cell.tone = tone
      cell.worst = event
    }
  }

  return cells.map((cell, index): RibbonCell => {
    const bucketStart = startSeconds + index * bucketSeconds
    const timeLabel = formatClockTime(bucketStart)
    if (cell.count === 0) {
      return { tone: 'nominal', description: `${timeLabel} — no anomalies observed` }
    }
    const worstLabel = cell.worst ? cell.worst.severity : 'nominal'
    return {
      tone: cell.tone,
      description: `${timeLabel} — ${cell.count} event${cell.count === 1 ? '' : 's'}, worst severity ${worstLabel}`,
    }
  })
}
