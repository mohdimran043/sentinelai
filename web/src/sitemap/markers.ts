import type { RecentEventEntry } from '@/api/engineClient'
import { toneForSeverity, type Tone } from '@/lib/severity'

/**
 * The most recent event for each `camera_id` in a merged event list (see
 * `mergeEngineEvents` — this expects its output, one row per `event_id`
 * already de-duplicated by `sequence`). Ties on `occurred_at` are broken by
 * `sequence`, so a version update that lands with the same timestamp as the
 * copy it replaces is still recognised as the newer one.
 */
export function latestEventByCamera(
  events: readonly RecentEventEntry[],
): Map<string, RecentEventEntry> {
  const latest = new Map<string, RecentEventEntry>()
  for (const event of events) {
    const current = latest.get(event.camera_id)
    if (
      !current ||
      event.occurred_at > current.occurred_at ||
      (event.occurred_at === current.occurred_at && event.sequence > current.sequence)
    ) {
      latest.set(event.camera_id, event)
    }
  }
  return latest
}

/**
 * A camera with no event yet is `inert` (grey/dim), never `nominal` (green).
 * Defaulting an unescalated marker to green would make a quiet map a wall of
 * green competing with the amber -> red escalation ramp — the design brief's
 * palette rule this project has held throughout. Green is earned the same
 * way it is everywhere else in this console: by an actual info/low-severity
 * event (`toneForSeverity`), not by the mere absence of trouble.
 */
export function markerTone(event: RecentEventEntry | undefined): Tone {
  return event ? toneForSeverity(event.severity) : 'inert'
}
