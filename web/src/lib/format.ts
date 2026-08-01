/**
 * Formatting helpers shared by telemetry readouts. Numbers render through
 * `Intl.NumberFormat` so digit grouping is consistent; pair with the `.readout`
 * class (tabular-nums) so values do not jitter as they update.
 */

const integerFormatter = new Intl.NumberFormat('en-US')

export function formatCount(value: number): string {
  return integerFormatter.format(value)
}

/** `occurred_at` / `last_frame_at` etc. arrive as epoch seconds (a Python `time.time()`). */
export function epochSecondsToDate(epochSeconds: number): Date {
  return new Date(epochSeconds * 1000)
}

export function formatClockTime(epochSeconds: number): string {
  return epochSecondsToDate(epochSeconds).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** "3s ago", "4m ago" — coarse enough for an operator glance, never invents precision. */
export function formatAgo(epochSeconds: number, now: number = Date.now()): string {
  const deltaMs = now - epochSeconds * 1000
  if (deltaMs < 0) return 'just now'
  const seconds = Math.floor(deltaMs / 1000)
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

/** `new_salient_track` -> "New salient track". Enums come off the wire as snake_case; never render them raw. */
export function humanizeEnum(value: string): string {
  const words = value.split('_').filter(Boolean)
  if (words.length === 0) return value
  return words
    .map((word, index) => (index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : word))
    .join(' ')
}

export function formatPercent(ratio: number): string {
  return `${(ratio * 100).toFixed(0)}%`
}

export function formatThreatScore(score: number): string {
  return score.toFixed(2)
}
