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

/**
 * The recorder stamps everything in nanoseconds since the epoch. `Date` takes
 * milliseconds, so divide — and note that the incoming number has already lost
 * sub-microsecond precision to `JSON.parse` (see `recorder/recorder.types.ts`).
 */
export function nanosToDate(atNs: number): Date {
  return new Date(atNs / 1_000_000)
}

const pad = (value: number, width = 2): string => String(value).padStart(width, '0')

/**
 * `1785578902544883331` -> `"2026-08-01 01:28:22"`, in the operator's local
 * time zone.
 *
 * Deliberately not `toLocaleString()` (which is what the recorder's own console
 * uses): a fixed, zero-padded, year-first layout sorts correctly by eye, is the
 * same width on every row so a tabular-mono column does not jitter, and cannot
 * be read as either D/M/Y or M/D/Y.
 */
export function formatNanosDateTime(atNs: number): string {
  const date = nanosToDate(atNs)
  if (Number.isNaN(date.getTime())) return '—'
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    ` ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  )
}

/** Same instant with milliseconds, for a detail view where exactness matters. */
export function formatNanosPrecise(atNs: number): string {
  const date = nanosToDate(atNs)
  if (Number.isNaN(date.getTime())) return '—'
  return `${formatNanosDateTime(atNs)}.${pad(date.getMilliseconds(), 3)}`
}

/**
 * SI bytes: 3200627168 -> "3.2 GB". Decimal rather than binary units because
 * that is how model downloads are advertised, and calling 3.0 GiB "3.0 GB"
 * would understate a download against the number the operator was quoted.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return '—'
  if (bytes < 1_000) return `${formatCount(bytes)} B`
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(1)} KB`
  if (bytes < 1_000_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`
}

/** Frontier pricing is quoted per million tokens: `5` -> `"$5.00"`. */
export function formatUsd(amount: number): string {
  return `$${amount.toFixed(2)}`
}

export function formatThreatScore(score: number): string {
  return score.toFixed(2)
}
