import type {
  RecorderCoverageInterval,
  RecorderCoverageReport,
  RecorderRetrievalHole,
} from '@/recorder/recorder.types'

/**
 * Pure helpers for the 86-114 KB coverage report/journal-day documents.
 *
 * A full day carries hundreds of `intervals` (594 on the live instance for one
 * camera-day) and dozens of `retrieval_holes` (91 observed). Rendering one row
 * per entry is not what an operator needs — a grouped cause breakdown is.
 * Kept as pure functions, separate from the page, for the same reason
 * `alertQuery.ts` is: testable without a DOM, and the page cannot quietly
 * diverge from this logic.
 */

export interface CauseTally {
  cause: string
  count: number
  totalMs: number
}

/** Every gap across every interval, grouped by cause and summed, worst first. */
export function summarizeGapCauses(intervals: readonly RecorderCoverageInterval[]): CauseTally[] {
  const byCause = new Map<string, CauseTally>()
  for (const interval of intervals) {
    for (const gap of interval.gaps) {
      const existing = byCause.get(gap.cause)
      if (existing) {
        existing.count += 1
        existing.totalMs += gap.duration_ms
      } else {
        byCause.set(gap.cause, { cause: gap.cause, count: 1, totalMs: gap.duration_ms })
      }
    }
  }
  return [...byCause.values()].sort((a, b) => b.totalMs - a.totalMs)
}

export interface RetrievalHoleSummary {
  count: number
  totalMs: number
  byCause: CauseTally[]
}

/** Retrieval holes grouped by cause, worst first, plus the overall total. */
export function summarizeRetrievalHoles(
  holes: readonly RecorderRetrievalHole[],
): RetrievalHoleSummary {
  const byCause = new Map<string, CauseTally>()
  let totalMs = 0
  for (const hole of holes) {
    const durationMs = (hole.end_ns - hole.start_ns) / 1_000_000
    totalMs += durationMs
    const existing = byCause.get(hole.cause)
    if (existing) {
      existing.count += 1
      existing.totalMs += durationMs
    } else {
      byCause.set(hole.cause, { cause: hole.cause, count: 1, totalMs: durationMs })
    }
  }
  return {
    count: holes.length,
    totalMs,
    byCause: [...byCause.values()].sort((a, b) => b.totalMs - a.totalMs),
  }
}

export interface IntegrityTimeRow {
  state: string
  seconds: number
}

/** `integrity_time`'s record turned into a sorted list — worst (most time) first. */
export function integrityTimeBreakdown(
  integrityTime: Record<string, number>,
): IntegrityTimeRow[] {
  return Object.entries(integrityTime)
    .map(([state, seconds]) => ({ state, seconds }))
    .sort((a, b) => b.seconds - a.seconds)
}

/**
 * True when the recorder has no coverage record at all for this camera-day —
 * an unknown camera, or a date entirely outside anything ever recorded. This
 * is distinct from "expected coverage exists but none of it was recorded"
 * (e.g. a camera that was down all day), which is a real, alarming report and
 * must not be flattened into the same empty state.
 */
export function hasNoCoverageRecord(report: RecorderCoverageReport): boolean {
  return report.coverage.expected_seconds === 0 && report.intervals.length === 0
}
