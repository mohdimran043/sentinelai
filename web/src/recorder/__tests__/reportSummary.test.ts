import { describe, expect, it } from 'vitest'
import {
  hasNoCoverageRecord,
  integrityTimeBreakdown,
  summarizeGapCauses,
  summarizeRetrievalHoles,
} from '@/recorder/reportSummary'
import type {
  RecorderCoverageInterval,
  RecorderCoverageReport,
  RecorderRetrievalHole,
} from '@/recorder/recorder.types'

function interval(
  state: string,
  gaps: RecorderCoverageInterval['gaps'],
): RecorderCoverageInterval {
  return { start_ns: 0, expected_seconds: 60, recorded_seconds: 0, integrity_state: state, gaps }
}

describe('summarizeGapCauses', () => {
  it('sums duration and count per cause across many intervals, worst first', () => {
    const intervals = [
      interval('LOST', [{ start_ns: 0, duration_ms: 1000, cause: 'STREAM_LOST' }]),
      interval('LOST', [{ start_ns: 0, duration_ms: 2000, cause: 'STREAM_LOST' }]),
      interval('DEGRADED_QUALITY', [{ start_ns: 0, duration_ms: 500, cause: 'DECODE_ERROR' }]),
    ]

    const result = summarizeGapCauses(intervals)

    expect(result).toEqual([
      { cause: 'STREAM_LOST', count: 2, totalMs: 3000 },
      { cause: 'DECODE_ERROR', count: 1, totalMs: 500 },
    ])
  })

  it('returns nothing for an interval list with no gaps', () => {
    expect(summarizeGapCauses([interval('HEALTHY', [])])).toEqual([])
  })

  it('handles an interval with more than one gap', () => {
    const intervals = [
      interval('LOST', [
        { start_ns: 0, duration_ms: 100, cause: 'A' },
        { start_ns: 0, duration_ms: 200, cause: 'A' },
        { start_ns: 0, duration_ms: 50, cause: 'B' },
      ]),
    ]
    expect(summarizeGapCauses(intervals)).toEqual([
      { cause: 'A', count: 2, totalMs: 300 },
      { cause: 'B', count: 1, totalMs: 50 },
    ])
  })
})

describe('summarizeRetrievalHoles', () => {
  const hole = (startNs: number, endNs: number, cause: string): RecorderRetrievalHole => ({
    start_ns: startNs,
    end_ns: endNs,
    cause,
  })

  it('computes each hole duration from end minus start, in milliseconds', () => {
    const holes = [hole(0, 1_000_000, 'expired_by_retention')] // 1,000,000 ns = 1 ms
    const summary = summarizeRetrievalHoles(holes)
    expect(summary.count).toBe(1)
    expect(summary.totalMs).toBe(1)
    expect(summary.byCause).toEqual([{ cause: 'expired_by_retention', count: 1, totalMs: 1 }])
  })

  it('groups by cause and orders worst-total first', () => {
    const holes = [
      hole(0, 1_000_000, 'no_segment'), // 1ms
      hole(0, 5_000_000, 'expired_by_retention'), // 5ms
      hole(0, 5_000_000, 'expired_by_retention'), // 5ms
    ]
    const summary = summarizeRetrievalHoles(holes)
    expect(summary.count).toBe(3)
    expect(summary.totalMs).toBe(11)
    expect(summary.byCause[0]).toEqual({
      cause: 'expired_by_retention',
      count: 2,
      totalMs: 10,
    })
    expect(summary.byCause[1]).toEqual({ cause: 'no_segment', count: 1, totalMs: 1 })
  })

  it('returns a zeroed summary for no holes at all, not undefined', () => {
    expect(summarizeRetrievalHoles([])).toEqual({ count: 0, totalMs: 0, byCause: [] })
  })
})

describe('integrityTimeBreakdown', () => {
  it('turns the record into a list sorted by seconds, worst first', () => {
    const result = integrityTimeBreakdown({ UNDERLIT: 100, OBSTRUCTED: 500, LOST: 10 })
    expect(result).toEqual([
      { state: 'OBSTRUCTED', seconds: 500 },
      { state: 'UNDERLIT', seconds: 100 },
      { state: 'LOST', seconds: 10 },
    ])
  })

  it('returns an empty list for an empty record rather than throwing', () => {
    expect(integrityTimeBreakdown({})).toEqual([])
  })
})

describe('hasNoCoverageRecord', () => {
  function makeReport(overrides: Partial<RecorderCoverageReport> = {}): RecorderCoverageReport {
    return {
      blind_spots: null,
      camera_id: 'room_4b',
      coverage: {
        expected_seconds: 0,
        gap_seconds: 0,
        intervals_with_records: 0,
        recorded_pct: 0,
        recorded_seconds: 0,
        unexplained_shortfall_secs: 0,
      },
      date: '2026-08-01',
      integrity_time: {},
      intervals: [],
      note: '',
      retrieval_holes: [],
      segments: { count: 0, seconds: 0 },
      ...overrides,
    }
  }

  it('is true for an unknown camera or a date entirely outside the record', () => {
    expect(hasNoCoverageRecord(makeReport())).toBe(true)
  })

  it('is false once anything was expected, even if nothing was recorded', () => {
    // A camera that was down (e.g. a fully-LOST day) still has real intervals.
    const report = makeReport({
      coverage: {
        expected_seconds: 44089.86,
        gap_seconds: 44089.57,
        intervals_with_records: 595,
        recorded_pct: 0,
        recorded_seconds: 0,
        unexplained_shortfall_secs: 0,
      },
      intervals: [interval('LOST', [])],
    })
    expect(hasNoCoverageRecord(report)).toBe(false)
  })
})
