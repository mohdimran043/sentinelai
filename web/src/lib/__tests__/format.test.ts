// oxlint-disable no-loss-of-precision -- These are real recorder timestamps.
// The rounding the rule warns about is the production behaviour under test.
import { describe, expect, it } from 'vitest'
import {
  formatBytes,
  formatNanosDateTime,
  formatNanosPrecise,
  formatUsd,
  nanosToDate,
} from '@/lib/format'

// `vite.config.ts` pins TZ=UTC for the suite, so these are the UTC renderings.
describe('nanosecond timestamps', () => {
  it('converts nanoseconds to a Date at millisecond resolution', () => {
    // 1785578902544883331 ns = 2026-08-01T10:08:22.544Z
    expect(nanosToDate(1785578902544883331).toISOString()).toBe('2026-08-01T10:08:22.544Z')
  })

  it('renders a fixed-width, year-first stamp so a mono column cannot jitter', () => {
    expect(formatNanosDateTime(1785578902544883331)).toBe('2026-08-01 10:08:22')
    // Single-digit month/day/hour still pad to the same width.
    expect(formatNanosDateTime(Date.UTC(2026, 0, 5, 3, 7, 9) * 1_000_000)).toBe(
      '2026-01-05 03:07:09',
    )
    expect(formatNanosDateTime(1785578902544883331)).toHaveLength(19)
  })

  it('adds milliseconds for the detail view', () => {
    expect(formatNanosPrecise(1785578902544883331)).toBe('2026-08-01 10:08:22.544')
  })

  it('says "—" rather than "Invalid Date" for an unusable value', () => {
    expect(formatNanosDateTime(Number.NaN)).toBe('—')
    expect(formatNanosPrecise(Number.NaN)).toBe('—')
  })
})

describe('formatBytes', () => {
  it('uses SI units, matching how model downloads are advertised', () => {
    // The recorder reports qwen2.5vl:3b at exactly this size.
    expect(formatBytes(3200627168)).toBe('3.2 GB')
    expect(formatBytes(5969245856)).toBe('6.0 GB')
    expect(formatBytes(84)).toBe('84 B')
    expect(formatBytes(28_000)).toBe('28.0 KB')
    expect(formatBytes(4_500_000)).toBe('4.5 MB')
  })
})

describe('formatUsd', () => {
  it('renders per-MTok prices to two decimals so a column lines up', () => {
    expect(formatUsd(5)).toBe('$5.00')
    expect(formatUsd(0.8)).toBe('$0.80')
    expect(formatUsd(25)).toBe('$25.00')
  })
})
