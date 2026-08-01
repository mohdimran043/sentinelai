import { describe, expect, it } from 'vitest'
import { joinNotificationRules } from '@/recorder/notificationJoin'
import type {
  RecorderNotificationChannel,
  RecorderNotificationEventType,
} from '@/recorder/recorder.types'

function eventType(overrides: Partial<RecorderNotificationEventType> = {}): RecorderNotificationEventType {
  return {
    type: 'LOST',
    description: 'No frames arrived for long enough to count as an outage.',
    locked: true,
    default_enabled: true,
    ...overrides,
  }
}

const DASHBOARD: RecorderNotificationChannel = {
  name: 'dashboard',
  configured: true,
  detail: 'In-app alert feed.',
}
const WEBHOOK_UNCONFIGURED: RecorderNotificationChannel = {
  name: 'webhook',
  configured: false,
  detail: 'No URL configured. Any alert routed here reaches nobody.',
}

describe('joinNotificationRules', () => {
  it('attaches each event type to its rule by event_type', () => {
    const rows = joinNotificationRules(
      [eventType({ type: 'LOST' }), eventType({ type: 'OBSTRUCTED', locked: false })],
      [
        { event_type: 'LOST', channels: ['dashboard'], enabled: true },
        { event_type: 'OBSTRUCTED', channels: ['dashboard'], enabled: true },
      ],
      [DASHBOARD],
    )

    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ type: 'LOST', enabled: true, channels: ['dashboard'] })
  })

  it('carries the description through untouched — it is the substance of the page', () => {
    const rows = joinNotificationRules(
      [eventType({ description: 'The GPU pipeline failed. Decode, masking and encode all stop.' })],
      [{ event_type: 'LOST', channels: ['dashboard'], enabled: true }],
      [DASHBOARD],
    )

    expect(rows[0]!.description).toBe(
      'The GPU pipeline failed. Decode, masking and encode all stop.',
    )
  })

  it('marks enabled undefined, not false, when the recorder declares a type with no matching rule', () => {
    const rows = joinNotificationRules([eventType({ type: 'CLOCK_STEP_DETECTED' })], [], [DASHBOARD])

    expect(rows[0]!.enabled).toBeUndefined()
    expect(rows[0]!.channels).toEqual([])
  })

  it('flags a locked type as undeliverable when its only channel is unconfigured', () => {
    const rows = joinNotificationRules(
      [eventType({ type: 'STREAM_LOST', locked: true })],
      [{ event_type: 'STREAM_LOST', channels: ['webhook'], enabled: true }],
      [DASHBOARD, WEBHOOK_UNCONFIGURED],
    )

    expect(rows[0]!.lockedButUndeliverable).toBe(true)
  })

  it('flags a locked type as undeliverable when it has no channel at all', () => {
    const rows = joinNotificationRules(
      [eventType({ type: 'MASK_LOAD_FAILED', locked: true })],
      [{ event_type: 'MASK_LOAD_FAILED', channels: [], enabled: true }],
      [DASHBOARD],
    )

    expect(rows[0]!.lockedButUndeliverable).toBe(true)
  })

  it('does not flag a locked type routed to a configured channel', () => {
    const rows = joinNotificationRules(
      [eventType({ type: 'FAILED', locked: true })],
      [{ event_type: 'FAILED', channels: ['dashboard'], enabled: true }],
      [DASHBOARD, WEBHOOK_UNCONFIGURED],
    )

    expect(rows[0]!.lockedButUndeliverable).toBe(false)
  })

  it('never flags an unlocked type as lockedButUndeliverable, even with no channel', () => {
    const rows = joinNotificationRules(
      [eventType({ type: 'STREAM_CONNECTED', locked: false })],
      [{ event_type: 'STREAM_CONNECTED', channels: [], enabled: false }],
      [DASHBOARD],
    )

    expect(rows[0]!.lockedButUndeliverable).toBe(false)
  })

  it('reflects the real 16-type live payload shape: every locked type routes to the configured dashboard channel', () => {
    const rows = joinNotificationRules(
      ['STREAM_LOST', 'LOST', 'FAILED', 'MASK_LOAD_FAILED', 'ENCODER_WRITE_FAILED', 'CUDA_ERROR'].map(
        (type) => eventType({ type, locked: true }),
      ),
      ['STREAM_LOST', 'LOST', 'FAILED', 'MASK_LOAD_FAILED', 'ENCODER_WRITE_FAILED', 'CUDA_ERROR'].map(
        (event_type) => ({ event_type, channels: ['dashboard'], enabled: true }),
      ),
      [DASHBOARD, WEBHOOK_UNCONFIGURED],
    )

    expect(rows.every((row) => row.lockedButUndeliverable === false)).toBe(true)
  })
})
