import type {
  RecorderNotificationChannel,
  RecorderNotificationEventType,
  RecorderNotificationRule,
} from '@/recorder/recorder.types'

/**
 * `GET /api/notifications` returns `event_types` (description, locked,
 * default_enabled) and `rules` (the currently-applied channels/enabled) as two
 * parallel arrays keyed by `type`/`event_type`. This is their join, kept pure
 * so it is testable without rendering `NotificationsPage`.
 */
export interface JoinedNotificationRow {
  type: string
  description: string
  locked: boolean
  defaultEnabled: boolean
  /** `undefined` when the recorder declares a type with no matching rule — a gap worth showing, not hiding. */
  enabled: boolean | undefined
  channels: string[]
  /**
   * True when this row is locked (so it may never be suppressed) but its own
   * rule cannot actually deliver anything — no channel at all, or only
   * channels that are not configured. That is exactly the fault the
   * recorder's own `note` warns must never happen; the join computes it so
   * the page can say so plainly rather than silently rendering channel chips.
   */
  lockedButUndeliverable: boolean
}

export function joinNotificationRules(
  eventTypes: RecorderNotificationEventType[],
  rules: RecorderNotificationRule[],
  channels: RecorderNotificationChannel[],
): JoinedNotificationRow[] {
  const ruleByType = new Map(rules.map((rule) => [rule.event_type, rule]))
  const configuredChannels = new Set(
    channels.filter((channel) => channel.configured).map((channel) => channel.name),
  )

  return eventTypes.map((eventType) => {
    const rule = ruleByType.get(eventType.type)
    const ruleChannels = rule?.channels ?? []
    const hasDeliverableChannel = ruleChannels.some((channel) => configuredChannels.has(channel))

    return {
      type: eventType.type,
      description: eventType.description,
      locked: eventType.locked,
      defaultEnabled: eventType.default_enabled,
      enabled: rule?.enabled,
      channels: ruleChannels,
      lockedButUndeliverable: eventType.locked && !hasDeliverableChannel,
    }
  })
}
