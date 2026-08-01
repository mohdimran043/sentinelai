import type { SentinelAIAnomalyEvent } from '@/events/anomalyEvent.types'

/** Deterministic PRNG (mulberry32) so mock data — and therefore tests — is stable
 * across reloads without needing a fixture file. */
function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashSeed(value: string): number {
  let hash = 0
  for (let i = 0; i < value.length; i++) {
    hash = (Math.imul(31, hash) + value.charCodeAt(i)) | 0
  }
  return hash
}

function uuidFrom(rand: () => number): string {
  const bytes = Array.from({ length: 16 }, () => Math.floor(rand() * 256))
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = bytes.map((b) => b.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

const REASONS = [
  'new_salient_track',
  'scene_change',
  'dwell_exceeded',
  'speed_anomaly',
  'track_count_spike',
  'periodic_summary',
] as const

const LABEL_POOL = ['person', 'vehicle', 'backpack', 'bicycle', 'dog'] as const

const DESCRIPTIONS: Record<(typeof REASONS)[number], string> = {
  new_salient_track: 'A new person entered the frame and is being tracked.',
  scene_change: 'The scene changed significantly from the learned baseline.',
  dwell_exceeded: 'A person has remained in the same area longer than expected.',
  speed_anomaly: 'A tracked object is moving faster than the camera normally sees.',
  track_count_spike: 'The number of simultaneously tracked objects rose sharply.',
  periodic_summary: 'Routine scene check — nothing unusual observed.',
}

const SUGGESTED_ACTIONS: Record<(typeof REASONS)[number], string> = {
  new_salient_track: 'Review the live feed if unexpected.',
  scene_change: 'Confirm the camera has not been moved or obstructed.',
  dwell_exceeded: 'Check whether this is a known routine before escalating.',
  speed_anomaly: 'Review footage for a vehicle or running person.',
  track_count_spike: 'Check for a crowd or gathering.',
  periodic_summary: 'No action needed.',
}

function severityFor(threatScore: number): SentinelAIAnomalyEvent['severity'] {
  if (threatScore >= 0.85) return 'critical'
  if (threatScore >= 0.65) return 'high'
  if (threatScore >= 0.35) return 'medium'
  if (threatScore >= 0.15) return 'low'
  return 'info'
}

export interface MockEventOptions {
  /** How many events to generate, spread backwards from `now`. */
  count?: number
  /** Total span the events are spread across, in seconds. */
  spanSeconds?: number
  now?: number
}

/** Generates a deterministic, schema-valid event history for one camera. Same
 * `cameraId` always produces the same events, which keeps both the UI (on
 * reload) and tests reproducible without a checked-in fixture file. */
export function generateMockEvents(
  cameraId: string,
  { count = 24, spanSeconds = 2 * 60 * 60, now = Date.now() }: MockEventOptions = {},
): SentinelAIAnomalyEvent[] {
  const rand = mulberry32(hashSeed(cameraId))
  const nowSeconds = now / 1000
  const events: SentinelAIAnomalyEvent[] = []

  for (let i = 0; i < count; i++) {
    const reason = REASONS[Math.floor(rand() * REASONS.length)]!
    const occurredAt = nowSeconds - rand() * spanSeconds
    const threatScore = Math.round(rand() * 100) / 100
    const severity = severityFor(threatScore)
    const descriptionUnavailable = rand() < 0.08
    const labelCount = 1 + Math.floor(rand() * 2)
    const labels = Array.from(
      new Set(Array.from({ length: labelCount }, () => LABEL_POOL[Math.floor(rand() * LABEL_POOL.length)]!)),
    )
    const trackIdCount = 1 + Math.floor(rand() * 3)
    const trackIds = Array.from({ length: trackIdCount }, () => Math.floor(rand() * 200))

    events.push({
      schema_version: 1,
      event_id: uuidFrom(rand),
      camera_id: cameraId,
      occurred_at: occurredAt,
      reason,
      threat_score: threatScore,
      severity,
      description: descriptionUnavailable ? '' : DESCRIPTIONS[reason],
      suggested_action: SUGGESTED_ACTIONS[reason],
      labels,
      track_ids: trackIds,
      keyframe_uri: null,
      clip_uri: null,
      description_unavailable: descriptionUnavailable,
      metadata: { mock: 'true', generator: 'sentinelai-web-msw' },
    })
  }

  return events.sort((a, b) => b.occurred_at - a.occurred_at)
}
