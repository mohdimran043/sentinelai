import type { AlertEntry, Capability, EscalationReason } from '@/api/engineClient'
import type { Tone } from '@/lib/severity'

/**
 * What an operator reads instead of a wire enum.
 *
 * Written out rather than derived by replacing underscores, because the two
 * disagree in exactly the cases that matter. `fall_suspected` must not render as
 * "Fall Suspected" title-cased into something that looks like a classifier's
 * verdict, and `camera_tamper` is about the camera rather than about a person —
 * which "Camera Tamper" does not convey and "Camera may be blind" does.
 */
export const REASON_LABELS: Record<EscalationReason, string> = {
  fall_suspected: 'Possible fall',
  unauthorized_person: 'Unrecognised person',
  camera_tamper: 'Camera may be blind',
  zone_intrusion: 'Restricted area entered',
  line_crossing: 'Boundary crossed',
  abandoned_object: 'Object left behind',
  dwell_exceeded: 'Loitering',
  speed_anomaly: 'Unusual speed',
  track_count_spike: 'Crowd forming',
  scene_change: 'Scene changed',
  new_salient_track: 'New arrival',
  periodic_summary: 'Routine look',
  user_requested: 'Requested by operator',
}

export function reasonLabel(reason: string): string {
  return REASON_LABELS[reason as EscalationReason] ?? reason.replaceAll('_', ' ')
}

/**
 * Priority drives the rail's colour, not severity.
 *
 * The two are different questions — severity is how alarming the scene looked to
 * a model, priority is how bad it would be to get this one wrong — and priority
 * is the one an operator triages on. A suspected fall the model described calmly
 * still belongs at the top in red.
 */
export function toneForPriority(priority: string): Tone {
  switch (priority) {
    case 'critical':
    case 'high':
      return 'breach'
    case 'medium':
      return 'caution'
    default:
      return 'inert'
  }
}

/** Alerts nobody has looked at yet. The count worth putting on a badge. */
export function unseenAlerts(alerts: readonly AlertEntry[]): AlertEntry[] {
  return alerts.filter((alert) => alert.state === 'active')
}

/**
 * Human-readable capability names.
 *
 * `camera_tamper` is deliberately phrased as what it watches for rather than
 * what it is called: an operator choosing capabilities is deciding what this
 * camera should notice, and "Camera health" says that where "Camera tamper"
 * reads like an accusation the camera makes.
 */
export const CAPABILITY_LABELS: Record<Capability, string> = {
  scene_description: 'Scene description',
  anomaly_detection: 'Anomaly triggers',
  fall_detection: 'Fall detection',
  abandoned_object: 'Abandoned objects',
  camera_tamper: 'Camera health',
  zone_monitoring: 'Zones & boundaries',
  person_authorization: 'Person authorisation',
}

/**
 * What each capability costs, for the camera configuration screen.
 *
 * Shown because §13's promise — that a disabled capability loads no model — is
 * only meaningful to an operator who can see what enabling one would cost. The
 * strings match `domain/capabilities.py`'s role table.
 */
export const CAPABILITY_COST: Record<Capability, string> = {
  scene_description: 'Vision-language model',
  anomaly_detection: 'Object detector',
  fall_detection: 'Object detector + pose',
  abandoned_object: 'Object detector',
  camera_tamper: 'No model',
  zone_monitoring: 'Object detector',
  person_authorization: 'Object detector + face',
}

export const CAPABILITY_ORDER: Capability[] = [
  'anomaly_detection',
  'scene_description',
  'fall_detection',
  'abandoned_object',
  'zone_monitoring',
  'camera_tamper',
  'person_authorization',
]

/**
 * How long an alert has been running, as a phrase.
 *
 * Duration rather than "x minutes ago", because an alert is an episode: what an
 * operator needs is how long this has been going on, which for a person on the
 * floor is the single most important number on the screen.
 */
export function formatSpan(firstSeen: number, lastSeen: number): string {
  const seconds = Math.max(0, Math.round(lastSeen - firstSeen))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}
