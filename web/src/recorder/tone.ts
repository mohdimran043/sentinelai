import type { Tone } from '@/lib/severity'

/**
 * Recorder enum -> escalation-ramp tone.
 *
 * Kept here rather than in `lib/severity.ts` because these are the recorder's
 * enums, not the AI engine's, and the two products must not share a vocabulary
 * by accident.
 *
 * The recorder's own bundle ships this map:
 *   HEALTHY: ok, CONNECTING: caution, DEGRADED_QUALITY: caution,
 *   UNDERLIT: caution, OBSTRUCTED: breach, FOV_SHIFTED: breach,
 *   LOST: breach, FAILED: breach, everything else: inert.
 *
 * We match it, and DELIBERATELY EXTEND it. Its `everything else: inert` default
 * catches STREAM_LOST, MASK_LOAD_FAILED, ENCODER_WRITE_FAILED and CUDA_ERROR —
 * four of the eight types the recorder itself marks `locked: true`, i.e. the
 * ones an operator is never allowed to suppress. Rendering "the GPU pipeline
 * failed" in the same grey as an informational message would be the opposite
 * of the honesty the rest of this design is built on. The extension only ever
 * raises a tone; it never quiets one.
 */
const EVENT_TYPE_TONE: Record<string, Tone> = {
  // --- matches the recorder's shipped map exactly ---
  HEALTHY: 'nominal',
  CONNECTING: 'caution',
  DEGRADED_QUALITY: 'caution',
  UNDERLIT: 'caution',
  OBSTRUCTED: 'breach',
  FOV_SHIFTED: 'breach',
  LOST: 'breach',
  FAILED: 'breach',

  // --- our extension: locked types the recorder's default left inert ---
  STREAM_LOST: 'breach',
  MASK_LOAD_FAILED: 'breach',
  ENCODER_WRITE_FAILED: 'breach',
  CUDA_ERROR: 'breach',

  // --- our extension: the remaining observed types ---
  DECODE_ERROR_BURST: 'caution',
  CLOCK_STEP_DETECTED: 'caution',
  UNMASKED_FRAME_SERVED: 'caution',
  STREAM_CONNECTED: 'nominal',
  CAMERA_RECOVERED: 'nominal',
}

/**
 * An unrecognised event type renders `inert`, not `nominal`. We do not know how
 * serious it is, and grey says "unknown" where green would say "fine".
 */
export function toneForRecorderEventType(eventType: string): Tone {
  return EVENT_TYPE_TONE[eventType] ?? 'inert'
}

/** `integrity_state` from GET /api/status uses the same vocabulary. */
export function toneForIntegrityState(state: string): Tone {
  return toneForRecorderEventType(state)
}

/** Delivery outcome -> tone. Matches the recorder's `{delivered: ok, failed: breach, unrouted: caution}`. */
export function toneForDeliveryStatus(status: string): Tone {
  switch (status) {
    case 'delivered':
      return 'nominal'
    case 'failed':
      return 'breach'
    case 'unrouted':
      return 'caution'
    default:
      return 'inert'
  }
}

/**
 * Perception tier build status. `PARTIAL` is caution and `NOT_BUILT` is breach
 * in the recorder's own console — not inert — because a tier that does not
 * exist is a thing the operator must not assume is working.
 */
export function toneForTierStatus(status: string): Tone {
  switch (status) {
    case 'BUILT':
      return 'nominal'
    case 'PARTIAL':
      return 'caution'
    default:
      return 'breach'
  }
}

/**
 * A journal day's own `status`, and a journal revision's `status` — same
 * vocabulary. `sealed` is the settled, evidentiary end state (nominal, not
 * merely "ok"); `provisional` means the day is still being written and the
 * numbers can still move, which is worth a caution rather than a shrug.
 */
export function toneForJournalStatus(status: string): Tone {
  switch (status) {
    case 'sealed':
      return 'nominal'
    case 'provisional':
      return 'caution'
    default:
      return 'inert'
  }
}
