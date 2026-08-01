/**
 * Everything that maps a wire enum onto the escalation ramp.
 *
 * Brief: "Severity maps onto the ramp: info/low -> --dim/--signal, medium ->
 * --caution, high/critical -> --breach." Green is nominal-only; it is never used
 * to mean "escalated".
 */

export type Tone = 'nominal' | 'caution' | 'breach' | 'inert'

export type EventSeverity = 'info' | 'low' | 'medium' | 'high' | 'critical'

export function toneForSeverity(severity: string): Tone {
  switch (severity) {
    case 'info':
    case 'low':
      return 'nominal'
    case 'medium':
      return 'caution'
    case 'high':
    case 'critical':
      return 'breach'
    default:
      return 'inert'
  }
}

/**
 * `ModelHealth.state` is one of the eight `LifecycleState` values defined in
 * `ai-engine/sentinel_ai/ports/model_runtime.py` (loaded, unloaded, sleeping,
 * downloading, updating, offline, healthy, unhealthy). The API's `HealthResponse
 * .status` field is always the literal string "ok" regardless of model state —
 * it is not a meaningful liveness signal, so the UI derives tone from each
 * model's own state instead of trusting the top-level status.
 */
export function toneForModelState(state: string): Tone {
  switch (state) {
    case 'loaded':
    case 'healthy':
      return 'nominal'
    case 'downloading':
    case 'updating':
    case 'sleeping':
      return 'caution'
    case 'offline':
    case 'unhealthy':
      return 'breach'
    case 'unloaded':
    default:
      return 'inert'
  }
}

/**
 * `CameraStatus` carries no explicit online/offline flag — only counters and
 * `last_frame_at`. This derives a liveness label from recency, which is the
 * most honest reading of the data actually on the wire. Anything stale reads
 * as "stale" rather than a fabricated "offline", because the engine may simply
 * be running slower than this threshold assumes.
 */
export const STALE_AFTER_SECONDS = 15

export type CameraLiveness = 'live' | 'stale' | 'no-data'

export function cameraLiveness(
  lastFrameAtEpochSeconds: number | null,
  nowMs: number = Date.now(),
): CameraLiveness {
  if (lastFrameAtEpochSeconds === null) return 'no-data'
  const ageSeconds = (nowMs - lastFrameAtEpochSeconds * 1000) / 1000
  return ageSeconds <= STALE_AFTER_SECONDS ? 'live' : 'stale'
}

export function toneForLiveness(liveness: CameraLiveness): Tone {
  switch (liveness) {
    case 'live':
      return 'nominal'
    case 'stale':
      return 'caution'
    case 'no-data':
      return 'inert'
  }
}
