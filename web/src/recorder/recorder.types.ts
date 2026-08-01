/**
 * Types for the recorder API (`sentinel-ingest` on :8080).
 *
 * PROVENANCE: the recorder publishes no OpenAPI document (`/openapi.json`
 * returns its SPA `index.html`, `/api/` returns
 * `{"error":"no such endpoint"}`), so unlike `src/api/engine.types.ts` these
 * cannot be generated. Every field below was read off a live 200 response on
 * 2026-08-01 from the running appliance. Where a field's full domain could not
 * be observed (only one or two values were in the data), the type is widened
 * with `(string & {})` rather than pretending the observed set is exhaustive.
 *
 * Anything NOT observed is not declared. If the recorder starts sending a field
 * this file does not know about, the UI ignores it — it never invents one.
 */

/* ------------------------------------------------------------------ *
 * A note on nanosecond timestamps
 * ------------------------------------------------------------------ *
 * `at_ns`, `server_time_ns` and `modified_ns` are nanoseconds since the epoch,
 * e.g. 1785578902544883331. That exceeds `Number.MAX_SAFE_INTEGER`
 * (9007199254740991), so `JSON.parse` rounds them: the value that reaches the
 * UI is accurate to roughly a quarter of a microsecond, not to the nanosecond.
 *
 * That is harmless for everything this console does with them (millisecond
 * display, ordering, bucketing) but it means a `ns` value from here must never
 * be used as an identity or sent back to the recorder as an exact match key.
 * Alerts carry a separate string `id` for that.
 */
export type Nanoseconds = number

/* ------------------------------------------------------------------ *
 * GET /api/cameras
 * ------------------------------------------------------------------ */

export interface RecorderCamera {
  id: string
  name: string
  /** Observed: "room", "common_area". */
  mode: 'room' | 'common_area' | (string & {})
  /** Observed: "room", "corridor", "dayroom". */
  space_type: string
  /** Observed: "v4l2:/dev/video0", "rtsp://host:554/path". */
  source: string
  width: number
  height: number
  fps: number
  /** Empty string when no mask file is configured — not null. */
  mask_path: string
  preroll_seconds: number
  elevated_watch: boolean
  capacity: number
  audio_enabled: boolean
  face_recognition_enabled: boolean
  /** Always `[]` in every probe, so the element shape is genuinely unknown. */
  ignored_zones: unknown[]
}

export interface RecorderCamerasResponse {
  cameras: RecorderCamera[]
  /**
   * The recorder's own disclaimer that `elevated_watch`, `capacity`,
   * `audio_enabled`, `face_recognition_enabled` and `ignored_zones` are stored
   * and validated but read by nothing. Render it; do not summarise it away.
   */
  context_note: string
}

/* ------------------------------------------------------------------ *
 * GET /api/status
 * ------------------------------------------------------------------ */

export interface RecorderWorkerStatus {
  camera_id: string
  mode: string
  /** Observed: "LOST", "OBSTRUCTED". The recorder's bundle also names HEALTHY, CONNECTING, FOV_SHIFTED, FAILED. */
  integrity_state: string
  running: boolean
  failed: boolean
  restarts: number
  /** `-1` means no frame has ever arrived — it is a sentinel, not an age. */
  last_frame_age_ms: number
  frames_dropped: number
}

export interface RecorderStatusResponse {
  cameras: RecorderWorkerStatus[]
  server_time_ns: Nanoseconds
}

/* ------------------------------------------------------------------ *
 * POST /api/cameras/{camera_id}/start and /stop
 * ------------------------------------------------------------------ *
 * NOT PROBED: unlike every GET in this file, calling one of these live would
 * itself stop or start a real running recorder worker, so no response body
 * was captured. `recorderClient.ts` trusts a 2xx by HTTP status alone and does
 * not parse or type its body; the one fact the UI relies on afterward is a
 * fresh GET /api/status. There is deliberately no response type declared
 * here — see the "anything NOT observed is not declared" rule above.
 */

/* ------------------------------------------------------------------ *
 * GET /api/alerts
 * ------------------------------------------------------------------ */

export interface RecorderAlert {
  /** `"<at_ns>-<seq>"`, e.g. "1785578902552363255-11105". Stable; use this as the key. */
  id: string
  camera_id: string
  /** One of the 16 types listed by GET /api/notifications. */
  event_type: string
  at_ns: Nanoseconds
  detail: string
}

export interface RecorderDelivery {
  alert_id: string
  camera_id: string
  event_type: string
  /** Observed: "dashboard". GET /api/notifications also names "webhook". */
  channel: string
  /** Observed: "delivered". The recorder's bundle also handles "failed" and "unrouted". */
  status: 'delivered' | 'failed' | 'unrouted' | (string & {})
  attempts: number
  at_ns: Nanoseconds
  /** Present only on a failure. */
  error?: string
  /** Set when delivery ignored the operator's rule because the event type is locked. */
  overridden?: boolean
}

export interface RecorderAlertsResponse {
  /** Newest first. Capped by the recorder (500 in the live instance) — a rolling window, not the full history. */
  alerts: RecorderAlert[]
  deliveries: RecorderDelivery[]
  dropped: {
    total: number
    /**
     * Alerts about lost observation or lost privacy enforcement that were
     * discarded before delivery was attempted. These may never be suppressed,
     * so a non-zero value is a system fault, not a quiet night.
     */
    welfare_locked: number
  }
  failed_deliveries: number
  unrouted_deliveries: number
  note: string
}

/* ------------------------------------------------------------------ *
 * GET /api/models
 * ------------------------------------------------------------------ */

export interface RecorderModelStatus {
  /** Observed: "off". The recorder's settings form also offers "interval", "event", "both". */
  cadence: 'off' | 'interval' | 'event' | 'both' | (string & {})
  interval_sec: number
  /** Whether the configured model can actually accept images. */
  vision: boolean
  /** Whether frames leave this machine. */
  offsite: boolean
  /** False means nothing is reading the footage. Say so, loudly. */
  ready: boolean
  /** Machine token, e.g. "no_model_configured". */
  reason: string
  /** Sentence explaining `reason` in the recorder's own words. */
  detail: string
}

export interface RecorderLocalModel {
  name: string
  size_bytes: number
  vision: boolean
  family: string
}

export interface RecorderFrontierModel {
  id: string
  label: string
  input_per_mtok: number
  output_per_mtok: number
  note: string
}

export interface RecorderModelTier {
  /** "detection", "pose", "tracking", "action", "face", "vlm", "welfare". */
  tier: string
  /** The model the design specifies for this tier. */
  model: string
  /** Observed: "NOT_BUILT", "PARTIAL". */
  status: 'NOT_BUILT' | 'PARTIAL' | 'BUILT' | (string & {})
  detail: string
}

export interface RecorderCatalogEntry {
  local_id: string
  label: string
  /** Parameter count as text, e.g. "3.8B". Empty for undeclared models. */
  parameters: string
  download_bytes: number
  /** Long-form measured notes, including recorded prompt violations. Never truncate this silently. */
  runtime_note: string
  vision: boolean
  /** Frontier model id an operator would reach for instead. Empty when none is declared. */
  online_equivalent: string
  why_this_pairing: string
  not_equivalent_because: string
  online_equivalent_in_force: string
  /** Observed: "declared", "none". */
  online_equivalent_source: string
  /** Whether this build declares the model, as opposed to merely finding it installed. */
  declared: boolean
  installed: boolean
}

export interface RecorderPerception {
  remote: boolean
  reachable: boolean
  offsite: boolean
  note: string
}

export interface RecorderModelsResponse {
  status: RecorderModelStatus
  local: RecorderLocalModel[]
  frontier: RecorderFrontierModel[]
  /** False means no API key is present, so the offsite path cannot run at all. */
  frontier_credentialed: boolean
  frontier_note: string
  /**
   * Present on the live instance, but optional on purpose.
   *
   * The recorder's own shipped console guards `tiers` before reading it
   * (`r && r.tiers && ...`) and never reads `catalog`, `perception` or
   * `selected` at all — which says these blocks are build-dependent, not
   * guaranteed. Declaring them required would turn a recorder upgrade that
   * drops one into a white screen on a console someone is meant to be
   * watching cameras through.
   */
  catalog?: RecorderCatalogEntry[]
  catalog_note?: string
  perception?: RecorderPerception
  selected?: {
    mode: string
    cadence: string
    interval_sec: number
  }
  tiers?: RecorderModelTier[]
  /**
   * Neither is present on the live instance, but the recorder's own shipped
   * bundle reads both, so they appear when the configured model is unusable or
   * no local runtime answers. Optional here for exactly that reason.
   */
  config_error?: string
  local_error?: string
}

/* ------------------------------------------------------------------ *
 * GET /api/masks/{camera_id}
 * PUT /api/masks/{camera_id}                (save — validates, then persists)
 * POST /api/masks/{camera_id}/validate       (dry run — never persists)
 * GET /api/masks/{camera_id}/calibration-frame
 *
 * All four verified live 2026-08-01 against `room_4b`, `corridor_1`, `room_2a`
 * and `dayroom_1`. Every camera on the reference instance is `recording`, so
 * `calibration_available: true` and a real calibration-frame image were never
 * observed live; that branch is built from the same documented shape, not a
 * second live capture — see `MasksPage.tsx` for how it is handled.
 * ------------------------------------------------------------------ */

/** `[x, y]` in the camera's CONFIGURED frame pixels — see `RecorderMaskResponse.note`. */
export type MaskPolygonPoint = [number, number]

export interface RecorderMaskRegion {
  region_id: string
  polygon: MaskPolygonPoint[]
}

export interface RecorderMaskInventoryRegion {
  region_id: string
  area_px: number
  area_pct: number
  /** `[x_min, y_min, x_max, y_max]`. */
  bbox: [number, number, number, number]
}

export interface RecorderMaskInventory {
  path: string
  regions: RecorderMaskInventoryRegion[]
  total_px: number
  total_pct: number
  frame_px: number
  /** Whether any two regions cover overlapping pixels. Not itself a fault — a
   * pixel counted by two regions is still masked exactly once. */
  may_overlap: boolean
  note: string
}

export interface RecorderMaskResponse {
  camera_id: string
  /** Observed: "room", "common_area" — same vocabulary as `RecorderCamera.mode`. */
  mode: string
  frame_width: number
  frame_height: number
  /** Empty string when no mask file is configured — not null. */
  mask_path: string
  regions: RecorderMaskRegion[]
  editable: boolean
  recording: boolean
  /**
   * False on every camera observed live, because every camera on the
   * reference instance is recording. When false, the recorder will not serve
   * `GET .../calibration-frame` (404) — this console must never substitute the
   * live snapshot for it; the snapshot is already masked, so drawing against
   * it would either mislead the operator about what they are covering or
   * expose exactly what the mask exists to hide.
   */
  calibration_available: boolean
  /** Present (and only meaningful) when `calibration_available` is false: the
   * recorder's own words for why, and what to do about it. Render verbatim. */
  calibration_refusal?: string
  /** Present only when `regions.length > 0` — verified absent on an empty set. */
  inventory?: RecorderMaskInventory
  /** Explains the frame-pixel coordinate space; see `MaskPolygonPoint`. */
  note: string
}

/** Per-region outcome from a validate or save call. */
export interface RecorderMaskRegionResult {
  region_id: string
  valid: boolean
  /** Present when this region failed — the recorder's own polygon rule that
   * rejected it. Never re-derived or second-guessed client-side. */
  error?: string
}

/**
 * Body of both `POST .../validate` (always dry-run, HTTP 200 either way) and
 * `PUT /api/masks/{camera_id}` (HTTP 200 and persisted when `valid`, HTTP 422
 * and left untouched on disk when not — verified live: an invalid `PUT` did
 * not change the camera's saved regions).
 */
export interface RecorderMaskValidation {
  valid: boolean
  /** Present when invalid — a top-level summary, usually the first failing region's own error. */
  error?: string
  regions: RecorderMaskRegionResult[]
  /** Present only when `valid` is true. */
  inventory?: RecorderMaskInventory
}

 * GET /api/capabilities
 * ------------------------------------------------------------------ */

export interface RecorderCapability {
  key: string
  label: string
  available: boolean
  /** Present on most entries; explains what the capability can and cannot do. */
  reason?: string
}

export interface RecorderCapabilitiesResponse {
  capabilities: RecorderCapability[]
}

/* ------------------------------------------------------------------ *
 * GET /api/notifications
 * ------------------------------------------------------------------ */

export interface RecorderNotificationChannel {
  name: string
  configured: boolean
  detail: string
}

export interface RecorderNotificationEventType {
  type: string
  description: string
  /** A locked type may never be suppressed by an operator rule. Render it as locked. */
  locked: boolean
  default_enabled: boolean
}

/**
 * The currently-applied routing for one event type. Distinct from
 * `RecorderNotificationEventType.default_enabled` — that is what ships by
 * default, this is what is in force now. VERIFIED 2026-08-01: one rule per
 * event type, 16 total, `channels` always `["dashboard"]` on the live
 * instance (the only configured channel).
 */
export interface RecorderNotificationRule {
  event_type: string
  channels: string[]
  enabled: boolean
}

export interface RecorderNotificationsResponse {
  channel_config: Record<string, unknown>
  channels: RecorderNotificationChannel[]
  delivery_note: string
  delivery_wired: boolean
  event_types: RecorderNotificationEventType[]
  /** VERIFIED 2026-08-01: 16 entries, one per event type. */
  rules: RecorderNotificationRule[]
  /** The event `type` values that are locked — the same set as `event_types[].locked`, restated. */
  locked: string[]
  /**
   * "Locked event types report loss of observation or loss of protection ...
   * and cannot be disabled, left without channels, or routed only to a
   * channel that cannot deliver." The invariant this whole page exists to
   * make legible.
   */
  note: string
}

/* ------------------------------------------------------------------ *
 * GET /api/settings
 * ------------------------------------------------------------------ */

export interface RecorderSettingsResponse {
  note: string
  online_warning: string
  settings: {
    inference_mode: 'local' | 'online' | (string & {})
    online_acknowledged: boolean
  }
  /** False means the setting is recorded but nothing acts on it. */
  wired: boolean
}

/* ------------------------------------------------------------------ *
 * GET /api/storage/untracked
 * ------------------------------------------------------------------ */

export interface RecorderUntrackedFile {
  path: string
  size_bytes: number
  modified_ns: Nanoseconds
}

export interface RecorderUntrackedCamera {
  camera_id: string
  dir: string
  count: number
  total_bytes: number
  /** `null` when the directory is empty; a partial list when `truncated`. */
  sample: RecorderUntrackedFile[] | null
  truncated: boolean
}

export interface RecorderUntrackedResponse {
  cameras: RecorderUntrackedCamera[]
  note: string
  total_bytes: number
  total_count: number
}
