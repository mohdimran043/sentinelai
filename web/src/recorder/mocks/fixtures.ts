// oxlint-disable no-loss-of-precision -- The recorder's nanosecond timestamps
// genuinely exceed Number.MAX_SAFE_INTEGER, and JSON.parse rounds them exactly
// the way these literals do. Writing a "safe" number here would make the
// fixture stop matching what the live API delivers, which is the one thing a
// fixture must never do. See recorder/recorder.types.ts.
import type {
  RecorderAlert,
  RecorderAlertsResponse,
  RecorderCamerasResponse,
  RecorderCapabilitiesResponse,
  RecorderCoverageReport,
  RecorderJournalDayResponse,
  RecorderJournalResponse,
  RecorderModelsResponse,
  RecorderSettingsResponse,
  RecorderStatusResponse,
  RecorderUntrackedResponse,
} from '@/recorder/recorder.types'

/**
 * Fixtures for the recorder API.
 *
 * `REAL_*` values are verbatim excerpts of live 200 responses captured from the
 * running appliance on 2026-08-01 — not invented shapes. They exist so a test
 * that passes here would also pass against the real recorder.
 *
 * The live alert window happens to contain exactly one camera and two event
 * types today, which is not enough to prove a filter works. `makeAlerts` builds
 * a wider set with the same field shapes for the filtering, ordering and paging
 * tests; the real excerpt is used wherever the point is "we parse what the
 * recorder actually sends".
 */

export const REAL_MODELS: RecorderModelsResponse = {
  "status": {
    "cadence": "off",
    "interval_sec": 30,
    "vision": false,
    "offsite": false,
    "ready": false,
    "reason": "no_model_configured",
    "detail": "no model is configured, so no footage is being described"
  },
  "local": [
    {
      "name": "qwen2.5vl:3b",
      "size_bytes": 3200627168,
      "vision": true,
      "family": "qwen25vl"
    },
    {
      "name": "qwen2.5vl:7b",
      "size_bytes": 5969245856,
      "vision": true,
      "family": "qwen25vl"
    },
    {
      "name": "gemma2:latest",
      "size_bytes": 5443152417,
      "vision": false,
      "family": "gemma2"
    }
  ],
  "frontier": [
    {
      "id": "claude-opus-5",
      "label": "Claude Opus 5",
      "input_per_mtok": 5,
      "output_per_mtok": 25,
      "note": "default; strongest scene reading"
    },
    {
      "id": "claude-sonnet-5",
      "label": "Claude Sonnet 5",
      "input_per_mtok": 3,
      "output_per_mtok": 15,
      "note": "cheaper per frame, near-Opus on description"
    },
    {
      "id": "claude-haiku-4-5",
      "label": "Claude Haiku 4.5",
      "input_per_mtok": 1,
      "output_per_mtok": 5,
      "note": "cheapest; use when sampling many cameras often"
    }
  ],
  "frontier_credentialed": false,
  "frontier_note": "Frontier models are vision-capable, and every frame sent to one leaves this machine. master-prompt.txt lists offsite inference under Skip; this build enables it because the operator asked for it on 2026-07-31, gated on a recorded acknowledgement. The API key is read from the recorder's environment and is never accepted or displayed here.",
  "catalog": [
    {
      "local_id": "qwen2.5vl:3b",
      "label": "Qwen2.5-VL 3B (Q4_K_M)",
      "parameters": "3.8B",
      "download_bytes": 3200627168,
      "runtime_note": "Half the download of the 7B and the practical option on a small GPU. Measured through sentinel-perception on this box against a real recorded segment: 87s for the first description including model load, then 6.5s warm. That is the only figure in this catalog that makes the periodic cadence usable on this hardware at all. The masked-frame adversarial probe the 7B was put through has NOT been run against it, so the mask prohibition is unverified for this model.",
      "vision": true,
      "online_equivalent": "claude-haiku-4-5",
      "why_this_pairing": "Both are the cheap option in their tier — the one you choose when sampling many cameras often, and the one whose per-description cost is low enough not to shape the cadence.",
      "not_equivalent_because": "This is the weakest pairing in the catalog and should be read as 'the budget choice on each side', not as comparable output. A 3B model miscounts people, misreads posture and abandons dark frames far more often than any frontier model, and in this system a wrong description is not a worse caption — it is a wrong observation filed against a real person. Neither is the mask prohibition verified here. Do not treat a 3B description as evidence of anything beyond what a reviewer can confirm from the segment itself, which every description links to by hash. MEASURED PROMPT VIOLATION (2026-07-31): handed a real recorded frame containing people, under the unmodified ScenePrompt, this model wrote \"A man is seated... He is wearing... there is a child standing near a table\" on three runs out of three. ScenePrompt forbids guessing the age or sex of anyone. The 7B model on the SAME frame under the SAME prompt wrote \"Two individuals are visible; one is seated on a couch, and the other is bending over near a table\" — so this is a weakness of this model, not of the prompt, and the difference between the two rows of this catalog is a real person's sex and age being asserted in a record their family and an inquiry may read. Treat the 3B as unsuitable for a custodial record until that is re-measured.",
      "online_equivalent_in_force": "claude-haiku-4-5",
      "online_equivalent_source": "declared",
      "declared": true,
      "installed": true
    },
    {
      "local_id": "qwen2.5vl:7b",
      "label": "Qwen2.5-VL 7B (Q4_K_M)",
      "parameters": "8.3B",
      "download_bytes": 5969245856,
      "runtime_note": "The model the master prompt budgets for (line 19, ~7GB on a 24GB 4090). Measured on the 8GB RTX 4060 laptop this was built on it does NOT fit beside the recorder's CUDA context: ollama reported size_vram 0 and ran it on the CPU at 124 seconds per description of one 720p frame — re-measured through sentinel-perception at 113s on a real recorded segment. It is the only model here observed to hold the scene prompt's identity prohibition on a frame containing people; see the 3B entry for what the alternative did.",
      "vision": true,
      "online_equivalent": "claude-sonnet-5",
      "why_this_pairing": "The closest online counterpart in intended role: the model you reach for when a description has to be good enough to read, without paying top-tier prices per frame per camera.",
      "not_equivalent_because": "Different models from different vendors — the sentences differ and neither is a corrected version of the other. Sonnet is substantially stronger on dark, blurred and crowded frames, which is most custodial footage at night. And every frame described this way LEAVES THIS MACHINE, which the local model never does; that difference is recorded permanently on each description.",
      "online_equivalent_in_force": "claude-sonnet-5",
      "online_equivalent_source": "declared",
      "declared": true,
      "installed": true
    },
    {
      "local_id": "gemma2:latest",
      "label": "gemma2:latest",
      "parameters": "",
      "download_bytes": 5443152417,
      "runtime_note": "Installed on this machine but not part of this build's catalog. Nothing here has been measured against it and its behaviour under the scene prompt's prohibitions — in particular the one that stops it guessing what a privacy mask covers — is unknown.",
      "vision": false,
      "online_equivalent": "",
      "why_this_pairing": "",
      "not_equivalent_because": "No pairing has been declared, so nothing is being claimed about an online counterpart either.",
      "online_equivalent_in_force": "",
      "online_equivalent_source": "none",
      "declared": false,
      "installed": true
    }
  ],
  "catalog_note": "An online equivalent is the model an operator would reach for instead, not a model that produces the same answer. They are different models from different vendors: the sentences differ, the quality differs — a 3B local model and a frontier model are not interchangeable — and only one of them keeps the frame on this machine. Each row states where its own pairing breaks down.",
  "perception": {
    "remote": false,
    "reachable": false,
    "offsite": false,
    "note": "Inference runs inside the recorder process. Frames are read from recorded segments on this machine and do not cross a process boundary unless an offsite model is selected."
  },
  "selected": {
    "mode": "local",
    "cadence": "off",
    "interval_sec": 30
  },
  "tiers": [
    {
      "tier": "detection",
      "model": "YOLOv11m",
      "status": "NOT_BUILT",
      "detail": "No detector runs. Nothing counts people, and no event can be raised."
    },
    {
      "tier": "pose",
      "model": "RTMPose-m",
      "status": "NOT_BUILT",
      "detail": "No pose estimation. Posture, falls and stillness are not derived."
    },
    {
      "tier": "tracking",
      "model": "ByteTrack",
      "status": "NOT_BUILT",
      "detail": "No tracking. Identity across frames is not maintained."
    },
    {
      "tier": "action",
      "model": "VideoMAEv2-S",
      "status": "NOT_BUILT",
      "detail": "No action recognition. Nothing classifies what is happening."
    },
    {
      "tier": "face",
      "model": "InsightFace",
      "status": "NOT_BUILT",
      "detail": "No face recognition. Disabled in common areas by policy regardless."
    },
    {
      "tier": "vlm",
      "model": "Qwen2.5-VL-7B-AWQ (spec) / configured model (built)",
      "status": "PARTIAL",
      "detail": "Scene description runs, but on a schedule or on demand rather than event-triggered, because no detector exists to raise an event. It describes one still frame and asserts nothing about welfare."
    },
    {
      "tier": "welfare",
      "model": "—",
      "status": "NOT_BUILT",
      "detail": "No welfare inference. Descriptions are explicitly forbidden from implying distress, and no alert is derived from them."
    }
  ]
}

export const REAL_ALERTS: RecorderAlertsResponse = {
  "alerts": [
    {
      "id": "1785578902552363255-11105",
      "camera_id": "room_4b",
      "event_type": "UNDERLIT",
      "at_ns": 1785578902544883331,
      "detail": "luma_mean 0.063 below floor 0.08"
    },
    {
      "id": "1785578897595698657-11104",
      "camera_id": "room_4b",
      "event_type": "OBSTRUCTED",
      "at_ns": 1785578897580927791,
      "detail": "frame_variance 0.0000 below 5% of rolling median 0.1608"
    },
    {
      "id": "1785578892635562524-11103",
      "camera_id": "room_4b",
      "event_type": "UNDERLIT",
      "at_ns": 1785578892620955927,
      "detail": "luma_mean 0.063 below floor 0.08"
    },
    {
      "id": "1785578887664473845-11102",
      "camera_id": "room_4b",
      "event_type": "OBSTRUCTED",
      "at_ns": 1785578887656754979,
      "detail": "frame_variance 0.0000 below 5% of rolling median 0.1608"
    }
  ],
  "deliveries": [
    {
      "alert_id": "1785578902552363255-11105",
      "camera_id": "room_4b",
      "event_type": "UNDERLIT",
      "channel": "dashboard",
      "status": "delivered",
      "attempts": 1,
      "at_ns": 1785578902552376281
    },
    {
      "alert_id": "1785578897595698657-11104",
      "camera_id": "room_4b",
      "event_type": "OBSTRUCTED",
      "channel": "dashboard",
      "status": "delivered",
      "attempts": 1,
      "at_ns": 1785578897595708396
    },
    {
      "alert_id": "1785578892635562524-11103",
      "camera_id": "room_4b",
      "event_type": "UNDERLIT",
      "channel": "dashboard",
      "status": "delivered",
      "attempts": 1,
      "at_ns": 1785578892635573782
    },
    {
      "alert_id": "1785578887664473845-11102",
      "camera_id": "room_4b",
      "event_type": "OBSTRUCTED",
      "channel": "dashboard",
      "status": "delivered",
      "attempts": 1,
      "at_ns": 1785578887664492058
    }
  ],
  "dropped": {
    "total": 0,
    "welfare_locked": 0
  },
  "failed_deliveries": 0,
  "unrouted_deliveries": 0,
  "note": "Alerts are facts about cameras — loss of observation, loss of privacy enforcement, integrity changes. None of them is a statement about a person, and an empty feed is not confirmation that anyone is well."
}

/** Nanosecond epoch used as the base for generated alerts: 2026-08-01 00:00:00 UTC. */
export const FIXTURE_BASE_NS = 1785_542_400_000_000_000

export interface MakeAlertOptions {
  cameraId: string
  eventType: string
  /** Whole seconds after `FIXTURE_BASE_NS`. */
  offsetSeconds: number
  detail?: string
}

export function makeAlert({
  cameraId,
  eventType,
  offsetSeconds,
  detail,
}: MakeAlertOptions): RecorderAlert {
  const atNs = FIXTURE_BASE_NS + offsetSeconds * 1_000_000_000
  return {
    id: `${atNs}-${offsetSeconds}`,
    camera_id: cameraId,
    event_type: eventType,
    at_ns: atNs,
    detail: detail ?? `${eventType.toLowerCase()} on ${cameraId} at +${offsetSeconds}s`,
  }
}

/** Wraps alerts in the envelope the recorder actually returns, newest first. */
export function makeAlertsResponse(
  alerts: RecorderAlert[],
  overrides: Partial<Omit<RecorderAlertsResponse, 'alerts'>> = {},
): RecorderAlertsResponse {
  return {
    alerts: [...alerts].sort((a, b) => b.at_ns - a.at_ns),
    deliveries: [],
    dropped: { total: 0, welfare_locked: 0 },
    failed_deliveries: 0,
    unrouted_deliveries: 0,
    note: REAL_ALERTS.note,
    ...overrides,
  }
}

/* ------------------------------------------------------------------ *
 * GET /api/settings — captured live 2026-08-01
 * ------------------------------------------------------------------ */

export const REAL_SETTINGS: RecorderSettingsResponse = {
  note: 'No model is running, so nothing reads the footage. The setting is recorded so the choice is explicit and auditable.',
  online_warning:
    'master-prompt.txt lists offsite model APIs under Skip ("send data offsite = compliance problem"). In a custodial setting, sending frames of detained people to a third party is a legal exposure.',
  settings: {
    inference_mode: 'local',
    online_acknowledged: false,
  },
  wired: false,
}

/* ------------------------------------------------------------------ *
 * GET /api/storage/untracked — captured live 2026-08-01
 * ------------------------------------------------------------------ */

export const REAL_STORAGE: RecorderUntrackedResponse = {
  cameras: [
    {
      camera_id: 'corridor_1',
      dir: '/var/tmp/sentinel-wall/preroll/corridor_1',
      count: 0,
      total_bytes: 0,
      sample: null,
      truncated: false,
    },
    {
      camera_id: 'dayroom_1',
      dir: '/var/tmp/sentinel-wall/preroll/dayroom_1',
      count: 0,
      total_bytes: 0,
      sample: null,
      truncated: false,
    },
    {
      camera_id: 'room_2a',
      dir: '/var/tmp/sentinel-wall/preroll/room_2a',
      count: 0,
      total_bytes: 0,
      sample: null,
      truncated: false,
    },
    {
      camera_id: 'room_4b',
      dir: '/var/tmp/sentinel-wall/preroll/room_4b',
      count: 3,
      total_bytes: 84,
      sample: [
        {
          path: '/var/tmp/sentinel-wall/preroll/room_4b/room_4b_1785502196892950204.mp4',
          size_bytes: 28,
          modified_ns: 1785502196899459663,
        },
        {
          path: '/var/tmp/sentinel-wall/preroll/room_4b/room_4b_1785502277303285389.mp4',
          size_bytes: 28,
          modified_ns: 1785502277307621050,
        },
        {
          path: '/var/tmp/sentinel-wall/preroll/room_4b/room_4b_1785580404648961122.mp4',
          size_bytes: 28,
          modified_ns: 1785580405161197947,
        },
      ],
      truncated: false,
    },
  ],
  note: 'Footage on disk that the segment index does not account for. It is never deleted automatically: an un-indexed file is one the system has lost its record of, which is exactly when its contents are least safe to assume are worthless. Reclaiming the space is a deliberate act.',
  total_bytes: 84,
  total_count: 3,
}

/* ------------------------------------------------------------------ *
 * GET /api/capabilities — captured live 2026-08-01
 * ------------------------------------------------------------------ */

export const REAL_CAPABILITIES: RecorderCapabilitiesResponse = {
  capabilities: [
    { key: 'coverage', label: 'Coverage & gaps', available: true },
    { key: 'integrity', label: 'Camera integrity', available: true },
    { key: 'segments', label: 'Recorded video', available: true },
    { key: 'masked_scalars', label: 'Masked-region presence/motion', available: true },
    {
      key: 'mask_editor',
      label: 'Draw privacy masks',
      available: true,
      reason:
        "Draws mask polygons over the camera's own frame and validates them server-side, so the console never disagrees with the daemon about what is maskable. It cannot draw ignored_zones: those are not masks, their pixels are still recorded, and one canvas for both would invite an operator to believe an area was blanked when it was merely ignored.",
    },
    {
      key: 'notifications',
      label: 'Alert delivery',
      available: true,
      reason:
        'Delivers the facts this slice records — integrity transitions and loss of observation. It cannot notify anyone about a person, a fall or distress, because nothing detects those.',
    },
    {
      key: 'detection',
      label: 'Person detection & pose',
      available: false,
      reason:
        'No perception tier is connected. The CUDA IPC ring is published and tested, but nothing consumes it yet (slice 3).',
    },
    {
      key: 'welfare',
      label: 'Welfare & distress detection',
      available: false,
      reason:
        'Not implemented (slice 6). Nothing detects stillness, collapse, distress or self-harm, so no welfare judgement of any kind is available. Absence of detection is not confirmation of wellbeing.',
    },
    {
      key: 'events',
      label: 'Incidents & severity',
      available: false,
      reason:
        'Not implemented (slice 7). Slice 1 emits facts only; severity needs occupancy and elevated_watch context that does not exist yet.',
    },
    {
      key: 'risk',
      label: 'Pre-incident risk score',
      available: false,
      reason: 'Not implemented (slice 9).',
    },
    {
      key: 'narrative',
      label: 'Written narrative / the Book',
      available: false,
      reason:
        'Not implemented (slice 4). No narrative is generated, and none will be invented for periods that were not analysed.',
    },
    {
      key: 'inference',
      label: 'Model inference (local or online)',
      available: false,
      reason:
        'No model is loaded. The mode setting is recorded for when a perception tier is wired in, and changes nothing today.',
    },
    {
      key: 'per_camera_context',
      label: 'Per-camera context',
      available: false,
      reason:
        'elevated_watch, capacity, audio_enabled, face_recognition_enabled and ignored_zones are validated, stored and reported — and acted on by NOTHING in this slice. Setting elevated_watch does not cause a camera to be watched more closely, because nothing yet watches for anything. The schema lands early so a deployment is not configured against one shape and then silently reinterpreted when welfare logic arrives.',
    },
  ],
}

/* ------------------------------------------------------------------ *
 * GET /api/cameras — captured live 2026-08-01, 4 cameras
 * ------------------------------------------------------------------ */

export const REAL_CAMERAS: RecorderCamerasResponse = {
  cameras: [
    {
      id: 'room_4b',
      name: 'Room 4B',
      mode: 'room',
      space_type: 'room',
      source: 'v4l2:/dev/video0',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '/var/tmp/sentinel-wall/masks/room_4b.json',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
    {
      id: 'corridor_1',
      name: 'Corridor 1',
      mode: 'common_area',
      space_type: 'corridor',
      source: 'rtsp://10.0.0.99:554/nothing-here',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
    {
      id: 'room_2a',
      name: 'Room 2A',
      mode: 'room',
      space_type: 'room',
      source: 'rtsp://10.0.0.98:554/nothing-here',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
    {
      id: 'dayroom_1',
      name: 'Dayroom 1',
      mode: 'common_area',
      space_type: 'dayroom',
      source: 'rtsp://10.0.0.97:554/nothing-here',
      width: 1280,
      height: 720,
      fps: 15,
      mask_path: '',
      preroll_seconds: 300,
      elevated_watch: false,
      capacity: 0,
      audio_enabled: false,
      face_recognition_enabled: false,
      ignored_zones: [],
    },
  ],
  context_note:
    'elevated_watch, capacity, audio_enabled, face_recognition_enabled and ignored_zones are stored and validated. Nothing in this slice reads them. Setting elevated_watch does not cause a camera to be watched more closely.',
}

/* ------------------------------------------------------------------ *
 * GET /api/status — captured live 2026-08-01, same instant as REAL_CAMERAS.
 * One real camera (room_4b) reads UNDERLIT rather than LOST at this capture;
 * the brief's "one real camera is currently LOST" is reflected in the other
 * three, captured moments apart as the corridor/room_2a/dayroom feeds (which
 * have no real source behind them) aged past the outage threshold.
 * ------------------------------------------------------------------ */

export const REAL_STATUS: RecorderStatusResponse = {
  cameras: [
    {
      camera_id: 'corridor_1',
      mode: 'common_area',
      integrity_state: 'LOST',
      running: true,
      failed: false,
      restarts: 0,
      last_frame_age_ms: -1,
      frames_dropped: 0,
    },
    {
      camera_id: 'dayroom_1',
      mode: 'common_area',
      integrity_state: 'LOST',
      running: true,
      failed: false,
      restarts: 0,
      last_frame_age_ms: -1,
      frames_dropped: 0,
    },
    {
      camera_id: 'room_2a',
      mode: 'room',
      integrity_state: 'LOST',
      running: true,
      failed: false,
      restarts: 0,
      last_frame_age_ms: -1,
      frames_dropped: 0,
    },
    {
      camera_id: 'room_4b',
      mode: 'room',
      integrity_state: 'UNDERLIT',
      running: true,
      failed: false,
      restarts: 0,
      last_frame_age_ms: 0,
      frames_dropped: 98893490696,
    },
  ],
  server_time_ns: 1785580431295388577,
}

/* ------------------------------------------------------------------ *
 * GET /api/journal/{camera}, /api/journal/{camera}/{date} and
 * GET /api/report?camera=&date= — captured live 2026-08-01 against room_4b.
 *
 * `intervals` and `retrieval_holes` are excerpted, not complete: the live
 * documents carry 364-594 intervals and up to 91 retrieval holes for one
 * camera-day. Every entry kept here is a real one, copied verbatim — trimmed
 * to a variety-preserving sample (every `integrity_state` and gap `cause`
 * that appeared at least once survives the trim) because the point of these
 * fixtures is shape and value fidelity, not volume fidelity; nothing in this
 * console renders per-interval rows anyway (see `reportSummary.ts`).
 * ------------------------------------------------------------------ */

export const REAL_JOURNAL_ROOM_4B: RecorderJournalResponse = {
  camera_id: 'room_4b',
  days: [
    { date: '2026-07-31', status: 'sealed', revisions: 2, sealed_at_ns: 1785540246958127621 },
    { date: '2026-07-30', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152908981707 },
    { date: '2026-07-29', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152923271550 },
    { date: '2026-07-28', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152929242398 },
    { date: '2026-07-27', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152935529956 },
    { date: '2026-07-26', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152941560553 },
    { date: '2026-07-25', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152947736645 },
    { date: '2026-07-24', status: 'sealed', revisions: 1, sealed_at_ns: 1785502152953897219 },
  ],
  note: 'A sealed day was computed while the footage behind it still existed and is never recomputed. Asking the live report for the same day after retention has run gives a smaller answer about the same day.',
}

/** The `report` embedded in room_4b/2026-07-31's revision 2 (sealed). */
const REAL_REPORT_2026_07_31: RecorderCoverageReport = {
  blind_spots: {
    path: '/var/tmp/sentinel-wall/masks/room_4b.json',
    regions: [
      { region_id: 'toilet', area_px: 78200, area_pct: 8.485243055555555, bbox: [80, 90, 420, 320] },
      {
        region_id: 'washbasin',
        area_px: 56000,
        area_pct: 6.076388888888889,
        bbox: [900, 400, 1180, 600],
      },
    ],
    total_px: 134200,
    total_pct: 14.561631944444445,
    frame_px: 921600,
    may_overlap: false,
    note: 'Masked regions are never observed and never recorded. Coverage percentages elsewhere describe only the unmasked part of the frame.',
  },
  camera_id: 'room_4b',
  coverage: {
    expected_seconds: 34524.09797090699,
    gap_seconds: 12761.046,
    intervals_with_records: 364,
    recorded_pct: 64.17862852273936,
    recorded_seconds: 22157.092587574993,
    unexplained_shortfall_secs: -394.0406166680059,
  },
  date: '2026-07-31',
  integrity_time: {
    DEGRADED_QUALITY: 425.351729525,
    FOV_SHIFTED: 360.002298424,
    LOST: 60.001412034,
    OBSTRUCTED: 10439.997581901993,
    UNDERLIT: 23238.744949022002,
  },
  intervals: [
    {
      start_ns: 1785502152907557839,
      expected_seconds: 60.001412034,
      recorded_seconds: 41.08136856,
      integrity_state: 'LOST',
      gaps: [{ start_ns: 1785502197000060702, duration_ms: 15908, cause: 'LOST' }],
    },
    {
      start_ns: 1785502212908969873,
      expected_seconds: 60.000320173,
      recorded_seconds: 56.725630217,
      integrity_state: 'DEGRADED_QUALITY',
      gaps: [{ start_ns: 1785502212908969873, duration_ms: 2521, cause: 'LOST' }],
    },
    {
      start_ns: 1785502272909290046,
      expected_seconds: 5.350849348,
      recorded_seconds: 5.010755344,
      integrity_state: 'DEGRADED_QUALITY',
      gaps: [],
    },
    {
      start_ns: 1785502362391993458,
      expected_seconds: 60.000032891,
      recorded_seconds: 59.127872027,
      integrity_state: 'FOV_SHIFTED',
      gaps: [],
    },
    {
      start_ns: 1785503022392518702,
      expected_seconds: 59.999929933,
      recorded_seconds: 59.702867862,
      integrity_state: 'UNDERLIT',
      gaps: [],
    },
    {
      start_ns: 1785503382393825067,
      expected_seconds: 59.998982814,
      recorded_seconds: 59.506092168,
      integrity_state: 'OBSTRUCTED',
      gaps: [],
    },
  ],
  note: 'Coverage, integrity and segments are recorded. Welfare, incidents, severity, risk and narrative are NOT available — see /api/capabilities. Absence of detection is not confirmation of wellbeing.',
  retrieval_holes: [
    { start_ns: 1785445200000000000, end_ns: 1785531600000000000, cause: 'expired_by_retention' },
  ],
  segments: { count: 0, seconds: 0 },
}

export const REAL_JOURNAL_DAY_SEALED: RecorderJournalDayResponse = {
  camera_id: 'room_4b',
  date: '2026-07-31',
  sealed: true,
  revision: {
    camera_id: 'room_4b',
    date: '2026-07-31',
    revision: 2,
    status: 'sealed',
    supersedes: 1,
    reason:
      'day complete; sealed from the coverage journal and segment index while the evidence behind it still existed',
    created_at_ns: 1785540246958127621,
    report: REAL_REPORT_2026_07_31,
    report_sha256: 'b326bb2ffc3de165018964e33956d4019b122c8f0208185961e532654ceefdc3',
  },
}

/**
 * `GET /api/journal/room_4b/2020-01-01` — VERIFIED: a date with no written
 * chapter is 200, not 404, and carries no `revision` key at all.
 */
export const REAL_JOURNAL_DAY_NO_CHAPTER: RecorderJournalDayResponse = {
  camera_id: 'room_4b',
  date: '2020-01-01',
  detail:
    'No chapter has been written for this day. It is either still in progress, or predates this journal. The live report can still assemble it from whatever coverage records and segments remain.',
  sealed: false,
}

/**
 * `GET /api/journal/room_4b/2026-07-31?history=1` — VERIFIED: `history` sits
 * alongside `revision` (the latest), oldest revision first. Revision 1's own
 * `report` body is never rendered by the history view (only revision/status/
 * reason/created_at/sha256 are), so it is not re-captured separately here.
 */
export const REAL_JOURNAL_DAY_HISTORY: RecorderJournalDayResponse = {
  ...REAL_JOURNAL_DAY_SEALED,
  history: [
    {
      camera_id: 'room_4b',
      date: '2026-07-31',
      revision: 1,
      status: 'provisional',
      reason: 'written on request while the day was still in progress; expect a sealed revision once the day is complete',
      created_at_ns: 1785502572964166316,
      report: REAL_REPORT_2026_07_31,
      report_sha256: '5f66c041efed54c98663a5d8991aa3f64d840f65dfc1165473ebfaffdfeb4862',
    },
    REAL_JOURNAL_DAY_SEALED.revision!,
  ],
}

/** `GET /api/report?camera=room_4b&date=2026-08-01` — the live, standalone report. */
export const REAL_REPORT: RecorderCoverageReport = {
  blind_spots: {
    path: '/var/tmp/sentinel-wall/masks/room_4b.json',
    regions: [
      { region_id: 'toilet', area_px: 78200, area_pct: 8.485243055555555, bbox: [80, 90, 420, 320] },
      {
        region_id: 'washbasin',
        area_px: 56000,
        area_pct: 6.076388888888889,
        bbox: [900, 400, 1180, 600],
      },
    ],
    total_px: 134200,
    total_pct: 14.561631944444445,
    frame_px: 921600,
    may_overlap: false,
    note: 'Masked regions are never observed and never recorded. Coverage percentages elsewhere describe only the unmasked part of the frame.',
  },
  camera_id: 'room_4b',
  coverage: {
    expected_seconds: 44029.86481698496,
    gap_seconds: 8395.891,
    intervals_with_records: 594,
    recorded_pct: 79.93334421500266,
    recorded_seconds: 35194.54340156094,
    unexplained_shortfall_secs: 439.4304154240199,
  },
  date: '2026-08-01',
  integrity_time: {
    OBSTRUCTED: 17646.30325166999,
    UNDERLIT: 26383.561565314983,
  },
  intervals: [
    {
      start_ns: 1785536701135049626,
      expected_seconds: 65.803826719,
      recorded_seconds: 0,
      integrity_state: 'OBSTRUCTED',
      gaps: [{ start_ns: 1785536721718623657, duration_ms: 9091, cause: 'LOST' }],
    },
    {
      start_ns: 1785561606937900088,
      expected_seconds: 8443.551353708,
      recorded_seconds: 326.935158802,
      integrity_state: 'UNDERLIT',
      gaps: [{ start_ns: 1785561643973639983, duration_ms: 8386800, cause: 'LOST' }],
    },
    {
      start_ns: 1785536766938876345,
      expected_seconds: 59.999101954,
      recorded_seconds: 0,
      integrity_state: 'OBSTRUCTED',
      gaps: [],
    },
    {
      start_ns: 1785536826937978299,
      expected_seconds: 60.00067125,
      recorded_seconds: 0,
      integrity_state: 'OBSTRUCTED',
      gaps: [],
    },
    {
      start_ns: 1785537066938672451,
      expected_seconds: 60.000074814,
      recorded_seconds: 5.432388217,
      integrity_state: 'UNDERLIT',
      gaps: [],
    },
    {
      start_ns: 1785537126938747265,
      expected_seconds: 60.00009093,
      recorded_seconds: 60.485240129,
      integrity_state: 'UNDERLIT',
      gaps: [],
    },
  ],
  note: 'Coverage, integrity and segments are recorded. Welfare, incidents, severity, risk and narrative are NOT available — see /api/capabilities. Absence of detection is not confirmation of wellbeing.',
  retrieval_holes: [
    { start_ns: 1785531600000000000, end_ns: 1785580460225729122, cause: 'expired_by_retention' },
    { start_ns: 1785580466178913777, end_ns: 1785580466181788122, cause: 'no_segment' },
    { start_ns: 1785580467170793984, end_ns: 1785580467173806122, cause: 'no_segment' },
  ],
  segments: { count: 328, seconds: 325.52253916199976 },
}

/** `GET /api/report?camera=nope&date=2026-08-01` — an unknown camera: 200, all-zero, one retrieval hole. */
export const REAL_REPORT_UNKNOWN_CAMERA: RecorderCoverageReport = {
  blind_spots: null,
  camera_id: 'nope',
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
  note: 'Coverage, integrity and segments are recorded. Welfare, incidents, severity, risk and narrative are NOT available — see /api/capabilities. Absence of detection is not confirmation of wellbeing.',
  retrieval_holes: [
    { start_ns: 1785531600000000000, end_ns: 1785618000000000000, cause: 'no_segment' },
  ],
  segments: { count: 0, seconds: 0 },
}

/**
 * `GET /api/report?camera=corridor_1&date=2026-08-01` — a camera whose source
 * never connects: coverage was fully expected but 0% recorded. Distinct from
 * `REAL_REPORT_UNKNOWN_CAMERA` — expected_seconds is real here — and the page
 * must not render the two the same way. `intervals` excerpted as elsewhere.
 */
export const REAL_REPORT_FULLY_LOST: RecorderCoverageReport = {
  blind_spots: {
    path: '',
    regions: [],
    total_px: 0,
    total_pct: 0,
    frame_px: 921600,
    may_overlap: false,
    note: 'Masked regions are never observed and never recorded. Coverage percentages elsewhere describe only the unmasked part of the frame.',
  },
  camera_id: 'corridor_1',
  coverage: {
    expected_seconds: 44089.86468379996,
    gap_seconds: 44089.572,
    intervals_with_records: 595,
    recorded_pct: 0,
    recorded_seconds: 0,
    unexplained_shortfall_secs: 0.2926837999621057,
  },
  date: '2026-08-01',
  integrity_time: { LOST: 44089.86468379996 },
  intervals: [
    {
      start_ns: 1785536701135049626,
      expected_seconds: 65.803826719,
      recorded_seconds: 0,
      integrity_state: 'LOST',
      gaps: [{ start_ns: 1785536701135049626, duration_ms: 65803, cause: 'STREAM_LOST' }],
    },
    {
      start_ns: 1785536766938876345,
      expected_seconds: 59.999101954,
      recorded_seconds: 0,
      integrity_state: 'LOST',
      gaps: [{ start_ns: 1785536766938876345, duration_ms: 59999, cause: 'STREAM_LOST' }],
    },
  ],
  note: 'Coverage, integrity and segments are recorded. Welfare, incidents, severity, risk and narrative are NOT available — see /api/capabilities. Absence of detection is not confirmation of wellbeing.',
  retrieval_holes: [],
  segments: { count: 0, seconds: 0 },
}
