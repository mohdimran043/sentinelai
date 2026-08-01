// oxlint-disable no-loss-of-precision -- The recorder's nanosecond timestamps
// genuinely exceed Number.MAX_SAFE_INTEGER, and JSON.parse rounds them exactly
// the way these literals do. Writing a "safe" number here would make the
// fixture stop matching what the live API delivers, which is the one thing a
// fixture must never do. See recorder/recorder.types.ts.
import type {
  RecorderAlert,
  RecorderAlertsResponse,
  RecorderCapabilitiesResponse,
  RecorderModelsResponse,
  RecorderSettingsResponse,
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
