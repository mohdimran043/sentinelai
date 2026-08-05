# Configuration

Two mechanisms, and the split is deliberate:

- **`SENTINEL_*` environment variables** (or a `.env` file in `ai-engine/`) for
  everything that is a process-wide scalar. Defined in
  `ai-engine/sentinel_ai/config.py` — 36 settings, all listed below.
- **`cameras.json`** for the camera list, because a camera is a nested record
  with a nested per-camera profile. Flattening a list of those into environment
  names is a worse interface than one small document you can diff, review and
  mount into a container.

Settings are read once and cached (`get_settings()` is `@lru_cache`), so
changing the environment at runtime has no effect. Unknown `SENTINEL_*`
variables are ignored, not rejected — a typo fails silently, so check your
spelling.

## Mode

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_MODE` | `development` | `development` \| `production`. The **only** place the transport seam is chosen. **Production is not built** — nothing branches on it yet, so setting it changes no behaviour today. |

## Cameras and device

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_CAMERAS_FILE` | `./cameras.json` | Path to the camera list. See [below](#camerasjson). |
| `SENTINEL_ENABLE_CAMERA_WRITES` | `false` | Whether `PATCH /cameras/{id}` may edit a camera's `label`, `zone` and per-camera welfare policy (`notify_on`, `notify_min_confidence`, the clip and summary overrides) and write the change back to this file. The welfare fields take effect on that camera's next escalation. **Off by default: the engine has no authentication**, so an enabled write endpoint is reconfigurable by anything that can reach the port. Read [Operations → Editing cameras from the console](operations.md#editing-cameras-from-the-console) before turning it on. |
| `SENTINEL_DEVICE` | *(unset)* | Torch device for **both** models. Unset auto-detects via `yolo11.select_device()` — CUDA when visible, else CPU. Set it to pin a device, or to force CPU on a box that has a GPU. |

## VRAM budget

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_VRAM_TOTAL_MIB` | `8192` | Total VRAM the planner believes it has. |
| `SENTINEL_VRAM_RESERVED_MIB` | `2048` | Held back for the desktop compositor, CUDA context and fragmentation. Usable budget is `total - reserved` = **6144 MiB** by default. |

`plan_residency()` admits models against the usable figure. On a 24 GB card,
raise `TOTAL`; on a headless server, you can lower `RESERVED`.

## Models

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_DETECTOR_MODEL_ID` | `yolo11s.pt` | Weights file. Resolved relative to the engine's model cache unless absolute. Startup **fails** if the model's class names are missing any `DEFAULT_SALIENT_CLASSES` — otherwise every salient-class trigger would silently never fire. |
| `SENTINEL_DETECTOR_VRAM_MIB` | `432` | **Measured, not guessed** — see [below](#the-two-vram-numbers). |
| `SENTINEL_VLM_MODEL_ID` | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | Names the model *family*. The `-AWQ` suffix is **stripped before loading** — the engine loads the unquantised checkpoint and quantises with bitsandbytes NF4. AWQ is never touched. See [ADR 1](decisions.md#1-bitsandbytes-nf4-not-autoawq). |
| `SENTINEL_VLM_VRAM_MIB` | `2766` | **Measured, not guessed.** |
| `SENTINEL_VLM_IDLE_UNLOAD_SECONDS` | `600.0` | How long the VLM may sit idle before `ResidentSet.sweep_idle()` evicts it and returns its VRAM. It is transparently reloaded on the next escalation. |

### The two VRAM numbers

`detector_vram_mib=432` and `vlm_vram_mib=2766` are **figures measured on an
RTX 4060**, not entries from the design table. This distinction matters enough
that it is called out in the code, because getting it wrong has already caused
one wrong decision:

- Spec §4.3's design table estimated **900 MiB** (detector) and **4400 MiB**
  (VLM). Against 6144 usable, a second detector alongside the VLM
  (`4400 + 2*900 = 6200`) would *not* fit.
- The measured figures are **432** and **2766**. `2766 + 2*432 = 3630` leaves
  2514 MiB free, and `InsufficientVram` would not appear until roughly eight
  cameras.

So "a second detector would not fit" was never a valid reason for sharing one
detector. The real reason is different and still holds — see
[ADR 4](decisions.md#4-one-shared-detector-behind-a-lock). If you find yourself
reasoning from the design table, stop and read the measured numbers.

Both settings are only a **startup seed**. `plan_residency()` must decide
whether a model fits *before* it is loaded, and `capabilities().vram_mib` is 0
until `warmup()` has run — so a configured estimate is unavoidable.
Immediately after startup, `main.refresh_specs_from_capabilities()` replaces
each seed with the measured value.

Both measured values include a deliberate `+300 MiB` CUDA-context overhead on
top of `torch.cuda.memory_reserved()`. That is an intentional over-estimate:
the planner evicting a model too early is recoverable; an OOM mid-escalation
from under-reporting is not.

**Known limitation:** the measurement is taken during `warmup()` against a
small dummy frame. On a real 810×1080 frame the VLM's true peak was observed
938 MiB higher than the warmup figure. `RESERVED` absorbs this today; do not
tune `RESERVED` to zero.

## Detector tuning

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_DETECTOR_CONF_THRESHOLD` | `0.35` | Minimum detection confidence. Must be in (0, 1). |
| `SENTINEL_DETECTOR_IOU_THRESHOLD` | `0.45` | NMS IoU threshold. Must be in (0, 1). |
| `SENTINEL_DETECTOR_IMGSZ` | `640` | Inference input size, pixels. |
| `SENTINEL_DETECT_EVERY_N_FRAMES` | `1` | Run detection on every Nth frame. Raise to trade responsiveness for GPU headroom. |

## VLM scheduling

These are the *process-wide* governors. The *per-camera* governors live in
`cameras.json` profiles.

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_VLM_QUEUE_MAXSIZE` | `4` | Escalation queue depth. Submission is non-blocking and **drops when full**, counted as `escalations_dropped`, so a stalled VLM never back-pressures a camera. |
| `SENTINEL_VLM_TIMEOUT_SECONDS` | `30.0` | Per-describe timeout. On timeout the event is still published, with `description_unavailable=true`. |
| `SENTINEL_VLM_MAX_NEW_TOKENS` | `256` | Generation cap. |
| `SENTINEL_VLM_GLOBAL_CONCURRENCY` | `1` | `AdmissionGate` semaphore — how many describes may be in flight across **all** cameras. |
| `SENTINEL_VLM_GLOBAL_MIN_INTERVAL_SECONDS` | `2.0` | Minimum spacing between GPU admissions, process-wide. Enforced *after* the semaphore, so a caller that already queued does not pay both delays. |

## Video sources

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_SOURCE_REALTIME` | `true` | Pace a `FileSource` at wall-clock speed. Set `false` to replay a file as fast as it decodes (what the e2e smoke test does). No effect on RTSP. |
| `SENTINEL_RTSP_RECONNECT_INITIAL_SECONDS` | `1.0` | First reconnect backoff. Doubles each failure. |
| `SENTINEL_RTSP_RECONNECT_MAX_SECONDS` | `30.0` | Backoff ceiling. |
| `SENTINEL_DECODE_HWACCEL` | *(unset)* | **Inert.** Decode runs on CPU via PyAV; no source adapter branches on this yet. It exists so re-enabling hardware decode is a configuration change rather than a rewrite. See [ADR 2](decisions.md#2-cpu-decode-not-nvdec). |

## Evidence clips

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_CLIP_PREROLL_SECONDS` | `3.0` | How much footage before the trigger the clip must contain. |
| `SENTINEL_CLIP_POSTROLL_SECONDS` | `5.0` | How long the clip keeps recording after the trigger. A later escalation while recording *extends* this deadline rather than opening a second clip. |
| `SENTINEL_CLIP_TEMP_DIR` | `./var/clips` | Where clips are remuxed before upload. |

Pre-roll is a floor, not an exact figure. `PreRollBuffer.flush()` walks back to
the oldest keyframe at or before the horizon, because a clip starting mid-GOP is
undecodable. With a typical 2 s GOP a requested 3.0 s pre-roll yields 3.0–5.0 s
in the delivered clip — **always more context, never less**.

## Event publishing

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_RABBITMQ_URL` | `amqp://sentinel:sentinel@localhost:5672/` | Broker. Matches `deploy/compose/docker-compose.core.yml`. |
| `SENTINEL_RABBITMQ_EXCHANGE` | `sentinel.events` | Topic exchange. **No queue is bound by default** — to see events you must bind a consumer to `anomaly.#` before the engine publishes. |
| `SENTINEL_EVENT_SPOOL_DIR` | `./var/spool/events` | Disk buffer for validated events waiting on a broker that is merely down. Replayed automatically. |
| `SENTINEL_BROKER_REPLAY_INTERVAL_SECONDS` | `30.0` | How often `main.BrokerLink` re-drains the spool while the broker is up. A disk buffer is only half a guarantee without something replaying it; this is that something's period. |
| `SENTINEL_DEAD_LETTER_DIR` | `./var/spool/dead-letter` | Last resort for events that failed for a reason replay cannot fix — a schema violation, an unwritable spool, a transport bug — **and** welfare notes the webhook notifier could not deliver (a non-2xx/3xx response, a connection failure or timeout after retries were exhausted). Each record carries `record_type` (`"Event"` or `"WelfareNote"`) so a repair script can tell the two shapes apart. **Deliberately separate** from the spool: mixing them would put a permanently-unacceptable payload at the front of the replay queue. |

## Welfare notifications

Who gets told when the vision model reports a welfare concern. The routing rule
itself is per camera and lives in `cameras.json` (`notify_on`,
`notify_min_confidence`); these settings choose the channel it goes out on.

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_NOTIFIER_KIND` | `logging` | `logging` \| `webhook`. `logging` writes one line per routed note and touches no network, so a deployment that configures nothing still leaves a trail. There is no "off" — muting is per camera, via `notify_on: []`. |
| `SENTINEL_NOTIFIER_WEBHOOK_URL` | *(unset)* | Where `kind=webhook` POSTs. **Required** by that kind: `webhook` with no URL fails at startup rather than falling back to logging, because an operator who mistyped this would otherwise get a process that starts cleanly and never sends the one alert it exists for. **Treat it as a credential** — ntfy and Slack both carry a token in the path, so the engine never logs it, never follows a redirect that could re-send it elsewhere, and installs a redaction filter over httpx's own request logging for as long as the notifier lives. |
| `SENTINEL_NOTIFIER_TIMEOUT_SECONDS` | `20.0` | Wall-clock ceiling on **one whole notification, retries included** — not the per-request HTTP timeout, which stays at the webhook adapter's own 5 s. That adapter retries a transient failure (5xx, 429, 408, connection error) up to three times with backoff, a worst case of 18 s, so a ceiling below that would cancel the retries midway and turn every transient 429 into a lost note. |

Delivery runs on its own worker, off the escalation path: a hanging endpoint
costs the note it belongs to and nothing else — never the GPU admission slot,
never the next describe, never a clip. Notes queue up to 32 deep behind a slow
notifier and are dropped (counted and logged) beyond that. A shutdown drains the
queue, bounded by the same cap as the escalation drain.

## Object storage

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_MINIO_ENDPOINT` | `localhost:9000` | S3 API endpoint. |
| `SENTINEL_MINIO_ACCESS_KEY` | `sentinel` | |
| `SENTINEL_MINIO_SECRET_KEY` | `sentinel123` | Development credential. Change it anywhere real. |
| `SENTINEL_MINIO_BUCKET` | `sentinel-clips` | Created by the `minio-init` one-shot in the compose stack. |
| `SENTINEL_MINIO_SECURE` | `false` | Use TLS. |

## `cameras.json`

Path from `SENTINEL_CAMERAS_FILE`, default `./cameras.json` relative to
`ai-engine/`. The template is committed as `ai-engine/cameras.example.json`;
the real file is **gitignored**, because it carries site names and URLs with
credentials in them.

```json
{
  "cameras": [
    {
      "id": "avenue_01",
      "label": "Avenue (demo)",
      "url": "rtsp://localhost:8554/avenue_01"
    },
    {
      "id": "replay_01",
      "label": "Recorded footage (replay)",
      "url": "./datasets/avenue/avenue_01.mp4",
      "profile": { "cooldown_seconds": 5.0, "summary_interval_seconds": 45.0 }
    }
  ]
}
```

- `id` and `url` are **required**; `label` defaults to `id`.
- A `url` with an `rtsp://` or `rtsps://` scheme builds an `RtspSource`.
  **Anything else is treated as a path to a video file** and builds a
  `FileSource` — which is what makes replaying recorded footage the same code
  path as a live camera.
- `profile` overrides any `CameraProfile` field. It is validated against that
  dataclass's own field names, so a typo **fails at startup** rather than
  silently doing nothing.
- `zone` is optional, and constrained to `room` | `corridor` | `dayroom`. Omit
  it and the camera is ungrouped, which is a legitimate deployment. Give it a
  value the engine does not know and **startup fails**: a typo'd zone is a
  camera the operator meant to group and silently did not.

### This file is written as well as read

With `SENTINEL_ENABLE_CAMERA_WRITES=true`, `PATCH /cameras/{id}` edits `label`,
`zone` and the per-camera welfare policy (`notify_on`,
`notify_min_confidence`, `clip_preroll_seconds`, `clip_postroll_seconds`,
`summary_interval_seconds`) **in this file** — write-then-rename, with the whole
document re-validated before anything is written. Consequences worth knowing:

- The welfare-policy fields are written, **survive a restart**, and take effect
  on that camera's **next escalation** — never retroactively, and never on an
  escalation already in flight. `notify_on: []` is the mute switch: that camera
  stops notifying anyone about anything, while still detecting, still recording
  clips and still publishing events. Nothing about a muted camera looks different
  from a quiet one, which is why the engine writes an audit line naming the old
  and the new policy on every edit.
- Comment keys (`_comment`, `_note`), profiles, URLs, other cameras and any
  field a later version adds are all preserved; the file is edited, not
  regenerated. Formatting is normalised to 2-space JSON, so expect a reflow on
  the first write.
- `url` and `profile` are **never** written by the console and are rejected with
  a 422 if a request names them. Both require editing this file and restarting.
- If the console and a hand-edit race, the console loses: an edit naming a
  camera this file no longer holds is a 409 and nothing is written.

### Per-camera profile fields

Defaults from `ai-engine/sentinel_ai/domain/camera_profile.py`. Slice 1 uses
fixed values; Phase 4 replaces the percentile and baseline fields with learned
per-camera ones.

| Field | Default | Effect |
|---|---|---|
| `salient_classes` | person, car, truck, bus, motorcycle, bicycle, backpack, handbag, suitcase | Which detected classes the triggers care about. |
| `vlm_enabled` | `true` | `false` short-circuits the whole gate for this camera. |
| `min_track_frames` | `8` | Track age before `new_salient_track` fires. |
| `scene_delta_threshold` | `0.35` | Signature distance counting as a change. Must be in (0, 1]. |
| `scene_delta_frames` | `5` | Consecutive frames the change must persist for `scene_change`. |
| `dwell_radius_px` | `48.0` | How far a track may drift and still count as dwelling. |
| `dwell_seconds` | `30.0` | Dwell duration before `dwell_exceeded` fires. |
| `speed_percentile` | `0.95` | Reserved for Phase 4's learned threshold. **Unused today.** |
| `speed_fallback_px_s` | `180.0` | The speed threshold actually in force today. |
| `track_count_baseline` | `6` | Salient-track count above which `track_count_spike` fires. |
| `summary_interval_seconds` | `45.0` | Cadence of `periodic_summary`. |
| `bucket_capacity` | `2` | Token-bucket burst allowance. |
| `bucket_refill_seconds` | `10.0` | Seconds per token. |
| `cooldown_seconds` | `5.0` | Hard floor between VLM calls. |

`cooldown_seconds` is deliberately **shorter than** `bucket_refill_seconds`. At
or above it, the cooldown would dominate the bucket entirely and
`bucket_capacity` would become dead configuration — the burst allowance that
lets a genuinely novel scene get a second look could never be spent. At half the
refill interval it caps the worst case at 2 calls in 5 s rather than 2 calls in
100 ms, and it comfortably exceeds one VLM turnaround, so the next call is never
admitted before the previous description exists. Every field is validated in
`__post_init__`; an invalid combination raises at startup.

## Web console environment

All optional, all `VITE_`-prefixed, each read in exactly one file.

| Variable | Default | Read in |
|---|---|---|
| `VITE_ENGINE_API_URL` | `/engine` | `web/src/api/config.ts` |
| `VITE_RECORDER_API_URL` | `/recorder/api` | `web/src/recorder/config.ts` |
| `VITE_EVENTS_API_URL` | *(unset → MSW mock)* | `web/src/events/eventClient.ts`. **Also gates MSW**: if unset, `web/src/main.tsx` starts the mock service worker. |

The two defaults are Vite dev-server proxy prefixes and work only under
`npm run dev`. See [operations](operations.md#known-deployment-gaps).
