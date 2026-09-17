# Configuration

Two mechanisms, and the split is deliberate:

- **`SENTINEL_*` environment variables** (or a `.env` file in `ai-engine/`) for
  everything that is a process-wide scalar. Defined in
  `ai-engine/sentinel_ai/config.py` — 58 settings, all listed below.
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

## Pose (fall detection)

Loaded only when some camera enables `fall_detection`.

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_POSE_MODEL_ID` | `yolo11n-pose.pt` | The keypoint model. `n`, not the detector's `s`: pose runs on people the detector already found and degrades to box geometry when a skeleton is missing, so its misses are recoverable where the detector's are not. |
| `SENTINEL_POSE_VRAM_MIB` | `420` | Startup estimate, replaced by the measured figure after warmup. Measured on an RTX 4090: **462 MiB**. |
| `SENTINEL_POSE_IDLE_UNLOAD_SECONDS` | unset (never) | Unlike the VLM's 600 s. Pose runs on every sampled frame of a fall-detection camera, so it is idle only when those cameras are empty — and evicting it then means the reload lands exactly when someone walks into an empty room. |
| `SENTINEL_POSE_CONF_THRESHOLD` | `0.4` | Person confidence inside the pose model's own pass. Distinct from `SENTINEL_DETECTOR_CONF_THRESHOLD`. |
| `SENTINEL_POSE_IMGSZ` | `640` | Inference size. |

## Person authorization (faces)

Loaded only when some camera enables `person_authorization`.

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_FACE_ENCRYPTION_KEY` | **unset** | Base64 AES key for the biometric store. **No default, and the engine refuses to start without it** when the capability is enabled — a default key is no key. Generate one with `python -m sentinel_ai.adapters.face.encrypted_store`. |
| `SENTINEL_FACE_STORE_PATH` | `./var/faces/faces.json` | Enrolled people and their sealed embeddings. Treat as a credential store. It holds **no images**. |
| `SENTINEL_FACE_MODEL_NAME` | `buffalo_l` | InsightFace bundle: SCRFD detection + ArcFace 512-d embedding, under ONNX Runtime rather than torch. Only the detection and recognition modules are loaded — the bundle also ships age and gender estimators, and this system has no use for either. |
| `SENTINEL_FACE_VRAM_MIB` | `704` | What `plan_residency()` reserves. Measured: 608 MiB standalone, 654 MiB warming beside the detector. **0 on a box with no CUDA provider for ONNX Runtime**, where the pipeline falls back to CPU — which keeps the capability working, at 125 ms per frame instead of 15. Install the `face-gpu` extra if that matters. |

**Rotating the key makes every enrolled face undecryptable**, so everybody must be
re-enrolled. That is the honest consequence of the data being genuinely encrypted rather
than obfuscated.

## Alerts

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_ALERT_REGISTER_CAPACITY` | `500` | How many alerts the in-process register holds before evicting the oldest **closed** one. Bounds memory; not a curation limit. |
| `SENTINEL_ALERT_MERGE_WINDOW_SECONDS` | `120.0` | How long after its last sighting an alert still absorbs a recurrence. The knob that decides whether an operator sees an *incident* or a *category*. |
| `SENTINEL_ALERT_STORE_PATH` | `./var/alerts/alerts.json` | Where triage state is written, so **an acknowledgement survives a restart**. Set to `null` for the in-memory-only behaviour this had before the store existed. |
| `SENTINEL_ALERT_FLUSH_INTERVAL_SECONDS` | `5.0` | How long a *machine-driven* change (an alert opening, an occurrence count rising) may sit unwritten. **Does not apply to acknowledging or resolving**, which are flushed before the API answers. |

**The alert store is not the record of what happened.** That is the anomaly event
published to RabbitMQ. This file records what a human *did about it* — which alerts were
seen and which were closed — and that exists nowhere else. It is rewritten whole on
every save, by write-then-rename, so a process killed mid-write leaves either the old
complete set or the new one.

A file that cannot be parsed, or one written by a build with a different schema version,
is treated as "no alerts to restore" and logged. Triage state is lost; the engine starts.
Taking surveillance down over a bookkeeping file would be the worse failure.

## Evidence clips and notification clips

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_CLIP_RETENTION_DAYS` | `1` | How long a clip is kept, as a **bucket lifecycle rule** the object store enforces. `0` disables it and keeps clips forever. |
| `SENTINEL_NOTIFY_CLIP_SECONDS` | `3.0` | How much footage a *notification* carries, and what the console plays inline on an alert row. `0` disables the short clip entirely: notifications then carry the full recording, and `GET /alerts/{id}/clip` falls back to it. |
| `SENTINEL_NOTIFY_CLIP_MIN_SEVERITY` | `high` | The severity at or above which a *notification* gets the short clip instead of the full one. Does not gate the console: `GET /alerts/{id}/clip` serves the short copy for any alert that has one, because an operator scanning a list wants three seconds regardless of band. |

**Two clips, because the two readers want opposite things.** The evidence clip keeps the
pre-roll and the post-roll for somebody who sits down with it later. A notification is
read on a phone by somebody deciding whether to walk down a corridor, and the useful
length is however long it takes to see what happened — so above the severity threshold
the note links a short cut instead. Below it, the full clip goes, because nobody is
running anywhere and the context is worth more than the shorter download.

The short clip is **cut from the start** of the finished one, so it is the pre-roll: a
clip that opened on the fall would show a person already on the floor. It is a remux,
never a re-encode ([ADR 3](decisions.md#3-clips-are-remuxed-never-re-encoded)) — and the
rule is sharper here, because this copy is the one most notifications are ever actually
watched as.

Retention is a lifecycle rule rather than a sweeper inside this process, because a
sweeper deletes nothing while the engine is down — and an engine that was down for a
week comes back to a week of clips it should have expired. Set `0` where the bucket
already has a policy of its own; layering a second one on top is how evidence disappears
a week before anybody expected.

## Keeping the event loop free

Two settings that exist because this engine is one Python process, and the thread it
runs on is the resource it is short of
([ADR 17](decisions.md#17-scale-by-processes-not-by-threads)).

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_SOURCE_MAX_FPS` | `0` (every frame) | Cap what a live source hands to the pipeline. Took three 1080p 30 fps cameras from **85% dropped to 0.0%**. |
| `SENTINEL_VLM_QUANTIZATION` | `nf4` | `none` loads the vision model in fp16: **1.42 s per describe against 1.97 s**, for 7224 MiB instead of 2820. |

**The frame cap is a cap, not an optimisation.** A camera delivers 30 fps; this pipeline
processes five to ten. The rest existed only to wake the loop and overwrite a mailbox
slot. The codec still decodes every frame — an inter-coded frame is meaningless without
its references — so this saves loop time, not decode time. What it costs is stated
plainly: **it sets the temporal resolution of everything downstream**, including the fall
machine's descent-rate measurement. Set it near what the pipeline actually achieves;
never below what a detector needs.

**`nf4` stays the default** because it is what makes the vision model fit beside a
detector on an 8 GiB card ([ADR 1](decisions.md#1-bitsandbytes-nf4-not-autoawq)). It
dequantises weights on every forward pass, which is a good trade when VRAM is scarce and
a poor one when it is not — and a describe holds the GIL in its per-token loop, so its
duration is time no camera's frame loop runs.

## Camera sources

`url` decides which source is built, and nothing else does:

| `url` looks like | Source | Notes |
|---|---|---|
| `rtsp://` or `rtsps://` | `RtspSource` | Reconnects with backoff |
| a page under `earthcam.com` | `EarthCamSource` | Resolves a fresh signed stream on every connection |
| anything else | `FileSource` | Read as a path |

### EarthCam pages

Configure the page a person would open in a browser — not an `.m3u8`:

```json
{
  "id": "linkou",
  "label": "Linkou Old Street, Taiwan",
  "url": "https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/"
}
```

A page carrying several cameras is disambiguated with EarthCam's own `?cam=` parameter,
exactly as the site does:

```json
"url": "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=bourbonstreet"
```

Naming a camera the page does not have is a startup error listing the ones it does —
falling back to a different camera would hand somebody a working stream of the wrong
street.

**Nothing is cached.** The page hands out a playlist URL signed with `?t=…&td=…` that
expires, so the source fetches the page again before *every* connection attempt. That is
the entire recovery strategy: a signature that dies mid-stream is repaired by the
reconnect its own death triggers, on the inherited exponential backoff. A 403 is
recognised and logged as `EarthCam stream expired; refreshing stream metadata` rather
than left to look like a camera that went private.

**The signature is never logged.** Every URL that reaches a log goes through
`redact_url`, and the field holding the token is kept out of the dataclass's `repr`, so
a traceback cannot leak what the log statements are careful about.

Check one without starting the engine:

```bash
python -m sentinel_ai.stream_tools resolve --url "<page url>"   # what the page points at
python -m sentinel_ai.stream_tools test    --url "<page url>"   # …and decode frames from it
```

Only `earthcam.com` and its subdomains are fetched, checked on the page URL *and* on
every media URL the page points at. Without that, a camera entry in `cameras.json` would
be a server-side request forgery primitive.

## Decode

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_DECODE_HWACCEL` | unset | Hardware decode device type — `cuda`, `qsv`, `drm`, `amf`. **Leave it unset unless CPU is the resource you are short of.** |

This was an inert placeholder until it was implemented and measured, and the measurement
is why it stays off. On an RTX 4090 at 1080p, CUDA decode alone runs at 801 fps and 0.26
cores against software's 580 fps and 1.00 core — but the frame still has to reach a numpy
array in host memory, and pulling an NV12 surface off the GPU to convert it there costs
more than decoding straight to YUV420P: end to end **1.84 cores against 1.60**. It buys
camera count on a core-starved box and costs throughput everywhere else.

A device type this FFmpeg build cannot open is a **startup error**, not a silent
fallback. A deployment that asked for hardware decode and quietly got software is one
whose capacity planning is wrong and whose logs agree with it.

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

Read [Operations → Welfare notifications](operations.md#welfare-notifications)
before turning this on anywhere real. It asks a vision-language model about
single still frames, it is not a fall detector, and it misses things.

| Setting | Default | Effect |
|---|---|---|
| `SENTINEL_NOTIFIER_KIND` | `logging` | `logging` \| `webhook`. `logging` writes one line per routed note and touches no network, so a deployment that configures nothing still leaves a trail. There is no "off" — muting is per camera, via `notify_on: []`. |
| `SENTINEL_NOTIFIER_WEBHOOK_URL` | *(unset)* | Where `kind=webhook` POSTs. **Required** by that kind: `webhook` with no URL fails at startup rather than falling back to logging, because an operator who mistyped this would otherwise get a process that starts cleanly and never sends the one alert it exists for. **Treat it as a credential** — ntfy and Slack both carry a token in the path, so the engine never logs it, never follows a redirect that could re-send it elsewhere, and installs a redaction filter over httpx's own request logging for as long as the notifier lives. |
| `SENTINEL_NOTIFIER_TIMEOUT_SECONDS` | `20.0` | Wall-clock ceiling on **one whole notification, retries included** — not the per-request HTTP timeout, which stays at the webhook adapter's own 5 s. That adapter retries a transient failure (5xx, 429, 408, connection error) up to three times with backoff, a worst case of 18 s, so a ceiling below that cancels the retries midway and turns every transient 429 into a note that is not delivered, not retried and **not spooled** — the cancellation never reaches the adapter's own dead-letter write, so the only trace is one warning line. A `webhook` deployment whose value here does not exceed that 18 s therefore **fails at startup**, naming both numbers; unset it to get a default already sized for this adapter. |

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
- `capabilities` is optional, and constrained to `scene_description` |
  `anomaly_detection`. See below.

### Per-camera AI capabilities

Which AI runs on one camera. Optional, and its absence means both capabilities —
exactly what every camera did before the field existed — so an existing
`cameras.json` keeps loading and behaves identically.

```json
{
  "id": "corridor_3",
  "url": "rtsp://localhost:8554/corridor_3",
  "capabilities": ["anomaly_detection"]
}
```

| Capability | What it turns on | What it costs |
|---|---|---|
| `scene_description` | The VLM describes an escalation the gate allowed. | Detector + VLM |
| `anomaly_detection` | The six automatic escalation triggers. | Detector |
| `fall_detection` | The temporal fall/collapse state machine (§7). | Detector + pose |
| `abandoned_object` | A bag a person was with, left behind and still. | Detector |
| `camera_tamper` | This camera going blind — lens covered, view obstructed, unlit. | **No model** |
| `zone_monitoring` | Restricted areas entered and virtual boundaries crossed. | Detector |
| `person_authorization` | Faces checked against the enrolled roster (§8–§12). | Detector + face |

**`camera_tamper` is free.** It reads the luma histogram the motion stage already
computes, so it can be switched on across an entire site for the cost of one comparison
per frame. That matters more than it sounds: every *other* detector goes silent when a
camera is obstructed, and silence is exactly what a healthy camera watching an empty
corridor looks like.

**`person_authorization` is the one capability that is a decision about people rather
than about compute.** It is off by default and enabled per camera, and the engine
**refuses to start** if any camera enables it without `SENTINEL_FACE_ENCRYPTION_KEY`
set — biometric data is not written in the clear on a fallback path.

`fall_detection` is independent of `anomaly_detection`, deliberately: the triggers ask
"is this scene worth a look", while fall detection asks one specific question about one
person over several seconds. A dayroom that should raise a fall but not a track-count
spike is an ordinary configuration, not a contradiction.

**This is not cosmetic, and it is the lever that makes many cameras affordable.**
The process loads the union of what every camera asks for, decided *before
anything is constructed*: a file in which nobody asks for `scene_description`
never downloads a vision-language model, never gives it VRAM, and never lists it
in `GET /health`. Skipping is also per camera — a camera with `[]` is handed no
detector at all, so it costs a decode and nothing else.

An explicit `[]` is a real instruction and is **not** the same as omitting the
field: the camera is still decoded and its liveness still reported, and no model
runs against it. Collapsing the two would re-enable monitoring on a camera
somebody deliberately switched off. A name the engine does not know **fails at
startup**, for `zone`'s reason and a stronger one — a silently dropped capability
leaves an operator believing monitoring is running that never was.

Turning a capability off changes behaviour rather than removing information. A
camera with `anomaly_detection` but not `scene_description` still escalates and
still publishes events; those events carry a metadata-derived description,
`description_unavailable: true`, and a `description_skipped` key in `metadata`
saying the reason was configuration rather than a model failure. A console must
render those two differently, or an operator learns to ignore a flag that
otherwise means a real fault.

**Editing at runtime.** `PATCH /cameras/{id}` takes `capabilities` as a whole
replacement list. Disabling always works. *Enabling* one whose model this process
did not load is a **409 naming the restart**, not a 200 that quietly did nothing:
which models exist is fixed at startup, and placing a 3B vision model under an
HTTP request would stall every camera sharing the GPU. Nothing is written when
that happens, so the file and the running camera never disagree.

### Per-camera fall thresholds

`fall_policy` on a camera entry, read only when `capabilities` includes
`fall_detection`. Every field is optional and every one has a default, so a camera that
omits the object entirely gets the defaults below.

```json
{
  "id": "dayroom_1",
  "url": "rtsp://localhost:8554/dayroom_1",
  "capabilities": ["anomaly_detection", "fall_detection"],
  "fall_policy": { "settle_seconds": 4.0, "min_descent_rate": 0.8 }
}
```

An unknown field name **fails at startup**, exactly as an unknown `profile` field does,
and for a sharper reason: a silently-ignored threshold leaves an operator believing they
tuned a camera that is still running the defaults — and here the defaults decide whether
anyone is told a person is on the floor.

| Field | Default | Effect |
|---|---|---|
| `min_upright_seconds` | `0.4` | How long a person must have been upright before a descent can count. Guards against someone already on the floor when the track begins. |
| `upright_aspect_max` | `0.75` | Box width/height at or below which a person reads as upright. |
| `horizontal_aspect_min` | `1.1` | Width/height at or above which they read as horizontal. Must exceed `upright_aspect_max`. |
| `upright_torso_degrees_max` | `35.0` | Torso angle from vertical below which **pose** says upright. |
| `horizontal_torso_degrees_min` | `60.0` | Torso angle above which pose says horizontal. Short of 90 on purpose: someone slumped against a wall has not landed flat. |
| `min_keypoint_confidence` | `0.4` | Per-joint score below which a keypoint is ignored and the frame falls back to box geometry. |
| `min_descent_rate` | `0.7` | **Body heights per second** of downward movement that separates a fall from sitting down. |
| `descent_window_seconds` | `2.5` | How long the body has to reach horizontal after the descent began, before the episode is abandoned. |
| `settle_seconds` | `3.0` | How long they must stay down and still before anything is raised. The clause that separates "fell" from "fell and is not getting up" — and the dominant term in time-to-alert. |
| `still_radius` | `0.35` | How far the centroid may drift, in body heights, and still count as settled. Movement **restarts** the timer rather than cancelling the episode. |

**Everything is in body heights, never pixels.** A person three metres from the camera
and the same person twenty metres away fall at wildly different pixel rates, so a `px/s`
threshold is really a threshold on distance-from-camera — tuned on one camera and wrong
on the next. The person's own bounding box supplies the unit, so one policy means the
same thing on a corridor camera and a car-park camera.

**What it detects, and what it does not.** The *transition*: upright, then a rapid
descent, then horizontal, then still. Someone already lying down when they enter frame
raises nothing, because there is no transition to observe. That is a real limitation and
the honest one — this detects falling, not lying. Pose improves the reading and is never
required; a camera without it runs the same machine on box geometry and records which it
used.

### Per-camera behaviour thresholds

Three more optional objects per camera, each read only when the matching capability is
enabled, each validated against its own field names so a typo fails at startup rather
than at the first incident.

| Object | Read when | Notable fields |
|---|---|---|
| `abandonment_policy` | `abandoned_object` | `unattended_seconds` (30.0) — how long the object stands alone before anyone is told. `attend_radius` (1.5) — how close a person must be, **in that person's own heights**, to count as with it. |
| `tamper_policy` | `camera_tamper` | `concentration_threshold` (0.85) — share of the frame in one luma bin that counts as blank. `obstructed_seconds` (10.0) — how long the collapse must persist. `min_clear_seconds` (30.0) — how long the view must have been varied first, which is what stops a permanently dark camera alarming forever. |
| `authorization_policy` | `person_authorization` | `match_threshold` (0.42), `min_observations` (4), `min_duration_seconds` (2.0), `min_box_pixels` (48), `min_frontality` (0.35). |

`zone_policy` is the one with nested geometry:

```json
"zone_policy": {
  "min_frames_inside": 3,
  "zones": [
    { "name": "stairwell", "polygon": [[0.0, 0.3], [0.35, 0.3], [0.35, 1.0], [0.0, 1.0]] }
  ],
  "lines": [
    { "name": "atrium threshold", "start": [0.55, 0.0], "end": [0.55, 1.0], "direction": "both" }
  ]
}
```

**Coordinates are fractions of frame width and height, never pixels.** An RTSP camera
can renegotiate resolution mid-stream, and a zone drawn against 1920×1080 silently
becomes a quarter of the intended area at 960×540. A coordinate outside `[0, 1]` is
rejected at startup — which is what catches a zone drawn in pixels by mistake.

`direction` is `both` | `a_to_b` | `b_to_a`, named by the line's own endpoints because
"northbound" stops meaning anything the moment somebody re-points the camera.

**Loitering is not in this list** because it already exists: it is the `dwell_exceeded`
escalation trigger, tuned by `dwell_radius_px` and `dwell_seconds` on the camera profile.

### The per-camera welfare policy

Five more optional fields per camera. Every one of them has a default, so an
existing `cameras.json` that mentions none of them keeps loading and behaves
exactly as it did.

| Field | Default when absent | Meaning |
|---|---|---|
| `notify_on` | **every kind** | Which welfare concern kinds this camera notifies a human about. An array of `collapse` \| `altercation` \| `self_harm` \| `medication` \| `distress` \| `other` |
| `notify_min_confidence` | `likely` | The lowest tier that may notify: `possible` or `likely`. There is no `certain` — one still frame cannot earn it |
| `clip_preroll_seconds` | `SENTINEL_CLIP_PREROLL_SECONDS` (3.0) | This camera's own pre-roll. `>= 0`; `0` means no lead-in at all |
| `clip_postroll_seconds` | `SENTINEL_CLIP_POSTROLL_SECONDS` (5.0) | This camera's own post-roll. `> 0` |
| `summary_interval_seconds` | the camera **profile's** `summary_interval_seconds` | This camera's forced-look interval. `> 0`. Distinct from `profile.summary_interval_seconds`, which is restart-only; this is the runtime-editable override, and it wins where both are set |

```json
{
  "id": "cell_14",
  "url": "rtsp://localhost:8554/cell_14",
  "zone": "room",
  "notify_on": ["collapse", "self_harm", "medication"],
  "notify_min_confidence": "possible",
  "clip_postroll_seconds": 12.0
}
```

Two things about this that are easy to get backwards:

- **Absent and empty are different**, exactly as `zone` already distinguishes
  them. `notify_on` absent means *every kind*; `notify_on: []` means *never
  notify from this camera*. Get this the wrong way round and either a camera the
  operator silenced starts alerting, or one they meant to route goes quiet.
- **An unknown concern kind fails at startup**, the same way an unknown `zone`
  does. A typo'd kind is a concern the operator meant to be told about and
  silently would not be.

`notify_min_confidence: possible` still does not notify on a `possible` concern
unless the event's threat score is already in the caution band or above — see
[Operations → Welfare notifications](operations.md#when-a-notification-actually-goes-out)
for the full rule, and
[what this is and is not](operations.md#read-this-before-you-rely-on-it) before
relying on any of it. It asks a vision-language model about single frames; it is
not a fall detector, and a stretcher carry was missed entirely in measurement.

Note the direct trade-off in `clip_postroll_seconds`: the notification is
dispatched after the clip is finalised, so this value is the floor on how long
it takes to reach a person. **A shorter post-roll is a faster alert and less
evidence.**

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
| `auto_escalation_enabled` | `true` | `false` short-circuits the whole gate for this camera, leaving only `user_requested`. **Derived from `capabilities` at load time — do not set it here.** Formerly `vlm_enabled`, renamed because it never controlled the VLM and, now that the VLM is separately optional, that name was actively misleading. |
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
