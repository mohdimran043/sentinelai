# Operations

## The compose stack

`deploy/compose/docker-compose.core.yml`, all under the `core` profile. Image
tags are pinned, not `latest`, so `make up` is reproducible.

```bash
make up      # docker compose --profile core up -d
make logs
make down
```

| Service | Image | Ports | Used by |
|---|---|---|---|
| `rabbitmq` | `rabbitmq:3.13-management` | 5672 (amqp), 15672 (UI) | **The engine.** Event publishing. |
| `minio` | `minio/minio:RELEASE.2024-10-13…` | 9000 (S3), 9001 (console) | **The engine.** Evidence clips. |
| `minio-init` | `minio/mc` | — | One-shot: creates the `sentinel-clips` bucket, then exits 0. |
| `mediamtx` | `bluenviron/mediamtx:1.9.3` | 8554 (RTSP), 1935, 8888, 8889 | Video-source normaliser for the demo. |
| `postgres` | `postgres:16` | 5432 | **Nothing yet.** Scaffolded for the Phase 1C Go backend. |
| `redis` | `redis:7` | 6379 | **Nothing yet.** Same. |

The engine talks only to `rabbitmq` and `minio`. Postgres and Redis share this
one compose file rather than forking a second one for Phase 1C.

`mediamtx` has no Docker healthcheck: the official image is built `FROM scratch`
— no shell, no curl — so no in-container probe can run. Reachability is
asserted from outside over the RTSP port by the integration smoke test.

> **Credentials are development-only.** `sentinel/sentinel`,
> `sentinel/sentinel123`, Redis with no password at all, and every port
> published on `0.0.0.0`. Do not put this compose file on a network you do not
> control.

## Running the engine

```bash
make install-runtime            # CPU: pipeline + fakes, no models
make install-gpu                # adds torch, ultralytics, transformers, bitsandbytes

cd ai-engine
cp cameras.example.json cameras.json     # then edit it
uvicorn sentinel_ai.main:app --host 0.0.0.0 --port 8000
```

`sentinel_ai/main.py` is the composition root and the real entry point — not
`api.app:app`, which only exposes `create_app(service)` so tests never build a
real `EngineService`.

Importing `main` reads no camera file, opens no socket and touches no GPU.
Composition happens inside the FastAPI lifespan, on the running loop, because
`RtspSource.__init__` calls `asyncio.create_task` and so cannot run before there
is a loop. That is what lets CI import the module with neither the `gpu` extra
nor a `cameras.json` present.

### Endpoints

| Endpoint | Returns |
|---|---|
| `GET /health` | Per-model `state`, `detail`, `vram_mib` |
| `GET /cameras` | `CameraStatus` for every camera, plus `config_writable` — whether the PATCH below will do anything on this deployment |
| `GET /cameras/{id}/telemetry` | One camera's counters, its `label`, its `zone` and its stored welfare policy. 404 if unknown |
| `PATCH /cameras/{id}` | **Write.** Edits `label`, `zone` and the per-camera welfare policy, persisted to `cameras.json`. **Off by default** — 403 unless `SENTINEL_ENABLE_CAMERA_WRITES=true`. See [Editing cameras from the console](#editing-cameras-from-the-console) |
| `POST /cameras/{id}/describe` | Forces a `user_requested` escalation, returns `event_id` |
| `GET /alerts` | Every alert the running engine holds, worst first then most recent first. **Volatile** — see [Alerts](#alerts) |
| `POST /alerts/{id}/acknowledge` | Records that a person has seen it. 409 if already resolved |
| `POST /alerts/{id}/resolve` | Records that a person has finished with it. Idempotent |
| `GET /authorized-persons` | The enrolled roster: names, permissions, and a **count** of reference faces. **Carries no biometric data of any kind** |
| `PUT /authorized-persons/{id}` | Create or replace a person. Carries no faces |
| `POST /authorized-persons/{id}/faces` | Enrol one reference face from a photograph (multipart). 422 with a stated reason if no usable face is found |
| `DELETE /authorized-persons/{id}` | Delete the person **and every face enrolled for them**. Not a soft delete |
| `GET /cameras/{id}/events` | A bounded ring of that camera's recent events, capped at 200, plus `latest` and `latest_description_state` (`none`/`available`/`unavailable`). Volatile — this is the console's view, not the event store. RabbitMQ plus the Go consumer is the durable record. 404 if unknown |
| `GET /events/stream` | `text/event-stream`. Opens with `event: backlog` carrying a JSON array, then streams live events. A sequence watermark makes the backlog-to-live handover gapless and duplicate-free; the same `event_id` at a higher sequence is a legitimate new version, not a repeat — that is how a clip URI back-fills onto an event a client already displayed. The ring is bounded, so a long disconnect genuinely loses history |

Telemetry counters: `frames_seen`, `frames_dropped`, `detections_run`,
`escalations`, `escalations_dropped`, `discontinuities`, `last_frame_at`,
`last_escalation_at`.

Two things to know when reading `/health`:

- **`status` is the hardcoded literal `"ok"`**, regardless of model state.
  Derive health from the per-model `state` field, never from `status`.
- The VLM sitting at `UNLOADED` is usually **correct** — it is the 600 s idle
  unload having fired, not a fault. A camera watching an empty corridor is
  supposed to look like that.

### Editing cameras from the console

`PATCH /cameras/{id}` is the engine's only endpoint that changes anything
durable. It is **disabled by default** and answers 403 until a deployment sets:

```bash
SENTINEL_ENABLE_CAMERA_WRITES=true
```

> **Read this before you set it.** The engine has **no authentication** — see
> [Known deployment gaps §2](#2-no-authentication). Every other endpoint is a
> read, so an exposed port has so far cost you information. This one is a write,
> and a persisted one: anything that can open a TCP connection to `:8000` can
> rename a camera to another camera's name and move it into another wing, and
> the change is written to `cameras.json` and survives the restart that would
> otherwise undo it. In a custodial setting that is the console's account of
> *where an incident happened*, altered anonymously and permanently.
>
> Enable it only where you have already decided that everything which can reach
> the port is permitted to reconfigure cameras — which in practice means binding
> uvicorn to `127.0.0.1`, or putting an authenticating reverse proxy in front of
> it. The engine logs a warning at startup when the flag is on. Revisit this in
> Phase 1C, when JWT arrives and the flag can default to on behind real auth.

What the endpoint will and will not change:

| Field | Editable at runtime? | Why |
|---|---|---|
| `label` | **Yes** — applied live, written to `cameras.json` | Metadata. Nothing in the pipeline branches on it |
| `zone` | **Yes** — same. `null` ungroups; omitting the field leaves the grouping alone | Metadata. Changes how a console groups a camera, not how the engine watches one |
| `notify_on`, `notify_min_confidence`, `clip_preroll_seconds`, `clip_postroll_seconds`, `summary_interval_seconds` | **Yes** — live from that camera's **next escalation**, written to `cameras.json` and **survive a restart** | The per-camera welfare policy, and the sharpest edge on this endpoint. `notify_on: []` is the mute switch: an anonymous caller can stop a camera notifying anyone about anything, from the next escalation onward, while it keeps detecting, keeps recording clips and keeps publishing events — so a muted camera looks exactly like a quiet one. The startup warning and the audit line naming the old and the new policy on every edit are the only places it shows |
| `url` | **No — restart required** | Changing the source means tearing down the running `CameraRunner`, its pre-roll buffer and any clip mid-recording, and building a new source. Separately, an RTSP URL routinely carries credentials, so an unauthenticated API neither accepts nor returns it |
| `profile` | **No — restart required** | It is the escalation policy the gate is part-way through applying (cooldowns, a token bucket with live state) |

A body carrying `url` or `profile` is a **422 naming the field**, never a 200
that quietly dropped it. Edit `cameras.json` and restart for those two.

The write is atomic and never half-applied. The edited document is re-parsed in
full before anything is written, so a request that would produce a file the
engine could not load at its next startup is refused outright; the new document
is then written to a `.tmp` sibling, `fsync`ed, and renamed into place. Comment
keys, other cameras, profiles and any field this version has no model of are
preserved — the file is edited, not regenerated. The in-memory camera is updated
only *after* the file is on disk, and from what the file now says, so the two
cannot disagree about an edit that succeeded.

If `cameras.json` has been edited by hand since the engine started and no longer
holds the camera being edited, the answer is **409** and nothing is written.
Reconcile the file and restart rather than letting the console overwrite it.

In the console, this is the **Camera record** panel on a camera's own page. When
`config_writable` is false the panel still shows the stored record, read-only,
and names the environment variable — hiding it would leave a camera's label,
zone and welfare policy with nowhere in the console they can be read. `url` and
`profile` are listed as restart-required with no value, because no endpoint
returns one.

## Welfare notifications

When the gate escalates and the VLM describes the keyframe, the same prompt also
asks it to look for five specific things and say plainly whether it sees them:

- a person collapsed, fallen, lying on the ground or apparently unresponsive;
- a physical altercation between people;
- apparent self-harm;
- an apparent medication or unlabelled-container ingestion;
- any other apparent distress or harm — restraint, dragging, a clutched injury,
  a raised weapon.

What it says comes back as a list of `{kind, confidence, evidence}` concerns
carried on the event, and — if it clears the camera's bar — is sent to somebody
who is not watching the console.

### Read this before you rely on it

**This asks a vision-language model about single still frames. It is not a fall
detector, and nothing downstream may treat it as one.**

- **One frame, no memory.** The model sees the keyframe of an escalation the
  gate already chose to spend a GPU slot on, and the token bucket and cooldown
  bound that to one escalation per 10 s per camera in steady state (a burst of
  2, then refill). Something that begins and ends between two escalations is
  never looked at by anything. The forced-look interval —
  `summary_interval_seconds`, 45 s by default — is the only thing that guarantees
  a camera is looked at at all when nothing trips a trigger.
- **No action recognition exists anywhere in this system.** The detector reports
  that a `person` box is present, never what that person is doing, and there is
  no action classifier behind any of this. A camera with `fall_detection`
  enabled adds a pose model and a geometry state machine — see the section
  below, which has limits of its own — but that is still not action
  recognition, and a camera without it has no temporal reasoning at all.
- **It misses things.** In prior measurement on real footage, **a stretcher
  carry was missed entirely** — a person carried out of a room on a stretcher,
  and the system said nothing. Absence of a notification is not evidence that
  nothing happened, and must never be used as one.
- **Fallible in both directions.** A single frame cannot reliably separate
  someone lying down from someone who has collapsed, or horseplay from an
  assault, and the model will confidently assert either.
- **There are two confidence tiers and there is no `certain`.** `possible` and
  `likely` are the whole scale, because one still frame cannot honestly earn
  more. A `likely collapse` is the model saying what it saw clearly supports
  that reading — not that it happened.
- **The medication check is deliberately narrow.** The prompt does not ask the
  model to name a substance, estimate a dose, or judge whether medication was
  prescribed, and the evidence it returns describes only what was visible.
  Clinical judgement is not something a single-frame VLM may be asked for in a
  custodial setting.

Treat a notification as a reason to go and look at the clip. Never as a finding.

### Fall detection: read this before you rely on it either

Enabling `fall_detection` on a camera adds something the single-frame check above does
not have — **memory**. A pure state machine watches each tracked person across seconds
and raises a concern only when all four of these happen in order: they were upright long
enough to be believed, they moved downward fast, they ended up horizontal, and they
stayed down and still. Only then is the vision-language model asked to confirm, and only
a concern it agrees with carries `basis: "temporal_pose_vlm"` on the published event.

That is materially more evidence than one frame. It is still not a fall detector in the
sense a clinician or a regulator would mean, and these limits are not hypothetical:

- **It detects falling, not lying.** Someone already on the floor when they enter frame
  raises nothing, because the machine never observed a transition. A person who
  collapsed just out of view and crawled into frame is invisible to it.
- **It has now been measured against real fall footage, and the result needs reading
  carefully.** On URFall — 30 clips containing a fall, 40 of daily activity — the state
  machine saw the full temporal signature in **29 of 30 falls**, and at the shipped
  settings **raised none of them**. URFall clips end about a second after the person
  lands, and `settle_seconds` is 3.0, so the machine is still waiting for stillness when
  the footage stops. A real camera keeps recording, so this is not the failure it looks
  like — but it does mean the detector has **never been validated end to end on footage
  that runs long enough to validate it**, and no public dataset this harness can consume
  provides any.

  The other half of that measurement is the one to act on: **23 of the 40
  daily-activity clips produced the same signature** — sitting down heavily, lying down
  deliberately, bending to pick something up. The confirmation window is what separates
  those from a fall. Shortening it to make the detector fire sooner trades directly
  against a 57.5% false-alarm rate. Reproduce all of this with
  `python -m sentinel_ai.validation.falls --dataset ../datasets/urfd`, and use
  `--settle-seconds` to see the trade rather than argue about it.

  Treat the default thresholds as a starting point to be tuned against your own cameras,
  not as a calibration.
- **Occlusion defeats it.** A person who falls behind furniture, or whose box merges
  with another person's, is a person the tracker loses — and a lost track takes its
  episode with it, silently.
- **`settle_seconds` is a floor on how late you are told.** At the 3.0 s default,
  nothing is raised until the person has been down and still for three seconds, and the
  clip's post-roll adds more. Lowering it trades notice against a stumble raising an
  alarm.
- **Movement restarts that timer.** Someone struggling to get up is still down, and the
  machine keeps waiting rather than cancelling — so a person in distress who keeps
  moving is reported *later*, not never, and possibly much later.
- **Pose is an improvement, not a guarantee.** Where a skeleton is unavailable or
  low-confidence the machine falls back to bounding-box shape, which a crouch, a carried
  object or a box that grew to include a chair can all confound. The event records which
  reading was used.
- **One escalation per frame.** If two people fall in the same instant, one is reported
  and the other is not — the keyframe is shared, so a second call would describe the
  same image.

The same rule as above applies, for the same reason: a fall concern is a reason to go
and look at the clip. Never a finding, and never a medical one.

### When a notification actually goes out

A concern notifies when **all three** hold:

1. its `kind` is in that camera's `notify_on` (default: every kind), **and**
2. its `confidence` meets that camera's `notify_min_confidence` (default:
   `likely`), **and**
3. if the concern is only `possible`, the event's threat score is already in the
   **caution band or above** — `medium`, `high` or `critical`, i.e. a threat
   score of 0.4 or more.

The third clause holds even for a camera that lowered its own bar to `possible`.
Asking to be told about maybes is not the same as asking to be told about every
maybe in an otherwise unremarkable scene, and a notifier that cries wolf is one
an operator learns to ignore — which is a muted notifier with extra steps.

The note carries **only the concerns that routed**, never the whole assessment:
including a kind the operator muted would leak exactly what `notify_on` exists
to suppress.

**The console is the exception, deliberately.** `GET /cameras/{id}/events` and
the camera page show *every* concern the model reported, unfiltered by
`notify_on`. That policy decides who gets paged; it has no business deciding
what an operator looking straight at the camera page may see. Muting a camera
silences its notifications without also blinding its console — so a muted camera
still looks quiet on the pager and still tells the truth on screen.

Per-camera policy lives in `cameras.json` and is editable at runtime — see
[Editing cameras from the console](#editing-cameras-from-the-console) and
[Configuration](configuration.md#the-per-camera-welfare-policy).

### Choosing the channel

| `SENTINEL_NOTIFIER_KIND` | What it does |
|---|---|
| `logging` *(default)* | One structured INFO line per routed note. No network, no configuration. A deployment that has decided nothing still leaves a trail an operator can `grep` |
| `webhook` | POSTs the note as JSON to `SENTINEL_NOTIFIER_WEBHOOK_URL` |

There is no `off`. Muting is per camera, via `notify_on: []` — which is a
deliberate, auditable, per-camera decision rather than a global switch somebody
flips during a noisy week and nobody re-flips.

`webhook` with no URL **fails at startup**. It does not fall back to logging: an
operator who mistyped the variable would otherwise get a process that starts
cleanly and never sends the one alert it exists for.

**Treat the webhook URL as a credential.** ntfy and Slack both put a
per-recipient token in the URL path. The engine never logs it, never follows a
redirect that could re-send it to another host, and installs a redaction filter
over httpx's own request logging for as long as the notifier lives — httpx logs
`HTTP Request: POST <url>` at INFO by default, which would otherwise put that
token in the logs of every deployment that configures one.

Delivery runs on its own worker, off the escalation path. One attempt is bounded
at 5 s; a transient failure (5xx, 429, 408, connection error) is retried up to
three times with backoff, a worst case of 18 s. A **4xx is not retried** — a 400
will be 400 again, and retrying a 401 just replays a rejected credential.
`SENTINEL_NOTIFIER_TIMEOUT_SECONDS` (default 20 s) is the outer ceiling on all of
that, and a webhook deployment whose value does not clear the adapter's 18 s
**refuses to start**. A note that still cannot be delivered is written to
`SENTINEL_DEAD_LETTER_DIR` with `record_type: "WelfareNote"` rather than dropped.

A hanging endpoint costs the note it belongs to and nothing else — never the GPU
admission slot, never the next describe, never a clip. Notes queue up to 32 deep
behind a slow notifier and are dropped beyond that, counted and logged.

### The clip travels as a URL, and it may not open

The note carries `clip_uri`, a MinIO object URL — not the bytes. Whoever
receives the notification needs credentials for that bucket and a route to it,
and **may have neither**. The webhook body therefore carries a literal
`clip_uri_note` field saying exactly that, always present rather than only when
there is a clip: a responder who cannot open the link needs to know that
immediately, not after two minutes of clicking.

`clip_uri` is also legitimately `null`: a clip that failed to write does not
suppress the notification. A note with no clip is still worth sending.

### Latency is bounded below by the post-roll

The order is fixed: describe → clip finalised → publish → **notify**. The clip is
finalised when its post-roll deadline passes, so a notification cannot go out any
sooner than `clip_postroll_seconds` after the keyframe — 5 s by default, and
per-camera overridable.

That is a real trade-off and it points both ways: **a shorter post-roll is a
faster alert and less evidence.** Notifying before the clip was finalised would
send a note whose `clip_uri` pointed at nothing, which is why the order is what
it is.

## Alerts

An event is what happened. An **alert** is an episode that events accumulate into.

The engine keys an alert on `(camera, reason, subject)` and merges every later matching
event within `SENTINEL_ALERT_MERGE_WINDOW_SECONDS` (120 s) into it, incrementing
`occurrences` and extending `last_seen_at`. One person walking a corridor for twenty
seconds is one row saying it happened seventeen times, not seventeen rows. Measured on
real footage: forty-five seconds of one corridor produced eleven events and would have
produced eleven rows without this.

Acknowledge records that a person saw it. Resolve records that a person finished with
it, and a resolved alert stops absorbing recurrences — the same thing happening again
opens a new alert rather than quietly reopening a closed judgement. **Nothing resolves
itself**; there is no timeout anywhere in this path.

### Alerts: read this before you rely on them

**The register is in engine memory and a restart empties it.** An acknowledgement is not
durable. The durable record is the anomaly event published to RabbitMQ. Treat the alert
list as the view an operator works from right now, not as an audit trail — the absence
of an alert means "not held by this process run", never "did not happen".

It is bounded at `SENTINEL_ALERT_REGISTER_CAPACITY` (500) and evicts closed alerts
before open ones, so a very busy site loses old *resolved* rows first.

### Watching an alert's clip

`GET /alerts/{alert_id}/clip` is the **only** route that serves a recording, and the
console's alert rows play it inline.

```bash
curl -o clip.mp4 localhost:8000/alerts/<id>/clip              # the short copy
curl -o full.mp4 'localhost:8000/alerts/<id>/clip?short=false' # pre-roll, event, post-roll
```

`short=true` is the default and returns the `SENTINEL_NOTIFY_CLIP_SECONDS` trim — three
seconds, the length somebody watches while deciding where to go rather than scrolls
past. It falls back to the full recording when no short copy was made, so it never 404s
for a reason the caller could have avoided.

**The alert id is the whole of the authorisation.** The object is resolved from the
alert inside the engine and never taken from the request, so no caller can name a clip
this engine did not itself attach to an alert; the reader re-checks the bucket rather
than trusting that. There is still **no authentication** on this port — anyone who can
reach it and knows an alert id can watch that footage. That is the same exposure the
rest of this API has, and it is footage of people, so treat the port accordingly.

A `404` here means one of three things and does not distinguish them: no clip was ever
recorded, the clip has passed `SENTINEL_CLIP_RETENTION_DAYS`, or this engine has no
object store configured. `AlertEntry.clip_uri` tells you whether the first applies.

## Person authorization

Off by default. It is enabled per camera, and enabling it anywhere makes
`SENTINEL_FACE_ENCRYPTION_KEY` mandatory — the engine **will not start** without it.

```bash
python -m sentinel_ai.adapters.face.encrypted_store    # prints a fresh base64 key
export SENTINEL_FACE_ENCRYPTION_KEY=...
```

Then create a person and give them a reference face:

```bash
curl -X PUT localhost:8000/authorized-persons/$(uuidgen) \
  -H 'content-type: application/json' \
  -d '{"display_name": "A. Operator", "camera_ids": ["front-door"], "status": "active"}'

curl -X POST localhost:8000/authorized-persons/<id>/faces -F 'image=@photo.jpg'
```

Enrol **several** faces per person, from the angles that camera actually sees. One
reference is the usual reason somebody is not recognised in profile, which is why the
roster reports a count rather than hiding it.

### What is stored, and what is not

| | |
|---|---|
| Stored | A 512-d ArcFace embedding per reference face, each sealed individually with AES-256-GCM, in a 0600 file written by write-then-rename |
| **Not stored** | **Any face image.** The photograph is embedded and discarded inside the request that carried it |
| **Not logged** | Embeddings, similarity scores, or any biometric value. The roster API returns none either |

`DELETE /authorized-persons/{id}` removes the person *and* their embeddings. It is not a
soft delete — a record that dropped the name and kept the vectors would keep precisely
the part that identifies somebody. Revoking access without deleting the record is a
different act: `PUT` with `"status": "disabled"`.

**Rotating the key makes every enrolled face undecryptable.** Everybody must re-enrol.
That is what it means for the data to be genuinely encrypted rather than obscured.

### Person authorization: read this before you rely on it

**A miss is the safe failure here; a false accusation is not.** `UNAUTHORIZED_PERSON`
is deliberately low severity, well below a suspected fall, because the system's
confidence that it has correctly identified a stranger is much lower than its confidence
that somebody fell — and the cost of being wrong lands on a person.

- **It is a recognition system, not an access control system.** It has no authentication
  in front of it, it does not open doors, and it must not be wired to anything that
  does.
- **The 0.42 default threshold is now measured**, on the standard LFW verification
  protocol: 92.7% recall and **zero false matches in 1100 different-person pairs**, with
  the threshold sitting in a wide empty band between the impostor 95th percentile (0.098)
  and the genuine median (0.676). Reproduce it with
  `python -m sentinel_ai.validation.faces --pairs <lfw pairs parquet>`.
- **That was measured on portrait photographs, not on your cameras.** LFW faces are
  large, frontal and well lit. A ceiling-mounted camera eight metres down a corridor
  produces none of those, and `min_box_pixels` and `min_frontality` will reject most of
  what it sees — which is the safe failure, but it means recall on *your* footage is not
  92.7% and is not known. Tune per camera against faces from that camera.
- **Even-handedness is now partly measured, and it is not uniform.** On 1800 FairFace
  images (`python -m sentinel_ai.validation.faces --fairness 1800`), face detection is
  even across every labelled group (99.1–100%) and so is the share clearing the quality
  bars (67–76%). But **impostor similarity is not**: near-misses between different people
  sit at p99 = 0.156 for White faces and 0.179–0.237 for every other group. All are far
  below the shipped 0.42, so nothing fails today — but the margin protecting against a
  false match is 15–50% thinner for non-White faces, and a site that lowers
  `match_threshold` spends that margin unevenly.
- **Verification accuracy by group is still unmeasured**, and cannot be measured with
  what is publicly reachable: FairFace has no identity labels, so same-person pairs
  cannot be built from it, and the datasets that do (RFW, DemogPairs) require an
  institutional application. **This has not been validated as fair in the sense that
  matters most.**
- Recognition is **sticky for the life of a track**: once somebody is matched, turning
  away from the camera does not un-recognise them. A dropped and re-acquired track
  starts over.
- Nothing is reported until `min_observations` (4) frames across `min_duration_seconds`
  (2.0) agree. A single bad frame cannot accuse anybody.

## Camera tamper

Enabled per camera, needs **no model**, so it costs nothing to switch on site-wide. It
asks whether most of the frame has collapsed into one luma bin — a covered lens, a
sprayed dome, a light switched off — and reports only after `min_clear_seconds` (30 s)
of varied view followed by `obstructed_seconds` (10 s) of a blank one. The second
threshold rides out a lorry pulling across the view, headlights sweeping the lens, and
an auto-exposure hunt after a light is switched on.

**A camera that is dark all night does not alarm**, because the clear run resets the
moment it ends; the detector must have seen something varied *first*. That also means a
camera obstructed before the engine started is never reported — it has no clear run to
compare against.

Worth enabling everywhere for a reason that is easy to miss: an obstructed camera is
silent in every other signal the system has, and silence looks exactly like a quiet
corridor.

## Running the console

```bash
cd web && npm install && npm run dev     # http://localhost:5173
```

`npm run build` · `npm run test` · `npm run typecheck` · `npm run lint` ·
`npm run gen` (regenerate types from `contracts/`).

The console reads **two separate backends** and never conflates them: the AI
engine on `:8000` and the recorder appliance (`sentinel-ingest`, a different
product) on `:8080`. The anomaly event feed is **mocked with MSW**, because
those events arrive via RabbitMQ → Go → Postgres and Go does not exist yet.
The mock worker starts unless `VITE_EVENTS_API_URL` is set. Details in
[web/README.md](../web/README.md).

## The end-to-end demo

Both levels below have actually been run on a real GPU box against real
services — not merely documented.

### Automated proxy (one command)

`ai-engine/tests/e2e/test_demo_smoke.py`, marked `gpu` + `integration`. Runs the
real detector/tracker/VLM over the downloaded Avenue clip through
`FileSource(realtime=False)` — no mediamtx, no RabbitMQ, no MinIO. Passes in
about 23 s, producing one real `Event` with a non-empty description and a threat
score in [0, 1].

```bash
bash datasets/download_sample.sh
cd ai-engine && .venv/bin/python -m pytest tests/e2e/test_demo_smoke.py -v
```

### Full stack

```bash
# 1. Fetch the clip (once). ~780 MiB archive, deleted after extraction.
bash datasets/download_sample.sh

# 2. Core services.
docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d

# 3. Loop the clip into mediamtx as the avenue_01 RTSP path.
ffmpeg -stream_loop -1 -re -i datasets/avenue/avenue_01.mp4 -t 200 \
  -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/avenue_01

# 4. Run the pipeline against it.
cd ai-engine
cat > cameras.json <<'EOF'
{"cameras": [{"id": "avenue_01", "label": "Avenue (demo)",
              "url": "rtsp://localhost:8554/avenue_01"}]}
EOF
uvicorn sentinel_ai.main:app --host 127.0.0.1 --port 8000
```

**`-rtsp_transport tcp` is not optional.** The compose file publishes only
mediamtx's TCP port (8554), not its RTP/RTCP UDP ports (8000/8001). ffmpeg's
default RTSP-publish transport is UDP, so media packets silently never reach the
container and mediamtx kills the "publishing" session as idle after its 10 s
read timeout — even though the RTSP handshake itself succeeds. This failure mode
is silent and looks like a working connection.

### Seeing the output

The `sentinel.events` exchange is a **topic exchange with no queue bound by
default**. To see events you must bind a consumer to `anomaly.#` *before* the
engine publishes; otherwise messages are routed nowhere and silently discarded.

- RabbitMQ: `http://localhost:15672` (`sentinel`/`sentinel`)
- MinIO: `http://localhost:9001`, or `mc ls local/sentinel-clips/avenue_01/`

A representative clean 120 s run: `frames_seen=2988`, `frames_dropped=56`,
`detections_run=2932`, `escalations=5`, `escalations_dropped=0`,
`discontinuities=0`. Five events published (`track_count_spike`,
`speed_anomaly` ×2, `dwell_exceeded` ×2), threat scores 0.10–0.20, five clips in
MinIO. One clip verified with `ffprobe` at 8.36 s / 209 frames / 5 keyframes,
decoding cleanly under `ffmpeg -f null -`, with real pre-roll ahead of the
trigger frame.

## Tests and CI

```bash
make check      # lint + typecheck + test
make test       # pytest -m "not gpu"
make test-gpu   # pytest -m gpu
```

CI (`.github/workflows/ci.yml`) runs on `main` and `master`: `ruff check`,
`ruff format --check`, `mypy` (strict, over `sentinel_ai` **and** `tests`), then
`pytest -m "not gpu and not integration"` with `--cov-fail-under=85`.

Two deliberate choices worth not undoing:

- **`-W error` is not a CI flag.** `filterwarnings = ["error"]` lives in
  `pyproject.toml`, so it applies to every invocation — CI's, a developer's, an
  IDE's. A flag is only enforced by whoever remembers to type it. The `httpx2`
  dependency and the `supervision>=0.24,<0.28` pin exist *solely* because a
  warning is fatal under it; as a command-line flag, either could have been
  silently undone by a passing build.
- **The OpenAPI drift check is a test, not a workflow step.** CI already runs
  the suite, and it fails on the developer's machine the moment they change a
  route rather than twenty minutes later on a push.

Web CI (`.github/workflows/web-ci.yml`) is a separate workflow so a broken
`npm install` can never block the Python job. It runs `npm ci`, then checks the
generated API types, then `typecheck`, `lint`, `test` and the production build.

The generated-types check is the one place the bullet above is deliberately
inverted — it *is* a workflow step rather than a test. `web/src/api/engine.types.ts`
is generated from `contracts/openapi/ai-engine.yaml` by `npm run gen:api` and
committed, and verifying it means regenerating the file and asking git whether
anything moved; that is a working-tree question, not one a unit test should be
answering. It runs before `typecheck` on purpose, so a contract someone forgot
to regenerate against fails as "you forgot to regenerate" rather than as a
confusing type error several steps later. This is not hypothetical: the
camera-editing contract landed with the types 197 lines behind it, and `tsc`
stayed green throughout — a stale generated file is internally consistent, it
simply describes a schema the engine no longer serves.

## Known deployment gaps

These are real, known, and **currently unowned**. Do not discover them in
production.

### 1. The engine sends no CORS headers

`ai-engine/sentinel_ai/api/app.py` adds no CORS middleware — verified, there is
no `CORSMiddleware` anywhere in the engine source. A browser calling it
cross-origin is blocked.

The console works around this with a **Vite dev-server proxy** (`/engine/*` →
`http://127.0.0.1:8000`), which makes the request same-origin as far as the
browser is concerned. **That proxy exists only under `npm run dev`.** A built
bundle served by anything else has no proxy and no CORS, and will fail.

The same applies to the recorder appliance: `/recorder/*` →
`http://127.0.0.1:8080`, also dev-only, also no CORS.

A deployment needs one of:

- CORS middleware on the engine (and on the recorder) for the console's origin;
- a reverse proxy in front of both, doing what the dev proxy does;
- the console served from the same origin as the API.

Set `VITE_ENGINE_API_URL` / `VITE_RECORDER_API_URL` to the reachable base in
that case. **Nobody owns this decision yet.** It is the single largest gap
between "works on a laptop" and "deployed".

### 2. No authentication

`web/src/store/session.ts` holds an operator display name in `sessionStorage`
and is explicitly **not** an authentication boundary; `RequireSession.tsx` is a
routing convenience, not a security guard. JWT is Phase 1C's Go backend. The
engine's API has no auth of any kind.

This is why `PATCH /cameras/{id}` — the engine's only write endpoint — ships
**disabled**, rather than being treated as one more route. Reaching an
unauthenticated read endpoint costs disclosure; reaching an unauthenticated
*write* endpoint costs control, and the two are not the same size. Gating it
behind `SENTINEL_ENABLE_CAMERA_WRITES` does not make the port safe — nothing
short of authentication does — but it does mean a deployment that has not
thought about this is not silently reconfigurable by whoever finds it, and that
turning it on is a decision someone made rather than a default they inherited.
See [Editing cameras from the console](#editing-cameras-from-the-console).

### 3. The dead-letter spool has no operator story

`SENTINEL_DEAD_LETTER_DIR` accumulates events that could not be published and
that replay cannot fix. Nothing replays it, nothing surfaces its depth on
`/health` or in telemetry, and nothing alerts on it. **Check it manually.** A
directory that silently accumulates evidence is only half a guarantee.

### 4. VRAM measurement undershoots real peaks

`capabilities().vram_mib` is measured during `warmup()` against a small dummy
frame. On a real 810×1080 frame the VLM's true peak was observed 938 MiB above
the warmup figure, and `plan_residency()`'s admission control works off the
warmup number. `SENTINEL_VRAM_RESERVED_MIB` (2048) absorbs this today. Do not
tune it toward zero.

Related: `plan_residency()` has **no notion of externally consumed VRAM** — it
assumes it owns the whole card. On a headless server that is correct. On a
desktop it survives only because the reserve is larger than the compositor's
usage, which is a coincidence, not a guarantee.

### 5. `RtspSource.close()` can take up to ~20 s

PyAV/ffmpeg blocking calls cannot be interrupted from another thread, so
shutdown is bounded by the socket timeout (`stimeout=20000000`) rather than
being immediate. Expect a slow stop on a camera whose stream has gone away.
