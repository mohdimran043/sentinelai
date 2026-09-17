# SentinelAI AI Engine

Behaviour-intelligence inference orchestrator: ingests a camera stream,
detects and tracks objects, decides when a scene is worth describing,
describes it with a vision-language model, and publishes a validated
`anomaly_event` with an evidence clip. See
`docs/superpowers/specs/2026-08-01-sentinelai-phase1b-ai-engine-slice-design.md`
for the full design.

## Running it

```bash
cd ai-engine
python -m venv .venv && . .venv/bin/activate
pip install -e ".[runtime,dev]"      # CPU-only pipeline with fakes
# pip install -e ".[gpu]"            # add this on the GPU box for real models
# pip install -e ".[face-gpu]"       # and this only if a camera enables person_authorization

docker compose -f ../deploy/compose/docker-compose.core.yml --profile core up -d   # Postgres, Redis, RabbitMQ, MinIO, mediamtx

cp cameras.example.json cameras.json    # then edit it: see "Camera configuration"
# Only if some camera enables person_authorization — the engine refuses to start without it:
# export SENTINEL_FACE_ENCRYPTION_KEY=$(python -m sentinel_ai.adapters.face.encrypted_store)

uvicorn sentinel_ai.main:app --host 0.0.0.0 --port 8000
```

That last line is the real entry point. `sentinel_ai/main.py` is the
composition root: it reads the camera list, builds `Yolo11Detector`,
`Qwen25VLDescriber`, `ByteTrackTracker`, `MotionAnalyzer`, `MinioClipWriter`,
`RabbitMQPublisher` and one `RtspSource`/`FileSource` per camera, registers
both models with a `ModelSpec`, wires `ModelRegistry` + `ResidentSet` +
`VlmScheduler` + one `CameraRunner` per camera into an `EngineService`, owns
the reconnect loop that calls `connect()`/`replay_spool()` on the publisher,
and hands the result to `create_app`. `sentinel_ai/api/app.py` still exposes
only `create_app(service)`, so tests still never build a real
`EngineService`.

Process-wide configuration is environment variables prefixed `SENTINEL_`
(see `sentinel_ai/config.py`), or a `.env` file in `ai-engine/`.

## Camera configuration

Cameras live in a JSON file — `SENTINEL_CAMERAS_FILE`, default
`./cameras.json`, template committed as `cameras.example.json` (the real one
is gitignored, since it carries site names and URLs with credentials in
them):

```json
{
  "cameras": [
    {
      "id": "avenue_01",
      "label": "Avenue (demo)",
      "url": "rtsp://localhost:8554/avenue_01",
      "profile": {"cooldown_seconds": 5.0}
    }
  ]
}
```

`id` and `url` are required, `label` defaults to `id`, and `profile`
overrides any `CameraProfile` field — validated against that dataclass's own
field names, so a typo fails at startup instead of silently doing nothing. A
`url` with an `rtsp://`/`rtsps://` scheme builds an `RtspSource`; anything
else is taken as a path to a video file and builds a `FileSource`, so a
replay over recorded footage is the same code path as a live camera.

Six more optional per-camera fields exist, all with defaults, so a file that
mentions none of them keeps loading: `zone` (`room`/`corridor`/`dayroom`, for
grouping), and the welfare notification policy — `notify_on`,
`notify_min_confidence`, `clip_preroll_seconds`, `clip_postroll_seconds` and
`summary_interval_seconds`. An unknown `zone` or an unknown concern kind
**fails at startup** rather than being ignored, on the same reasoning as a
typo'd profile field. Full reference, with defaults and the absent-vs-`[]`
distinction that decides whether a camera notifies anyone, is in
[docs/configuration.md](../docs/configuration.md#the-per-camera-welfare-policy).

A file rather than more `SENTINEL_*` variables because a camera is a nested
record with a nested profile, and flattening a list of those into environment
names is a worse interface than one small document that can be diffed,
reviewed and mounted into a container.

### VRAM specs

`ModelSpec.vram_mib` starts from `SENTINEL_DETECTOR_VRAM_MIB` /
`SENTINEL_VLM_VRAM_MIB` (defaults: the figures measured on an RTX 4060) and is
replaced with the measured `capabilities().vram_mib` immediately after
startup. A configured seed is unavoidable — `plan_residency()` must decide
whether a model fits *before* it is loaded, and `capabilities().vram_mib` is 0
until `warmup()` has run.

## Development vs. Production Mode

`SENTINEL_MODE` (`development` | `production`) is the **only** place this
seam is chosen (`sentinel_ai/config.py`). Development is the default and
binds an in-process transport. Production Mode — gRPC — is not built in
this phase; setting `SENTINEL_MODE=production` today does not change engine
behaviour, since nothing downstream branches on it yet.

## `decode_hwaccel`

Video decode runs on CPU by default (PyAV, no NVDEC), a deliberate Phase 1A
deviation: this box's CPU cores comfortably absorb software decode at this
scale, and the VRAM a hardware decode surface would have used instead goes
toward the VRAM budget (spec §2.1). `decode_hwaccel`
(`sentinel_ai/config.py`, currently `None`) is the escape hatch if a future
deployment needs it — no source adapter (`FileSource`, `RtspSource`)
branches on it yet, so setting it today has no effect; it exists so
re-enabling hardware decode is a configuration change, not a rewrite.

One consequence of encoded (not raw) pre-roll buffering worth knowing:
`PreRollBuffer.flush()` walks back to the oldest keyframe at or before the
`clip_preroll_seconds` horizon, because a clip that starts mid-GOP is
undecodable. With a typical 2s GOP, a requested 3.0s pre-roll yields
3.0–5.0s in the actual clip — always more context, never less.

## `RtspSource` and reconnect

`sentinel_ai/adapters/sources/rtsp.py` implements `FrameSource` over a live
RTSP URL, structured exactly like `FileSource`: one PyAV demux pass fans out
to a decoded `FrameData` stream and an encoded `EncodedPacket` stream, and
repeats `FileSource`'s end-of-stream fix (the synthetic flush packet at
end-of-stream has no `dts` but must still be decoded, or trailing B-frames
are silently dropped).

On top of that, `RtspSource` reconnects with exponential backoff
(`rtsp_reconnect_initial_seconds` → doubling → capped at
`rtsp_reconnect_max_seconds`), via a separately-testable `_ReconnectLoop`
(no PyAV, no queues — see `tests/adapters/sources/test_rtsp.py`, CI-safe with
an injected fake clock/sleep). Every reconnect resets `RtspSource`'s own
`frame_index` counter to 0 while the injected monotonic `clock()` keeps
advancing `FrameData.timestamp`/`EncodedPacket.pts`. `CameraRunner`
(`pipeline/runner.py`, `_is_timeline_regression`) already watches exactly
this: a `frame_index` that drops below the previous frame's is treated as a
stream discontinuity, resetting the tracker, motion analyzer, gate state,
and pre-roll.

The RTSP decode path itself (`_connect_once`/`_pump_stream`) needs a real
stream and is not asserted in CI — there is no CI-safe way to test a live
RTSP reconnect without a running mediamtx. Only `_ReconnectLoop`'s
backoff sequencing (and the pure `_annexb_keyframe_bytes` helper below) is
unit-tested.

One RTSP-specific fix worth knowing about: RTP H.264 (RFC 6184) carries
SPS/PPS out-of-band, in the RTSP SDP's `sprop-parameter-sets` negotiated
once at session setup, unlike an MP4-sourced Annex-B stream (where
`h264_mp4toannexb` repeats SPS/PPS before every keyframe). `MinioClipWriter`
(Task 9)'s "extract_extradata" bitstream filter was built and tested only
against the latter. Without `RtspSource` stitching the SDP-derived
`codec_context.extradata` back onto every keyframe packet's bytes
(`_annexb_keyframe_bytes`), every clip opened against a real RTSP source
came back an empty file — found live, during this task's own manual demo,
not a hypothetical. See the task report for the full diagnosis.

## Datasets

`datasets/download_sample.sh` fetches one CUHK Avenue clip for GPU
verification and the end-to-end demo; see `datasets/README.md` for
provenance and licence. The footage itself is never committed.

## The end-to-end demo

Two levels exist, and **both were actually run** during this task, on a real
GPU box with real core services — not just documented:

1. **Automated proxy** (`ai-engine/tests/e2e/test_demo_smoke.py`, marked
   `gpu` + `integration`): runs the real detector/tracker/VLM trio over the
   downloaded Avenue clip through `FileSource(realtime=False)` — no
   mediamtx, no RabbitMQ, no MinIO. Ran green: 1 passed in ~23s, producing
   one real `Event` with a non-empty description and a threat score in
   `[0, 1]`. This is what CI deselects (`-m "not gpu and not integration"`).
2. **Manual, full-stack demo**: mediamtx → `RtspSource` → real models →
   RabbitMQ → MinIO, started with `uvicorn sentinel_ai.main:app`. It used to
   need a bespoke script because there was no composition root; it no longer
   does. Full details of the original run, and the two real bugs it found
   (and fixed) were recorded during development. Those working notes live in
   `.superpowers/sdd/`, which is **gitignored** — they are not in a fresh clone.
   What survives of them is in `docs/decisions.md`, which is the durable record.
   The short version:

### 1. Fetch the clip once

```bash
bash datasets/download_sample.sh
```

### 2. Start core services

```bash
docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d
```

### 3. Loop the clip into mediamtx as the `avenue_01` RTSP path

```bash
ffmpeg -stream_loop -1 -re -i datasets/avenue/avenue_01.mp4 -t 200 \
  -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/avenue_01
```

`-rtsp_transport tcp` is not optional here: `docker-compose.core.yml` only
publishes mediamtx's TCP port (8554), not its RTP/RTCP UDP ports
(8000/8001). ffmpeg's default RTSP-publish transport is UDP; without
forcing TCP, the media packets silently never reach the container and
mediamtx kills the "publishing" session as idle after its 10s read timeout,
even though the RTSP handshake itself succeeds. Found by exactly this
failure during this task's own run — see the report.

### 4. Run the pipeline against it

```bash
cd ai-engine
cat > cameras.json <<'EOF'
{"cameras": [{"id": "avenue_01", "label": "Avenue (demo)",
              "url": "rtsp://localhost:8554/avenue_01"}]}
EOF
uvicorn sentinel_ai.main:app --host 127.0.0.1 --port 8000
```

Then `GET /health`, `GET /cameras`, `GET /cameras/avenue_01/telemetry` and
`POST /cameras/avenue_01/describe`. Note the `sentinel.events` exchange is a
topic exchange with no queue bound by default, so to *see* the events you
need a consumer bound to `anomaly.#` before the engine publishes.

### Verification checklist — all checked against a real run

- [x] RabbitMQ: `http://localhost:15672`'s API (or `rabbitmqctl
      list_queues`) confirms messages arrived on the `sentinel.events`
      exchange — a second, independent `aio-pika` consumer bound to it
      decoded 5 real payloads with non-empty prose `description`s (e.g.
      *"A person is walking through an indoor area with a purple handbag on
      the ground nearby."*) and `threat.value` in `[0, 1]` (observed 0.10–0.20).
- [x] MinIO: `mc ls local/sentinel-clips/avenue_01/` (or
      `http://localhost:9001`) showed 5 new `.mp4` objects, one per
      `clip_uri` the events carried.
- [x] Downloaded one clip and ran `ffprobe` — 8.36s duration (within the
      spec §5.5 3–5s pre-roll + 5s post-roll range), decoded cleanly
      end-to-end with `ffmpeg -f null -`, 5 keyframes / 209 total frames,
      confirming real pre-roll content precedes the trigger rather than the
      clip starting at it.
