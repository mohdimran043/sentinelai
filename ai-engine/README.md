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

docker compose -f ../deploy/compose/docker-compose.core.yml --profile core up -d   # Postgres, Redis, RabbitMQ, MinIO, mediamtx

uvicorn sentinel_ai.api.app:app --host 0.0.0.0 --port 8000
```

**The last line above is aspirational, not runnable today.**
`sentinel_ai/api/app.py` exposes `create_app(service: EngineServiceProtocol)`,
a factory that takes an already-built `EngineService` — deliberately, "so
tests never build a real `EngineService`" (its own docstring). No task
through Phase 1B has added the composition root that would actually build
one: something that reads a camera list (id, RTSP URL, `CameraProfile`),
constructs a real `RtspSource` per camera, wires the real `Yolo11Detector` /
`ByteTrackTracker` / `Qwen25VLDescriber` / `MinioClipWriter` /
`RabbitMQPublisher` into a `ModelRegistry` + `ResidentSet` +
`VlmScheduler` + one `CameraRunner` per camera, and hands the result to
`create_app`. There is no `cameras.yaml`/`.toml` loader anywhere in the
codebase either. Until that composition root exists, `uvicorn
sentinel_ai.api.app:app` has no module-level `app` to serve, and starting
the engine against a live RTSP source is not a one-line command — see
"What was actually run" below for what this task verified instead.

Configuration is environment variables prefixed `SENTINEL_` (see
`sentinel_ai/config.py`), or a `.env` file in `ai-engine/`.

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
   RabbitMQ → MinIO. This needs the composition root described above, which
   does not exist yet, so the commands below wire the pieces by hand instead
   of starting the (not-yet-buildable) FastAPI app. Full details, output,
   and the two real bugs this run found (and fixed) are in
   `.superpowers/sdd/task-14-report.md`; the short version:

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

Without a composition root, this is a short Python script rather than
`uvicorn`, wiring the same adapters `EngineService` would: `RtspSource`
pointed at `rtsp://localhost:8554/avenue_01`, `Yolo11Detector`,
`ByteTrackTracker`, `Qwen25VLDescriber`, `MinioClipWriter`,
`RabbitMQPublisher`, one `CameraRunner`, one `VlmScheduler`. This task's
report (`.superpowers/sdd/task-14-report.md`) has the exact script used and
its full output for the run this task actually performed.

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
