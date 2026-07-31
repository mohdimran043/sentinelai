# SentinelAI Phase 1B — AI Engine Vertical Slice

**Date:** 2026-08-01
**Status:** Approved for planning
**Predecessor:** [Phase 0+1 design](2026-07-31-sentinelai-phase0-1-design.md) · [Phase 1A plan](../plans/2026-07-31-phase1a-foundation-and-ai-engine.md)

## 1. Goal

Turn Phase 1A's pure core into a running system: ingest a video stream, detect and track
objects, decide when a scene is worth describing, describe it with a vision-language model,
and publish a validated event with an evidence clip.

Phase 1A delivered the algorithms with no way to run them. Phase 1B delivers the machinery
around them and the three real models, on one camera, on an 8 GB laptop GPU.

**Done means:** a looped video file arrives over RTSP from mediamtx, YOLO11s and ByteTrack
run on it, the escalation gate fires on genuine activity, Qwen2.5-VL describes the scene, and
an `anomaly_event` lands on RabbitMQ with an MP4 clip in MinIO that starts 3 s before the
event. Everything except the three model adapters is green in CI on CPU.

## 2. Inherited constraints

From the Phase 0+1 design and Phase 1A, unchanged and binding:

- `domain/` and `ports/` stay pure — no third-party I/O, no clock reads, no dependency on
  outer layers or on `config`. `tests/test_architecture.py` enforces this and, since the
  post-review hardening, catches aliased imports, from-imports, dynamic imports and relative
  outer-layer imports. Phase 1B must not weaken it.
- Development Mode is the default. The in-process transport binds by default; gRPC is
  Production Mode and is **not** built in this phase.
- Default VLM is Qwen2.5-VL-**3B**. The 7B model is never a default. The checkpoint is the
  unquantised `Qwen/Qwen2.5-VL-3B-Instruct` loaded 4-bit NF4, not the `-AWQ` build — see §7.
- VRAM ceiling 8192 MiB, reserved headroom 2048 MiB.
- Every GPU-requiring test is marked `@pytest.mark.gpu` and every docker-dependent test
  `@pytest.mark.integration`; CI runs `-m "not gpu and not integration"`.
- Events carry no model identity.
- Clip pre-roll is 3.0 s.
- All domain timestamps are float seconds from a monotonic clock, passed in explicitly.

### 2.1 Hardware reality, measured

Probed on the target machine rather than assumed:

| Resource | Actual | Consequence |
|---|---|---|
| GPU | RTX 4060 Laptop, **8188 MiB** | 4 MiB under the 8192 assumption; the 2048 MiB reserve absorbs it. No change. |
| CPU | 20 cores | Ample for PyAV CPU decode; vindicates the Phase 1A deviation away from NVDEC. |
| System RAM | 15 GB total, **~7 GB free** | The binding constraint. Drives the clip-writer design in §5.5. |
| Docker | 29.1.3 / Compose 2.40.3 | Present. |
| torch | installed by the Task 1 spike | 2.13 with Triton 3.7.1 — the combination autoawq could not work against. |

The spec's §4.3 VRAM table budgets ~7.6 GB of 8 GB. Phase 1A's CPU-decode deviation frees the
~0.3 GB that table assigned to NVDEC surfaces, so the projected figure was **~7.3 GB accounted
of 8.0 GB usable**, leaving ~0.9 GB unallocated *on top of* the 2.0 GB reserve already inside
that 7.3.

**Measured, after the Task 1 spike.** The VLM turned out cheaper than budgeted. bitsandbytes
NF4 peaks at 3517 MiB total with the desktop session running — 694 MiB of that is the
compositor, so the model itself costs ~2.8 GB against the 4.4 GB the table assigned it:

| Component | Budgeted | Actual |
|---|---|---|
| Desktop compositor | not modelled | ~0.7 GB |
| YOLO11s | ~0.9 GB | not yet measured (Task 11) |
| Qwen2.5-VL-3B | ~4.4 GB (AWQ) | **~2.8 GB (NF4)** |
| Reserved headroom | ~2.0 GB | ~2.0 GB |

Roughly 1.6 GB better off than planned, which absorbs the compositor overhead the original
table never accounted for and still leaves the residency planner room to keep both models
resident. The planner's behaviour is unchanged — it reads these numbers from
`Capabilities.vram_mib`, so the improvement arrives as data, not code.


## 3. Scope

### In

| Area | Delivered |
|---|---|
| Sources | `file.py`, `rtsp.py`, shared `PreRollBuffer` |
| Stages | Motion energy + scene signature |
| Pipeline | Per-camera async runner, stages 1–5 |
| Detectors | YOLO11s via Ultralytics |
| Trackers | ByteTrack via supervision |
| Vision | Qwen2.5-VL-3B, 4-bit NF4 (§7 — the AWQ path was tried and rejected) |
| Orchestrator | Registry, resident set, scheduler, global admission gate, service |
| Publishers | `inmemory.py`, disk-buffered `rabbitmq.py` |
| Storage | MinIO clip writer |
| API | FastAPI: health, camera status, describe-now |
| Infra | `docker-compose.core.yml`, `mediamtx.yml` |
| Data | Download script for a public sample clip |

### Out, with destination

| Deferred | Phase |
|---|---|
| Go backend (JWT, consumer, REST, WS hub) | 1C |
| React slice (login, dashboard, camera page) | 1D |
| gRPC transport / Production Mode | 2 |
| Multi-camera scheduling and fairness | 2 |
| Detector batching | 2 — see §4.2 |
| TensorRT export for YOLO11s | 2 |
| Behaviour learning, the 18 anomaly detectors | 4 |

## 4. Port changes

Three ports are settled before any adapter exists — two amended, one deliberately left alone.
Changing them now is cheap; changing them once the adapters and the pipeline hot path exist is
surgery, which is why the Phase 1A whole-branch review flagged them.

### 4.1 `ClipWriter` becomes a handle

Current signature takes the whole clip at once:

```python
async def write(camera_id, event_id, frames: Sequence[FrameData], fps) -> str
```

At 1080p30 a 3 s pre-roll plus post-roll tail is ~180 raw RGB frames — roughly **1.1 GB of
host RAM per concurrent clip** on a machine with ~7 GB free. Replace it with an
open/append/finish handle so media is written as it arrives and never accumulates.

The handle appends **encoded packets, not decoded frames**. This is the pivot the whole
clip path turns on: the bytes already arrived encoded, the pre-roll buffer already holds
them encoded (§5.1), and the output is an MP4 — so the clip is a **remux**, with no decode
and no re-encode anywhere in it. That is both the cheapest memory path and the cheapest CPU
path, and it keeps the evidence bit-identical to what the camera sent, which matters for a
clip whose purpose is evidentiary.

This requires a packet type in the ports layer alongside `FrameData`:

```python
@dataclass(frozen=True, slots=True)
class EncodedPacket:
    """A demuxed, still-encoded packet. `data` is bytes so ports stay codec-agnostic."""

    camera_id: str
    data: bytes
    pts: float          # seconds, monotonic, same clock as FrameData.timestamp
    is_keyframe: bool
    codec: str          # "h264", "hevc" — a plain str, so no codec library enters ports


class ClipWriter(ABC):
    @abstractmethod
    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle: ...


class ClipHandle(ABC):
    @abstractmethod
    async def append(self, packet: EncodedPacket) -> None: ...
    @abstractmethod
    async def finish(self) -> str:
        """Finalise the clip and return its URI."""
    @abstractmethod
    async def abort(self) -> None:
        """Discard a partial clip; must not raise."""
```

Peak memory becomes the pre-roll buffer plus one packet. `abort()` exists because a pipeline
shutdown mid-clip must not leak a temp file or a half-written MinIO object.

**Fallback.** Remuxing requires an MP4-compatible codec. H.264 and H.265 cover RTSP cameras
and everything mediamtx emits, so this holds for every source in scope. The writer dispatches
on `EncodedPacket.codec` and raises on a codec it cannot remux, rather than silently emitting
an empty clip — a clip writer that fails quietly is worse than one that fails. If a source
ever supplies something else, the adapter re-encodes from decoded frames — slower and lossy,
but the port signature does not change, because `EncodedPacket` is what the writer consumes
either way.

### 4.2 `FrameSource` also emits packets

The clip path consumes `EncodedPacket` and the inference path consumes `FrameData`, both
derived from one demux pass (§5.1). The port must therefore expose both. Keep the existing
`__aiter__` yielding `FrameData` as the primary iteration protocol — the pipeline is its main
consumer and should not have to filter a union — and add a separate accessor for the packet
stream:

```python
class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...
    @abstractmethod
    def packets(self) -> AsyncIterator[EncodedPacket]: ...
    @abstractmethod
    async def close(self) -> None: ...
```

Implementations fan one demux loop out to both consumers. `FakeSource` gains a synthetic
packet stream so the clip path is exercisable in CI without real media.

### 4.3 `ObjectDetector.detect` stays per-frame

`Capabilities.batch_max` advertises batching, but this slice is single-camera, so every batch
would have size 1. Adding a batch method now would be untested, unused code. **Decision:**
leave `detect` per-frame and document in the port's docstring that `batch_max` is advertised
for Phase 2 multi-camera use and is not yet honoured. Revisit when a second camera exists.

## 5. Components

### 5.1 Sources and the pre-roll buffer

`FileSource` reads a local file; `RtspSource` reads from mediamtx with exponential-backoff
reconnect. Both decode with PyAV on CPU (`decode_hwaccel` config flag remains the escape
hatch) and yield `FrameData`.

**Sources emit two streams from one demux pass.** Inference needs decoded pixels; the clip
needs encoded bytes. Decoding twice, or re-encoding for the clip, would waste both CPU and
memory — so a source yields `FrameData` for the pipeline *and* `EncodedPacket` for the
pre-roll buffer, from the same demuxed packet. Only packets the pipeline actually samples are
decoded; every packet is buffered.

`PreRollBuffer` is a fixed-duration ring over `EncodedPacket`, sized from
`clip_preroll_seconds`. Holding encoded packets costs ~2 MB where raw frames would cost
~340 MB. On flush it walks back to the **oldest keyframe at or before** the pre-roll horizon,
because a clip that starts mid-GOP is undecodable. The actual pre-roll is therefore quantised
up to the GOP boundary: with a typical 2 s GOP, a requested 3.0 s yields 3.0–5.0 s. That errs
toward more context, never less, and must be documented at the config flag.

`FileSource` additionally supports a `realtime` flag. Off, it yields frames as fast as
possible (deterministic, for tests). On, it paces to the file's own framerate (for the demo
path). Tests use `realtime=False` exclusively so nothing sleeps in CI.

### 5.2 Motion stage

`pipeline/stages/motion.py` computes the two cheap signals the gate consumes:

- **Motion energy** — mean absolute difference against the previous frame, normalised to
  [0, 1].
- **Scene signature** — a fixed-bin normalised histogram, which is what `signature_delta`'s
  total-variation metric expects.

This is the outermost layer, so numpy is allowed here and only here among the things the gate
touches. The bin count is a constant in this module. Because a mid-stream bin-count change is
exactly the crash the review found (B1), the pipeline must treat a change in signature length
as a stream discontinuity and reset `GateState`; the domain now returns 0.0 rather than
raising, but the streak reset belongs to the caller.

Note: `SceneState.motion_energy` is currently carried but read by no trigger. Phase 1B wires
it as a real signal but does not add a trigger for it — that is Phase 4's behaviour engine.
Spec §4.1 describes motion energy *per zone*; this slice keeps it a scalar and the field
docstring should say so.

### 5.3 Pipeline runner

One asyncio task per camera, running stages 1–5 in order: decode → detect → track → motion →
gate.

PyAV decode and YOLO inference are blocking calls. They run in a thread executor so the event
loop stays responsive; the tracker, motion stage and gate are pure CPU and run inline.

**Backpressure.** If detection cannot keep up with decode, the runner drops the stale frame
and processes the newest — surveillance wants current reality, not a delayed complete record.
Dropped-frame counts are exposed as telemetry, because silently dropping frames while
reporting healthy is exactly the failure that erodes trust in a monitoring system.

The runner never awaits the VLM. A positive gate decision is handed to the scheduler and the
loop continues immediately.

### 5.4 Orchestrator

- **`registry.py`** — model discovery and the eight §5 lifecycle states.
- **`resident_set.py`** — executes the Phase 1A `plan_residency()` output: loads, unloads, and
  applies the 600 s VLM idle-unload.
- **`scheduler.py`** — a bounded queue with **exactly one VLM worker**. One worker is correct,
  not a simplification: there is one GPU and one set of resident weights, so concurrent VLM
  calls would contend for the same VRAM. Drop-on-full is safe because the token bucket and the
  cooldown already bound the arrival rate; every drop is counted and logged.
- **`admission.py`** *(new — not in the original §6 layout)* — a global admission gate.

**Why the admission gate exists.** Phase 1A's governors — token bucket, cooldown, dedup — are
all *per camera*, but the GPU they protect is *global*. With one camera these are equivalent,
so this is not a bug today. With N cameras, N independent buckets can each legitimately permit
a call and collectively saturate the GPU. Building the seam now, while it is trivially
testable with one camera, avoids retrofitting it into the hot path in Phase 2. It is a
process-wide semaphore plus a global rate ceiling, sitting between the scheduler queue and the
VLM worker.

- **`service.py`** — the single entry point the API delegates to.

### 5.5 Clip writer

On escalation: open a PyAV MP4 muxer writing to a temp file, flush the pre-roll buffer into it
from the chosen keyframe, continue appending live packets for `clip_postroll_seconds`,
finalise, upload to MinIO, delete the temp file, return the URI. No decode, no encode — the
packets are copied through. Peak RAM is the ring buffer plus one packet.

`clip_postroll_seconds` is a new setting, defaulting to **5.0 s**. Chosen so that a clip
covers a person crossing a typical frame after the trigger, and so total clip length stays
bounded at roughly 8–10 s including the quantised pre-roll — short enough that MinIO storage
stays trivial at this scale.

A second escalation on the same camera while a clip is still recording extends the existing
clip's post-roll rather than opening a second overlapping one.

Failure to write or upload a clip must **not** lose the event. The event publishes with
`clip_uri=None` and the failure is logged — spec §9's governing rule is that an anomaly event
is never lost to an infrastructure failure.

### 5.6 Publishers

`InMemoryPublisher` for tests. `RabbitMQPublisher` publishes to the `sentinel.events` exchange
and, when the broker is unreachable, spools to `event_spool_dir` on disk and replays on
reconnect. Every payload passes through `encode_event`, which now validates against
`contracts/events/anomaly_event.schema.json` before publishing.

### 5.7 API

FastAPI, thin, delegating to `orchestrator/service.py`:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness plus per-model lifecycle state |
| `GET /cameras` | Configured cameras and their status |
| `GET /cameras/{id}/telemetry` | FPS, dropped frames, discontinuities, last escalation |
| `POST /cameras/{id}/describe` | `USER_REQUESTED` — bypasses the governors via `force()` |

VRAM is reported on `/health`, not on the per-camera telemetry endpoint: it is a property of a
model, and one GPU serves every camera, so per-camera attribution would be a fiction.

No WebSocket in this phase; the Go backend owns the WS hub in 1C.

## 6. Error handling

Implements spec §9. Governing rule: **an anomaly event is never lost to an infrastructure
failure.**

| Failure | Behaviour |
|---|---|
| Camera disconnect | Exponential backoff reconnect; status → offline; event emitted |
| Signature length change | Delta treated as 0.0, `GateState` reset — no crash (review finding B1) |
| VLM out of memory | Evict LRU → retry once → mark unhealthy → continue detection-only |
| VLM timeout | Event still created from metadata, flagged `description_unavailable` |
| Clip write/upload failure | Event published with `clip_uri=None` |
| RabbitMQ unavailable | Disk-backed spool, replayed on reconnect |
| Scheduler queue full | Escalation dropped and counted; bucket and cooldown make this rare |

## 7. The autoawq risk and its fallback

> **RESOLVED 2026-08-01 by the Task 1 spike — Branch B.** AutoAWQ installed and its model
> loaded into VRAM, but inference failed inside autoawq's own bundled Triton GEMM kernel
> against Triton 3.7.1, surviving three cheap fixes (transformers 4.49.0, transformers 4.51.3,
> explicit `torch_dtype=float16`). Task 13 therefore loads the **unquantised**
> `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint with `BitsAndBytesConfig(load_in_4bit=True,
> bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=float16)`. Measured with the desktop
> session running: 694 MiB baseline → 3359 MiB loaded → **3517 MiB peak during inference**.
> That is ~2.8 GB for the VLM against the 4.4 GB this design budgeted — see §2.1. Full record:
> `docs/superpowers/plans/phase1b-spike-result.md`.
>
> The section below is retained as written because it is why the spike existed and why the
> outcome cost a config change rather than a redesign.

The spec mandates `transformers` + `autoawq`. AutoAWQ was deprecated by its maintainer in 2025
and is fragile against torch ≥ 2.5 / transformers ≥ 4.49 — the versions pinned in
`pyproject.toml`. `torch` is not currently installed on the target machine, so this is unproven.

**Task 1 of the plan is a throwaway spike:** install the `gpu` extra and load
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ` on the 4060. Nothing else depends on its *code* — only on
its answer.

- **If it loads:** proceed exactly as specced.
- **If it fails:** fall back to `transformers` + `bitsandbytes` 4-bit NF4. Comparable VRAM
  (~3.5 GB), actively maintained. The `VisionLanguageModel` port is unchanged either way, so
  this is an adapter-internal swap, not a redesign — which is precisely what the port
  abstraction was for. Record the deviation in the plan and update the §4.3 VRAM figure.

The spike is throwaway: its output is a decision and a documented VRAM measurement, not code.

## 8. Test footage

Per spec §5, the deterministic tier is looped video files via mediamtx, and the labelled tier
names CUHK Avenue — which is already the `avenue_01` path in the spec's own mediamtx example.

`datasets/` is gitignored except for `*.sh`, `*.md` and `.gitkeep`, so the repo carries a
**download script and a licence note**, never the footage. The script must state the source and
licence before downloading, and the plan task must verify the licence permits research use
before the URL is hardcoded.

Two tiers of asset:

- **Synthetic, committed:** an ffmpeg-generated clip of moving shapes, a few hundred KB, for
  pre-roll-buffer and pipeline tests. Deterministic, no licence questions, no network in CI.
- **Real, downloaded:** the public sample, for GPU verification of YOLO11s and Qwen.

Spec §5's ethical constraint carries forward unchanged: public feeds only, nothing that indexes
unsecured private cameras.

## 9. Testing

| Layer | Approach | GPU |
|---|---|---|
| Sources, pre-roll buffer | Real PyAV against the committed synthetic clip | no |
| Motion stage | Table-driven over synthetic frames | no |
| Pipeline runner | Phase 1A fakes; full file → gate → event → publisher trace | no |
| Orchestrator, admission gate | Fakes, simulated clocks | no |
| Publishers | `InMemoryPublisher`; RabbitMQ marked `integration` | no |
| Clip writer | Real PyAV encode to temp file; MinIO marked `integration` | no |
| YOLO11s, ByteTrack, Qwen | `@pytest.mark.gpu`, local only | yes |

The end-to-end CPU trace is the phase's keystone test: it proves the whole slice composes
without a GPU, which is what keeps CI meaningful.

**Contract tests.** Each real adapter must pass the same port-contract tests its fake passes.
The fakes are now type-checked under mypy, so they stay honest reference implementations.

## 10. Task sequencing

Vertical slice first, then substitute real components one at a time. Every task leaves the
system working, and each real adapter is verified against a known-good baseline rather than
being debugged simultaneously with everything else.

| # | Task | GPU |
|---|---|---|
| 1 | ~~autoawq spike~~ — **done**, resolved to bitsandbytes NF4 | yes |
| 2 | Compose core services + mediamtx config | no |
| 3 | `ClipWriter` port change + fake update | no |
| 4 | `FileSource` + `PreRollBuffer` + synthetic clip | no |
| 5 | Motion stage | no |
| 6 | Pipeline runner, end-to-end with fakes | no |
| 7 | Orchestrator: registry, resident set, scheduler, admission gate | no |
| 8 | `InMemoryPublisher` + disk-buffered `RabbitMQPublisher` | no |
| 9 | MinIO clip writer | no |
| 10 | FastAPI surface | no |
| 11 | YOLO11s adapter | yes |
| 12 | ByteTrack adapter | no |
| 13 | Qwen2.5-VL adapter | yes |
| 14 | `RtspSource` + dataset download script + end-to-end demo | yes |

Tasks 1–10 are CI-green on CPU. Tasks 11–14 are where the GPU and the real weights enter.

## 11. Risks

| Risk | Mitigation |
|---|---|
| ~~autoawq will not install~~ | **Materialised.** The spike caught it before any adapter code existed; cost was a `pyproject.toml` line, not a redesign. Retired as a risk. |
| 15 GB RAM with ~7 GB free | Encoded pre-roll and streaming clip encode keep peak RAM low; compose services can run selectively |
| VRAM margin | Measured better than budgeted: the VLM costs ~2.8 GB, not 4.4 GB (§2.1). The 600 s idle-unload reclaims it between events, and the planner evicts if a future model tightens things again. |
| Qwen latency exceeds the escalation rate | Bounded queue with counted drops; the cooldown widens the gap between calls |
| Pre-roll quantised to GOP boundary | Documented; errs toward more context, never less |
| RTSP timing bugs surface late (task 14) | `FileSource` shares the `PreRollBuffer` and decode path, so only reconnect logic is genuinely new |

## 12. Decisions log

| Decision | Rationale |
|---|---|
| Vertical slice before real models | Every task keeps a working system; real adapters get a known-good baseline |
| autoawq spike as task 1 | Throwaway, minutes long, and its answer shapes the VLM adapter |
| `ClipWriter` → open/append/finish | ~1.1 GB/clip otherwise, on a box with ~7 GB free |
| Clips are remuxed, never re-encoded | Cheapest CPU and memory path, and keeps evidence bit-identical to what the camera sent |
| Sources emit frames *and* packets | One demux pass serves both inference and the clip; decoding twice would waste CPU and RAM |
| `detect` stays per-frame | Single-camera slice; batching would be untested unused code |
| Queue + one VLM worker | One GPU, one set of weights; concurrent calls would contend for VRAM |
| Global admission gate now | Per-camera governors cannot bound a global GPU; trivial to test with one camera |
| Encoded pre-roll packets | ~2 MB vs ~340 MB; GOP quantisation is an acceptable price |
| Drop stale frames under backpressure | Surveillance wants current reality, not a delayed complete record |
| Public sample via download script | Keeps footage out of git; licence verified before the URL is hardcoded |
