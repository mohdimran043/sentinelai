# Architecture

## The central bet

SentinelAI runs expensive models on a single consumer GPU. Almost every
interesting decision the system makes — *is this scene worth describing? does
this model still deserve its VRAM? may this camera call the GPU right now?* — is
a decision about spending a scarce resource.

The bet is that **those decisions can be pure functions**, and therefore tested
exhaustively on a CPU in CI, in milliseconds, with no GPU and no model weights.

That bet is why `domain/` and `ports/` are pure, and it pays: the full CPU suite
runs **896 tests in a few seconds** (`make test`). The escalation gate, the
VRAM planner, the token bucket, the welfare routing rule and all seven triggers
are covered without CUDA ever initialising.
The 19 tests that genuinely need a GPU or running infrastructure are marked
`gpu` / `integration` and deselected.

## Layers

Dependencies point inward only. Nothing in an inner layer knows an outer layer
exists.

```
        ┌──────────────────────────────────────────────────┐
        │ main.py — composition root                       │  builds everything
        ├──────────────────────────────────────────────────┤
        │ api/         FastAPI routes                      │
        │ pipeline/    CameraRunner, motion, behaviour     │  impure: I/O, clocks,
        │ orchestrator/ registry, resident set, scheduler, │  frameworks, GPU
        │              alert register, priority queue      │
        │ adapters/    YOLO11(+pose), Qwen2.5-VL, PyAV,    │
        │              InsightFace, MinIO, RabbitMQ,       │
        │              ByteTrack, encrypted face store     │
        ├──────────────────────────────────────────────────┤
        │ ports/       abstract interfaces — the seams     │  PURE
        │ domain/      entities, behaviour/, policy/       │  PURE
        └──────────────────────────────────────────────────┘
```

### What "pure" means here, exactly

`domain/` and `ports/` may not:

- import third-party I/O or model libraries — `torch`, `cv2`, `av`, `numpy`,
  `ultralytics`, `transformers`, `fastapi`, `minio`, `aio_pika`, and others;
- import `time`, `datetime`, `random`, `os`, or `importlib` **at all** (the
  module is banned, not just the call, so aliases and `from`-imports cannot
  sneak through);
- call anything that reads a clock — `monotonic`, `perf_counter`, `now`,
  `utcnow`, `sleep`, and the rest;
- import dynamically (`importlib.import_module`, `__import__`);
- import `adapters`, `orchestrator`, `pipeline`, `api`, **or `config`**.

`config` is on that list because `Settings()` reads `.env` from disk. Letting
policy import it would make the escalation gate depend on process environment.

Time is always an **argument**. `decide(scene, profile, state)` takes `now` from
`scene.timestamp`; `plan_residency(...)` takes `now=`. That is what makes
"what happens 601 seconds later" a one-line test instead of a sleep.

The consequence you will meet first: `FrameData.pixels` is typed `object`, not
`np.ndarray`, because typing it properly would drag numpy into `ports/`. The
adapters `isinstance`-check it at the boundary.

### The rule is enforced mechanically

`ai-engine/tests/test_architecture.py` parses every module in `domain/` and
`ports/` with `ast` and fails the build on any violation. It is an AST walk, not
a substring grep — prose in a docstring mentioning `sentinel_ai.adapters` is
documentation, and a relative `from ..adapters import x` is a violation even
though that string never appears.

**All of it has been proven to fail when violated.** The same file carries 16
known-bad module sources — an aliased `import time as t`, a
`from ...adapters import`, an `importlib.import_module("torch")`,
`from sentinel_ai.config import get_settings` — and asserts the detectors flag
each one, plus one clean module asserting no false positives. 24 tests in that
file. A fitness function nobody has watched fail is a hypothesis, not a guard.

The suite also raises if a pure layer is missing or empty, so the checks can
never pass vacuously against zero files.

**The `clock-read` matcher is pinned too, and that took a second attempt.** The
first fifteen known-bad cases all expected a `forbidden-import:*`,
`dynamic-import` or `outer-layer:*` offence, and the assertion is an
`any(startswith(...))` — so on every "clock" case the *import* ban satisfied it
and `_clock_reads` was never the thing under test. Meanwhile
`test_pure_layers_do_not_read_the_clock` scans real modules with zero
violations, so it passed vacuously. Replacing `_clock_reads`'s body with
`return []` survived the entire suite.

The sixteenth case closes it: `import asyncio` plus `await asyncio.sleep(1.0)`,
expecting `clock-read:asyncio.sleep`. `asyncio` is deliberately *not* on the
forbidden-import list — the pure layers may not read a clock but may be async —
so it is the one case whose **only** available offence is the clock read. The
same mutation now fails exactly that case and nothing else.

Worth keeping in mind when adding a control: a positive control that trips two
detectors pins neither of them.

## The ports

Eleven abstract interfaces in `ai-engine/sentinel_ai/ports/`. Everything
replaceable is replaceable through one of them.

| Port | File | Contract |
|---|---|---|
| `FrameSource` | `frame_source.py` | Async-iterates `FrameData`; `packets()` fans the *same demux pass* out as still-encoded `EncodedPacket`s |
| `ObjectDetector` | `detector.py` | `detect(frame) -> tuple[Detection, ...]`, per-frame |
| `Tracker` | `tracker.py` | `update(detections, timestamp) -> tuple[Track, ...]`. **Synchronous** — association is cheap CPU work |
| `VisionLanguageModel` | `vision_llm.py` | `describe(VisionRequest) -> SceneDescription` |
| `EventPublisher` / `FailedEventSink` | `event_publisher.py` | `publish(event)`. Must **raise** on failure, never swallow |
| `ClipWriter` / `ClipHandle` | `clip_writer.py` | `open()` returns a handle; `append`/`finish`/`abort` stream packets |
| `Notifier` | `notifier.py` | `notify(WelfareNote)`. Must **never raise** — the exact opposite of `EventPublisher` |
| `PoseEstimator` | `pose.py` | `estimate(frame, tracks) -> Mapping[int, PersonPose]`, keyed by track id. Loaded only where a capability needs it |
| `FaceDetector` / `FaceEmbedder` | `face.py` | Find faces, turn one into a 512-d embedding. Two ports, one adapter — see below |
| `FaceStore` | `face.py` | Persist and read enrolled people and their sealed embeddings |
| `ModelRuntime` | `model_runtime.py` | Lifecycle: `initialize`, `warmup`, `predict`, `shutdown`, `mark_unhealthy`, `health`, `version`, `capabilities` |

Four shapes worth understanding:

**`EventPublisher` must raise; `Notifier` must never.** They look alike — both
take one record and send it somewhere — and their failure contracts are
opposites. A publisher that swallows an error loses an event that nothing else
will ever write down, so `publish()` raising is what hands it to
`FailedEventSink`. A notifier is best-effort commentary on a pipeline that has
already published; one that raised would take down the thing it exists to
observe. Implement the wrong one of these and the failure is silent in both
directions.

**`FrameSource` is dual-stream.** One demux pass produces both decoded frames
(for detection, sampled) and encoded packets (for the pre-roll buffer and clip
writer, *every* packet). This is what lets a clip be a remux rather than a
re-encode — see [decisions](decisions.md#3-clips-are-remuxed-never-re-encoded).

**`ModelRuntime` is separate from the capability ports.** `Yolo11Detector`
implements both `ObjectDetector` and `ModelRuntime`; `Qwen25VLDescriber`
implements both `VisionLanguageModel` and `ModelRuntime`. The capability port
says *what the model does*; `ModelRuntime` says *how the orchestrator owns its
VRAM*. A model that the registry should load, warm, health-check and evict needs
both. One that manages its own lifetime needs only the capability port.

**`FaceDetector` and `FaceEmbedder` are separate ports that one adapter implements.**
Detection and embedding are genuinely different capabilities — a deployment could detect
faces without ever comparing them — but InsightFace produces both from a single
`FaceAnalysis.get()` call, and running it twice to honour the separation would double
the cost for nothing. Splitting the ports keeps the *substitution* open; sharing the
adapter keeps the *frame* cheap. `FaceStore` is a third port rather than part of either,
because where biometric data is kept is a decision about storage and encryption, not
about vision.

`mark_unhealthy` is abstract rather than a defaulted no-op on purpose: a model
can see its own load failures, but "this model has OOMed on two consecutive
describes" is knowledge the *scheduler* holds. A silently unimplemented
`mark_unhealthy` would leave `/health` reporting a model as fine after the
orchestrator had given up on it.

## The two model-runtime transports

`SENTINEL_MODE` (`config.py`) is the **only** place this seam is chosen.

- **`development`** (default) — models are in-process Python objects. The
  orchestrator calls `ModelRuntime` methods directly.
- **`production`** — intended to bind a gRPC transport so models run out of
  process.

**Production is not built.** Setting `SENTINEL_MODE=production` today changes
nothing, because nothing downstream branches on it yet. The seam exists so that
adding it later is a change in one module rather than a rewrite; treat the
setting as reserved, not functional.

## Data flow: frame to published event

One asyncio task per camera (`pipeline/runner.py`, `CameraRunner`).

1. **Demux.** `FrameSource` runs one PyAV pass. Decoded frames go one way,
   encoded packets the other.
2. **Backpressure.** Frames land in a `_LatestSlot` — a single-slot mailbox that
   *overwrites*. An overwrite before the consumer collects is a drop, counted in
   `CameraTelemetry.frames_dropped`. A naive `async for frame: await detect()`
   loop never drops anything and therefore produces a delayed complete record,
   where surveillance wants current reality.
3. **Detect.** `ObjectDetector.detect()` — one shared `Yolo11Detector` behind an
   `asyncio.Lock` ([why](decisions.md#4-one-shared-detector-behind-a-lock)).
   Results are computed once and reused by tracking, the gate, the VLM and
   reporting; detection is never re-run per consumer.
4. **Track.** `Tracker.update()`, inline, synchronous.
5. **Motion.** `MotionAnalyzer` produces `motion_energy` and a normalised
   `scene_signature` histogram.
6. **Behaviour.** For cameras that enabled a capability needing it, `PoseEstimator`
   runs and its skeletons are attributed to tracks by IoU. A `BehaviourObservation`
   — `SceneState` plus poses plus frame dimensions — goes to the four pure state
   machines in `domain/behaviour/`: falls, abandonment, tamper, zones. Each returns
   `(next_state, candidates)`; the highest-priority candidate escalates directly.
7. **Authorize.** Then, for cameras that enabled `person_authorization`, faces are
   detected, embedded and compared against the sealed roster. An unrecognised person
   must persist across `min_observations` frames and `min_duration_seconds` before
   anything is raised.

   Both steps run **before** the gate and return early when they fire. A completed
   temporal signature is already deduplicated to one report per episode by its own
   state machine, so the gate's three governors have nothing left to protect against —
   and a cooldown window swallowing a suspected fall is the failure the subsystem
   exists to prevent. They are ordered falls-before-strangers because only one
   escalation per frame is possible (the keyframe is shared) and a person on the floor
   outranks a person who is merely unrecognised.
8. **Gate.** A `SceneState` — deliberately carrying **no pixels** — goes to the
   pure `decide()`. It returns a decision *and the next state*, so the runner
   threads state through frames and the gate never holds mutable state or reads
   a clock.
9. **Escalate.** If the gate says yes, an `EscalationRequest` is submitted to
   `VlmScheduler`. Submission is synchronous and drops when full, so a stalled
   VLM can never back-pressure the camera.
10. **Clip.** In parallel, the packet loop keeps a `PreRollBuffer` ring. On
   escalation it opens a `ClipHandle`, which continues receiving packets until
   the post-roll deadline, then remuxes to MP4 and uploads to MinIO.
11. **Describe.** The scheduler passes `AdmissionGate` (process-wide GPU
   concurrency + minimum interval), calls `ResidentSet.ensure()` to load the VLM
   if it was idle-evicted, and runs `describe()`.
12. **Publish.** `VlmScheduler._assemble()` builds the `Event` — the one place in
    the codebase an `Event` is constructed — stamping `occurred_at` in Unix epoch
    seconds via the per-camera anchor. `event_codec` validates it against
    `contracts/events/anomaly_event.schema.json`, then `EventPublisher.publish()`
    sends it. A publisher that raises hands the event to `FailedEventSink`.
13. **Notify.** Last, and only after the publish: if the VLM reported a welfare
    concern this camera routes, a `WelfareNote` is handed to `Notifier` — by
    `put_nowait` onto a separate worker, never awaited here, so a hanging
    webhook cannot hold the GPU admission slot the step above is still inside.
    Deliberately runs even when the publish fell back to the dead-letter sink: a
    broker outage is the case *most* worth reaching a person over. See
    [operations](operations.md#welfare-notifications).

### The three governors

The gate evaluates in a fixed order, and the order is load-bearing
(`domain/policy/escalation.py`):

1. `auto_escalation_enabled` — a disabled camera short-circuits everything
2. **triggers** — is anything worth looking at?
3. **cooldown** — is it simply too soon since the last call?
4. **duplicate suppression** — have we already described this exact scene?
   (scene-signature distance below `DEDUP_EPSILON = 0.05`)
5. **rate budget** — a `TokenBucket`. Can we afford it?

The bucket is last because it is the only stage that *consumes* anything;
neither cooldown nor dedup may run after it and throw a spent token away.
Cooldown precedes dedup because it is an unconditional temporal floor.

Every suppression still reports the trigger reason, so telemetry shows what the
gate declined and why.

### The escalation gate

The seven triggers live in `domain/policy/triggers.py`, each a pure predicate of
a `TriggerContext`:

| Trigger | Fires when |
|---|---|
| `speed_anomaly` | a salient track exceeds the speed threshold |
| `dwell_exceeded` | a track stays within `dwell_radius_px` of its anchor for `dwell_seconds` |
| `track_count_spike` | salient track count exceeds `track_count_baseline` |
| `scene_change` | signature delta ≥ threshold, sustained `scene_delta_frames` frames |
| `new_salient_track` | a salient-class track reaches `min_track_frames` |
| `periodic_summary` | `summary_interval_seconds` since the last summary |
| `user_requested` | not a predicate — `force()`, from `POST /cameras/{id}/describe` |

They are evaluated in exactly that order, and the first to fire names the
reason. `ALL_TRIGGERS` is the tuple that encodes it.

### Two clocks, deliberately

`CameraRunner` has **no clock of its own**. Every time-valued decision reads the
frame or packet that occasioned it. `VlmScheduler` *does* hold a real clock,
because `AdmissionGate` spaces admissions with `asyncio.sleep`, which runs on
wall time.

Conflating them is a real bug that was hit: a runner clock reading ahead of a
`FileSource` timeline (which starts at 0.0) makes `TokenBucket.refilled`'s
regressing-clock guard turn every refill into a no-op. The camera spends its
burst and is rate-limited into silence for the life of the process.

This is also why an `Event` carries two timestamps — see
[decisions](decisions.md#5-occurred_at-is-unix-epoch-with-a-per-camera-anchor).

## VRAM residency

`domain/policy/vram_budget.py` is pure and decides *what* to load and unload;
`orchestrator/resident_set.py` holds the bookkeeping and performs the I/O.

`plan_residency()` takes the budget, the currently-resident set, the required
set, and last-used times. It idle-evicts first (freeing memory nobody asked
for), then for each required model evicts **LRU among lower-or-equal-priority,
non-required models** until it fits, raising `InsufficientVram` if it cannot.

Priority is `ModelSpec.priority`, higher survives. The always-on detector
outranks the VLM. `idle_unload_seconds=None` means never idle-evict.

Because residency is a function of the budget, the same planner serves an 8 GB
laptop GPU and a 24 GB 4090 with no change.

## The contract seam

`contracts/` is the only coupling between `ai-engine/` and `web/`. The UI
generates its types from it and must never import from `ai-engine/`. There is a
drift test. See [contracts/README.md](../contracts/README.md).
