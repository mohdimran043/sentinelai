# Architecture decision record

These are the decisions a newcomer would otherwise re-litigate, or undo without
realising what they were for. Several were expensive to learn.

**Format.** Each entry states what was decided, why, what it cost, and what
would change it. New decisions go at the bottom with the next number. Never
delete an entry — supersede it, and say which one it replaces. If a decision
turns out to rest on a wrong number, correct the entry rather than quietly
patching the code; ADR 4 is an example of exactly that.

| # | Decision | Status |
|---|---|---|
| [1](#1-bitsandbytes-nf4-not-autoawq) | bitsandbytes NF4, not AutoAWQ | Accepted |
| [2](#2-cpu-decode-not-nvdec) | CPU decode, not NVDEC | Accepted |
| [3](#3-clips-are-remuxed-never-re-encoded) | Clips are remuxed, never re-encoded | Accepted |
| [4](#4-one-shared-detector-behind-a-lock) | One shared detector behind a lock | Accepted, rationale corrected |
| [5](#5-occurred_at-is-unix-epoch-with-a-per-camera-anchor) | `occurred_at` is Unix epoch with a per-camera anchor | Accepted |
| [6](#6-domain-and-ports-are-pure-and-a-test-enforces-it) | `domain/` and `ports/` are pure, and a test enforces it | Accepted |
| [7](#7-the-escalation-gate-exists-at-all) | The escalation gate exists at all | Accepted — this is the thesis |
| [8](#8-contracts-is-the-only-engine--ui-coupling) | `contracts/` is the only engine ↔ UI coupling | Accepted, partly unenforced |
| [9](#9-a-file-for-cameras-environment-variables-for-scalars) | A file for cameras, environment variables for scalars | Accepted |

---

## 1. bitsandbytes NF4, not AutoAWQ

**Decided.** The VLM loads the *unquantised* `Qwen/Qwen2.5-VL-3B-Instruct`
checkpoint through plain `transformers`, quantised at load time with
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)`.
AutoAWQ is not a dependency.

**Why.** AWQ was the design's first choice, and it was tried properly. It
failed. Four escalating attempts produced four *different* failures:

1. **As installed** — current `transformers` has removed AutoAWQ as a supported
   AWQ backend entirely; `AwqQuantizer.validate_environment()` now
   unconditionally demands `gptqmodel`.
2. **`transformers==4.49.0`** (the project's floor) — `autoawq` 0.2.9's model
   registry unconditionally imports Qwen3 modeling code that does not exist in
   4.49: `ModuleNotFoundError: No module named 'transformers.models.qwen3'`.
3. **`transformers==4.51.3`** — the version AutoAWQ's own deprecation banner
   names as its last tested configuration. The model *loads* (4249 MiB).
   `generate()` fails on a Triton dtype mismatch between AWQ's fp16 scales and
   auto-resolved bf16 activations.
4. **Same, plus explicit `torch_dtype=torch.float16`** — loads (4279 MiB),
   clears every decoder layer, and dies at the final `lm_head` projection
   **inside AutoAWQ's own bundled Triton GEMM kernel**, with a nonsensical type
   error between two float16 operands.

AutoAWQ ships no compiled CUDA extension wheel for this torch/CUDA build, so it
falls back to that broken Triton path. This is a hard unmaintained-library
incompatibility — AutoAWQ was deprecated by its maintainer in 2025 — not a
configuration or pinning problem.

The NF4 path was then verified end to end on the target GPU and produced a real
caption. Full transcript: `docs/superpowers/plans/phase1b-spike-result.md`.

**What it cost.** Very little, which is the surprise. Measured on an RTX 4060
laptop (8188 MiB, desktop session live):

| | AWQ (Branch A) | NF4 (Branch B) |
|---|---|---|
| Baseline | 694 MiB | 694 MiB |
| After load | 4279 MiB | **3359 MiB** |
| Inference peak | crashed | **3517 MiB** |
| Inference wall time | — | 2.66 s / 64 new tokens |

NF4 used *less* VRAM than the AWQ attempt and is actively maintained.

**A trap to know about.** `SENTINEL_VLM_MODEL_ID` still defaults to
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`. That suffix names the model *family* and is
**stripped by `_base_checkpoint()` before loading** — the AWQ repo is never
fetched. `version()` and `capabilities().model_key` deliberately report the
checkpoint actually loaded, not the configured id, so an operator reading
`/health` cannot wrongly conclude AWQ is active. Do not "fix" the config
default by pointing it at a real AWQ repo; that would change nothing about what
loads, and would make the id lie.

**What would change it.** AutoAWQ gaining a maintained release that works
against current Triton, or a move to a serving runtime (vLLM, TensorRT-LLM)
that brings its own quantisation. The `VisionLanguageModel` port exists so that
swap is one adapter, not a rewrite — which is precisely why the port
abstraction was worth having before it was needed.

---

## 2. CPU decode, not NVDEC

**Decided.** Video decode runs on the CPU through PyAV. No NVDEC, no hardware
decode surface.

**Why.** VRAM is the binding constraint on this hardware; CPU is not. The box
has 20 cores that comfortably absorb software decode at this scale, while a
hardware decode surface would consume VRAM that the models need. Trading an
abundant resource for a scarce one is the right direction.

**What it cost.** CPU headroom, which is available, and a ceiling on camera
count that is higher than the VRAM ceiling anyway. At Phase 1B's single-camera
scale this is not close to binding.

**What would change it.** Many more cameras per box, or a deployment where CPU
is the scarce resource. `SENTINEL_DECODE_HWACCEL` exists as the escape hatch
and is currently **inert** — no source adapter branches on it. It is a
placeholder that makes re-enabling hardware decode a configuration change
rather than a rewrite; do not assume setting it does anything today.

---

## 3. Clips are remuxed, never re-encoded

**Decided.** `MinioClipWriter` copies the camera's already-encoded H.264/HEVC
packets into an MP4 container. There is no decode and no encode anywhere in the
clip path.

**Why.** Three reasons compound. Re-encoding costs GPU or CPU at exactly the
moment the system is already busy escalating. It degrades the picture. And it
means the evidence is no longer what the camera sent — for a surveillance
product, evidence that is bit-identical to the source is a property worth
protecting.

This is also why `FrameSource` is dual-stream: one demux pass fans out decoded
frames *and* still-encoded packets, so the pre-roll ring buffer holds encoded
bytes and the clip writer has something to remux.

**What it cost.** Real complexity, most of it in recovering information the
port deliberately does not carry. `ClipHandle.open()` receives only
`camera_id`, `event_id` and `fps`; a packet carries bytes, a pts, a keyframe
flag and a codec string. Width, height and the SPS/PPS an MP4 muxer needs for
its `stsd` box are all absent, and are recovered from the bytes themselves —
dimensions from an `av.CodecContext` *parser* (not a decoder), extradata from a
separate `extract_extradata` bitstream filter.

Two specific costs worth knowing:

- **Pre-roll is a floor, not an exact figure.** A clip cannot start mid-GOP, so
  `PreRollBuffer.flush()` walks back to the oldest keyframe at or before the
  horizon. A requested 3.0 s pre-roll yields 3.0–5.0 s at a 2 s GOP. Always
  more, never less.
- **RTSP needed a fix that MP4 did not.** RTP H.264 (RFC 6184) carries SPS/PPS
  out-of-band in the SDP's `sprop-parameter-sets`, negotiated once at session
  setup, whereas MP4-sourced Annex-B repeats them before every keyframe. The
  bitstream filter was built against the latter. Until `RtspSource`
  (`_annexb_keyframe_bytes`) stitched the SDP-derived extradata back onto every
  keyframe, **every clip from a real RTSP camera came back an empty file**.
  Found during a live demo, not in review.

### The monotonic-DTS guard

Inside `_RemuxSession._mux`, a non-increasing tick is nudged to `last + 1`:

```python
if self._last_ticks is not None and ticks <= self._last_ticks:
    ticks = self._last_ticks + 1
```

This is not defensive padding. It fixes a **1-in-5 clip loss**. A non-increasing
DTS makes ffmpeg's mov muxer return `EINVAL`, surfacing as
`av.error.ArgumentError: ... returned 22`. It is not a rare race: on RTSP,
`au_pts` is packet *arrival* time, and a TCP-interleaved socket delivers a burst
after any stall, so two access units routinely land inside one 11.1 µs tick and
`round()` maps them to the same integer.

Nudging rather than dropping, because one tick is 1/90000 s — three orders of
magnitude below a frame at any framerate in scope, invisible on playback —
whereas dropping would silently delete evidence, and a burst is exactly the
moment something is happening. It only ever moves a timestamp forward, so the
rebased zero point and the clip's overall span are untouched.

**Do not remove this guard because the arithmetic looks like it cannot happen.**
It happened, in production, one clip in five.

**What would change it.** A source delivering a codec outside
`SUPPORTED_CODECS` (`h264`, `hevc`), or a requirement to burn overlays into the
evidence — either would force a decode/encode path, and both should be resisted.
A codec outside the set raises `UnsupportedCodecError` on the very first packet
rather than finalising an empty MP4, deliberately: a clip writer that fails
silently is worse than one that fails, because the operator only finds out when
they go looking for evidence that was never written.

---

## 4. One shared detector behind a lock

**Decided.** `compose()` builds exactly **one** `Yolo11Detector` and hands it to
every `CameraRunner`. It serialises its own access with an `asyncio.Lock` held
across the executor offload.

**Why the lock is mandatory.** `YOLO.predict` keeps per-call state on
`self.predictor` (`batch`, `results`, `source`); Ultralytics documents one model
instance per thread for this reason. Unsynchronised, two cameras' calls
interleave on that shared state and **one camera's detections are attributed to
the other** — silently, because both calls still return a well-formed result.

**Why the sharing, and the corrected rationale.** The original argument was that
a second detector would not fit in the VRAM budget. **That argument is wrong**,
and it is worth understanding why, because it is the kind of error that
propagates.

It came from spec §4.3's *design table* (900 MiB detector, 4400 MiB VLM). The
**measured** figures on the RTX 4060 are 432 and 2766. Against 6144 usable
(8192 total − 2048 reserved):

- design table: `4400 + 2*900 = 6200` → does not fit
- measured: `2766 + 2*432 = 3630` → fits, leaving 2514 MiB free

`InsufficientVram` would not appear until around eight cameras. A second
detector would comfortably fit.

The argument that does carry: **there is one GPU, so the kernels serialise
whatever we do.** A lock makes that explicit at zero VRAM cost. Per-camera
weights buy no parallelism on a single card and scale VRAM linearly in cameras.

**What it cost.** Latency under contention — which the pipeline already
absorbs, because a camera waiting on the lock simply drops staler frames in its
`_LatestSlot` mailbox, which is what that mailbox is for. A dedicated
single-thread executor was the other candidate; it buys the same mutual
exclusion but adds an executor to create and tear down on every
initialize/shutdown cycle, and with the lock held the model is already only ever
touched by one thread at a time.

**What would change it.** Multiple GPUs — at which point per-device detectors
buy real parallelism and the argument inverts.

**The lesson.** Configured numbers in `config.py` that carry a "measured"
docstring are measurements. Design-table numbers are estimates. Reasoning from
the wrong one produced a correct decision supported by a false premise, which is
worse than it sounds: the false premise would have blocked a legitimate change
later.

---

## 5. `occurred_at` is Unix epoch with a per-camera anchor

**Decided.** `Event` carries **two** timestamps, and they are not
interchangeable:

- **`occurred_at`** — Unix epoch seconds, UTC, fractional. The field a consumer
  sorts and displays on. Survives restarts; means the same thing for an RTSP
  camera and a replay camera in the same process.
- **`source_timestamp`** — the raw source timeline. `time.monotonic()` for live
  RTSP, seconds-from-start-of-file for a replay. This is the timeline a clip's
  pts and `CameraTelemetry` are on, so it is what correlates an event with its
  evidence. Comparable **only** within one process run for one camera. Never
  sort on it.

Neither is computed in `domain/` — the domain reads no clock, so both arrive
already calculated at `VlmScheduler._assemble()`, the one place an `Event` is
constructed.

**Why epoch.** `occurred_at` used to be `scene.timestamp` verbatim. A live run
published `"occurred_at": 51181.128868795` — a monotonic reading. That value
resets on every restart, shares no origin with a replay camera in the same
process, and cannot be turned into a wall time by a consumer, while the codec's
own docstring tells the Go consumer to sort by it.

**Why an anchor rather than `time.time()` per event.** Sampling the wall clock
per event makes `occurred_at` a fresh reading of a clock that `ntpd`, `chronyd`
or a hypervisor can step backwards at any moment. Two events a second apart
could then land out of order — precisely the ordering the consumer is told to
rely on. Offsetting from one anchor makes the difference between any two
`occurred_at` values from the same camera *exactly* the difference between their
source timestamps: absolute accuracy is that of the anchor, ordering is that of
the source's own clock.

**Why the anchor is per camera, and not per process.** This is the expensive
part. The first fix anchored the whole process once at construction, to
`(wall_clock(), clock())`, rebasing every event as
`wall_at_anchor + (source_timestamp - mono_at_anchor)`. That is exact only when
a camera's `source_timestamp` shares `clock`'s timeline — true for `RtspSource`
(monotonic), **false for `FileSource`**, whose frames carry container pts
starting at 0.0 per file. `time.monotonic()` counts from boot, so a replay
event's `occurred_at` came out roughly `wall_at_anchor - mono_at_anchor` —
**about 14 hours in the past** on the box it was found on. A replay camera and an
RTSP camera in the same process landed hours apart for events observed seconds
apart, with both anchors read correctly.

The fix is to stop assuming a shared timeline at all. Each camera's anchor is
established lazily from *that camera's own* first-seen `source_timestamp`,
paired with the wall clock read at that moment, cached in `_camera_anchors`.

**What it cost, stated honestly.** A camera's first event is anchored at that
event's assembly, not at a shared moment, so comparing `occurred_at` *across two
cameras* is precise only to when each camera's anchor was established — not to
true simultaneity. Typically well under a second. It degrades gracefully: never
off by more than each camera's own anchoring delay, never by an unrelated
system's uptime.

**Known drift — the contract is stale here.** `contracts/events/anomaly_event.schema.json`
still describes `occurred_at` as "derived from a single wall+monotonic anchor
taken once per engine process". That describes the *superseded* per-process
design, not the code. The code is correct; the schema prose is out of date. The
OpenAPI drift test does not cover the event schema, and prose descriptions are
not machine-checkable, so nothing caught it. **Fix the description before Phase
1C's Go consumer is written against it.**

**What would change it.** A source that carries true wall-clock timestamps
(many IP cameras do, via RTCP sender reports) would let `occurred_at` be read
directly rather than anchored, removing the cross-camera imprecision entirely.

---

## 6. `domain/` and `ports/` are pure, and a test enforces it

**Decided.** No third-party I/O libraries, no clock reads, no dynamic imports,
and no dependency on `adapters`, `orchestrator`, `pipeline`, `api` or `config`
in the two inner layers. `ai-engine/tests/test_architecture.py` enforces it by
AST walk.

**Why this is not style.** It is what makes the escalation gate and the VRAM
planner testable on CPU in CI without a GPU — the project's central bet. The
full CPU suite is **896 tests in a few seconds**. "What does the gate do 601
seconds later, with a spent bucket, on a camera whose signature bin count just
changed?" is a one-line test because `now` is an argument, not a clock read.
Without the rule those become integration tests with sleeps, or they do not get
written.

`config` is on the forbidden list specifically because `Settings()` reads `.env`
from disk. Importing it from `domain/` would make policy depend on process
environment.

**How it is enforced, and why you should trust it.** The detectors ban the
*module* (`time`, `datetime`, `random`, `os`, `importlib`), not just the call, so
`import time as t`, `from time import monotonic` and
`datetime.datetime.now()` are all unrepresentable rather than merely
undetected. A separate matcher catches clock reads by trailing identifier, and
another catches `importlib.import_module` / `__import__`. Import analysis is an
AST walk, so prose in a docstring mentioning `sentinel_ai.adapters` is
documentation while a relative `from ..adapters import x` is a violation.

**The positive controls are the point.** The same file carries 16 known-bad
module sources — the aliased clock read, the `perf_counter`, the relative
outer-layer import, the `from sentinel_ai.config import get_settings`, the
`import numpy` — and asserts the detectors flag each one, plus one clean module
asserting there are no false positives. 24 tests. It also raises if a pure layer
is missing or empty, so the checks cannot pass vacuously against zero files. A
fitness function that has never been observed to fail is a hypothesis, not a
guard.

**The sixteenth control is the interesting one, and it is worth knowing why it
exists.** For a long time the `clock-read` matcher was unpinned: every "clock"
known-bad case *also* imported a banned module, the assertion is an
`any(startswith(...))`, so `forbidden-import:time` satisfied it and the clock
matcher was never under test. The real-module scan passed vacuously alongside
it, having no violations to find. Gutting `_clock_reads` to `return []` survived
the whole suite — that mutation was run, and it confirmed the hole.

The fix was one entry: `import asyncio` plus `await asyncio.sleep(1.0)`,
expecting `clock-read:asyncio.sleep`. `asyncio` is deliberately *not* on the
forbidden-import list — the pure layers may not read a clock, but they are
allowed to be asynchronous — which makes it the only case whose sole available
offence is the clock read. The same mutation now fails, and fails only there.

The general lesson outlived the specific bug: **a positive control that trips
two detectors pins neither.** When adding one, check what else it would set off.

**What it cost.** Small, real ergonomic friction. `FrameData.pixels` is typed
`object` rather than `np.ndarray`, and adapters `isinstance`-check at the
boundary. `EncodedPacket.codec` is a plain lowercase `str` rather than a library
enum. Policy functions take a `now: float` parameter that callers must thread
through. All three are worth it.

**What would change it.** Nothing foreseeable. If a genuinely pure third-party
library were needed in `domain/`, add it to the allowlist deliberately, in the
same commit as the reason.

---

## 7. The escalation gate exists at all

**Decided.** A VLM call happens only when a pure predicate says the scene
warrants one, and only if three governors then permit it.

**Why this is the thesis, not an optimisation.** Describing every frame with a
3B vision model is unaffordable on one consumer GPU — a describe takes ~2.7 s
for 64 tokens, against 25–30 fps arriving. Describing nothing is a motion
detector. The entire product proposition is *spending GPU only where it buys
information*. Everything else in the engine — the VRAM planner, the admission
gate, the resident set, the pre-roll buffer — exists to serve escalations that
the gate decided were worth making.

If someone proposes removing the gate to "simplify", they are proposing
removing the product.

**The seven triggers** are pure predicates in `domain/policy/triggers.py`,
evaluated in a fixed priority order, first-to-fire naming the reason:
`speed_anomaly`, `dwell_exceeded`, `track_count_spike`, `scene_change`,
`new_salient_track`, `periodic_summary`. The seventh, `user_requested`, is not a
predicate — it bypasses them entirely via `force()`, from
`POST /cameras/{id}/describe`.

**The three governors**, in this order and for these reasons:

1. **Cooldown** — an unconditional temporal floor. Inside the window nothing
   escalates, whatever the scene looks like.
2. **Duplicate suppression** — scene-signature distance below
   `DEDUP_EPSILON = 0.05` means we have already described this.
3. **Token bucket** — the rate budget. **Last, because it is the only stage that
   consumes anything.** Neither cooldown nor dedup may run after it and throw a
   spent token away.

Cooldown precedes dedup because inside the cooldown window comparing signatures
is wasted work and reports the less actionable of two reasons.

The bucket is immutable: `try_consume` returns `(allowed, new_bucket)`.
Suppressed decisions still carry the trigger reason, so telemetry shows what the
gate declined and why — that is what makes the gate tunable rather than opaque.

**A second, process-wide governor exists** because the per-camera ones are not
enough. `AdmissionGate` (`orchestrator/admission.py`) caps concurrency and
minimum interval across *all* cameras. With one camera the two are equivalent;
with N cameras, N independently-permitting buckets could each legitimately allow
a call and collectively saturate the GPU. It was built while it was still
trivially testable rather than retrofitted into the hot path later.

**What it cost.** Real behaviour is missed between escalations. A dwell that
starts and ends inside a cooldown window is never described. Tuning is a
per-camera burden (`CameraProfile`), and the defaults are fixed values that
Phase 4 intends to replace with learned per-camera baselines.

**What would change it.** Cheap enough continuous VLM inference, or a
lightweight always-on behaviour classifier that makes the coarse triggers
unnecessary. Neither is close.

---

## 8. `contracts/` is the only engine ↔ UI coupling

**Decided.** `contracts/openapi/ai-engine.yaml` and
`contracts/events/anomaly_event.schema.json` are the entire interface between
`ai-engine/` and `web/`. The UI generates its types from them. `web/` never
imports from `ai-engine/`; `ai-engine/` never reads `web/`.

**Why.** Two trees, two toolchains, two languages, and a third consumer (the
Phase 1C Go backend) arriving later. A shared schema that both sides generate
from is the only coupling that survives that.

**What it cost.** A regeneration step (`npm run gen`) and a drift test. Both are
cheap. The generated files are committed so a checkout builds without running
codegen.

**Enforcement, honestly.** The *OpenAPI* side is enforced:
`ai-engine/tests/test_openapi_contract.py` re-renders the document from the live
FastAPI app and asserts byte-equality with the committed copy. It is a pytest
assertion rather than a CI step deliberately — CI already runs the suite, and it
fails on the developer's machine the moment they change a route, not twenty
minutes later on a push.

The *directory boundary* is *not* yet enforced. It is true structurally today,
but no fitness test asserts it, and there is no web CI at all — `.github/workflows/ci.yml`
runs Python only, so the console's tests, `tsc -b` and `vite build` are ungated.
Both gaps are known and unowned. See [contracts/README.md](../contracts/README.md).

**What would change it.** Nothing about the seam. The enforcement gap should be
closed, not the rule relaxed.

---

## 9. A file for cameras, environment variables for scalars

**Decided.** Process-wide scalars are `SENTINEL_*` environment variables in
`config.py`. The camera list is a JSON file at `SENTINEL_CAMERAS_FILE`.

**Why.** A camera is a nested record containing a nested `CameraProfile` with
fourteen fields. Flattening a *list* of those into environment variable names
produces something like `SENTINEL_CAMERA_3_PROFILE_DWELL_SECONDS`, which cannot
be diffed, reviewed or mounted as a unit. One small document can be.

**What it cost.** A second configuration mechanism to learn, a file that must
exist before startup, and a `cameras.example.json` to keep in step. The real
file is gitignored, because it carries site names and URLs with credentials in
them.

**Consequences worth knowing.** `profile` overrides are validated against
`CameraProfile`'s own field names, so a typo fails at startup rather than
silently doing nothing. Camera-file problems are fail-loud on purpose: every
alternative — skip the bad camera, fall back to defaults — produces a process
that runs while silently watching fewer cameras than the operator configured.
And a `url` without an `rtsp://`/`rtsps://` scheme is treated as a file path,
which is what makes replay and live capture the same code path.

**What would change it.** A real control plane owning camera configuration
(Phase 1C's Go backend and Postgres) would make the file a bootstrap default
rather than the source of truth.
