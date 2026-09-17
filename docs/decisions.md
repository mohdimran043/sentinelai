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
| [10](#10-welfare-concerns-are-an-opinion-not-a-detection) | Welfare concerns are an opinion, not a detection | Accepted |
| [11](#11-a-capability-decides-which-models-exist-not-which-ones-run) | A capability decides which models exist, not which ones run | Accepted |
| [12](#12-fall-detection-is-a-temporal-signature-and-it-gets-its-own-basis) | Fall detection is a temporal signature, and it gets its own `basis` | Accepted, unvalidated on real footage |
| [13](#13-behaviour-detectors-are-pure-state-machines-measured-in-body-heights) | Behaviour detectors are pure state machines, measured in body heights | Accepted |
| [14](#14-an-alert-is-an-episode-not-an-event) | An alert is an episode, not an event | Accepted |
| [15](#15-person-authorization-stores-sealed-embeddings-and-nothing-else) | Person authorization stores sealed embeddings, and nothing else | Accepted |
| [16](#16-triage-state-is-durable-the-event-stream-is-still-the-record) | Triage state is durable; the event stream is still the record | Accepted |
| [17](#17-scale-by-processes-not-by-threads) | Scale by processes, not by threads | Accepted, supersedes a wrong diagnosis |
| [18](#18-an-earthcam-camera-is-a-page-url-resolved-every-time) | An EarthCam camera is a page URL, resolved every time | Accepted |

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

---

## 10. Welfare concerns are an opinion, not a detection

**Decided.** What the VLM reports about a person's wellbeing is stored as
`WelfareConcern{kind, confidence, evidence, evidence_stated}`, gathered into a
`WelfareAssessment` carrying a mandatory `basis: Literal["single_frame_vlm"]`.
`Confidence` is exactly `possible` | `likely`. There are no booleans, no
`certain` tier, and **no new `EscalationReason`** — the seven reasons are
unchanged, and there is deliberately no `fall_detected` among them.

**Why a boolean would have been the whole bug.** `fall_detected: true` reads to
any downstream consumer — the Phase 1C store, a dashboard, an auditor — as a
trained classifier's verdict. There is no fall detector in this system. There is
no pose estimation, no action recognition and no per-limb tracking; there is a
language model looking at one still frame. A boolean forces that judgment
through a threshold *before* it reaches storage, and once stored it is
indistinguishable from a real detector's output by anything reading the payload.
Keeping `confidence` as data instead lets each consumer choose its own bar
rather than inheriting one baked in here.

**Why there is no `certain`.** A single frame, sampled at most once per ~10 s by
the gate, cannot rule out an innocent explanation: someone lying down is not
necessarily someone who has collapsed, two people standing close are not
necessarily fighting. The tier is absent so that no caller — and no future
maintainer padding out an enum "for completeness" — can claim a certainty the
evidence cannot earn. Leaving it out is the honest option, not a missing one.

**`basis` is mandatory and single-valued** so a consumer reading the payload
alone, with no side channel and no tribal knowledge of which pipeline produced
it, can see where the opinion came from and weigh it accordingly.

**The notifier inverts the publisher's failure contract, and that is not an
oversight.** `EventPublisher.publish()` must **raise** on failure; a raise is
what hands the event to `FailedEventSink`, and a publisher that logs and returns
drops it past the last component able to save it. `Notifier.notify()` must
**never** raise. It is best-effort commentary on a pipeline that has already
published, dispatched off the escalation path onto its own worker, and one that
threw would take down the thing it exists to observe. The two ports look alike —
one record in, one destination — so this is written down here because
implementing the wrong contract fails silently in both directions.

**What it cost.** Real complexity, in three places:

- **Routing is not a boolean check.** `concerns_to_notify` is three clauses:
  kind in `notify_on`, confidence meets `notify_min_confidence`, and — if the
  concern is only `possible` — the event's threat score must already be in the
  caution band or above. That third clause exists because acting on `possible`
  alone, with nothing else in the system agreeing, is how a welfare notifier
  becomes noise an operator learns to ignore. An ignored notifier is a muted one.
- **`evidence_stated` had to be promoted into the domain, then onto the wire.**
  It began as an adapter-private placeholder constant, which meant the only way
  to tell "the model named a concern but described nothing" from a genuinely
  evidenced one was to string-match a leading-underscore adapter internal from
  wherever routing lived. It is a domain field for that reason, and a later task
  had to extend the event schema and codec so a Phase 1C consumer would not read
  every concern as evidenced.
- **Absent and empty had to stay distinguishable end to end**, through
  `cameras.json`, the `UNSET` sentinel in `CameraEdit`, the PATCH body and the
  console's diff. `notify_on` absent means every kind; `notify_on: []` means this
  camera notifies nobody. Collapse the two anywhere along that path and either a
  camera someone silenced starts alerting, or one they meant to route goes quiet.

**Consequences worth knowing.** Duplicate kinds collapse to the highest-
confidence concern rather than stacking, so a model reporting `collapse` under
two prompt phrasings does not double-count in any rule reading `len(concerns)`.
`evidence` is mandatory and rejected when blank, so a concern always points at
something. And the honest limits belong in operator-facing documentation, not
only in these docstrings — see
[operations](operations.md#read-this-before-you-rely-on-it), including that a
stretcher carry was missed entirely in measurement.

**What would change it.** A second source of welfare judgement — multi-frame
reasoning, or a purpose-built pose model — does **not** widen
`basis: "single_frame_vlm"` to cover it. It gets its own `basis` value, and the
routing rule is rewritten against the pair. Widening this one would retroactively
relabel every opinion already stored under it.

---

## 11. A capability decides which models exist, not which ones run

**Decided.** Each camera carries a `capabilities` set in `cameras.json`. The union over
every camera is computed **before anything is constructed**, and `main.build_models`
builds only the model roles that union names. A role nobody asked for is never
instantiated, never registered, never given VRAM, and never appears in `/health`.

**Why not a runtime flag.** The obvious cheaper design is to load everything and skip
the calls a camera does not want. That gets the behaviour right and the cost wrong: a
site running thirty corridor cameras on triggers alone would still hold 2.7 GiB of
vision-language weights and 460 MiB of pose weights for the life of the process, and
`ResidentSet` would dutifully keep them resident because residency is a function of what
is *registered*, not of what is used. §13's "automatically avoid loading unnecessary
models for disabled capabilities" is a statement about VRAM, and only a decision taken
before construction can honour it.

**Skipping is per camera as well as per process.** The detector is shared (ADR 4), so it
exists as soon as any camera needs one — but a camera with no capabilities is handed
`None` and runs no inference at all. Otherwise "switch this camera off" would still cost
a forward pass per frame to produce detections nothing reads.

**The pure layer names roles, not checkpoints.** `domain/capabilities.py` says a
capability needs a `ModelRole.VLM`; which checkpoint fills that is `Settings`, and
`domain/` may not import `config` (ADR 6). Swapping Qwen for Moondream is a setting.

**What it cost.**

- **A field that had to be renamed.** `CameraProfile.vlm_enabled` never controlled the
  VLM — it short-circuits the whole gate — and that was a distinction without a
  difference only while every camera that escalated also described. Once
  `scene_description` became separately optional, a camera could escalate and publish
  while never calling a model, and a field called `vlm_enabled` sitting `True` on
  exactly that camera was the most misleading thing in the record. It is
  `auto_escalation_enabled` now, derived from the capability set at composition rather
  than stored beside it.
- **An event with no description is now two different facts.** `description_unavailable`
  has always meant "the model was asked and did not answer". A camera with description
  switched off produces an event with no description either, and rendering that as a
  model failure teaches an operator to ignore a flag that otherwise means a real fault.
  Rather than widen a required boolean into a tri-state — a breaking change for every
  consumer already reading it — the reason rides in `Event.metadata` under
  `description_skipped`.
- **Enabling a capability at runtime can be refused.** `PATCH /cameras/{id}` takes
  `capabilities`, and disabling always works. *Enabling* one whose model this process
  never loaded answers **409 naming the restart**, because placing a 3B model under an
  HTTP request would stall every camera sharing the GPU. The check runs before the file
  is written, so a refusal leaves the record and the running camera agreeing.

**What would change it.** A model server that could place and evict weights out of
process on demand (the `SENTINEL_MODE=production` gRPC seam) would make enabling a
capability live a bounded operation rather than a restart.

---

## 12. Fall detection is a temporal signature, and it gets its own `basis`

**Decided.** `domain/behaviour/fall.py` is a pure state machine over bounding-box
geometry and, where available, pose keypoints: **upright → rapid descent → horizontal →
still, for long enough**. All four are required. It emits `FallEvidence` — measurements,
no score — raises `EscalationReason.FALL_SUSPECTED`, and the welfare concern that
survives vision-language confirmation carries `basis="temporal_pose_vlm"`.

**This is ADR 10's own escape clause being used, not overturned.** That decision said a
second source of welfare judgement "does **not** widen `basis: 'single_frame_vlm'` to
cover it. It gets its own `basis` value, and the routing rule is rewritten against the
pair." That is exactly what happened: `basis` went from a JSON `const` to a two-member
enum, and every opinion already stored under `single_frame_vlm` still means what it
meant when it was written. A consumer that hard-coded the old `const` now rejects a
`temporal_pose_vlm` payload, which is the correct failure — it is being handed evidence
of a kind it has no handling for.

**Why the transition and not the posture.** A person lying on the ground is not a fall;
it is a person lying on the ground, and there are innocent reasons for it. A detector
firing on posture alone produces exactly the alert an operator learns to dismiss, and a
dismissed alert is worse than none because it costs the attention a real collapse then
does not get. The cost of this choice is stated plainly rather than hidden: **someone
already on the floor when they enter frame raises nothing.** This detects falling, not
lying.

**Why body heights, never pixels.** Every rate and distance is normalised by the
person's own bounding-box height. A `px/s` threshold is a threshold on
distance-from-camera wearing a speed's clothes — tuned on one camera and wrong on the
next.

**Why no confidence score.** `FallEvidence` carries the descent rate, how long the
person has been down, and whether pose or geometry did the reading. A single float would
be read as a calibrated probability by everything downstream, and nothing here has
earned one — the same argument ADR 10 makes about the VLM's own opinion.

**It bypasses the gate's three governors**, like `user_requested` and unlike the six
automatic triggers. The machine has already deduplicated to one report per episode and
will not raise again until the person stands up, so the governors have nothing left to
protect against — and a cooldown window swallowing the one escalation that mattered is
the failure this subsystem exists to prevent.

**Pose is an enhancement, never a requirement.** A camera without it runs the same
machine on box aspect ratio and records `used_pose=False`. A pose model that raises
degrades the reading and does not cost the frame.

**What it cost.**

- A ninth port (`PoseEstimator`) and a third registrable model.
- Single-stage pose output has to be reconciled with the pipeline's own tracks by IoU,
  and a skeleton that cannot be confidently attributed is dropped. A pose bound to the
  wrong person does not add noise — it makes one person's posture read as another's for
  as long as the confusion lasts.
- The machine must be reset on a stream discontinuity, for the tracker's reason one
  layer up: every phase is keyed by a track id the reconnect invalidated and every
  timestamp is on a timeline that no longer exists.

**What has not been validated.** The state machine is exercised exhaustively against
scripted geometry — falls, sits, stumbles, people already down, tracking artefacts — and
end to end through the real frame loop. It has **not** been measured against real fall
footage, because none is in this repository. Its false-positive and false-negative rates
on real video are unknown. Read `operations.md` before relying on it.

**What would change it.** A trained action-recognition model would be a third source and
would get a third `basis` value, not this one.

---

## 13. Behaviour detectors are pure state machines, measured in body heights

**Decided.** Abandoned object, camera tamper, zone intrusion and line crossing are
implemented the same way ADR 12 implemented falls: a frozen dataclass of thresholds, a
`(state, observation) -> (state, candidates)` function in `domain/behaviour/`, and no
clock, no pixels beyond geometry, no model call. `BehaviourEngine` (`pipeline/`) owns
the mutable per-camera state and nothing else; every judgement is in `domain/`.

**Why no framework.** Each detector is 100–200 lines and shares a shape, not code. An
abstract `Detector` base class would have bought one thing — a registry — at the cost of
forcing four genuinely different state machines through one interface. `observe_falls`,
`observe_abandonment`, `observe_tamper` and `observe_zones` are four functions the
engine calls in sequence; adding a fifth is adding a function and one call site.

**Why loitering is not among them.** It already exists. `dwell_exceeded` has been an
escalation trigger since the gate was written, with its own radius and duration. §6 of
the brief asked for loitering; implementing a second one would have been a duplicate
under a different name, and the honest answer was to say so and tune the existing one.

**Every distance is in body heights, every share is a fraction.** A person must be
within `1.5` of their *own* bounding-box height of a bag to count as attending it; a
zone vertex is a fraction of frame width. Both for ADR 12's reason: a pixel threshold is
a threshold on distance-from-camera and resolution, wearing a length's clothes. This one
bit us concretely — an RTSP source that renegotiates from 1920×1080 to 960×540 would
silently shrink every pixel-specified zone to a quarter of its intended area, and
nothing would report an error.

**Camera tamper needs no model at all.** It reads the luma histogram the motion stage
already computes and asks whether one bin holds most of the frame. That makes camera
health free to enable on every camera in a site — which matters because an obstructed
camera and a quiet camera look identical in every other signal the system has.

**What it cost.**

- A `min_clear_seconds` gate on tamper, because a camera that is dark at 3am is not
  being tampered with. The detector must have seen a varied view *first*, and the clear
  run resets the moment it ends — an early version accrued "clear" time while already
  blank and would have alarmed on a permanently covered lens forever.
- Object abandonment is keyed on the tracker, so a bag whose track is lost and
  re-acquired starts over. Preferred to the alternative: a detector that re-identifies
  objects across track breaks would report the same bag repeatedly.
- Ground points, not centroids, for person↔object distance. A tall person's centroid is
  a metre above the floor; the bag is on it.

**What would change it.** A fifth detector that genuinely needed cross-detector state —
say, "this bag was left by the person who is now loitering" — would justify a shared
context object. Four independent ones do not.

---

## 14. An alert is an episode, not an event

**Decided.** `Alert` is a first-class domain entity keyed by
`(camera_id, reason, subject)`. The first qualifying event opens one; every later event
matching that key within `SENTINEL_ALERT_MERGE_WINDOW_SECONDS` increments
`occurrences` and extends `last_seen_at` rather than opening another. An operator
acknowledges the episode, not each sighting.

**Why.** The brief's own example: a person seen 17 times in 20 seconds is *one* thing
happening, and 17 rows is a UI that teaches operators to stop reading it. The engine
already had three deduplication governors on the *escalation* side (cooldown, scene
dedup, token bucket), but those exist to protect the GPU. They are tuned for compute,
not attention, and they cannot merge across them — a fall and a zone intrusion by the
same person are two escalations and should stay two events, while five zone intrusions
by that person are one alert.

**Why the subject is `subject_track_ids`, not `track_ids`.** This is the decision that
cost the most to get right. `Event.track_ids` is every track in the scene, so a busy
corridor produced a different key on every frame — a live run generated eleven separate
alerts for one person walking past, each keyed by whoever else happened to be in shot.
`subject_track_ids` was added to `Event` for exactly this: the tracks the detector
attributed the event *to*. It is omitted from the wire payload when empty so that every
trigger-raised event stays byte-identical to what it was before the field existed.

**Severity is the maximum of the model's opinion and the reason's floor.** A VLM that
describes a collapse mildly cannot lower a `FALL_SUSPECTED` below critical. The floor
lives in `domain/policy/priority.py` next to the priority ordering, because they answer
the same question — how much of a human's attention this deserves.

**The register is volatile, and that is a real limitation.** It lives in engine memory,
bounded at `SENTINEL_ALERT_REGISTER_CAPACITY`, evicting resolved before acknowledged
before active. An acknowledgement does not survive a restart. The durable record is the
published event stream; the alert layer is a view over it for an operator at a screen.
Phase 1C's Postgres is where this becomes durable, and nothing about the domain model
has to change when it does.

**What would change it.** Multi-operator use. Two people acknowledging from two consoles
against an in-memory register is a race the current design does not address, because
there is currently one console and no authentication.

---

## 15. Person authorization stores sealed embeddings, and nothing else

**Decided.** Per-camera opt-in face recognition. Enrolment produces a 512-d ArcFace
embedding, sealed individually with AES-256-GCM under `SENTINEL_FACE_ENCRYPTION_KEY`,
written to a 0600 file by write-then-rename. **No face image is ever stored**, and no
embedding or similarity score is logged.

**Why no images.** The system does not need them. Recognition needs the embedding;
review needs the event clip, which the evidence pipeline already produces. Storing a
reference photo would add a second, more sensitive copy of a person's biometrics for no
capability the system lacks without it.

**The engine refuses to start without a key.** If any camera enables
`person_authorization` and `SENTINEL_FACE_ENCRYPTION_KEY` is unset, startup fails.
There is no default key and no plaintext fallback, because a fallback path is the path
everything ends up on. The cost is stated honestly: rotating the key makes every
enrolled face undecryptable and everybody must re-enrol.

**No threshold is hard-coded.** All eight — match similarity, minimum observations,
minimum duration, minimum face pixels, minimum frontality, and the three quality bars —
are fields on a per-camera `AuthorizationPolicy` with documented defaults. The measured
starting point (`0.42` cosine) came from running the real pipeline: self-match `1.000`,
a different person `0.065`. A site with a different camera height and lens will need a
different number, and the config is where that conversation happens.

**Recognition is sticky for the life of a track.** Once a track has matched an
authorized person, it stays recognised even when the face becomes unreadable. An
earlier version re-accumulated evidence every time somebody turned away from the camera,
and would eventually accuse a person it had already identified. A person does not stop
being authorized by turning their head.

**Three independent quality bars, never blended.** Face size, frontality and detector
score each have a floor, and a face must clear all three. A weighted score would let a
large, badly-angled face pass on size alone — and the whole point of the bars is to keep
a bad embedding out of a comparison whose output is an accusation.

**Unauthorized is a low-severity finding, not an alarm.** `UNAUTHORIZED_PERSON` sits
well below `FALL_SUSPECTED` in `priority.py`. The system's confidence that it has
correctly identified a stranger is much lower than its confidence that somebody fell,
and the consequence of being wrong lands on a person.

**What it cost.**

- A second inference runtime. InsightFace runs under ONNX Runtime, not torch, so the
  process carries two model stacks. Worth it: it falls back to CPU cleanly, and faces
  are rare relative to frames.
- Enrolment is an API call with an image body, which is the one place a face image
  enters the process. It is embedded and discarded within the request.

**What would change it.** A requirement to show operators *who* a stranger resembles
would need reference images and is a different decision, made with a different set of
people in the room.

---

## 16. Triage state is durable; the event stream is still the record

**Decided.** The alert register gets an `AlertStore` port and a JSON-file adapter.
Operator actions — acknowledge, resolve — are flushed **before the API answers**.
Machine-driven changes — an alert opening, an occurrence count rising — are flushed by a
coalescing five-second timer. `SENTINEL_ALERT_STORE_PATH=null` restores the previous
memory-only behaviour.

**Why the two paths are paced differently.** They are different kinds of fact. An
occurrence count is recoverable — every event behind it is already published to the
broker — and it changes as fast as the site is busy. An acknowledgement exists nowhere
else in the system and changes a handful of times a minute, because there is a human in
the loop. Flushing everything synchronously would write the whole register on every
event; flushing everything lazily would let a `kill -9` throw away the one thing that
cannot be reconstructed. So the rare, precious path waits and the common, recoverable
path does not.

**A 200 from `POST /alerts/{id}/acknowledge` now means the decision is on disk.** An
operator told "acknowledged" by an engine that then restarts and shows the row as unseen
has been lied to about the only state they created, and after that they stop trusting
the list. Verified against `kill -9`, not just a clean shutdown.

**A revision counter, not a dirty flag.** The coordinator records the revision it last
saved and compares. A boolean would have to be cleared either before the write — losing
a change that arrives during it — or after, re-writing an unchanged set. There is a test
for exactly that interleaving.

**Acknowledgement became first-wins and idempotent.** Two consoles watching one wall
both acknowledge the same row; last-write-wins would push `acknowledged_at` later every
time somebody looked, turning "when did this stop being unseen" into "when did somebody
last click". A second acknowledgement is not an error — the operator is reading a live
list — so it returns the existing state rather than a 409.

**This does not make the alert store the record of what happened.** That is still the
anomaly event on the broker. This file records what a human *did about it*. The
distinction is written into the file's own `_comment` field, because somebody will find
it on a server one day and need to know.

**What it cost.**

- A store that cannot be read — truncated, or written by a build with a different schema
  version — loses triage state and lets the engine start. Taking surveillance down over
  a bookkeeping file is the worse failure, and the file is kept rather than deleted so
  there is something to debug.
- A seventh shutdown phase, after the notification drain, because publishing is what
  opens an alert and the register is not settled until the last publish has happened.

**What would change it.** Phase 1C's Postgres. It replaces this adapter and nothing
else, which is what the port is for — and it is also what makes multi-operator use
addressable, which an in-process register is not.

---

## 17. Scale by processes, not by threads

**Decided.** More cameras than one process comfortably carries are run as several
processes, each with its own `cameras.json` and port. Measured on the 24-core box, for
the same twenty cameras: one process 49.3 fps, two processes 114.4, **four processes
176.0** — 3.6x, on hardware that did not change.

**This supersedes a wrong diagnosis, and the wrongness is the interesting part.**
`docs/performance.md` previously said the binding constraint was CPU H.264 decode and
pointed at ADR 2's "what would change it". One measurement disproved it: at twenty
cameras the engine uses **113% of a single core on a 24-core box**. Not CPU-bound. The
GPU does its forward pass in about 5 ms and is delivering 49 of them a second, so not
GPU-bound either. The constraint was never a resource — it was the single Python
process: one event loop, one shared detector behind a lock (ADR 4), and a GIL
serialising the executor thread against the loop. Twenty-three cores idle, one
saturated.

**Two optimisations were built before that was understood, and both are kept anyway.**
Lazy pixel conversion (`DeferredPixels`) takes a 1080p decode from 1.51 to 1.09 cores
when nine frames in ten are dropped, and is strictly less work for identical output.
`SENTINEL_DECODE_HWACCEL` is no longer inert. Neither moved the sweep, because neither
addressed the constraint. They are documented as such so nobody measures them again.

**Over-sharding collapses, and the cliff is steep.** Ten processes fell to 11.6 fps
aggregate — worse than one. Each pays its own CUDA context and model placement, and ten
of them time-slicing one GPU spend more on context switching than on inference. Four is
the measured recommendation here; the knee is somewhere between four and ten and will
move with the GPU.

**What it cost.** Nothing in code, which is the appeal. It costs one thing in operations:
each process holds its own alert register and its own event ring, so an operator console
pointed at one process cannot see another's alerts. That is tolerable while the console
talks to one engine and becomes Phase 1C's problem the moment it should not be.

**What would change it.** Real parallelism inside one process — a free-threaded Python
build, or moving inference behind the gRPC transport `SENTINEL_MODE=production` reserves,
so the GIL stops being shared with the decode and orchestration loop.

---

## 18. An EarthCam camera is a page URL, resolved every time

**Decided.** A public EarthCam camera is configured as the page a person would open in a
browser. `EarthCamSource` fetches that page before **every** connection attempt, reads
the configuration the page embeds for its own player, picks the best variant from the
master playlist, and opens it with the request context the player uses.

**Nothing is cached, and that is the design rather than an omission.** The page hands out
a playlist URL signed with `?t=…&td=…` that expires. Caching it is the failure this
exists to prevent — a stale URL is the 403 that made the naive version look broken. So
there is no token store, no refresh timer and no separate "403 handler": a signature that
dies mid-stream is repaired by the reconnect its own death triggers, on the backoff
`RtspSource` already had. The 403 is still *recognised*, because "your URL is old" and
"the camera went private" are different things an operator should not have to tell apart
from an ffmpeg error string.

**It resolves; it does not circumvent.** There is no login, paywall or DRM. The signature
is issued to anonymous visitors by the page itself, and this asks the page for a current
one rather than forging, extending or replaying anything. The `Referer`, `Origin` and
`User-Agent` sent are the player's own — a truthful statement of where the request came
from, not a disguise. Nothing here retries past a refusal.

**`EarthCamSource` subclasses `RtspSource` rather than copying it.** The two differ in
exactly one thing: what to open. The reconnect loop, the single demux pass fanning out to
decoded frames *and* still-encoded packets, the discontinuity signal that resets the
tracker, the bounded queues and the shutdown are the same problem — and that loop is the
trickiest code in the pipeline, so two copies of it was the worst available outcome.
`_open_container` and `_stream_label` were extracted as the two seams this needed.

**The signature never reaches a log.** `redact_url` drops the whole query rather than
named parameters, so a future one nobody has thought about does not have to be added to a
denylist. The field holding the token is excluded from the dataclass's `repr`, because the
usual way a secret reaches a log is a traceback nobody wrote.

**Only earthcam.com is fetched**, checked on the page URL *and* on every media URL the
page points at. `build_source` takes that URL straight from `cameras.json`; without the
check, a camera entry would be a server-side request forgery primitive, and the second
check matters because the page decides where the media lives.

**What it cost.**

- `httpx` became a runtime dependency rather than a dev one.
- A camera that depends on somebody else's website being up, and on its page layout. The
  parse fails loudly with a message saying the layout may have changed, and the two
  integration tests that hit the real pages are marked `network` so a change on their end
  reports without failing the build.

**What would change it.** A camera the page serves over something other than HLS, or an
EarthCam that starts requiring an account — the second would make this a credentialed
client, which is a different decision with a different answer.

