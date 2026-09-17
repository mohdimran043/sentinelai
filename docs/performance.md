# Measured performance

Every number on this page was produced by `python -m sentinel_ai.benchmark` on the
hardware named below. Nothing here is extrapolated from a model card, scaled from a
smaller run, or carried over from another machine. Where a figure is absent it is
because it has not been measured — not because it was inconvenient.

Reproduce any of it with one of two entry points. The sweep, which composes the whole
engine over N cameras:

```bash
cd ai-engine
python -m sentinel_ai.benchmark --video ../datasets/avenue/avenue_01.mp4 \
    --cameras 1,5,10,20 --seconds 60
```

…and the micro-benchmarks, for the two stages the sweep cannot isolate — the behaviour
state machines, which round to nothing inside a 5 ms detector pass, and the face
pipeline, which only runs on frames that contain a readable face:

```bash
python -m sentinel_ai.benchmark.micro
```

## The box

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GiB, driver 580.173.02, CUDA 13.0 |
| CPU | 24 logical cores |
| Torch | 2.14.0+cu130 |
| Footage | CUHK Avenue test clip — 640×360, H.264, 25 fps, 1439 frames |

**The clip is 640×360.** That is small for a security camera, and it flatters every
throughput number here. A 1080p stream is roughly nine times the pixels to decode, and
decode — which the sweeps below identify as the binding constraint — would rise
substantially. The single-model latencies in the next section *were* measured at 1080p
and are directly comparable; the camera counts were not.

## Models

Loaded only where a camera's capabilities require them (see
[configuration](configuration.md#per-camera-ai-capabilities)).

| Model | Role | Quantisation | VRAM, standalone | VRAM, marginal | Latency (1080p, solo) |
|---|---|---|---|---|---|
| YOLO11s | detector | fp16 | 452 MiB | 452 MiB | **4.0 ms** p50, 5.4 ms p95 |
| YOLO11n-pose | pose | fp16 | 462 MiB | 11 MiB | **4.4 ms** p50, 5.8 ms p95 |
| Qwen2.5-VL-3B-Instruct | scene description | bitsandbytes NF4 4-bit | ~2.8 GiB | 2364 MiB | **3.3 s** p50, 3.9 s p95 |
| InsightFace `buffalo_l` (SCRFD + ArcFace) | person authorization | fp32 ONNX | 608 MiB | 654 MiB | **15.2 ms** p50, 17.8 ms p95 |

## Camera scaling — with person authorization

`anomaly_detection` + `person_authorization`, so the detector *and* the face pipeline on
every sampled frame that has a person in it. This was previously unmeasured.

| Cameras | Aggregate fps | Dropped | Detector p50 | Face p50 | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 25.0 | 0.0% | 6.2 ms | 5.3 ms | 1018 MiB |
| 3 | 72.6 | 3.2% | 29.5 ms | 6.2 ms | 1038 MiB |
| 5 | 71.9 | 42.4% | 57.7 ms | 6.3 ms | 1090 MiB |

**Three cameras run essentially clean and five do not**, which is the same ceiling as
every other configuration — and for the same reason, because the number that grows with
camera count is the *detector's*, not the face pipeline's.

**The face figure here is 6 ms, not the 15 ms in the model table, and the difference is
the footage rather than the code.** These are 640x360 street frames in which SCRFD finds
few usable faces, so most calls are the fixed detection pass with no embeddings to
compute. A doorway camera at 1080p with faces in most frames pays the per-face term from
the model table instead. **Do not read 6 ms as the cost of face recognition** — read it
as the cost on footage where there is almost nothing to recognise.

**Two VRAM columns, because the difference is large and the second is the useful one.**
Standalone is what the model costs as the only thing on the card — which for the pose
model is mostly the CUDA context it had to create. Marginal is what it costs to add it
to a process that already has the others, which is the question `plan_residency()`
actually asks, and which is why pose comes to 11 MiB rather than 462: it reuses
allocator slack and pays nothing toward a context that already exists.

**The face figure is the one to read carefully, because it is 8x worse without a
dependency that is easy to miss.** InsightFace runs under ONNX Runtime rather than
torch, and ONNX Runtime ships in two packages: `onnxruntime` is CPU-only and is what
`pip install insightface` pulls in. On the same box, the same 1080p frame holding six
faces:

| ONNX Runtime provider | p50 | p95 | VRAM |
|---|---:|---:|---:|
| `CUDAExecutionProvider` (`onnxruntime-gpu`) | **15.2 ms** | 17.8 ms | 608 MiB |
| `CPUExecutionProvider` (`onnxruntime`) | **124.7 ms** | 170.4 ms | 0 |

The engine falls back to CPU automatically and says so in `version()`, which is
deliberate — the capability keeps working — but 125 ms per frame caps a camera near
8 fps. Install the `face-gpu` extra rather than `face` if authorization matters.

The cost decomposes into a fixed detection pass plus a per-face embedding, measured by
running the same pipeline over a frame with no faces in it:

| | detection only | + 6 faces | per face |
|---|---:|---:|---:|
| GPU | 3.5 ms | 15.3 ms | ~2.0 ms |
| CPU | 27.4 ms | 124.7 ms | ~16.2 ms |

Frame size barely matters: SCRFD resizes to 640×640, so 640×360 and 1920×1080 come out
within 2 ms of each other. The per-face term is the one that scales with a crowd, and
the fixed term is why `_detect_unauthorized` returns before touching the models at all
when a camera has no person tracks — which is most frames on most cameras.

**608 MiB, and only 480 of it is visible at warmup.** ONNX Runtime allocates the
recognition workspace lazily, on the first frame that actually contains a face, so the
figure the adapter measures at the end of warmup is a floor rather than the cost. It is
also the one model here whose marginal figure is *larger* than its standalone one — 654
MiB warming up beside the detector in the live engine, against 608 alone.
`SENTINEL_FACE_VRAM_MIB` (704) is what `plan_residency()` reserves and covers both.

This model is also the reason `adapters/vram.py` grew a second instrument. Every other
figure in this table comes from torch's caching allocator, which cannot see ONNX
Runtime at all: measured that way, the face pipeline reported **0 MiB** while holding
608. `device_used_mib()` asks the driver instead.

The VLM figure is per describe at the default 256 `max_new_tokens`, measured in the
pipeline rather than in isolation. It is consistent with [ADR 7](decisions.md)'s "~2.7 s
for 64 tokens" and is the reason the escalation gate exists: at 3.3 s a call, describing
every frame of one 25 fps camera would need 82 GPUs.

Model placement (`service.start()`, weights already cached) takes **9–14 s** for all
three. That is the cold-start cost of a restart, and the reason `ResidentSet` keeps
models warm rather than loading on demand.

## The behaviour detectors cost nothing measurable

The four state machines in `domain/behaviour/` are pure Python over geometry — no model,
no GPU, no allocation beyond a frozen dataclass. Per frame, on the CPU above:

| Detector | 1 tracked person | 10 tracked people |
|---|---:|---:|
| `fall` | 5.3 µs | 36.4 µs |
| `zones` (1 zone + 1 line) | 3.7 µs | 22.4 µs |
| `tamper` | 1.4 µs | 1.1 µs |
| `abandonment` | 0.6 µs | 0.7 µs |

Microseconds against a 5 ms detector pass: **all four together are under 0.1% of the
frame budget at ten people.** Fall and zones scale with the number of tracked people, as
they must — they ask a question per person. Tamper and abandonment do not: tamper reads
only the luma histogram the motion stage already computed, which is why
[ADR 13](decisions.md#13-behaviour-detectors-are-pure-state-machines-measured-in-body-heights)
calls it free to enable site-wide, and abandonment's cost is proportional to abandonable
*objects*, of which this scene has none.

This is the payoff from keeping them in `domain/`. A detector that cannot read a clock
or touch a pixel is a detector whose entire cost is arithmetic.

## Camera scaling — triggers only

`--capabilities anomaly_detection`, so detector only. 25 fps arrival per camera, no
frame sampling.

| Cameras | Aggregate fps | Frames dropped | Detector p50 | Detector p95 |
|---:|---:|---:|---:|---:|
| 1 | 23.7 | 0.2% | 5.0 ms | 7.3 ms |
| 5 | 70.5 | 35.9% | 54.7 ms | 62.7 ms |
| 10 | 85.1 | 63.9% | 104.0 ms | 115.7 ms |
| 20 | 49.6 | 90.0% | 392.8 ms | 405.6 ms |

## Camera scaling — every capability

`scene_description` + `anomaly_detection` + `fall_detection`, so detector **and** pose on
every sampled frame, and the VLM on every escalation.

| Cameras | Aggregate fps | Dropped | Detector p50 | Pose p50 | VLM p50 | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 23.9 | 0.2% | 5.1 ms | 5.6 ms | 3342 ms | 3520 MiB |
| 5 | 52.5 | 56.2% | 66.6 ms | 18.0 ms | 6249 ms | 5278 MiB |

## What these numbers say

**One camera runs clean.** 23.9 fps against a 25 fps source with 0.2% dropped, with
detection, tracking, pose, the fall machine and the escalation gate all running. That is
the pipeline keeping up in real time.

**The binding constraint is not the GPU.** Detector p50 rises almost linearly with
camera count — 5 ms solo, 393 ms at twenty — while the GPU itself still does the forward
pass in about 5 ms. The 388 ms of difference is queueing, not compute. Aggregate
throughput plateaus near **85 detections/second** and then *falls*, which is the
signature of contention rather than saturation.

Two mechanisms, and the second is the larger:

1. **One shared detector behind a lock** ([ADR 4](decisions.md)). Every camera
   serialises through it. This is deliberate and correct — there is one GPU, so the
   kernels serialise anyway — but it means the wait is proportional to camera count.
2. **CPU decode and inference share a thread pool.** PyAV decode and
   `run_in_executor(None, ...)` both use the default executor. Giving the detector its
   own single-thread executor was measured: at 20 cameras it moved aggregate throughput
   from 49.6 to 54.6 fps and detector p50 from 393 ms to 328 ms — real, but about 10%,
   so it is not the dominant term and was **not** adopted (it would need its own
   lifecycle and an ADR amendment for a modest gain).

**This paragraph used to blame CPU decode. It was wrong** — see
[Scaling by processes](#scaling-by-processes-not-by-cameras), where a single measurement
of total CPU (113% of one core on a 24-core box) rules out decode, inference and the GPU
alike, and leaves the single-process architecture itself.

## Dropping frames before they reach the loop

**Measured on three 1080p 30 fps cameras: 85% dropped, then 0.0%.** The fix was not to
make anything faster — it was to stop decoding-and-discarding on the one thread that
could not afford it.

| | Dropped | Detections |
|---|---:|---:|
| Uncapped (30 fps per camera) | 80–92% | ~10/s total |
| **`SENTINEL_SOURCE_MAX_FPS=8`** | **0.0%** | every delivered frame |

A camera delivers 30 frames a second; this pipeline processes five to ten. The other
twenty-five existed only to wake the event loop and overwrite a mailbox slot. Capping
what the *source hands on* — the codec still decodes every frame, because an inter-coded
frame is meaningless without its references — gives that loop time back to the frames
that are actually looked at.

It is a cap, not an optimisation, and it is honest about what it costs: **8 fps is the
temporal resolution of everything downstream**, including the fall machine's descent
rate. Set it near what the pipeline actually achieves, never below what a detector needs.

### What was ruled out first, by measurement

Chasing the drop rate without measuring would have led somewhere useless. In order:

| Suspect | Measured | Verdict |
|---|---|---|
| The detector | 3.7 ms/frame — a 271/s ceiling against ~10/s observed | Not it |
| The GPU | 1–2% utilisation | Not it, and no GPU change could fix it |
| CPU exhaustion | 69% of **one** core, 24 available | Not starved — serialised |
| The motion stage | 16.75 ms/frame on the event loop | **Real**, fixed below |
| The VLM | ~50% of wall time generating, GIL held per token | **Real**, reduced below |

### The motion stage was the loop's largest per-frame cost

`MotionAnalyzer.analyze` ran inline on the event loop and promoted a full 1920x1080
frame to `float64` — twice, for current and previous — before subtracting. 50 MiB per
operand, per frame, on the thread this engine is short of.

| | Per frame |
|---|---:|
| `float64`, full frame | 16.75 ms |
| **`int16`, every 4th pixel** | **1.50 ms** |

`int16` is exactly wide enough for a difference of two `uint8`s, so that half is
bit-identical. The quarter-resolution sample moves a whole-frame mean and a 16-bin
histogram by about a thousandth — measured on the same frames: `0.333474` against
`0.333813`. Process CPU fell from 77% to 45% of one core.

### The vision model is the largest single GIL holder

A describe holds the GIL in its per-token loop, so its duration is time no camera's
frame loop runs. On an RTX 4090 the 4-bit weights were costing more than the memory they
saved:

| `SENTINEL_VLM_QUANTIZATION` | Per describe | VRAM |
|---|---:|---:|
| `nf4` (ADR 1's default) | 1.97 s | 2820 MiB |
| **`none` (fp16)** | **1.42 s** | 7224 MiB |

NF4 dequantises weights on every forward pass — a good trade when VRAM is the binding
constraint, a poor one when 17 GiB is free. `nf4` stays the default because it is what
makes this model fit on an 8 GiB card.

## Reading the drop rate

**A high `frames_dropped` is the design working, not a fault.** Frames land in a
single-slot mailbox that *overwrites*: an overwrite before the consumer collects the
previous frame is counted as a drop, and that is what makes "always work on the freshest
frame" true. A queue would drop nothing and produce a delayed complete record, which is
the wrong trade for surveillance.

So the number to optimise is **detections per second**, not the drop percentage. Two
1080p 30 fps cameras sit around 85% dropped while detecting steadily, and forcing that
percentage down would mean either a slower source or a queue that falls behind.

### The motion stage was the loop's largest cost

Measured on those two cameras, `MotionAnalyzer.analyze` ran **inline on the event loop**
and promoted a full 1920x1080 frame to `float64` — twice, for the current and previous
frame — before subtracting. That is a 50 MiB allocation per operand, per frame, on the
one thread this engine is short of ([ADR 17](decisions.md#17-scale-by-processes-not-by-threads)).

Two changes, both of which leave the answer intact:

| | Per frame |
|---|---:|
| `float64`, full frame | 16.75 ms |
| **`int16`, every 4th pixel** | **1.50 ms** |

`int16` is exactly wide enough for a difference of two `uint8`s, so the result is
identical to the bit. The quarter-resolution sample changes a whole-frame mean and a
16-bin histogram by about a thousandth — measured, on the same frames: `0.333474`
against `0.333813`.

**Process CPU fell from 77% to 45% of one core** on the live two-camera deployment. The
drop rate did not move, because it is not what that cost was limiting: the remaining
ceiling is the single shared detector and the single event loop, and the fix for those
is the one below.

## Scaling by processes, not by cameras

**The single most useful measurement in this document.** The same twenty cameras and
the same box, sharded differently:

| Layout | Aggregate fps | Drop |
|---|---:|---:|
| 1 process x 20 cameras | 49.3 | 90.1% |
| 2 processes x 10 cameras | 114.4 | 77.0% |
| **4 processes x 5 cameras** | **176.0** | **64.8%** |
| 10 processes x 2 cameras | 11.6 | — |

Four processes carry **3.6x** the throughput of one, on hardware that has not changed.

### Why, and why the earlier explanation in this document was wrong

An earlier version of this page said the remaining cost was CPU H.264 decode and
pointed at [ADR 2](decisions.md#2-cpu-decode-not-nvdec). That was wrong, and one
measurement disproves it: at twenty cameras the engine uses **113% of a single core on a
24-core box**. It is not CPU-bound, it is not GPU-bound — the detector's forward pass is
about 5 ms and it is delivering 49 of them a second — and it is not decode-bound.

It is bound by being one Python process. Everything funnels through a single event loop
and a single shared detector behind a lock ([ADR 4](decisions.md#4-one-shared-detector-behind-a-lock)),
and the GIL serialises the executor thread against that loop. Twenty-three cores sit
idle while one is saturated. Sharding the cameras across processes is the fix that
follows directly, and it needs no code: each process gets its own `cameras.json` and its
own port.

**Over-sharding collapses.** Ten processes fell to 11.6 fps aggregate — worse than one.
Each pays its own CUDA context and its own model placement, and ten of them time-slicing
one GPU spend more on context switching than on inference. The knee is between four and
ten; four is the measured recommendation on this box.

### What this did *not* come from

Two decode optimisations were built and measured during this work, and neither moved the
sweep, because neither addressed the constraint:

- **Lazy pixel conversion** (`DeferredPixels`). Frames now defer their BGR conversion
  until something reads them, so a frame the mailbox overwrites is never converted.
  Measured in isolation, converting one frame in ten instead of all of them takes a
  1080p decode from **1.51 to 1.09 cores**. It is strictly less work for identical
  output and it is kept — but on a box that was using 1.13 cores of 24, saving CPU was
  never going to raise throughput.
- **Hardware decode** (`SENTINEL_DECODE_HWACCEL`, previously inert). Now implemented,
  and off by default because the numbers say so — see the table in
  `adapters/sources/hwaccel.py`. CUDA decode alone is nearly four times cheaper on CPU
  (0.26 cores against 1.00) and the *pipeline* is more expensive end to end (1.84 cores
  against 1.60), because pulling an NV12 surface back to host memory costs more than
  software-decoding to YUV420P there in the first place.

Both are honest improvements to the wrong bottleneck. They are documented here so the
next person does not measure them again.

## Accuracy — does it work, not what does it cost

Everything above is throughput and footprint. These are the answers to the other
question, produced by `python -m sentinel_ai.validation.*` on public datasets with the
real models. Two capabilities previously had no accuracy measurement of any kind.

### Person authorization: LFW, 2200 pairs

The standard Labeled Faces in the Wild verification protocol — 1100 pairs of the same
person, 1100 of different people — scored **at the threshold this engine ships**
(`match_threshold = 0.42`), not at one tuned on these pairs.

| | |
|---|---:|
| Recall (same person, matched) | **92.7%** (1020/1100) |
| **False match rate** | **0.0%** (0/1100) |
| Accuracy | 96.4% |
| Same-person similarity, median | 0.676 |
| Different-person similarity, 95th percentile | 0.098 |

**Not one false match in 1100 opportunities**, and the shipped threshold sits in a wide
empty band between 0.098 and 0.676. That is the right direction for this system: a miss
means a person is not recognised, which raises a low-severity `unauthorized_person`
finding; a false match would mean wrongly treating a stranger as authorised.

Four pairs had no detectable face at all and were counted as non-matches, which is what
the engine would do — no usable face means no observation.

### Person authorization: even-handedness, FairFace, 1800 images

FairFace carries race, gender and age labels but no identity labels, so same-person
pairs cannot be built from it and **verification accuracy by group is out of reach**.
What is reachable is the three things this pipeline actually gates on.

| Group | n | Face detected | Passes the quality bars | Impostor similarity, p99 |
|---|---:|---:|---:|---:|
| Black | 290 | 100.0% | 70.3% | 0.233 |
| East Asian | 266 | 100.0% | 69.2% | 0.226 |
| Indian | 221 | 100.0% | 69.7% | 0.219 |
| Latino / Hispanic | 257 | 100.0% | 75.9% | 0.187 |
| Middle Eastern | 196 | 99.5% | 70.4% | 0.179 |
| Southeast Asian | 218 | 99.5% | 69.7% | 0.237 |
| White | 352 | 99.1% | 66.8% | **0.156** |

**Detection and quality gating are even-handed here.** Every group is detected at
99.1–100%, and the share clearing `min_box_pixels`, `min_frontality` and
`min_detector_confidence` sits in a 67–76% band with White faces at the *bottom* of it.
Neither is the disparity that was expected.

**Impostor similarity is not even-handed.** Every pair sampled above is two different
people, so the last column is how close near-misses get inside a group. White faces sit
at 0.156; every other group is 0.179 to 0.237, consistently 15–50% closer to the
decision boundary. All of them remain far below the shipped 0.42, so this is a **margin
difference rather than an observed failure at this threshold** — but the direction is the
one the literature describes, and it has a concrete consequence: a threshold validated on
LFW, which is predominantly White, is more conservative for White faces than for
everybody else. A site lowering `match_threshold` to improve recall spends that margin
unevenly.

### Fall detection: URFall, 70 clips

30 clips each containing one fall, 40 of daily activities containing none, from the
camera parallel to the floor. **This is the measurement the documentation has been
saying was missing, and it does not flatter the detector.**

| | Raised (what an operator gets) | Signature seen |
|---|---:|---:|
| Falls detected | **0 / 30** | 29 / 30 (96.7%) |
| Daily activities flagged | 0 / 40 | 23 / 40 (57.5%) |

**At the shipped settings this detector raises nothing on this dataset.** Three separate
causes compound, and all three were measured rather than guessed:

1. **The clips end before the confirmation window closes.** A URFall clip stops about a
   second after the person lands; `settle_seconds` is 3.0. Re-running with
   `--settle-seconds 1.0` still raises nothing, so this is real but not sufficient on
   its own.
2. **A third of falls never register as a descent at all.** Peak descent rate over 15
   fall clips: median **0.91** body heights/second, 25th percentile 0.59, minimum 0.32.
   `min_descent_rate` is 0.7, so **10 of 15 clear it**. A fall that does not is treated
   as somebody lying down deliberately — `down` with no timer started, which by design
   can never complete.
3. **The signal itself overlaps.** The same measurement over 15 daily-activity clips:
   median 0.64, 75th percentile 0.73, maximum 1.38. **4 of 15 exceed the same
   threshold.** Falls and sitting down are not separated by how fast a bounding box
   centre drops; the distributions sit on top of each other.

Point 3 is the one that matters, and it is why lowering the threshold is not the fix.
**The signature alone is not a fall detector, and that is the point of ADR 12**: 23 of
the 40 daily-activity clips produce the whole upright-descent-horizontal sequence —
sitting down heavily, lying down deliberately, bending to pick something up. Trading the
confirmation window away to make the first column non-zero buys a 57.5% false-alarm rate,
which is the number that decides whether an operator keeps the feature switched on.

ADR 12 already names what would change this: a trained action-recognition model, as a
third source with its own `basis`. This measurement is the evidence for that, and it is
the first evidence the project has had either way.

Pose was available on 72.6% of frames, so the "pose is an enhancement, geometry is the
fallback" claim in ADR 12 holds in practice rather than only in principle.

**What this does not say.** It does not say the detector never fires on a real
deployment: a real camera keeps recording after somebody falls, so cause 1 disappears
entirely and the two-thirds of falls that do clear the descent threshold would complete.
It says the detector is **unvalidated end to end on footage that runs long enough to
validate it**, that **a third of real falls do not look like falls to it**, and that the
geometric signal it rests on does not separate falling from sitting down on its own.

## What has **not** been measured

Read this before quoting anything above.

- **§14's 20–50 camera target is not met at full frame rate on this footage.** Twenty
  cameras drop 90% of frames. The engine stays up, publishes events and never crashes —
  `_LatestSlot` is doing its job — but a camera processing one frame in ten is not being
  watched the way an operator would assume. The honest figure on this box, this clip and
  these settings is **1 camera clean, roughly 3–4 cameras at acceptable drop rates, and
  10+ only with frame sampling and an explicit decision about what a 60% drop rate means
  for the detectors that depend on frame-to-frame continuity.**
- **Frame sampling was measured and did not help as expected.** `--detect-every-n 5` at
  20 cameras gave 36.2 fps and 50.4% drops — fewer drops, but no more detections,
  because decode still happens for every frame. Sampling reduces inference load, not
  decode load, and decode is the constraint.
- **1080p camera counts.** The scaling sweeps used 640×360 footage. Do not assume these
  counts hold at 1080p.
- **RTSP.** Every measurement replays a local file. Network jitter, reconnects and
  variable-bitrate streams are not represented.
- **Face verification accuracy by demographic group.** Detection rate, quality-bar pass
  rate and impostor similarity per group are measured above. Verification accuracy per
  group is not, and cannot be with publicly reachable data — FairFace has no identity
  labels, and the datasets that do (RFW, DemogPairs) require an institutional
  application.
- **Face recognition on surveillance geometry.** LFW is portrait photography. Nothing
  here measures recall on a face eight metres down a corridor, which is the case a site
  actually runs.
- **Abandoned object and camera tamper against real incidents.** Falls and faces now
  have public datasets behind them; these two do not. No real abandoned bag and no real
  obstructed lens has been through them. Zone and line crossing are the exception — both
  fired on real pedestrians in the Avenue clip.
- **Fall detection on footage that outlasts the fall.** See the URFall section: no
  public dataset this harness can consume keeps recording long enough for the
  confirmation window to close.
- **Anything at 1080p beyond single-model latency.** Every camera-count sweep on this
  page used 640x360 footage.
- **§33's end-to-end alert latency.** The benchmark substitutes in-memory stand-ins for
  RabbitMQ and MinIO, so it cannot answer it. The floor is structural and can be stated
  from the parts: `settle_seconds` (3.0 s default for a fall) + VLM describe (~3.3 s) +
  clip post-roll (5.0 s default) + broker round trip. **The 0.5–2 s target in §33 is not
  achievable with these defaults**, and most of the budget is the two configured waits
  rather than inference.
- **Fall detection accuracy.** The state machine is exercised exhaustively against
  scripted geometry and end to end through the real frame loop. It has **never been run
  against real fall footage.** Its false-positive and false-negative rates are unknown.
  See [operations](operations.md#fall-detection-read-this-before-you-rely-on-it-either).
- **VRAM is reported as a marginal cost, and the process total under-reports.** With
  all three models resident `/health` reports 452 + 2364 + 11 = 2827 MiB while
  `nvidia-smi` attributes about 4.1 GiB to the process. The difference is the CUDA
  context, cuDNN workspaces and allocator fragmentation, which no torch API attributes
  to a model. Per-model figures answer "can I fit one more", which is what
  `plan_residency()` asks; `SENTINEL_VRAM_RESERVED_MIB` covers the rest and is
  load-bearing rather than a rounding allowance.
- **CPU and GPU utilisation percentages** are not captured by the harness. `nvidia-smi`
  showed 35% GPU during a 5-camera full-capability run, which is consistent with the
  CPU-bound conclusion above, but it is a spot reading rather than a measurement.
