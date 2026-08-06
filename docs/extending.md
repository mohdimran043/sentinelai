# Extending SentinelAI

Every extension point already exists. This document says where they are and
what a new implementation has to satisfy.

The shape is always the same: **implement a port, add it to the composition
root, prove it against the port-contract tests.** Nothing in `domain/` or
`ports/` should need to change to add a source, a model, or a publisher — if it
does, that is a signal the port is wrong, not that you should edit the domain.

| I want to add… | Implement | In |
|---|---|---|
| A video source (USB, ONVIF, WebRTC, S3 replay) | `FrameSource` | `adapters/sources/` |
| A detector | `ObjectDetector` (+ `ModelRuntime` if the registry should own its VRAM) | `adapters/detectors/` |
| A vision model (Gemini, GPT, LLaVA, Gemma Vision) | `VisionLanguageModel` (+ `ModelRuntime`) | `adapters/vision/` |
| A tracker | `Tracker` | `adapters/trackers/` |
| An event publisher (Kafka, webhook, NATS) | `EventPublisher` | `adapters/publishers/` |
| A welfare notifier (SMS, pager, Matrix, ntfy) | `Notifier` | `adapters/notifiers/` |
| A clip store (S3, local disk, NAS) | `ClipWriter` + `ClipHandle` | `adapters/storage/` |
| An escalation trigger | a pure predicate | `domain/policy/triggers.py` |
| A console section | an entry + a page | `web/src/routes/recorder/` |

## Rules that apply to every adapter

1. **Import heavy dependencies lazily**, inside the method that needs them —
   never at module scope. `Yolo11Detector` and `Qwen25VLDescriber` both do this.
   It is what lets their pure helpers be unit-tested on CPU in CI with no `gpu`
   extra installed, and it is why `main.py` can be imported without a GPU.
2. **Anything blocking goes to an executor.** PyAV demux and model inference are
   blocking C calls. The `async def` on the port is the promise that you have
   handled the offload; `CameraRunner` will not add a second executor hop.
3. **Validate at the boundary.** `FrameData.pixels` is typed `object` because
   `ports/` may not import numpy. `isinstance`-check it and raise `TypeError`.
4. **Never read a clock in a pure helper** you want to test. Take `now: float`.
5. **Fail loudly at startup, degrade gracefully at runtime.** A misconfigured
   adapter should raise during `initialize()`. A transient runtime failure
   should not take the camera down.

## Worked example 1: a new video source

Say you want `UsbSource` for a locally attached camera.

### The contract

```python
class FrameSource(ABC):
    def __aiter__(self) -> AsyncIterator[FrameData]: ...
    def packets(self) -> AsyncIterator[EncodedPacket]: ...
    async def close(self) -> None: ...
```

Three obligations that are easy to miss and will break things downstream:

- **`packets()` must be the same demux pass as `__aiter__`**, not a second
  read of the device. The pre-roll buffer and clip writer line packets up
  against frames on a shared timeline.
- **`FrameData.timestamp` and `EncodedPacket.pts` must be the same timeline.**
  `CameraRunner` has no clock of its own — it takes every time-valued decision
  from the frame or packet in hand. Two timelines here is
  [the bug that silences a camera](decisions.md#5-occurred_at-is-unix-epoch-with-a-per-camera-anchor).
- **Every demuxed packet is offered to `packets()`**, even though only sampled
  frames get decoded. Clips are remuxes of those packets.

If your source can reconnect, reset `frame_index` to 0 on reconnect and keep
the timestamp clock advancing. `CameraRunner._is_timeline_regression` treats a
`frame_index` that drops below its predecessor as a discontinuity, and resets
the tracker, motion analyzer, gate state and pre-roll accordingly. This is the
existing, tested contract — do not invent a different discontinuity signal.

### Model it on `FileSource`

`adapters/sources/file.py` is the simplest complete implementation. `rtsp.py`
adds reconnect via a separately testable `_ReconnectLoop` that takes an injected
clock and sleep function, so backoff sequencing is unit-tested with no network.
Copy that structure — it is what makes reconnect logic CI-safe.

### Wire it up

`main.py`'s `_build_source` selects on URL scheme: `rtsp://`/`rtsps://` builds
`RtspSource`, anything else builds `FileSource`. Add your scheme there.

### The tests it must pass

```bash
cd ai-engine && .venv/bin/python -m pytest tests/ports/test_port_contracts.py -v
```

Add your class to the parametrised list so `test_fakes_satisfy_their_ports` and
`test_every_port_is_abstract_and_cannot_be_instantiated` cover it. Then follow
`tests/adapters/test_file_source.py` for behaviour: frames and packets share a
timeline, `close()` is idempotent, the stream terminates.

If your source needs real hardware, mark those tests `@pytest.mark.integration`
so CI deselects them — and make sure the *pure* parts (backoff, URL parsing,
frame indexing) are testable without it.

## Worked example 2: a new vision model

Say you want `GeminiVisionDescriber` — a hosted API rather than a local model.

### The contract

```python
class VisionLanguageModel(ABC):
    async def describe(self, request: VisionRequest) -> SceneDescription: ...
```

`VisionRequest` carries `keyframe`, `scene`, `history` (the last 5 escalation
details), `camera_label` and `reason_detail`. `SceneDescription` returns
`description`, `threat_value` (0.0–1.0) and `suggested_action`.

### Do you also need `ModelRuntime`?

**Yes if the model occupies VRAM the orchestrator must manage.** `ModelRuntime`
is how a model enters the registry, gets warmed, health-checked, VRAM-budgeted
and idle-evicted. `Qwen25VLDescriber` implements both.

**No if the model runs elsewhere.** A hosted API has no VRAM footprint, so
`plan_residency()` has nothing to plan. Implement only `VisionLanguageModel`
and skip the registry — but then nothing reports it on `/health`, so add your
own health signal if operators need one.

### Three obligations `Qwen25VLDescriber` teaches

1. **Never let a malformed reply reach the event.** Any parse failure — no JSON,
   invalid JSON, wrong types, missing key — must fall through to a *fixed*
   `SceneDescription`, never the raw model text. This is not fussiness: a model
   confused enough to ignore the format instruction is exactly the model most
   likely to answer "As Qwen2.5-VL, I can see…", and published events must
   carry no model identity. Sanitising raw text was considered and rejected —
   it means enumerating every way a model might name itself, an open-ended
   problem, where a fixed fallback cannot leak by construction. Log the raw text
   at DEBUG; logs are operator infrastructure, not the published event.
2. **Distinguish the two failure modes.** A malformed *success* is the fallback
   description above. A timeout or OOM is an *infrastructure* failure and
   surfaces as `Event.description_unavailable=True`. Do not conflate them.
3. **Release VRAM properly in `shutdown()`.** `gc.collect()` **before**
   `torch.cuda.empty_cache()`, and run both on a worker thread. An `nn.Module`
   graph is full of reference cycles, so dropping the reference does not free it
   under refcounting alone. Measured: `empty_cache()` alone left ~2.4 GB
   reserved; adding `gc.collect()` first dropped it to ~54 MiB. The 600 s idle
   unload depends on that being a real reclaim — a stale reservation starves
   `plan_residency()` of VRAM it believes it just got back. The full collect
   goes on a worker thread because it blocks its caller for the duration, and
   the idle sweep can fire while other cameras are running.

### Register it

In `main.py`, build your describer and register a `ModelSpec` with a
`vram_mib` seed, a `priority` (the detector's is 100, the VLM's 50 — higher
survives eviction), and `idle_unload_seconds` (`None` = never evict). Then
`refresh_specs_from_capabilities()` replaces the seed with the measured figure
after warmup.

### The tests it must pass

- `tests/ports/test_port_contracts.py` — add a `issubclass` assertion alongside
  `test_qwen25vl_describer_satisfies_vision_llm_and_model_runtime`. These are
  class-level, so they need no weights and no GPU.
- Follow `tests/adapters/vision/test_qwen25vl.py` for the pure parts: prompt
  construction and response parsing are plain functions and must be tested
  without loading anything. **Test the malformed-reply paths explicitly** —
  empty string, prose with no JSON, JSON with a string `threat_value`, JSON with
  a trailing aside containing braces.
- Real inference goes behind `@pytest.mark.gpu`.

## Adding an escalation trigger

The cheapest extension in the codebase, because it is pure.

Write a function in `domain/policy/triggers.py`:

```python
def loitering_after_hours(ctx: TriggerContext) -> TriggerOutcome:
    if not ctx.profile.some_new_field:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.LOITERING_AFTER_HOURS,
        detail="...",           # human-readable, ends up in telemetry and the VLM prompt
    )
```

Then:

1. Add the value to `EscalationReason` in `domain/entities.py`.
2. Add it to the `reason` enum in
   `contracts/events/anomaly_event.schema.json` — **the event will fail
   validation at publish time otherwise.** Regenerate the web types
   (`cd web && npm run gen`).
3. Insert it in `ALL_TRIGGERS` at the right **priority position**. The tuple
   order is the priority order: when several fire, the first one names the
   reason. Put a specific, actionable trigger above a general one.
4. Add any tuning knobs to `CameraProfile`, with validation in `__post_init__`.
5. Test it in `tests/domain/policy/test_triggers.py`. No fixtures, no mocks, no
   clock — construct a `TriggerContext` and assert. This is the payoff of
   [ADR 6](decisions.md#6-domain-and-ports-are-pure-and-a-test-enforces-it).

**Constraints.** The predicate must be pure: no clock reads (`ctx.scene.timestamp`
is your `now`), no I/O, no imports outside the allowlist. `SceneState` carries
**no pixel data** by design — the gate must be decidable from cheap signals
alone, because deciding it is what avoids paying for the expensive ones. If your
trigger needs pixels, it belongs in the pipeline stage that computes
`motion_energy` and `scene_signature`, not in the gate.

## Adding a publisher, notifier, or clip store

**Publisher** (`EventPublisher`): the one rule that matters is **every failure
must raise**. `VlmScheduler` treats a raise as "the publisher did not take this
event" and hands it to `FailedEventSink`. A publisher that logs and returns
drops the event past the last component able to save it. If your transport can
be merely *down* (as opposed to rejecting), spool to disk and replay — see
`adapters/publishers/rabbitmq.py`, and note that a spool is only half a
guarantee without something replaying it (`main.BrokerLink`).

**Notifier** (`Notifier`): the rule is the exact **opposite** of the publisher's
— `notify()` must **never raise**. Read that twice before writing one, because
the two ports look alike and their failure contracts are inverted. A publisher
that swallows an error loses the only durable record of an event; a notifier
that raises takes down the pipeline it exists to observe. Best-effort commentary
is the whole job.

Concretely, from `adapters/notifiers/webhook.py`, which is the worked example:

- **Bound every attempt, and bound the whole call.** A hanging endpoint must not
  park the single dispatch worker. Retry transient failures (5xx, 429, 408,
  connection errors) with backoff; **never retry a 4xx** — a 400 will be 400
  again, and retrying a 401 just replays a rejected credential.
- **Spool, do not drop.** A note that cannot be delivered goes to the existing
  dead-letter writer with `record_type: "WelfareNote"`, not into a log line and
  oblivion. Reuse `adapters/publishers/dead_letter.py`; do not write a second one.
- **Treat a destination URL as a credential.** ntfy and Slack put tokens in the
  path. Do not log it, do not follow redirects (a redirect target could re-send
  it to another host), and if your client library logs request URLs, expose an
  explicit redaction installer for the composition root to call — a hidden side
  effect of constructing the adapter is one nobody can find later.
- **Send only what routed.** `concerns_to_notify` has already filtered to the
  kinds this camera is configured for; forwarding the whole assessment would
  leak exactly what `notify_on` exists to suppress.

**Clip store** (`ClipWriter`/`ClipHandle`): `open()` returns a handle, then
`append`/`finish`/`abort` stream packets through. Do not buffer the clip in
memory — 3 s pre-roll plus 5 s post-roll at 1080p30 is roughly 1.1 GB of raw
frames per concurrent clip. Streaming encoded packets keeps peak memory at the
pre-roll ring plus one packet. **`abort()` must not raise**: it runs on shutdown
paths, and a pipeline shutdown mid-clip must not turn cleanup into a second
failure. If you are writing MP4, read
[ADR 3](decisions.md#3-clips-are-remuxed-never-re-encoded) first, especially the
monotonic-DTS guard.

## Adding a console section

> `web/` currently has several agents working in it. Coordinate before editing
> shared files — `sections.ts`, `App.tsx` and the recorder mock fixtures have
> been clobbered by concurrent edits before.

The recorder console is registry-driven. `web/src/routes/recorder/sections.ts`
holds the list; read its header comment, which already documents the convention.

1. Add a `RecorderSection` entry:

```ts
{
  path: 'retention',
  label: 'Retention',
  built: false,
  endpoints: ['GET /api/retention'],
  summary: 'How long footage is kept per camera, and what has already aged out.',
}
```

2. Ship it with **`built: false`** until the page really works. That is not a
   TODO marker, it is a product rule: the route exists and the rail shows the
   section, but the screen behind it is a *stated* placeholder
   (`SectionNotBuiltPage.tsx`) rather than a working page. The rail renders a
   literal `soon` badge, and `AppShell.test.tsx` asserts the badge count equals
   `RECORDER_SECTIONS.filter(s => !s.built).length` — so the honesty marker is
   test-enforced against the registry, not hand-maintained. As the file puts it:
   a nav item that silently leads nowhere useful is the same lie as a tile that
   renders a disabled capability as if it worked.
3. `endpoints` and `summary` are shown on the placeholder, so the next person
   knows where to start. Write them for that reader.
4. Build the page, flip `built: true`, and remove nothing.

Two conventions to follow in the page itself:

- **Never render a fabricated zero.** If data is missing, use `Absent` (states
  what is missing and why) or `Notice`. An unreachable backend gets an explicit
  banner, not blank tiles.
- **Never render a raw enum.** `severity` and `reason` arrive as snake_case and
  go through `humanizeEnum` (`web/src/lib/format.ts`). Telemetry numerics use
  the `.readout` class so tabular figures do not jitter on update.

Tone mapping lives in exactly two files and must stay there:
`web/src/lib/severity.ts` (engine) and `web/src/recorder/tone.ts` (recorder).
Nothing under `src/recorder/` may import from `src/api/`, or the reverse — two
products, two clients, two type sets.

## Before you commit

```bash
cd ai-engine
.venv/bin/python -m pytest -q -m "not gpu and not integration"   # whole CPU suite, seconds
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy                                                    # strict, over src AND tests

cd ../web && npm run typecheck && npm run test                    # not yet gated by CI
```

Or `make check` for the Python side. Note `filterwarnings = ["error"]` is set in
`pyproject.toml`, not as a command-line flag — a new warning is a test failure
for everyone, including your IDE. `mypy` covers `tests/` too, deliberately: the
fakes are CI's stand-ins for every real adapter, so port-signature drift in them
is exactly what static checking should catch.
