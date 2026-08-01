# SentinelAI Phase 1B — AI Engine Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Phase 1A's pure core into a running system — ingest RTSP, detect and track with YOLO11s + ByteTrack, escalate through the Phase 1A gate, describe with Qwen2.5-VL, and publish a validated event with an MP4 evidence clip that starts 3 s before the trigger.

**Architecture:** Clean architecture, continued from Phase 1A. `domain/` and `ports/` stay pure and are enforced by architecture fitness tests. This phase builds only outward: `adapters/` implement the ports, `orchestrator/` owns model lifecycle and the VLM queue, `pipeline/` runs one async task per camera, and `api/` is a thin FastAPI delegation layer. The vertical slice is built with Phase 1A's fakes first (Tasks 4–10, all CI-green on CPU), then real models are substituted one at a time (Tasks 11–14).

**Tech Stack:** Python 3.12, PyAV (CPU decode + MP4 remux), numpy, Ultralytics YOLO11, supervision ByteTrack, transformers + autoawq (Qwen2.5-VL-3B-Instruct-AWQ), aio-pika, MinIO SDK, FastAPI, uvicorn, pytest, ruff, mypy, Docker Compose, mediamtx.

**Reference spec:** `docs/superpowers/specs/2026-08-01-sentinelai-phase1b-ai-engine-slice-design.md`
**Predecessor plan:** `docs/superpowers/plans/2026-07-31-phase1a-foundation-and-ai-engine.md`

## Global Constraints

- Python **3.12** exactly; `ai-engine/` targets `py312` in ruff/mypy config.
- `sentinel_ai/domain/` and `sentinel_ai/ports/` MUST NOT import any third-party I/O library, MUST NOT read the clock, and MUST NOT import `sentinel_ai.config` or any outer layer (`adapters`, `orchestrator`, `pipeline`, `api`). `tests/test_architecture.py` enforces all of this and now also catches aliased imports, from-imports, dynamic imports, and relative outer-layer imports. **Do not weaken any existing fitness rule.** numpy is forbidden in `domain/` and `ports/`; it is permitted from `pipeline/` and `adapters/` outward.
- All timestamps crossing into `domain/` are **float seconds from a monotonic clock**, passed in explicitly.
- Development Mode is the **default** (`SENTINEL_MODE=development`). gRPC / Production Mode is NOT built in this phase.
- Default Vision LLM: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`. 7B must never be a default.
- VRAM ceiling **8192 MiB**; reserved headroom **2048 MiB**.
- Every test that requires a CUDA GPU or downloaded model weights MUST be marked `@pytest.mark.gpu`. Every test that requires docker-compose services MUST be marked `@pytest.mark.integration`. CI runs `-m "not gpu and not integration"`.
- Events carry **no model identity** (spec §3.3).
- Clip pre-roll **3.0 s**; clip post-roll **5.0 s**.
- Ruff: line-length 100, selects `E,F,I,UP,B,SIM,ANN,RUF`, `tests/**` exempt from `ANN`. Mypy `strict` over both `sentinel_ai` and `tests`.
- Nothing in CI may sleep, hit the network, or require a GPU. `FileSource(realtime=False)` is the only source used in CPU tests.
- Test output must stay pristine — the suite passes under `-W error`. A new warning is a defect.
- Commit after every task, Conventional Commits format. Work directly on `master`; no branches.

## Baseline at the start of this plan

`master` at `5751a11`. 188 tests passing, ruff clean, mypy clean over 39 files. Phase 1A delivered:

- `domain/entities.py` — `BBox`, `Detection`, `Track`, `SceneState`, `EscalationReason`, `Severity`, `ThreatScore`, `Event`
- `domain/camera_profile.py` — `CameraProfile` (includes `cooldown_seconds: float = 5.0`)
- `domain/policy/` — `TokenBucket`, the seven triggers, `decide()`/`force()`, `plan_residency()`
- `ports/` — `ModelRuntime`, `ObjectDetector`, `Tracker`, `VisionLanguageModel`, `FrameSource`, `EventPublisher`, `ClipWriter`
- `adapters/serialization/event_codec.py` — `encode_event` (validates), `decode_event`, `validate_payload`
- `contracts/events/anomaly_event.schema.json`
- `tests/fakes/` — `FakeDetector`, `FakeTracker`, `FakeVisionLLM`, `FakeSource`, `FakePublisher`, `FakeClipWriter`

## File Structure

| File | Responsibility |
|---|---|
| `deploy/compose/docker-compose.core.yml` | Postgres, Redis, RabbitMQ, MinIO, mediamtx |
| `deploy/compose/mediamtx.yml` | Source registry — the `avenue_01` path |
| `ai-engine/sentinel_ai/config.py` | **Modify** — new settings (see Shared Interfaces) |
| `ai-engine/sentinel_ai/ports/frame_source.py` | **Modify** — add `EncodedPacket`, `FrameSource.packets()` |
| `ai-engine/sentinel_ai/ports/clip_writer.py` | **Modify** — `ClipWriter.open()` + `ClipHandle` ABC |
| `ai-engine/sentinel_ai/ports/detector.py` | **Modify** — docstring only: `batch_max` is Phase 2 |
| `ai-engine/sentinel_ai/adapters/sources/preroll.py` | `PreRollBuffer` — keyframe-aligned encoded ring |
| `ai-engine/sentinel_ai/adapters/sources/file.py` | `FileSource` — PyAV demux, dual frame/packet fan-out |
| `ai-engine/sentinel_ai/adapters/sources/rtsp.py` | `RtspSource` — same, plus backoff reconnect |
| `ai-engine/sentinel_ai/pipeline/stages/motion.py` | `MotionAnalyzer` — motion energy + scene signature |
| `ai-engine/sentinel_ai/pipeline/runner.py` | `CameraRunner` — one async task per camera |
| `ai-engine/sentinel_ai/orchestrator/registry.py` | `ModelRegistry` — discovery + §5 lifecycle states |
| `ai-engine/sentinel_ai/orchestrator/resident_set.py` | `ResidentSet` — executes `plan_residency()`, idle-unload |
| `ai-engine/sentinel_ai/orchestrator/admission.py` | `AdmissionGate` — process-wide GPU admission |
| `ai-engine/sentinel_ai/orchestrator/scheduler.py` | `VlmScheduler` — bounded queue, one worker |
| `ai-engine/sentinel_ai/orchestrator/service.py` | `EngineService` — the single entry point |
| `ai-engine/sentinel_ai/adapters/publishers/inmemory.py` | `InMemoryPublisher` |
| `ai-engine/sentinel_ai/adapters/publishers/rabbitmq.py` | `RabbitMQPublisher` + disk spool |
| `ai-engine/sentinel_ai/adapters/storage/minio_clips.py` | `MinioClipWriter` + `MinioClipHandle` |
| `ai-engine/sentinel_ai/adapters/detectors/yolo11.py` | `Yolo11Detector` |
| `ai-engine/sentinel_ai/adapters/trackers/bytetrack.py` | `ByteTrackTracker` |
| `ai-engine/sentinel_ai/adapters/vision/qwen25vl.py` | `Qwen25VLDescriber` |
| `ai-engine/sentinel_ai/api/app.py` | FastAPI app factory + lifespan |
| `ai-engine/sentinel_ai/api/routes.py` | Endpoints — thin delegation only |
| `ai-engine/sentinel_ai/api/schemas.py` | Pydantic request/response models |
| `datasets/download_sample.sh` | Public sample fetch + licence note |
| `datasets/README.md` | Provenance, licence, ethical constraint |

---

## Shared Interfaces

**Every task depends on this section.** These names, signatures and types are fixed. Do not
rename, re-shape, or "improve" them — later tasks are written against them verbatim.

### S1. New settings on `Settings` (`sentinel_ai/config.py`)

Added in Task 3 as one block, so no later task edits `config.py` for these:

```python
    clip_postroll_seconds: float = Field(default=5.0, gt=0)

    detector_conf_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    detector_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    detector_imgsz: int = Field(default=640, gt=0)

    vlm_queue_maxsize: int = Field(default=4, ge=1)
    vlm_timeout_seconds: float = Field(default=30.0, gt=0)
    vlm_max_new_tokens: int = Field(default=256, gt=0)
    vlm_global_concurrency: int = Field(default=1, ge=1)
    vlm_global_min_interval_seconds: float = Field(default=2.0, ge=0)

    detect_every_n_frames: int = Field(default=1, ge=1)
    source_realtime: bool = True
    rtsp_reconnect_initial_seconds: float = Field(default=1.0, gt=0)
    rtsp_reconnect_max_seconds: float = Field(default=30.0, gt=0)

    clip_temp_dir: str = "./var/clips"
```

### S2. `EncodedPacket` (`sentinel_ai/ports/frame_source.py`, Task 3)

```python
@dataclass(frozen=True, slots=True)
class EncodedPacket:
    """A demuxed, still-encoded packet. `data` is bytes so ports stay codec-agnostic."""

    camera_id: str
    data: bytes
    pts: float
    is_keyframe: bool
    codec: str
```

`pts` is float seconds on the same monotonic timeline as `FrameData.timestamp`.

`codec` is the container's short codec name, lowercase — `"h264"`, `"hevc"`. It is a plain
`str`, not a library enum, so no codec library enters the ports layer. Sources set it from
the demuxed stream; the clip writer uses it to construct its parser and output stream instead
of assuming H.264. Without it the remux is silently wrong for any non-H.264 source — it would
drop every packet rather than fail — which is the kind of defect that surfaces as "clips are
empty" months later.

### S3. `FrameSource` (`sentinel_ai/ports/frame_source.py`, Task 3)

```python
class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...

    @abstractmethod
    def packets(self) -> AsyncIterator[EncodedPacket]: ...

    @abstractmethod
    async def close(self) -> None: ...
```

### S4. `ClipWriter` / `ClipHandle` (`sentinel_ai/ports/clip_writer.py`, Task 3)

```python
class ClipHandle(ABC):
    @abstractmethod
    async def append(self, packet: EncodedPacket) -> None: ...

    @abstractmethod
    async def finish(self) -> str:
        """Finalise the clip and return its URI."""

    @abstractmethod
    async def abort(self) -> None:
        """Discard a partial clip. MUST NOT raise."""


class ClipWriter(ABC):
    @abstractmethod
    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle: ...
```

### S5. `PreRollBuffer` (`adapters/sources/preroll.py`, Task 4)

```python
class PreRollBuffer:
    def __init__(self, preroll_seconds: float) -> None: ...

    def append(self, packet: EncodedPacket) -> None:
        """Add a packet and evict everything older than the keyframe-aligned horizon."""

    def flush(self) -> tuple[EncodedPacket, ...]:
        """Return buffered packets from the oldest keyframe at or before the horizon.

        Returns () when no keyframe has been seen yet.
        """

    def clear(self) -> None: ...

    @property
    def span_seconds(self) -> float:
        """pts span currently held; 0.0 when empty."""
```

### S6. `MotionSignals` + `MotionAnalyzer` (`pipeline/stages/motion.py`, Task 5)

```python
SIGNATURE_BINS: int = 16


@dataclass(frozen=True, slots=True)
class MotionSignals:
    motion_energy: float
    scene_signature: tuple[float, ...]


class MotionAnalyzer:
    def __init__(self, bins: int = SIGNATURE_BINS) -> None: ...

    def analyze(self, frame: FrameData) -> MotionSignals:
        """Motion energy vs the previous frame, plus a normalised luma histogram.

        The first frame yields motion_energy == 0.0.
        """

    def reset(self) -> None:
        """Drop the previous frame — call on stream discontinuity."""
```

`motion_energy` is in [0, 1]. `scene_signature` sums to 1.0 and has length `bins`.

### S7. `EscalationRequest` (`orchestrator/scheduler.py`, Task 7)

What the runner hands the scheduler. Carries everything the VLM and the event need, so the
worker never reaches back into the pipeline:

```python
@dataclass(frozen=True, slots=True)
class EscalationRequest:
    camera_id: str
    event_id: UUID
    reason: EscalationReason
    detail: str
    scene: SceneState
    keyframe: FrameData
    profile: CameraProfile
    camera_label: str
    history: tuple[str, ...]
    clip: ClipHandle | None
```

### S8. `AdmissionGate` (`orchestrator/admission.py`, Task 7)

```python
class AdmissionGate:
    def __init__(self, concurrency: int, min_interval_seconds: float) -> None: ...

    async def acquire(self, now: float) -> None:
        """Block until a global slot is free AND min_interval has elapsed."""

    def release(self, now: float) -> None: ...

    @property
    def in_flight(self) -> int: ...
```

Clock is injected as an argument, matching the domain convention.

### S9. `VlmScheduler` (`orchestrator/scheduler.py`, Task 7)

```python
class VlmScheduler:
    def __init__(
        self,
        vlm: VisionLanguageModel,
        publisher: EventPublisher,
        admission: AdmissionGate,
        *,
        maxsize: int,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> None: ...

    def submit(self, request: EscalationRequest) -> bool:
        """Non-blocking. Returns False and counts a drop when the queue is full."""

    async def run(self) -> None:
        """The single worker loop. Cancel to stop."""

    async def drain(self) -> None:
        """Await completion of queued work — tests only."""

    @property
    def dropped(self) -> int: ...
```

### S10. `ModelRegistry` / `ResidentSet` (`orchestrator/`, Task 7)

**`ModelSpec` already exists** — `sentinel_ai.domain.policy.vram_budget.ModelSpec`, with fields
`model_key: str`, `vram_mib: int`, `priority: int`, `idle_unload_seconds: float | None`. Import
it; do **not** define a second one. `plan_residency()` consumes exactly that type, so a parallel
registry-local spec would have to be converted on every call for no benefit. There is no `kind`
field: a model's kind is already on `ModelRuntime.capabilities().kind`, which is where the
registry reads it from.

`plan_residency` is keyword-only:

```python
plan_residency(
    *, specs: Mapping[str, ModelSpec], currently_resident: Sequence[str],
    required: Sequence[str], last_used_at: Mapping[str, float],
    now: float, total_mib: int, reserved_mib: int,
) -> ResidencyPlan  # .load, .unload, .resident, .free_mib — all tuples but free_mib
```

It raises `InsufficientVram` when a required model cannot fit even with everything evictable
unloaded, and `KeyError` for an unknown model key.

```python
class ModelRegistry:
    def register(self, spec: ModelSpec, runtime: ModelRuntime) -> None: ...
    def get(self, key: str) -> ModelRuntime: ...
    def state(self, key: str) -> LifecycleState: ...
    def specs(self) -> Mapping[str, ModelSpec]: ...
    def health(self) -> dict[str, HealthReport]: ...


class ResidentSet:
    def __init__(self, registry: ModelRegistry, total_mib: int, reserved_mib: int) -> None: ...
    async def ensure(self, required: tuple[str, ...], now: float) -> None:
        """Apply plan_residency(): load required, evict what must go."""
    async def sweep_idle(self, now: float) -> None: ...
    def resident(self) -> frozenset[str]: ...
```

### S11. `CameraRunner` (`pipeline/runner.py`, Task 6)

```python
@dataclass(frozen=True, slots=True)
class CameraTelemetry:
    camera_id: str
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    last_frame_at: float | None
    last_escalation_at: float | None


class CameraRunner:
    def __init__(
        self,
        *,
        camera_id: str,
        camera_label: str,
        source: FrameSource,
        detector: ObjectDetector,
        tracker: Tracker,
        motion: MotionAnalyzer,
        profile: CameraProfile,
        scheduler: VlmScheduler,
        clip_writer: ClipWriter | None,
        preroll: PreRollBuffer,
        clock: Callable[[], float],
        detect_every_n_frames: int = 1,
    ) -> None: ...

    async def run(self) -> None: ...
    async def describe_now(self) -> UUID:
        """USER_REQUESTED path — uses domain force(); returns the event id."""
    def telemetry(self) -> CameraTelemetry: ...
```

### S12. `EngineService` (`orchestrator/service.py`, Task 7)

```python
class EngineService:
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def cameras(self) -> tuple[CameraTelemetry, ...]: ...
    def telemetry(self, camera_id: str) -> CameraTelemetry: ...
    def health(self) -> dict[str, HealthReport]: ...
    async def describe_now(self, camera_id: str) -> UUID: ...
```

`camera_id` not found raises `UnknownCameraError` (defined in `orchestrator/service.py`); the
API maps it to HTTP 404.

### S13. Adapter constructors (Tasks 8–13)

```python
class InMemoryPublisher(EventPublisher):
    def __init__(self) -> None: ...
    @property
    def events(self) -> tuple[Event, ...]: ...


class RabbitMQPublisher(EventPublisher):
    def __init__(self, url: str, exchange: str, spool_dir: Path) -> None: ...
    async def connect(self) -> None: ...
    async def replay_spool(self) -> None: ...


class MinioClipWriter(ClipWriter):
    def __init__(
        self, endpoint: str, access_key: str, secret_key: str,
        bucket: str, secure: bool, temp_dir: Path,
    ) -> None: ...


class Yolo11Detector(ObjectDetector, ModelRuntime):
    def __init__(self, model_id: str, conf: float, iou: float, imgsz: int, device: str) -> None: ...


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: int = 30) -> None: ...


class Qwen25VLDescriber(VisionLanguageModel, ModelRuntime):
    def __init__(self, model_id: str, max_new_tokens: int, device: str) -> None: ...
```

`Yolo11Detector` and `Qwen25VLDescriber` implement **both** their typed port and `ModelRuntime`,
so the registry can manage their lifecycle while the pipeline uses the typed interface.
`ModelRuntime.predict()` on these delegates to the typed method.

### S14. Escalation → event assembly

Owned by the scheduler worker (Task 7), so exactly one place builds an `Event`:

1. `admission.acquire(now)`
2. `vlm.describe(VisionRequest(...))` under `asyncio.timeout(vlm_timeout_seconds)`
3. On success → `Event(... description=..., threat=ThreatScore.from_value(...), description_unavailable=False)`
4. On `TimeoutError` or any VLM exception → same `Event` with a metadata-derived description
   and `description_unavailable=True`. **The event is still published.**
5. Clip: `await clip.finish()` → `clip_uri`; on failure log and publish with `clip_uri=None`
6. `await publisher.publish(event)`
7. `admission.release(now)` in a `finally`

---

## Reconciliation Log

This plan's task bodies were drafted in parallel against the Shared Interfaces above. Where a
draft disagreed with the contract or with another draft, the conflict and its resolution are
recorded here rather than silently patched — an executor hitting one of these should know it
was decided, not overlooked.

| # | Conflict | Resolution |
|---|---|---|
| R1 | S2's `EncodedPacket` had four fields; the clip writer needs to know what it is remuxing, and was assuming H.264 | **`codec: str` added** to `EncodedPacket`. `MinioClipWriter` gates on `SUPPORTED_CODECS = {"h264", "hevc"}` and raises `UnsupportedCodecError` on the first packet of anything else, rather than parsing forever and finalising a valid-but-empty MP4. A mid-clip codec change is rejected for the same reason. |
| R2 | S10 defined a `ModelSpec` with `key`/`kind` that duplicated `domain.policy.vram_budget.ModelSpec` (`model_key`, no `kind`) | **The domain type is reused.** `plan_residency()` consumes exactly it, so a registry-local twin would need converting on every call. A model's kind already lives on `ModelRuntime.capabilities().kind`. |
| R3 | Task 6's runner counts stream discontinuities; S11's `CameraTelemetry` and Task 10's `CameraStatus` had no such field | **`discontinuities: int` added** to both, and to the telemetry endpoint's response. Silent resets are exactly what erodes trust in a monitoring system. |
| R4 | Spec §5.7 lists VRAM on `GET /cameras/{id}/telemetry`, but VRAM is model-scoped, not camera-scoped | **VRAM stays on `/health`**, where `HealthReport.vram_mib` already puts it, and spec §5.7 is amended. Per-camera VRAM attribution is meaningless while one GPU serves every camera. |
| R5 | Task 14's `RtspSource` signals reconnect by resetting `frame_index`, assuming the runner watches for the regression — written before Task 6 existed | **Task 6 implements exactly that**, plus a second trigger (scene-signature length change). Both reset tracker, motion, pre-roll and `GateState`. The contract is now stated explicitly in Task 6 and both signals are tested. |
| R6 | `supervision` sat in the `gpu` extra, but ByteTrack is numpy/scipy with no torch dependency — Task 12's "runs in CI" claim was false | **Moved to the `runtime` extra**, pinned `>=0.24,<0.28`: `sv.ByteTrack` is deprecated from 0.28 and its `DeprecationWarning` would break the suite's `-W error` requirement. Migrating to the `trackers` package is Phase 2. |
| R7 | Tasks 1, 3, 11 and 12 all edit `pyproject.toml` | Their edits touch disjoint regions — Task 1 the `gpu` extra, Task 3 `[project]` settings, Task 11 the mypy overrides, Task 12 the `runtime` extra. Each task's Files block names `pyproject.toml` so the conflict is visible to whoever executes them out of order. |
| R8 | S7–S9 (`EscalationRequest`, `AdmissionGate`, `VlmScheduler`) were attributed to Task 7, but Task 6's runner takes a `VlmScheduler` in its constructor — and every task's verify step re-runs the whole suite, so Task 6 could not reach a green commit against a type that did not exist | **S7–S9 are implemented in Task 6**; Task 7 consumes them and adds `ModelRegistry`, `ResidentSet` and `EngineService`. The interfaces are unchanged — only which task creates the file moved. An earlier draft tried a stub instead and produced a Task 6 that imported a `VlmScheduler` its own stub never defined. |

### Open deviation requiring sign-off

**Spec §6 says a camera disconnect emits an event; this plan does not emit one.**
`EscalationReason` has no connectivity member, and adding one is not a local change: the enum is
mirrored in `contracts/events/anomaly_event.schema.json`, which the Go backend and the React app
will both consume in later phases. Phase 1B therefore covers the "status → offline" half via
telemetry staleness (`last_frame_at` stops advancing) and leaves the "event emitted" half
unbuilt. The natural home is Phase 1C, when a consumer exists that could actually act on such an
event. **This is a knowing deviation from spec §6, not an oversight** — if you want it in 1B
instead, it needs a task that changes the domain enum and the event contract together.

## Task Index

| # | Task | Needs GPU | Runs in CI |
|---|---|---|---|
| 1 | AutoAWQ load spike — decide the VLM path | yes | n/a — throwaway spike |
| 2 | Compose core services + mediamtx config | no | `integration` only |
| 3 | Port changes: `EncodedPacket`, `ClipHandle`, settings | no | yes |
| 4 | `PreRollBuffer` + `FileSource` + synthetic clip fixture | no | yes |
| 5 | `MotionAnalyzer` | no | yes |
| 6 | `CameraRunner` — end-to-end with fakes | no | yes |
| 7 | Orchestrator: registry, resident set, admission, scheduler, service | no | yes |
| 8 | `InMemoryPublisher` + `RabbitMQPublisher` with disk spool | no | yes |
| 9 | `MinioClipWriter` | no | yes, except the MinIO upload |
| 10 | FastAPI surface | no | yes |
| 11 | `Yolo11Detector` | for inference only | partly — see below |
| 12 | `ByteTrackTracker` | no | yes |
| 13 | `Qwen25VLDescriber` | for inference only | partly — see below |
| 14 | `RtspSource` + dataset script + end-to-end demo | yes | no |

**Tasks 11 and 13 are partly CI-testable.** Both lazy-import `torch` / `ultralytics` /
`transformers` inside the methods that need them rather than at module scope, so the modules
import cleanly without the `gpu` extra installed. The port-contract subclass checks, the
Ultralytics-box-to-`Detection` conversion, and Qwen's prompt construction and
malformed-output parsing all run in CI. Only tests that genuinely need CUDA or downloaded
weights carry `@pytest.mark.gpu`.
---

### Task 1: AutoAWQ load spike — decide the VLM path

Throwaway spike (spec §7). Its deliverable is a decision and a measured VRAM number
committed to `docs/superpowers/plans/phase1b-spike-result.md` — not a module, and nothing
under `sentinel_ai/` depends on this task's *code*, only on its written answer. No unit test
exists for this task; "verify" means reading the spike's own log output.

**Files:**
- Modify: `ai-engine/pyproject.toml` (the `gpu` extra and its mypy overrides — exactly one of
  the two branches in Step 4 applies)
- Create: `docs/superpowers/plans/phase1b-spike-result.md`

**Interfaces:**
- Consumes: nothing — this is the first task in the plan.
- Produces: the VLM-loading decision, read by a human implementing Task 13
  (`Qwen25VLDescriber`, S13) to choose between an `autoawq`-backed load and a `transformers` +
  `bitsandbytes` 4-bit NF4 load. `VisionLanguageModel` (unchanged either way) and the
  `Qwen25VLDescriber.__init__(model_id, max_new_tokens, device)` signature from S13 are not
  affected by which branch is taken — that is the entire point of the port abstraction. Also
  produces the finalised `gpu` extra in `pyproject.toml`, which `make install-gpu` and Task 13
  both rely on.

- [ ] **Step 1: Install the `gpu` extra into the existing venv, logging everything**

  The venv at `ai-engine/.venv` already has `[dev]` installed (Phase 1A); `torch` has never
  been pulled in (spec §2.1). `pillow` is added ad hoc here for the probe's synthetic image —
  it is not added to `pyproject.toml` because `qwen-vl-utils` already pulls it in transitively
  for the real adapter, so committing it separately would be a redundant pin.

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI/ai-engine
  . .venv/bin/activate
  LOG=/tmp/spike_qwen_awq.log
  : > "$LOG"
  { pip install -e '.[dev,runtime,gpu]' pillow; echo "PIP_EXIT=$?"; } 2>&1 | tee -a "$LOG"
  ```
  Expected: either it completes with `PIP_EXIT=0`, or `autoawq`'s build step fails (it is a
  compiled CUDA extension against a torch/transformers pin combination its own maintainer
  deprecated in 2025) and `PIP_EXIT` is non-zero. Both outcomes are valid spike results — do
  not troubleshoot a failed `autoawq` build; a failure here already answers the question Step 4
  asks, so proceed straight to Step 3's probe only if `PIP_EXIT=0`, otherwise skip to Step 4's
  failure branch with this log as the evidence.

- [ ] **Step 2: Confirm the hardware the result will be measured against**

  Run:
  ```bash
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv | tee -a /tmp/spike_qwen_awq.log
  ```
  Expected: `NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 580.173.02, 8.9`. This machine's
  desktop compositor is a live GNOME/KDE/etc. session, not a headless X server — the readings
  in Step 3 below are therefore already the "measured under the actual desktop session"
  baseline spec §2.1 requires, without any extra setup: `nvidia-smi` always reports system-wide
  usage, compositor included, so as long as this command is run from a normal logged-in desktop
  (not a fresh SSH session to a display-manager-only boot), the number is real.

- [ ] **Step 3: Run the load-and-infer probe**

  This is the throwaway script — it exists only in `/tmp` for the duration of this step, is
  never added to the repo, and is deleted at the end of the step. It loads
  `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` through plain `transformers` (which auto-detects the AWQ
  quantisation config in the checkpoint and dispatches to `autoawq` as the installed backend —
  there is no separate "autoawq API" to call), runs one synthetic image through it, and takes an
  `nvidia-smi` reading at three checkpoints: idle-desktop baseline, after weights are resident,
  and after inference (the peak).

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI/ai-engine
  . .venv/bin/activate
  SPIKE=$(mktemp /tmp/spike_qwen_awq_probe.XXXXXX.py)
  cat > "$SPIKE" <<'PYEOF'
  """Throwaway spike probe (Phase 1B Task 1). Never imported by sentinel_ai; deleted after run."""

  import subprocess
  import time

  import torch
  from PIL import Image
  from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

  MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


  def smi(label: str) -> None:
      out = subprocess.run(
          ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
          capture_output=True,
          text=True,
          check=True,
      ).stdout.strip()
      print(f"NVIDIA_SMI[{label}]: {out}")


  smi("baseline_desktop_idle")

  image = Image.new("RGB", (640, 480), color=(60, 120, 200))

  processor = AutoProcessor.from_pretrained(MODEL_ID)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      MODEL_ID, torch_dtype="auto", device_map="cuda:0"
  )
  smi("after_model_load")

  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image", "image": image},
              {"type": "text", "text": "Describe this scene in one sentence."},
          ],
      }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

  start = time.monotonic()
  with torch.inference_mode():
      output = model.generate(**inputs, max_new_tokens=64)
  elapsed = time.monotonic() - start
  smi("after_inference_peak")

  decoded = processor.batch_decode(output, skip_special_tokens=True)
  print(f"INFERENCE_OK elapsed_seconds={elapsed:.2f}")
  print(f"OUTPUT_TAIL={decoded[0][-300:]!r}")
  print(f"TORCH_VERSION={torch.__version__}")

  import transformers as _tf

  print(f"TRANSFORMERS_VERSION={_tf.__version__}")
  print(f"TORCH_CUDA_MAX_ALLOCATED_MIB={torch.cuda.max_memory_allocated() / (1024 * 1024):.0f}")
  PYEOF
  { python "$SPIKE"; echo "PROBE_EXIT=$?"; } 2>&1 | tee -a /tmp/spike_qwen_awq.log
  rm -f "$SPIKE"
  ```
  Expected: one of two outcomes, both are the spike's answer, not a bug to fix:
  - **Loads and infers**: the log contains an `INFERENCE_OK` line and `PROBE_EXIT=0`. →
    Branch A in Step 4.
  - **Fails**: an exception — most plausibly `ImportError` (autoawq's CUDA kernels rejecting
    the installed torch ABI), a `RuntimeError` from a kernel launch, or a CUDA OOM if the
    compositor's baseline usage plus the model does not fit in 8188 MiB — and `PROBE_EXIT`
    non-zero. → Branch B in Step 4. Do not debug the traceback; do not retry with different
    flags. The traceback itself is the evidence this spike exists to produce.

- [ ] **Step 4: Apply the decision rule to `ai-engine/pyproject.toml`**

  **Branch A — it loaded and inferred (`PROBE_EXIT=0`, `INFERENCE_OK` present).** Proceed
  exactly as specced: `autoawq` at Task 13. Pin the `gpu` extra to the exact versions that were
  just proven to work — a bare `>=` range is not what was tested, and `autoawq`'s own fragility
  against newer torch/transformers is precisely the risk this spike exists to retire, so leaving
  it floating would silently reopen it on the next `pip install`.

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI/ai-engine
  . .venv/bin/activate
  python3 - <<'PYEOF'
  import re
  from importlib.metadata import version
  from pathlib import Path

  path = Path("pyproject.toml")
  text = path.read_text()

  pins = {
      "torch": version("torch"),
      "transformers": version("transformers"),
      "autoawq": version("autoawq"),
  }

  for name, pin in pins.items():
      pattern = rf'"{name}>=[^"]+",'
      replacement = f'"{name}=={pin}",'
      assert re.search(pattern, text), f"{name} entry not found in gpu extra"
      text = re.sub(pattern, replacement, text, count=1)

  path.write_text(text)
  print(f"pinned exact versions that loaded and inferred successfully: {pins}")
  PYEOF
  ```
  This leaves `qwen-vl-utils>=0.0.8`, `ultralytics>=8.3`, `supervision>=0.24`, `accelerate>=1.0`
  as ranges (none of those are the fragile dependency) and leaves the mypy override list
  (`"autoawq.*"` already present) untouched.

  **Branch B — it failed.** Fall back to `transformers` + `bitsandbytes` 4-bit NF4, per spec
  §7: comparable VRAM (~3.5 GB), actively maintained, and `VisionLanguageModel` is unchanged, so
  this is an adapter-internal swap for Task 13, not a redesign.

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI/ai-engine
  . .venv/bin/activate
  python3 - <<'PYEOF'
  from pathlib import Path

  path = Path("pyproject.toml")
  text = path.read_text()

  old_gpu_block = '''gpu = [
      "torch>=2.5",
      "ultralytics>=8.3",
      "supervision>=0.24",
      "transformers>=4.49",
      "accelerate>=1.0",
      "autoawq>=0.2.6",
      "qwen-vl-utils>=0.0.8",
  ]'''

  new_gpu_block = '''gpu = [
      "torch>=2.5",
      "ultralytics>=8.3",
      "supervision>=0.24",
      "transformers>=4.49",
      "accelerate>=1.0",
      "bitsandbytes>=0.43",
      "qwen-vl-utils>=0.0.8",
  ]'''

  assert old_gpu_block in text, "gpu extra block not found verbatim — pyproject.toml has already changed"
  text = text.replace(old_gpu_block, new_gpu_block)

  old_overrides = '''module = [
      "ultralytics.*",
      "supervision.*",
      "av.*",
      "autoawq.*",
      "qwen_vl_utils.*",
      "minio.*",
      "jsonschema.*",
  ]'''

  new_overrides = '''module = [
      "ultralytics.*",
      "supervision.*",
      "av.*",
      "bitsandbytes.*",
      "qwen_vl_utils.*",
      "minio.*",
      "jsonschema.*",
  ]'''

  assert old_overrides in text, "mypy overrides block not found verbatim"
  text = text.replace(old_overrides, new_overrides)

  path.write_text(text)
  print("gpu extra switched to the bitsandbytes NF4 fallback")
  PYEOF
  ```

- [ ] **Step 5: Write the spike record and commit**

  The record is built entirely from `/tmp/spike_qwen_awq.log` — every number in it (the three
  `nvidia-smi` readings, the torch/transformers versions, the pass/fail outcome) is copied
  verbatim from what Steps 1–3 actually printed, not retyped by hand.

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI
  LOG=/tmp/spike_qwen_awq.log

  if grep -q INFERENCE_OK "$LOG"; then
    DECISION="autoawq. Task 13 loads Qwen/Qwen2.5-VL-3B-Instruct-AWQ via transformers + autoawq, as specced. pyproject.toml's gpu extra now pins the exact versions that loaded and inferred successfully."
  else
    DECISION="transformers + bitsandbytes 4-bit NF4 fallback. Task 13 loads the unquantised Qwen/Qwen2.5-VL-3B-Instruct checkpoint with BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=\"nf4\", bnb_4bit_compute_dtype=torch.float16). pyproject.toml's gpu extra now carries bitsandbytes instead of autoawq."
  fi

  {
    echo "# Phase 1B spike result — AutoAWQ load (Task 1)"
    echo
    echo "Ran: $(date -I)"
    echo "Machine: RTX 4060 Laptop, 8188 MiB, driver 580.173.02, compute capability 8.9,"
    echo "20 CPU cores, 15 GB RAM (~7 GB free)."
    echo
    echo "## Decision"
    echo
    echo "$DECISION"
    echo
    echo "## Raw probe log"
    echo
    echo '```'
    cat "$LOG"
    echo '```'
  } > docs/superpowers/plans/phase1b-spike-result.md

  git add docs/superpowers/plans/phase1b-spike-result.md ai-engine/pyproject.toml
  git commit -m "$(cat <<'EOF'
  chore(spike): decide the VLM loading path for Task 13

  Ran the autoawq load spike from spec §7 on the target 4060 Laptop (8188 MiB).
  The outcome and the measured VRAM are committed verbatim in
  docs/superpowers/plans/phase1b-spike-result.md; the gpu extra in
  pyproject.toml is pinned to match. Task 13 reads the decision from that
  file rather than re-deciding — the VisionLanguageModel port is identical
  either way, so nothing else in the plan depends on which branch was taken.
  EOF
  )"
  ```
  Expected: the commit succeeds and `docs/superpowers/plans/phase1b-spike-result.md` contains a
  non-empty `## Raw probe log` section with real `NVIDIA_SMI[...]` lines in it — a file with an
  empty log block means Step 3 was skipped or its output was not captured, and this task is not
  done.

---

### Task 2: Compose core services + mediamtx config

**Files:**
- Create: `deploy/compose/docker-compose.core.yml`
- Create: `deploy/compose/mediamtx.yml`
- Modify: `Makefile`
- Create: `ai-engine/tests/integration/__init__.py`
- Create: `ai-engine/tests/integration/test_compose_services.py`

**Interfaces:**
- Consumes: `sentinel_ai.config.get_settings()` — reads `rabbitmq_url` and `minio_endpoint`
  read-only, to keep the smoke test's assertions tied to the same defaults the compose file
  matches. No changes to `config.py` in this task.
- Produces: `make up` / `make down` / `make logs`, and the `avenue_01` mediamtx path, consumed
  manually by whoever runs Task 9's `MinioClipWriter` integration test or Task 14's end-to-end
  demo. Nothing under `sentinel_ai/` imports anything from this task.

Postgres and Redis have no settings in `config.py` — `sentinel_ai` does not talk to them in this
phase (spec's Phase 1C is the Go backend that owns them); they're scaffolded here only because
`docker-compose.core.yml` is the one core-infra file the whole system shares, per spec §5 of the
Phase 0+1 design (`repo/postgres/`, `cache/redis/`, `mq/rabbitmq/`). Their credentials below are
therefore local convention, not a contract anything currently reads.

- [ ] **Step 1: Write the failing integration smoke test**

  `@pytest.mark.integration` per the global constraints — this test opens real TCP sockets and
  must never run in the default CI command.

  ```python
  # ai-engine/tests/integration/__init__.py
  ```
  (empty file — marks the directory as a package, matching every other `tests/` subpackage.)

  ```python
  # ai-engine/tests/integration/test_compose_services.py
  """Smoke test for deploy/compose/docker-compose.core.yml.

  Excluded from CI: CI runs `-m "not gpu and not integration"` (see pyproject.toml's
  `[tool.pytest.ini_options]` markers and the Global Constraints). Run manually after
  `make up`:

      cd ai-engine && . .venv/bin/activate && python -m pytest -m integration \
          tests/integration/test_compose_services.py -v
  """

  from __future__ import annotations

  import socket
  from urllib.parse import urlparse

  import pytest

  from sentinel_ai.config import get_settings

  # Postgres and Redis have no sentinel_ai.config settings yet (spec: the Go backend owns
  # them from Phase 1C onward) — these mirror docker-compose.core.yml's port mappings
  # directly rather than a setting that does not exist.
  POSTGRES_HOST_PORT = ("localhost", 5432)
  REDIS_HOST_PORT = ("localhost", 6379)
  RABBITMQ_MANAGEMENT_HOST_PORT = ("localhost", 15672)
  MEDIAMTX_RTSP_HOST_PORT = ("localhost", 8554)

  TIMEOUT_SECONDS = 3.0


  def _assert_reachable(host: str, port: int) -> None:
      with socket.create_connection((host, port), timeout=TIMEOUT_SECONDS):
          pass


  @pytest.mark.integration
  def test_postgres_is_reachable() -> None:
      _assert_reachable(*POSTGRES_HOST_PORT)


  @pytest.mark.integration
  def test_redis_is_reachable() -> None:
      _assert_reachable(*REDIS_HOST_PORT)


  @pytest.mark.integration
  def test_rabbitmq_amqp_is_reachable() -> None:
      settings = get_settings()
      parsed = urlparse(settings.rabbitmq_url)
      assert parsed.hostname is not None, f"unparseable rabbitmq_url: {settings.rabbitmq_url!r}"
      _assert_reachable(parsed.hostname, parsed.port or 5672)


  @pytest.mark.integration
  def test_rabbitmq_management_ui_is_reachable() -> None:
      _assert_reachable(*RABBITMQ_MANAGEMENT_HOST_PORT)


  @pytest.mark.integration
  def test_minio_is_reachable() -> None:
      settings = get_settings()
      host, _, port = settings.minio_endpoint.partition(":")
      assert port, f"minio_endpoint has no port: {settings.minio_endpoint!r}"
      _assert_reachable(host, int(port))


  @pytest.mark.integration
  def test_mediamtx_rtsp_port_is_reachable() -> None:
      _assert_reachable(*MEDIAMTX_RTSP_HOST_PORT)
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -m integration tests/integration/test_compose_services.py -v`
  Expected: FAIL — every test raises `ConnectionRefusedError: [Errno 111] Connection refused`,
  because nothing is listening on any of these ports yet.

- [ ] **Step 3: Implement — the compose file, mediamtx config, and Makefile targets**

  ```yaml
  # deploy/compose/docker-compose.core.yml
  #
  # Core infra for SentinelAI's `core` profile (spec §5 of the Phase 0+1 design): the
  # datastores and the video-source normaliser. `sentinel_ai` only talks to rabbitmq and
  # minio directly (config.py); postgres and redis are scaffolded here for the Phase 1C Go
  # backend, sharing this one compose file rather than forking a second one.
  #
  # Image tags are pinned, not `latest`, so `make up` is reproducible; bump them with
  # `docker compose -f deploy/compose/docker-compose.core.yml pull` when a newer release is
  # wanted.

  services:
    postgres:
      image: postgres:16
      profiles: ["core"]
      restart: unless-stopped
      environment:
        POSTGRES_USER: sentinel
        POSTGRES_PASSWORD: sentinel
        POSTGRES_DB: sentinel
      ports:
        - "5432:5432"
      volumes:
        - postgres_data:/var/lib/postgresql/data
      healthcheck:
        test: ["CMD-SHELL", "pg_isready -U sentinel -d sentinel"]
        interval: 5s
        timeout: 5s
        retries: 10

    redis:
      image: redis:7
      profiles: ["core"]
      restart: unless-stopped
      ports:
        - "6379:6379"
      volumes:
        - redis_data:/data
      healthcheck:
        test: ["CMD", "redis-cli", "ping"]
        interval: 5s
        timeout: 5s
        retries: 10

    rabbitmq:
      image: rabbitmq:3.13-management
      profiles: ["core"]
      restart: unless-stopped
      environment:
        RABBITMQ_DEFAULT_USER: sentinel
        RABBITMQ_DEFAULT_PASS: sentinel
      ports:
        - "5672:5672"   # amqp — matches sentinel_ai.config.Settings.rabbitmq_url
        - "15672:15672" # management UI
      volumes:
        - rabbitmq_data:/var/lib/rabbitmq
      healthcheck:
        test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
        interval: 5s
        timeout: 5s
        retries: 10

    minio:
      image: minio/minio:RELEASE.2024-10-13T13-34-11Z
      profiles: ["core"]
      restart: unless-stopped
      command: server /data --console-address ":9001"
      environment:
        MINIO_ROOT_USER: sentinel
        MINIO_ROOT_PASSWORD: sentinel123
      ports:
        - "9000:9000"   # S3 API — matches sentinel_ai.config.Settings.minio_endpoint
        - "9001:9001"   # console
      volumes:
        - minio_data:/data
      healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
        interval: 5s
        timeout: 5s
        retries: 10

    minio-init:
      # One-shot bucket creation: sentinel_ai.config.Settings.minio_bucket ("sentinel-clips")
      # must exist before Task 9's MinioClipWriter uploads a clip. Runs once per `make up` and
      # exits 0 whether or not the bucket already existed (`mc mb --ignore-existing`).
      image: minio/mc:RELEASE.2024-10-08T09-37-26Z
      profiles: ["core"]
      depends_on:
        minio:
          condition: service_healthy
      entrypoint: >
        /bin/sh -c "
        mc alias set local http://minio:9000 sentinel sentinel123 &&
        mc mb --ignore-existing local/sentinel-clips &&
        exit 0
        "

    mediamtx:
      image: bluenviron/mediamtx:1.9.3
      profiles: ["core"]
      restart: unless-stopped
      ports:
        - "8554:8554" # rtsp
        - "1935:1935" # rtmp
        - "8888:8888" # hls
        - "8889:8889" # webrtc
      volumes:
        - ./mediamtx.yml:/mediamtx.yml:ro
      # No Docker healthcheck: the official image is built FROM scratch (no shell, no
      # curl/wget), so a CMD-SHELL or CMD probe cannot run inside the container. Reachability
      # is instead asserted from outside, over the RTSP port, by this task's own smoke test.

  volumes:
    postgres_data:
    redis_data:
    rabbitmq_data:
    minio_data:
  ```

  ```yaml
  # deploy/compose/mediamtx.yml
  #
  # avenue_01 is a `source: publisher` path (spec §5): mediamtx does not pull this stream on
  # its own — an external client pushes it in over RTSP. The exact ffmpeg command that loops
  # a file into this path is documented in this task's own report; every camera in this slice,
  # real or looped file, arrives at RtspSource (Task 14) through the same rtsp:// URL, which is
  # the entire point of normalising every source through mediamtx.

  paths:
    avenue_01:
      source: publisher
  ```

  Restore the `up`/`down`/`logs` Makefile targets. They were deliberately removed in
  `730c718` because this compose file did not exist yet; it now does. Replace the `.PHONY` line
  and the trailing comment block with:

  ```makefile
  .PHONY: help install install-runtime install-gpu lint format typecheck test test-gpu check up down logs
  ```

  and replace:

  ```makefile
  check: lint typecheck test

  # `up` / `down` / `logs` are deliberately absent: they referenced
  # deploy/compose/docker-compose.core.yml, which does not exist yet, so `make
  # help` advertised three targets that could only fail. The compose file lands in
  # Phase 1B; the targets come back with it.
  ```

  with:

  ```makefile
  check: lint typecheck test

  up:
  	docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d

  down:
  	docker compose -f deploy/compose/docker-compose.core.yml --profile core down

  logs:
  	docker compose -f deploy/compose/docker-compose.core.yml logs -f
  ```

- [ ] **Step 4: Verify**

  Run:
  ```bash
  cd /home/imran/Documents/github/SentinelAI
  make up
  cd ai-engine && . .venv/bin/activate
  python -m pytest -m integration tests/integration/test_compose_services.py -v
  ```
  Expected: PASS — all six reachability tests green, including `minio-init` having created
  `sentinel-clips` (`docker compose -f ../deploy/compose/docker-compose.core.yml logs minio-init`
  shows `Bucket created successfully` or `already own it`).

  Then confirm this test stays out of the default run and the rest of the suite is untouched:
  ```bash
  python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy
  ```
  Expected: all PASS — the integration test does not appear in this run's collected tests.

  Tear down:
  ```bash
  cd /home/imran/Documents/github/SentinelAI && make down
  ```

  **The ffmpeg command to loop a file into mediamtx as `avenue_01`** (exercised once Task 14's
  dataset script has fetched a sample; recorded here since it belongs with the compose/mediamtx
  config it targets, not with the dataset script itself):
  ```bash
  ffmpeg -re -stream_loop -1 -i /path/to/sample.mp4 -c copy -f rtsp rtsp://localhost:8554/avenue_01
  ```
  `-re` paces the push at the file's native frame rate (a real camera does not send faster than
  real time); `-stream_loop -1` loops forever; `-c copy` remuxes without re-encoding, so this one
  ffmpeg process costs negligible CPU even on a loop.

- [ ] **Step 5: Commit**

  ```bash
  git add deploy/compose/docker-compose.core.yml deploy/compose/mediamtx.yml Makefile \
      ai-engine/tests/integration/__init__.py ai-engine/tests/integration/test_compose_services.py
  git commit -m "$(cat <<'EOF'
  feat(infra): add core compose services and mediamtx, restore make up/down/logs

  Postgres 16, Redis 7, RabbitMQ 3.13 (management plugin), MinIO and mediamtx
  under the `core` profile, with healthchecks, named volumes, and a one-shot
  mc container that creates the sentinel-clips bucket. Ports and MinIO/RabbitMQ
  credentials match sentinel_ai.config.Settings' existing defaults. Restores
  the make targets 730c718 removed pending this file's existence, and adds a
  socket-level integration smoke test (excluded from CI) asserting every
  service is reachable.
  EOF
  )"
  ```

---

### Task 3: Port changes — `EncodedPacket`, `ClipHandle`, and the new settings

Implements S1–S4 verbatim. This is the last task ever allowed to touch `config.py`,
`ports/frame_source.py`, or `ports/clip_writer.py` for these names — every later task
consumes them as fixed. `tests/fakes/io.py` changes in the same commit, because `FakeSource`
and `FakeClipWriter` must satisfy the changed ports or mypy (which type-checks `tests/`, per
`ai-engine/pyproject.toml`'s `files = ["sentinel_ai", "tests"]`) fails the build.

**Files:**
- Modify: `ai-engine/sentinel_ai/config.py` (S1)
- Modify: `ai-engine/sentinel_ai/ports/frame_source.py` (S2, S3)
- Modify: `ai-engine/sentinel_ai/ports/clip_writer.py` (S4)
- Modify: `ai-engine/sentinel_ai/ports/detector.py` (docstring only — spec §4.3)
- Modify: `ai-engine/tests/fakes/io.py` (`FakeSource.packets()`, `FakeClipWriter`/
  `FakeClipHandle` rewritten against S4)
- Modify: `ai-engine/tests/ports/test_port_contracts.py`
- Modify: `ai-engine/tests/test_config.py`

**Interfaces:**
- Consumes: nothing from an earlier task — this is the first task to touch these files.
- Produces (verbatim from the skeleton): `EncodedPacket` (S2), `FrameSource.packets()` (S3),
  `ClipHandle` and `ClipWriter.open()` (S4), and the fourteen new `Settings` fields (S1). Task 4
  (`PreRollBuffer`, `FileSource`) and Task 6 (`CameraRunner`) consume `EncodedPacket` and
  `FrameSource.packets()` directly; Task 9 (`MinioClipWriter`) implements `ClipHandle` against
  this exact shape; Tasks 4–14 read the new `Settings` fields by name.

- [ ] **Step 1: Write the failing tests**

  Add to `ai-engine/tests/test_config.py` (full file — the new tests are appended after
  `test_clips_carry_three_seconds_of_pre_roll`, before `test_env_prefix_overrides_mode`):

  ```python
  from __future__ import annotations

  import pytest

  from sentinel_ai.config import Mode, Settings


  def defaults() -> Settings:
      """Settings with no `.env` applied.

      `Settings()` honours `env_file=".env"`, so a developer with an
      `ai-engine/.env` that sets SENTINEL_MODE would otherwise fail these tests.
      These assertions pin the *code* defaults, not the local environment.
      """
      # `_env_file` is a pydantic-settings runtime override; it is not part of the
      # model's generated __init__ signature, hence the ignore.
      return Settings(_env_file=None)  # type: ignore[call-arg]


  def test_development_is_the_default_mode() -> None:
      assert defaults().mode is Mode.DEVELOPMENT


  def test_default_vlm_is_the_3b_awq_build() -> None:
      assert defaults().vlm_model_id == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


  def test_vram_defaults_match_the_8gb_budget() -> None:
      settings = defaults()
      assert settings.vram_total_mib == 8192
      assert settings.vram_reserved_mib == 2048


  def test_the_vlm_unloads_after_ten_idle_minutes() -> None:
      assert defaults().vlm_idle_unload_seconds == 600.0


  def test_clips_carry_three_seconds_of_pre_roll() -> None:
      assert defaults().clip_preroll_seconds == 3.0


  def test_clips_carry_five_seconds_of_post_roll() -> None:
      assert defaults().clip_postroll_seconds == 5.0


  def test_detector_thresholds_match_the_yolo11s_defaults() -> None:
      settings = defaults()
      assert settings.detector_conf_threshold == 0.35
      assert settings.detector_iou_threshold == 0.45
      assert settings.detector_imgsz == 640


  def test_vlm_queue_defaults_bound_the_scheduler() -> None:
      settings = defaults()
      assert settings.vlm_queue_maxsize == 4
      assert settings.vlm_timeout_seconds == 30.0
      assert settings.vlm_max_new_tokens == 256
      assert settings.vlm_global_concurrency == 1
      assert settings.vlm_global_min_interval_seconds == 2.0


  def test_detect_every_n_frames_defaults_to_every_frame() -> None:
      assert defaults().detect_every_n_frames == 1


  def test_source_realtime_defaults_to_true() -> None:
      assert defaults().source_realtime is True


  def test_rtsp_reconnect_backoff_defaults() -> None:
      settings = defaults()
      assert settings.rtsp_reconnect_initial_seconds == 1.0
      assert settings.rtsp_reconnect_max_seconds == 30.0


  def test_clip_temp_dir_default() -> None:
      assert defaults().clip_temp_dir == "./var/clips"


  def test_env_prefix_overrides_mode(monkeypatch: pytest.MonkeyPatch) -> None:
      monkeypatch.setenv("SENTINEL_MODE", "production")
      assert Settings().mode is Mode.PRODUCTION
  ```

  Replace `ai-engine/tests/ports/test_port_contracts.py` in full:

  ```python
  from __future__ import annotations

  import inspect
  from uuid import uuid4

  import pytest

  from sentinel_ai.domain.entities import (
      BBox,
      Detection,
      EscalationReason,
      Event,
      SceneState,
      ThreatScore,
  )
  from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
  from sentinel_ai.ports.detector import ObjectDetector
  from sentinel_ai.ports.event_publisher import EventPublisher
  from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource
  from sentinel_ai.ports.model_runtime import LifecycleState, ModelRuntime
  from sentinel_ai.ports.tracker import Tracker
  from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
  from tests.fakes.io import FakeClipHandle, FakeClipWriter, FakePublisher, FakeSource
  from tests.fakes.models import FakeDetector, FakeTracker, FakeVisionLLM

  ALL_PORTS = [
      ModelRuntime,
      ObjectDetector,
      Tracker,
      VisionLanguageModel,
      FrameSource,
      EventPublisher,
      ClipWriter,
      ClipHandle,
  ]

  BOX = BBox(0.0, 0.0, 10.0, 10.0)


  def a_frame(frame_index: int = 0, timestamp: float = 0.0) -> FrameData:
      return FakeSource.make_frame("cam-1", frame_index, timestamp)


  def a_packet(
      frame_index: int = 0, timestamp: float = 0.0, is_keyframe: bool = True
  ) -> EncodedPacket:
      return EncodedPacket(
          camera_id="cam-1",
          data=f"packet-{frame_index}".encode(),
          pts=timestamp,
          is_keyframe=is_keyframe,
          codec="h264",
      )


  def a_scene() -> SceneState:
      return SceneState(
          camera_id="cam-1",
          frame_index=0,
          timestamp=0.0,
          detections=(),
          tracks=(),
          motion_energy=0.0,
          scene_signature=(1.0,),
      )


  def a_vision_request() -> VisionRequest:
      return VisionRequest(
          keyframe=a_frame(),
          scene=a_scene(),
          history=(),
          camera_label="Front Door",
          reason_detail="test",
      )


  @pytest.mark.parametrize("port", ALL_PORTS, ids=lambda p: p.__name__)
  def test_every_port_is_abstract_and_cannot_be_instantiated(port: type) -> None:
      assert inspect.isabstract(port), f"{port.__name__} has no abstract methods"
      with pytest.raises(TypeError):
          port()


  def test_model_runtime_exposes_the_seven_spec_section_10_methods() -> None:
      required = {
          "initialize",
          "health",
          "predict",
          "warmup",
          "shutdown",
          "version",
          "capabilities",
      }
      # Equality, not a subset: an eighth abstract method added to the §10
      # interface must be a deliberate, visible change to this test.
      assert required == set(ModelRuntime.__abstractmethods__)


  def test_lifecycle_states_cover_all_eight_from_spec_section_5() -> None:
      assert {s.value for s in LifecycleState} == {
          "loaded",
          "unloaded",
          "sleeping",
          "downloading",
          "updating",
          "offline",
          "healthy",
          "unhealthy",
      }


  @pytest.mark.parametrize(
      ("fake", "port"),
      [
          (FakeDetector, ObjectDetector),
          (FakeTracker, Tracker),
          (FakeVisionLLM, VisionLanguageModel),
          (FakeSource, FrameSource),
          (FakePublisher, EventPublisher),
          (FakeClipWriter, ClipWriter),
          (FakeClipHandle, ClipHandle),
      ],
      ids=lambda x: x.__name__,
  )
  def test_fakes_satisfy_their_ports(fake: type, port: type) -> None:
      assert issubclass(fake, port)


  class TestFakeDetector:
      async def test_it_replays_scripted_detections_in_order(self) -> None:
          first = (Detection("person", 0.9, BOX),)
          second = (Detection("car", 0.8, BBox(5.0, 5.0, 25.0, 25.0)),)
          detector = FakeDetector(script=[first, second])

          assert await detector.detect(a_frame()) == first
          assert await detector.detect(a_frame()) == second

      async def test_it_repeats_the_final_entry_once_the_script_runs_out(self) -> None:
          only = (Detection("person", 0.9, BOX),)
          detector = FakeDetector(script=[only])

          assert await detector.detect(a_frame()) == only
          assert await detector.detect(a_frame()) == only

      async def test_it_counts_calls_so_tests_can_assert_reuse(self) -> None:
          detector = FakeDetector(script=[()])
          await detector.detect(a_frame())
          await detector.detect(a_frame())
          assert detector.call_count == 2

      def test_an_empty_script_is_rejected(self) -> None:
          with pytest.raises(ValueError, match="at least one entry"):
              FakeDetector(script=[])


  class TestFakeVisionLLM:
      async def test_it_records_every_request_for_assertion(self) -> None:
          vlm = FakeVisionLLM(
              response=SceneDescription(
                  description="Two people talking near the entrance.",
                  threat_value=0.1,
                  suggested_action="No action required.",
              )
          )
          request = a_vision_request()
          result = await vlm.describe(request)

          assert result.description == "Two people talking near the entrance."
          assert vlm.requests == [request]
          assert vlm.call_count == 1

      async def test_it_can_be_configured_to_raise_for_failure_path_tests(self) -> None:
          """Needed for spec §9: a VLM timeout must still produce an event."""
          vlm = FakeVisionLLM(error=TimeoutError("vlm timed out"))
          with pytest.raises(TimeoutError):
              await vlm.describe(a_vision_request())
          assert vlm.call_count == 1, "the request is recorded even when it fails"


  class TestFakeTracker:
      def test_it_assigns_stable_ids_and_ages_tracks(self) -> None:
          tracker = FakeTracker()
          detection = Detection("person", 0.9, BOX)

          first = tracker.update((detection,), timestamp=0.0)
          second = tracker.update((detection,), timestamp=0.1)

          assert first[0].track_id == second[0].track_id
          assert second[0].age_frames == first[0].age_frames + 1

      def test_it_computes_speed_from_centroid_displacement(self) -> None:
          tracker = FakeTracker()
          tracker.update((Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),), timestamp=0.0)
          moved = tracker.update(
              (Detection("person", 0.9, BBox(20.0, 0.0, 30.0, 10.0)),), timestamp=1.0
          )
          assert moved[0].speed_px_s == pytest.approx(20.0)

      def test_a_distant_detection_starts_a_new_track(self) -> None:
          tracker = FakeTracker(match_radius_px=10.0)
          first = tracker.update((Detection("person", 0.9, BOX),), timestamp=0.0)
          far = tracker.update(
              (Detection("person", 0.9, BBox(500.0, 500.0, 510.0, 510.0)),), timestamp=0.1
          )
          assert far[0].track_id != first[0].track_id

      def test_reset_clears_all_state(self) -> None:
          """Re-sight the object after the reset — an empty update proves nothing.

          `tracker.update((), ...) == ()` holds for any implementation, including
          one whose `reset` does nothing at all, because an empty detection tuple
          always yields an empty result.
          """
          tracker = FakeTracker(match_radius_px=10.0)
          near = Detection("person", 0.9, BOX)
          far = Detection("person", 0.9, BBox(500.0, 500.0, 510.0, 510.0))

          tracker.update((near,), timestamp=0.0)
          before = tracker.update((near, far), timestamp=0.1)
          far_id = before[1].track_id
          assert far_id != 1, "the id counter has advanced past its starting value"

          tracker.reset()
          after = tracker.update((far,), timestamp=1.0)

          assert after[0].track_id != far_id, "the association must be forgotten"
          assert after[0].track_id == 1, "the id counter must be rewound"
          assert after[0].age_frames == 1, "a re-sighted object is new, not aged"


  class TestFakeSource:
      async def test_it_yields_every_frame_in_order(self) -> None:
          source = FakeSource.constant("cam-1", count=3, fps=10.0)
          indices = [frame.frame_index async for frame in source]
          assert indices == [0, 1, 2]

      async def test_constant_spaces_timestamps_by_the_frame_interval(self) -> None:
          source = FakeSource.constant("cam-1", count=3, fps=10.0)
          stamps = [frame.timestamp async for frame in source]
          assert stamps == pytest.approx([0.0, 0.1, 0.2])

      async def test_close_is_recorded(self) -> None:
          source = FakeSource.constant("cam-1", count=1)
          await source.close()
          assert source.closed is True

      async def test_it_yields_a_synthetic_packet_per_frame_when_none_are_given(self) -> None:
          source = FakeSource.constant("cam-1", count=3, fps=10.0)
          packets = [packet async for packet in source.packets()]
          assert [p.pts for p in packets] == pytest.approx([0.0, 0.1, 0.2])
          assert all(p.camera_id == "cam-1" for p in packets)
          assert all(p.is_keyframe for p in packets), "every synthetic packet is a keyframe"

      async def test_it_yields_explicit_packets_when_given(self) -> None:
          frames = [FakeSource.make_frame("cam-1", i, i / 10.0) for i in range(2)]
          packets = [a_packet(0, 0.0, is_keyframe=True), a_packet(1, 0.1, is_keyframe=False)]
          source = FakeSource(frames, packets=packets)

          collected = [packet async for packet in source.packets()]

          assert collected == packets


  class TestFakePublisher:
      async def test_it_collects_published_events(self) -> None:
          publisher = FakePublisher()
          event = Event(
              event_id=uuid4(),
              camera_id="cam-1",
              occurred_at=1.0,
              reason=EscalationReason.SPEED_ANOMALY,
              threat=ThreatScore.from_value(0.7),
              description="A person is running.",
              suggested_action="Review the clip.",
          )
          await publisher.publish(event)
          assert publisher.events == [event]

      async def test_it_can_be_configured_to_fail(self) -> None:
          publisher = FakePublisher(error=ConnectionError("broker down"))
          with pytest.raises(ConnectionError):
              await publisher.publish(
                  Event(
                      event_id=uuid4(),
                      camera_id="cam-1",
                      occurred_at=1.0,
                      reason=EscalationReason.PERIODIC_SUMMARY,
                      threat=ThreatScore.from_value(0.1),
                      description="",
                      suggested_action="",
                  )
              )
          assert publisher.events == []


  class TestFakeClipWriter:
      async def test_open_records_the_call_and_returns_a_handle(self) -> None:
          writer = FakeClipWriter()
          event_id = uuid4()

          handle = await writer.open("cam-1", event_id, fps=30.0)

          assert writer.opened == [("cam-1", event_id, 30.0)]
          assert isinstance(handle, FakeClipHandle)

      async def test_append_records_every_packet_in_order(self) -> None:
          writer = FakeClipWriter()
          handle = await writer.open("cam-1", uuid4(), fps=30.0)
          packets = [a_packet(i, i / 30.0) for i in range(3)]

          for packet in packets:
              await handle.append(packet)

          assert handle.packets == packets

      async def test_finish_returns_a_deterministic_uri(self) -> None:
          writer = FakeClipWriter()
          event_id = uuid4()
          handle = await writer.open("cam-1", event_id, fps=30.0)

          uri = await handle.finish()

          assert uri == f"s3://sentinel-clips/cam-1/{event_id}.mp4"
          assert handle.finished is True

      async def test_abort_cannot_raise_even_after_finish(self) -> None:
          writer = FakeClipWriter()
          handle = await writer.open("cam-1", uuid4(), fps=30.0)
          await handle.finish()

          await handle.abort()  # must not raise

          assert handle.aborted is True

      async def test_each_open_call_returns_an_independent_handle(self) -> None:
          writer = FakeClipWriter()
          first = await writer.open("cam-1", uuid4(), fps=30.0)
          second = await writer.open("cam-1", uuid4(), fps=30.0)

          await first.append(a_packet(0, 0.0))

          assert first.packets != second.packets
          assert len(writer.handles) == 2
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/test_config.py tests/ports/test_port_contracts.py -v`
  Expected: FAIL —
  `tests/ports/test_port_contracts.py` errors at collection with
  `ImportError: cannot import name 'EncodedPacket' from 'sentinel_ai.ports.frame_source'`
  (and, once that import is fixed in isolation, `ImportError: cannot import name 'ClipHandle'
  from 'sentinel_ai.ports.clip_writer'`, then `ImportError: cannot import name 'FakeClipHandle'
  from 'tests.fakes.io'`); `tests/test_config.py::test_clips_carry_five_seconds_of_post_roll`
  fails with `AttributeError: 'Settings' object has no attribute 'clip_postroll_seconds'`.

- [ ] **Step 3: Implement**

  ```python
  # ai-engine/sentinel_ai/config.py
  """Runtime configuration. This is the ONLY place the Development/Production seam is chosen."""

  from __future__ import annotations

  from enum import StrEnum
  from functools import lru_cache

  from pydantic import Field
  from pydantic_settings import BaseSettings, SettingsConfigDict


  class Mode(StrEnum):
      """Spec §3.1/§3.2. Development binds in-process transport; Production binds gRPC."""

      DEVELOPMENT = "development"
      PRODUCTION = "production"


  class Settings(BaseSettings):
      model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

      mode: Mode = Mode.DEVELOPMENT

      vram_total_mib: int = Field(default=8192, gt=0)
      vram_reserved_mib: int = Field(default=2048, ge=0)

      detector_model_id: str = "yolo11s.pt"
      vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
      vlm_idle_unload_seconds: float = Field(default=600.0, gt=0)

      decode_hwaccel: str | None = None

      rabbitmq_url: str = "amqp://sentinel:sentinel@localhost:5672/"
      rabbitmq_exchange: str = "sentinel.events"
      event_spool_dir: str = "./var/spool/events"

      minio_endpoint: str = "localhost:9000"
      minio_access_key: str = "sentinel"
      minio_secret_key: str = "sentinel123"
      minio_bucket: str = "sentinel-clips"
      minio_secure: bool = False

      clip_preroll_seconds: float = Field(default=3.0, ge=0)
      clip_postroll_seconds: float = Field(default=5.0, gt=0)

      detector_conf_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
      detector_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
      detector_imgsz: int = Field(default=640, gt=0)

      vlm_queue_maxsize: int = Field(default=4, ge=1)
      vlm_timeout_seconds: float = Field(default=30.0, gt=0)
      vlm_max_new_tokens: int = Field(default=256, gt=0)
      vlm_global_concurrency: int = Field(default=1, ge=1)
      vlm_global_min_interval_seconds: float = Field(default=2.0, ge=0)

      detect_every_n_frames: int = Field(default=1, ge=1)
      source_realtime: bool = True
      rtsp_reconnect_initial_seconds: float = Field(default=1.0, gt=0)
      rtsp_reconnect_max_seconds: float = Field(default=30.0, gt=0)

      clip_temp_dir: str = "./var/clips"


  @lru_cache(maxsize=1)
  def get_settings() -> Settings:
      return Settings()
  ```

  ```python
  # ai-engine/sentinel_ai/ports/frame_source.py
  """Video input port. Every source (RTSP, file, USB) reduces to this."""

  from __future__ import annotations

  from abc import ABC, abstractmethod
  from collections.abc import AsyncIterator
  from dataclasses import dataclass


  @dataclass(frozen=True, slots=True)
  class FrameData:
      """A decoded frame.

      `pixels` is typed `object` deliberately: it carries a numpy array at
      runtime, but typing it as such would drag numpy into the ports layer and
      the architecture fitness test would (correctly) fail the build.
      """

      camera_id: str
      frame_index: int
      timestamp: float
      width: int
      height: int
      pixels: object


  @dataclass(frozen=True, slots=True)
  class EncodedPacket:
      """A demuxed, still-encoded packet. `data` is bytes so ports stay codec-agnostic.

      `pts` is float seconds on the same monotonic timeline as `FrameData.timestamp` — the
      pre-roll buffer and the clip writer line packets and frames up on it.
      """

      camera_id: str
      data: bytes
      pts: float
      is_keyframe: bool
      codec: str
      """Container short codec name, lowercase — "h264", "hevc".

      A plain `str`, deliberately: the clip writer needs to know what it is remuxing, but a
      library enum here would drag a codec dependency into the pure ports layer and the
      architecture fitness test would reject it. Without this field the clip writer has to
      *assume* H.264, and a non-H.264 source produces an empty clip instead of an error.
      """


  class FrameSource(ABC):
      @abstractmethod
      def __aiter__(self) -> AsyncIterator[FrameData]: ...

      @abstractmethod
      def packets(self) -> AsyncIterator[EncodedPacket]:
          """The same demux pass as `__aiter__`, fanned out as still-encoded packets.

          Feeds the pre-roll buffer and the clip writer (spec §4.2). Every demuxed packet is
          offered here even though only the sampled ones are decoded into frames.
          """

      @abstractmethod
      async def close(self) -> None: ...
  ```

  ```python
  # ai-engine/sentinel_ai/ports/clip_writer.py
  """Evidence clip port.

  Clips must begin 3 s before the anomaly (spec §23, §26), which is why the decode stage keeps
  a pre-roll ring buffer rather than starting to record on escalation.

  `open` returns a handle so a clip is written as packets arrive instead of assembled in memory
  first: a 3 s pre-roll plus 5 s post-roll at 1080p30 is ~180 raw frames, roughly 1.1 GB of
  host RAM per concurrent clip on a machine with ~7 GB free (Phase 1B spec §4.1). Streaming
  encoded packets through open/append/finish keeps peak memory at the pre-roll ring plus one
  packet.
  """

  from __future__ import annotations

  from abc import ABC, abstractmethod
  from uuid import UUID

  from sentinel_ai.ports.frame_source import EncodedPacket


  class ClipHandle(ABC):
      @abstractmethod
      async def append(self, packet: EncodedPacket) -> None: ...

      @abstractmethod
      async def finish(self) -> str:
          """Finalise the clip and return its URI."""

      @abstractmethod
      async def abort(self) -> None:
          """Discard a partial clip. MUST NOT raise.

          A pipeline shutdown mid-clip must not turn cleanup into a second failure.
          """


  class ClipWriter(ABC):
      @abstractmethod
      async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle: ...
  ```

  ```python
  # ai-engine/sentinel_ai/ports/detector.py
  """Object detection port (pipeline stage 2)."""

  from __future__ import annotations

  from abc import ABC, abstractmethod

  from sentinel_ai.domain.entities import Detection
  from sentinel_ai.ports.frame_source import FrameData


  class ObjectDetector(ABC):
      @abstractmethod
      async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
          """Detections for a single frame.

          Results are reused by tracking, the gate, the VLM, and reporting — detection is
          never re-run per downstream consumer (spec §7).

          An implementation's `ModelRuntime.capabilities().batch_max` may advertise a batch
          size greater than 1, but this method stays per-frame in Phase 1B: the slice is
          single-camera, so every batch would have size 1 and a batch method would be
          untested, unused code. `batch_max` is advertised for Phase 2 multi-camera use only
          (spec §4.3); no Phase 1B adapter honours it.
          """
  ```

  ```python
  # ai-engine/tests/fakes/io.py
  """I/O fakes: frame sources, publishers, clip writers."""

  from __future__ import annotations

  from collections.abc import AsyncIterator, Sequence
  from uuid import UUID

  from sentinel_ai.domain.entities import Event
  from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
  from sentinel_ai.ports.event_publisher import EventPublisher
  from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource


  class FakeSource(FrameSource):
      def __init__(
          self,
          frames: Sequence[FrameData],
          packets: Sequence[EncodedPacket] | None = None,
      ) -> None:
          self._frames = list(frames)
          self._packets = (
              list(packets)
              if packets is not None
              else [self._synthetic_packet(frame) for frame in self._frames]
          )
          self.closed = False

      @staticmethod
      def make_frame(
          camera_id: str, frame_index: int, timestamp: float, value: int = 0
      ) -> FrameData:
          return FrameData(
              camera_id=camera_id,
              frame_index=frame_index,
              timestamp=timestamp,
              width=4,
              height=4,
              pixels=[[value] * 4 for _ in range(4)],
          )

      @staticmethod
      def _synthetic_packet(frame: FrameData) -> EncodedPacket:
          """A deterministic stand-in for an encoded packet, paired to `frame` by camera id
          and pts. Always a keyframe: this fake exercises the clip *plumbing* (open/append/
          finish wiring through the runner and scheduler), not GOP-quantised flush semantics
          — that is Task 4's job, against real PyAV output.
          """
          return EncodedPacket(
              camera_id=frame.camera_id,
              data=f"packet-{frame.frame_index}".encode(),
              pts=frame.timestamp,
              is_keyframe=True,
              codec="h264",
          )

      @classmethod
      def constant(
          cls, camera_id: str, count: int, fps: float = 10.0, value: int = 0
      ) -> FakeSource:
          return cls(
              [cls.make_frame(camera_id, index, index / fps, value) for index in range(count)]
          )

      def __aiter__(self) -> AsyncIterator[FrameData]:
          """Sync, matching the port and the async-iterator protocol.

          `async for` calls `__aiter__()` without awaiting it, so the method must return the
          iterator directly. Writing it as `async def` happened to work only because an
          `async def` containing `yield` is an async *generator* function — remove the yield
          and it breaks. The port's shape is the correct one, so the fake follows it.
          """

          async def frames() -> AsyncIterator[FrameData]:
              for frame in self._frames:
                  yield frame

          return frames()

      def packets(self) -> AsyncIterator[EncodedPacket]:
          """Sync for the same reason `__aiter__` is — see above."""

          async def stream() -> AsyncIterator[EncodedPacket]:
              for packet in self._packets:
                  yield packet

          return stream()

      async def close(self) -> None:
          self.closed = True


  class FakePublisher(EventPublisher):
      def __init__(self, error: Exception | None = None) -> None:
          self.events: list[Event] = []
          self.closed = False
          self._error = error

      async def publish(self, event: Event) -> None:
          if self._error is not None:
              raise self._error
          self.events.append(event)

      async def close(self) -> None:
          self.closed = True


  class FakeClipHandle(ClipHandle):
      """Records every appended packet; `finish` and `abort` never raise."""

      def __init__(self, camera_id: str, event_id: UUID) -> None:
          self.camera_id = camera_id
          self.event_id = event_id
          self.packets: list[EncodedPacket] = []
          self.finished = False
          self.aborted = False

      async def append(self, packet: EncodedPacket) -> None:
          self.packets.append(packet)

      async def finish(self) -> str:
          self.finished = True
          return f"s3://sentinel-clips/{self.camera_id}/{self.event_id}.mp4"

      async def abort(self) -> None:
          self.aborted = True


  class FakeClipWriter(ClipWriter):
      def __init__(self) -> None:
          self.opened: list[tuple[str, UUID, float]] = []
          self.handles: list[FakeClipHandle] = []

      async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle:
          self.opened.append((camera_id, event_id, fps))
          handle = FakeClipHandle(camera_id, event_id)
          self.handles.append(handle)
          return handle
  ```

- [ ] **Step 4: Verify**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
  Expected: all PASS, ruff clean, mypy `Success: no issues found in 41 source files` (39 from
  the Phase 1A baseline plus this task's two new test files' worth of exercised modules — the
  exact count is whatever mypy reports; the requirement is zero errors, not a specific number).

- [ ] **Step 5: Commit**

  ```bash
  git add ai-engine/sentinel_ai/config.py ai-engine/sentinel_ai/ports/frame_source.py \
      ai-engine/sentinel_ai/ports/clip_writer.py ai-engine/sentinel_ai/ports/detector.py \
      ai-engine/tests/fakes/io.py ai-engine/tests/ports/test_port_contracts.py \
      ai-engine/tests/test_config.py
  git commit -m "$(cat <<'EOF'
  feat(ports): add EncodedPacket, ClipHandle, and the Phase 1B settings

  ClipWriter.write(frames) becomes open/append/finish over EncodedPacket
  (spec §4.1): a 3s pre-roll plus 5s post-roll at 1080p30 is ~1.1GB of raw
  frames per concurrent clip against ~7GB free RAM, so the clip is now
  streamed as still-encoded packets and remuxed, never assembled in memory.
  FrameSource gains packets() so one demux pass feeds both inference and the
  clip path (spec §4.2). detector.py gets a docstring-only note that
  Capabilities.batch_max is Phase 2 only (spec §4.3) — detect() stays
  per-frame.

  FakeSource and FakeClipWriter are rewritten in the same commit against the
  new shapes, since mypy type-checks tests/ and a stale fake would break the
  build before any real adapter exists.
  EOF
  )"
  ```

---

### Task 4: `PreRollBuffer` + `FileSource` + the synthetic clip fixture

Three deliverables in one task because none is independently useful and two of them share a
test fixture: `PreRollBuffer` is exercised with hand-built packets (fast, exact control over
pts/keyframe placement) and separately proven against real PyAV demuxing of the committed
clip; `FileSource` can only be tested against a real container.

**Files:**
- Create: `ai-engine/tests/assets/synthetic_clip.mp4`
- Create: `ai-engine/sentinel_ai/adapters/sources/__init__.py`
- Create: `ai-engine/sentinel_ai/adapters/sources/preroll.py`
- Create: `ai-engine/sentinel_ai/adapters/sources/file.py`
- Test: `ai-engine/tests/adapters/test_preroll.py`
- Test: `ai-engine/tests/adapters/test_file_source.py`

**Interfaces:**
- Consumes: `EncodedPacket`, `FrameData`, `FrameSource` (`sentinel_ai/ports/frame_source.py`,
  Task 3). Task 3 lands `EncodedPacket` with a fifth field, `codec: str`, beyond skeleton S2's
  four — see the `` at the end of this document. This task's code
  constructs `EncodedPacket` with all five fields, matching what Task 3 actually produces.
- Produces (verbatim, S5): `PreRollBuffer.__init__(preroll_seconds)`, `.append(packet)`,
  `.flush() -> tuple[EncodedPacket, ...]`, `.clear()`, `.span_seconds`. Also produces
  `FileSource(path: str, camera_id: str, realtime: bool)` implementing `FrameSource`. Task 6
  (`CameraRunner`) constructs both; Task 9 (`MinioClipWriter`) consumes the packets
  `PreRollBuffer.flush()` returns; Task 14 (`RtspSource`) reuses `FileSource`'s fan-out pattern.

- [ ] **Step 1: Write the failing tests for `PreRollBuffer`**

  ```python
  # ai-engine/tests/adapters/test_preroll.py
  from __future__ import annotations

  import pytest

  from sentinel_ai.adapters.sources.preroll import PreRollBuffer
  from sentinel_ai.ports.frame_source import EncodedPacket


  def packet(pts: float, is_keyframe: bool) -> EncodedPacket:
      return EncodedPacket(
          camera_id="cam-1", data=b"x", pts=pts, is_keyframe=is_keyframe, codec="h264"
      )


  class TestPreRollBuffer:
      def test_rejects_a_non_positive_horizon(self) -> None:
          with pytest.raises(ValueError, match="preroll_seconds"):
              PreRollBuffer(preroll_seconds=0.0)

      def test_returns_empty_before_any_keyframe_has_been_seen(self) -> None:
          buffer = PreRollBuffer(preroll_seconds=1.0)
          buffer.append(packet(0.0, is_keyframe=False))
          buffer.append(packet(0.1, is_keyframe=False))
          assert buffer.flush() == ()

      def test_flush_starts_from_a_keyframe(self) -> None:
          buffer = PreRollBuffer(preroll_seconds=0.15)
          buffer.append(packet(0.0, is_keyframe=True))
          buffer.append(packet(0.1, is_keyframe=False))
          buffer.append(packet(0.2, is_keyframe=False))
          flushed = buffer.flush()
          assert flushed[0].is_keyframe
          assert [p.pts for p in flushed] == [0.0, 0.1, 0.2]

      def test_flush_advances_to_the_newest_keyframe_at_or_before_the_horizon(self) -> None:
          """Two keyframes exist; only the closer one anchors a minimal clip."""
          buffer = PreRollBuffer(preroll_seconds=0.05)
          buffer.append(packet(0.0, is_keyframe=True))
          buffer.append(packet(0.1, is_keyframe=True))
          buffer.append(packet(0.2, is_keyframe=False))
          flushed = buffer.flush()
          # horizon = 0.2 - 0.05 = 0.15; the keyframe at 0.1 qualifies, so the
          # one at 0.0 no longer anchors anything once a closer one exists.
          assert [p.pts for p in flushed] == [0.1, 0.2]

      def test_eviction_is_bounded_and_keeps_the_keyframe_anchor(self) -> None:
          """A working buffer stays small; a no-op implementation keeps growing."""
          buffer = PreRollBuffer(preroll_seconds=0.25)
          # A 1s GOP (10 packets @ 0.1s apart) fed for 5s -- far more than the horizon.
          for i in range(50):
              buffer.append(packet(round(i * 0.1, 4), is_keyframe=(i % 10 == 0)))
          flushed = buffer.flush()
          assert flushed[0].is_keyframe
          assert buffer.span_seconds < 1.5, "retained span must stay bounded, not grow forever"
          assert flushed[0].pts == pytest.approx(4.0)
          assert flushed[-1].pts == pytest.approx(4.9)

      def test_append_never_evicts_a_keyframe_later_packets_still_depend_on(self) -> None:
          buffer = PreRollBuffer(preroll_seconds=10.0)  # horizon far in the past
          for i in range(20):
              buffer.append(packet(round(i * 0.1, 4), is_keyframe=(i % 10 == 0)))
          # Horizon never reached: everything since the first keyframe is kept.
          assert buffer.flush()[0].pts == 0.0
          assert len(buffer.flush()) == 20

      def test_clear_genuinely_empties_the_buffer(self) -> None:
          buffer = PreRollBuffer(preroll_seconds=0.15)
          buffer.append(packet(0.0, is_keyframe=True))
          buffer.append(packet(0.1, is_keyframe=False))
          buffer.clear()
          assert buffer.flush() == ()
          assert buffer.span_seconds == 0.0

      def test_span_seconds_is_zero_when_empty(self) -> None:
          assert PreRollBuffer(preroll_seconds=1.0).span_seconds == 0.0

      def test_span_seconds_reports_the_retained_pts_range(self) -> None:
          buffer = PreRollBuffer(preroll_seconds=10.0)
          buffer.append(packet(0.0, is_keyframe=True))
          buffer.append(packet(0.3, is_keyframe=False))
          assert buffer.span_seconds == pytest.approx(0.3)
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/test_preroll.py -v`
  Expected: FAIL — errors at collection with
  `ModuleNotFoundError: No module named 'sentinel_ai.adapters.sources'` (the subpackage does
  not exist yet).

- [ ] **Step 3: Implement `PreRollBuffer`**

  ```python
  # ai-engine/sentinel_ai/adapters/sources/__init__.py
  """Video sources and the shared pre-roll ring buffer."""
  ```

  ```python
  # ai-engine/sentinel_ai/adapters/sources/preroll.py
  """Keyframe-aligned encoded packet ring (spec §5.1).

  Holds `EncodedPacket`s for `preroll_seconds` of the live stream so an escalation's
  evidence clip can start *before* the trigger. Packets, not decoded frames: a few
  seconds of H.264 costs ~2 MB where the equivalent raw RGB frames cost ~340 MB.
  """

  from __future__ import annotations

  from collections import deque

  from sentinel_ai.ports.frame_source import EncodedPacket


  class PreRollBuffer:
      """A clip cannot start mid-GOP (spec §5.1): `flush()` always begins at a
      keyframe, which quantises the actual pre-roll up to the GOP boundary --
      more context than requested, never less.
      """

      def __init__(self, preroll_seconds: float) -> None:
          if preroll_seconds <= 0:
              raise ValueError(f"preroll_seconds must be positive, got {preroll_seconds}")
          self._preroll_seconds = preroll_seconds
          self._packets: deque[EncodedPacket] = deque()

      def _anchor_index(self) -> int | None:
          """Index of the keyframe `flush()` would start from, or None.

          The target is the *newest* keyframe at or before the horizon -- the
          smallest clip that still satisfies the pre-roll target. Before the
          buffer has accumulated `preroll_seconds` of history, no keyframe is
          old enough yet; in that ramp-up window the earliest keyframe held is
          the best available anchor. `None` only when no keyframe has been
          buffered at all, per `flush()`'s contract.
          """
          if not self._packets:
              return None
          horizon = self._packets[-1].pts - self._preroll_seconds
          first_keyframe: int | None = None
          anchor: int | None = None
          for index, candidate in enumerate(self._packets):
              if not candidate.is_keyframe:
                  continue
              if first_keyframe is None:
                  first_keyframe = index
              if candidate.pts <= horizon:
                  anchor = index
          return anchor if anchor is not None else first_keyframe

      def append(self, packet: EncodedPacket) -> None:
          """Add a packet and evict everything older than the keyframe-aligned horizon."""
          self._packets.append(packet)
          anchor = self._anchor_index()
          if anchor is None:
              return
          # Everything before `anchor` is provably unreachable from any future
          # flush(): once a newer keyframe already satisfies the horizon, an
          # older one can never become the flush point again (pts only increases).
          for _ in range(anchor):
              self._packets.popleft()

      def flush(self) -> tuple[EncodedPacket, ...]:
          """Return buffered packets from the oldest keyframe at or before the horizon.

          Returns () when no keyframe has been seen yet.
          """
          anchor = self._anchor_index()
          if anchor is None:
              return ()
          return tuple(self._packets)[anchor:]

      def clear(self) -> None:
          self._packets.clear()

      @property
      def span_seconds(self) -> float:
          """pts span currently held; 0.0 when empty."""
          if not self._packets:
              return 0.0
          return self._packets[-1].pts - self._packets[0].pts
  ```

- [ ] **Step 4: Verify `PreRollBuffer` in isolation**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/test_preroll.py -v`
  Expected: 9 PASS.

- [ ] **Step 5: Generate and verify the synthetic clip fixture**

  A committed, deterministic H.264 MP4 the buffer and source tests demux for real — not under
  `datasets/` (gitignored; download-only), but under `ai-engine/tests/assets/`, which is
  ordinary tracked source. `-g 10 -keyint_min 10 -sc_threshold 0` pins the GOP at exactly 10
  frames regardless of scene content, so keyframe positions are deterministic; `-profile:v
  baseline` forbids B-frames, so decode order equals display order and `packet.pts` is
  monotonic — both are load-bearing for the keyframe-index assertions above and in Step 6.

  Run:
  ```bash
  mkdir -p ai-engine/tests/assets
  ffmpeg -y -f lavfi -i "testsrc2=size=320x240:rate=10:duration=5" \
    -c:v libx264 -pix_fmt yuv420p -profile:v baseline \
    -g 10 -keyint_min 10 -sc_threshold 0 \
    -movflags +faststart \
    ai-engine/tests/assets/synthetic_clip.mp4
  ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate,duration \
    -of default=noprint_wrappers=1 ai-engine/tests/assets/synthetic_clip.mp4
  ffprobe -v error -select_streams v:0 -show_entries frame=key_frame -of csv \
    ai-engine/tests/assets/synthetic_clip.mp4 | grep -c ',1'
  ```
  Expected: `nb_frames=50`, `r_frame_rate=10/1`, `duration=5.000000`; the keyframe count is
  `5` (at frame indices 0, 10, 20, 30, 40 — verified against this exact ffmpeg invocation).
  File size is 133,160 bytes (~130 KB).

- [ ] **Step 6: Write the failing tests for `FileSource`**

  ```python
  # ai-engine/tests/adapters/test_file_source.py
  from __future__ import annotations

  import asyncio
  import time
  from pathlib import Path

  import numpy as np
  import pytest

  from sentinel_ai.adapters.sources.file import FileSource
  from sentinel_ai.adapters.sources.preroll import PreRollBuffer
  from sentinel_ai.ports.frame_source import EncodedPacket

  ASSET = str(Path(__file__).resolve().parents[1] / "assets" / "synthetic_clip.mp4")


  async def _drain(source: FileSource) -> None:
      async for _ in source:
          pass


  async def _collect_packets(source: FileSource) -> list[EncodedPacket]:
      return [p async for p in source.packets()]


  async def test_decodes_every_frame_in_order() -> None:
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)
      frames = [f async for f in source]
      assert [f.frame_index for f in frames] == list(range(50))
      assert frames[0].width == 320
      assert frames[0].height == 240
      assert isinstance(frames[0].pixels, np.ndarray)
      await source.close()


  async def test_timestamps_start_at_zero_and_are_spaced_by_the_frame_interval() -> None:
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)
      stamps = [f.timestamp async for f in source]
      assert stamps[0] == 0.0
      assert stamps == pytest.approx([i * 0.1 for i in range(50)], abs=1e-6)
      await source.close()


  async def test_packets_emit_every_packet_with_correct_keyframes() -> None:
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)
      # packets() must be drained alongside __aiter__: frame delivery holds real
      # backpressure, so a frame consumer that never runs at all would eventually
      # stall the shared pump. Every real caller (CameraRunner, Task 6) drains
      # both from the start, so this test does too.
      drain_frames = asyncio.ensure_future(_drain(source))
      packets = await asyncio.wait_for(_collect_packets(source), timeout=5.0)
      await drain_frames
      assert len(packets) == 50
      keyframe_indices = [i for i, p in enumerate(packets) if p.is_keyframe]
      assert keyframe_indices == [0, 10, 20, 30, 40]
      assert packets[0].pts == 0.0
      assert packets[-1].pts == pytest.approx(4.9, abs=1e-6)
      assert all(p.codec == "h264" for p in packets)
      await source.close()


  async def test_concurrent_drain_of_both_streams_agrees_on_counts() -> None:
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)

      async def collect_frames() -> int:
          return len([f async for f in source])

      frames_count, packets = await asyncio.wait_for(
          asyncio.gather(collect_frames(), _collect_packets(source)), timeout=5.0
      )
      assert frames_count == 50
      assert len(packets) == 50
      await source.close()


  async def test_only_iterating_frames_never_touching_packets_does_not_deadlock() -> None:
      """The literal crux case: packets() is never even called."""
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)

      async def collect_frames() -> int:
          return len([f async for f in source])

      frames_count = await asyncio.wait_for(collect_frames(), timeout=5.0)
      assert frames_count == 50
      await source.close()


  async def test_realtime_false_yields_as_fast_as_possible() -> None:
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)
      start = time.monotonic()
      _ = [f async for f in source]
      elapsed = time.monotonic() - start
      assert elapsed < 2.0, "realtime=False must not pace to the file's own framerate"
      await source.close()


  async def test_preroll_buffer_against_the_real_synthetic_clip() -> None:
      """Real PyAV demuxing feeding the buffer proves GOP quantisation end to end."""
      source = FileSource(ASSET, camera_id="cam-1", realtime=False)
      buffer = PreRollBuffer(preroll_seconds=0.25)
      drain_frames = asyncio.ensure_future(_drain(source))

      async def fill_buffer() -> None:
          async for packet in source.packets():
              buffer.append(packet)

      await asyncio.wait_for(fill_buffer(), timeout=5.0)
      await drain_frames
      flushed = buffer.flush()
      assert flushed[0].is_keyframe
      assert flushed[0].pts == 4.0  # last GOP boundary <= (4.9 - 0.25)
      await source.close()
  ```

- [ ] **Step 7: Run it to verify it fails**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/test_file_source.py -v`
  Expected: FAIL — errors at collection with
  `ModuleNotFoundError: No module named 'sentinel_ai.adapters.sources.file'`.

- [ ] **Step 8: Implement `FileSource`**

  One PyAV demux pass must serve two independent async iterators: `__aiter__` (decoded
  `FrameData`, for inference) and `packets()` (encoded `EncodedPacket`, for the pre-roll
  buffer). The demux+decode loop runs on a background thread (PyAV calls are blocking; the
  event loop must not stall on them), which hands items to each consumer through a
  thread-safe `queue.Queue`, bridged into `asyncio` via `loop.run_in_executor`.

  The two queues get different backpressure policies, and that asymmetry is the actual design:

  - **Frame queue: real (blocking) backpressure.** `__aiter__` must never silently drop a
    frame, so the pump's `put()` blocks when the queue is full, naturally throttling the pump
    to the consumer's pace. This is what makes "a consumer that only iterates `__aiter__` and
    never touches `packets()`" work correctly and deterministically: the pump only ever waits
    on *this* queue, and since that consumer is by definition draining it, the pump always
    makes progress. (A non-blocking/drop policy here would be actively wrong, not just
    unnecessary: since the pump runs on a real OS thread with no cooperative yield points of
    its own, it could race through the entire file before the consumer's coroutine is even
    scheduled, silently dropping every frame but the last few — verified by first writing the
    implementation this way and watching `test_decodes_every_frame_in_order` fail
    non-deterministically.)
  - **Packet queue: non-blocking, drop-oldest.** If nothing ever calls `packets()`, this queue
    would otherwise grow without bound for the lifetime of the source. `_drop_oldest_put`
    never blocks the pump, so an absent packet consumer cannot stall frame delivery either —
    the two queues cannot deadlock each other.

  The accepted asymmetry: draining *only* `packets()` and never `__aiter__` is not supported —
  the pump would eventually block on the frame queue once it fills, since nothing is asked to
  bound frame delivery instead. This does not arise in the real system: `CameraRunner` (Task 6)
  drains both from the moment a camera starts, because the pre-roll buffer must be fed
  continuously regardless of whether an escalation ever fires. Every test in Step 6 that
  exercises `packets()` therefore also drains frames concurrently, matching that invariant.

  `close()` polls the stop flag every 100ms while blocked on the frame queue, so a shutdown
  mid-stream can never hang waiting for a consumer that has already stopped reading.

  ```python
  # ai-engine/sentinel_ai/adapters/sources/file.py
  """FileSource: one PyAV demux pass fanned out to decoded frames and encoded
  packets (spec §4.2, §5.1)."""

  from __future__ import annotations

  import asyncio
  import contextlib
  import queue
  import threading
  import time
  from collections.abc import AsyncIterator

  import av

  from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource

  _FRAME_QUEUE_MAXSIZE = 8
  _PACKET_QUEUE_MAXSIZE = 256
  _STOP_POLL_SECONDS = 0.1


  class _QueueEnd:
      """Sentinel pushed onto both queues when the demux loop finishes or fails."""


  _QUEUE_END = _QueueEnd()


  def _drop_oldest_put[T](q: queue.Queue[T | _QueueEnd], item: T | _QueueEnd) -> None:
      """Non-blocking put that discards the oldest entry instead of blocking.

      Used only for the packet queue: see the design note above `FileSource`.
      """
      while True:
          try:
              q.put_nowait(item)
              return
          except queue.Full:
              with contextlib.suppress(queue.Empty):
                  q.get_nowait()


  class FileSource(FrameSource):
      """Demuxes `path` once and fans out to decoded frames and encoded packets.

      Frame delivery uses real (blocking) backpressure; the packet queue never
      blocks the pump. See the design note in the plan for why they differ.
      """

      def __init__(self, path: str, camera_id: str, realtime: bool) -> None:
          self._path = path
          self._camera_id = camera_id
          self._realtime = realtime
          self._frame_queue: queue.Queue[FrameData | _QueueEnd] = queue.Queue(
              maxsize=_FRAME_QUEUE_MAXSIZE
          )
          self._packet_queue: queue.Queue[EncodedPacket | _QueueEnd] = queue.Queue(
              maxsize=_PACKET_QUEUE_MAXSIZE
          )
          self._stop = threading.Event()
          self._pump_started = False
          self._pump_error: Exception | None = None
          self._thread: threading.Thread | None = None

      def _ensure_pump(self) -> None:
          if self._pump_started:
              return
          self._pump_started = True
          self._thread = threading.Thread(target=self._pump, daemon=True)
          self._thread.start()

      def _put_frame_blocking(self, item: FrameData | _QueueEnd) -> bool:
          """Blocking put, polling `_stop` every 100ms so `close()` always wins.

          A plain blocking `put()` would ignore a `close()` requested while the
          queue is full and the consumer has stopped pulling.
          """
          while not self._stop.is_set():
              try:
                  self._frame_queue.put(item, timeout=_STOP_POLL_SECONDS)
                  return True
              except queue.Full:
                  continue
          return False

      def _pump(self) -> None:
          try:
              container = av.open(self._path)
              try:
                  stream = container.streams.video[0]
                  time_base = stream.time_base
                  if time_base is None:
                      raise RuntimeError(f"video stream has no time_base: {self._path}")
                  rate = float(stream.average_rate) if stream.average_rate else 0.0
                  first_pts: int | None = None
                  frame_index = 0
                  start_wall = time.monotonic()
                  for packet in container.demux(stream):
                      if self._stop.is_set():
                          return
                      if packet.pts is None:
                          continue  # the trailing flush packet carries no timing
                      if first_pts is None:
                          first_pts = packet.pts
                      pts_seconds = float((packet.pts - first_pts) * time_base)

                      _drop_oldest_put(
                          self._packet_queue,
                          EncodedPacket(
                              camera_id=self._camera_id,
                              data=bytes(packet),
                              pts=pts_seconds,
                              is_keyframe=bool(packet.is_keyframe),
                              codec=stream.codec_context.name,
                          ),
                      )

                      for frame in packet.decode():
                          if self._realtime and rate > 0:
                              due = start_wall + frame_index / rate
                              delay = due - time.monotonic()
                              if delay > 0:
                                  time.sleep(delay)
                          pixels = frame.to_ndarray(format="bgr24")
                          frame_data = FrameData(
                              camera_id=self._camera_id,
                              frame_index=frame_index,
                              timestamp=pts_seconds,
                              width=pixels.shape[1],
                              height=pixels.shape[0],
                              pixels=pixels,
                          )
                          if not self._put_frame_blocking(frame_data):
                              return
                          frame_index += 1
              finally:
                  container.close()
          except Exception as exc:
              # Surfaced to whichever consumer notices the sentinel next, not
              # swallowed: a decode failure must not look like a clean end of stream.
              self._pump_error = exc
          finally:
              self._put_frame_blocking(_QUEUE_END)
              _drop_oldest_put(self._packet_queue, _QUEUE_END)

      def __aiter__(self) -> AsyncIterator[FrameData]:
          self._ensure_pump()

          async def frames() -> AsyncIterator[FrameData]:
              loop = asyncio.get_running_loop()
              while True:
                  item = await loop.run_in_executor(None, self._frame_queue.get)
                  if isinstance(item, _QueueEnd):
                      if self._pump_error is not None:
                          raise self._pump_error
                      return
                  yield item

          return frames()

      def packets(self) -> AsyncIterator[EncodedPacket]:
          self._ensure_pump()

          async def packet_stream() -> AsyncIterator[EncodedPacket]:
              loop = asyncio.get_running_loop()
              while True:
                  item = await loop.run_in_executor(None, self._packet_queue.get)
                  if isinstance(item, _QueueEnd):
                      if self._pump_error is not None:
                          raise self._pump_error
                      return
                  yield item

          return packet_stream()

      async def close(self) -> None:
          self._stop.set()
          if self._thread is not None:
              await asyncio.get_running_loop().run_in_executor(None, self._thread.join)
  ```

- [ ] **Step 9: Verify `FileSource`**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/test_file_source.py -v`
  Expected: 7 PASS.

- [ ] **Step 10: Full verify**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
  Expected: all PASS, ruff clean, mypy `Success` (no errors).

- [ ] **Step 11: Commit**

  ```bash
  git add ai-engine/tests/assets/synthetic_clip.mp4 \
      ai-engine/sentinel_ai/adapters/sources/__init__.py \
      ai-engine/sentinel_ai/adapters/sources/preroll.py \
      ai-engine/sentinel_ai/adapters/sources/file.py \
      ai-engine/tests/adapters/test_preroll.py \
      ai-engine/tests/adapters/test_file_source.py
  git commit -m "$(cat <<'EOF'
  feat(adapters): add PreRollBuffer, FileSource, and a synthetic clip fixture

  PreRollBuffer holds encoded packets keyframe-aligned to the pre-roll
  horizon (spec §5.1): a clip cannot start mid-GOP, so flush() walks back to
  the newest keyframe at or before the target and append() evicts everything
  provably unreachable from any future flush.

  FileSource fans one PyAV demux pass out to __aiter__ (decoded frames) and
  packets() (encoded packets) via a background thread and two queues with
  different backpressure policies: frame delivery blocks (never drops),
  packet delivery drops the oldest entry rather than ever stalling the pump
  -- so a consumer that only iterates frames and never touches packets()
  cannot deadlock or grow memory without bound.

  The synthetic clip is ffmpeg-generated with a pinned 1s GOP so keyframe
  positions are deterministic in CI, with no network and no GPU.
  EOF
  )"
  ```

---

### Task 5: `MotionAnalyzer`

Implements S6 exactly. `pipeline/stages/motion.py` is the outermost layer the escalation gate's
inputs pass through, so numpy is allowed here — and only here among the things the gate
touches; `domain/` and `ports/` stay numpy-free, enforced by `tests/test_architecture.py`.

**Files:**
- Create: `ai-engine/sentinel_ai/pipeline/__init__.py`
- Create: `ai-engine/sentinel_ai/pipeline/stages/__init__.py`
- Create: `ai-engine/sentinel_ai/pipeline/stages/motion.py`
- Create: `ai-engine/tests/pipeline/__init__.py` (new test package directory)
- Test: `ai-engine/tests/pipeline/test_motion.py`

**Interfaces:**
- Consumes: `FrameData` (`sentinel_ai/ports/frame_source.py`, unchanged by this task).
- Produces (verbatim, S6): `SIGNATURE_BINS`, `MotionSignals(motion_energy, scene_signature)`,
  `MotionAnalyzer(bins=SIGNATURE_BINS).analyze(frame) -> MotionSignals`, `.reset()`. Task 6
  (`CameraRunner`) constructs one `MotionAnalyzer` per camera and feeds `MotionSignals` into
  `SceneState.motion_energy` / `.scene_signature`.

- [ ] **Step 1: Write the failing tests**

  ```python
  # ai-engine/tests/pipeline/test_motion.py
  from __future__ import annotations

  import numpy as np
  import pytest

  from sentinel_ai.pipeline.stages.motion import SIGNATURE_BINS, MotionAnalyzer
  from sentinel_ai.ports.frame_source import FrameData


  def a_frame(pixels: np.ndarray) -> FrameData:
      height, width = pixels.shape[0], pixels.shape[1]
      return FrameData(
          camera_id="cam-1",
          frame_index=0,
          timestamp=0.0,
          width=width,
          height=height,
          pixels=pixels,
      )


  def solid(value: int, height: int = 8, width: int = 8) -> np.ndarray:
      return np.full((height, width, 3), value, dtype=np.uint8)


  def signature_delta(a: tuple[float, ...], b: tuple[float, ...]) -> float:
      """Mirrors `SceneState.signature_delta`'s total-variation formula, so
      this test pins the exact property the escalation gate's SceneChange
      trigger depends on without importing the domain layer into a pipeline test."""
      return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / 2.0


  class TestMotionAnalyzer:
      def test_first_frame_yields_exactly_zero_energy(self) -> None:
          analyzer = MotionAnalyzer()
          signals = analyzer.analyze(a_frame(solid(128)))
          assert signals.motion_energy == 0.0

      def test_identical_frames_yield_zero_energy(self) -> None:
          analyzer = MotionAnalyzer()
          analyzer.analyze(a_frame(solid(128)))
          signals = analyzer.analyze(a_frame(solid(128)))
          assert signals.motion_energy == 0.0

      def test_a_fully_inverted_frame_yields_energy_near_one(self) -> None:
          analyzer = MotionAnalyzer()
          analyzer.analyze(a_frame(solid(0)))
          signals = analyzer.analyze(a_frame(solid(255)))
          assert signals.motion_energy == pytest.approx(1.0)

      def test_signature_sums_to_one_and_has_the_configured_length(self) -> None:
          analyzer = MotionAnalyzer(bins=SIGNATURE_BINS)
          signals = analyzer.analyze(a_frame(solid(128)))
          assert len(signals.scene_signature) == SIGNATURE_BINS
          assert sum(signals.scene_signature) == pytest.approx(1.0)

      def test_black_and_white_frames_have_disjoint_signatures_with_delta_one(self) -> None:
          analyzer = MotionAnalyzer()
          black = analyzer.analyze(a_frame(solid(0))).scene_signature
          white = analyzer.analyze(a_frame(solid(255))).scene_signature
          assert signature_delta(black, white) == pytest.approx(1.0)

      def test_reset_drops_the_previous_frame(self) -> None:
          """A no-op reset would still compare against the black frame below,
          giving energy near 1.0 instead of exactly 0.0."""
          analyzer = MotionAnalyzer()
          analyzer.analyze(a_frame(solid(0)))
          analyzer.reset()
          signals = analyzer.analyze(a_frame(solid(255)))
          assert signals.motion_energy == 0.0

      def test_a_resolution_change_is_treated_as_a_discontinuity_not_motion(self) -> None:
          analyzer = MotionAnalyzer()
          analyzer.analyze(a_frame(solid(0, height=8, width=8)))
          signals = analyzer.analyze(a_frame(solid(255, height=16, width=16)))
          assert signals.motion_energy == 0.0

      def test_rejects_non_positive_bin_count(self) -> None:
          with pytest.raises(ValueError, match="bins"):
              MotionAnalyzer(bins=0)

      def test_rejects_non_ndarray_pixels(self) -> None:
          analyzer = MotionAnalyzer()
          bad_frame = FrameData(
              camera_id="cam-1", frame_index=0, timestamp=0.0, width=1, height=1, pixels=[[0]]
          )
          with pytest.raises(TypeError, match="numpy array"):
              analyzer.analyze(bad_frame)
  ```

- [ ] **Step 2: Run it to verify it fails**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/pipeline/test_motion.py -v`
  Expected: FAIL — errors at collection with
  `ModuleNotFoundError: No module named 'sentinel_ai.pipeline'` (the package does not exist
  anywhere in the tree yet).

- [ ] **Step 3: Implement**

  `FrameData.pixels` is typed `object` so numpy stays out of `ports/`; `_as_pixels` below is
  the one place that narrows it back, via a real `isinstance` check against a concrete numpy
  type — which is what lets mypy strict narrow `object` to `npt.NDArray[np.uint8]` without a
  `cast`, and it is a genuine runtime check, not a type-checker placation, because nothing
  upstream guarantees a caller always hands in a numpy array.

  ```python
  # ai-engine/sentinel_ai/pipeline/__init__.py
  """Per-camera inference pipeline: decode -> detect -> track -> motion -> gate."""
  ```

  ```python
  # ai-engine/sentinel_ai/pipeline/stages/__init__.py
  """Individual pipeline stages, run in order by `pipeline.runner.CameraRunner`."""
  ```

  ```python
  # ai-engine/sentinel_ai/pipeline/stages/motion.py
  """Motion stage (spec §5.2): motion energy and scene signature.

  Outermost layer, so numpy is allowed here -- and nowhere in `domain/` or
  `ports/` (the architecture fitness test enforces that; this module must never
  be imported from either).
  """

  from __future__ import annotations

  from dataclasses import dataclass

  import numpy as np
  import numpy.typing as npt

  from sentinel_ai.ports.frame_source import FrameData

  SIGNATURE_BINS: int = 16


  @dataclass(frozen=True, slots=True)
  class MotionSignals:
      motion_energy: float
      scene_signature: tuple[float, ...]


  def _as_pixels(value: object) -> npt.NDArray[np.uint8]:
      """Narrow `FrameData.pixels` (typed `object` to keep numpy out of ports)."""
      if not isinstance(value, np.ndarray):
          raise TypeError(f"FrameData.pixels must be a numpy array, got {type(value).__name__}")
      return value


  def _luma(pixels: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
      """BT.601 luma. `pixels` is `to_ndarray(format='bgr24')` shaped (H, W, 3),
      or already single-channel (H, W)."""
      if pixels.ndim == 2:
          return pixels.astype(np.float64)
      blue = pixels[..., 0].astype(np.float64)
      green = pixels[..., 1].astype(np.float64)
      red = pixels[..., 2].astype(np.float64)
      return 0.114 * blue + 0.587 * green + 0.299 * red


  def _signature(luma: npt.NDArray[np.float64], bins: int) -> tuple[float, ...]:
      counts, _ = np.histogram(luma, bins=bins, range=(0.0, 255.0))
      total = counts.sum()
      if total == 0:
          return tuple(1.0 / bins for _ in range(bins))
      normalized = counts / total
      return tuple(float(value) for value in normalized)


  class MotionAnalyzer:
      def __init__(self, bins: int = SIGNATURE_BINS) -> None:
          if bins <= 0:
              raise ValueError(f"bins must be positive, got {bins}")
          self._bins = bins
          self._previous: npt.NDArray[np.uint8] | None = None

      def analyze(self, frame: FrameData) -> MotionSignals:
          """Motion energy vs the previous frame, plus a normalised luma histogram.

          The first frame yields motion_energy == 0.0.
          """
          pixels = _as_pixels(frame.pixels)

          if self._previous is None or self._previous.shape != pixels.shape:
              # No previous frame, or a resolution change (a discontinuity, not
              # motion) -- treated exactly like the very first frame of a stream.
              energy = 0.0
          else:
              current = pixels.astype(np.float64)
              previous = self._previous.astype(np.float64)
              energy = float(np.mean(np.abs(current - previous))) / 255.0

          self._previous = pixels
          signature = _signature(_luma(pixels), self._bins)
          return MotionSignals(motion_energy=energy, scene_signature=signature)

      def reset(self) -> None:
          """Drop the previous frame -- call on stream discontinuity."""
          self._previous = None
  ```

- [ ] **Step 4: Full verify**

  Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
  Expected: all PASS, ruff clean, mypy `Success` (no errors).

- [ ] **Step 5: Commit**

  ```bash
  git add ai-engine/sentinel_ai/pipeline/__init__.py \
      ai-engine/sentinel_ai/pipeline/stages/__init__.py \
      ai-engine/sentinel_ai/pipeline/stages/motion.py \
      ai-engine/tests/pipeline/__init__.py \
      ai-engine/tests/pipeline/test_motion.py
  git commit -m "$(cat <<'EOF'
  feat(pipeline): add MotionAnalyzer -- motion energy and scene signature

  Mean absolute luma difference against the previous frame, normalised to
  [0, 1], plus a normalised 16-bin luma histogram (spec §5.2) for
  SceneState.signature_delta's total-variation metric. A resolution change
  is treated as a discontinuity (energy 0.0), matching how the domain layer
  already treats a signature-length change (review finding B1) rather than
  raising mid-stream.

  This is the first module under pipeline/, and the first place numpy is
  imported outside adapters/: FrameData.pixels stays typed `object` in
  ports/ so the architecture fitness test's numpy ban there holds.
  EOF
  )"
  ```


---

### Task 6: `CameraRunner` — end-to-end with fakes

Implements S11. Also implements S7 (`EscalationRequest`), S8 (`AdmissionGate`) and S9
(`VlmScheduler`, including the full S14 event-assembly worker) — see Reconciliation Log R8 for why those three land here instead of Task 7: `CameraRunner`'s
constructor takes a `scheduler: VlmScheduler` (S11) and its keystone test is a full
`source → gate → scheduler → publisher` trace, so a working scheduler must exist first. Task 7
consumes `VlmScheduler`/`AdmissionGate`/`EscalationRequest` from this task unchanged and adds
`registry.py`, `resident_set.py`, `service.py`, plus the S14 error-path tests.

**Files:**
- Modify: `ai-engine/tests/fakes/io.py` — `FakeSource` gains a synthetic `packets()` stream
  (S2/S3); `FakeClipWriter`/`FakeClipHandle` replace the old `write()` shape with S4's
  open/append/finish/abort.
- Create: `ai-engine/sentinel_ai/orchestrator/__init__.py`
- Create: `ai-engine/sentinel_ai/orchestrator/admission.py` — `AdmissionGate` (S8).
- Create: `ai-engine/sentinel_ai/orchestrator/scheduler.py` — `EscalationRequest` (S7),
  `VlmScheduler` (S9), the S14 worker.
- Create: `ai-engine/sentinel_ai/pipeline/runner.py` — `CameraTelemetry`, `CameraRunner` (S11).
  (`sentinel_ai/pipeline/__init__.py` and `pipeline/stages/motion.py` already exist from Task 5.)
- Create: `ai-engine/tests/orchestrator/__init__.py`
- Test: `ai-engine/tests/orchestrator/test_admission.py`
- Test: `ai-engine/tests/orchestrator/test_scheduler.py`
- Create: `ai-engine/tests/pipeline/__init__.py`
- Test: `ai-engine/tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: `EncodedPacket`/`FrameSource`/`FrameData` (S2/S3), `ClipWriter`/`ClipHandle` (S4),
  `PreRollBuffer` (S5), `MotionSignals`/`MotionAnalyzer` (S6), `GateState`/`GateOutcome`/
  `decide`/`force` (`sentinel_ai.domain.policy.escalation`), `CameraProfile`, `SceneState`,
  `Event`, `EscalationReason`, `ThreatScore` (`sentinel_ai.domain.entities`),
  `VisionLanguageModel`/`VisionRequest` (`sentinel_ai.ports.vision_llm`), `EventPublisher`.
- Produces: `EscalationRequest` (S7), `AdmissionGate` (S8), `VlmScheduler` (S9),
  `CameraTelemetry`/`CameraRunner` (S11) — all used verbatim by Task 7 and later by
  `orchestrator/service.py`.

- [ ] **Step 1: Extend the fakes for packets and clip handles**

`FakeSource` needs a packet stream so the pre-roll/clip path is testable without real media;
`FakeClipWriter` needs to satisfy S4's handle shape instead of the old single-shot `write()`.

```python
# ai-engine/tests/fakes/io.py
"""I/O fakes: frame sources, publishers, clip writers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource


class FakeSource(FrameSource):
    def __init__(
        self,
        frames: Sequence[FrameData],
        packets: Sequence[EncodedPacket] = (),
    ) -> None:
        self._frames = list(frames)
        self._packets = list(packets)
        self.closed = False

    @staticmethod
    def make_frame(camera_id: str, frame_index: int, timestamp: float, value: int = 0) -> FrameData:
        return FrameData(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp=timestamp,
            width=4,
            height=4,
            pixels=[[value] * 4 for _ in range(4)],
        )

    @staticmethod
    def make_packet(
        camera_id: str, pts: float, is_keyframe: bool = False, data: bytes = b"\x00"
    ) -> EncodedPacket:
        return EncodedPacket(
            camera_id=camera_id, data=data, pts=pts, is_keyframe=is_keyframe, codec="h264"
        )

    @classmethod
    def constant(cls, camera_id: str, count: int, fps: float = 10.0, value: int = 0) -> FakeSource:
        return cls([cls.make_frame(camera_id, index, index / fps, value) for index in range(count)])

    @classmethod
    def with_packets(
        cls,
        camera_id: str,
        count: int,
        fps: float = 10.0,
        keyframe_every: int = 5,
        value: int = 0,
    ) -> FakeSource:
        """A source whose packet stream mirrors its frames one-for-one.

        Every `keyframe_every`-th packet is a keyframe, matching a realistic GOP so
        `PreRollBuffer.flush()` (which walks back to a keyframe boundary) has one to find.
        """
        frames = [cls.make_frame(camera_id, index, index / fps, value) for index in range(count)]
        packets = [
            cls.make_packet(camera_id, index / fps, is_keyframe=(index % keyframe_every == 0))
            for index in range(count)
        ]
        return cls(frames, packets)

    def __aiter__(self) -> AsyncIterator[FrameData]:
        """Sync, matching the port and the async-iterator protocol.

        `async for` calls `__aiter__()` without awaiting it, so the method must
        return the iterator directly. Writing it as `async def` happened to work
        only because an `async def` containing `yield` is an async *generator*
        function — remove the yield and it breaks. The port's shape is the
        correct one, so the fake follows it.
        """

        async def frames() -> AsyncIterator[FrameData]:
            for frame in self._frames:
                yield frame

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        async def stream() -> AsyncIterator[EncodedPacket]:
            for packet in self._packets:
                yield packet

        return stream()

    async def close(self) -> None:
        self.closed = True


class FakePublisher(EventPublisher):
    def __init__(self, error: Exception | None = None) -> None:
        self.events: list[Event] = []
        self.closed = False
        self._error = error

    async def publish(self, event: Event) -> None:
        if self._error is not None:
            raise self._error
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class FakeClipHandle(ClipHandle):
    def __init__(
        self,
        camera_id: str,
        event_id: UUID,
        *,
        finish_error: Exception | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.event_id = event_id
        self.appended: list[EncodedPacket] = []
        self.finished = False
        self.aborted = False
        self._finish_error = finish_error

    async def append(self, packet: EncodedPacket) -> None:
        self.appended.append(packet)

    async def finish(self) -> str:
        if self._finish_error is not None:
            raise self._finish_error
        self.finished = True
        return f"s3://sentinel-clips/{self.camera_id}/{self.event_id}.mp4"

    async def abort(self) -> None:
        self.aborted = True


class FakeClipWriter(ClipWriter):
    """Records every handle it opens so tests can inspect what was appended.

    `finish_error`, when set, is attached to every handle opened afterwards —
    enough to exercise the S14 "clip failure must not lose the event" path
    (Task 7) without needing per-handle configuration.
    """

    def __init__(self, finish_error: Exception | None = None) -> None:
        self.handles: list[FakeClipHandle] = []
        self._finish_error = finish_error

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle:
        handle = FakeClipHandle(camera_id, event_id, finish_error=self._finish_error)
        self.handles.append(handle)
        return handle
```

- [ ] **Step 2: Run the fake tests to confirm they still type-check and pass**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/fakes -q && mypy tests/fakes`
Expected: PASS (no test file targets `tests/fakes` directly, so this simply proves the module
imports and mypy accepts it before anything downstream depends on it).

- [ ] **Step 3: Write the failing test for `AdmissionGate`**

```python
# ai-engine/tests/orchestrator/test_admission.py
from __future__ import annotations

import asyncio

import pytest

from sentinel_ai.orchestrator import admission as admission_module
from sentinel_ai.orchestrator.admission import AdmissionGate


def test_construction_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        AdmissionGate(concurrency=0, min_interval_seconds=0.0)
    with pytest.raises(ValueError, match="min_interval_seconds"):
        AdmissionGate(concurrency=1, min_interval_seconds=-1.0)


async def test_in_flight_tracks_acquire_and_release() -> None:
    gate = AdmissionGate(concurrency=2, min_interval_seconds=0.0)
    await gate.acquire(now=0.0)
    assert gate.in_flight == 1
    await gate.acquire(now=0.0)
    assert gate.in_flight == 2
    gate.release(now=0.0)
    assert gate.in_flight == 1


async def test_two_concurrent_acquires_are_serialised_at_concurrency_one() -> None:
    """The keystone claim for S8: N per-camera governors cannot bound a global GPU,
    so a single global slot must make a second caller wait for the first to finish."""
    gate = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
    order: list[str] = []
    release_first = asyncio.Event()

    async def first() -> None:
        await gate.acquire(now=0.0)
        order.append("first-acquired")
        await release_first.wait()
        gate.release(now=0.0)
        order.append("first-released")

    async def second() -> None:
        # Give `first` a tick to acquire before this one even tries — asyncio.sleep(0)
        # only yields to the event loop once, it waits zero wall-clock time.
        await asyncio.sleep(0)
        assert gate.in_flight == 1, "first must already hold the only slot"
        await gate.acquire(now=0.0)
        order.append("second-acquired")

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0)
    assert not second_task.done(), "second must block on the semaphore, not run to completion"

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert order == ["first-acquired", "first-released", "second-acquired"]


async def test_min_interval_computes_the_correct_deficit_without_real_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`asyncio.sleep` is replaced with a recorder: this proves the deficit math is
    right (spec: "block until ... min_interval has elapsed") without CI ever waiting."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(admission_module, "_sleep", fake_sleep)
    gate = AdmissionGate(concurrency=5, min_interval_seconds=2.0)

    await gate.acquire(now=0.0)
    gate.release(now=0.0)
    await gate.acquire(now=0.5)  # only 0.5s later: 1.5s deficit against a 2.0s floor
    assert slept == pytest.approx([1.5])


async def test_ample_spacing_never_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(admission_module, "_sleep", fake_sleep)
    gate = AdmissionGate(concurrency=5, min_interval_seconds=2.0)

    await gate.acquire(now=0.0)
    gate.release(now=0.0)
    await gate.acquire(now=5.0)
    assert slept == []
```

- [ ] **Step 4: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_admission.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.orchestrator'`

- [ ] **Step 5: Implement `AdmissionGate`**

```python
# ai-engine/sentinel_ai/orchestrator/__init__.py
"""Model lifecycle, VLM admission and scheduling, and the engine entry point."""
```

```python
# ai-engine/sentinel_ai/orchestrator/admission.py
"""Process-wide GPU admission gate (spec §5.4) — new in Phase 1B, not in the original layout.

Phase 1A's governors — the token bucket, cooldown, dedup — are all per camera, but the
GPU they protect is global. With one camera the two are equivalent, so this is not a
bug today; with N cameras, N independently-permitting buckets could each legitimately
allow a call and collectively saturate the GPU. Building this seam now, while it is
trivially testable with a single camera, avoids retrofitting a global limiter into the
hot path once a second camera exists.
"""

from __future__ import annotations

import asyncio
from asyncio import sleep as _sleep


class AdmissionGate:
    def __init__(self, concurrency: int, min_interval_seconds: float) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        if min_interval_seconds < 0:
            raise ValueError(f"min_interval_seconds must be >= 0, got {min_interval_seconds}")
        self._semaphore = asyncio.Semaphore(concurrency)
        self._min_interval_seconds = min_interval_seconds
        self._last_acquired_at: float | None = None
        self._in_flight = 0

    async def acquire(self, now: float) -> None:
        """Block until a global slot is free AND min_interval has elapsed."""
        await self._semaphore.acquire()
        self._in_flight += 1
        if self._last_acquired_at is not None:
            deficit = self._min_interval_seconds - (now - self._last_acquired_at)
            if deficit > 0:
                await _sleep(deficit)
        self._last_acquired_at = now

    def release(self, now: float) -> None:
        del now  # no release-side interval policy today; kept for symmetry with acquire
        self._in_flight -= 1
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        return self._in_flight
```

- [ ] **Step 6: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_admission.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add ai-engine/sentinel_ai/orchestrator/__init__.py ai-engine/sentinel_ai/orchestrator/admission.py ai-engine/tests/orchestrator/__init__.py ai-engine/tests/orchestrator/test_admission.py
git commit -m "feat(orchestrator): add the process-wide VLM admission gate

Per-camera governors cannot bound a shared GPU once a second camera exists;
this seam is cheap to build and test now, with one camera, than to retrofit
into the hot path later."
```

- [ ] **Step 8: Write the failing test for `EscalationRequest` + `VlmScheduler`**

Covers the happy path and drop-on-full; the S14 error paths (VLM timeout, clip-finish
failure) are Task 7's job, against this same implementation.

```python
# ai-engine/tests/orchestrator/test_scheduler.py
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, Detection, EscalationReason, SceneState
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.ports.vision_llm import SceneDescription
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeVisionLLM

CLOCK = iter([0.0] * 1000)


def clock() -> float:
    return next(CLOCK)


def a_request(clip=None) -> EscalationRequest:
    frame = FakeSource.make_frame("cam-1", 0, 0.0)
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),),
        tracks=(),
        motion_energy=0.1,
        scene_signature=(1.0,),
    )
    return EscalationRequest(
        camera_id="cam-1",
        event_id=uuid4(),
        reason=EscalationReason.PERIODIC_SUMMARY,
        detail="initial scene summary",
        scene=scene,
        keyframe=frame,
        profile=CameraProfile(camera_id="cam-1"),
        camera_label="Front Door",
        history=(),
        clip=clip,
    )


def new_scheduler(
    vlm: FakeVisionLLM | None = None,
    publisher: FakePublisher | None = None,
    maxsize: int = 4,
) -> tuple[VlmScheduler, FakeVisionLLM, FakePublisher]:
    vlm = vlm or FakeVisionLLM(
        response=SceneDescription(
            description="A person is standing near the door.",
            threat_value=0.3,
            suggested_action="Monitor.",
        )
    )
    publisher = publisher or FakePublisher()
    scheduler = VlmScheduler(
        vlm=vlm,
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=maxsize,
        timeout_seconds=5.0,
        clock=clock,
    )
    return scheduler, vlm, publisher


async def test_a_submitted_escalation_is_described_and_published() -> None:
    scheduler, vlm, publisher = new_scheduler()
    worker = asyncio.create_task(scheduler.run())
    try:
        assert scheduler.submit(a_request()) is True
        await scheduler.drain()
        assert len(publisher.events) == 1
        event = publisher.events[0]
        assert event.description == "A person is standing near the door."
        assert event.threat.value == pytest.approx(0.3)
        assert event.description_unavailable is False
        assert vlm.call_count == 1
    finally:
        worker.cancel()


async def test_a_clip_is_finished_and_its_uri_attached() -> None:
    writer = FakeClipWriter()
    handle = await writer.open("cam-1", uuid4(), fps=10.0)
    scheduler, _vlm, publisher = new_scheduler()
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request(clip=handle))
        await scheduler.drain()
        assert publisher.events[0].clip_uri == f"s3://sentinel-clips/cam-1/{handle.event_id}.mp4"
        assert handle.finished is True
    finally:
        worker.cancel()


async def test_no_clip_leaves_clip_uri_none() -> None:
    scheduler, _vlm, publisher = new_scheduler()
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request(clip=None))
        await scheduler.drain()
        assert publisher.events[0].clip_uri is None
    finally:
        worker.cancel()


async def test_a_full_queue_drops_and_counts_without_raising() -> None:
    scheduler, _vlm, _publisher = new_scheduler(maxsize=1)
    # No worker running: nothing drains the queue, so the second submit finds it full.
    assert scheduler.submit(a_request()) is True
    assert scheduler.submit(a_request()) is False
    assert scheduler.dropped == 1
```

- [ ] **Step 9: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.orchestrator.scheduler'`

- [ ] **Step 10: Implement `EscalationRequest` and `VlmScheduler`**

```python
# ai-engine/sentinel_ai/orchestrator/scheduler.py
"""The VLM escalation queue (spec §5.4) and the S14 event-assembly worker (spec §6, §9).

Exactly one worker processes escalations: there is one GPU and one set of resident
VLM weights, so concurrent `describe()` calls would contend for the same VRAM.
Drop-on-full is safe because the token bucket and the cooldown already bound the
camera-side arrival rate (spec §4.1) — a full queue should be rare in practice, and
when it happens the drop is counted, never raised.

Event assembly happens here and only here (S14): this is the one place in the system
that constructs an `Event`. Every error path below still produces one — spec §9's
governing rule is that an anomaly event is never lost to an infrastructure failure.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import Event, EscalationReason, SceneState, ThreatScore
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.vision_llm import VisionLanguageModel, VisionRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EscalationRequest:
    """What the runner hands the scheduler. Carries everything the VLM and the event
    need, so the worker never reaches back into the pipeline for anything."""

    camera_id: str
    event_id: UUID
    reason: EscalationReason
    detail: str
    scene: SceneState
    keyframe: FrameData
    profile: CameraProfile
    camera_label: str
    history: tuple[str, ...]
    clip: ClipHandle | None


_UNAVAILABLE_THREAT_VALUE = 0.5
"""A conservative mid-range placeholder: severity truly is unknown without a
description, and 0.5 neither over- nor under-states it for downstream triage."""


def _labels_and_tracks(scene: SceneState) -> tuple[tuple[str, ...], tuple[int, ...]]:
    labels = tuple(sorted({track.label for track in scene.tracks}))
    track_ids = tuple(track.track_id for track in scene.tracks)
    return labels, track_ids


def _metadata_description(request: EscalationRequest) -> str:
    """Fallback description built from cheap signals alone — no VLM call required."""
    labels, _ = _labels_and_tracks(request.scene)
    what = ", ".join(labels) if labels else "motion"
    return f"{request.reason.value}: {what} ({request.detail})"


class VlmScheduler:
    def __init__(
        self,
        vlm: VisionLanguageModel,
        publisher: EventPublisher,
        admission: AdmissionGate,
        *,
        maxsize: int,
        timeout_seconds: float,
        clock: Callable[[], float],
    ) -> None:
        self._vlm = vlm
        self._publisher = publisher
        self._admission = admission
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._queue: asyncio.Queue[EscalationRequest] = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0

    def submit(self, request: EscalationRequest) -> bool:
        """Non-blocking. Returns False and counts a drop when the queue is full."""
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning("vlm queue full: dropping escalation for camera %s", request.camera_id)
            return False
        return True

    async def run(self) -> None:
        """The single worker loop. Cancel to stop."""
        while True:
            request = await self._queue.get()
            try:
                await self._process(request)
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        """Await completion of queued work — tests only."""
        await self._queue.join()

    @property
    def dropped(self) -> int:
        return self._dropped

    async def _process(self, request: EscalationRequest) -> None:
        now = self._clock()
        await self._admission.acquire(now)
        try:
            event = await self._describe(request)
            event = await self._attach_clip(event, request)
            await self._publisher.publish(event)
        finally:
            self._admission.release(self._clock())

    async def _describe(self, request: EscalationRequest) -> Event:
        labels, track_ids = _labels_and_tracks(request.scene)
        vlm_request = VisionRequest(
            keyframe=request.keyframe,
            scene=request.scene,
            history=request.history,
            camera_label=request.camera_label,
            reason_detail=request.detail,
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                description = await self._vlm.describe(vlm_request)
        except Exception as error:  # noqa: BLE001 -- S14 step 4: never lose the event
            logger.warning(
                "vlm describe failed for camera %s event %s: %s",
                request.camera_id,
                request.event_id,
                error,
            )
            return Event(
                event_id=request.event_id,
                camera_id=request.camera_id,
                occurred_at=request.scene.timestamp,
                reason=request.reason,
                threat=ThreatScore.from_value(_UNAVAILABLE_THREAT_VALUE),
                description=_metadata_description(request),
                suggested_action="Review the clip when available.",
                labels=labels,
                track_ids=track_ids,
                description_unavailable=True,
            )
        return Event(
            event_id=request.event_id,
            camera_id=request.camera_id,
            occurred_at=request.scene.timestamp,
            reason=request.reason,
            threat=ThreatScore.from_value(description.threat_value),
            description=description.description,
            suggested_action=description.suggested_action,
            labels=labels,
            track_ids=track_ids,
            description_unavailable=False,
        )

    async def _attach_clip(self, event: Event, request: EscalationRequest) -> Event:
        if request.clip is None:
            return event
        try:
            clip_uri = await request.clip.finish()
        except Exception as error:  # noqa: BLE001 -- S14 step 5: never lose the event
            logger.warning(
                "clip finish failed for camera %s event %s: %s",
                request.camera_id,
                request.event_id,
                error,
            )
            return event
        return replace(event, clip_uri=clip_uri)
```

`asyncio.CancelledError` is a `BaseException`, not an `Exception`, so `except Exception` above
never swallows the worker's own cancellation — `run()` still stops cleanly.

- [ ] **Step 11: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_scheduler.py -v && ruff check ai-engine/sentinel_ai/orchestrator && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 12: Commit**

```bash
git add ai-engine/sentinel_ai/orchestrator/scheduler.py ai-engine/tests/orchestrator/test_scheduler.py
git commit -m "feat(orchestrator): add the bounded VLM scheduler and S14 event assembly

One worker because there is one GPU; every error path still publishes an
event, per spec §9's governing rule."
```

- [ ] **Step 13: Write the failing test for `CameraRunner`**

One file, four behaviours: backpressure actually drops (fails against a naive
sequential implementation), the keystone end-to-end trace, discontinuity reset, and the
clip open/extend/postroll lifecycle including `describe_now`.

```python
# ai-engine/tests/pipeline/test_runner.py
from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID

import pytest

from sentinel_ai.config import Settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline import runner as runner_module
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeTracker, FakeVisionLLM


def patch_postroll(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """`CameraRunner` reads `clip_postroll_seconds` from settings (S11's fixed
    constructor has no such parameter); tests pin it small and explicit."""
    monkeypatch.setattr(
        runner_module, "get_settings", lambda: Settings(clip_postroll_seconds=seconds)
    )


def clock_from(sequence: list[float]) -> Callable[[], float]:
    it = iter(sequence)

    def clock() -> float:
        try:
            return next(it)
        except StopIteration:
            return sequence[-1]

    return clock


def make_runner(
    *,
    frames_and_packets: FakeSource,
    detector: FakeDetector,
    scheduler: VlmScheduler,
    clip_writer: FakeClipWriter | None,
    profile: CameraProfile | None = None,
    detect_every_n_frames: int = 1,
) -> CameraRunner:
    return CameraRunner(
        camera_id="cam-1",
        camera_label="Front Door",
        source=frames_and_packets,
        detector=detector,
        tracker=FakeTracker(),
        motion=MotionAnalyzer(),
        profile=profile or CameraProfile(camera_id="cam-1"),
        scheduler=scheduler,
        clip_writer=clip_writer,
        preroll=PreRollBuffer(preroll_seconds=3.0),
        clock=clock_from([0.0]),
        detect_every_n_frames=detect_every_n_frames,
    )


def new_scheduler(vlm: FakeVisionLLM, publisher: FakePublisher) -> VlmScheduler:
    return VlmScheduler(
        vlm=vlm,
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock_from([0.0]),
    )


class TestKeystoneTrace:
    async def test_source_to_gate_to_scheduler_to_publisher_on_cpu_with_no_sleeping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The phase's keystone test: proves the whole slice composes without a GPU."""
        patch_postroll(monkeypatch, seconds=0.0)
        source = FakeSource.with_packets("cam-1", count=3, fps=10.0)
        detector = FakeDetector(script=[()])  # empty detections: PERIODIC_SUMMARY still fires
        publisher = FakePublisher()
        vlm = FakeVisionLLM()
        scheduler = new_scheduler(vlm, publisher)
        runner = make_runner(
            frames_and_packets=source, detector=detector, scheduler=scheduler, clip_writer=None
        )

        worker = asyncio.create_task(scheduler.run())
        try:
            await runner.run()
            await scheduler.drain()
        finally:
            worker.cancel()

        telemetry = runner.telemetry()
        assert telemetry.frames_seen == 3
        assert telemetry.detections_run == 3
        assert telemetry.escalations == 1  # PERIODIC_SUMMARY fires on the first scene
        assert len(publisher.events) == 1
        assert publisher.events[0].reason.value == "periodic_summary"


class TestBackpressure:
    async def test_a_slow_detector_causes_frame_drops_not_a_full_queue_of_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Must fail against a naive `async for frame in source: await detector.detect(frame)`
        sequential implementation, because that implementation never drops anything —
        it just processes every frame, however late."""
        patch_postroll(monkeypatch, seconds=5.0)
        source = FakeSource.with_packets("cam-1", count=50, fps=100.0)
        gate = asyncio.Event()

        class SlowDetector(FakeDetector):
            async def detect(self, frame):  # type: ignore[override]
                if self.call_count == 0:
                    await gate.wait()  # first call blocks; the source races ahead meanwhile
                return await super().detect(frame)

        detector = SlowDetector(script=[()])
        publisher = FakePublisher()
        scheduler = new_scheduler(FakeVisionLLM(), publisher)
        runner = make_runner(
            frames_and_packets=source, detector=detector, scheduler=scheduler, clip_writer=None
        )

        run_task = asyncio.create_task(runner.run())
        await asyncio.sleep(0)  # let the producer task race ahead of the gated detector
        await asyncio.sleep(0)
        gate.set()
        await run_task

        telemetry = runner.telemetry()
        assert telemetry.frames_seen == 50
        assert telemetry.frames_dropped > 0, "the producer must have outrun the gated detector"
        assert telemetry.detections_run < telemetry.frames_seen


class TestDiscontinuity:
    async def test_a_signature_length_change_resets_tracker_motion_and_gate_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=5.0)

        class VariableBinMotion(MotionAnalyzer):
            """A test double: the real `MotionAnalyzer` never changes bin count for a
            fixed instance, but the review's B1 crash was exactly a length mismatch, so
            this proves the runner's own defence independent of whether it can occur
            naturally today."""

            def __init__(self) -> None:
                super().__init__()
                self._script = iter([(0.1, (0.5, 0.5)), (0.2, (0.3, 0.3, 0.4))])

            def analyze(self, frame):  # type: ignore[override]
                energy, signature = next(self._script)
                return MotionSignals(motion_energy=energy, scene_signature=signature)

        source = FakeSource.constant("cam-1", count=2, fps=10.0)
        near = Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0))
        detector = FakeDetector(script=[(near,), (near,)])
        publisher = FakePublisher()
        scheduler = new_scheduler(FakeVisionLLM(), publisher)
        runner = CameraRunner(
            camera_id="cam-1",
            camera_label="Front Door",
            source=source,
            detector=detector,
            tracker=FakeTracker(),
            motion=VariableBinMotion(),
            profile=CameraProfile(camera_id="cam-1", min_track_frames=1),
            scheduler=scheduler,
            clip_writer=None,
            preroll=PreRollBuffer(preroll_seconds=3.0),
            clock=clock_from([0.0]),
        )

        await runner.run()

        # A gate that stayed wedged in state built from an incomparable signature
        # would not escalate cleanly on the second, discontinuous frame; a reset one
        # does, and `last_escalation_at` reflects that second frame's own timestamp.
        assert runner.telemetry().last_escalation_at == pytest.approx(0.1)


class TestClipLifecycle:
    async def test_a_second_escalation_while_recording_extends_the_postroll_instead_of_opening_a_new_clip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_postroll(monkeypatch, seconds=1.0)
        # PERIODIC_SUMMARY fires on frame 0; NEW_SALIENT_TRACK fires again once the
        # track has aged past min_track_frames, both while the clip is still open.
        source = FakeSource.with_packets("cam-1", count=15, fps=10.0, keyframe_every=3)
        near = Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0))
        detector = FakeDetector(script=[(near,)])
        writer = FakeClipWriter()
        publisher = FakePublisher()
        scheduler = new_scheduler(FakeVisionLLM(), publisher)
        runner = make_runner(
            frames_and_packets=source,
            detector=detector,
            scheduler=scheduler,
            clip_writer=writer,
            profile=CameraProfile(camera_id="cam-1", min_track_frames=2),
        )

        worker = asyncio.create_task(scheduler.run())
        try:
            await runner.run()
            await scheduler.drain()
        finally:
            worker.cancel()

        assert len(writer.handles) == 1, "the second escalation must not open a second clip"
        handle = writer.handles[0]
        assert handle.finished is True
        assert sum(event.clip_uri is not None for event in publisher.events) == 1, (
            "only the clip-owning escalation's event carries the uri"
        )

    async def test_describe_now_uses_force_and_bypasses_the_governors(self) -> None:
        source = FakeSource.constant("cam-1", count=1, fps=10.0)
        detector = FakeDetector(script=[()])
        publisher = FakePublisher()
        scheduler = new_scheduler(FakeVisionLLM(), publisher)
        profile = CameraProfile(camera_id="cam-1", vlm_enabled=False)  # governors would refuse
        runner = make_runner(
            frames_and_packets=source,
            detector=detector,
            scheduler=scheduler,
            clip_writer=None,
            profile=profile,
        )

        worker = asyncio.create_task(scheduler.run())
        try:
            await runner.run()  # observes the one frame, vlm_enabled=False so nothing escalates
            assert runner.telemetry().escalations == 0

            event_id = await runner.describe_now()
            await scheduler.drain()
        finally:
            worker.cancel()

        assert isinstance(event_id, UUID)
        assert len(publisher.events) == 1
        assert publisher.events[0].reason.value == "user_requested"

    async def test_describe_now_before_any_frame_raises(self) -> None:
        source = FakeSource.constant("cam-1", count=0)
        detector = FakeDetector(script=[()])
        scheduler = new_scheduler(FakeVisionLLM(), FakePublisher())
        runner = make_runner(
            frames_and_packets=source, detector=detector, scheduler=scheduler, clip_writer=None
        )
        with pytest.raises(RuntimeError, match="has not processed a frame"):
            await runner.describe_now()
```

- [ ] **Step 14: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/pipeline/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.pipeline.runner'`

- [ ] **Step 15: Implement `CameraRunner`**

```python
# ai-engine/sentinel_ai/pipeline/runner.py
"""One asyncio task per camera (spec §5.3): decode → detect → track → motion → gate.

Threading: `FrameSource.__aiter__`/`.packets()` and `ObjectDetector.detect` are async
by contract precisely because PyAV decode and YOLO inference are blocking C-extension
calls. The adapters that perform them (`FileSource`/`RtspSource`, `Yolo11Detector`) hide
the offload behind that `async def`, e.g. `return await asyncio.to_thread(self._predict, frame)`
— the runner's obligation is only to `await` these calls and never add a second,
redundant executor hop around an already-async port method. `Tracker.update` and
`MotionAnalyzer.analyze` are cheap, pure CPU (spec explicitly keeps them synchronous)
and the escalation gate is a pure function, so all three run inline on the loop.

Backpressure: a `_LatestSlot` mailbox decouples frame *arrival* from frame
*processing*. A `_produce` task drains the source eagerly and always overwrites the
slot with the newest frame; an overwrite before the consumer collects the previous
one is a drop, counted in `CameraTelemetry.frames_dropped`. This is what makes
"drop the stale frame, process the newest" real: a naive `async for frame in source:
await detector.detect(frame)` loop never drops anything, because it only ever asks
the source for a new frame once the previous one is fully processed.

Clip lifecycle: only the escalation that *opens* a clip ever carries its `ClipHandle`
on an `EscalationRequest`. A later escalation while the clip is still recording only
extends `_ActiveClip.deadline` — submitting its own request with `clip=None` — because
finishing a shared handle twice, or before its (possibly extended) post-roll window has
actually closed, is undefined. The clip-owning request is queued lazily, from the
packet loop, the moment `packet.pts` first reaches the (possibly-extended) deadline —
never from the frame loop, so escalation submission is always non-blocking and
immediate for every *other* request.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.config import get_settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.domain.policy.escalation import GateState, decide, force
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer, MotionSignals
from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData, FrameSource
from sentinel_ai.ports.tracker import Tracker

_DEFAULT_FPS = 10.0
_HISTORY_MAXLEN = 5


@dataclass(frozen=True, slots=True)
class CameraTelemetry:
    camera_id: str
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    last_frame_at: float | None
    last_escalation_at: float | None


@dataclass(frozen=True, slots=True)
class _ActiveClip:
    handle: ClipHandle
    deadline: float
    request: EscalationRequest


class _LatestSlot:
    """Single-slot mailbox: only the newest unconsumed frame survives.

    An overwrite before the consumer collects the previous frame is exactly the
    backpressure drop spec §5.3 requires — surveillance wants current reality, not
    a delayed complete record. `close()` lets the producer signal end-of-stream so
    the consumer can stop instead of waiting forever.
    """

    def __init__(self) -> None:
        self._frame: FrameData | None = None
        self._closed = False
        self._event = asyncio.Event()
        self.dropped = 0

    def put(self, frame: FrameData) -> None:
        if self._frame is not None:
            self.dropped += 1
        self._frame = frame
        self._event.set()

    def close(self) -> None:
        self._closed = True
        self._event.set()

    async def get(self) -> FrameData | None:
        while True:
            if self._frame is not None:
                frame, self._frame = self._frame, None
                return frame
            if self._closed:
                return None
            self._event.clear()
            await self._event.wait()


class CameraRunner:
    def __init__(
        self,
        *,
        camera_id: str,
        camera_label: str,
        source: FrameSource,
        detector: ObjectDetector,
        tracker: Tracker,
        motion: MotionAnalyzer,
        profile: CameraProfile,
        scheduler: VlmScheduler,
        clip_writer: ClipWriter | None,
        preroll: PreRollBuffer,
        clock: Callable[[], float],
        detect_every_n_frames: int = 1,
    ) -> None:
        self._camera_id = camera_id
        self._camera_label = camera_label
        self._source = source
        self._detector = detector
        self._tracker = tracker
        self._motion = motion
        self._profile = profile
        self._scheduler = scheduler
        self._clip_writer = clip_writer
        self._preroll = preroll
        self._clock = clock
        self._detect_every_n_frames = detect_every_n_frames

        # `clip_postroll_seconds` has no S11 constructor slot: it is a process-wide
        # tuning value (S1), read once here rather than threaded through every
        # `CameraRunner` construction site.
        self._clip_postroll_seconds = get_settings().clip_postroll_seconds

        self._gate_state = GateState.initial(profile, clock())
        self._history: deque[str] = deque(maxlen=_HISTORY_MAXLEN)
        self._active_clip: _ActiveClip | None = None
        self._clip_lock = asyncio.Lock()
        self._slot: _LatestSlot | None = None

        self._last_scene: SceneState | None = None
        self._last_keyframe: FrameData | None = None
        self._last_signature_len: int | None = None
        self._last_processed_timestamp: float | None = None
        self._last_two_timestamps: tuple[float, float] | None = None

        self._frames_seen = 0
        self._detections_run = 0
        self._escalations = 0
        self._escalations_dropped = 0
        self._discontinuities = 0
        self._last_frame_at: float | None = None
        self._last_escalation_at: float | None = None

    async def run(self) -> None:
        self._slot = _LatestSlot()
        produce_task = asyncio.create_task(self._produce())
        packets_task = asyncio.create_task(self._packet_loop())
        try:
            await self._consume()
        finally:
            for task in (produce_task, packets_task):
                task.cancel()
            for task in (produce_task, packets_task):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if self._active_clip is not None:
                pending = self._active_clip
                self._active_clip = None
                await pending.handle.abort()
                self._submit(replace(pending.request, clip=None))
            await self._source.close()

    async def describe_now(self) -> UUID:
        """USER_REQUESTED path — uses domain `force()`; returns the event id."""
        if self._last_scene is None or self._last_keyframe is None:
            raise RuntimeError(f"camera {self._camera_id!r} has not processed a frame yet")
        now = self._clock()
        outcome = force(EscalationReason.USER_REQUESTED, self._gate_state, now)
        self._gate_state = outcome.state
        assert outcome.decision.reason is not None
        return await self._escalate(
            self._last_scene,
            self._last_keyframe,
            outcome.decision.reason,
            outcome.decision.detail,
            now,
        )

    def telemetry(self) -> CameraTelemetry:
        return CameraTelemetry(
            camera_id=self._camera_id,
            frames_seen=self._frames_seen,
            frames_dropped=self._slot.dropped if self._slot is not None else 0,
            detections_run=self._detections_run,
            escalations=self._escalations,
            escalations_dropped=self._escalations_dropped,
            discontinuities=self._discontinuities,
            last_frame_at=self._last_frame_at,
            last_escalation_at=self._last_escalation_at,
        )

    # -- stages 1-2: frame arrival, with backpressure -------------------------------

    async def _produce(self) -> None:
        assert self._slot is not None
        try:
            async for frame in self._source:
                self._frames_seen += 1
                self._last_frame_at = frame.timestamp
                if frame.frame_index % self._detect_every_n_frames == 0:
                    self._slot.put(frame)
        finally:
            self._slot.close()

    async def _consume(self) -> None:
        assert self._slot is not None
        while True:
            frame = await self._slot.get()
            if frame is None:
                return
            await self._process_frame(frame)

    # -- stages 2-5: detect, track, motion, gate ------------------------------------

    async def _process_frame(self, frame: FrameData) -> None:
        detections = await self._detector.detect(frame)
        self._detections_run += 1
        tracks = self._tracker.update(detections, frame.timestamp)
        signals = self._motion.analyze(frame)

        if self._is_discontinuous(frame, signals):
            self._discontinuities += 1
            self._tracker.reset()
            self._motion.reset()
            self._gate_state = GateState.initial(self._profile, frame.timestamp)

        self._last_signature_len = len(signals.scene_signature)
        if self._last_processed_timestamp is not None:
            self._last_two_timestamps = (self._last_processed_timestamp, frame.timestamp)
        self._last_processed_timestamp = frame.timestamp

        scene = SceneState(
            camera_id=self._camera_id,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            detections=detections,
            tracks=tracks,
            motion_energy=signals.motion_energy,
            scene_signature=signals.scene_signature,
        )
        self._last_scene = scene
        self._last_keyframe = frame

        outcome = decide(scene, self._profile, self._gate_state)
        self._gate_state = outcome.state
        if outcome.decision.should_escalate and outcome.decision.reason is not None:
            await self._escalate(
                scene, frame, outcome.decision.reason, outcome.decision.detail, frame.timestamp
            )

    def _is_discontinuous(self, frame: FrameData, signals: MotionSignals) -> bool:
        """Spec §5.2/§6: a signature-length change or a source reconnect (observed as
        a timestamp regression) is a stream discontinuity, not a `SceneState` the
        domain can compare against what came before. The domain already returns
        0.0/None rather than raising (review finding B1); resetting the streak/state
        so the next frame starts clean is the caller's job."""
        if (
            self._last_processed_timestamp is not None
            and frame.timestamp < self._last_processed_timestamp
        ):
            return True
        return (
            self._last_signature_len is not None
            and len(signals.scene_signature) != self._last_signature_len
        )

    # -- escalation, clip lifecycle, scheduler submission ---------------------------

    async def _escalate(
        self,
        scene: SceneState,
        keyframe: FrameData,
        reason: EscalationReason,
        detail: str,
        now: float,
    ) -> UUID:
        event_id = uuid4()
        history = tuple(self._history)
        self._history.append(detail)
        self._last_escalation_at = now

        base = {
            "camera_id": self._camera_id,
            "event_id": event_id,
            "reason": reason,
            "detail": detail,
            "scene": scene,
            "keyframe": keyframe,
            "profile": self._profile,
            "camera_label": self._camera_label,
            "history": history,
        }

        if self._clip_writer is None:
            self._submit(EscalationRequest(clip=None, **base))
            return event_id

        async with self._clip_lock:
            if self._active_clip is None:
                handle = await self._clip_writer.open(
                    self._camera_id, event_id, fps=self._estimated_fps()
                )
                for packet in self._preroll.flush():
                    await handle.append(packet)
                request = EscalationRequest(clip=handle, **base)
                self._active_clip = _ActiveClip(
                    handle=handle,
                    deadline=now + self._clip_postroll_seconds,
                    request=request,
                )
            else:
                # A clip is already recording: extend its post-roll instead of
                # opening a second, overlapping one (spec §5.5). This escalation
                # still gets its own event — just without a clip of its own, since
                # a `ClipHandle` may only ever be finished once.
                self._active_clip = replace(
                    self._active_clip, deadline=now + self._clip_postroll_seconds
                )
                self._submit(EscalationRequest(clip=None, **base))
        return event_id

    async def _packet_loop(self) -> None:
        async for packet in self._source.packets():
            self._preroll.append(packet)
            async with self._clip_lock:
                if self._active_clip is not None:
                    await self._active_clip.handle.append(packet)
                    if packet.pts >= self._active_clip.deadline:
                        pending = self._active_clip
                        self._active_clip = None
                        self._submit(pending.request)

    def _submit(self, request: EscalationRequest) -> None:
        if self._scheduler.submit(request):
            self._escalations += 1
        else:
            self._escalations_dropped += 1

    def _estimated_fps(self) -> float:
        if self._last_two_timestamps is None:
            return _DEFAULT_FPS
        previous, current = self._last_two_timestamps
        delta = current - previous
        return 1.0 / delta if delta > 0 else _DEFAULT_FPS
```

The one subtlety worth flagging: `_packet_loop` holds `self._clip_lock` around every
append, so a live packet arriving mid-preroll-flush (itself inside the same lock, in
`_escalate`) simply waits for the lock instead of racing ahead of older, still-buffered
pre-roll packets — the clip's packets stay pts-ordered without a second queue.

- [ ] **Step 16: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 17: Commit**

```bash
git add ai-engine/sentinel_ai/pipeline/runner.py ai-engine/tests/pipeline/__init__.py ai-engine/tests/pipeline/test_runner.py ai-engine/tests/fakes/io.py
git commit -m "feat(pipeline): add CameraRunner — decode/detect/track/motion/gate on one task per camera

Backpressure drops stale frames under a slow detector rather than queuing a
delayed complete record; a second escalation while a clip records extends its
post-roll instead of opening a second overlapping one; a signature-length
change or source reconnect resets tracker, motion and gate state."
```

---

### Task 7: Orchestrator — registry, resident set, service (consuming Task 6's admission gate and scheduler)

Implements S10 (`ModelRegistry`, `ResidentSet`) and S12 (`EngineService`), and adds the
S14 error-path tests against Task 6's `VlmScheduler` — those paths are the point of this
task, not an afterthought. `AdmissionGate` (S8) and `VlmScheduler`/`EscalationRequest` (S7,
S9) already exist from Task 6 (Reconciliation Log R8).

**Files:**
- Modify: `ai-engine/tests/fakes/models.py` — add `FakeModelRuntime` (a `ModelRuntime` fake;
  nothing earlier defined one).
- Create: `ai-engine/sentinel_ai/orchestrator/registry.py` — `ModelSpec`, `ModelRegistry` (S10).
- Create: `ai-engine/sentinel_ai/orchestrator/resident_set.py` — `ResidentSet` (S10).
- Create: `ai-engine/sentinel_ai/orchestrator/service.py` — `EngineService`, `UnknownCameraError` (S12).
- Test: `ai-engine/tests/orchestrator/test_registry.py`
- Test: `ai-engine/tests/orchestrator/test_resident_set.py`
- Test: `ai-engine/tests/orchestrator/test_service.py`
- Modify: `ai-engine/tests/orchestrator/test_scheduler.py` — add the S14 error-path tests
  (VLM timeout, clip-finish failure) against Task 6's `VlmScheduler`.

**Interfaces:**
- Consumes: `AdmissionGate` (S8), `EscalationRequest`/`VlmScheduler` (S7/S9) verbatim from
  Task 6; `plan_residency`/`ModelSpec`/`ResidencyPlan`/`InsufficientVram` from
  `sentinel_ai.domain.policy.vram_budget` (Phase 1A); `ModelRuntime`/`HealthReport`/
  `LifecycleState`/`Capabilities` from `sentinel_ai.ports.model_runtime`; `CameraRunner`/
  `CameraTelemetry` (S11) from Task 6.
- Produces: `ModelSpec`/`ModelRegistry`/`ResidentSet` (S10), `EngineService`/
  `UnknownCameraError` (S12) — used verbatim by the FastAPI surface (Task 10) and by the
  real adapters' wiring (Tasks 11-14).

**One `ModelSpec`, not two.** Reuse `sentinel_ai.domain.policy.vram_budget.ModelSpec`
(`model_key`, `vram_mib`, `priority`, `idle_unload_seconds`) — do not define a registry-local
twin. `plan_residency()` consumes exactly that type, so a second one would have to be
translated on every call for no benefit. There is no `kind` field because a model's kind is
already on `ModelRuntime.capabilities().kind`, which is where `ModelRegistry.kind()` reads it
from. See Reconciliation Log R2.

- [ ] **Step 1: Add `FakeModelRuntime`**

```python
# ai-engine/tests/fakes/models.py — append to the existing file
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime


class FakeModelRuntime(ModelRuntime):
    """A controllable `ModelRuntime`: tests drive its lifecycle state directly rather
    than simulating a real load/warmup/shutdown sequence."""

    def __init__(
        self,
        model_key: str,
        kind: str = "vision",
        vram_mib: int = 100,
        initialize_error: Exception | None = None,
    ) -> None:
        self._model_key = model_key
        self._kind = kind
        self._vram_mib = vram_mib
        self._initialize_error = initialize_error
        self._state = LifecycleState.UNLOADED
        self.initialize_calls = 0
        self.warmup_calls = 0
        self.shutdown_calls = 0
        self.predict_calls: list[object] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1
        if self._initialize_error is not None:
            self._state = LifecycleState.UNHEALTHY
            raise self._initialize_error
        self._state = LifecycleState.LOADED

    async def warmup(self) -> None:
        self.warmup_calls += 1
        self._state = LifecycleState.HEALTHY

    async def predict(self, request: object) -> object:
        self.predict_calls.append(request)
        return request

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._state = LifecycleState.UNLOADED

    def health(self) -> HealthReport:
        vram = self._vram_mib if self._state != LifecycleState.UNLOADED else 0
        return HealthReport(state=self._state, vram_mib=vram)

    def version(self) -> str:
        return "fake-1"

    def capabilities(self) -> Capabilities:
        return Capabilities(model_key=self._model_key, kind=self._kind, vram_mib=self._vram_mib)
```

- [ ] **Step 2: Run it to confirm the fake type-checks**

Run: `cd ai-engine && . .venv/bin/activate && mypy tests/fakes`
Expected: `Success`.

- [ ] **Step 3: Write the failing test for `ModelRegistry`**

```python
# ai-engine/tests/orchestrator/test_registry.py
from __future__ import annotations

import pytest

from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.ports.model_runtime import LifecycleState
from tests.fakes.models import FakeModelRuntime

DETECTOR_SPEC = ModelSpec(
    model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None
)
VLM_SPEC = ModelSpec(
    model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0
)


def test_register_and_get_round_trip() -> None:
    registry = ModelRegistry()
    runtime = FakeModelRuntime("yolo11s")
    registry.register(DETECTOR_SPEC, runtime)
    assert registry.get("yolo11s") is runtime
    assert registry.specs() == (DETECTOR_SPEC,)


def test_get_of_an_unknown_key_raises() -> None:
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="qwen25vl3b"):
        registry.get("qwen25vl3b")


async def test_state_reflects_the_runtimes_own_health() -> None:
    registry = ModelRegistry()
    runtime = FakeModelRuntime("yolo11s")
    registry.register(DETECTOR_SPEC, runtime)
    assert registry.state("yolo11s") == LifecycleState.UNLOADED
    await runtime.initialize()
    assert registry.state("yolo11s") == LifecycleState.LOADED


async def test_health_aggregates_every_registered_runtime() -> None:
    registry = ModelRegistry()
    detector = FakeModelRuntime("yolo11s")
    vlm = FakeModelRuntime("qwen25vl3b")
    registry.register(DETECTOR_SPEC, detector)
    registry.register(VLM_SPEC, vlm)
    await detector.initialize()

    health = registry.health()

    assert set(health) == {"yolo11s", "qwen25vl3b"}
    assert health["yolo11s"].state == LifecycleState.LOADED
    assert health["qwen25vl3b"].state == LifecycleState.UNLOADED
```

- [ ] **Step 4: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.orchestrator.registry'`

- [ ] **Step 5: Implement `ModelRegistry`**

```python
# ai-engine/sentinel_ai/orchestrator/registry.py
"""Model discovery and the eight §5 lifecycle states (spec §5.4).

A thin lookup layer, deliberately: lifecycle truth lives in each runtime's own
`health()` (spec §10), so the registry never keeps a second, driftable copy of it.
"""

from __future__ import annotations

from sentinel_ai.domain.policy.vram_budget import ModelSpec
from sentinel_ai.ports.model_runtime import HealthReport, LifecycleState, ModelRuntime

__all__ = ["ModelRegistry", "ModelSpec"]
# ModelSpec is re-exported so callers import one name from one place; it is the
# domain type, not a copy (Reconciliation Log R2).


class ModelRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._runtimes: dict[str, ModelRuntime] = {}

    def register(self, spec: ModelSpec, runtime: ModelRuntime) -> None:
        self._specs[spec.model_key] = spec
        self._runtimes[spec.model_key] = runtime

    def get(self, key: str) -> ModelRuntime:
        try:
            return self._runtimes[key]
        except KeyError:
            raise KeyError(f"unknown model key: {key}") from None

    def state(self, key: str) -> LifecycleState:
        return self.get(key).health().state

    def specs(self) -> tuple[ModelSpec, ...]:
        return tuple(self._specs.values())

    def health(self) -> dict[str, HealthReport]:
        return {key: runtime.health() for key, runtime in self._runtimes.items()}
```

- [ ] **Step 6: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_registry.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add ai-engine/sentinel_ai/orchestrator/registry.py ai-engine/tests/orchestrator/test_registry.py ai-engine/tests/fakes/models.py
git commit -m "feat(orchestrator): add ModelRegistry — discovery plus the eight spec §5 lifecycle states

Lifecycle truth lives in each runtime's own health() so the registry never
keeps a second, driftable copy of it."
```

- [ ] **Step 8: Write the failing test for `ResidentSet`**

```python
# ai-engine/tests/orchestrator/test_resident_set.py
from __future__ import annotations

from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from tests.fakes.models import FakeModelRuntime

TOTAL_MIB = 8192
RESERVED_MIB = 2048

DETECTOR_SPEC = ModelSpec(
    model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None
)
VLM_SPEC = ModelSpec(
    model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0
)


def new_resident_set() -> tuple[ResidentSet, FakeModelRuntime, FakeModelRuntime]:
    registry = ModelRegistry()
    detector = FakeModelRuntime("yolo11s", vram_mib=900)
    vlm = FakeModelRuntime("qwen25vl3b", vram_mib=4400)
    registry.register(DETECTOR_SPEC, detector)
    registry.register(VLM_SPEC, vlm)
    return ResidentSet(registry, total_mib=TOTAL_MIB, reserved_mib=RESERVED_MIB), detector, vlm


class TestEnsure:
    async def test_a_required_model_is_loaded_and_warmed_up(self) -> None:
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        assert detector.initialize_calls == 1
        assert detector.warmup_calls == 1
        assert resident_set.resident() == frozenset({"yolo11s"})

    async def test_an_already_resident_model_is_not_reloaded(self) -> None:
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        await resident_set.ensure(("yolo11s",), now=1.0)
        assert detector.initialize_calls == 1

    async def test_both_models_fit_the_budget_together(self) -> None:
        resident_set, _detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s", "qwen25vl3b"), now=0.0)
        assert resident_set.resident() == frozenset({"yolo11s", "qwen25vl3b"})


class TestSweepIdle:
    async def test_the_vlm_is_unloaded_600s_after_its_last_ensure_call(self) -> None:
        resident_set, _detector, vlm = new_resident_set()
        await resident_set.ensure(("yolo11s", "qwen25vl3b"), now=0.0)
        await resident_set.sweep_idle(now=599.0)
        assert resident_set.resident() == frozenset({"yolo11s", "qwen25vl3b"})

        await resident_set.sweep_idle(now=600.0)
        assert resident_set.resident() == frozenset({"yolo11s"})
        assert vlm.shutdown_calls == 1

    async def test_the_detector_never_idle_evicts(self) -> None:
        resident_set, detector, _vlm = new_resident_set()
        await resident_set.ensure(("yolo11s",), now=0.0)
        await resident_set.sweep_idle(now=10_000.0)
        assert resident_set.resident() == frozenset({"yolo11s"})
        assert detector.shutdown_calls == 0

    async def test_re_ensuring_the_vlm_refreshes_its_idle_clock(self) -> None:
        resident_set, _detector, vlm = new_resident_set()
        await resident_set.ensure(("qwen25vl3b",), now=0.0)
        await resident_set.ensure(("qwen25vl3b",), now=500.0)  # a fresh "use" before 600s
        await resident_set.sweep_idle(now=1000.0)  # only 500s since the refresh
        assert resident_set.resident() == frozenset({"qwen25vl3b"})
        assert vlm.shutdown_calls == 0
```

- [ ] **Step 9: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_resident_set.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.orchestrator.resident_set'`

- [ ] **Step 10: Implement `ResidentSet`**

```python
# ai-engine/sentinel_ai/orchestrator/resident_set.py
"""Executes the Phase 1A `plan_residency()` output (spec §5.4): loads, unloads, and
applies the 600s VLM idle-unload via `sweep_idle()`.

`plan_residency()` is pure and takes the orchestrator's own bookkeeping (currently
resident, last-used) as arguments — this class is that bookkeeping plus the I/O
(`ModelRuntime.initialize`/`shutdown`) the plan calls for.

Freshening a model's idle clock is just calling `ensure()` again with that key in
`required`: `plan_residency` stamps `last_used_at` for every required key on every
call, so a caller invoking a model (e.g. wrapping a `VisionLanguageModel.describe`
call with `await resident_set.ensure((vlm_key,), now)` first) keeps it alive for as
long as it is genuinely being used, and `sweep_idle` — `ensure` with nothing
required — evicts it exactly `idle_unload_seconds` after the last such call.
"""

from __future__ import annotations

from sentinel_ai.domain.policy import vram_budget
from sentinel_ai.orchestrator.registry import ModelRegistry


class ResidentSet:
    def __init__(self, registry: ModelRegistry, total_mib: int, reserved_mib: int) -> None:
        self._registry = registry
        self._total_mib = total_mib
        self._reserved_mib = reserved_mib
        self._resident: set[str] = set()
        self._last_used_at: dict[str, float] = {}

    async def ensure(self, required: tuple[str, ...], now: float) -> None:
        """Apply plan_residency(): load required, evict what must go."""
        specs = {
            spec.model_key: vram_budget.ModelSpec(
                model_key=spec.model_key,
                vram_mib=spec.vram_mib,
                priority=spec.priority,
                idle_unload_seconds=spec.idle_unload_seconds,
            )
            for spec in self._registry.specs()
        }
        plan = vram_budget.plan_residency(
            specs=specs,
            currently_resident=tuple(self._resident),
            required=required,
            last_used_at=self._last_used_at,
            now=now,
            total_mib=self._total_mib,
            reserved_mib=self._reserved_mib,
        )
        for key in plan.unload:
            await self._registry.get(key).shutdown()
            self._resident.discard(key)
        for key in plan.load:
            runtime = self._registry.get(key)
            await runtime.initialize()
            await runtime.warmup()
            self._resident.add(key)
        for key in required:
            self._last_used_at[key] = now

    async def sweep_idle(self, now: float) -> None:
        await self.ensure((), now)

    def resident(self) -> frozenset[str]:
        return frozenset(self._resident)
```

- [ ] **Step 11: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_resident_set.py -v && mypy`
Expected: all PASS, mypy `Success`.

- [ ] **Step 12: Commit**

```bash
git add ai-engine/sentinel_ai/orchestrator/resident_set.py ai-engine/tests/orchestrator/test_resident_set.py
git commit -m "feat(orchestrator): add ResidentSet — executes plan_residency and the 600s VLM idle-unload"
```

- [ ] **Step 13: Write the S14 error-path tests against Task 6's `VlmScheduler`**

```python
# ai-engine/tests/orchestrator/test_scheduler.py — append to the file Task 6 created
async def test_a_vlm_timeout_still_publishes_an_event_flagged_unavailable() -> None:
    scheduler, _vlm, publisher = new_scheduler(vlm=FakeVisionLLM(error=TimeoutError("vlm timed out")))
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request())
        await scheduler.drain()
    finally:
        worker.cancel()

    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert event.description_unavailable is True
    assert "periodic_summary" in event.description
    assert event.threat.value == pytest.approx(0.5)


async def test_a_vlm_exception_other_than_timeout_also_still_publishes() -> None:
    scheduler, _vlm, publisher = new_scheduler(vlm=FakeVisionLLM(error=RuntimeError("cuda oom")))
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request())
        await scheduler.drain()
    finally:
        worker.cancel()

    assert publisher.events[0].description_unavailable is True


async def test_a_clip_finish_failure_still_publishes_with_clip_uri_none() -> None:
    writer = FakeClipWriter(finish_error=OSError("minio unreachable"))
    handle = await writer.open("cam-1", uuid4(), fps=10.0)
    scheduler, _vlm, publisher = new_scheduler()
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request(clip=handle))
        await scheduler.drain()
    finally:
        worker.cancel()

    assert len(publisher.events) == 1
    assert publisher.events[0].clip_uri is None
    assert publisher.events[0].description_unavailable is False  # the VLM call itself succeeded


async def test_admission_is_released_even_when_the_vlm_raises() -> None:
    """The `finally` around `admission.release()` (S14 step 7) must run on every path,
    not just the happy one — otherwise one failed escalation would permanently
    starve every later one of its slot."""
    admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
    scheduler = VlmScheduler(
        vlm=FakeVisionLLM(error=RuntimeError("boom")),
        publisher=FakePublisher(),
        admission=admission,
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )
    worker = asyncio.create_task(scheduler.run())
    try:
        scheduler.submit(a_request())
        await scheduler.drain()
        assert admission.in_flight == 0

        scheduler.submit(a_request())
        await scheduler.drain()  # would hang forever if the first call never released
        assert admission.in_flight == 0
    finally:
        worker.cancel()
```

- [ ] **Step 14: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_scheduler.py -v`
Expected: all PASS — Task 6's `_describe`/`_attach_clip` `except Exception` branches and the
`try`/`finally` around `admission.release()` already implement every path these tests exercise.

- [ ] **Step 15: Commit**

```bash
git add ai-engine/tests/orchestrator/test_scheduler.py
git commit -m "test(orchestrator): cover the S14 error paths — VLM timeout, clip failure, admission release

Spec §9's governing rule: an anomaly event is never lost to an infrastructure
failure. These are first-class tests, not an afterthought."
```

- [ ] **Step 16: Write the failing test for `EngineService`**

```python
# ai-engine/tests/orchestrator/test_service.py
from __future__ import annotations

import asyncio

import pytest

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.orchestrator.service import EngineService, UnknownCameraError
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from tests.fakes.io import FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeModelRuntime, FakeTracker, FakeVisionLLM

DETECTOR_SPEC = ModelSpec(
    model_key="yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None
)
VLM_SPEC = ModelSpec(
    model_key="qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0
)


def clock() -> float:
    return 0.0


def build_service(monkeypatch: pytest.MonkeyPatch) -> tuple[EngineService, FakePublisher]:
    from sentinel_ai.config import Settings
    from sentinel_ai.pipeline import runner as runner_module

    monkeypatch.setattr(runner_module, "get_settings", lambda: Settings(clip_postroll_seconds=0.0))

    registry = ModelRegistry()
    registry.register(DETECTOR_SPEC, FakeModelRuntime("yolo11s", vram_mib=900))
    registry.register(VLM_SPEC, FakeModelRuntime("qwen25vl3b", vram_mib=4400))
    resident_set = ResidentSet(registry, total_mib=8192, reserved_mib=2048)

    publisher = FakePublisher()
    scheduler = VlmScheduler(
        vlm=FakeVisionLLM(),
        publisher=publisher,
        admission=AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        maxsize=4,
        timeout_seconds=5.0,
        clock=clock,
    )
    runner = CameraRunner(
        camera_id="cam-1",
        camera_label="Front Door",
        source=FakeSource.constant("cam-1", count=1, fps=10.0),
        detector=FakeDetector(script=[()]),
        tracker=FakeTracker(),
        motion=MotionAnalyzer(),
        profile=CameraProfile(camera_id="cam-1"),
        scheduler=scheduler,
        clip_writer=None,
        preroll=PreRollBuffer(preroll_seconds=3.0),
        clock=clock,
    )
    service = EngineService(
        cameras={"cam-1": runner},
        registry=registry,
        resident_set=resident_set,
        scheduler=scheduler,
        required_model_keys=("yolo11s", "qwen25vl3b"),
        clock=clock,
    )
    return service, publisher


async def test_start_loads_the_required_models_and_runs_every_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _publisher = build_service(monkeypatch)
    await service.start()
    await asyncio.sleep(0)  # let the one-frame camera task run to completion
    await service.stop()

    assert service.health()["yolo11s"].state.value == "healthy"
    telemetry = service.telemetry("cam-1")
    assert telemetry.frames_seen == 1


async def test_cameras_lists_every_configured_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    await service.start()
    await service.stop()
    assert {t.camera_id for t in service.cameras()} == {"cam-1"}


async def test_telemetry_of_an_unknown_camera_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    with pytest.raises(UnknownCameraError, match="cam-404"):
        service.telemetry("cam-404")


async def test_describe_now_of_an_unknown_camera_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    service, _publisher = build_service(monkeypatch)
    with pytest.raises(UnknownCameraError, match="cam-404"):
        await service.describe_now("cam-404")
```

- [ ] **Step 17: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/orchestrator/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.orchestrator.service'`

- [ ] **Step 18: Implement `EngineService`**

```python
# ai-engine/sentinel_ai/orchestrator/service.py
"""The single entry point the API delegates to (spec §5.4, §5.7)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from uuid import UUID

from sentinel_ai.orchestrator.registry import ModelRegistry
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner, CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport


class UnknownCameraError(KeyError):
    def __init__(self, camera_id: str) -> None:
        super().__init__(f"unknown camera: {camera_id}")
        self.camera_id = camera_id


class EngineService:
    def __init__(
        self,
        cameras: Mapping[str, CameraRunner],
        registry: ModelRegistry,
        resident_set: ResidentSet,
        scheduler: VlmScheduler,
        required_model_keys: tuple[str, ...],
        clock: Callable[[], float],
    ) -> None:
        self._cameras = dict(cameras)
        self._registry = registry
        self._resident_set = resident_set
        self._scheduler = scheduler
        self._required_model_keys = required_model_keys
        self._clock = clock
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        await self._resident_set.ensure(self._required_model_keys, self._clock())
        self._tasks.append(asyncio.create_task(self._scheduler.run()))
        for runner in self._cameras.values():
            self._tasks.append(asyncio.create_task(runner.run()))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return tuple(runner.telemetry() for runner in self._cameras.values())

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        return self._get_runner(camera_id).telemetry()

    def health(self) -> dict[str, HealthReport]:
        return self._registry.health()

    async def describe_now(self, camera_id: str) -> UUID:
        return await self._get_runner(camera_id).describe_now()

    def _get_runner(self, camera_id: str) -> CameraRunner:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise UnknownCameraError(camera_id) from None
```

`start()` deliberately does not spin up a periodic idle-sweep loop calling
`resident_set.sweep_idle()` on a real timer: doing so with `asyncio.sleep(interval)` would be
untestable in CI without threading a fake clock all the way through the service, and every
path that actually needs `sweep_idle()`'s behaviour is already covered where it is cheap to
test precisely — `ResidentSet`'s own unit tests, with an injected `now`. A production
deployment (Task 14, alongside `RtspSource`, outside this task's CI-green scope) wraps a
running `EngineService` in a `while True: await asyncio.sleep(30); await
resident_set.sweep_idle(time.monotonic())` loop and wraps the VLM passed into `VlmScheduler`
so that each `describe()` call first does `await resident_set.ensure((vlm_key,), now)` —
refreshing the VLM's idle clock exactly when it is genuinely used, so it survives for as long
as escalations keep arriving and unloads 600s after the last one, per spec. Neither wrapper
changes S9's or S10's fixed shape; both are composition Task 14 adds around them.

- [ ] **Step 19: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 20: Commit**

```bash
git add ai-engine/sentinel_ai/orchestrator/service.py ai-engine/tests/orchestrator/test_service.py
git commit -m "feat(orchestrator): add EngineService — the single entry point for start/stop/telemetry/health/describe_now

UnknownCameraError is defined here; the API layer (Task 10) maps it to 404."
```


---

### Task 8: `InMemoryPublisher` + `RabbitMQPublisher` with disk spool

**Files:**
- Create: `sentinel_ai/adapters/publishers/__init__.py`
- Create: `sentinel_ai/adapters/publishers/inmemory.py`
- Create: `sentinel_ai/adapters/publishers/rabbitmq.py`
- Test: `tests/adapters/publishers/__init__.py` (new test package dir)
- Test: `tests/adapters/publishers/test_inmemory.py`
- Test: `tests/adapters/publishers/test_rabbitmq_spool.py`
- Test: `tests/adapters/publishers/test_rabbitmq_integration.py` (`@pytest.mark.integration`, excluded from CI)

**Interfaces:**
- Consumes: `EventPublisher` (`sentinel_ai/ports/event_publisher.py`, unchanged since Phase 1A —
  `async def publish(self, event: Event) -> None` / `async def close(self) -> None`);
  `encode_event`, `validate_payload` (`sentinel_ai/adapters/serialization/event_codec.py`);
  `Settings.rabbitmq_url`, `.rabbitmq_exchange`, `.event_spool_dir` (`sentinel_ai/config.py`,
  already present).
- Produces: `InMemoryPublisher`, `RabbitMQPublisher` exactly as fixed by S13 — later tasks
  (wiring in `main.py`, out of this plan's scope) construct whichever one `Settings.mode`
  selects.

`aio-pika` ships inline types (`py.typed`), so no `[[tool.mypy.overrides]]` entry is needed for
it — unlike `minio`/`av`, which are already listed because they don't.

- [ ] **Step 1: Write the failing test for `InMemoryPublisher`**

```python
# tests/adapters/publishers/test_inmemory.py
"""InMemoryPublisher — the real (not fake) EventPublisher used for CPU-only
runs and the keystone end-to-end test when no broker is configured (spec §5.6).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from jsonschema import ValidationError

from sentinel_ai.adapters.publishers.inmemory import InMemoryPublisher
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore


def _event(**overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.7),
        "description": "A person is running.",
        "suggested_action": "Review the clip.",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


async def test_publish_collects_events_in_order() -> None:
    publisher = InMemoryPublisher()
    first, second = _event(), _event()

    await publisher.publish(first)
    await publisher.publish(second)

    assert publisher.events == (first, second)


async def test_close_is_idempotent_and_does_not_clear_events() -> None:
    publisher = InMemoryPublisher()
    await publisher.publish(_event())
    await publisher.close()
    await publisher.close()
    assert len(publisher.events) == 1


async def test_publish_validates_before_accepting_the_event() -> None:
    """Every payload goes through `encode_event`, so a schema-invalid event
    (here, a threat score that bypassed `ThreatScore.from_value`'s range
    check by direct construction) fails loudly at publish time rather than
    reaching a consumer."""
    publisher = InMemoryPublisher()
    bypassed = _event(threat=ThreatScore(value=5.0, severity=Severity.CRITICAL))

    with pytest.raises(ValidationError):
        await publisher.publish(bypassed)

    assert publisher.events == ()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/publishers/test_inmemory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.publishers'`

- [ ] **Step 3: Implement `InMemoryPublisher`**

```python
# sentinel_ai/adapters/publishers/__init__.py
"""Publisher adapters — EventPublisher implementations (spec §5.6)."""
```

```python
# sentinel_ai/adapters/publishers/inmemory.py
"""In-memory EventPublisher — the real adapter for CPU-only runs (spec §5.6).

Not a test fake: it lives in `adapters/`, is what the end-to-end CPU trace
asserts against, and validates every event through `encode_event` exactly
like `RabbitMQPublisher` does, so a schema violation is caught the same way
in every environment the AI Engine runs in.
"""

from __future__ import annotations

from sentinel_ai.adapters.serialization.event_codec import encode_event
from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.event_publisher import EventPublisher


class InMemoryPublisher(EventPublisher):
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._closed = False

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    async def publish(self, event: Event) -> None:
        encode_event(event)  # validates; raises before the event is ever accepted
        self._events.append(event)

    async def close(self) -> None:
        self._closed = True
```

- [ ] **Step 4: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/publishers/test_inmemory.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the failing test for `RabbitMQPublisher`'s disk spool**

```python
# tests/adapters/publishers/test_rabbitmq_spool.py
"""RabbitMQPublisher's disk spool, exercised with no broker running (spec
§6, §9): everything that must survive a dead broker and a process restart.
A live-broker round trip is `@pytest.mark.integration`
(test_rabbitmq_integration.py) and excluded from CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import aio_pika
import pytest

from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore


class _FakeExchange:
    """Stands in for `aio_pika`'s exchange: records publishes, never touches
    a socket. Assigned directly to `_exchange` so `replay_spool()` and
    `publish()` are testable without a real `connect()`."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def publish(self, message: aio_pika.Message, routing_key: str) -> None:
        body = json.loads(message.body.decode("utf-8"))
        self.published.append((routing_key, body))


def _event(camera_id: str = "cam-1", **overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": camera_id,
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.7),
        "description": "A person is running.",
        "suggested_action": "Review the clip.",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


async def test_publish_without_a_connection_spools_to_disk(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )

    await publisher.publish(_event())

    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_two_events_in_the_same_instant_replay_in_call_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same-second (here, same-nanosecond) collision: the filename's
    sequence tie-breaker, not wall-clock resolution, must decide order."""
    import sentinel_ai.adapters.publishers.rabbitmq as rabbitmq_module

    monkeypatch.setattr(rabbitmq_module.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    first, second = _event(occurred_at=1.0), _event(occurred_at=2.0)

    await publisher.publish(first)
    await publisher.publish(second)

    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]
    await publisher.replay_spool()

    replayed_ids = [body["event_id"] for _routing_key, body in fake_exchange.published]
    assert replayed_ids == [str(first.event_id), str(second.event_id)]
    assert list(tmp_path.glob("*.json")) == [], "successfully replayed spool files are removed"


async def test_a_corrupt_spool_file_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    good = tmp_path / "10000000000000000000-00000001-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json"
    good.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "event_id": str(uuid4()),
                "camera_id": "cam-1",
                "occurred_at": 1.0,
                "reason": "speed_anomaly",
                "threat_score": 0.7,
                "severity": "high",
                "description": "d",
                "suggested_action": "a",
                "labels": [],
                "track_ids": [],
                "keyframe_uri": None,
                "clip_uri": None,
                "description_unavailable": False,
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    corrupt = tmp_path / "20000000000000000000-00000001-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.json"
    corrupt.write_text("{not valid json", encoding="utf-8")

    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]

    await publisher.replay_spool()  # must not raise

    assert len(fake_exchange.published) == 1, "the valid file replayed"
    assert not good.exists(), "a replayed file is deleted"
    assert corrupt.with_suffix(".json.corrupt").exists(), (
        "the corrupt file is set aside, not retried forever"
    )


async def test_replay_before_connect_raises_a_clear_error(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="connect"):
        await publisher.replay_spool()


async def test_routing_key_carries_camera_and_reason(tmp_path: Path) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    fake_exchange = _FakeExchange()
    publisher._exchange = fake_exchange  # type: ignore[assignment]

    await publisher.publish(_event(camera_id="cam-7", reason=EscalationReason.DWELL_EXCEEDED))

    routing_key, _body = fake_exchange.published[0]
    assert routing_key == "anomaly.cam-7.dwell_exceeded"


async def test_a_broker_failure_mid_replay_leaves_remaining_files_for_next_time(
    tmp_path: Path,
) -> None:
    publisher = RabbitMQPublisher(
        url="amqp://unused/", exchange="sentinel.events", spool_dir=tmp_path
    )
    await publisher.publish(_event())
    await publisher.publish(_event())

    class _FlakyExchange(_FakeExchange):
        async def publish(self, message: aio_pika.Message, routing_key: str) -> None:
            raise ConnectionError("broker dropped again")

    publisher._exchange = _FlakyExchange()  # type: ignore[assignment]
    await publisher.replay_spool()  # must not raise

    assert len(list(tmp_path.glob("*.json"))) == 2, "nothing was deleted on a failed replay"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/publishers/test_rabbitmq_spool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.publishers.rabbitmq'`

- [ ] **Step 7: Implement `RabbitMQPublisher`**

```python
# sentinel_ai/adapters/publishers/rabbitmq.py
"""RabbitMQ EventPublisher with a disk-backed spool (spec §5.6, §9).

Governing rule (spec §9): an anomaly event is never lost to an
infrastructure failure. When the broker is unreachable, the already-validated
payload is written to `spool_dir` and replayed once `connect()` succeeds
again. `connect()` and `replay_spool()` are separate, explicit calls (S13) —
this publisher never retries on its own; the caller (a reconnect loop with
its own backoff policy) decides when each runs.

Spool file naming: `<ts_ns:020d>-<seq:08d>-<event_id.hex>.json`. The
zero-padded nanosecond wall-clock timestamp sorts chronologically across
process restarts — wall time keeps advancing across a crash, unlike the
domain's injected monotonic clock, which resets to 0 on the next process. The
sequence number breaks ties when two events land in the same nanosecond
(a real possibility: two cameras can both escalate inside one scheduler
tick). Both are baked into the filename, so `sorted(glob(...))` alone gives
replay order — no separate index file to keep consistent with the spool
directory.

Partial-write detection is write-then-rename, not a checksum: the file is
written to a `.json.tmp` sibling and moved into place with `Path.replace`,
which is an atomic rename on the same filesystem. A `.json` file therefore
either doesn't exist yet or is complete — `replay_spool`'s `*.json` glob can
never observe a half-written one. A checksum would also work, but it requires
computing and storing a hash for every payload just to protect against a
window that write-then-rename closes for free.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractRobustConnection
from aio_pika.exceptions import AMQPError
from jsonschema import ValidationError

from sentinel_ai.adapters.serialization.event_codec import encode_event, validate_payload
from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.event_publisher import EventPublisher

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (AMQPError, OSError, ConnectionError)


def _routing_key(camera_id: str, reason: str) -> str:
    """`anomaly.<camera_id>.<reason>` on the topic exchange — lets a consumer
    bind on one camera, one reason across all cameras, or everything
    (`anomaly.#`) without the publisher knowing which."""
    return f"anomaly.{camera_id}.{reason}"


class RabbitMQPublisher(EventPublisher):
    def __init__(self, url: str, exchange: str, spool_dir: Path) -> None:
        self._url = url
        self._exchange_name = exchange
        self._spool_dir = spool_dir
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._spool_seq = 0

    async def connect(self) -> None:
        """Open the connection, channel and durable topic exchange.

        Raises on failure; the caller decides whether and when to retry.
        Durable exchange + persistent messages (in `_publish_payload`) mean a
        broker restart drops neither the topology nor an in-flight message.
        """
        connection = await aio_pika.connect_robust(self._url)
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            self._exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        self._connection = connection
        self._channel = channel
        self._exchange = exchange

    async def publish(self, event: Event) -> None:
        payload = encode_event(event)  # validates against the committed schema
        routing_key = _routing_key(event.camera_id, event.reason.value)
        if self._exchange is None:
            self._spool(event.event_id, payload)
            return
        try:
            await self._publish_payload(routing_key, payload)
        except _TRANSPORT_ERRORS:
            logger.warning("broker unreachable, spooling event %s", event.event_id)
            self._spool(event.event_id, payload)

    async def replay_spool(self) -> None:
        """Publish every spooled payload, oldest first, deleting each as it
        succeeds. Stops at the first broker failure so the remaining files
        wait for the next call, and sets aside (rather than crashes on) a
        corrupt file so it is not retried forever.
        """
        if self._exchange is None:
            raise RuntimeError("replay_spool() called before connect() succeeded")
        for path in sorted(self._spool_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                validate_payload(payload)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError, OSError):
                logger.warning("skipping corrupt spool file %s", path.name, exc_info=True)
                path.rename(path.with_suffix(".json.corrupt"))
                continue

            routing_key = _routing_key(str(payload["camera_id"]), str(payload["reason"]))
            try:
                await self._publish_payload(routing_key, payload)
            except _TRANSPORT_ERRORS:
                logger.warning("broker dropped mid-replay, stopping at %s", path.name)
                return
            path.unlink(missing_ok=True)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def _publish_payload(self, routing_key: str, payload: Mapping[str, object]) -> None:
        assert self._exchange is not None
        message = aio_pika.Message(
            body=json.dumps(payload).encode("utf-8"),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            content_type="application/json",
        )
        await self._exchange.publish(message, routing_key=routing_key)

    def _spool(self, event_id: UUID, payload: Mapping[str, object]) -> None:
        self._spool_seq += 1
        base = f"{time.time_ns():020d}-{self._spool_seq:08d}-{event_id.hex}"
        final_path = self._spool_dir / f"{base}.json"
        tmp_path = self._spool_dir / f"{base}.json.tmp"
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(final_path)  # atomic rename on the same filesystem
```

- [ ] **Step 8: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/publishers -v -m "not integration"`
Expected: all pass (`test_rabbitmq_integration.py` collected but skipped/deselected by the marker).

- [ ] **Step 9: Add the excluded live-broker integration test**

```python
# tests/adapters/publishers/test_rabbitmq_integration.py
"""Live-broker round trip: requires `docker compose -f deploy/compose/docker-compose.core.yml
up -d rabbitmq` (spec §2, Task 2). Excluded from CI by `-m "not integration"`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import uuid4

import aio_pika
import pytest

from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.config import get_settings
from sentinel_ai.domain.entities import EscalationReason, Event, ThreatScore

pytestmark = pytest.mark.integration


def _event() -> Event:
    return Event(
        event_id=uuid4(),
        camera_id="cam-integration",
        occurred_at=1_700_000_000.0,
        reason=EscalationReason.SPEED_ANOMALY,
        threat=ThreatScore.from_value(0.7),
        description="A person is running.",
        suggested_action="Review the clip.",
    )


async def test_publish_reaches_a_bound_queue(tmp_path: Path) -> None:
    settings = get_settings()
    publisher = RabbitMQPublisher(
        url=settings.rabbitmq_url, exchange=settings.rabbitmq_exchange, spool_dir=tmp_path
    )
    await publisher.connect()
    try:
        assert publisher._channel is not None
        queue = await publisher._channel.declare_queue(exclusive=True)
        await queue.bind(publisher._exchange, routing_key="anomaly.#")

        event = _event()
        await publisher.publish(event)

        message = await asyncio.wait_for(queue.get(), timeout=5.0)
        body = json.loads(message.body.decode("utf-8"))
        assert body["event_id"] == str(event.event_id)
        await message.ack()
    finally:
        await publisher.close()


async def test_spooled_events_replay_once_the_broker_is_reachable(tmp_path: Path) -> None:
    settings = get_settings()

    # An unreachable port forces every publish to spool, without touching the
    # real broker's lifecycle.
    down_publisher = RabbitMQPublisher(
        url="amqp://sentinel:sentinel@localhost:1/",
        exchange=settings.rabbitmq_exchange,
        spool_dir=tmp_path,
    )
    event = _event()
    await down_publisher.publish(event)
    assert len(list(tmp_path.glob("*.json"))) == 1

    up_publisher = RabbitMQPublisher(
        url=settings.rabbitmq_url, exchange=settings.rabbitmq_exchange, spool_dir=tmp_path
    )
    await up_publisher.connect()
    try:
        assert up_publisher._channel is not None
        queue = await up_publisher._channel.declare_queue(exclusive=True)
        await queue.bind(up_publisher._exchange, routing_key="anomaly.#")

        await up_publisher.replay_spool()

        message = await asyncio.wait_for(queue.get(), timeout=5.0)
        body = json.loads(message.body.decode("utf-8"))
        assert body["event_id"] == str(event.event_id)
        await message.ack()
    finally:
        await up_publisher.close()
    assert list(tmp_path.glob("*.json")) == []
```

- [ ] **Step 10: Verify the full suite once more**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 11: Commit**

```bash
git add sentinel_ai/adapters/publishers tests/adapters/publishers
git commit -m "$(cat <<'EOF'
feat(adapters): add InMemoryPublisher and disk-spooled RabbitMQPublisher

An anomaly event must never be lost to a dead broker (spec §9). The
RabbitMQ adapter spools encoded, already-schema-validated payloads to disk
with a write-then-rename filename that sorts chronologically across
restarts, and replays them once connect() succeeds again — connect() and
replay_spool() stay separate so the caller owns retry/backoff.
EOF
)"
```

---

### Task 9: `MinioClipWriter`

**Files:**
- Create: `sentinel_ai/adapters/storage/__init__.py`
- Create: `sentinel_ai/adapters/storage/minio_clips.py`
- Test: `tests/adapters/storage/__init__.py` (new test package dir)
- Test: `tests/adapters/storage/test_minio_clips.py`

**Interfaces:**
- Consumes: `ClipWriter` / `ClipHandle` (S4, `sentinel_ai/ports/clip_writer.py`); `EncodedPacket`
  (S2, `sentinel_ai/ports/frame_source.py`); `Settings.minio_endpoint/.minio_access_key/
  .minio_secret_key/.minio_bucket/.minio_secure` (already in `sentinel_ai/config.py`);
  `ai-engine/tests/assets/synthetic_clip.mp4` (Task 4's committed ffmpeg-generated fixture — a
  few hundred KB, H.264-in-MP4, moving shapes. If Task 4 named the file differently, update the
  `FIXTURE` constant at the top of the test file; nothing else changes).
- Produces: `MinioClipWriter` exactly as fixed by S13.

- [ ] **Step 1: Write the failing test — the remux, with no MinIO involved**

```python
# tests/adapters/storage/test_minio_clips.py
"""Remux + MinIO clip writer tests (spec §5.5). The remux itself is pure PyAV
and is tested without any MinIO server; the upload path is
`@pytest.mark.integration`, requiring `docker compose ... up -d minio`.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import av
import pytest

from sentinel_ai.adapters.storage.minio_clips import MinioClipWriter, _RemuxSession
from sentinel_ai.ports.frame_source import EncodedPacket

FIXTURE = Path(__file__).resolve().parents[2] / "assets" / "synthetic_clip.mp4"


def _annexb_packets_from_fixture(camera_id: str) -> list[EncodedPacket]:
    """Read the committed fixture and re-wrap it as Annex-B `EncodedPacket`s.

    `FileSource` (Task 4) demuxes this same fixture for the pipeline; this
    independently re-derives what `RtspSource` hands a `ClipWriter` in
    production. RTP H.264 (RFC 6184) carries SPS/PPS in-band, unlike the MP4
    container's out-of-band `avcC` record — so this converts the fixture's
    AVCC packets to Annex-B with the `h264_mp4toannexb` bitstream filter,
    which is exactly the ffmpeg-side transform that already happens for free
    on an RTSP source. `_RemuxSession` below depends on packets looking like
    this; see its docstring for why.
    """
    container = av.open(str(FIXTURE))
    in_stream = container.streams.video[0]
    bsf = av.BitStreamFilterContext("h264_mp4toannexb", in_stream)
    packets: list[EncodedPacket] = []
    for packet in container.demux(in_stream):
        if packet.dts is None:
            continue
        for filtered in bsf.filter(packet):
            packets.append(
                EncodedPacket(
                    camera_id=camera_id,
                    data=bytes(filtered),
                    pts=float(filtered.pts * filtered.time_base),
                    is_keyframe=bool(filtered.is_keyframe),
                    codec=in_stream.codec_context.name,
                )
            )
    container.close()
    assert packets, "fixture produced no packets — is FIXTURE the right file?"
    assert packets[0].is_keyframe, "fixture must start on a keyframe for this test to be valid"
    return packets


class TestRemuxSession:
    def test_a_written_clip_is_a_valid_mp4_starting_on_a_keyframe(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"

        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()

        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        out_packets = [p for p in out.demux(out_stream) if p.dts is not None]
        out.close()

        assert out_packets, "no packets were muxed into the clip"
        assert out_packets[0].is_keyframe

    def test_clip_duration_matches_the_appended_packets(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"

        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()

        expected_span = packets[-1].pts - packets[0].pts
        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        actual_span = float(out_stream.duration * out_stream.time_base)
        out.close()

        assert actual_span == pytest.approx(expected_span, abs=1.0 / 25.0)

    def test_abort_without_finish_deletes_the_temp_file(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        session.append(packets[0])

        session.abort()
        assert not temp_path.exists()

        session.abort()  # idempotent

    def test_abort_after_finish_never_raises(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()
        assert temp_path.exists(), "finish() leaves the local file for the caller to upload"

        session.abort()  # after finish(): must be a safe no-op, not raise
        session.abort()  # and idempotent on top of that


class TestMinioClipHandle:
    async def test_finish_uploads_then_deletes_the_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)
        uploaded: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            writer._client,
            "fput_object",
            lambda bucket, name, path: uploaded.append((bucket, name, path)),
        )

        camera_id, event_id = "cam-1", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        for packet in _annexb_packets_from_fixture(camera_id):
            await handle.append(packet)
        uri = await handle.finish()

        assert uri == f"s3://sentinel-clips/{camera_id}/{event_id}.mp4"
        assert len(uploaded) == 1
        bucket, object_name, local_path = uploaded[0]
        assert bucket == "sentinel-clips"
        assert object_name == f"{camera_id}/{event_id}.mp4"
        assert not Path(local_path).exists(), "the temp file is deleted after upload"

    async def test_abort_deletes_the_temp_file_and_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)

        handle = await writer.open("cam-1", uuid4(), fps=25.0)
        packets = _annexb_packets_from_fixture("cam-1")
        await handle.append(packets[0])

        await handle.abort()
        await handle.abort()  # idempotent, must not raise
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/storage/test_minio_clips.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.storage'`

- [ ] **Step 3: Implement**

```python
# sentinel_ai/adapters/storage/__init__.py
"""Storage adapters — ClipWriter implementations (spec §5.5)."""
```

```python
# sentinel_ai/adapters/storage/minio_clips.py
"""MinIO evidence clip writer (spec §5.5). A remux, never a re-encode: the
packets that arrive are already H.264, the pre-roll buffer already holds
them encoded, and the output is an MP4 container around the same bytes —
so there is no decode and no encode anywhere in this path, and the evidence
stays bit-identical to what the camera sent.

The crux this module resolves: `ClipHandle.open()` receives only
`camera_id`, `event_id` and `fps` (S4); `append()` receives an
`EncodedPacket` — bytes, a pts, a keyframe flag and a codec name (S2). The
codec name says *what* to parse, but nothing carries width, height, or the
SPS/PPS an MP4 muxer needs to write a valid `stsd` box, because the port is
deliberately codec-agnostic beyond that one string. Those are recovered
from the bytes themselves: H.264 over RTSP/RTP (RFC 6184, the deployment
source in scope) carries SPS/PPS in-band on every keyframe, unlike an MP4
file's out-of-band `avcC` record. Feeding the raw bytes through
`av.CodecContext.create(codec, "r")` — an elementary-stream parser, not a
decoder — reassembles NAL units into access units and populates `.width`,
`.height` and `.extradata` the moment it has seen a keyframe's parameter
sets. That is where the output stream's configuration comes from: nowhere
else, because nowhere else has it.

`SUPPORTED_CODECS` gates this explicitly. A codec outside it raises
`UnsupportedCodecError` on the first packet rather than parsing forever and
finalising an empty MP4 — a clip writer that fails silently is worse than
one that fails, because the operator only discovers it when they go looking
for evidence that was never written.
"""

from __future__ import annotations

import asyncio
import logging
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import av
from minio import Minio

from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.frame_source import EncodedPacket

logger = logging.getLogger(__name__)

_OUTPUT_TIME_BASE = Fraction(1, 90_000)
"""A conventional high-resolution MP4 video timebase, independent of fps —
gives sub-frame pts precision regardless of the source framerate."""

SUPPORTED_CODECS: frozenset[str] = frozenset({"h264", "hevc"})
"""Codecs that can be remuxed straight into MP4 and that carry parameter
sets in-band. Everything mediamtx emits and every RTSP camera in scope is
one of these."""


class UnsupportedCodecError(RuntimeError):
    """Raised when a source delivers a codec the clip writer cannot remux."""


class _RemuxSession:
    """Copies `EncodedPacket` bytes into a local MP4 file. No MinIO, no
    network — isolated from `MinioClipHandle` so the remux, the part with
    real logic, is unit-testable without a running MinIO server.
    """

    def __init__(self, temp_path: Path, fps: float) -> None:
        self._temp_path = temp_path
        self._fps = fps
        self._container = av.open(str(temp_path), mode="w")
        self._codec: str | None = None
        self._parser: av.CodecContext | None = None
        self._out_stream: av.video.stream.VideoStream | None = None
        self._start_pts: float | None = None
        self._closed = False

    def _ensure_parser(self, codec: str) -> av.CodecContext:
        """Bind the session to the first packet's codec.

        A mid-clip codec change cannot be remuxed into one MP4 track, so it is
        rejected rather than silently producing a corrupt file.
        """
        if self._parser is None:
            if codec not in SUPPORTED_CODECS:
                raise UnsupportedCodecError(
                    f"cannot remux codec {codec!r}; supported: {sorted(SUPPORTED_CODECS)}"
                )
            self._codec = codec
            self._parser = av.CodecContext.create(codec, "r")
        elif codec != self._codec:
            raise UnsupportedCodecError(
                f"codec changed mid-clip: {self._codec!r} -> {codec!r}"
            )
        return self._parser

    @property
    def temp_path(self) -> Path:
        return self._temp_path

    def append(self, packet: EncodedPacket) -> None:
        if self._closed:
            return
        if self._out_stream is None and not packet.is_keyframe:
            # Cannot start a clip mid-GOP; PreRollBuffer.flush() guarantees
            # the first packet handed to a real ClipHandle is a keyframe, so
            # this only guards a caller that violates that contract.
            logger.debug("dropping non-keyframe packet before the first keyframe")
            return

        parser = self._ensure_parser(packet.codec)

        for parsed in parser.parse(packet.data):
            if self._out_stream is None:
                if not (parser.extradata and parser.width and parser.height):
                    continue  # SPS/PPS not fully seen yet
                self._out_stream = self._container.add_stream(
                    packet.codec, rate=max(1, round(self._fps))
                )
                self._out_stream.codec_context.width = parser.width
                self._out_stream.codec_context.height = parser.height
                self._out_stream.codec_context.extradata = parser.extradata
                self._out_stream.codec_context.pix_fmt = "yuv420p"
                self._out_stream.time_base = _OUTPUT_TIME_BASE
                self._start_pts = packet.pts

            assert self._out_stream is not None
            assert self._start_pts is not None
            # PTS/DTS rebasing: the clip's own clock starts at 0 regardless
            # of where the camera's monotonic pts happened to be.
            ticks = round((packet.pts - self._start_pts) / float(_OUTPUT_TIME_BASE))
            parsed.pts = ticks
            parsed.dts = ticks  # no B-frames on the low-latency IPPP profiles in scope
            parsed.time_base = _OUTPUT_TIME_BASE
            parsed.stream = self._out_stream
            self._container.mux(parsed)

    def finish(self) -> Path:
        if self._closed:
            return self._temp_path
        for parsed in self._parser.parse(None):  # flush the parser's trailing access unit
            if self._out_stream is None:
                continue
            parsed.stream = self._out_stream
            self._container.mux(parsed)
        self._container.close()
        self._closed = True
        return self._temp_path

    def abort(self) -> None:
        """Discard a partial clip. Must not raise — this runs on shutdown paths."""
        if self._closed:
            return
        self._closed = True
        try:
            self._container.close()
        except Exception:  # noqa: BLE001 - best-effort; abort() must never raise
            logger.warning("error closing partial clip %s", self._temp_path, exc_info=True)
        try:
            self._temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to remove temp clip file %s", self._temp_path, exc_info=True)


class MinioClipHandle(ClipHandle):
    def __init__(
        self, session: _RemuxSession, client: Minio, bucket: str, camera_id: str, event_id: UUID
    ) -> None:
        self._session = session
        self._client = client
        self._bucket = bucket
        self._camera_id = camera_id
        self._event_id = event_id
        self._done = False

    async def append(self, packet: EncodedPacket) -> None:
        await asyncio.to_thread(self._session.append, packet)

    async def finish(self) -> str:
        if self._done:
            raise RuntimeError("clip handle already finished or aborted")
        self._done = True
        temp_path = await asyncio.to_thread(self._session.finish)
        object_name = f"{self._camera_id}/{self._event_id}.mp4"
        try:
            await asyncio.to_thread(
                self._client.fput_object, self._bucket, object_name, str(temp_path)
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return f"s3://{self._bucket}/{object_name}"

    async def abort(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            await asyncio.to_thread(self._session.abort)
        except Exception:  # noqa: BLE001 - best-effort; abort() must never raise
            logger.warning(
                "error aborting clip %s/%s", self._camera_id, self._event_id, exc_info=True
            )


class MinioClipWriter(ClipWriter):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        temp_dir: Path,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._temp_dir = temp_dir
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._bucket_ready = False

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle:
        if not self._bucket_ready:
            await asyncio.to_thread(self._ensure_bucket)
            self._bucket_ready = True
        temp_path = self._temp_dir / f"{camera_id}-{event_id}.mp4"
        session = await asyncio.to_thread(_RemuxSession, temp_path, fps)
        return MinioClipHandle(session, self._client, self._bucket, camera_id, event_id)

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
```

- [ ] **Step 4: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/storage -v -m "not integration"`
Expected: all pass.

- [ ] **Step 5: Add the excluded live-MinIO integration test**

```python
# tests/adapters/storage/test_minio_clips.py  (append to the file above)


@pytest.mark.integration
class TestMinioClipWriterIntegration:
    """Requires `docker compose -f deploy/compose/docker-compose.core.yml up -d minio`."""

    async def test_finish_returns_a_uri_the_object_exists_at(self, tmp_path: Path) -> None:
        from sentinel_ai.config import get_settings

        settings = get_settings()
        writer = MinioClipWriter(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            temp_dir=tmp_path,
        )
        camera_id, event_id = "cam-integration", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        for packet in _annexb_packets_from_fixture(camera_id):
            await handle.append(packet)
        uri = await handle.finish()

        assert uri == f"s3://{settings.minio_bucket}/{camera_id}/{event_id}.mp4"
        stat = await asyncio.to_thread(
            writer._client.stat_object, settings.minio_bucket, f"{camera_id}/{event_id}.mp4"
        )
        assert stat.size > 0
```

(Add `import asyncio` to the top of the test file alongside the existing imports.)

- [ ] **Step 6: Verify the full suite once more**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 7: Commit**

```bash
git add sentinel_ai/adapters/storage tests/adapters/storage
git commit -m "$(cat <<'EOF'
feat(adapters): add MinioClipWriter — packet-copy remux to MP4

Clips are a remux, not a re-encode: packets copy through untouched and the
output stream's codec parameters are recovered from the in-band H.264
SPS/PPS via CodecContext.parse(), since EncodedPacket deliberately carries
no codec identity. Keeps evidence bit-identical to what the camera sent and
peak RAM at one packet, matching the ~7 GB-free budget in spec §2.1.
EOF
)"
```

---

### Task 10: FastAPI surface

**Files:**
- Create: `sentinel_ai/api/__init__.py`
- Create: `sentinel_ai/api/app.py`
- Create: `sentinel_ai/api/routes.py`
- Create: `sentinel_ai/api/schemas.py`
- Modify: `pyproject.toml` (add `httpx` to the `dev` extra — `fastapi.testclient.TestClient` needs
  it and nothing currently in `dev` or `runtime` pulls it in)
- Test: `tests/api/__init__.py` (new test package dir)
- Test: `tests/api/test_routes.py`

**Interfaces:**
- Consumes: `EngineService` (S12, `orchestrator/service.py`) — `start()`, `stop()`, `cameras()
  -> tuple[CameraTelemetry, ...]`, `telemetry(camera_id) -> CameraTelemetry`, `health() ->
  dict[str, HealthReport]`, `describe_now(camera_id) -> UUID`; `UnknownCameraError` (also
  `orchestrator/service.py` per S12 — assumed to take the camera id and produce a message
  containing it, matching the codebase's `ValueError`-with-offending-field convention);
  `CameraTelemetry` (S11, `pipeline/runner.py`); `HealthReport`, `LifecycleState`
  (`ports/model_runtime.py`).
- Produces: `create_app`, `router`, `EngineServiceProtocol` (`api/app.py` / `api/routes.py`) —
  `main.py` (out of this plan) constructs the real `EngineService` and calls `create_app(service)`.

No WebSocket route in this phase — the Go backend owns the WS hub in Phase 1C (spec §5.7); this
task adds none, and nobody should add one here.

`EngineService` is a concrete class, not a port, so the app factory is typed against a
structural `Protocol` capturing exactly the six methods it calls — this is what lets the test
suite inject a lightweight fake without constructing a real `EngineService` (which pulls in the
registry, resident set, scheduler and admission gate) while `main.py` passes the real one
unchanged, satisfying the protocol by shape alone.

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_routes.py
"""FastAPI surface tests (spec §5.7): thin delegation to EngineService, so
every test runs against a fake service and asserts status codes and response
shapes only — no business logic lives here to test.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from sentinel_ai.api.app import create_app
from sentinel_ai.orchestrator.service import UnknownCameraError
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport, LifecycleState


class _FakeEngineService:
    def __init__(
        self,
        *,
        cameras: tuple[CameraTelemetry, ...] = (),
        health: dict[str, HealthReport] | None = None,
        describe_result: UUID | None = None,
    ) -> None:
        self._cameras = {t.camera_id: t for t in cameras}
        self._health = health or {}
        self._describe_result = describe_result or uuid4()
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def cameras(self) -> tuple[CameraTelemetry, ...]:
        return tuple(self._cameras.values())

    def telemetry(self, camera_id: str) -> CameraTelemetry:
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        return self._cameras[camera_id]

    def health(self) -> dict[str, HealthReport]:
        return self._health

    async def describe_now(self, camera_id: str) -> UUID:
        if camera_id not in self._cameras:
            raise UnknownCameraError(camera_id)
        return self._describe_result


def _telemetry(camera_id: str = "cam-1") -> CameraTelemetry:
    return CameraTelemetry(
        camera_id=camera_id,
        frames_seen=100,
        frames_dropped=2,
        detections_run=98,
        escalations=3,
        escalations_dropped=0,
        discontinuities=1,
        last_frame_at=12.5,
        last_escalation_at=10.0,
    )


def test_health_returns_per_model_lifecycle_state() -> None:
    service = _FakeEngineService(
        health={"yolo11": HealthReport(state=LifecycleState.LOADED, detail="", vram_mib=1400)}
    )
    with TestClient(create_app(service)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models"] == [{"key": "yolo11", "state": "loaded", "detail": "", "vram_mib": 1400}]


def test_list_cameras_returns_configured_cameras() -> None:
    service = _FakeEngineService(cameras=(_telemetry("cam-1"), _telemetry("cam-2")))
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras")

    assert response.status_code == 200
    ids = [camera["camera_id"] for camera in response.json()["cameras"]]
    assert ids == ["cam-1", "cam-2"]


def test_camera_telemetry_returns_the_expected_shape() -> None:
    service = _FakeEngineService(cameras=(_telemetry("cam-1"),))
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras/cam-1/telemetry")

    assert response.status_code == 200
    assert response.json() == {
        "camera_id": "cam-1",
        "frames_seen": 100,
        "frames_dropped": 2,
        "detections_run": 98,
        "escalations": 3,
        "escalations_dropped": 0,
        "discontinuities": 1,
        "last_frame_at": 12.5,
        "last_escalation_at": 10.0,
    }


def test_camera_telemetry_for_an_unknown_camera_is_404() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)) as client:
        response = client.get("/cameras/does-not-exist/telemetry")

    assert response.status_code == 404


def test_describe_now_returns_the_event_id() -> None:
    event_id = uuid4()
    service = _FakeEngineService(cameras=(_telemetry("cam-1"),), describe_result=event_id)
    with TestClient(create_app(service)) as client:
        response = client.post("/cameras/cam-1/describe")

    assert response.status_code == 200
    assert response.json() == {"event_id": str(event_id)}


def test_describe_now_for_an_unknown_camera_is_404() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)) as client:
        response = client.post("/cameras/does-not-exist/describe")

    assert response.status_code == 404


def test_lifespan_starts_and_stops_the_service() -> None:
    service = _FakeEngineService()
    with TestClient(create_app(service)):
        assert service.started is True
    assert service.stopped is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/api/test_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.api'`

- [ ] **Step 3: Implement**

```python
# sentinel_ai/api/__init__.py
"""FastAPI surface — thin delegation to orchestrator.service.EngineService (spec §5.7).

No business logic lives here; every endpoint validates the path/body and
calls straight into EngineService. No WebSocket route in this phase — the Go
backend owns the WS hub starting in Phase 1C.
"""
```

```python
# sentinel_ai/api/schemas.py
"""Pydantic wire models. Domain dataclasses never cross the API boundary
directly — these mirror the relevant fields so the wire shape and the
orchestrator's internal shape can change independently."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel

from sentinel_ai.pipeline.runner import CameraTelemetry


class ModelHealth(BaseModel):
    key: str
    state: str
    detail: str
    vram_mib: int


class HealthResponse(BaseModel):
    status: str
    models: list[ModelHealth]


class CameraStatus(BaseModel):
    camera_id: str
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    last_frame_at: float | None
    last_escalation_at: float | None

    @classmethod
    def from_telemetry(cls, telemetry: CameraTelemetry) -> CameraStatus:
        return cls(
            camera_id=telemetry.camera_id,
            frames_seen=telemetry.frames_seen,
            frames_dropped=telemetry.frames_dropped,
            detections_run=telemetry.detections_run,
            escalations=telemetry.escalations,
            escalations_dropped=telemetry.escalations_dropped,
            discontinuities=telemetry.discontinuities,
            last_frame_at=telemetry.last_frame_at,
            last_escalation_at=telemetry.last_escalation_at,
        )


class CamerasResponse(BaseModel):
    cameras: list[CameraStatus]


class DescribeResponse(BaseModel):
    event_id: UUID
```

```python
# sentinel_ai/api/routes.py
"""Endpoints (spec §5.7). Every handler is a one-line delegation to
EngineServiceProtocol; `UnknownCameraError` is the only exception translated
here, to HTTP 404."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sentinel_ai.api.schemas import (
    CamerasResponse,
    CameraStatus,
    DescribeResponse,
    HealthResponse,
    ModelHealth,
)
from sentinel_ai.orchestrator.service import UnknownCameraError
from sentinel_ai.pipeline.runner import CameraTelemetry
from sentinel_ai.ports.model_runtime import HealthReport


class EngineServiceProtocol(Protocol):
    """The exact EngineService (S12) surface this API uses, expressed
    structurally so tests can inject a fake without constructing the real
    orchestrator stack. `orchestrator.service.EngineService` satisfies this
    by shape; nothing here changes S12."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def cameras(self) -> tuple[CameraTelemetry, ...]: ...
    def telemetry(self, camera_id: str) -> CameraTelemetry: ...
    def health(self) -> dict[str, HealthReport]: ...
    async def describe_now(self, camera_id: str) -> UUID: ...


def get_service(request: Request) -> EngineServiceProtocol:
    service: EngineServiceProtocol = request.app.state.service
    return service


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def get_health(
    service: EngineServiceProtocol = Depends(get_service),
) -> HealthResponse:
    reports = service.health()
    return HealthResponse(
        status="ok",
        models=[
            ModelHealth(key=key, state=report.state.value, detail=report.detail,
                        vram_mib=report.vram_mib)
            for key, report in reports.items()
        ],
    )


@router.get("/cameras", response_model=CamerasResponse)
async def list_cameras(
    service: EngineServiceProtocol = Depends(get_service),
) -> CamerasResponse:
    return CamerasResponse(cameras=[CameraStatus.from_telemetry(t) for t in service.cameras()])


@router.get("/cameras/{camera_id}/telemetry", response_model=CameraStatus)
async def get_camera_telemetry(
    camera_id: str, service: EngineServiceProtocol = Depends(get_service)
) -> CameraStatus:
    try:
        telemetry = service.telemetry(camera_id)
    except UnknownCameraError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return CameraStatus.from_telemetry(telemetry)


@router.post("/cameras/{camera_id}/describe", response_model=DescribeResponse)
async def describe_camera_now(
    camera_id: str, service: EngineServiceProtocol = Depends(get_service)
) -> DescribeResponse:
    try:
        event_id = await service.describe_now(camera_id)
    except UnknownCameraError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DescribeResponse(event_id=event_id)
```

```python
# sentinel_ai/api/app.py
"""App factory + lifespan (spec §5.7). The service is injected, never
constructed here, so tests never build a real EngineService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from sentinel_ai.api.routes import EngineServiceProtocol, router


def create_app(service: EngineServiceProtocol) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="SentinelAI AI Engine", lifespan=lifespan)
    app.state.service = service
    app.include_router(router)
    return app
```

- [ ] **Step 4: Add `httpx` for `TestClient`**

`fastapi.testclient.TestClient` requires `httpx`, which nothing in `dev` or `runtime` currently
pulls in. In `pyproject.toml`:

```toml
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.7",
    "mypy>=1.13",
    "httpx>=0.27",
]
```

Run: `cd ai-engine && . .venv/bin/activate && pip install -e ".[runtime,dev]"`

- [ ] **Step 5: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/api -v`
Expected: 7 passed.

- [ ] **Step 6: Verify the full suite once more**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 7: Commit**

```bash
git add sentinel_ai/api tests/api pyproject.toml
git commit -m "$(cat <<'EOF'
feat(api): add the FastAPI surface as thin delegation to EngineService

health/cameras/telemetry/describe-now, all one-line calls into
EngineService behind a structural Protocol so tests inject a fake service
instead of building the real orchestrator stack. UnknownCameraError maps to
404. No WebSocket here — the Go backend owns the WS hub from Phase 1C.
EOF
)"
```

---


---

### Task 11: `Yolo11Detector`

**Files:**
- Create: `ai-engine/sentinel_ai/adapters/detectors/__init__.py`
- Create: `ai-engine/sentinel_ai/adapters/detectors/yolo11.py`
- Create: `ai-engine/tests/adapters/detectors/__init__.py`
- Create: `ai-engine/tests/adapters/detectors/test_yolo11.py`
- Modify: `ai-engine/tests/ports/test_port_contracts.py` — add a class-level subclass check (no instantiation, no weights, no GPU)
- Modify: `ai-engine/pyproject.toml` — add `torch.*` to `[[tool.mypy.overrides]]`

**Interfaces:**
- Consumes: `ObjectDetector.detect` (ports/detector.py), `ModelRuntime` (S13, ports/model_runtime.py: `LifecycleState`, `HealthReport`, `Capabilities`), `Detection`/`BBox` (domain/entities.py), `DEFAULT_SALIENT_CLASSES` (domain/camera_profile.py), `FrameData` (ports/frame_source.py).
- Produces: `Yolo11Detector(model_id, conf, iou, imgsz, device)` per S13; `select_device() -> str` (module-level helper, used by whatever composes this adapter — S1 has no `device` setting, so device selection is this task's own concern, not a config field).

**Design decisions this task settles**

- **Lazy heavy imports.** `torch` and `ultralytics` are imported inside the methods that need them (`initialize`, `warmup`, `detect`, `shutdown`, `version`, and inside `select_device`), never at module scope. `pyproject.toml`'s `gpu` extra (which bundles these) is not installed in CI — CI runs `dev` + `runtime` only. Because the module has no heavy import at load time, `yolo11.py` itself, the pure conversion function, and the `issubclass` port checks are importable and testable in CI with zero `gpu`-extra packages present. This is a deliberate refinement of the plan's own coarse "Task 11: CI no" — only the tests that truly need torch/ultralytics/weights/CUDA are `@pytest.mark.gpu`; the rest run in CI. (Flagged as an author note at the end of this plan, since it reads as a stronger claim than the Task Index table implies.)
- **Weights caching.** A bare `model_id` like `"yolo11s.pt"` makes Ultralytics download into the process's current working directory — unpredictable across a dev shell, a container, and systemd. `initialize()` resolves a bare (non-absolute) `model_id` against `_MODEL_CACHE_DIR`, defaulting to `./var/models` and overridable via `SENTINEL_MODEL_CACHE_DIR` (a plain env var read directly in this module, mirroring the `SENTINEL_` prefix without adding a field to the frozen `Settings` class — S1 is closed to this task). An absolute path is used as-is.
- **Device selection.** `device` is a required constructor argument (S13), not computed internally, because Ultralytics does not probe CUDA availability itself — passing `device="cuda"` on a machine with none raises deep inside torch, not at construction, which would be a confusing place to discover it. `select_device()` is the one place that calls `torch.cuda.is_available()` and returns `"cuda"` or `"cpu"`; whatever composes this adapter (already-landed orchestrator wiring, out of this task's file scope) is expected to call it before constructing.
- **COCO name mapping.** Ultralytics' bundled `coco.yaml` (ships inside the `ultralytics` package, no download needed) names index 0 `person`, 1 `bicycle`, 2 `car`, 3 `motorcycle`, 5 `bus`, 7 `truck`, 24 `backpack`, 26 `handbag`, 28 `suitcase` — an exact string match, lowercase and singular, for every member of `DEFAULT_SALIENT_CLASSES`. `initialize()` asserts `DEFAULT_SALIENT_CLASSES <= set(model.names.values())` at load time and raises loudly (marking the runtime `UNHEALTHY`) rather than silently proceeding, because a mismatched `names` mapping (e.g. a custom-trained checkpoint) would otherwise disable every salient-class escalation trigger with no error anywhere.
- **Thresholds.** `conf`/`iou`/`imgsz` are passed straight through to `model.predict(...)` per call; the constructor just stores them (`detector_conf_threshold`, `detector_iou_threshold`, `detector_imgsz` from S1).
- **`Capabilities`.** `vram_mib` is measured, not guessed: `warmup()` runs one real inference and then reads the VRAM cost, so `capabilities().vram_mib` is 0 until warmed and real afterward.
> **CORRECTED 2026-08-01 after the Task 11 review.** `torch.cuda.memory_allocated()` counts only
> currently-live tensors. It excludes the caching allocator's reserved pool, the CUDA context
> itself (~150–300 MiB, created on first kernel launch and held until process exit), and
> cuDNN/cuBLAS workspaces — all of which `nvidia-smi` sees and the GPU genuinely holds. Measured
> on YOLO11s: `memory_allocated()` 68 MiB, `memory_reserved()` 132 MiB, real `nvidia-smi` delta
> ~281 MiB. A ~4× understatement.
>
> This number feeds `plan_residency()`, the system's GPU admission control, which does hard
> arithmetic against 6144 MiB usable. Under-reporting makes it authorise residency combinations
> that do not fit, and the OOM surfaces on whichever model allocates next — plausibly the VLM
> mid-escalation.
>
> **Use `torch.cuda.memory_reserved()` plus `_CUDA_CONTEXT_OVERHEAD_MIB` (a documented constant,
> 300 MiB) rather than `memory_allocated()`.** The constant is a deliberate over-estimate: the
> planner erring toward evicting one model too early is recoverable; erring toward an OOM
> mid-escalation is not.
 `labels` is the live `model.names` set, not a hardcoded literal — so a future checkpoint swap can never drift silently from what `capabilities()` reports.
- **`FrameData.pixels` narrowing.** `pixels` is typed `object` at the port boundary (it carries a numpy array at runtime, but the port cannot import numpy). `detect()` does an explicit `isinstance(frame.pixels, np.ndarray)` check and raises `TypeError` naming the offending type otherwise — this is the narrowing mypy strict needs, and it turns a silent wrong-type bug into a loud one.
- **Threading.** `detect()` offloads the blocking Ultralytics call to the default executor itself via `loop.run_in_executor`, so it is safe to call directly from the pipeline's event loop regardless of whether `CameraRunner` (spec §5.3) also wraps it in its own executor — a harmless redundant hop if so.

- [ ] **Step 1: Write the failing tests — the CPU-safe ones**

```python
# ai-engine/tests/adapters/detectors/__init__.py
```

```python
# ai-engine/tests/adapters/detectors/test_yolo11.py
"""Tests for Yolo11Detector (Task 11).

Only the parts that need no torch, no ultralytics weights, and no GPU run
without a marker — see yolo11.py's module docstring for why the lazy-import
split makes that possible. The one fact that genuinely needs the
`ultralytics` package (its bundled COCO names) and real inference are
`@pytest.mark.gpu` further down; CI does not install the `gpu` extra, so
those are skipped there, not merely deselected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, _to_detections
from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.model_runtime import ModelRuntime


def test_yolo11_detector_satisfies_both_ports() -> None:
    assert issubclass(Yolo11Detector, ObjectDetector)
    assert issubclass(Yolo11Detector, ModelRuntime)


def test_to_detections_maps_class_ids_through_the_injected_names() -> None:
    names = {0: "person", 2: "car"}
    result = _to_detections(
        boxes_xyxy=[(0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 25.0, 25.0)],
        confidences=[0.91, 0.77],
        class_ids=[0, 2],
        names=names,
    )
    assert result == (
        Detection(label="person", confidence=pytest.approx(0.91), box=BBox(0.0, 0.0, 10.0, 10.0)),
        Detection(label="car", confidence=pytest.approx(0.77), box=BBox(5.0, 5.0, 25.0, 25.0)),
    )


def test_to_detections_on_no_boxes_is_empty() -> None:
    assert _to_detections([], [], [], {}) == ()


@pytest.mark.gpu
def test_real_coco_names_cover_every_default_salient_class() -> None:
    """The bundled coco.yaml ships inside the installed `ultralytics` package
    — no download, no CUDA — but the package itself is `gpu`-extra only, so
    this still cannot run in plain CI.
    """
    import ultralytics
    import yaml

    coco_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    names = set(yaml.safe_load(coco_yaml.read_text())["names"].values())
    assert DEFAULT_SALIENT_CLASSES <= names


@pytest.mark.gpu
async def test_real_detector_loads_warms_up_and_reports_health() -> None:
    from sentinel_ai.adapters.detectors.yolo11 import select_device

    detector = Yolo11Detector(
        model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device=select_device()
    )
    await detector.initialize()
    await detector.warmup()
    assert detector.health().state.value == "healthy"
    assert detector.capabilities().vram_mib >= 0
    assert DEFAULT_SALIENT_CLASSES <= detector.capabilities().labels
    await detector.shutdown()
    assert detector.health().state.value == "unloaded"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/detectors/test_yolo11.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.detectors'`

- [ ] **Step 3: Implement**

```python
# ai-engine/sentinel_ai/adapters/detectors/__init__.py
```

```python
# ai-engine/sentinel_ai/adapters/detectors/yolo11.py
"""Ultralytics YOLO11 object detector (spec §5; ports/detector.py, ports/model_runtime.py).

Implements both `ObjectDetector` (the typed interface the pipeline calls) and
`ModelRuntime` (the lifecycle interface the registry manages) per S13 —
`ModelRuntime.predict()` delegates to `detect()`.

Heavy ML dependencies (`torch`, `ultralytics`) are imported lazily, inside the
methods that need them, never at module scope. This is not a style
preference: it is what lets this module — and every pure helper in it — be
imported and unit-tested on CPU in CI without the `gpu` extra installed.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ultralytics import YOLO

_MODEL_CACHE_DIR = Path(os.environ.get("SENTINEL_MODEL_CACHE_DIR", "./var/models"))
"""Where a bare weights filename (e.g. "yolo11s.pt") resolves to on disk.

Ultralytics downloads a bare filename into the current working directory,
which is unpredictable across dev/container/systemd invocations. Resolving
through one fixed cache directory keeps the download in one place regardless
of launch context. The env var mirrors the `SENTINEL_` settings prefix
without adding a field to the frozen `Settings` class (S1 is closed here).
"""


def select_device() -> str:
    """"cuda" if a CUDA device is visible, else "cpu".

    Ultralytics does not probe availability itself: passing device="cuda" on
    a machine with none raises deep inside torch, not at construction. The
    caller decides *before* construction — hence `device` is a required
    constructor argument (S13), not a default computed inside the adapter.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_detections(
    boxes_xyxy: Sequence[tuple[float, float, float, float]],
    confidences: Sequence[float],
    class_ids: Sequence[int],
    names: Mapping[int, str],
) -> tuple[Detection, ...]:
    """Pure conversion from Ultralytics' box arrays to domain `Detection`s.

    Takes plain Python values, not ultralytics/torch tensors, so it is
    unit-testable with an injected `names` mapping and no model at all.
    """
    return tuple(
        Detection(
            label=names[class_id],
            confidence=float(confidence),
            box=BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
        )
        for (x1, y1, x2, y2), confidence, class_id in zip(
            boxes_xyxy, confidences, class_ids, strict=True
        )
    )


class Yolo11Detector(ObjectDetector, ModelRuntime):
    def __init__(self, model_id: str, conf: float, iou: float, imgsz: int, device: str) -> None:
        self._model_id = model_id
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._model: YOLO | None = None
        self._names: dict[int, str] = {}
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0

    async def initialize(self) -> None:
        self._state = LifecycleState.DOWNLOADING
        try:
            from ultralytics import YOLO

            from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES

            _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            weights_path = (
                self._model_id
                if Path(self._model_id).is_absolute()
                else str(_MODEL_CACHE_DIR / self._model_id)
            )
            model = YOLO(weights_path)
            names: dict[int, str] = dict(model.names)

            missing = DEFAULT_SALIENT_CLASSES - set(names.values())
            if missing:
                raise ValueError(
                    f"model_id={self._model_id!r} names are missing salient classes "
                    f"{missing!r} — every salient-class escalation trigger would "
                    "silently never fire"
                )

            self._model = model
            self._names = names
            self._state = LifecycleState.LOADED
        except Exception as exc:  # noqa: BLE001 — any load failure marks UNHEALTHY, never crashes
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Yolo11Detector.warmup called before initialize()")
        dummy = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=self._imgsz,
            height=self._imgsz,
            pixels=np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8),
        )
        await self.detect(dummy)
        import torch

        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(self._device) // (1024 * 1024)
            self._vram_mib = int(reserved) + _CUDA_CONTEXT_OVERHEAD_MIB
        self._state = LifecycleState.HEALTHY

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        if self._model is None:
            raise RuntimeError("Yolo11Detector.detect called before initialize()")
        if not isinstance(frame.pixels, np.ndarray):
            raise TypeError(f"FrameData.pixels must be a numpy array, got {type(frame.pixels)!r}")
        pixels: np.ndarray = frame.pixels
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, pixels)

    def _detect_sync(self, pixels: np.ndarray) -> tuple[Detection, ...]:
        assert self._model is not None
        results = self._model.predict(
            source=pixels,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()
        xyxy = [tuple(row) for row in boxes.xyxy.tolist()]
        confidences = boxes.conf.tolist()
        class_ids = [int(c) for c in boxes.cls.tolist()]
        return _to_detections(xyxy, confidences, class_ids, self._names)

    async def predict(self, request: object) -> object:
        if not isinstance(request, FrameData):
            raise TypeError(f"Yolo11Detector.predict expects FrameData, got {type(request)!r}")
        return await self.detect(request)

    async def shutdown(self) -> None:
        self._model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state = LifecycleState.UNLOADED

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        import ultralytics

        return f"ultralytics=={ultralytics.__version__} model={self._model_id}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_key=self._model_id,
            kind="detector",
            labels=frozenset(self._names.values()),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
```

Add to `tests/ports/test_port_contracts.py` (append; needs no new import of anything heavy):

```python
from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector


def test_yolo11_detector_satisfies_object_detector_and_model_runtime() -> None:
    """Class-level only — no instantiation, no weights, no GPU needed, so
    this belongs beside the fakes' contract checks, not behind @pytest.mark.gpu.
    """
    assert issubclass(Yolo11Detector, ObjectDetector)
    assert issubclass(Yolo11Detector, ModelRuntime)
```

Add to `pyproject.toml`'s `[[tool.mypy.overrides]]` `module` list: `"torch.*"`.

- [ ] **Step 4: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`. (The `@pytest.mark.gpu` tests are not run here; run them on the GPU box with the `gpu` extra installed: `pip install -e ".[gpu]" && python -m pytest tests/adapters/detectors/test_yolo11.py -v -m gpu`.)

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/adapters/detectors/ ai-engine/tests/adapters/detectors/ ai-engine/tests/ports/test_port_contracts.py ai-engine/pyproject.toml
git commit -m "feat(adapters): add Yolo11Detector

Lazy-imports torch/ultralytics so the module, its pure box-to-Detection
mapping, and the port-contract check run in CI without the gpu extra;
verifies the loaded checkpoint's COCO names cover every DEFAULT_SALIENT_CLASS
at load time instead of trusting it silently."
```

---

### Task 12: `ByteTrackTracker`

**Files:**
- Create: `ai-engine/sentinel_ai/adapters/trackers/__init__.py`
- Create: `ai-engine/sentinel_ai/adapters/trackers/bytetrack.py`
- Create: `ai-engine/tests/adapters/trackers/__init__.py`
- Create: `ai-engine/tests/adapters/trackers/test_bytetrack.py`
- Modify: `ai-engine/tests/ports/test_port_contracts.py` — add a class-level subclass check
- Modify: `ai-engine/pyproject.toml` — move `supervision` from the `gpu` extra to `runtime`, pinned `<0.28`

**Interfaces:**
- Consumes: `Tracker` (ports/tracker.py), `Detection`/`Track`/`BBox` (domain/entities.py).
- Produces: `ByteTrackTracker(frame_rate: int = 30)` per S13.

**Design decisions this task settles**

- **`pyproject.toml` move, not just a note.** `supervision` currently sits under the `gpu` extra alongside `torch`/`ultralytics`/`autoawq`, but supervision's `ByteTrack` is a numpy/scipy association algorithm with no torch dependency. Left there, this task's own instruction ("tests belong in CI") would be false — CI installs `dev` + `runtime` only, never `gpu`. Moving `supervision` into `runtime` is what makes that true. Pinned `supervision>=0.24,<0.28`: `sv.ByteTrack` is deprecated (in favour of a separate `trackers` package) starting supervision 0.28, and a `DeprecationWarning` at construction time would break the suite's `-W error` requirement. Migrating to the `trackers` package is out of scope for this task; the pin defers it cleanly.
- **`Detection` ⇄ `sv.Detections` round trip.** `sv.Detections` carries a `class_id: np.ndarray[int]`, not a string label. This adapter maintains a bijective `label ⇄ class_id` map (`_label_ids`/`_id_labels`), assigning a new small integer the first time a label is seen, so it never needs `sv.Detections.data` (whose pass-through behaviour across `update_with_detections` this task does not rely on).
- **`age_frames`.** ByteTrack returns `tracker_id`, `xyxy`, `class_id`, `confidence` per call — it does not track how long a `tracker_id` has been alive. This adapter maintains `self._age: dict[track_id, int]`, incrementing by one every time a `tracker_id` reappears in the output and never decaying it for a track that briefly leaves frame (ByteTrack itself, via `lost_track_buffer`, is what decides whether a reappearing id is "the same" track or a new one — this adapter just counts how many times it has seen whatever id ByteTrack hands back). This is what the escalation gate's `min_track_frames=8` debounce reads, so getting the increment-exactly-once-per-sighting semantics right matters more than anything else in this file.
- **`speed_px_s`.** Centroid displacement over elapsed wall time: `self._last_position: dict[track_id, (BBox, float)]` remembers the previous box and timestamp per id; `update()`'s `timestamp` parameter (explicit, per the `Tracker` port) is what elapsed time is computed against, matching `FakeTracker`'s existing convention exactly.
- **`reset()`.** Calls supervision's own `ByteTrack.reset()` (clears its tracked/lost/removed lists and rewinds its internal id counter to 0), then clears this adapter's own `_age`, `_last_position`, and label maps. The Phase 1A review found `FakeTracker`'s original reset test vacuous (it passed against a no-op `reset()`); the test below is written the same corrected way: track two objects so the id counter advances past 1, reset, re-sight one of them, and assert both a *fresh* `track_id == 1` and `age_frames == 1` — a no-op `reset()` would fail both.

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/adapters/trackers/__init__.py
```

```python
# ai-engine/tests/adapters/trackers/test_bytetrack.py
"""Tests for ByteTrackTracker (Task 12). Runs on CPU: supervision's ByteTrack
has no GPU/torch dependency, so — unlike the model tasks either side of this
one — these tests belong in CI.
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.tracker import Tracker


def _det(label: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, box=BBox(x1, y1, x2, y2))


def test_bytetrack_tracker_satisfies_its_port() -> None:
    assert issubclass(ByteTrackTracker, Tracker)


def test_stable_ids_across_frames() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    first = tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    second = tracker.update((_det("person", 1.0, 0.0, 11.0, 10.0),), timestamp=0.1)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_age_frames_increments_once_per_sighting() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    tracker.update((_det("person", 1.0, 0.0, 11.0, 10.0),), timestamp=0.1)
    third = tracker.update((_det("person", 2.0, 0.0, 12.0, 10.0),), timestamp=0.2)
    assert third[0].age_frames == 3


def test_speed_from_known_displacement_and_elapsed_time() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    moved = tracker.update((_det("person", 20.0, 0.0, 30.0, 10.0),), timestamp=1.0)
    assert moved[0].speed_px_s == pytest.approx(20.0, rel=0.05)


def test_reset_clears_state_and_a_re_sighted_object_gets_a_fresh_id() -> None:
    """Re-sight `far` after reset — an id or age that merely happens to look
    right is not enough (the Phase 1A review found exactly this gap in
    FakeTracker's original reset test)."""
    tracker = ByteTrackTracker(frame_rate=10)
    near = _det("person", 0.0, 0.0, 10.0, 10.0)
    far = _det("person", 500.0, 500.0, 510.0, 510.0)

    tracker.update((near,), timestamp=0.0)
    before = tracker.update((near, far), timestamp=0.1)
    far_before = next(t for t in before if t.box.x1 > 100.0)
    assert far_before.track_id != 1, "the id counter has advanced past its starting value"

    tracker.reset()
    after = tracker.update((far,), timestamp=1.0)

    assert after[0].track_id != far_before.track_id, "the association must be forgotten"
    assert after[0].track_id == 1, "the id counter must be rewound"
    assert after[0].age_frames == 1, "a re-sighted object is new, not aged"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/trackers/test_bytetrack.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.trackers'`

- [ ] **Step 3: Implement**

```python
# ai-engine/sentinel_ai/adapters/trackers/__init__.py
```

```python
# ai-engine/sentinel_ai/adapters/trackers/bytetrack.py
"""Supervision ByteTrack tracker (spec §5; ports/tracker.py).

Runs entirely on CPU — supervision's ByteTrack is a numpy/scipy association
algorithm with no torch dependency — which is why, unlike the model adapters
either side of it in the task list, its tests belong in CI.

ByteTrack hands back `tracker_id`, `xyxy`, `confidence`, `class_id` per call;
it does not hand back how long a track has existed or how fast it is moving,
both of which the escalation gate needs (`min_track_frames`,
`speed_fallback_px_s`). This adapter maintains both itself, keyed by
`tracker_id`.
"""

from __future__ import annotations

from math import hypot

import numpy as np
import supervision as sv

from sentinel_ai.domain.entities import BBox, Detection, Track
from sentinel_ai.ports.tracker import Tracker


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: int = 30) -> None:
        self._tracker = sv.ByteTrack(frame_rate=frame_rate)
        self._age: dict[int, int] = {}
        self._last_position: dict[int, tuple[BBox, float]] = {}
        self._label_ids: dict[str, int] = {}
        self._id_labels: dict[int, str] = {}

    def _class_id_for(self, label: str) -> int:
        if label not in self._label_ids:
            new_id = len(self._label_ids)
            self._label_ids[label] = new_id
            self._id_labels[new_id] = label
        return self._label_ids[label]

    def _to_sv_detections(self, detections: tuple[Detection, ...]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()
        xyxy = np.array(
            [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections], dtype=np.float32
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([self._class_id_for(d.label) for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Track, ...]:
        sv_detections = self._to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        if len(tracked) == 0:
            return ()

        results: list[Track] = []
        for xyxy, tracker_id, class_id in zip(
            tracked.xyxy, tracked.tracker_id, tracked.class_id, strict=True
        ):
            track_id = int(tracker_id)
            label = self._id_labels[int(class_id)]
            box = BBox(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]))

            age = self._age.get(track_id, 0) + 1
            self._age[track_id] = age

            previous = self._last_position.get(track_id)
            if previous is not None:
                prev_box, prev_ts = previous
                elapsed = timestamp - prev_ts
                speed = (
                    hypot(box.cx - prev_box.cx, box.cy - prev_box.cy) / elapsed
                    if elapsed > 0
                    else 0.0
                )
            else:
                speed = 0.0
            self._last_position[track_id] = (box, timestamp)

            results.append(
                Track(track_id=track_id, label=label, box=box, age_frames=age, speed_px_s=speed)
            )
        return tuple(results)

    def reset(self) -> None:
        self._tracker.reset()
        self._age.clear()
        self._last_position.clear()
        self._label_ids.clear()
        self._id_labels.clear()
```

Add to `tests/ports/test_port_contracts.py` (append):

```python
from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker


def test_bytetrack_tracker_satisfies_tracker() -> None:
    assert issubclass(ByteTrackTracker, Tracker)
```

Edit `pyproject.toml`:

```toml
runtime = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.31",
    "av>=13.0",
    "numpy>=1.26,<2.2",
    "aio-pika>=9.4",
    "minio>=7.2",
    "supervision>=0.24,<0.28",
]
gpu = [
    "torch>=2.5",
    "ultralytics>=8.3",
    "transformers>=4.49",
    "accelerate>=1.0",
    "autoawq>=0.2.6",
    "qwen-vl-utils>=0.0.8",
]
```

(`supervision.*` is already in `[[tool.mypy.overrides]]` — no change needed there.)

- [ ] **Step 4: Verify**

Run: `cd ai-engine && . .venv/bin/activate && pip install -e ".[runtime,dev]" && python -m pytest -q -m "not gpu and not integration" -W error && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`, no warnings.

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/adapters/trackers/ ai-engine/tests/adapters/trackers/ ai-engine/tests/ports/test_port_contracts.py ai-engine/pyproject.toml
git commit -m "feat(adapters): add ByteTrackTracker

Moves supervision from the gpu extra to runtime (ByteTrack is numpy-based,
not torch-based) so this is the one model adapter whose tests run in CI;
pins <0.28 to dodge ByteTrack's deprecation warning under -W error."
```

---

### Task 13: `Qwen25VLDescriber`

**Files:**
- Create: `ai-engine/sentinel_ai/adapters/vision/__init__.py`
- Create: `ai-engine/sentinel_ai/adapters/vision/qwen25vl.py`
- Create: `ai-engine/tests/adapters/vision/__init__.py`
- Create: `ai-engine/tests/adapters/vision/test_qwen25vl.py`
- Modify: `ai-engine/tests/ports/test_port_contracts.py` — add a class-level subclass check
- Modify: `ai-engine/pyproject.toml` — add `bitsandbytes` to the `gpu` extra (Branch B needs it even if Branch A is the one taken); add `transformers.*` and `bitsandbytes.*` to `[[tool.mypy.overrides]]` (`torch.*` already added by Task 11)

**Interfaces:**
- Consumes: `VisionLanguageModel`, `VisionRequest`, `SceneDescription` (ports/vision_llm.py), `ModelRuntime` (S13), `SceneState` (domain/entities.py), `FrameData` (ports/frame_source.py).
- Produces: `Qwen25VLDescriber(model_id, max_new_tokens, device)` per S13.

**Design decisions this task settles**

- **Which branch.** Read `docs/superpowers/plans/phase1b-spike-result.md` (written by Task 1). If it records that `autoawq` loaded successfully on the target GPU, implement `initialize()` with **Branch A** below. If it records failure, implement `initialize()` with **Branch B**. Every other method is identical either way — that is the entire point of the `VisionLanguageModel` port abstraction (spec §7). Do not implement both; the file has exactly one `initialize()` method.
- **Lazy heavy imports**, same rationale as Task 11: `torch`, `transformers`, `qwen_vl_utils`'s transitive `PIL` dependency are imported inside the methods that need them. `_build_text_prompt` and `_parse_response` are pure stdlib + domain-type functions and need none of it, which is what makes them CI-testable with no `gpu` extra installed — the task's own instruction that "parsing and prompt-construction tests run on CPU in CI" is only true because of this split.
- **Prompt template.** `_build_text_prompt` is pure string construction from `VisionRequest`'s text fields (camera label, reason detail, tracked-object counts, motion energy, prior descriptions) plus a strict-JSON response instruction. It never mentions a model name.
- **Parsing robustness.** `_parse_response` finds the first `{...}` block, `json.loads`s it, validates the three required keys and types, and clamps `threat_value` into `[0, 1]` (a raw value of `1.4` or `-0.2` must not blow up `ThreatScore.from_value` downstream). Any failure at any step — no JSON found, invalid JSON, wrong types, missing keys — falls through to a fixed, safe `SceneDescription` built from the raw text, **not an exception**: a VLM that returns prose instead of JSON is a formatting slip, not an infrastructure failure, and per spec §9 must not be confused with the timeout/OOM path that flags `description_unavailable=True` at the scheduler (S14). This class's `describe()` always returns a `SceneDescription`.
- **No model identity leak (spec §3.3).** Neither `_build_text_prompt` nor `_parse_response` nor the `SceneDescription` they produce ever contains "Qwen" or any model name. `version()` and `Capabilities.model_key` do carry the model id, but that is orchestrator-internal (the registry's health view) — a different boundary than the published `Event`, which is assembled elsewhere (S14) from `SceneDescription` alone.
- **Idle-unload interaction.** The 600 s idle-unload (`vlm_idle_unload_seconds`, S1) is entirely `ResidentSet`'s decision (Task 7, already landed) via `ModelSpec(idle_unload_seconds=600.0)` — this class has no timer of its own. Its only obligations toward that mechanism: `shutdown()` must actually free VRAM (`del` the model, `torch.cuda.empty_cache()`) so the 600 s reclaim is real, not nominal, and `initialize()` must be safe to call again afterward so `ResidentSet.ensure()` can reload on the next escalation.
- **`Capabilities.vram_mib` measured, not guessed.** `warmup()` runs one real `describe()` call (on a synthetic frame) and then reads `torch.cuda.memory_reserved()` plus `_CUDA_CONTEXT_OVERHEAD_MIB` — see the correction under Task 11; do NOT use `memory_allocated()`. Spec §7 puts the bitsandbytes fallback at "~3.5 GB"; the AWQ path is expected to land near the same figure. Neither number is hardcoded — this is what the spike's own measurement is for.
- **Keyframe handoff.** `VisionRequest.keyframe.pixels` is `object` at the port boundary; `describe()` does the same `isinstance(..., np.ndarray)` narrowing as `Yolo11Detector.detect`, then wraps it as `PIL.Image.fromarray(...)` for the processor's chat-template image slot.
- **Determinism.** `do_sample=False, num_beams=1` (greedy decoding) — so the same frame and the same prompt produce a comparable description run to run, which matters for dedup/testing and for a human reviewing two similar events.

- [ ] **Step 1: Write the failing test — the CPU-safe parts**

```python
# ai-engine/tests/adapters/vision/__init__.py
```

```python
# ai-engine/tests/adapters/vision/test_qwen25vl.py
"""Tests for Qwen25VLDescriber (Task 13).

_build_text_prompt and _parse_response are pure — no torch, no transformers,
no GPU — which is what lets them run in CI. Real inference (either branch)
is @pytest.mark.gpu and exercised locally on the GPU box.
"""

from __future__ import annotations

from sentinel_ai.adapters.vision.qwen25vl import (
    Qwen25VLDescriber,
    _base_checkpoint,
    _build_text_prompt,
    _parse_response,
)
from sentinel_ai.domain.entities import BBox, Detection, SceneState, Track
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import ModelRuntime
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest


def _scene(tracks: tuple[Track, ...] = ()) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=tracks,
        motion_energy=0.4,
        scene_signature=(1.0,),
    )


def _request(**overrides: object) -> VisionRequest:
    frame = FrameData(
        camera_id="cam-1", frame_index=0, timestamp=0.0, width=4, height=4, pixels=None
    )
    defaults: dict[str, object] = dict(
        keyframe=frame,
        scene=_scene(),
        history=(),
        camera_label="Front Door",
        reason_detail="new salient track: person",
    )
    defaults.update(overrides)
    return VisionRequest(**defaults)  # type: ignore[arg-type]


def test_qwen25vl_describer_satisfies_both_ports() -> None:
    assert issubclass(Qwen25VLDescriber, VisionLanguageModel)
    assert issubclass(Qwen25VLDescriber, ModelRuntime)


def test_base_checkpoint_strips_the_awq_suffix() -> None:
    assert _base_checkpoint("Qwen/Qwen2.5-VL-3B-Instruct-AWQ") == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_base_checkpoint_is_unchanged_without_the_suffix() -> None:
    assert _base_checkpoint("Qwen/Qwen2.5-VL-3B-Instruct") == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_prompt_includes_camera_label_reason_and_tracked_object_counts() -> None:
    track = Track(track_id=1, label="person", box=BBox(0.0, 0.0, 1.0, 1.0), age_frames=9, speed_px_s=0.0)
    prompt = _build_text_prompt(_request(scene=_scene(tracks=(track, track))))
    assert "Front Door" in prompt
    assert "new salient track: person" in prompt
    assert "2 person" in prompt
    assert "Qwen" not in prompt


def test_prompt_never_mentions_a_model_name() -> None:
    assert "qwen" not in _build_text_prompt(_request()).lower()


def test_parse_response_reads_well_formed_json() -> None:
    raw = '{"description": "A person walks by.", "threat_value": 0.2, "suggested_action": "None."}'
    result = _parse_response(raw)
    assert result == SceneDescription(
        description="A person walks by.", threat_value=0.2, suggested_action="None."
    )


def test_parse_response_extracts_json_embedded_in_prose_or_fences() -> None:
    raw = 'Sure, here you go:\n```json\n{"description": "Ok.", "threat_value": 0.1, "suggested_action": "None."}\n```'
    result = _parse_response(raw)
    assert result.description == "Ok."
    assert result.threat_value == 0.1


def test_parse_response_clamps_an_out_of_range_threat_value() -> None:
    raw = '{"description": "X", "threat_value": 1.7, "suggested_action": "Y"}'
    assert _parse_response(raw).threat_value == 1.0


def test_parse_response_falls_back_on_prose_with_no_json() -> None:
    result = _parse_response("There is a person near the entrance, nothing concerning.")
    assert result.description == "There is a person near the entrance, nothing concerning."
    assert 0.0 <= result.threat_value <= 1.0
    assert result.suggested_action


def test_parse_response_falls_back_on_json_missing_a_required_key() -> None:
    raw = '{"description": "X", "suggested_action": "Y"}'
    result = _parse_response(raw)
    assert result.description == raw
    assert 0.0 <= result.threat_value <= 1.0


def test_parse_response_never_raises_on_empty_text() -> None:
    result = _parse_response("")
    assert result.description
    assert 0.0 <= result.threat_value <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/vision/test_qwen25vl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.vision'`

- [ ] **Step 3: Implement**

```python
# ai-engine/sentinel_ai/adapters/vision/__init__.py
```

```python
# ai-engine/sentinel_ai/adapters/vision/qwen25vl.py
"""Qwen2.5-VL vision-language describer (spec §5; ports/vision_llm.py, ports/model_runtime.py).

Loading path is whatever Task 1's spike decided
(`docs/superpowers/plans/phase1b-spike-result.md`): Branch A's `initialize()`
below if AWQ loaded on the target GPU, Branch B's if it did not. Every other
method is identical either way — that is the entire reason the
`VisionLanguageModel` port abstraction exists (spec §7). Use exactly one.

Heavy ML dependencies (`torch`, `transformers`, `PIL`) are imported lazily
inside the methods that need them, never at module scope — see
`adapters/detectors/yolo11.py`'s module docstring for why: it is what makes
`_build_text_prompt` and `_parse_response` unit-testable on CPU in CI with no
`gpu` extra installed.

Spec §3.3: the event carries no model identity. `_build_text_prompt` and
`_parse_response` never mention a model name, and neither does the
`SceneDescription` this class returns; `version()` and
`Capabilities.model_key` are orchestrator-internal (the registry's health
view) — a different boundary from the published event.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np

from sentinel_ai.domain.entities import SceneState
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import Capabilities, HealthReport, LifecycleState, ModelRuntime
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest

if TYPE_CHECKING:
    import torch as torch_module

_AWQ_SUFFIX = "-AWQ"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK_ACTION = "Review the clip manually — automated threat assessment unavailable."

_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON object of this exact shape, no other text:\n"
    '{"description": "<one or two plain sentences describing what is visible>", '
    '"threat_value": <float between 0.0 and 1.0>, '
    '"suggested_action": "<one short, concrete sentence for a human reviewer>"}'
)


def _base_checkpoint(model_id: str) -> str:
    """Branch B loads the unquantized checkpoint; the `-AWQ` suffix only
    names the AWQ-prequantized repo Branch A uses."""
    return model_id[: -len(_AWQ_SUFFIX)] if model_id.endswith(_AWQ_SUFFIX) else model_id


def _format_detections(scene: SceneState) -> str:
    if not scene.tracks:
        return "no tracked objects"
    counts: dict[str, int] = {}
    for track in scene.tracks:
        counts[track.label] = counts.get(track.label, 0) + 1
    return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


def _format_history(history: tuple[str, ...]) -> str:
    if not history:
        return "(none yet)"
    return "\n".join(f"- {line}" for line in history)


def _build_text_prompt(request: VisionRequest) -> str:
    """Pure string construction — no image, no model."""
    return (
        "You are a security monitoring assistant describing a single still frame "
        f"from a fixed security camera named '{request.camera_label}'.\n"
        f"This frame was captured because: {request.reason_detail}\n"
        f"Objects currently tracked in view: {_format_detections(request.scene)}\n"
        f"Recent motion energy (0=static, 1=high motion): {request.scene.motion_energy:.2f}\n"
        "Prior descriptions for this camera, most recent last:\n"
        f"{_format_history(request.history)}\n\n"
        "Describe what is visible, assess how concerning it is, and suggest one "
        f"action for a human reviewer.\n{_RESPONSE_INSTRUCTIONS}"
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_response(raw_text: str) -> SceneDescription:
    """Parse the model's reply, tolerating one that ignores the JSON
    instruction — a formatting slip must never crash the pipeline (spec §9).
    """
    candidate = raw_text.strip()
    match = _JSON_BLOCK.search(candidate)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            description = payload.get("description")
            threat_value = payload.get("threat_value")
            suggested_action = payload.get("suggested_action")
            if (
                isinstance(description, str)
                and description.strip()
                and isinstance(threat_value, int | float)
                and not isinstance(threat_value, bool)
                and isinstance(suggested_action, str)
                and suggested_action.strip()
            ):
                return SceneDescription(
                    description=description.strip(),
                    threat_value=_clamp01(float(threat_value)),
                    suggested_action=suggested_action.strip(),
                )
    fallback_description = candidate[:500] if candidate else "The vision model returned no text."
    return SceneDescription(
        description=fallback_description, threat_value=0.5, suggested_action=_FALLBACK_ACTION
    )


class Qwen25VLDescriber(VisionLanguageModel, ModelRuntime):
    def __init__(self, model_id: str, max_new_tokens: int, device: str) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._model: object | None = None
        self._processor: object | None = None
        self._torch: torch_module | None = None
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0

    async def initialize(self) -> None:
        """BRANCH A — use if phase1b-spike-result.md records that autoawq
        loaded. Delete this method and use Branch B's version instead
        (below) if it recorded failure.
        """
        self._state = LifecycleState.DOWNLOADING
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._torch = torch
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_id, torch_dtype=torch.float16, device_map=self._device
            )
            model.eval()
            self._model = model
            self._processor = AutoProcessor.from_pretrained(self._model_id)
            self._state = LifecycleState.LOADED
        except Exception as exc:  # noqa: BLE001 — any load failure marks UNHEALTHY, never crashes
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Qwen25VLDescriber.warmup called before initialize()")
        dummy_frame = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=64,
            height=64,
            pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        dummy_scene = SceneState(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            detections=(),
            tracks=(),
            motion_energy=0.0,
            scene_signature=(1.0,),
        )
        request = VisionRequest(
            keyframe=dummy_frame,
            scene=dummy_scene,
            history=(),
            camera_label="warmup",
            reason_detail="warmup",
        )
        await self.describe(request)
        assert self._torch is not None
        if self._torch.cuda.is_available():
            reserved = self._torch.cuda.memory_reserved(self._device) // (1024 * 1024)
            self._vram_mib = int(reserved) + _CUDA_CONTEXT_OVERHEAD_MIB
        self._state = LifecycleState.HEALTHY

    async def describe(self, request: VisionRequest) -> SceneDescription:
        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError("Qwen25VLDescriber.describe called before initialize()")
        if not isinstance(request.keyframe.pixels, np.ndarray):
            raise TypeError(
                f"FrameData.pixels must be a numpy array, got {type(request.keyframe.pixels)!r}"
            )
        import asyncio

        from PIL import Image

        pixels: np.ndarray = request.keyframe.pixels
        image = Image.fromarray(pixels.astype(np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _build_text_prompt(request)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(  # type: ignore[attr-defined]
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)  # type: ignore[attr-defined]

        loop = asyncio.get_running_loop()
        generated_ids = await loop.run_in_executor(None, self._generate_sync, inputs)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        raw_text = self._processor.batch_decode(  # type: ignore[attr-defined]
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _parse_response(raw_text)

    def _generate_sync(self, inputs: object) -> object:
        assert self._model is not None
        assert self._torch is not None
        with self._torch.inference_mode():
            return self._model.generate(  # type: ignore[attr-defined]
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False, num_beams=1
            )

    async def predict(self, request: object) -> object:
        if not isinstance(request, VisionRequest):
            raise TypeError(
                f"Qwen25VLDescriber.predict expects VisionRequest, got {type(request)!r}"
            )
        return await self.describe(request)

    async def shutdown(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._state = LifecycleState.UNLOADED

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        import transformers

        return f"transformers=={transformers.__version__} model={self._model_id}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_key=self._model_id,
            kind="vision",
            labels=frozenset(),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
```

**BRANCH B — replaces the `initialize` method above.** Use this instead if `phase1b-spike-result.md` recorded that autoawq failed to load. Everything else in the file (constructor, `warmup`, `describe`, `predict`, `shutdown`, `health`, `version`, `capabilities`, and the module-level `_build_text_prompt`/`_parse_response`/`_base_checkpoint`) is unchanged:

```python
    async def initialize(self) -> None:
        """BRANCH B — fallback: transformers + bitsandbytes 4-bit NF4 on the
        unquantized checkpoint. Comparable VRAM to AWQ (~3.5 GB per spec §7),
        actively maintained. `_base_checkpoint` strips the `-AWQ` suffix that
        only names the AWQ-prequantized repo, so `self._model_id` can stay
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" (the default in Settings) without
        this branch ever trying to load the AWQ repo itself.
        """
        self._state = LifecycleState.DOWNLOADING
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )

            self._torch = torch
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            checkpoint = _base_checkpoint(self._model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint, quantization_config=quantization_config, device_map=self._device
            )
            model.eval()
            self._model = model
            self._processor = AutoProcessor.from_pretrained(checkpoint)
            self._state = LifecycleState.LOADED
        except Exception as exc:  # noqa: BLE001 — any load failure marks UNHEALTHY, never crashes
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise
```

Add to `tests/ports/test_port_contracts.py` (append):

```python
from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber


def test_qwen25vl_describer_satisfies_both_ports() -> None:
    assert issubclass(Qwen25VLDescriber, VisionLanguageModel)
    assert issubclass(Qwen25VLDescriber, ModelRuntime)
```

Edit `pyproject.toml`'s `gpu` extra to add `bitsandbytes` (needed if Branch B is taken; harmless if Branch A is):

```toml
gpu = [
    "torch>=2.5",
    "ultralytics>=8.3",
    "transformers>=4.49",
    "accelerate>=1.0",
    "autoawq>=0.2.6",
    "qwen-vl-utils>=0.0.8",
    "bitsandbytes>=0.44",
]
```

Add `"transformers.*"` and `"bitsandbytes.*"` to `[[tool.mypy.overrides]]`'s `module` list (`"torch.*"` was added by Task 11 — confirm it is there rather than re-adding it).

- [ ] **Step 4: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" -W error && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`, no warnings. On the GPU box, after `pip install -e ".[gpu]"`: `python -m pytest tests/adapters/vision/test_qwen25vl.py -v -m gpu`.

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/adapters/vision/ ai-engine/tests/adapters/vision/ ai-engine/tests/ports/test_port_contracts.py ai-engine/pyproject.toml
git commit -m "feat(adapters): add Qwen25VLDescriber

Branches initialize() on the Task 1 spike's recorded result (autoawq vs
transformers+bitsandbytes NF4); every other method, including the
malformed-output-tolerant response parser, is identical either way."
```

---

### Task 14: `RtspSource` + dataset script + end-to-end demo

**Files:**
- Create: `ai-engine/sentinel_ai/adapters/sources/rtsp.py`
- Create: `ai-engine/tests/adapters/sources/test_rtsp.py` (assumes `ai-engine/tests/adapters/sources/__init__.py` already exists from Task 4; create it if it does not)
- Create: `datasets/download_sample.sh`
- Create: `datasets/README.md`
- Create: `ai-engine/tests/e2e/__init__.py`
- Create: `ai-engine/tests/e2e/test_demo_smoke.py`
- Create: `ai-engine/README.md`

**Interfaces:**
- Consumes: `FrameSource`, `EncodedPacket`, `FrameData` (ports/frame_source.py, S2/S3), `rtsp_reconnect_initial_seconds`/`rtsp_reconnect_max_seconds` (S1).
- Produces: `RtspSource(camera_id, url, *, reconnect_initial_seconds, reconnect_max_seconds, clock=time.monotonic)`. Not in S13 — sources are not model-runtime lifecycle objects, so this constructor is this task's own to define.

**Design decisions this task settles**

- **Reconnect is a separately testable unit.** `_ReconnectLoop` owns the retry/backoff sequencing and nothing else — no PyAV, no asyncio queues. It takes an injected `connect_once` coroutine and an injected `sleep` coroutine, so its exponential-backoff-doubling-capped-at-max behaviour is asserted with fakes: no real sleep, no real network, no GPU. `RtspSource` composes it with the real PyAV connect/decode loop as `connect_once`.
- **The discontinuity signal, concretely.** On every reconnect, `RtspSource` resets its own `frame_index` counter to 0 while leaving the injected monotonic `clock` untouched — `FrameData.timestamp` keeps advancing across the reconnect (so cooldown/rate-limit logic downstream, which assumes non-decreasing time, never sees time run backward), but `frame_index` restarting is the signal a consumer watches for. This reuses the same convention a looping `FileSource` already needs (spec §5.2's note that a mid-stream signature-length change must be treated as a discontinuity applies equally to a frame-index rewind) rather than adding a new callback to the already-landed `CameraRunner`/`FrameSource` contract. See the author note at the end of this plan: this is stated as a contract `CameraRunner` (Task 6) is expected to honour, not verified against Task 6's actual source here, since that file is out of this task's scope.
- **Timestamps are wall-clock, not the RTSP stream's own `pts`.** A file has one continuous timeline, so `FileSource` can trust its embedded `pts`. A live RTSP session's `pts` restarts (or is otherwise renegotiated) on every reconnect and cannot be trusted as the pipeline's monotonic clock. `RtspSource` stamps both `FrameData.timestamp` and `EncodedPacket.pts` from the injected `clock()` at the moment of arrival — the one thing guaranteed to keep moving forward across a reconnect.
- **Threading.** One worker thread per connection attempt runs the blocking `av.open`/`demux`/`decode` loop; each decoded `FrameData`/`EncodedPacket` crosses back to the event loop via `asyncio.run_coroutine_threadsafe(queue.put(...), loop).result()`, which is what keeps the two `asyncio.Queue`s single-writer-safe from the event loop's perspective while still blocking the worker thread on backpressure (the queues are bounded).
- **Camera disconnect → offline + event (spec §6).** This task delivers the reconnect/backoff half fully. The "status → offline" half is observable through telemetry that already exists without a new field: `CameraTelemetry.last_frame_at` (S11) stops advancing while wall-clock time keeps moving, which is what a health consumer polls for staleness. The "event emitted" half cannot be built inside this task's file scope: `EscalationReason` (domain/entities.py, frozen, not in this task's file list) has no connectivity-related member, and inventing one here would violate "use S1–S14 verbatim" while also touching a file this task does not own. This is flagged as an author note at the end of this plan rather than silently only half-implemented.
- **Dataset licence, verified before hardcoding the URL.** CUHK Avenue's own host page (`http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html`) publishes no separate licence file, but is consistently described — by the dataset's own citation request and by every academic redistribution checked (e.g. the Anomalib datamodule docs) — as released for **academic research use**; there is no commercial-use grant published anywhere. The URL was confirmed reachable and correctly named (`Avenue_Dataset.zip`, 813,227,845 bytes, `Content-Type: application/zip`) directly against the CUHK host, not a third-party mirror. SentinelAI's use — one clip, looped locally for development/demo, never redistributed — sits inside that grant; `datasets/README.md` records this verification trail and the citation, not just a licence label.
- **Only one clip is kept.** The archive bundles 16 training + 21 testing videos (~780 MiB); the script downloads the full archive (no server-side partial-zip extraction is practical here), extracts only `testing_videos/01.avi`, re-encodes it to H.264/yuv420p (the source is Xvid-encoded AVI, which mediamtx/RTSP publishing does not want), and deletes the archive and everything else extracted from it.
- **What is automated vs. manual.** The RTSP-decode path itself (`_connect_once`/`_pump_stream`) needs a real stream and is not asserted in CI — there is no CI-safe way to test a live RTSP reconnect without a running mediamtx. The reconnect *sequencing* (`_ReconnectLoop`) is fully unit-tested in CI. The full mediamtx→RTSP→real-models→RabbitMQ→MinIO demo is manual, with an exact command sequence and a verification checklist below. One automated proxy exists: `tests/e2e/test_demo_smoke.py`, `@pytest.mark.gpu` and `@pytest.mark.integration`, runs the same real detector/tracker/VLM trio over the downloaded Avenue clip through `FileSource(realtime=False)` instead of RTSP (so it does not also require mediamtx) and asserts one real, complete `Event` results.

- [ ] **Step 1: Write the failing test — the CPU-safe reconnect logic**

```python
# ai-engine/tests/adapters/sources/test_rtsp.py
"""Tests for RtspSource (Task 14).

`_ReconnectLoop`'s backoff sequencing is fully decoupled from PyAV and
networking, so it is CPU/CI-safe: no real sleep, no real connection, no GPU.
The PyAV decode path itself needs a real RTSP source (mediamtx) and is
exercised manually — see the task's end-to-end demo section — there is no
CI-safe way to test a live RTSP reconnect without a running mediamtx.
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.sources.rtsp import RtspSource, _ReconnectLoop
from sentinel_ai.ports.frame_source import FrameSource


def test_rtsp_source_satisfies_its_port() -> None:
    assert issubclass(RtspSource, FrameSource)


async def test_reconnect_loop_backs_off_exponentially_and_caps_at_max() -> None:
    sleeps: list[float] = []
    attempts = 0

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise ConnectionError("simulated failure")
        # 4th attempt "succeeds" — the loop stops on the next should_stop check.

    loop = _ReconnectLoop(connect_once, initial_seconds=1.0, max_seconds=3.0, sleep=fake_sleep)
    await loop.run_forever(should_stop=lambda: attempts >= 4)

    assert sleeps == pytest.approx([1.0, 2.0, 3.0])  # doubling, capped at max_seconds=3.0
    assert loop.backoff_log == pytest.approx([1.0, 2.0, 3.0])


async def test_reconnect_loop_calls_on_reconnect_once_per_retry() -> None:
    calls = 0

    def on_reconnect() -> None:
        nonlocal calls
        calls += 1

    attempts = 0

    async def connect_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise ConnectionError("simulated failure")

    async def fake_sleep(_seconds: float) -> None:
        return None

    loop = _ReconnectLoop(
        connect_once, initial_seconds=1.0, max_seconds=30.0, sleep=fake_sleep,
        on_reconnect=on_reconnect,
    )
    await loop.run_forever(should_stop=lambda: attempts >= 3)
    assert calls == 2


def test_on_reconnect_resets_the_frame_index_counter() -> None:
    """Constructing RtspSource for real starts a background task that opens a
    real (here, nonexistent) RTSP URL — not CI-safe. `__new__` bypasses
    `__init__` to unit-test this one state transition in isolation.
    """
    source = RtspSource.__new__(RtspSource)
    source._frame_index = 42
    source._on_reconnect()
    assert source._frame_index == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/sources/test_rtsp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.sources.rtsp'`

- [ ] **Step 3: Implement `RtspSource`**

```python
# ai-engine/sentinel_ai/adapters/sources/rtsp.py
"""RTSP frame source with exponential-backoff reconnect (spec §5.1, §6).

Implements `FrameSource` (S3) exactly as `FileSource` does: one PyAV demux
loop fans out to a decoded `FrameData` stream (`__aiter__`) and an encoded
`EncodedPacket` stream (`packets()`), so the clip path never decodes and the
inference path never re-demuxes. The only genuinely new logic here is the
reconnect loop, split out into `_ReconnectLoop` — a class with no PyAV or
asyncio-queue state of its own, so its retry/backoff sequencing is
unit-testable with fakes: no network, no real sleep, no GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import av

from sentinel_ai.ports.frame_source import EncodedPacket, FrameData, FrameSource

logger = logging.getLogger(__name__)


class _CloseSentinel:
    """A dedicated sentinel type, not `None` or `object()`, so `isinstance`
    narrows the queue's union type for mypy strict."""

    __slots__ = ()


_CLOSE = _CloseSentinel()


class _ReconnectLoop:
    """Retries `connect_once` with exponential backoff; owns no I/O itself."""

    def __init__(
        self,
        connect_once: Callable[[], Awaitable[None]],
        *,
        initial_seconds: float,
        max_seconds: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        self._connect_once = connect_once
        self._initial = initial_seconds
        self._max = max_seconds
        self._sleep = sleep
        self._on_reconnect = on_reconnect
        self.backoff_log: list[float] = []

    async def run_forever(self, should_stop: Callable[[], bool]) -> None:
        backoff = self._initial
        while not should_stop():
            try:
                await self._connect_once()
                backoff = self._initial
            except Exception as exc:  # noqa: BLE001 — any connect/stream failure retries
                if should_stop():
                    return
                logger.warning("stream error (%s); reconnecting in %.1fs", exc, backoff)
                self.backoff_log.append(backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2.0, self._max)
                if self._on_reconnect is not None:
                    self._on_reconnect()


class RtspSource(FrameSource):
    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        reconnect_initial_seconds: float,
        reconnect_max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._camera_id = camera_id
        self._url = url
        self._clock = clock
        self._frame_index = 0
        self._closed = False
        self._frame_queue: asyncio.Queue[FrameData | _CloseSentinel] = asyncio.Queue(maxsize=8)
        self._packet_queue: asyncio.Queue[EncodedPacket | _CloseSentinel] = asyncio.Queue(
            maxsize=64
        )
        self._reconnect = _ReconnectLoop(
            self._connect_once,
            initial_seconds=reconnect_initial_seconds,
            max_seconds=reconnect_max_seconds,
            on_reconnect=self._on_reconnect,
        )
        self._task = asyncio.create_task(
            self._reconnect.run_forever(should_stop=lambda: self._closed)
        )

    def _on_reconnect(self) -> None:
        """The discontinuity signal: a new connection restarts frame
        numbering from 0 while `clock()` keeps advancing. `CameraRunner`
        treats a `frame_index` that drops back to (or below) the previous
        value as a stream discontinuity and resets `MotionAnalyzer`,
        `GateState`, and the tracker.
        """
        self._frame_index = 0

    async def _connect_once(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pump_stream, loop)

    def _pump_stream(self, loop: asyncio.AbstractEventLoop) -> None:
        container = av.open(self._url, options={"rtsp_transport": "tcp"})
        try:
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                if self._closed:
                    return
                if packet.dts is None:
                    continue  # flush packet at end-of-stream; nothing to decode
                arrival = self._clock()
                encoded = EncodedPacket(
                    camera_id=self._camera_id,
                    data=bytes(packet),
                    pts=arrival,
                    is_keyframe=bool(packet.is_keyframe),
                    codec=stream.codec_context.name,
                )
                asyncio.run_coroutine_threadsafe(self._packet_queue.put(encoded), loop).result()
                for decoded in packet.decode():
                    pixels = decoded.to_ndarray(format="rgb24")
                    frame = FrameData(
                        camera_id=self._camera_id,
                        frame_index=self._frame_index,
                        timestamp=arrival,
                        width=pixels.shape[1],
                        height=pixels.shape[0],
                        pixels=pixels,
                    )
                    asyncio.run_coroutine_threadsafe(self._frame_queue.put(frame), loop).result()
                    self._frame_index += 1
            raise ConnectionError(f"RTSP stream ended: {self._url}")
        finally:
            container.close()

    def __aiter__(self) -> AsyncIterator[FrameData]:
        async def frames() -> AsyncIterator[FrameData]:
            while True:
                item = await self._frame_queue.get()
                if isinstance(item, _CloseSentinel):
                    return
                yield item

        return frames()

    def packets(self) -> AsyncIterator[EncodedPacket]:
        async def pkts() -> AsyncIterator[EncodedPacket]:
            while True:
                item = await self._packet_queue.get()
                if isinstance(item, _CloseSentinel):
                    return
                yield item

        return pkts()

    async def close(self) -> None:
        self._closed = True
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        await self._frame_queue.put(_CLOSE)
        await self._packet_queue.put(_CLOSE)
```

- [ ] **Step 4: Verify RtspSource**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest tests/adapters/sources/test_rtsp.py -v -q -m "not gpu and not integration" && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`.

- [ ] **Step 5: Dataset script and provenance note**

```bash
# datasets/download_sample.sh
#!/usr/bin/env bash
# Downloads one CUHK Avenue test clip for the Phase 1B GPU demo.
#
# Source: CUHK Avenue Dataset — Lu, C., Shi, J., & Jia, J. (2013). "Abnormal
# Event Detection at 150 FPS in Matlab." ICCV 2013.
# http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
#
# Licence: see datasets/README.md for the full verification trail. Summary:
# the host page publishes no separate licence file but is consistently
# described, by the authors' own citation request and by every academic
# redistribution checked, as released for academic research use. This
# script's use — local dev/demo footage, never redistributed — fits inside
# that grant. Cite the ICCV 2013 paper above if this footage or a
# description of it appears in any publication.
#
# Ethical constraint (spec §5): deliberately-public, licensed research
# footage only — never a feed that indexes an unsecured private camera.
# Do not point this script at a different dataset without repeating this
# verification in datasets/README.md.

set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${DATASET_DIR}/avenue"
ZIP_URL="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip"
ZIP_PATH="${WORK_DIR}/Avenue_Dataset.zip"
OUTPUT_MP4="${WORK_DIR}/avenue_01.mp4"

if [[ -f "${OUTPUT_MP4}" ]]; then
  echo "Already present: ${OUTPUT_MP4}"
  exit 0
fi

mkdir -p "${WORK_DIR}"

echo "Downloading Avenue Dataset (~780 MiB; the full 16-training/21-testing"
echo "archive — only one testing clip is kept) from CUHK..."
curl -fL --retry 3 -o "${ZIP_PATH}" "${ZIP_URL}"

echo "Extracting the first testing clip..."
UNZIP_DIR="${WORK_DIR}/extracted"
mkdir -p "${UNZIP_DIR}"
unzip -o "${ZIP_PATH}" -d "${UNZIP_DIR}" >/dev/null

SOURCE_AVI="$(find "${UNZIP_DIR}" -ipath "*testing_videos*" -iname "01.avi" | head -n1)"
if [[ -z "${SOURCE_AVI}" ]]; then
  echo "Could not locate testing video 01 inside the archive." >&2
  echo "Inspect ${UNZIP_DIR} and adjust the find pattern above." >&2
  exit 1
fi

echo "Re-encoding to H.264/yuv420p for the RTSP/remux pipeline (the archive"
echo "ships Xvid-encoded AVI, which mediamtx/RTSP publishing does not want)..."
ffmpeg -y -i "${SOURCE_AVI}" -c:v libx264 -profile:v baseline -pix_fmt yuv420p \
  -movflags +faststart "${OUTPUT_MP4}"

echo "Cleaning up the archive (kept: ${OUTPUT_MP4})..."
rm -rf "${ZIP_PATH}" "${UNZIP_DIR}"

echo "Done: ${OUTPUT_MP4}"
```

```markdown
# datasets/README.md

# Sample footage

This directory is gitignored except for this file, `download_sample.sh`, and
`.gitkeep` — the footage itself is never committed (spec §8 / Phase 1B §8).

## Source

**CUHK Avenue Dataset** — Lu, C., Shi, J., & Jia, J. (2013). *Abnormal Event
Detection at 150 FPS in Matlab.* ICCV 2013.
http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

## Licence — verified before the URL was hardcoded

The dataset's own page publishes no separate licence file. What is
verifiable:

- It is consistently described, across every academic redistribution
  checked (e.g. the Anomalib datamodule documentation), as released for
  **academic research use**. No commercial-use grant is published anywhere.
- The paper accompanying it (cited above) is the expected citation if this
  footage, or a description of it, appears in any publication.
- The download URL used by `download_sample.sh`
  (`http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip`)
  was confirmed to resolve directly against the CUHK host — not a
  third-party mirror — returning `Content-Type: application/zip`,
  813,227,845 bytes.

SentinelAI's use here is non-commercial: one clip, used as local
development/demo footage for a research project, looped through a private
mediamtx instance and never redistributed or re-published. That fits inside
the academic-research grant described above.

**Do not point this script at a different dataset without repeating this
verification.** A dataset without a published academic-research grant, or
one sourced from an indexed feed of unsecured private cameras, is out of
scope regardless of convenience.

## Ethical constraint (binding, from spec §5)

Only deliberately-public, licensed research footage is used for
verification. Nothing that indexes unsecured private cameras (e.g. via
Shodan-style scanners) is ever a source, for testing or otherwise.

## What the script does

`download_sample.sh` downloads the full Avenue archive, extracts a single
testing clip (`testing_videos/01.avi`), re-encodes it to H.264/yuv420p MP4,
and deletes the archive and everything else extracted from it. Output:
`datasets/avenue/avenue_01.mp4`.

Run once, from the repo root or `ai-engine/`:

    bash datasets/download_sample.sh

Requires `curl`, `unzip`, and `ffmpeg` on `PATH`.
```

- [ ] **Step 6: Automated smoke test — the CI-impossible parts, minus mediamtx**

```python
# ai-engine/tests/e2e/__init__.py
```

```python
# ai-engine/tests/e2e/test_demo_smoke.py
"""One real, end-to-end pass with real models over the downloaded Avenue
clip — no mediamtx, no RTSP, no RabbitMQ, no MinIO, so it only needs the GPU
extra and the dataset script's output, not the full docker-compose stack.
This is the closest thing to an automated version of the manual demo below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

AVENUE_CLIP = Path(__file__).resolve().parents[3] / "datasets" / "avenue" / "avenue_01.mp4"


@pytest.mark.gpu
@pytest.mark.integration
async def test_full_pipeline_produces_a_real_event_with_real_models() -> None:
    if not AVENUE_CLIP.exists():
        pytest.skip(f"run datasets/download_sample.sh first — {AVENUE_CLIP} not found")

    from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, select_device
    from sentinel_ai.adapters.sources.file import FileSource
    from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
    from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber
    from sentinel_ai.domain.camera_profile import CameraProfile
    from sentinel_ai.orchestrator.admission import AdmissionGate
    from sentinel_ai.orchestrator.scheduler import VlmScheduler
    from sentinel_ai.pipeline.runner import CameraRunner
    from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
    from tests.fakes.io import FakePublisher

    device = select_device()
    detector = Yolo11Detector(model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device=device)
    vlm = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=256, device=device
    )
    await detector.initialize()
    await detector.warmup()
    await vlm.initialize()
    await vlm.warmup()

    publisher = FakePublisher()
    admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
    scheduler = VlmScheduler(
        vlm, publisher, admission, maxsize=4, timeout_seconds=30.0, clock=__import__("time").monotonic
    )
    worker = __import__("asyncio").create_task(scheduler.run())

    source = FileSource(str(AVENUE_CLIP), camera_id="avenue_01", realtime=False)
    runner = CameraRunner(
        camera_id="avenue_01",
        camera_label="Avenue (demo)",
        source=source,
        detector=detector,
        tracker=ByteTrackTracker(frame_rate=25),
        motion=MotionAnalyzer(),
        profile=CameraProfile(camera_id="avenue_01"),
        scheduler=scheduler,
        clip_writer=None,
        preroll=__import__("sentinel_ai.adapters.sources.preroll", fromlist=["PreRollBuffer"]).PreRollBuffer(
            preroll_seconds=3.0
        ),
        clock=__import__("time").monotonic,
    )
    await runner.run()
    await scheduler.drain()
    worker.cancel()

    assert publisher.events, "no event was produced over the whole clip"
    event = publisher.events[0]
    assert event.description
    assert 0.0 <= event.threat.value <= 1.0

    await detector.shutdown()
    await vlm.shutdown()
```

- [ ] **Step 7: The manual end-to-end demo**

Not automated beyond Step 6's smoke test — this exercises mediamtx, RabbitMQ, and MinIO, none of which the smoke test touches. Run on the GPU box with the `gpu` extra installed and `docker-compose.core.yml` (Task 2) available.

```bash
# 1. Fetch the clip once.
bash datasets/download_sample.sh

# 2. Start core services (Postgres, Redis, RabbitMQ, MinIO, mediamtx).
docker compose -f deploy/compose/docker-compose.core.yml up -d

# 3. Loop the clip into mediamtx as the avenue_01 RTSP path.
ffmpeg -stream_loop -1 -re -i datasets/avenue/avenue_01.mp4 \
  -c copy -f rtsp rtsp://localhost:8554/avenue_01

# 4. In another shell, start the engine (Development Mode is the default —
#    SENTINEL_MODE need not be set). Point the avenue_01 camera's source URL
#    at rtsp://localhost:8554/avenue_01 via whatever camera configuration
#    Task 7/10 already exposes.
cd ai-engine && . .venv/bin/activate
uvicorn sentinel_ai.api.app:app --host 0.0.0.0 --port 8000
```

Verification checklist:

- [ ] `curl http://localhost:8000/cameras/avenue_01/telemetry` shows `escalations` increment over a couple of minutes.
- [ ] RabbitMQ: open `http://localhost:15672` (or `rabbitmqctl list_queues`) and confirm messages have arrived on the `sentinel.events` exchange/queue; alternatively run a one-off `aio-pika` consumer to print one decoded payload and confirm `description` is non-empty prose (not a placeholder) and `threat_score` is in `[0, 1]`.
- [ ] MinIO: open `http://localhost:9001` (or `mc ls localminio/sentinel-clips/avenue_01/`) and confirm a new `.mp4` object exists for the event's `clip_uri`.
- [ ] Download that clip and run `ffprobe -show_entries format=duration -of default=noprint_wrappers=1 <clip>.mp4` — expect roughly 8–10 s (3–5 s quantised pre-roll + 5 s post-roll per spec §5.5), and play it to confirm the pre-roll segment shows activity *before* whatever triggered the escalation, not starting at the trigger frame.

- [ ] **Step 8: `ai-engine/README.md`**

```markdown
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

docker compose -f ../deploy/compose/docker-compose.core.yml up -d   # Postgres, Redis, RabbitMQ, MinIO, mediamtx

uvicorn sentinel_ai.api.app:app --host 0.0.0.0 --port 8000
```

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
deviation: this box's 20 CPU cores comfortably absorb software decode at
this scale, and the ~0.3 GB of VRAM a hardware decode surface would have
used instead goes toward the VRAM budget (spec §2.1). `decode_hwaccel`
(`sentinel_ai/config.py`, currently `None`) is the escape hatch if a future
deployment needs it — no source adapter branches on it yet, so setting it
today has no effect; it exists so re-enabling hardware decode is a
configuration change, not a rewrite.

One consequence of encoded (not raw) pre-roll buffering worth knowing:
`PreRollBuffer.flush()` walks back to the oldest keyframe at or before the
`clip_preroll_seconds` horizon, because a clip that starts mid-GOP is
undecodable. With a typical 2 s GOP, a requested 3.0 s pre-roll yields
3.0–5.0 s in the actual clip — always more context, never less.

## Datasets

`datasets/download_sample.sh` fetches one CUHK Avenue clip for GPU
verification and the end-to-end demo; see `datasets/README.md` for
provenance and licence. The footage itself is never committed.
```

- [ ] **Step 9: Verify**

Run: `cd ai-engine && . .venv/bin/activate && python -m pytest -q -m "not gpu and not integration" -W error && ruff check . && ruff format --check . && mypy`
Expected: all PASS, ruff clean, mypy `Success`, no warnings. (Step 6's smoke test and Step 7's manual demo both require the GPU box, the dataset script's output, and — for Step 7 — the full compose stack; neither runs here.)

- [ ] **Step 10: Commit**

```bash
git add ai-engine/sentinel_ai/adapters/sources/rtsp.py ai-engine/tests/adapters/sources/test_rtsp.py \
  datasets/download_sample.sh datasets/README.md \
  ai-engine/tests/e2e/ ai-engine/README.md
git commit -m "feat(adapters): add RtspSource, dataset download script, and the end-to-end demo

Reconnect backoff is a separately-testable unit (_ReconnectLoop) so its
sequencing runs in CI without network, sleep, or GPU. Adds the CUHK Avenue
download script with its licence verification trail, a GPU+integration
smoke test that proves the model trio end-to-end without mediamtx, and the
manual RTSP/RabbitMQ/MinIO demo checklist. Documents decode_hwaccel and the
Development/Production seam in the engine's first README."
```

---


