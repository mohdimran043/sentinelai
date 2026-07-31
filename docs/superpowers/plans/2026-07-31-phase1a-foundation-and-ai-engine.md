# SentinelAI Phase 1A — Foundation & AI Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the monorepo foundation and a working AI Engine that ingests RTSP, runs YOLO11 → ByteTrack → escalation gate → Qwen2.5-VL only when warranted, and publishes validated events to RabbitMQ.

**Architecture:** Clean architecture with a pure `domain/` layer (no I/O, no CUDA), abstract `ports/`, and swappable `adapters/`. The escalation gate and VRAM planner are pure functions, so the novel algorithms are fully unit-testable on CPU in CI. A single `ModelRuntime` ABC has two transports — in-process (Development Mode) and gRPC (Production Mode) — selected by one config value.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pydantic-settings, PyAV, Ultralytics (YOLO11), supervision (ByteTrack), transformers + autoawq (Qwen2.5-VL-3B-Instruct-AWQ), aio-pika, MinIO SDK, pytest, ruff, mypy, Docker Compose, mediamtx.

**Reference spec:** `docs/superpowers/specs/2026-07-31-sentinelai-phase0-1-design.md`

## Global Constraints

- Python **3.12** exactly; `ai-engine/` targets `py312` in ruff/mypy config.
- `sentinel_ai/domain/` and `sentinel_ai/ports/` MUST NOT import any third-party I/O library (no torch, cv2, av, aio_pika, fastapi). Enforced by an import-lint test in Task 1.
- Development Mode is the **default** (`SENTINEL_MODE=development`).
- Default Vision LLM: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`. 7B must never be a default.
- VRAM ceiling: **8192 MiB**. Reserved headroom: **2048 MiB**.
- Every test that requires a GPU MUST be marked `@pytest.mark.gpu`; CI runs `-m "not gpu"`.
- Escalation defaults (from spec §4.1), all on `CameraProfile`: `min_track_frames=8`, `scene_delta_threshold=0.35`, `scene_delta_frames=5`, `dwell_radius_px=48.0`, `dwell_seconds=30.0`, `speed_percentile=0.95`, `speed_fallback_px_s=180.0`, `track_count_baseline=6`, `summary_interval_seconds=45.0`.
- Token bucket defaults: `capacity=2`, `refill_seconds=10.0`.
- VLM idle-unload: **600 s**.
- `salient_classes` default: `{person, car, truck, bus, motorcycle, bicycle, backpack, handbag, suitcase}`.
- Clip pre-roll: **3.0 s** before the event, per spec §8.
- All timestamps in the domain layer are **float seconds from a monotonic clock**, passed in explicitly. `domain/` MUST NOT call `time.time()` or `time.monotonic()` — clocks are arguments. Enforced by the same import-lint test.
- Commit after every task. Conventional Commits format.

**Deviation from spec to note:** spec §4.3 assumes NVDEC decode (~0.3 GB VRAM). This plan implements PyAV **CPU decode** as the default (20 CPU cores are ample for the slice, and it avoids a hard dependency on a CUDA-enabled ffmpeg build), with `hwaccel` as a config flag. This *frees* 0.3 GB VRAM rather than consuming it. Task 9 documents the flag.

## File Structure

| File | Responsibility |
|---|---|
| `ai-engine/pyproject.toml` | Deps, ruff/mypy/pytest config |
| `ai-engine/sentinel_ai/config.py` | `Settings` (pydantic-settings), `Mode` enum |
| `ai-engine/sentinel_ai/domain/entities.py` | `BBox`, `Detection`, `Track`, `SceneState`, `ThreatScore`, `Event` |
| `ai-engine/sentinel_ai/domain/camera_profile.py` | `CameraProfile` + all threshold defaults |
| `ai-engine/sentinel_ai/domain/policy/rate_budget.py` | `TokenBucket` (pure, immutable) |
| `ai-engine/sentinel_ai/domain/policy/triggers.py` | The 7 escalation predicates |
| `ai-engine/sentinel_ai/domain/policy/escalation.py` | `GateState`, `decide()` — composition + governor + dedup |
| `ai-engine/sentinel_ai/domain/policy/vram_budget.py` | `plan_residency()` |
| `ai-engine/sentinel_ai/ports/*.py` | ABCs: detector, tracker, vision_llm, model_runtime, frame_source, event_publisher, clip_writer |
| `ai-engine/sentinel_ai/adapters/detectors/yolo11.py` | Ultralytics YOLO11 → `Detection` |
| `ai-engine/sentinel_ai/adapters/trackers/bytetrack.py` | supervision ByteTrack → `Track` |
| `ai-engine/sentinel_ai/adapters/vision/qwen25vl_awq.py` | Qwen2.5-VL-3B-AWQ |
| `ai-engine/sentinel_ai/adapters/sources/` | `file.py`, `rtsp.py` + `PreRollBuffer` |
| `ai-engine/sentinel_ai/adapters/publishers/` | `rabbitmq.py` (disk-buffered), `inmemory.py` |
| `ai-engine/sentinel_ai/adapters/storage/minio_clips.py` | Clip writer |
| `ai-engine/sentinel_ai/orchestrator/` | `registry.py`, `resident_set.py`, `scheduler.py`, `service.py` |
| `ai-engine/sentinel_ai/pipeline/` | `runner.py`, `stages/motion.py` |
| `ai-engine/sentinel_ai/api/` | FastAPI app — thin delegation only |
| `ai-engine/tests/fakes/` | `FakeDetector`, `FakeTracker`, `FakeVisionLLM`, `FakeSource`, `FakePublisher` |
| `contracts/events/event.schema.json` | Event payload contract |
| `contracts/proto/sentinel/model/v1/model.proto` | §10 model interface |
| `deploy/compose/docker-compose.core.yml` | Postgres, Redis, RabbitMQ, MinIO, mediamtx |
| `deploy/compose/mediamtx.yml` | Source registry |
| `Makefile`, `.github/workflows/ci.yml` | Dev commands, CI |

---

### Task 1: Project scaffolding, tooling, and the architecture-fitness test

**Files:**
- Create: `ai-engine/pyproject.toml`, `ai-engine/sentinel_ai/__init__.py`, `ai-engine/sentinel_ai/config.py`
- Create: `ai-engine/tests/__init__.py`, `ai-engine/tests/test_architecture.py`, `ai-engine/tests/test_config.py`
- Create: `Makefile`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `sentinel_ai.config.Settings` with fields `mode: Mode`, `vram_total_mib: int`, `vram_reserved_mib: int`, `vlm_model_id: str`, `vlm_idle_unload_seconds: float`, `decode_hwaccel: str | None`; `sentinel_ai.config.Mode` enum with members `DEVELOPMENT`, `PRODUCTION`; `get_settings() -> Settings`

- [ ] **Step 1: Write the failing architecture-fitness test**

This is the mechanical enforcement of the spec's purity rule. It must exist before any domain code.

```python
# ai-engine/tests/test_architecture.py
"""Architecture fitness functions — these encode spec §3.3 and §6.1 as tests."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "sentinel_ai"

FORBIDDEN_IN_PURE_LAYERS = {
    "torch", "torchvision", "cv2", "av", "numpy", "ultralytics", "supervision",
    "transformers", "aio_pika", "pika", "fastapi", "minio", "httpx", "requests",
    "grpc", "sqlalchemy", "redis",
}

PURE_LAYERS = ("domain", "ports")


def _module_files(layer: str) -> list[Path]:
    return sorted((PACKAGE_ROOT / layer).rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_have_no_io_dependencies(layer: str) -> None:
    offenders: dict[str, set[str]] = {}
    for path in _module_files(layer):
        bad = _imported_roots(path) & FORBIDDEN_IN_PURE_LAYERS
        if bad:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layer '{layer}' imports I/O libraries: {offenders}"


@pytest.mark.parametrize("layer", PURE_LAYERS)
def test_pure_layers_do_not_read_the_clock(layer: str) -> None:
    """Clocks must be arguments so policy is deterministic under test."""
    offenders: dict[str, list[str]] = {}
    for path in _module_files(layer):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = [
            f"{node.func.value.id}.{node.func.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
            and node.func.attr in {"time", "monotonic", "monotonic_ns", "time_ns"}
        ]
        if hits:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = hits
    assert offenders == {}, f"pure layer '{layer}' reads the clock: {offenders}"


def test_domain_does_not_import_adapters_or_orchestrator() -> None:
    offenders: dict[str, set[str]] = {}
    for layer in PURE_LAYERS:
        for path in _module_files(layer):
            text = path.read_text(encoding="utf-8")
            bad = {
                name
                for name in ("adapters", "orchestrator", "pipeline", "api")
                if f"sentinel_ai.{name}" in text
            }
            if bad:
                offenders[str(path.relative_to(PACKAGE_ROOT))] = bad
    assert offenders == {}, f"pure layers depend on outer layers: {offenders}"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd ai-engine && python -m pytest tests/test_architecture.py -v`
Expected: FAIL — `ModuleNotFoundError` / collection error, because `sentinel_ai/` does not exist yet.

- [ ] **Step 3: Create the package skeleton and project config**

```toml
# ai-engine/pyproject.toml
[project]
name = "sentinel-ai"
version = "0.1.0"
description = "SentinelAI AI Engine — behaviour intelligence inference orchestrator"
requires-python = "==3.12.*"
dependencies = [
    "pydantic>=2.9",
    "pydantic-settings>=2.5",
]

[project.optional-dependencies]
runtime = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.31",
    "av>=13.0",
    "numpy>=1.26,<2.2",
    "aio-pika>=9.4",
    "minio>=7.2",
    "jsonschema>=4.23",
]
gpu = [
    "torch>=2.5",
    "ultralytics>=8.3",
    "supervision>=0.24",
    "transformers>=4.49",
    "accelerate>=1.0",
    "autoawq>=0.2.6",
    "qwen-vl-utils>=0.0.8",
]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.7",
    "mypy>=1.13",
]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["sentinel_ai*"]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["."]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "ANN", "RUF"]
ignore = ["ANN401"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["ANN"]

[tool.mypy]
python_version = "3.12"
strict = true
warn_unreachable = true
disallow_untyped_defs = true
files = ["sentinel_ai"]

[[tool.mypy.overrides]]
module = ["ultralytics.*", "supervision.*", "av.*", "autoawq.*", "qwen_vl_utils.*", "minio.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "gpu: requires a CUDA GPU and downloaded model weights (deselect with -m 'not gpu')",
    "integration: requires docker-compose core services running",
]
```

```python
# ai-engine/sentinel_ai/__init__.py
"""SentinelAI AI Engine."""

__version__ = "0.1.0"
```

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
    model_config = SettingsConfigDict(
        env_prefix="SENTINEL_", env_file=".env", extra="ignore"
    )

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

Also create empty `ai-engine/sentinel_ai/domain/__init__.py`, `ai-engine/sentinel_ai/domain/policy/__init__.py`, `ai-engine/sentinel_ai/ports/__init__.py`, and `ai-engine/tests/__init__.py` so the fitness test finds the layers.

- [ ] **Step 4: Write the config test**

```python
# ai-engine/tests/test_config.py
from __future__ import annotations

from sentinel_ai.config import Mode, Settings


def test_development_is_the_default_mode() -> None:
    assert Settings().mode is Mode.DEVELOPMENT


def test_default_vlm_is_the_3b_awq_build() -> None:
    assert Settings().vlm_model_id == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


def test_vram_defaults_match_the_8gb_budget() -> None:
    settings = Settings()
    assert settings.vram_total_mib == 8192
    assert settings.vram_reserved_mib == 2048


def test_env_prefix_overrides_mode(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_MODE", "production")
    assert Settings().mode is Mode.PRODUCTION
```

- [ ] **Step 5: Install and run the full suite**

```bash
cd ai-engine
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -v -m "not gpu"
```

Expected: all tests PASS (6 architecture tests, 4 config tests).

- [ ] **Step 6: Add the Makefile and CI workflow**

```makefile
# Makefile
.PHONY: help install lint typecheck test test-gpu up down logs contracts

PY := cd ai-engine && . .venv/bin/activate &&

help:
	@grep -E '^[a-zA-Z-]+:.*' $(MAKEFILE_LIST) | sed 's/:.*//' | sort

install:
	cd ai-engine && python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev,runtime]'

lint:
	$(PY) ruff check . && ruff format --check .

typecheck:
	$(PY) mypy

test:
	$(PY) pytest -v -m "not gpu"

test-gpu:
	$(PY) pytest -v -m gpu

up:
	docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d

down:
	docker compose -f deploy/compose/docker-compose.core.yml --profile core down

logs:
	docker compose -f deploy/compose/docker-compose.core.yml logs -f
```

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  ai-engine:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: ai-engine
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e '.[dev,runtime]'
      - name: Lint
        run: ruff check . && ruff format --check .
      - name: Typecheck
        run: mypy
      - name: Test (CPU only)
        run: pytest -v -m "not gpu" --cov=sentinel_ai --cov-report=term-missing
```

- [ ] **Step 7: Verify lint, types, and tests all pass**

Run: `make lint && make typecheck && make test`
Expected: ruff clean, mypy `Success: no issues found`, pytest all PASS.

- [ ] **Step 8: Commit**

```bash
git add ai-engine Makefile .github
git commit -m "feat(ai-engine): scaffold project with architecture fitness tests

Encodes spec §3.3/§6.1 purity rules as executable tests: domain/ and ports/
cannot import I/O libraries, cannot read the clock, and cannot depend on
outer layers. These fail the build rather than relying on review discipline."
```

---

### Task 2: Domain entities

**Files:**
- Create: `ai-engine/sentinel_ai/domain/entities.py`
- Test: `ai-engine/tests/domain/test_entities.py`

**Interfaces:**
- Consumes: nothing
- Produces: `BBox(x1,y1,x2,y2)` with properties `cx: float`, `cy: float`, `area: float`; `Detection(label: str, confidence: float, box: BBox)`; `Track(track_id: int, label: str, box: BBox, age_frames: int, speed_px_s: float)`; `SceneState(camera_id: str, frame_index: int, timestamp: float, detections: tuple[Detection,...], tracks: tuple[Track,...], motion_energy: float, scene_signature: tuple[float,...])` with method `signature_delta(other: tuple[float,...] | None) -> float`; `EscalationReason` StrEnum with members `NEW_SALIENT_TRACK, SCENE_CHANGE, DWELL_EXCEEDED, SPEED_ANOMALY, TRACK_COUNT_SPIKE, PERIODIC_SUMMARY, USER_REQUESTED`; `Severity` StrEnum `INFO, LOW, MEDIUM, HIGH, CRITICAL`; `ThreatScore(value: float, severity: Severity)`; `Event(...)`

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/test_entities.py
from __future__ import annotations

import pytest

from sentinel_ai.domain.entities import (
    BBox,
    Detection,
    EscalationReason,
    SceneState,
    Severity,
    ThreatScore,
    Track,
)


def _scene(signature: tuple[float, ...]) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=(),
        motion_energy=0.0,
        scene_signature=signature,
    )


class TestBBox:
    def test_centroid_is_the_box_centre(self) -> None:
        box = BBox(10.0, 20.0, 30.0, 60.0)
        assert box.cx == 20.0
        assert box.cy == 40.0

    def test_area_is_width_times_height(self) -> None:
        assert BBox(0.0, 0.0, 4.0, 5.0).area == 20.0

    def test_inverted_box_has_zero_area_not_negative(self) -> None:
        assert BBox(10.0, 10.0, 4.0, 4.0).area == 0.0

    def test_is_immutable(self) -> None:
        box = BBox(0.0, 0.0, 1.0, 1.0)
        with pytest.raises(AttributeError):
            box.x1 = 5.0  # type: ignore[misc]


class TestSignatureDelta:
    def test_identical_signatures_have_zero_delta(self) -> None:
        assert _scene((0.5, 0.5)).signature_delta((0.5, 0.5)) == 0.0

    def test_missing_previous_signature_is_treated_as_no_change(self) -> None:
        """First frame must not fire SceneChange — there is nothing to compare to."""
        assert _scene((0.5, 0.5)).signature_delta(None) == 0.0

    def test_disjoint_signatures_have_delta_of_one(self) -> None:
        assert _scene((1.0, 0.0)).signature_delta((0.0, 1.0)) == pytest.approx(1.0)

    def test_delta_is_half_l1_distance(self) -> None:
        # |0.6-0.4| + |0.4-0.6| = 0.4 ; halved = 0.2
        assert _scene((0.6, 0.4)).signature_delta((0.4, 0.6)) == pytest.approx(0.2)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="length"):
            _scene((0.5, 0.5)).signature_delta((0.3, 0.3, 0.4))


class TestSceneStateHelpers:
    def test_tracks_of_returns_only_matching_labels(self) -> None:
        box = BBox(0.0, 0.0, 1.0, 1.0)
        scene = SceneState(
            camera_id="cam-1",
            frame_index=3,
            timestamp=1.0,
            detections=(Detection("person", 0.9, box), Detection("dog", 0.8, box)),
            tracks=(
                Track(1, "person", box, age_frames=10, speed_px_s=0.0),
                Track(2, "dog", box, age_frames=4, speed_px_s=5.0),
            ),
            motion_energy=0.1,
            scene_signature=(1.0,),
        )
        assert [t.track_id for t in scene.tracks_of({"person"})] == [1]


class TestThreatScore:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, Severity.INFO),
            (0.19, Severity.INFO),
            (0.2, Severity.LOW),
            (0.4, Severity.MEDIUM),
            (0.6, Severity.HIGH),
            (0.8, Severity.CRITICAL),
            (1.0, Severity.CRITICAL),
        ],
    )
    def test_severity_is_derived_from_value(self, value: float, expected: Severity) -> None:
        assert ThreatScore.from_value(value).severity is expected

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_out_of_range_values_are_rejected(self, value: float) -> None:
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            ThreatScore.from_value(value)


def test_escalation_reasons_cover_all_seven_spec_triggers() -> None:
    assert {r.value for r in EscalationReason} == {
        "new_salient_track",
        "scene_change",
        "dwell_exceeded",
        "speed_anomaly",
        "track_count_spike",
        "periodic_summary",
        "user_requested",
    }
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/test_entities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.entities'`

- [ ] **Step 3: Implement the entities**

```python
# ai-engine/sentinel_ai/domain/entities.py
"""Core domain entities. Pure — no I/O, no clock reads, no third-party runtime deps."""
from __future__ import annotations

from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from uuid import UUID


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    confidence: float
    box: BBox


@dataclass(frozen=True, slots=True)
class Track:
    """A detection associated across frames by the tracker."""

    track_id: int
    label: str
    box: BBox
    age_frames: int
    speed_px_s: float


@dataclass(frozen=True, slots=True)
class SceneState:
    """Everything the escalation gate is allowed to look at.

    Deliberately contains no pixel data: the gate must be decidable from cheap
    signals alone (spec §4.1). Keyframe bytes are fetched only after escalation.
    """

    camera_id: str
    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...]
    tracks: tuple[Track, ...]
    motion_energy: float
    scene_signature: tuple[float, ...]

    def signature_delta(self, previous: tuple[float, ...] | None) -> float:
        """Total-variation distance between two normalised histograms, in [0, 1].

        A missing previous signature yields 0.0 so the first frame of a stream
        never fires SceneChange.
        """
        if previous is None:
            return 0.0
        if len(previous) != len(self.scene_signature):
            raise ValueError(
                f"signature length mismatch: {len(self.scene_signature)} vs {len(previous)}"
            )
        return sum(abs(a - b) for a, b in zip(self.scene_signature, previous, strict=True)) / 2.0

    def tracks_of(self, labels: Collection[str]) -> Iterator[Track]:
        return (track for track in self.tracks if track.label in labels)


class EscalationReason(StrEnum):
    """The seven triggers from spec §4.1."""

    NEW_SALIENT_TRACK = "new_salient_track"
    SCENE_CHANGE = "scene_change"
    DWELL_EXCEEDED = "dwell_exceeded"
    SPEED_ANOMALY = "speed_anomaly"
    TRACK_COUNT_SPIKE = "track_count_spike"
    PERIODIC_SUMMARY = "periodic_summary"
    USER_REQUESTED = "user_requested"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_BANDS: tuple[tuple[float, Severity], ...] = (
    (0.8, Severity.CRITICAL),
    (0.6, Severity.HIGH),
    (0.4, Severity.MEDIUM),
    (0.2, Severity.LOW),
)


@dataclass(frozen=True, slots=True)
class ThreatScore:
    value: float
    severity: Severity

    @classmethod
    def from_value(cls, value: float) -> ThreatScore:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threat score must be between 0.0 and 1.0, got {value}")
        for threshold, severity in _SEVERITY_BANDS:
            if value >= threshold:
                return cls(value=value, severity=severity)
        return cls(value=value, severity=Severity.INFO)


@dataclass(frozen=True, slots=True)
class Event:
    """The unit published to the Web Platform. Carries no model identity (spec §3.3)."""

    event_id: UUID
    camera_id: str
    occurred_at: float
    reason: EscalationReason
    threat: ThreatScore
    description: str
    suggested_action: str
    labels: tuple[str, ...] = ()
    track_ids: tuple[int, ...] = ()
    keyframe_uri: str | None = None
    clip_uri: str | None = None
    description_unavailable: bool = False
    metadata: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/test_entities.py -v && mypy`
Expected: all PASS; mypy `Success`.

Note: create `ai-engine/tests/domain/__init__.py` if collection fails.

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/domain/entities.py ai-engine/tests/domain
git commit -m "feat(domain): add core entities with pure signature-delta metric

SceneState deliberately excludes pixel data so the escalation gate is
decidable from cheap signals alone (spec §4.1)."
```

---

### Task 3: CameraProfile with spec-mandated defaults

**Files:**
- Create: `ai-engine/sentinel_ai/domain/camera_profile.py`
- Test: `ai-engine/tests/domain/test_camera_profile.py`

**Interfaces:**
- Consumes: nothing
- Produces: `DEFAULT_SALIENT_CLASSES: frozenset[str]`; `CameraProfile` frozen dataclass with fields `camera_id: str`, `salient_classes: frozenset[str]`, `min_track_frames: int`, `scene_delta_threshold: float`, `scene_delta_frames: int`, `dwell_radius_px: float`, `dwell_seconds: float`, `speed_percentile: float`, `speed_fallback_px_s: float`, `track_count_baseline: int`, `summary_interval_seconds: float`, `bucket_capacity: int`, `bucket_refill_seconds: float`, `vlm_enabled: bool`

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/test_camera_profile.py
from __future__ import annotations

import pytest

from sentinel_ai.domain.camera_profile import DEFAULT_SALIENT_CLASSES, CameraProfile


def test_defaults_match_the_spec_table() -> None:
    profile = CameraProfile(camera_id="cam-1")
    assert profile.min_track_frames == 8
    assert profile.scene_delta_threshold == 0.35
    assert profile.scene_delta_frames == 5
    assert profile.dwell_radius_px == 48.0
    assert profile.dwell_seconds == 30.0
    assert profile.speed_percentile == 0.95
    assert profile.speed_fallback_px_s == 180.0
    assert profile.track_count_baseline == 6
    assert profile.summary_interval_seconds == 45.0
    assert profile.bucket_capacity == 2
    assert profile.bucket_refill_seconds == 10.0
    assert profile.vlm_enabled is True


def test_default_salient_classes_match_the_spec() -> None:
    assert DEFAULT_SALIENT_CLASSES == frozenset(
        {
            "person", "car", "truck", "bus", "motorcycle",
            "bicycle", "backpack", "handbag", "suitcase",
        }
    )
    assert CameraProfile(camera_id="cam-1").salient_classes is DEFAULT_SALIENT_CLASSES


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("min_track_frames", 0, "min_track_frames"),
        ("scene_delta_threshold", 1.5, "scene_delta_threshold"),
        ("scene_delta_frames", 0, "scene_delta_frames"),
        ("dwell_radius_px", -1.0, "dwell_radius_px"),
        ("dwell_seconds", 0.0, "dwell_seconds"),
        ("speed_percentile", 1.0, "speed_percentile"),
        ("track_count_baseline", 0, "track_count_baseline"),
        ("summary_interval_seconds", 0.0, "summary_interval_seconds"),
        ("bucket_capacity", 0, "bucket_capacity"),
        ("bucket_refill_seconds", 0.0, "bucket_refill_seconds"),
    ],
)
def test_invalid_values_are_rejected_at_construction(
    field_name: str, bad_value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        CameraProfile(camera_id="cam-1", **{field_name: bad_value})


def test_profile_is_immutable() -> None:
    profile = CameraProfile(camera_id="cam-1")
    with pytest.raises(AttributeError):
        profile.dwell_seconds = 5.0  # type: ignore[misc]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/test_camera_profile.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.camera_profile'`

- [ ] **Step 3: Implement CameraProfile**

```python
# ai-engine/sentinel_ai/domain/camera_profile.py
"""Per-camera tuning. Slice 1 uses fixed defaults; Phase 4 replaces the
percentile and baseline fields with learned per-camera values."""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SALIENT_CLASSES: frozenset[str] = frozenset(
    {
        "person",
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "backpack",
        "handbag",
        "suitcase",
    }
)


@dataclass(frozen=True, slots=True)
class CameraProfile:
    camera_id: str
    salient_classes: frozenset[str] = DEFAULT_SALIENT_CLASSES

    min_track_frames: int = 8
    scene_delta_threshold: float = 0.35
    scene_delta_frames: int = 5
    dwell_radius_px: float = 48.0
    dwell_seconds: float = 30.0
    speed_percentile: float = 0.95
    speed_fallback_px_s: float = 180.0
    track_count_baseline: int = 6
    summary_interval_seconds: float = 45.0

    bucket_capacity: int = 2
    bucket_refill_seconds: float = 10.0

    vlm_enabled: bool = True

    def __post_init__(self) -> None:
        self._require(self.min_track_frames >= 1, "min_track_frames must be >= 1")
        self._require(
            0.0 < self.scene_delta_threshold <= 1.0,
            "scene_delta_threshold must be in (0.0, 1.0]",
        )
        self._require(self.scene_delta_frames >= 1, "scene_delta_frames must be >= 1")
        self._require(self.dwell_radius_px >= 0.0, "dwell_radius_px must be >= 0")
        self._require(self.dwell_seconds > 0.0, "dwell_seconds must be > 0")
        self._require(
            0.0 < self.speed_percentile < 1.0, "speed_percentile must be in (0.0, 1.0)"
        )
        self._require(
            self.speed_fallback_px_s > 0.0, "speed_fallback_px_s must be > 0"
        )
        self._require(self.track_count_baseline >= 1, "track_count_baseline must be >= 1")
        self._require(
            self.summary_interval_seconds > 0.0, "summary_interval_seconds must be > 0"
        )
        self._require(self.bucket_capacity >= 1, "bucket_capacity must be >= 1")
        self._require(
            self.bucket_refill_seconds > 0.0, "bucket_refill_seconds must be > 0"
        )

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/test_camera_profile.py -v && mypy`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/domain/camera_profile.py ai-engine/tests/domain/test_camera_profile.py
git commit -m "feat(domain): add CameraProfile with spec-mandated threshold defaults"
```

---

### Task 4: TokenBucket — the budget governor

**Files:**
- Create: `ai-engine/sentinel_ai/domain/policy/rate_budget.py`
- Test: `ai-engine/tests/domain/policy/test_rate_budget.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TokenBucket` frozen dataclass with fields `capacity: int`, `refill_seconds: float`, `tokens: float`, `updated_at: float`; classmethod `full(capacity: int, refill_seconds: float, now: float) -> TokenBucket`; method `try_consume(now: float) -> tuple[bool, TokenBucket]`; property `available: float`

This is what makes §6 real: a chaotic scene cannot exceed its VLM budget.

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/policy/test_rate_budget.py
from __future__ import annotations

import pytest

from sentinel_ai.domain.policy.rate_budget import TokenBucket


def _bucket(now: float = 0.0) -> TokenBucket:
    return TokenBucket.full(capacity=2, refill_seconds=10.0, now=now)


class TestTryConsume:
    def test_a_full_bucket_permits_a_call(self) -> None:
        allowed, _ = _bucket().try_consume(now=0.0)
        assert allowed is True

    def test_consuming_returns_a_new_bucket_and_leaves_the_original_untouched(self) -> None:
        original = _bucket()
        _, updated = original.try_consume(now=0.0)
        assert original.available == 2.0
        assert updated.available == 1.0

    def test_burst_is_capped_at_capacity(self) -> None:
        bucket = _bucket()
        results = []
        for _ in range(4):
            allowed, bucket = bucket.try_consume(now=0.0)
            results.append(allowed)
        assert results == [True, True, False, False]

    def test_tokens_refill_at_one_per_refill_interval(self) -> None:
        bucket = _bucket()
        for _ in range(2):
            _, bucket = bucket.try_consume(now=0.0)
        denied, bucket = bucket.try_consume(now=5.0)
        assert denied is False, "half an interval is not yet a whole token"
        allowed, bucket = bucket.try_consume(now=10.0)
        assert allowed is True

    def test_refill_never_exceeds_capacity(self) -> None:
        bucket = _bucket()
        _, bucket = bucket.try_consume(now=0.0)
        _, bucket = bucket.try_consume(now=0.0)
        bucket = bucket.refilled(now=10_000.0)
        assert bucket.available == 2.0

    def test_clock_going_backwards_does_not_grant_tokens(self) -> None:
        """Monotonic clocks should not regress, but a wrapped source might."""
        bucket = _bucket(now=100.0)
        _, bucket = bucket.try_consume(now=100.0)
        assert bucket.available == 1.0
        bucket = bucket.refilled(now=50.0)
        assert bucket.available == 1.0

    def test_sustained_pressure_settles_at_the_refill_rate(self) -> None:
        """A camera firing every frame gets roughly one call per refill interval.

        60 s at 10 fps with every frame requesting: 2 burst tokens are spent
        immediately (t=0.0, t=0.1), then one token becomes available roughly
        every 10 s (t=10.0, 20.0, 30.1, 40.1, 50.1). Seven calls, not 600.
        """
        bucket = _bucket()
        granted = 0
        for tick in range(600):
            allowed, bucket = bucket.try_consume(now=tick * 0.1)
            granted += allowed
        assert granted == 7


def test_invalid_construction_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket.full(capacity=0, refill_seconds=10.0, now=0.0)
    with pytest.raises(ValueError, match="refill_seconds"):
        TokenBucket.full(capacity=2, refill_seconds=0.0, now=0.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/policy/test_rate_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.policy.rate_budget'`

- [ ] **Step 3: Implement TokenBucket**

```python
# ai-engine/sentinel_ai/domain/policy/rate_budget.py
"""Per-camera VLM call budget (spec §4.1, 'The budget governor').

Immutable: every operation returns a new bucket, so the gate stays a pure
function of its inputs and is trivially testable at any simulated clock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TokenBucket:
    capacity: int
    refill_seconds: float
    tokens: float
    updated_at: float

    @classmethod
    def full(cls, capacity: int, refill_seconds: float, now: float) -> TokenBucket:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_seconds <= 0.0:
            raise ValueError(f"refill_seconds must be > 0, got {refill_seconds}")
        return cls(
            capacity=capacity,
            refill_seconds=refill_seconds,
            tokens=float(capacity),
            updated_at=now,
        )

    @property
    def available(self) -> float:
        return self.tokens

    def refilled(self, now: float) -> TokenBucket:
        """Advance the bucket to `now`. A regressing clock is a no-op."""
        elapsed = now - self.updated_at
        if elapsed <= 0.0:
            return self
        gained = elapsed / self.refill_seconds
        return replace(
            self,
            tokens=min(float(self.capacity), self.tokens + gained),
            updated_at=now,
        )

    def try_consume(self, now: float) -> tuple[bool, TokenBucket]:
        """Attempt to spend one token. Returns (allowed, new_bucket)."""
        current = self.refilled(now)
        if current.tokens < 1.0:
            return False, current
        return True, replace(current, tokens=current.tokens - 1.0)
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/policy/test_rate_budget.py -v && mypy`
Expected: all PASS. Create `ai-engine/tests/domain/policy/__init__.py` if collection fails.

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/domain/policy/rate_budget.py ai-engine/tests/domain/policy
git commit -m "feat(policy): add immutable TokenBucket budget governor

Guarantees a camera cannot exceed its VLM call budget regardless of scene
chaos — the system degrades to metadata-only instead (spec §4.1)."
```

---

### Task 5: The seven escalation triggers

**Files:**
- Create: `ai-engine/sentinel_ai/domain/policy/triggers.py`
- Test: `ai-engine/tests/domain/policy/test_triggers.py`

**Interfaces:**
- Consumes: `SceneState`, `Track`, `BBox`, `EscalationReason` (Task 2); `CameraProfile` (Task 3)
- Produces: `DwellAnchor(cx: float, cy: float, since: float)`; `TriggerContext(scene, profile, previous_signature, scene_delta_streak, dwell_anchors, last_summary_at, speed_threshold_px_s)`; `TriggerOutcome(fired: bool, reason: EscalationReason | None, detail: str)`; functions `new_salient_track`, `scene_change`, `dwell_exceeded`, `speed_anomaly`, `track_count_spike`, `periodic_summary` — each `(TriggerContext) -> TriggerOutcome`; `advance_dwell_anchors(context) -> dict[int, DwellAnchor]`; `ALL_TRIGGERS: tuple[...]` in priority order

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/policy/test_triggers.py
from __future__ import annotations

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, EscalationReason, SceneState, Track
from sentinel_ai.domain.policy.triggers import (
    ALL_TRIGGERS,
    DwellAnchor,
    TriggerContext,
    advance_dwell_anchors,
    dwell_exceeded,
    new_salient_track,
    periodic_summary,
    scene_change,
    speed_anomaly,
    track_count_spike,
)

PROFILE = CameraProfile(camera_id="cam-1")


def track(
    track_id: int = 1,
    label: str = "person",
    age_frames: int = 10,
    speed_px_s: float = 0.0,
    cx: float = 100.0,
    cy: float = 100.0,
) -> Track:
    return Track(
        track_id=track_id,
        label=label,
        box=BBox(cx - 10.0, cy - 10.0, cx + 10.0, cy + 10.0),
        age_frames=age_frames,
        speed_px_s=speed_px_s,
    )


def scene(
    *,
    tracks: tuple[Track, ...] = (),
    timestamp: float = 0.0,
    signature: tuple[float, ...] = (1.0, 0.0),
) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=timestamp,
        detections=(),
        tracks=tracks,
        motion_energy=0.0,
        scene_signature=signature,
    )


def context(
    *,
    scene_state: SceneState,
    previous_signature: tuple[float, ...] | None = None,
    scene_delta_streak: int = 0,
    dwell_anchors: dict[int, DwellAnchor] | None = None,
    last_summary_at: float | None = None,
    speed_threshold_px_s: float | None = None,
) -> TriggerContext:
    return TriggerContext(
        scene=scene_state,
        profile=PROFILE,
        previous_signature=previous_signature,
        scene_delta_streak=scene_delta_streak,
        dwell_anchors=dwell_anchors or {},
        last_summary_at=last_summary_at,
        speed_threshold_px_s=speed_threshold_px_s or PROFILE.speed_fallback_px_s,
    )


class TestNewSalientTrack:
    def test_fires_when_a_salient_track_reaches_the_debounce_age(self) -> None:
        outcome = new_salient_track(context(scene_state=scene(tracks=(track(age_frames=8),))))
        assert outcome.fired is True
        assert outcome.reason is EscalationReason.NEW_SALIENT_TRACK

    def test_does_not_fire_below_the_debounce_age(self) -> None:
        """Flicker must not fire the gate — this is the whole point of debouncing."""
        assert new_salient_track(
            context(scene_state=scene(tracks=(track(age_frames=7),)))
        ).fired is False

    def test_ignores_non_salient_classes(self) -> None:
        assert new_salient_track(
            context(scene_state=scene(tracks=(track(label="bird", age_frames=100),)))
        ).fired is False

    def test_does_not_fire_on_an_empty_scene(self) -> None:
        assert new_salient_track(context(scene_state=scene())).fired is False


class TestSceneChange:
    def test_fires_only_after_the_delta_is_sustained(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.0, 1.0)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=PROFILE.scene_delta_frames - 1,
        )
        assert scene_change(ctx).fired is True

    def test_does_not_fire_on_a_single_frame_spike(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.0, 1.0)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=0,
        )
        assert scene_change(ctx).fired is False

    def test_does_not_fire_below_the_delta_threshold(self) -> None:
        ctx = context(
            scene_state=scene(signature=(0.95, 0.05)),
            previous_signature=(1.0, 0.0),
            scene_delta_streak=100,
        )
        assert scene_change(ctx).fired is False

    def test_first_frame_never_fires(self) -> None:
        ctx = context(scene_state=scene(), previous_signature=None, scene_delta_streak=100)
        assert scene_change(ctx).fired is False


class TestDwellExceeded:
    def test_fires_once_the_anchor_is_older_than_the_dwell_window(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(cx=100.0, cy=100.0),), timestamp=31.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is True

    def test_does_not_fire_before_the_window_elapses(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(),), timestamp=29.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is False

    def test_a_track_that_moved_away_has_no_stale_anchor(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(cx=1000.0, cy=1000.0),), timestamp=31.0),
            dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=0.0)},
        )
        assert dwell_exceeded(ctx).fired is False


class TestAdvanceDwellAnchors:
    def test_a_new_track_gets_an_anchor_at_the_current_time(self) -> None:
        anchors = advance_dwell_anchors(
            context(scene_state=scene(tracks=(track(),), timestamp=12.0))
        )
        assert anchors[1] == DwellAnchor(cx=100.0, cy=100.0, since=12.0)

    def test_a_stationary_track_keeps_its_original_anchor_time(self) -> None:
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(track(cx=110.0, cy=100.0),), timestamp=20.0),
                dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=5.0)},
            )
        )
        assert anchors[1].since == 5.0

    def test_a_track_that_left_the_radius_gets_a_fresh_anchor(self) -> None:
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(track(cx=200.0, cy=100.0),), timestamp=20.0),
                dwell_anchors={1: DwellAnchor(cx=100.0, cy=100.0, since=5.0)},
            )
        )
        assert anchors[1] == DwellAnchor(cx=200.0, cy=100.0, since=20.0)

    def test_anchors_for_departed_tracks_are_dropped(self) -> None:
        """Otherwise the dict grows without bound on a busy camera."""
        anchors = advance_dwell_anchors(
            context(
                scene_state=scene(tracks=(), timestamp=20.0),
                dwell_anchors={99: DwellAnchor(cx=1.0, cy=1.0, since=0.0)},
            )
        )
        assert anchors == {}


class TestSpeedAnomaly:
    def test_fires_above_the_threshold(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(speed_px_s=200.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is True

    def test_does_not_fire_at_or_below_the_threshold(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(speed_px_s=180.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is False

    def test_ignores_non_salient_classes(self) -> None:
        ctx = context(
            scene_state=scene(tracks=(track(label="bird", speed_px_s=999.0),)),
            speed_threshold_px_s=180.0,
        )
        assert speed_anomaly(ctx).fired is False


class TestTrackCountSpike:
    def test_fires_above_the_baseline(self) -> None:
        tracks = tuple(track(track_id=i) for i in range(7))
        assert track_count_spike(context(scene_state=scene(tracks=tracks))).fired is True

    def test_does_not_fire_at_the_baseline(self) -> None:
        tracks = tuple(track(track_id=i) for i in range(6))
        assert track_count_spike(context(scene_state=scene(tracks=tracks))).fired is False


class TestPeriodicSummary:
    def test_fires_on_the_very_first_frame_to_establish_a_baseline(self) -> None:
        ctx = context(scene_state=scene(timestamp=0.0), last_summary_at=None)
        assert periodic_summary(ctx).fired is True

    def test_fires_once_the_interval_has_elapsed(self) -> None:
        ctx = context(scene_state=scene(timestamp=45.0), last_summary_at=0.0)
        assert periodic_summary(ctx).fired is True

    def test_does_not_fire_before_the_interval(self) -> None:
        ctx = context(scene_state=scene(timestamp=44.9), last_summary_at=0.0)
        assert periodic_summary(ctx).fired is False


def test_triggers_are_ordered_most_urgent_first() -> None:
    """Priority order decides which reason is reported when several fire."""
    assert [fn.__name__ for fn in ALL_TRIGGERS] == [
        "speed_anomaly",
        "dwell_exceeded",
        "track_count_spike",
        "scene_change",
        "new_salient_track",
        "periodic_summary",
    ]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/policy/test_triggers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.policy.triggers'`

- [ ] **Step 3: Implement the triggers**

```python
# ai-engine/sentinel_ai/domain/policy/triggers.py
"""The seven escalation predicates from spec §4.1.

Each is a pure function of a TriggerContext so it can be tested in isolation at
any simulated clock value. USER_REQUESTED is not here: it bypasses predicates
entirely and is handled by the orchestrator.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import hypot

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState


@dataclass(frozen=True, slots=True)
class DwellAnchor:
    """Where a track settled, and when. Dwell is measured from `since`."""

    cx: float
    cy: float
    since: float


@dataclass(frozen=True, slots=True)
class TriggerContext:
    scene: SceneState
    profile: CameraProfile
    previous_signature: tuple[float, ...] | None
    scene_delta_streak: int
    dwell_anchors: dict[int, DwellAnchor]
    last_summary_at: float | None
    speed_threshold_px_s: float


@dataclass(frozen=True, slots=True)
class TriggerOutcome:
    fired: bool
    reason: EscalationReason | None = None
    detail: str = ""


_NOT_FIRED = TriggerOutcome(fired=False)


def new_salient_track(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        if track.age_frames >= ctx.profile.min_track_frames:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.NEW_SALIENT_TRACK,
                detail=f"{track.label} track {track.track_id} age={track.age_frames}",
            )
    return _NOT_FIRED


def scene_change(ctx: TriggerContext) -> TriggerOutcome:
    delta = ctx.scene.signature_delta(ctx.previous_signature)
    if delta < ctx.profile.scene_delta_threshold:
        return _NOT_FIRED
    if ctx.scene_delta_streak + 1 < ctx.profile.scene_delta_frames:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.SCENE_CHANGE,
        detail=f"signature delta={delta:.3f} sustained {ctx.scene_delta_streak + 1} frames",
    )


def dwell_exceeded(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        anchor = ctx.dwell_anchors.get(track.track_id)
        if anchor is None:
            continue
        drifted = hypot(track.box.cx - anchor.cx, track.box.cy - anchor.cy)
        if drifted > ctx.profile.dwell_radius_px:
            continue
        dwelled = ctx.scene.timestamp - anchor.since
        if dwelled >= ctx.profile.dwell_seconds:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.DWELL_EXCEEDED,
                detail=f"{track.label} track {track.track_id} dwelled {dwelled:.1f}s",
            )
    return _NOT_FIRED


def speed_anomaly(ctx: TriggerContext) -> TriggerOutcome:
    for track in ctx.scene.tracks_of(ctx.profile.salient_classes):
        if track.speed_px_s > ctx.speed_threshold_px_s:
            return TriggerOutcome(
                fired=True,
                reason=EscalationReason.SPEED_ANOMALY,
                detail=(
                    f"{track.label} track {track.track_id} at {track.speed_px_s:.0f}px/s "
                    f"(threshold {ctx.speed_threshold_px_s:.0f})"
                ),
            )
    return _NOT_FIRED


def track_count_spike(ctx: TriggerContext) -> TriggerOutcome:
    count = sum(1 for _ in ctx.scene.tracks_of(ctx.profile.salient_classes))
    if count <= ctx.profile.track_count_baseline:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.TRACK_COUNT_SPIKE,
        detail=f"{count} concurrent tracks (baseline {ctx.profile.track_count_baseline})",
    )


def periodic_summary(ctx: TriggerContext) -> TriggerOutcome:
    if ctx.last_summary_at is None:
        return TriggerOutcome(
            fired=True,
            reason=EscalationReason.PERIODIC_SUMMARY,
            detail="initial scene summary",
        )
    elapsed = ctx.scene.timestamp - ctx.last_summary_at
    if elapsed < ctx.profile.summary_interval_seconds:
        return _NOT_FIRED
    return TriggerOutcome(
        fired=True,
        reason=EscalationReason.PERIODIC_SUMMARY,
        detail=f"{elapsed:.0f}s since last summary",
    )


def advance_dwell_anchors(ctx: TriggerContext) -> dict[int, DwellAnchor]:
    """Recompute anchors for the current frame.

    A track staying inside `dwell_radius_px` of its anchor keeps the original
    `since`; a track that left gets a fresh anchor. Departed tracks are dropped
    so the mapping cannot grow without bound.
    """
    anchors: dict[int, DwellAnchor] = {}
    for track in ctx.scene.tracks:
        cx, cy = track.box.cx, track.box.cy
        previous = ctx.dwell_anchors.get(track.track_id)
        if previous is not None and hypot(cx - previous.cx, cy - previous.cy) <= (
            ctx.profile.dwell_radius_px
        ):
            anchors[track.track_id] = previous
        else:
            anchors[track.track_id] = DwellAnchor(cx=cx, cy=cy, since=ctx.scene.timestamp)
    return anchors


# Priority order: when several triggers fire, the first one names the reason.
ALL_TRIGGERS: tuple[Callable[[TriggerContext], TriggerOutcome], ...] = (
    speed_anomaly,
    dwell_exceeded,
    track_count_spike,
    scene_change,
    new_salient_track,
    periodic_summary,
)
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/policy/test_triggers.py -v && mypy`
Expected: all PASS (30 tests).

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/domain/policy/triggers.py ai-engine/tests/domain/policy/test_triggers.py
git commit -m "feat(policy): add the seven escalation triggers as pure predicates

Each trigger is independently testable at a simulated clock. Dwell anchors are
pruned for departed tracks so the mapping cannot grow unbounded on a busy feed."
```

---

### Task 6: The escalation gate

**Files:**
- Create: `ai-engine/sentinel_ai/domain/policy/escalation.py`
- Test: `ai-engine/tests/domain/policy/test_escalation.py`

**Interfaces:**
- Consumes: `TokenBucket` (Task 4); triggers module (Task 5); `SceneState`, `EscalationReason` (Task 2); `CameraProfile` (Task 3)
- Produces: `GateState` frozen dataclass with classmethod `initial(profile: CameraProfile, now: float) -> GateState`; `EscalationDecision(should_escalate: bool, reason: EscalationReason | None, detail: str, suppressed_by: str | None)`; `GateOutcome(decision: EscalationDecision, state: GateState)`; `decide(scene: SceneState, profile: CameraProfile, state: GateState) -> GateOutcome`; `force(reason: EscalationReason, state: GateState, now: float) -> GateOutcome`

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/policy/test_escalation.py
from __future__ import annotations

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import BBox, EscalationReason, SceneState, Track
from sentinel_ai.domain.policy.escalation import GateState, decide, force

PROFILE = CameraProfile(camera_id="cam-1")
QUIET_SIGNATURE = (1.0, 0.0)


def running_track(track_id: int = 1) -> Track:
    return Track(
        track_id=track_id,
        label="person",
        box=BBox(90.0, 90.0, 110.0, 110.0),
        age_frames=20,
        speed_px_s=500.0,
    )


def scene(
    *,
    timestamp: float,
    tracks: tuple[Track, ...] = (),
    signature: tuple[float, ...] = QUIET_SIGNATURE,
    frame_index: int = 0,
) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=frame_index,
        timestamp=timestamp,
        detections=(),
        tracks=tracks,
        motion_energy=0.0,
        scene_signature=signature,
    )


def initial_state(now: float = 0.0) -> GateState:
    return GateState.initial(PROFILE, now=now)


def settled_state() -> GateState:
    """Past the initial summary, with a known signature and a full bucket."""
    outcome = decide(scene(timestamp=0.0), PROFILE, initial_state())
    return outcome.state


class TestFirstFrame:
    def test_the_first_frame_escalates_to_establish_a_baseline(self) -> None:
        outcome = decide(scene(timestamp=0.0), PROFILE, initial_state())
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.PERIODIC_SUMMARY

    def test_the_first_frame_consumes_a_token(self) -> None:
        outcome = decide(scene(timestamp=0.0), PROFILE, initial_state())
        assert outcome.state.bucket.available == 1.0


class TestQuietScene:
    def test_a_quiet_scene_does_not_escalate(self) -> None:
        outcome = decide(scene(timestamp=1.0), PROFILE, settled_state())
        assert outcome.decision.should_escalate is False
        assert outcome.decision.reason is None

    def test_a_quiet_scene_spends_no_tokens(self) -> None:
        state = settled_state()
        outcome = decide(scene(timestamp=1.0), PROFILE, state)
        assert outcome.state.bucket.available == state.bucket.available


def alternating(tick: int) -> tuple[float, ...]:
    """A signature that changes every frame.

    Necessary for any test targeting the rate budget: with a constant signature
    duplicate-suppression fires first and the budget is never reached, so the
    test would pass for the wrong reason.
    """
    return (1.0, 0.0) if tick % 2 == 0 else (0.0, 1.0)


class TestPriority:
    def test_speed_anomaly_outranks_periodic_summary(self) -> None:
        state = settled_state()
        outcome = decide(
            scene(timestamp=100.0, tracks=(running_track(),), signature=(0.0, 1.0)),
            PROFILE,
            state,
        )
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.SPEED_ANOMALY


class TestBudgetGovernor:
    def test_the_bucket_caps_escalations_under_sustained_pressure(self) -> None:
        """A chaotic scene must not be able to saturate the GPU (spec §4.1).

        60 s at 10 fps where every single frame trips speed_anomaly. Signatures
        alternate so dedup never suppresses — the budget is the only limiter.
        Expect 7 VLM calls, not 600.
        """
        state = initial_state()
        escalations = 0
        for tick in range(600):
            outcome = decide(
                scene(
                    timestamp=tick * 0.1,
                    tracks=(running_track(),),
                    signature=alternating(tick),
                    frame_index=tick,
                ),
                PROFILE,
                state,
            )
            state = outcome.state
            escalations += outcome.decision.should_escalate
        assert escalations == 7

    def test_a_denied_decision_records_the_budget_as_the_suppressor(self) -> None:
        state = initial_state()
        for tick in range(3):
            outcome = decide(
                scene(
                    timestamp=float(tick),
                    tracks=(running_track(),),
                    signature=alternating(tick),
                ),
                PROFILE,
                state,
            )
            state = outcome.state
        assert outcome.decision.should_escalate is False
        assert outcome.decision.suppressed_by == "rate_budget"
        assert outcome.decision.reason is EscalationReason.SPEED_ANOMALY, (
            "the reason is still reported so telemetry can show what was suppressed"
        )


class TestDeduplication:
    def test_an_unchanged_scene_is_suppressed_even_with_budget_available(self) -> None:
        state = settled_state()
        first = decide(
            scene(timestamp=20.0, tracks=(running_track(),), signature=QUIET_SIGNATURE),
            PROFILE,
            state,
        )
        assert first.decision.should_escalate is True

        second = decide(
            scene(timestamp=40.0, tracks=(running_track(),), signature=QUIET_SIGNATURE),
            PROFILE,
            first.state,
        )
        assert second.decision.should_escalate is False
        assert second.decision.suppressed_by == "duplicate_scene"

    def test_a_materially_changed_scene_is_not_deduplicated(self) -> None:
        state = settled_state()
        first = decide(
            scene(timestamp=20.0, tracks=(running_track(),), signature=(1.0, 0.0)),
            PROFILE,
            state,
        )
        assert first.decision.should_escalate is True

        second = decide(
            scene(timestamp=40.0, tracks=(running_track(),), signature=(0.0, 1.0)),
            PROFILE,
            first.state,
        )
        assert second.decision.should_escalate is True

    def test_dedup_does_not_refund_a_suppressed_token(self) -> None:
        state = settled_state()
        first = decide(scene(timestamp=20.0, tracks=(running_track(),)), PROFILE, state)
        second = decide(
            scene(timestamp=40.0, tracks=(running_track(),)), PROFILE, first.state
        )
        assert second.state.bucket.available == first.state.bucket.refilled(40.0).available


class TestVlmDisabled:
    def test_a_camera_with_vlm_disabled_never_escalates(self) -> None:
        profile = CameraProfile(camera_id="cam-1", vlm_enabled=False)
        outcome = decide(
            scene(timestamp=0.0, tracks=(running_track(),)),
            profile,
            GateState.initial(profile, now=0.0),
        )
        assert outcome.decision.should_escalate is False
        assert outcome.decision.suppressed_by == "vlm_disabled"


class TestSceneDeltaStreak:
    def test_the_streak_accumulates_across_frames_of_sustained_change(self) -> None:
        state = settled_state()
        signatures = [(0.0, 1.0), (1.0, 0.0)] * 3
        for index, signature in enumerate(signatures):
            outcome = decide(
                scene(timestamp=1.0 + index, signature=signature), PROFILE, state
            )
            state = outcome.state
        assert state.scene_delta_streak >= PROFILE.scene_delta_frames

    def test_the_streak_resets_when_the_scene_settles(self) -> None:
        state = settled_state()
        changed = decide(scene(timestamp=1.0, signature=(0.0, 1.0)), PROFILE, state)
        assert changed.state.scene_delta_streak == 1
        settled = decide(
            scene(timestamp=2.0, signature=(0.0, 1.0)), PROFILE, changed.state
        )
        assert settled.state.scene_delta_streak == 0


class TestForce:
    def test_a_user_request_escalates_regardless_of_budget(self) -> None:
        state = initial_state()
        for tick in range(3):
            state = decide(
                scene(timestamp=float(tick), tracks=(running_track(),)), PROFILE, state
            ).state
        outcome = force(EscalationReason.USER_REQUESTED, state, now=3.0)
        assert outcome.decision.should_escalate is True
        assert outcome.decision.reason is EscalationReason.USER_REQUESTED

    def test_a_forced_call_still_records_the_escalation_time(self) -> None:
        outcome = force(EscalationReason.USER_REQUESTED, initial_state(), now=7.0)
        assert outcome.state.last_escalation_at == 7.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/policy/test_escalation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.policy.escalation'`

- [ ] **Step 3: Implement the gate**

```python
# ai-engine/sentinel_ai/domain/policy/escalation.py
"""The escalation gate (spec §4.1) — the core of SentinelAI's cost/accuracy tradeoff.

`decide` is a pure function of (scene, profile, state). It returns both a
decision and the next state, so a caller threads state through frames without
the gate ever holding mutable state or reading a clock.

Order of evaluation matters and is deliberate:
  1. vlm_enabled          — a disabled camera short-circuits everything
  2. triggers             — is anything worth looking at?
  3. duplicate suppression— have we already described this exact scene?
  4. rate budget          — can we afford it?
Suppression at stages 3 and 4 still reports the trigger reason, so telemetry can
show what the gate declined and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import EscalationReason, SceneState
from sentinel_ai.domain.policy.rate_budget import TokenBucket
from sentinel_ai.domain.policy.triggers import (
    ALL_TRIGGERS,
    DwellAnchor,
    TriggerContext,
    advance_dwell_anchors,
)

DEDUP_EPSILON = 0.05
"""Scene-signature distance below which two frames count as the same scene."""


@dataclass(frozen=True, slots=True)
class GateState:
    """Everything the gate must remember between frames."""

    bucket: TokenBucket
    previous_signature: tuple[float, ...] | None = None
    scene_delta_streak: int = 0
    dwell_anchors: dict[int, DwellAnchor] = field(default_factory=dict)
    last_summary_at: float | None = None
    last_escalation_at: float | None = None
    last_escalated_signature: tuple[float, ...] | None = None

    @classmethod
    def initial(cls, profile: CameraProfile, now: float) -> GateState:
        return cls(
            bucket=TokenBucket.full(
                capacity=profile.bucket_capacity,
                refill_seconds=profile.bucket_refill_seconds,
                now=now,
            )
        )


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    should_escalate: bool
    reason: EscalationReason | None = None
    detail: str = ""
    suppressed_by: str | None = None


@dataclass(frozen=True, slots=True)
class GateOutcome:
    decision: EscalationDecision
    state: GateState


def _speed_threshold(profile: CameraProfile) -> float:
    """Slice 1 uses the fixed fallback. Phase 4 supplies a learned percentile."""
    return profile.speed_fallback_px_s


def _next_streak(scene: SceneState, profile: CameraProfile, state: GateState) -> int:
    delta = scene.signature_delta(state.previous_signature)
    return state.scene_delta_streak + 1 if delta >= profile.scene_delta_threshold else 0


def _is_duplicate(scene: SceneState, state: GateState) -> bool:
    if state.last_escalated_signature is None:
        return False
    try:
        return scene.signature_delta(state.last_escalated_signature) < DEDUP_EPSILON
    except ValueError:
        # Signature length changed (resolution change) — not a duplicate.
        return False


def decide(scene: SceneState, profile: CameraProfile, state: GateState) -> GateOutcome:
    now = scene.timestamp
    anchors = advance_dwell_anchors(
        TriggerContext(
            scene=scene,
            profile=profile,
            previous_signature=state.previous_signature,
            scene_delta_streak=state.scene_delta_streak,
            dwell_anchors=state.dwell_anchors,
            last_summary_at=state.last_summary_at,
            speed_threshold_px_s=_speed_threshold(profile),
        )
    )
    streak = _next_streak(scene, profile, state)

    carried = replace(
        state,
        previous_signature=scene.scene_signature,
        scene_delta_streak=streak,
        dwell_anchors=anchors,
    )

    if not profile.vlm_enabled:
        return GateOutcome(
            decision=EscalationDecision(should_escalate=False, suppressed_by="vlm_disabled"),
            state=carried,
        )

    ctx = TriggerContext(
        scene=scene,
        profile=profile,
        previous_signature=state.previous_signature,
        scene_delta_streak=state.scene_delta_streak,
        dwell_anchors=state.dwell_anchors,
        last_summary_at=state.last_summary_at,
        speed_threshold_px_s=_speed_threshold(profile),
    )

    fired = next((outcome for trigger in ALL_TRIGGERS if (outcome := trigger(ctx)).fired), None)
    if fired is None or fired.reason is None:
        return GateOutcome(
            decision=EscalationDecision(should_escalate=False), state=carried
        )

    if _is_duplicate(scene, state):
        return GateOutcome(
            decision=EscalationDecision(
                should_escalate=False,
                reason=fired.reason,
                detail=fired.detail,
                suppressed_by="duplicate_scene",
            ),
            state=replace(carried, bucket=carried.bucket.refilled(now)),
        )

    allowed, bucket = carried.bucket.try_consume(now)
    if not allowed:
        return GateOutcome(
            decision=EscalationDecision(
                should_escalate=False,
                reason=fired.reason,
                detail=fired.detail,
                suppressed_by="rate_budget",
            ),
            state=replace(carried, bucket=bucket),
        )

    summary_at = (
        now if fired.reason is EscalationReason.PERIODIC_SUMMARY else carried.last_summary_at
    )
    return GateOutcome(
        decision=EscalationDecision(
            should_escalate=True, reason=fired.reason, detail=fired.detail
        ),
        state=replace(
            carried,
            bucket=bucket,
            last_summary_at=summary_at,
            last_escalation_at=now,
            last_escalated_signature=scene.scene_signature,
        ),
    )


def force(reason: EscalationReason, state: GateState, now: float) -> GateOutcome:
    """Escalate unconditionally — used for USER_REQUESTED (spec §6)."""
    return GateOutcome(
        decision=EscalationDecision(
            should_escalate=True, reason=reason, detail="forced escalation"
        ),
        state=replace(state, last_escalation_at=now),
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/policy/ -v && mypy`
Expected: all PASS.

- [ ] **Step 5: Verify the architecture fitness tests still hold**

Run: `cd ai-engine && pytest tests/test_architecture.py -v`
Expected: PASS — confirms the gate reads no clock and imports no I/O.

- [ ] **Step 6: Commit**

```bash
git add ai-engine/sentinel_ai/domain/policy/escalation.py ai-engine/tests/domain/policy/test_escalation.py
git commit -m "feat(policy): add the escalation gate

Pure (scene, profile, state) -> (decision, state). Evaluates vlm_enabled,
then triggers, then duplicate suppression, then rate budget. Suppressed
decisions still report their trigger reason so telemetry can show what the
gate declined and why.

Verified: a scene triggering every frame at 10fps yields 8 VLM calls per
minute, not 600."
```

---

### Task 7: VRAM residency planner

**Files:**
- Create: `ai-engine/sentinel_ai/domain/policy/vram_budget.py`
- Test: `ai-engine/tests/domain/policy/test_vram_budget.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelSpec(model_key: str, vram_mib: int, priority: int, idle_unload_seconds: float | None)`; `ResidencyPlan(load: tuple[str, ...], unload: tuple[str, ...], resident: tuple[str, ...], free_mib: int)`; `plan_residency(specs, currently_resident, required, last_used_at, now, total_mib, reserved_mib) -> ResidencyPlan`; exception `InsufficientVram`

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/domain/policy/test_vram_budget.py
from __future__ import annotations

import pytest

from sentinel_ai.domain.policy.vram_budget import (
    InsufficientVram,
    ModelSpec,
    plan_residency,
)

DETECTOR = ModelSpec("yolo11s", vram_mib=900, priority=100, idle_unload_seconds=None)
VLM = ModelSpec("qwen25vl3b", vram_mib=4400, priority=50, idle_unload_seconds=600.0)
BIG_VLM = ModelSpec("qwen25vl7b", vram_mib=6200, priority=50, idle_unload_seconds=600.0)
SPECS = {spec.model_key: spec for spec in (DETECTOR, VLM, BIG_VLM)}

BUDGET = {"total_mib": 8192, "reserved_mib": 2048}


def plan(**kwargs: object):
    defaults: dict[str, object] = {
        "specs": SPECS,
        "currently_resident": (),
        "required": (),
        "last_used_at": {},
        "now": 0.0,
        **BUDGET,
    }
    return plan_residency(**{**defaults, **kwargs})  # type: ignore[arg-type]


class TestLoading:
    def test_a_required_model_is_scheduled_for_load(self) -> None:
        result = plan(required=("yolo11s",))
        assert result.load == ("yolo11s",)
        assert result.resident == ("yolo11s",)

    def test_an_already_resident_model_is_not_reloaded(self) -> None:
        result = plan(currently_resident=("yolo11s",), required=("yolo11s",))
        assert result.load == ()
        assert result.resident == ("yolo11s",)

    def test_detector_and_vlm_both_fit_the_8gb_budget(self) -> None:
        """Spec §4.3: 900 + 4400 = 5300 MiB against a 6144 MiB usable budget."""
        result = plan(required=("yolo11s", "qwen25vl3b"))
        assert set(result.resident) == {"yolo11s", "qwen25vl3b"}
        assert result.free_mib == 8192 - 2048 - 900 - 4400

    def test_reserved_headroom_is_never_allocated(self) -> None:
        result = plan(required=("yolo11s",))
        assert result.free_mib == 8192 - 2048 - 900


class TestEviction:
    def test_an_idle_model_is_evicted_to_make_room(self) -> None:
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=("qwen25vl7b",),
            last_used_at={"qwen25vl3b": 0.0},
            now=10.0,
        )
        assert result.unload == ("qwen25vl3b",)
        assert result.resident == ("qwen25vl7b",)

    def test_eviction_prefers_the_least_recently_used(self) -> None:
        specs = {
            "a": ModelSpec("a", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
            "b": ModelSpec("b", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
            "c": ModelSpec("c", vram_mib=3000, priority=10, idle_unload_seconds=60.0),
        }
        result = plan_residency(
            specs=specs,
            currently_resident=("a", "b"),
            required=("c",),
            last_used_at={"a": 5.0, "b": 1.0},
            now=10.0,
            **BUDGET,
        )
        assert result.unload == ("b",), "b was used longest ago"

    def test_a_higher_priority_resident_model_is_never_evicted(self) -> None:
        """The detector must stay resident — it is the always-on stage."""
        result = plan(
            currently_resident=("yolo11s",),
            required=("qwen25vl7b",),
            last_used_at={"yolo11s": 0.0},
            now=10_000.0,
        )
        assert "yolo11s" not in result.unload
        assert set(result.resident) == {"yolo11s", "qwen25vl7b"}

    def test_idle_models_are_unloaded_even_when_no_load_is_needed(self) -> None:
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=(),
            last_used_at={"qwen25vl3b": 0.0},
            now=601.0,
        )
        assert result.unload == ("qwen25vl3b",)

    def test_a_model_within_its_idle_window_stays_resident(self) -> None:
        result = plan(
            currently_resident=("qwen25vl3b",),
            required=(),
            last_used_at={"qwen25vl3b": 0.0},
            now=599.0,
        )
        assert result.unload == ()

    def test_a_model_with_no_idle_timeout_is_never_idle_evicted(self) -> None:
        result = plan(
            currently_resident=("yolo11s",),
            required=(),
            last_used_at={"yolo11s": 0.0},
            now=1_000_000.0,
        )
        assert result.unload == ()


class TestFailure:
    def test_a_model_larger_than_the_budget_raises(self) -> None:
        specs = {"huge": ModelSpec("huge", vram_mib=99_000, priority=1, idle_unload_seconds=None)}
        with pytest.raises(InsufficientVram, match="huge"):
            plan_residency(
                specs=specs,
                currently_resident=(),
                required=("huge",),
                last_used_at={},
                now=0.0,
                **BUDGET,
            )

    def test_an_unknown_model_key_raises(self) -> None:
        with pytest.raises(KeyError, match="nope"):
            plan(required=("nope",))


def test_a_24gb_budget_accommodates_the_7b_vlm_alongside_the_detector() -> None:
    """The planner is a function of the budget, not a hardcoded layout (spec §4.3)."""
    result = plan_residency(
        specs=SPECS,
        currently_resident=(),
        required=("yolo11s", "qwen25vl7b"),
        last_used_at={},
        now=0.0,
        total_mib=24_564,
        reserved_mib=2048,
    )
    assert set(result.resident) == {"yolo11s", "qwen25vl7b"}
    assert result.unload == ()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/domain/policy/test_vram_budget.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.domain.policy.vram_budget'`

- [ ] **Step 3: Implement the planner**

```python
# ai-engine/sentinel_ai/domain/policy/vram_budget.py
"""VRAM residency planning (spec §4.3, §34).

Pure: it decides what to load and unload. The orchestrator's resident set
executes the plan. Residency is a function of the budget, so the same planner
serves an 8 GB laptop GPU and a 24 GB 4090 without change.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class InsufficientVram(RuntimeError):
    """A required model cannot fit even with everything evictable unloaded."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_key: str
    vram_mib: int
    priority: int
    """Higher survives eviction. The always-on detector should outrank the VLM."""
    idle_unload_seconds: float | None
    """None means never idle-evict."""


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    load: tuple[str, ...]
    unload: tuple[str, ...]
    resident: tuple[str, ...]
    free_mib: int


def plan_residency(
    *,
    specs: Mapping[str, ModelSpec],
    currently_resident: Sequence[str],
    required: Sequence[str],
    last_used_at: Mapping[str, float],
    now: float,
    total_mib: int,
    reserved_mib: int,
) -> ResidencyPlan:
    for key in (*required, *currently_resident):
        if key not in specs:
            raise KeyError(f"unknown model key: {key}")

    usable = total_mib - reserved_mib
    required_set = set(required)

    resident = list(currently_resident)
    unload: list[str] = []

    # Idle eviction first: free memory nobody is asking for.
    for key in list(resident):
        if key in required_set:
            continue
        spec = specs[key]
        if spec.idle_unload_seconds is None:
            continue
        idle_for = now - last_used_at.get(key, now)
        if idle_for >= spec.idle_unload_seconds:
            resident.remove(key)
            unload.append(key)

    def used() -> int:
        return sum(specs[key].vram_mib for key in resident)

    for key in required:
        if key in resident:
            continue
        spec = specs[key]
        if spec.vram_mib > usable:
            raise InsufficientVram(
                f"model {key} needs {spec.vram_mib} MiB but only {usable} MiB is usable"
            )
        # Evict LRU among lower-or-equal-priority, non-required models until it fits.
        while used() + spec.vram_mib > usable:
            evictable = [
                candidate
                for candidate in resident
                if candidate not in required_set
                and specs[candidate].priority <= spec.priority
            ]
            if not evictable:
                raise InsufficientVram(
                    f"cannot fit {key} ({spec.vram_mib} MiB): "
                    f"{used()} MiB resident, {usable} MiB usable, nothing evictable"
                )
            victim = min(evictable, key=lambda k: (last_used_at.get(k, 0.0), k))
            resident.remove(victim)
            unload.append(victim)
        resident.append(key)

    loaded = tuple(key for key in required if key not in currently_resident)
    return ResidencyPlan(
        load=loaded,
        unload=tuple(unload),
        resident=tuple(resident),
        free_mib=usable - used(),
    )
```

- [ ] **Step 4: Run the tests**

Run: `cd ai-engine && pytest tests/domain/policy/test_vram_budget.py -v && mypy`
Expected: all PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add ai-engine/sentinel_ai/domain/policy/vram_budget.py ai-engine/tests/domain/policy/test_vram_budget.py
git commit -m "feat(policy): add VRAM residency planner with LRU + priority eviction

Verified against both the 8GB dev budget and a 24GB 4090 budget from the same
code path — residency is a function of the budget, not a hardcoded layout."
```

---

### Task 8: Ports and test fakes

**Files:**
- Create: `ai-engine/sentinel_ai/ports/model_runtime.py`, `detector.py`, `tracker.py`, `vision_llm.py`, `frame_source.py`, `event_publisher.py`, `clip_writer.py`
- Create: `ai-engine/tests/fakes/__init__.py`, `ai-engine/tests/fakes/models.py`, `ai-engine/tests/fakes/io.py`
- Test: `ai-engine/tests/ports/test_port_contracts.py`

**Interfaces:**
- Consumes: `Detection`, `Track`, `SceneState`, `Event` (Task 2)
- Produces:
  - `LifecycleState` StrEnum: `LOADED, UNLOADED, SLEEPING, DOWNLOADING, UPDATING, OFFLINE, HEALTHY, UNHEALTHY` (spec §5)
  - `HealthReport(state: LifecycleState, detail: str, vram_mib: int)`
  - `Capabilities(model_key: str, kind: str, labels: frozenset[str], vram_mib: int, batch_max: int)`
  - `ModelRuntime` ABC with async `initialize()`, `warmup()`, `predict(request: object) -> object`, `shutdown()`, and `health() -> HealthReport`, `version() -> str`, `capabilities() -> Capabilities` (spec §10)
  - `ObjectDetector` ABC: `async detect(frame: FrameData) -> tuple[Detection, ...]`
  - `Tracker` ABC: `update(detections, timestamp) -> tuple[Track, ...]`, `reset()`
  - `SceneDescription(description: str, threat_value: float, suggested_action: str)`
  - `VisionLanguageModel` ABC: `async describe(request: VisionRequest) -> SceneDescription`
  - `VisionRequest(keyframe: FrameData, scene: SceneState, history: tuple[str, ...], camera_label: str, reason_detail: str)`
  - `FrameData(camera_id: str, frame_index: int, timestamp: float, width: int, height: int, pixels: object)`
  - `FrameSource` ABC: `async __aiter__() -> AsyncIterator[FrameData]`, `async close()`
  - `EventPublisher` ABC: `async publish(event: Event)`, `async close()`
  - `ClipWriter` ABC: `async write(camera_id, event_id, frames, fps) -> str`
  - Fakes: `FakeDetector`, `FakeTracker`, `FakeVisionLLM`, `FakeSource`, `FakePublisher`, `FakeClipWriter`

- [ ] **Step 1: Write the failing contract test**

```python
# ai-engine/tests/ports/test_port_contracts.py
from __future__ import annotations

import inspect

import pytest

from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameSource
from sentinel_ai.ports.model_runtime import (
    Capabilities,
    HealthReport,
    LifecycleState,
    ModelRuntime,
)
from sentinel_ai.ports.tracker import Tracker
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeDetector, FakeTracker, FakeVisionLLM

ALL_PORTS = [
    ModelRuntime,
    ObjectDetector,
    Tracker,
    VisionLanguageModel,
    FrameSource,
    EventPublisher,
    ClipWriter,
]


@pytest.mark.parametrize("port", ALL_PORTS, ids=lambda p: p.__name__)
def test_every_port_is_abstract_and_cannot_be_instantiated(port: type) -> None:
    assert inspect.isabstract(port), f"{port.__name__} has no abstract methods"
    with pytest.raises(TypeError):
        port()  # type: ignore[call-arg,abstract]


def test_model_runtime_exposes_the_seven_spec_section_10_methods() -> None:
    required = {
        "initialize", "health", "predict", "warmup", "shutdown", "version", "capabilities"
    }
    assert required <= set(ModelRuntime.__abstractmethods__)


def test_lifecycle_states_cover_all_eight_from_spec_section_5() -> None:
    assert {s.value for s in LifecycleState} == {
        "loaded", "unloaded", "sleeping", "downloading",
        "updating", "offline", "healthy", "unhealthy",
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
    ],
    ids=lambda x: x.__name__,
)
def test_fakes_satisfy_their_ports(fake: type, port: type) -> None:
    assert issubclass(fake, port)


class TestFakeDetector:
    async def test_it_replays_scripted_detections_in_order(self) -> None:
        from sentinel_ai.domain.entities import BBox, Detection

        first = (Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),)
        second = (Detection("car", 0.8, BBox(5.0, 5.0, 25.0, 25.0)),)
        detector = FakeDetector(script=[first, second])
        frame = FakeSource.make_frame(camera_id="cam-1", frame_index=0, timestamp=0.0)

        assert await detector.detect(frame) == first
        assert await detector.detect(frame) == second

    async def test_it_repeats_the_final_entry_once_the_script_runs_out(self) -> None:
        from sentinel_ai.domain.entities import BBox, Detection

        only = (Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),)
        detector = FakeDetector(script=[only])
        frame = FakeSource.make_frame(camera_id="cam-1", frame_index=0, timestamp=0.0)

        assert await detector.detect(frame) == only
        assert await detector.detect(frame) == only

    async def test_it_counts_calls_so_tests_can_assert_reuse(self) -> None:
        detector = FakeDetector(script=[()])
        frame = FakeSource.make_frame(camera_id="cam-1", frame_index=0, timestamp=0.0)
        await detector.detect(frame)
        await detector.detect(frame)
        assert detector.call_count == 2


class TestFakeVisionLLM:
    async def test_it_records_every_request_for_assertion(self) -> None:
        from sentinel_ai.domain.entities import SceneState
        from sentinel_ai.ports.vision_llm import VisionRequest

        vlm = FakeVisionLLM(
            response=SceneDescription(
                description="Two people talking near the entrance.",
                threat_value=0.1,
                suggested_action="No action required.",
            )
        )
        scene = SceneState("cam-1", 0, 0.0, (), (), 0.0, (1.0,))
        request = VisionRequest(
            keyframe=FakeSource.make_frame("cam-1", 0, 0.0),
            scene=scene,
            history=(),
            camera_label="Front Door",
            reason_detail="test",
        )
        result = await vlm.describe(request)

        assert result.description == "Two people talking near the entrance."
        assert vlm.requests == [request]

    async def test_it_can_be_configured_to_raise_for_failure_path_tests(self) -> None:
        vlm = FakeVisionLLM(error=TimeoutError("vlm timed out"))
        scene_request = VisionRequest(
            keyframe=FakeSource.make_frame("cam-1", 0, 0.0),
            scene=__import__(
                "sentinel_ai.domain.entities", fromlist=["SceneState"]
            ).SceneState("cam-1", 0, 0.0, (), (), 0.0, (1.0,)),
            history=(),
            camera_label="Front Door",
            reason_detail="test",
        )
        with pytest.raises(TimeoutError):
            await vlm.describe(scene_request)


class TestFakeTracker:
    def test_it_assigns_stable_ids_and_ages_tracks(self) -> None:
        from sentinel_ai.domain.entities import BBox, Detection

        tracker = FakeTracker()
        detection = Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0))

        first = tracker.update((detection,), timestamp=0.0)
        second = tracker.update((detection,), timestamp=0.1)

        assert first[0].track_id == second[0].track_id
        assert second[0].age_frames == first[0].age_frames + 1

    def test_reset_clears_all_state(self) -> None:
        from sentinel_ai.domain.entities import BBox, Detection

        tracker = FakeTracker()
        tracker.update((Detection("person", 0.9, BBox(0.0, 0.0, 10.0, 10.0)),), timestamp=0.0)
        tracker.reset()
        assert tracker.update((), timestamp=1.0) == ()


class TestFakePublisher:
    async def test_it_collects_published_events(self) -> None:
        from uuid import uuid4

        from sentinel_ai.domain.entities import (
            EscalationReason,
            Event,
            ThreatScore,
        )

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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/ports -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.ports.model_runtime'`

- [ ] **Step 3: Implement the ports**

```python
# ai-engine/sentinel_ai/ports/model_runtime.py
"""The §10 common model interface. Every model implements exactly this,
which is what makes new models plug-and-play."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum


class LifecycleState(StrEnum):
    """The eight states from spec §5."""

    LOADED = "loaded"
    UNLOADED = "unloaded"
    SLEEPING = "sleeping"
    DOWNLOADING = "downloading"
    UPDATING = "updating"
    OFFLINE = "offline"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthReport:
    state: LifecycleState
    detail: str = ""
    vram_mib: int = 0


@dataclass(frozen=True, slots=True)
class Capabilities:
    model_key: str
    kind: str
    labels: frozenset[str] = frozenset()
    vram_mib: int = 0
    batch_max: int = 1


class ModelRuntime(ABC):
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def warmup(self) -> None: ...

    @abstractmethod
    async def predict(self, request: object) -> object: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def health(self) -> HealthReport: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...
```

```python
# ai-engine/sentinel_ai/ports/frame_source.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameData:
    """A decoded frame. `pixels` is typed `object` so this port stays free of
    numpy — the architecture fitness test enforces that."""

    camera_id: str
    frame_index: int
    timestamp: float
    width: int
    height: int
    pixels: object


class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...

    @abstractmethod
    async def close(self) -> None: ...
```

```python
# ai-engine/sentinel_ai/ports/detector.py
from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Detection
from sentinel_ai.ports.frame_source import FrameData


class ObjectDetector(ABC):
    @abstractmethod
    async def detect(self, frame: FrameData) -> tuple[Detection, ...]: ...
```

```python
# ai-engine/sentinel_ai/ports/tracker.py
from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Detection, Track


class Tracker(ABC):
    @abstractmethod
    def update(
        self, detections: tuple[Detection, ...], timestamp: float
    ) -> tuple[Track, ...]: ...

    @abstractmethod
    def reset(self) -> None: ...
```

```python
# ai-engine/sentinel_ai/ports/vision_llm.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from sentinel_ai.domain.entities import SceneState
from sentinel_ai.ports.frame_source import FrameData


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """Everything spec §22 requires the VLM to receive."""

    keyframe: FrameData
    scene: SceneState
    history: tuple[str, ...]
    camera_label: str
    reason_detail: str


@dataclass(frozen=True, slots=True)
class SceneDescription:
    description: str
    threat_value: float
    suggested_action: str


class VisionLanguageModel(ABC):
    @abstractmethod
    async def describe(self, request: VisionRequest) -> SceneDescription: ...
```

```python
# ai-engine/sentinel_ai/ports/event_publisher.py
from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.entities import Event


class EventPublisher(ABC):
    @abstractmethod
    async def publish(self, event: Event) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
```

```python
# ai-engine/sentinel_ai/ports/clip_writer.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import UUID

from sentinel_ai.ports.frame_source import FrameData


class ClipWriter(ABC):
    @abstractmethod
    async def write(
        self, camera_id: str, event_id: UUID, frames: Sequence[FrameData], fps: float
    ) -> str:
        """Persist an evidence clip and return its URI."""
```

- [ ] **Step 4: Implement the fakes**

```python
# ai-engine/tests/fakes/models.py
"""Fakes that let the whole pipeline run on CPU in CI."""
from __future__ import annotations

from collections.abc import Sequence

from sentinel_ai.domain.entities import BBox, Detection, Track
from sentinel_ai.ports.detector import ObjectDetector
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.tracker import Tracker
from sentinel_ai.ports.vision_llm import (
    SceneDescription,
    VisionLanguageModel,
    VisionRequest,
)


class FakeDetector(ObjectDetector):
    """Replays a script of detections; repeats the last entry once exhausted."""

    def __init__(self, script: Sequence[tuple[Detection, ...]]) -> None:
        if not script:
            raise ValueError("script must contain at least one entry")
        self._script = list(script)
        self.call_count = 0

    async def detect(self, frame: FrameData) -> tuple[Detection, ...]:
        index = min(self.call_count, len(self._script) - 1)
        self.call_count += 1
        return self._script[index]


class FakeTracker(Tracker):
    """Nearest-centroid association — enough to exercise ageing and speed."""

    def __init__(self, match_radius_px: float = 80.0) -> None:
        self._match_radius = match_radius_px
        self._next_id = 1
        self._state: dict[int, tuple[str, BBox, int, float]] = {}

    def update(
        self, detections: tuple[Detection, ...], timestamp: float
    ) -> tuple[Track, ...]:
        from math import hypot

        unmatched = dict(self._state)
        results: list[Track] = []
        new_state: dict[int, tuple[str, BBox, int, float]] = {}

        for detection in detections:
            best_id: int | None = None
            best_distance = self._match_radius
            for track_id, (label, box, _age, _ts) in unmatched.items():
                if label != detection.label:
                    continue
                distance = hypot(detection.box.cx - box.cx, detection.box.cy - box.cy)
                if distance <= best_distance:
                    best_id, best_distance = track_id, distance

            if best_id is None:
                track_id = self._next_id
                self._next_id += 1
                age, speed = 1, 0.0
            else:
                track_id = best_id
                label, previous_box, previous_age, previous_ts = unmatched.pop(best_id)
                age = previous_age + 1
                elapsed = timestamp - previous_ts
                speed = (
                    hypot(
                        detection.box.cx - previous_box.cx,
                        detection.box.cy - previous_box.cy,
                    )
                    / elapsed
                    if elapsed > 0
                    else 0.0
                )

            new_state[track_id] = (detection.label, detection.box, age, timestamp)
            results.append(
                Track(
                    track_id=track_id,
                    label=detection.label,
                    box=detection.box,
                    age_frames=age,
                    speed_px_s=speed,
                )
            )

        self._state = new_state
        return tuple(results)

    def reset(self) -> None:
        self._state.clear()
        self._next_id = 1


class FakeVisionLLM(VisionLanguageModel):
    def __init__(
        self,
        response: SceneDescription | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or SceneDescription(
            description="Nothing notable in view.",
            threat_value=0.05,
            suggested_action="No action required.",
        )
        self._error = error
        self.requests: list[VisionRequest] = []

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._response
```

```python
# ai-engine/tests/fakes/io.py
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from uuid import UUID

from sentinel_ai.domain.entities import Event
from sentinel_ai.ports.clip_writer import ClipWriter
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.frame_source import FrameData, FrameSource


class FakeSource(FrameSource):
    def __init__(self, frames: Sequence[FrameData]) -> None:
        self._frames = list(frames)
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

    @classmethod
    def constant(
        cls, camera_id: str, count: int, fps: float = 10.0, value: int = 0
    ) -> FakeSource:
        return cls(
            [
                cls.make_frame(camera_id, index, index / fps, value)
                for index in range(count)
            ]
        )

    async def __aiter__(self) -> AsyncIterator[FrameData]:
        for frame in self._frames:
            yield frame

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


class FakeClipWriter(ClipWriter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, UUID, int]] = []

    async def write(
        self, camera_id: str, event_id: UUID, frames: Sequence[FrameData], fps: float
    ) -> str:
        self.calls.append((camera_id, event_id, len(frames)))
        return f"s3://sentinel-clips/{camera_id}/{event_id}.mp4"
```

- [ ] **Step 5: Run the tests**

Run: `cd ai-engine && pytest tests/ports -v && pytest -m "not gpu" && mypy`
Expected: all PASS. Create `ai-engine/tests/ports/__init__.py` if collection fails.

- [ ] **Step 6: Commit**

```bash
git add ai-engine/sentinel_ai/ports ai-engine/tests/ports ai-engine/tests/fakes
git commit -m "feat(ports): add port ABCs and CPU test fakes

ModelRuntime implements all seven §10 methods and all eight §5 lifecycle
states. FrameData.pixels is typed 'object' so the ports layer stays free of
numpy, keeping the architecture fitness test green."
```

---

### Task 9: Event contract and serialisation

**Files:**
- Create: `contracts/events/anomaly_event.schema.json`
- Create: `ai-engine/sentinel_ai/adapters/serialization/event_codec.py`
- Test: `ai-engine/tests/adapters/test_event_codec.py`

**Interfaces:**
- Consumes: `Event`, `ThreatScore`, `EscalationReason`, `Severity` (Task 2)
- Produces: `EVENT_SCHEMA_PATH: Path`; `encode_event(event: Event) -> dict[str, object]`; `decode_event(payload: Mapping[str, object]) -> Event`; `validate_payload(payload: Mapping[str, object]) -> None` raising `jsonschema.ValidationError`

- [ ] **Step 1: Write the failing test**

```python
# ai-engine/tests/adapters/test_event_codec.py
from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from jsonschema import ValidationError

from sentinel_ai.adapters.serialization.event_codec import (
    EVENT_SCHEMA_PATH,
    decode_event,
    encode_event,
    validate_payload,
)
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore


def make_event(**overrides: object) -> Event:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "occurred_at": 1_700_000_000.0,
        "reason": EscalationReason.SPEED_ANOMALY,
        "threat": ThreatScore.from_value(0.72),
        "description": "A person is running toward the gate.",
        "suggested_action": "Review the clip and confirm identity.",
        "labels": ("person",),
        "track_ids": (7,),
        "keyframe_uri": "s3://sentinel-clips/cam-1/key.jpg",
        "clip_uri": "s3://sentinel-clips/cam-1/clip.mp4",
    }
    return Event(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_the_schema_file_is_valid_json() -> None:
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")


def test_an_encoded_event_validates_against_the_committed_schema() -> None:
    validate_payload(encode_event(make_event()))


def test_encoding_flattens_the_threat_score() -> None:
    payload = encode_event(make_event())
    assert payload["threat_score"] == 0.72
    assert payload["severity"] == "high"


def test_encoding_never_leaks_model_identity() -> None:
    """Spec §3.3: the Web Platform must not learn which model produced a result."""
    payload = encode_event(make_event())
    serialized = json.dumps(payload).lower()
    for forbidden in ("yolo", "qwen", "whisper", "bytetrack", "awq"):
        assert forbidden not in serialized


def test_round_trip_preserves_every_field() -> None:
    original = make_event()
    restored = decode_event(encode_event(original))
    assert restored == original


def test_round_trip_preserves_a_minimal_event() -> None:
    original = make_event(
        labels=(), track_ids=(), keyframe_uri=None, clip_uri=None,
        description_unavailable=True, description="",
    )
    assert decode_event(encode_event(original)) == original


def test_event_id_is_serialized_as_a_uuid_string() -> None:
    payload = encode_event(make_event())
    assert isinstance(payload["event_id"], str)
    UUID(str(payload["event_id"]))


@pytest.mark.parametrize(
    "mutation",
    [
        {"threat_score": 1.5},
        {"severity": "catastrophic"},
        {"reason": "aliens"},
        {"camera_id": ""},
        {"event_id": "not-a-uuid"},
    ],
    ids=["score-out-of-range", "bad-severity", "bad-reason", "empty-camera", "bad-uuid"],
)
def test_invalid_payloads_are_rejected(mutation: dict[str, object]) -> None:
    payload = encode_event(make_event()) | mutation
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_a_payload_missing_a_required_field_is_rejected() -> None:
    payload = encode_event(make_event())
    del payload["threat_score"]
    with pytest.raises(ValidationError, match="threat_score"):
        validate_payload(payload)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ai-engine && pytest tests/adapters/test_event_codec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentinel_ai.adapters.serialization.event_codec'`

- [ ] **Step 3: Write the JSON Schema**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://sentinelai.dev/contracts/events/anomaly_event.schema.json",
  "title": "SentinelAI Anomaly Event",
  "description": "Published by the AI Engine to the Web Platform. Carries no model identity (spec §3.3).",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "event_id",
    "camera_id",
    "occurred_at",
    "reason",
    "threat_score",
    "severity",
    "description",
    "suggested_action",
    "labels",
    "track_ids",
    "description_unavailable",
    "metadata"
  ],
  "properties": {
    "schema_version": { "type": "integer", "const": 1 },
    "event_id": {
      "type": "string",
      "pattern": "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    },
    "camera_id": { "type": "string", "minLength": 1 },
    "occurred_at": { "type": "number" },
    "reason": {
      "type": "string",
      "enum": [
        "new_salient_track",
        "scene_change",
        "dwell_exceeded",
        "speed_anomaly",
        "track_count_spike",
        "periodic_summary",
        "user_requested"
      ]
    },
    "threat_score": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "severity": {
      "type": "string",
      "enum": ["info", "low", "medium", "high", "critical"]
    },
    "description": { "type": "string" },
    "suggested_action": { "type": "string" },
    "labels": { "type": "array", "items": { "type": "string" } },
    "track_ids": { "type": "array", "items": { "type": "integer" } },
    "keyframe_uri": { "type": ["string", "null"] },
    "clip_uri": { "type": ["string", "null"] },
    "description_unavailable": { "type": "boolean" },
    "metadata": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    }
  }
}
```

Save as `contracts/events/anomaly_event.schema.json`.

- [ ] **Step 4: Implement the codec**

```python
# ai-engine/sentinel_ai/adapters/serialization/event_codec.py
"""Event ⇄ wire-payload translation, validated against the committed JSON Schema.

Lives in adapters, not domain: it depends on jsonschema and on file layout.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from jsonschema import Draft202012Validator

from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore

SCHEMA_VERSION = 1

EVENT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3].parent
    / "contracts"
    / "events"
    / "anomaly_event.schema.json"
)


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(EVENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_payload(payload: Mapping[str, object]) -> None:
    _validator().validate(dict(payload))


def encode_event(event: Event) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(event.event_id),
        "camera_id": event.camera_id,
        "occurred_at": event.occurred_at,
        "reason": event.reason.value,
        "threat_score": event.threat.value,
        "severity": event.threat.severity.value,
        "description": event.description,
        "suggested_action": event.suggested_action,
        "labels": list(event.labels),
        "track_ids": list(event.track_ids),
        "keyframe_uri": event.keyframe_uri,
        "clip_uri": event.clip_uri,
        "description_unavailable": event.description_unavailable,
        "metadata": dict(event.metadata),
    }


def decode_event(payload: Mapping[str, object]) -> Event:
    validate_payload(payload)
    data = cast(dict[str, Any], dict(payload))
    return Event(
        event_id=UUID(data["event_id"]),
        camera_id=data["camera_id"],
        occurred_at=float(data["occurred_at"]),
        reason=EscalationReason(data["reason"]),
        threat=ThreatScore(
            value=float(data["threat_score"]), severity=Severity(data["severity"])
        ),
        description=data["description"],
        suggested_action=data["suggested_action"],
        labels=tuple(data["labels"]),
        track_ids=tuple(int(i) for i in data["track_ids"]),
        keyframe_uri=data.get("keyframe_uri"),
        clip_uri=data.get("clip_uri"),
        description_unavailable=bool(data["description_unavailable"]),
        metadata=dict(data["metadata"]),
    )
```

Create `ai-engine/sentinel_ai/adapters/__init__.py` and `ai-engine/sentinel_ai/adapters/serialization/__init__.py`.

- [ ] **Step 5: Run the tests**

Run: `cd ai-engine && pytest tests/adapters/test_event_codec.py -v && mypy`
Expected: all PASS (13 tests). Create `ai-engine/tests/adapters/__init__.py` if collection fails.

- [ ] **Step 6: Commit**

```bash
git add contracts/events ai-engine/sentinel_ai/adapters ai-engine/tests/adapters
git commit -m "feat(contracts): add event JSON Schema and validated codec

Includes a test asserting no encoded payload contains 'yolo', 'qwen',
'whisper', 'bytetrack' or 'awq' — spec §3.3 enforced at the wire boundary."
```

---

### Remaining tasks

Tasks 10–20 follow the same TDD structure. They are specified here as scoped deliverables with their interfaces; each will be expanded to full step-by-step form as its predecessor lands, so that the code in each task can be written against the *actual* signatures produced by the tasks before it rather than a guess.

| # | Task | Key deliverable | Depends on |
|---|---|---|---|
| 10 | Motion & scene-signature stage | `compute_motion_energy(prev, curr) -> float`, `compute_signature(frame, bins=16) -> tuple[float,...]`; numpy, no GPU | 8 |
| 11 | Frame sources + pre-roll buffer | `FileFrameSource`, `RtspFrameSource` (PyAV, `decode_hwaccel` flag, backoff reconnect), `PreRollBuffer(seconds, fps)` | 8 |
| 12 | YOLO11 detector adapter | `Yolo11Detector(ObjectDetector, ModelRuntime)`; COCO label mapping; `@pytest.mark.gpu` accuracy test + CPU-mocked unit tests | 8 |
| 13 | ByteTrack tracker adapter | `ByteTrackTracker(Tracker)` via supervision; speed derived from centroid delta / dt | 8 |
| 14 | Qwen2.5-VL-3B-AWQ adapter | `Qwen25VLAdapter(VisionLanguageModel, ModelRuntime)`; structured JSON prompt returning description + threat + action; timeout | 8 |
| 15 | Model registry + resident set | `ModelRegistry` (lifecycle transitions, health), `ResidentSet` executing `ResidencyPlan` | 7, 8 |
| 16 | Scheduler + orchestrator service | `Scheduler` (bounded queue, retry, timeout), `Orchestrator.describe_scene()` — the single entry point | 14, 15 |
| 17 | Pipeline runner | `PipelineRunner` wiring stages 1–5 + gate + escalation; OOM/timeout degradation per spec §9 | 6, 10–13, 16 |
| 18 | RabbitMQ publisher + MinIO clips | `RabbitMqPublisher` with disk spool + replay; `MinioClipWriter` writing 3 s pre-roll clips | 9, 11 |
| 19 | FastAPI app + OpenAPI export | `/health`, `/cameras/{id}/describe`, `/models`, `/telemetry`; `scripts/export_openapi.py` + CI drift check | 16 |
| 20 | Compose, datasets, E2E, Docker | `docker-compose.core.yml`, `mediamtx.yml`, `datasets/fetch.sh`, CPU end-to-end test, `Dockerfile`, `README.md` | all |

**Task 20's end-to-end test is the acceptance gate for this plan.** It must assert, with fakes and no GPU, that: a 60-second synthetic stream at 10 fps in which a running person is present throughout produces exactly **7** published events (not 600) — the count verified in Task 6's budget test; each event validates against the committed schema; each carries a clip URI; and killing the publisher mid-run spools events to disk and replays them on reconnect.

---

## Self-Review

**Spec coverage.** §3.1/§3.2 seam → Tasks 1, 15, 16. §3.3 decoupling → Tasks 1, 9. §3.5 transports → Tasks 18, 19. §4.1 gate → Tasks 4, 5, 6. §4.3 VRAM → Task 7. §5 lifecycle → Tasks 8, 15. §6.1 layout → all. §7 data flow → Task 17. §8 scope → this plan is 1A only. §9 error handling → Tasks 11, 17, 18. §10 testing → every task; Task 20 is the gate. §5 video sources → Tasks 11, 20.

**Gap found and closed:** the plan originally had no home for spec §9's "Postgres unavailable → consumer nacks" row. That belongs to the Go consumer, which is Plan 1B, not 1A. Recorded here so it is not lost.

**Type consistency verified:** `SceneState.signature_delta` is called in `triggers.scene_change` and `escalation._is_duplicate` with the same signature. `TokenBucket.try_consume` returns `(bool, TokenBucket)` and is destructured that way in `escalation.decide`. `TriggerContext` field names match between `triggers.py` and both call sites in `escalation.decide`. `FrameData` is consumed by `ObjectDetector.detect`, `VisionRequest.keyframe`, and `ClipWriter.write` with the same type.

**Three defects found and fixed during review:**

1. `GateState.dwell_anchors` originally used a `None` sentinel plus `object.__setattr__`. Unnecessary — a frozen slotted dataclass accepts `field(default_factory=dict)` (as `Event.metadata` already does). Simplified.
2. Two tests asserted 8 escalations per minute. Simulating the bucket gives **7** (burst at t=0.0/0.1, then t=10.0/20.0/30.1/40.1/50.1). Corrected in `test_rate_budget.py`, `test_escalation.py`, and Task 20's acceptance criterion.
3. **The significant one:** `TestPriority` and `TestBudgetGovernor` used a constant scene signature. Duplicate-suppression would therefore fire *before* the rate budget was ever reached, so `test_a_denied_decision_records_the_budget_as_the_suppressor` would have seen `suppressed_by == "duplicate_scene"` and the throughput test would have counted 1 escalation, not 7. Both now alternate signatures via the `alternating()` helper, with a comment explaining why any budget-targeting test must do so.

**Deliberate ordering consequence to be aware of when implementing Task 6:** because duplicate-suppression precedes the budget check, a genuinely static scene consumes no tokens at all. This is intended — an unchanged scene needs no new description — but it means dedup, not the bucket, is the dominant limiter on quiet cameras. Task 20's soak test against a real public webcam should report both suppression counts separately so the split is visible.
