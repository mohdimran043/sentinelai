"""Run N cameras against real models and report what it actually cost (spec §32).

What this measures, and what it does not
-----------------------------------------
It composes **the real engine** — `main.compose`, the same function
`create_default_app` calls — over N `FileSource` cameras replaying one clip, with the
three model ports wrapped in timers and the three *outbound* adapters replaced by
in-memory stand-ins (publisher, clip writer, notifier). Everything between the decoder
and the event is the shipped code.

The substitutions are deliberate and they bound what a number from here means:

* **No broker, no object store, no webhook.** A benchmark that also measured RabbitMQ
  and MinIO would be measuring the network, and its answer to "how many cameras fit on
  this GPU" would move when a disk got busy. Publish and clip-upload latency are a
  separate question from inference capacity, and conflating them makes neither
  answerable.
* **`realtime=True`, and this took a wrong turn first.** The obvious choice is
  `realtime=False`, letting `FileSource` replay as fast as the pipeline will take
  frames — "measure capacity, not the clip's frame rate". Measured that way the first
  sweep reported 4.0 fps at one camera with a detector p50 of 8 ms, which cannot both
  be true: 85 detections of 8 ms is 0.7 s of work inside 21 s of wall clock.

  The explanation is that `Yolo11Detector.detect` offloads to the **default**
  `ThreadPoolExecutor`, and so does PyAV's decode loop. Unthrottled, the decoder
  saturates that pool and starves the detector it is feeding. The number that came out
  was a measurement of decoder-versus-detector contention wearing capacity's clothes,
  and at 10 cameras it reported a detector p50 of 573 ms for a model that takes 4 ms.

  A real camera delivers at a fixed rate and cannot flood anything, so `realtime=True`
  is both the honest mode and the one that answers §14's actual question: at a fixed
  arrival rate, how many cameras can this box serve before it starts dropping frames?
  `drop_rate` is then the answer rather than an artefact.

So: the figures here are honest about inference cost under a realistic arrival rate,
and silent about I/O. §33's alert-latency target cannot be answered from this alone,
because most of that budget is the clip's post-roll and the broker round trip.

Why cameras are counted rather than estimated
----------------------------------------------
§14 asks for 20-50 cameras and §33 forbids claiming a number without measuring it. The
camera sweep exists so that the answer is a table with a point where it stops scaling,
rather than an extrapolation from one camera multiplied by N — which is exactly the
estimate that a shared GPU, a shared detector lock and a single VLM worker make wrong.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID, uuid4

from sentinel_ai import main
from sentinel_ai.adapters.publishers.inmemory import InMemoryPublisher
from sentinel_ai.benchmark.timing import (
    Recorder,
    TimedDetector,
    TimedFaceDetector,
    TimedPoseEstimator,
    TimedVisionLanguageModel,
)
from sentinel_ai.config import Settings
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.capabilities import CameraCapabilities, Capability, required_roles
from sentinel_ai.domain.entities import Event
from sentinel_ai.domain.identity import (
    AuthorizedPerson,
    EnrolledFace,
    FaceEmbedding,
    MatchCandidate,
    cosine_similarity,
)
from sentinel_ai.ports.event_publisher import FailedEventSink
from sentinel_ai.ports.face import FaceStore
from sentinel_ai.ports.notifier import Notifier, WelfareNote

__all__ = [
    "BenchmarkResult",
    "RunSpec",
    "default_capabilities",
    "parse_capabilities",
    "run_once",
    "run_sweep",
]


class _DiscardingSink(FailedEventSink):
    """Counts what could not be published instead of writing it anywhere.

    A benchmark that dead-lettered to disk would be measuring the disk. The count is
    still reported, because a run where events failed is a run whose throughput number
    means something different.
    """

    def __init__(self) -> None:
        self.stored = 0

    async def store(self, event: Event, error: BaseException) -> None:
        self.stored += 1


class _SilentNotifier(Notifier):
    """Counts notes. A webhook here would measure someone else's server."""

    def __init__(self) -> None:
        self.notified = 0

    async def notify(self, note: WelfareNote) -> None:
        self.notified += 1


class _EnrolledFaceStore(FaceStore):
    """A roster in memory, holding one synthetic person.

    `person_authorization` cannot be benchmarked without a store: `CameraRunner` skips
    the whole face pass when it has none, so a sweep with the capability enabled would
    measure a pipeline that never ran and report it as free.

    One enrolled person rather than none, because an **empty** roster is also not the
    workload. Every detected face would fail to match anything without a comparison
    being done, and the cost this sweep exists to measure includes the comparison.

    The embedding is fixed and arbitrary. Nothing here is a face — the benchmark replays
    a street clip, and whether the model recognises the pedestrians is the subject of
    `sentinel_ai/validation/faces.py`, not of a throughput measurement. No encryption
    either: nothing that reaches this object is biometric data about a real person, and
    it never touches a disk.
    """

    def __init__(self, dimension: int = 512) -> None:
        self._person = AuthorizedPerson(
            person_id=uuid4(),
            display_name="Benchmark subject",
            camera_ids=frozenset(),
            zones=frozenset(),
        )
        self._embedding = FaceEmbedding.of(tuple([0.0] * (dimension - 1) + [1.0]))

    async def add_person(self, person: AuthorizedPerson) -> None: ...

    async def get_person(self, person_id: UUID) -> AuthorizedPerson | None:
        return self._person if person_id == self._person.person_id else None

    async def list_people(self) -> tuple[AuthorizedPerson, ...]:
        return (self._person,)

    async def add_embedding(
        self, person_id: UUID, embedding: FaceEmbedding, *, image: bytes | None = None
    ) -> UUID:
        return uuid4()

    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        return ()

    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        return None

    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        return False

    async def reference_count(self, person_id: UUID) -> int:
        return 1

    async def delete_person(self, person_id: UUID) -> bool:
        return False

    async def search(
        self, embedding: FaceEmbedding, *, camera_id: str, zone: str | None, now: float
    ) -> tuple[MatchCandidate, ...]:
        return (
            MatchCandidate(
                person_id=self._person.person_id,
                display_name=self._person.display_name,
                similarity=cosine_similarity(embedding.vector, self._embedding.vector),
                authorized_here=self._person.is_authorized_on(
                    camera_id=camera_id, zone=zone, now=now
                ),
            ),
        )

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One point in the sweep."""

    video: Path
    cameras: int
    capabilities: CameraCapabilities
    seconds: float
    settings: Settings


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """What one run cost. Every field is measured, none is derived from a model's
    datasheet or from another run."""

    cameras: int
    load_seconds: float
    """How long `service.start()` took: downloading nothing (the weights are cached by
    then), placing every model, and running each one's warmup pass.

    Reported separately and **excluded from `wall_seconds`**, because an early version
    folded it in and produced a 60-second run reporting 149.9 s of wall clock and 9.6
    fps for a pipeline that was actually doing 25. Placing a 3B vision-language model
    takes longer than the measurement window, so including it does not merely add
    noise — it dominates the number it is added to.

    Worth reporting rather than discarding: it is the cold-start cost a deployment pays
    on every restart, and the reason `ResidentSet` keeps models warm.
    """

    wall_seconds: float
    """The measurement window only — after startup, before shutdown."""

    frames_processed: int
    frames_dropped: int
    escalations: int
    events_published: int
    recorder: Recorder
    peak_vram_mib: int

    @property
    def frames_per_second(self) -> float:
        """Aggregate across every camera — the number that answers "does this box keep
        up", where a per-camera figure would not."""
        return self.frames_processed / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        """The share of arrived frames that were overwritten before being processed.

        **The number that decides whether a camera count is real.** Throughput alone
        cannot say: a box at 100% GPU serving fifty cameras looks busy whether it is
        processing every frame or discarding four in five. `_LatestSlot` drops
        silently by design (it is what keeps the pipeline current rather than
        delayed), so this is the only place that design becomes visible.
        """
        total = self.frames_processed + self.frames_dropped
        return self.frames_dropped / total if total else 0.0

    def summary(self) -> str:
        lines = [
            f"cameras={self.cameras}  load={self.load_seconds:.1f}s  "
            f"wall={self.wall_seconds:.1f}s  "
            f"frames={self.frames_processed}  dropped={self.frames_dropped} "
            f"({self.drop_rate:.1%})  fps={self.frames_per_second:.1f}",
            f"escalations={self.escalations}  published={self.events_published}  "
            f"peak_vram={self.peak_vram_mib} MiB",
            self.recorder.table(),
        ]
        return "\n".join(lines)


def _peak_vram_mib() -> int:
    """Torch's own peak reservation, or 0 on a CPU box.

    `max_memory_reserved` rather than `max_memory_allocated`: the caching allocator's
    reserved pool is memory the driver has genuinely handed this process, which is what
    a second model has to fit alongside. Allocated-only would under-report and make a
    box look roomier than it is. Neither includes the CUDA context — see
    `adapters/detectors/yolo11.py` on the fixed overhead that covers.
    """
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_reserved() // (1024 * 1024))


def _reset_peak_vram() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


async def run_once(spec: RunSpec) -> BenchmarkResult:
    """Compose N cameras over one clip, run for `spec.seconds`, and report.

    The cameras all replay the same file, which is the point rather than a shortcut:
    the question is how many *decode-and-infer* workloads this box carries, and one
    clip played N times is N independent decodes of identical difficulty. Different
    clips would make each run's number depend on which footage landed on which camera.
    """
    settings = spec.settings
    roles = required_roles(*([spec.capabilities] * spec.cameras))
    models = main.build_models(settings, roles)

    recorder = Recorder()
    timed = replace(
        models,
        detector=(
            TimedDetector(models.detector, recorder) if models.detector is not None else None
        ),
        pose=(TimedPoseEstimator(models.pose, recorder) if models.pose is not None else None),
        vlm=(TimedVisionLanguageModel(models.vlm, recorder) if models.vlm is not None else None),
        face=(TimedFaceDetector(models.face, recorder) if models.face is not None else None),
    )

    cameras = tuple(
        main.CameraConfig(
            camera_id=f"bench_{index:02d}",
            label=f"Bench {index:02d}",
            url=str(spec.video),
            profile=CameraProfile(camera_id=f"bench_{index:02d}"),
            capabilities=spec.capabilities,
        )
        for index in range(spec.cameras)
    )

    publisher = InMemoryPublisher()
    composition = main.compose(
        settings,
        cameras,
        timed,
        publisher,  # type: ignore[arg-type]
        None,
        _DiscardingSink(),
        notifier=_SilentNotifier(),
        # Only when a camera asked for it, so a sweep without `person_authorization`
        # composes exactly the graph it did before this existed.
        face_store=_EnrolledFaceStore() if models.face is not None else None,
        # In memory, never on disk: a sweep that wrote an alert file would be measuring
        # the disk, and `AlertRegister` on its own is what the throughput numbers were
        # taken against.
        alert_store=None,
    )

    _reset_peak_vram()
    load_started = time.perf_counter()
    await composition.service.start()
    load_seconds = time.perf_counter() - load_started

    # The window starts *after* startup. See `BenchmarkResult.load_seconds` for the
    # measurement this ordering exists to stop being wrong.
    started = time.perf_counter()
    try:
        await asyncio.sleep(spec.seconds)
    finally:
        wall = time.perf_counter() - started
        await composition.service.stop()

    telemetry = composition.service.cameras()
    return BenchmarkResult(
        cameras=spec.cameras,
        load_seconds=load_seconds,
        wall_seconds=wall,
        frames_processed=sum(t.detections_run for t in telemetry),
        frames_dropped=sum(t.frames_dropped for t in telemetry),
        escalations=sum(t.escalations for t in telemetry),
        events_published=len(publisher.events),
        recorder=recorder,
        peak_vram_mib=_peak_vram_mib(),
    )


async def run_sweep(
    video: Path,
    counts: tuple[int, ...],
    *,
    seconds: float,
    capabilities: CameraCapabilities,
    settings: Settings | None = None,
) -> list[BenchmarkResult]:
    """Run each camera count in turn and return every result.

    Sequentially, never concurrently: two counts sharing the GPU would measure each
    other. The engine is fully stopped between points so that each run starts from the
    same residency state rather than inheriting the previous one's warm models — which
    would make the second point of every sweep look faster than the first for a reason
    that has nothing to do with camera count.
    """
    base = settings if settings is not None else Settings(source_realtime=True)
    results: list[BenchmarkResult] = []
    for count in counts:
        spec = RunSpec(
            video=video,
            cameras=count,
            capabilities=capabilities,
            seconds=seconds,
            settings=base,
        )
        results.append(await run_once(spec))
    return results


def parse_capabilities(raw: str) -> CameraCapabilities:
    """`"scene_description,anomaly_detection"` -> the set, or `"none"` for none.

    Fails loud on an unknown name, exactly as `cameras.json` does: a benchmark that
    silently dropped a capability would report the cost of a pipeline nobody asked for.
    """
    if raw.strip().lower() == "none":
        return CameraCapabilities.none()
    return CameraCapabilities.from_names([name.strip() for name in raw.split(",") if name.strip()])


def default_capabilities() -> CameraCapabilities:
    """Everything, so the default run measures the most expensive configuration.

    A benchmark defaulting to the cheap path would quote a camera count that no
    deployment using the product's headline features could reach.
    """
    return CameraCapabilities.of(
        Capability.SCENE_DESCRIPTION,
        Capability.ANOMALY_DETECTION,
        Capability.FALL_DETECTION,
    )
