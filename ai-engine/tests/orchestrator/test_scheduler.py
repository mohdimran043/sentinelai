from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import pytest
from jsonschema import ValidationError

from sentinel_ai.adapters.publishers.rabbitmq import RabbitMQPublisher
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.entities import (
    BBox,
    Detection,
    EscalationReason,
    Event,
    SceneState,
    Track,
)
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.event_history import RECENT_EVENTS_PER_CAMERA
from sentinel_ai.orchestrator.registry import ModelRegistry, ModelSpec
from sentinel_ai.orchestrator.resident_set import ResidentSet
from sentinel_ai.orchestrator.scheduler import (
    EscalationRequest,
    VlmScheduler,
    _is_out_of_memory,
)
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher, FailedEventSink
from sentinel_ai.ports.model_runtime import LifecycleState
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
from tests.fakes.io import FakeClipWriter, FakeFailedEventSink, FakePublisher, FakeSource
from tests.fakes.models import FakeModelRuntime, FakeVisionLLM

BOX = BBox(0.0, 0.0, 10.0, 10.0)


def _named_error(name: str) -> type[Exception]:
    """A synthetic exception class with the given name, standing in for
    `torch.cuda.OutOfMemoryError` — which cannot be imported here, since CI installs
    no `gpu` extra and `_is_out_of_memory` matches on the class name for exactly that
    reason."""
    return type(name, (RuntimeError,), {})


def clock() -> float:
    """A frozen clock. Nothing in the scheduler's own logic depends on time advancing,
    and freezing it makes it impossible for a test to lean on wall-clock progress."""
    return 0.0


class FlakyPublisher(EventPublisher):
    """Fails the first `fail_first` publishes, then succeeds — one publisher instance
    across a whole test, so no test has to reach into scheduler internals to swap it."""

    def __init__(self, fail_first: int) -> None:
        self.events: list[Event] = []
        self.attempts = 0
        self.closed = False
        self._fail_first = fail_first

    async def publish(self, event: Event) -> None:
        self.attempts += 1
        if self.attempts <= self._fail_first:
            raise ConnectionError("broker down")
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class Worker:
    """Runs the single scheduler worker for the duration of an `async with` block."""

    def __init__(self, scheduler: VlmScheduler) -> None:
        self._scheduler = scheduler
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> VlmScheduler:
        self._task = asyncio.create_task(self._scheduler.run())
        return self._scheduler

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._task is not None
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


class HangingVisionLLM(VisionLanguageModel):
    """A `describe` that never returns — the only shape that reaches the timeout.

    `FakeVisionLLM(error=TimeoutError(...))` looks like a timeout and is not one: it
    raises synchronously on the first `await`, so it exercises `_describe`'s
    `except Exception` clause and never `asyncio.timeout` at all. A hung GPU call is
    the failure `vlm_timeout_seconds` actually exists for, and it does not raise —
    it simply never completes.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False
        self._never_set = asyncio.Event()

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.started.set()
        try:
            await self._never_set.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable: the event is never set")  # pragma: no cover


def wall_clock() -> float:
    """A frozen wall clock, so `occurred_at` is exactly predictable in a test.

    A real `time.time()` here would make every epoch assertion below a moving target
    and would put a wall-clock read in CI for no benefit."""
    return 1_700_000_000.0


def a_request(
    clip: ClipHandle | None = None,
    reason: EscalationReason = EscalationReason.PERIODIC_SUMMARY,
    tracks: tuple[Track, ...] = (),
    timestamp: float = 12.5,
    camera_id: str = "cam-1",
) -> EscalationRequest:
    scene = SceneState(
        camera_id=camera_id,
        frame_index=0,
        timestamp=timestamp,
        detections=(Detection("person", 0.9, BOX),),
        tracks=tracks,
        motion_energy=0.1,
        scene_signature=(1.0,),
    )
    return EscalationRequest(
        camera_id=camera_id,
        event_id=uuid4(),
        reason=reason,
        detail="initial scene summary",
        scene=scene,
        keyframe=FakeSource.make_frame(camera_id, 0, 0.0),
        profile=CameraProfile(camera_id=camera_id),
        camera_label="Front Door",
        history=(),
        clip=clip,
    )


VLM_KEY = "qwen25vl3b"
VLM_SPEC = ModelSpec(model_key=VLM_KEY, vram_mib=4400, priority=50, idle_unload_seconds=600.0)


class RecordingResidentSet(ResidentSet):
    """The real `ResidentSet` over a real `ModelRegistry`, recording every `ensure()`.

    A real one rather than a stub: the point of routing the reload through
    `ResidentSet` is that `plan_residency`'s bookkeeping stays authoritative, and a
    stub would assert the call happened while proving nothing about its effect.
    """

    def __init__(self) -> None:
        registry = ModelRegistry()
        self.runtime = FakeModelRuntime(VLM_KEY, vram_mib=4400)
        registry.register(VLM_SPEC, self.runtime)
        super().__init__(registry, total_mib=8192, reserved_mib=2048)
        self.ensure_calls: list[tuple[tuple[str, ...], float]] = []

    async def ensure(self, required: tuple[str, ...], now: float) -> None:
        self.ensure_calls.append((required, now))
        await super().ensure(required, now)


class FailingResidentSet(RecordingResidentSet):
    async def ensure(self, required: tuple[str, ...], now: float) -> None:
        self.ensure_calls.append((required, now))
        raise RuntimeError("insufficient vram")


def new_scheduler(
    vlm: VisionLanguageModel | None = None,
    publisher: EventPublisher | None = None,
    maxsize: int = 4,
    admission: AdmissionGate | None = None,
    timeout_seconds: float = 5.0,
    resident_set: ResidentSet | None = None,
    dead_letter: FailedEventSink | None = None,
    clock: Callable[[], float] = clock,
    wall_clock: Callable[[], float] = wall_clock,
    recent_events_per_camera: int = RECENT_EVENTS_PER_CAMERA,
) -> VlmScheduler:
    return VlmScheduler(
        vlm=vlm
        or FakeVisionLLM(
            response=SceneDescription(
                description="A person is standing near the door.",
                threat_value=0.3,
                suggested_action="Monitor.",
            )
        ),
        publisher=publisher or FakePublisher(),
        admission=admission or AdmissionGate(concurrency=1, min_interval_seconds=0.0),
        resident_set=resident_set or RecordingResidentSet(),
        vlm_model_key=VLM_KEY,
        dead_letter=dead_letter or FakeFailedEventSink(),
        maxsize=maxsize,
        timeout_seconds=timeout_seconds,
        clock=clock,
        wall_clock=wall_clock,
        recent_events_per_camera=recent_events_per_camera,
    )


async def test_a_submitted_escalation_is_described_and_published() -> None:
    vlm = FakeVisionLLM(
        response=SceneDescription(
            description="A person is standing near the door.",
            threat_value=0.3,
            suggested_action="Monitor.",
        )
    )
    publisher = FakePublisher()
    async with Worker(new_scheduler(vlm=vlm, publisher=publisher)) as scheduler:
        assert scheduler.submit(a_request()) is True
        await scheduler.drain()

    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert event.description == "A person is standing near the door."
    assert event.suggested_action == "Monitor."
    assert event.threat.value == pytest.approx(0.3)
    assert event.description_unavailable is False
    assert event.source_timestamp == pytest.approx(12.5), "the scene's timestamp, not the clock's"
    assert vlm.call_count == 1


async def test_the_vlm_receives_the_keyframe_history_and_reason_detail() -> None:
    vlm = FakeVisionLLM()
    async with Worker(new_scheduler(vlm=vlm)) as scheduler:
        scheduler.submit(a_request())
        await scheduler.drain()

    assert vlm.requests[0].camera_label == "Front Door"
    assert vlm.requests[0].reason_detail == "initial scene summary"


async def test_the_event_carries_the_scene_labels_and_track_ids() -> None:
    tracks = (
        Track(track_id=7, label="person", box=BOX, age_frames=9, speed_px_s=1.0),
        Track(track_id=3, label="car", box=BOX, age_frames=4, speed_px_s=2.0),
    )
    publisher = FakePublisher()
    async with Worker(new_scheduler(publisher=publisher)) as scheduler:
        scheduler.submit(a_request(tracks=tracks))
        await scheduler.drain()

    assert publisher.events[0].labels == ("car", "person"), "sorted and de-duplicated"
    assert publisher.events[0].track_ids == (7, 3)


async def test_a_clip_is_finished_and_its_uri_attached() -> None:
    writer = FakeClipWriter()
    handle = await writer.open("cam-1", uuid4(), fps=10.0)
    publisher = FakePublisher()
    async with Worker(new_scheduler(publisher=publisher)) as scheduler:
        scheduler.submit(a_request(clip=handle))
        await scheduler.drain()

    assert publisher.events[0].clip_uri == f"s3://sentinel-clips/cam-1/{handle.event_id}.mp4"
    assert handle.finished is True


async def test_the_console_ring_learns_the_clip_uri_too() -> None:
    """T3. The published event has carried the clip URI since Task 12; the ring did
    not, because it is written at assembly and the clip is attached after. A console
    that cannot link an event to its footage has to guess at the store's key layout."""
    writer = FakeClipWriter()
    handle = await writer.open("cam-1", uuid4(), fps=10.0)
    async with Worker(new_scheduler()) as scheduler:
        scheduler.submit(a_request(clip=handle))
        await scheduler.drain()

    latest = scheduler.event_history("cam-1").latest
    assert latest is not None
    assert latest.clip_uri == f"s3://sentinel-clips/cam-1/{handle.event_id}.mp4"


async def test_a_clip_that_fails_leaves_the_ring_entry_present_and_uri_less() -> None:
    """Spec §9's rule survives T3 intact: the event is recorded at assembly, *before*
    the clip is attempted, so a clip that never finishes costs the console its link to
    the footage and nothing else. Recording after the attach instead would delete this
    event from the console entirely — the one an operator most needs to see.
    """
    writer = FakeClipWriter(finish_error=OSError("minio unreachable"))
    handle = await writer.open("cam-1", uuid4(), fps=10.0)
    publisher = FakePublisher()
    async with Worker(new_scheduler(publisher=publisher)) as scheduler:
        scheduler.submit(a_request(clip=handle))
        await scheduler.drain()

    latest = scheduler.event_history("cam-1").latest
    assert latest is not None, "§9: a failed clip must never cost the event"
    assert latest.clip_uri is None, "and it must not claim a clip that was never written"
    assert publisher.events[0].clip_uri is None


async def test_no_clip_leaves_clip_uri_none() -> None:
    publisher = FakePublisher()
    async with Worker(new_scheduler(publisher=publisher)) as scheduler:
        scheduler.submit(a_request(clip=None))
        await scheduler.drain()

    assert publisher.events[0].clip_uri is None


async def test_a_full_queue_drops_and_counts_without_raising() -> None:
    scheduler = new_scheduler(maxsize=1)
    # No worker running: nothing drains the queue, so the second submit finds it full.
    assert scheduler.submit(a_request()) is True
    assert scheduler.submit(a_request()) is False
    assert scheduler.dropped == 1


class OomVisionLLM(VisionLanguageModel):
    """Raises an out-of-memory error for the first `fail_first` describes.

    The error is a bare `RuntimeError` carrying torch's real message rather than a
    `torch.cuda.OutOfMemoryError`: the CI box has no `gpu` extra, and torch itself
    raises this shape from several call sites, so it is the harder case to detect.
    """

    def __init__(self, fail_first: int) -> None:
        self.fail_first = fail_first
        self.calls = 0

    async def describe(self, request: VisionRequest) -> SceneDescription:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total "
                "capacity of 7.99 GiB of which 21.06 MiB is free."
            )
        return SceneDescription(
            description="A person is standing near the door.",
            threat_value=0.3,
            suggested_action="Monitor.",
        )


class TestVlmOutOfMemory:
    """Spec §9's VLM-OOM row: evict LRU -> retry once -> mark unhealthy -> pipeline
    continues detection-only. None of the four clauses existed before; a CUDA OOM was
    caught by the broad `except Exception`, degraded to `description_unavailable=True`
    and forgotten, so the model stayed resident on a card it had just exhausted."""

    async def test_an_oom_evicts_retries_once_and_publishes_a_real_description(self) -> None:
        """The clause that keeps the description. Fails against the old scheduler with
        `description_unavailable is True`, `vlm.calls == 1` and no eviction at all."""
        resident_set = RecordingResidentSet()
        vlm = OomVisionLLM(fail_first=1)
        publisher = FakePublisher()
        scheduler = new_scheduler(vlm=vlm, publisher=publisher, resident_set=resident_set)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert vlm.calls == 2, "exactly one retry, not zero and not a loop"
        assert resident_set.runtime.shutdown_calls == 1, "the eviction must actually unload"
        assert resident_set.runtime.initialize_calls == 2, "and the retry must reload"
        assert resident_set.resident() == frozenset({VLM_KEY})
        assert publisher.events[0].description_unavailable is False
        assert publisher.events[0].description == "A person is standing near the door."
        assert resident_set.runtime.health().state is LifecycleState.HEALTHY

    async def test_a_second_oom_marks_the_model_unhealthy_and_still_publishes(self) -> None:
        """The clause that makes the fault visible. `/health` previously reported the
        post-eviction `UNLOADED`, which an operator reads as *correctly idle* rather
        than as a card that cannot hold the model."""
        resident_set = RecordingResidentSet()
        vlm = OomVisionLLM(fail_first=2)
        publisher = FakePublisher()
        scheduler = new_scheduler(vlm=vlm, publisher=publisher, resident_set=resident_set)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert vlm.calls == 2, "one retry only — a second OOM is not retried again"
        report = resident_set.runtime.health()
        assert report.state is LifecycleState.UNHEALTHY
        assert "out of memory on two consecutive describes" in report.detail
        assert resident_set.resident() == frozenset(), "the model is off the card"
        # Spec §9 all the same: the event survives, degraded.
        assert len(publisher.events) == 1
        assert publisher.events[0].description_unavailable is True

    async def test_the_pipeline_keeps_going_and_can_recover(self) -> None:
        """Detection-only is not "the VLM is off until restart":
        the next escalation's `ensure()` reloads it, and if the card has room it
        describes again."""
        resident_set = RecordingResidentSet()
        vlm = OomVisionLLM(fail_first=2)
        publisher = FakePublisher()
        scheduler = new_scheduler(vlm=vlm, publisher=publisher, resident_set=resident_set)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()
            scheduler.submit(a_request())
            await scheduler.drain()

        assert [e.description_unavailable for e in publisher.events] == [True, False]
        assert resident_set.resident() == frozenset({VLM_KEY})
        assert resident_set.runtime.health().state is LifecycleState.HEALTHY

    async def test_a_non_oom_failure_is_not_retried(self) -> None:
        """A timeout or a malformed reply is one bad escalation. Retrying it would
        spend the process's only GPU slot twice for the same expected outcome, and
        would evict a model that has nothing wrong with it."""
        resident_set = RecordingResidentSet()
        vlm = FakeVisionLLM(error=TimeoutError("vlm timed out"))
        publisher = FakePublisher()
        scheduler = new_scheduler(vlm=vlm, publisher=publisher, resident_set=resident_set)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert vlm.call_count == 1
        assert resident_set.runtime.shutdown_calls == 0, "nothing was evicted"
        assert resident_set.runtime.health().state is not LifecycleState.UNHEALTHY
        assert publisher.events[0].description_unavailable is True

    @pytest.mark.parametrize(
        "error",
        [
            MemoryError("out of memory"),
            RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB"),
            _named_error("OutOfMemoryError")("CUDA driver ran dry"),
        ],
    )
    def test_every_shape_a_cuda_oom_arrives_in_is_recognised(self, error: Exception) -> None:
        """torch raises `torch.cuda.OutOfMemoryError` from some paths and a plain
        `RuntimeError` with the same message from others, and this module cannot import
        torch to check the type — CI has no `gpu` extra."""
        assert _is_out_of_memory(error) is True

    @pytest.mark.parametrize(
        "error",
        [TimeoutError("vlm timed out"), ValueError("threat_value 1.4 out of range")],
    )
    def test_an_ordinary_failure_is_not_mistaken_for_an_oom(self, error: Exception) -> None:
        assert _is_out_of_memory(error) is False


class TestVlmResidency:
    """C1: the describe path is the only thing that can keep the VLM alive, and it was
    not wired to `ResidentSet` at all — the scheduler held no reference to one."""

    async def test_every_describe_makes_the_vlm_resident_first(self) -> None:
        """`ResidentSet`'s own module docstring specifies this contract verbatim —
        "a caller invoking a model ... wrapping a `describe` call with
        `await resident_set.ensure((vlm_key,), now)` first" — and the caller was never
        written. Both halves matter: the call loads a VLM the idle sweeper has evicted,
        and it stamps `last_used_at`, without which the 600s idle window measures time
        since boot rather than time since last use.
        """
        resident_set = RecordingResidentSet()
        publisher = FakePublisher()
        scheduler = new_scheduler(publisher=publisher, resident_set=resident_set)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()
            scheduler.submit(a_request())
            await scheduler.drain()

        assert resident_set.ensure_calls == [((VLM_KEY,), 0.0), ((VLM_KEY,), 0.0)], (
            "every describe, not merely the first, must freshen the idle clock"
        )
        assert resident_set.resident() == frozenset({VLM_KEY})
        assert len(publisher.events) == 2

    async def test_an_evicted_vlm_is_brought_back_rather_than_degrading_forever(
        self,
    ) -> None:
        """The self-heal, at the level the scheduler owns: after an idle eviction the
        very next escalation must reload the model, not publish a stub description.

        Fails against a scheduler with no `ResidentSet`: the runtime stays UNLOADED,
        `initialize_calls` stays at 1, and the event carries
        `description_unavailable=True` for the lifetime of the process.
        """
        resident_set = RecordingResidentSet()
        await resident_set.ensure((VLM_KEY,), 0.0)
        await resident_set.sweep_idle(1_000.0)  # past idle_unload_seconds=600
        assert resident_set.resident() == frozenset()
        assert resident_set.runtime.shutdown_calls == 1

        publisher = FakePublisher()
        scheduler = new_scheduler(publisher=publisher, resident_set=resident_set)
        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert resident_set.runtime.initialize_calls == 2, "the describe must reload it"
        assert resident_set.resident() == frozenset({VLM_KEY})
        assert publisher.events[0].description_unavailable is False

    async def test_a_residency_failure_costs_the_description_but_never_the_event(
        self,
    ) -> None:
        """Spec §9 again, one layer further out. A reload can genuinely fail — the card
        may have filled up since boot — and it happens inside the admission-gated
        section, so raising from there would both lose the event and, at concurrency 1,
        risk wedging the only GPU slot.
        """
        admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(
            vlm=FakeVisionLLM(error=RuntimeError("describe called before initialize()")),
            publisher=publisher,
            admission=admission,
            resident_set=FailingResidentSet(),
        )

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()
            assert len(publisher.events) == 1
            assert publisher.events[0].description_unavailable is True
            assert admission.in_flight == 0

            # The worker must still be usable, not wedged on a leaked slot.
            scheduler.submit(a_request())
            await scheduler.drain()

        assert len(publisher.events) == 2


class TestNeverLoseAnEvent:
    """Spec §9's governing rule: an anomaly event is never lost to an infrastructure
    failure. Each case breaks a different piece of infrastructure and still demands
    an `Event` at the publisher."""

    async def test_a_failing_vlm_still_publishes_a_metadata_derived_description(self) -> None:
        tracks = (Track(track_id=1, label="person", box=BOX, age_frames=9, speed_px_s=1.0),)
        publisher = FakePublisher()
        scheduler = new_scheduler(
            vlm=FakeVisionLLM(error=TimeoutError("vlm timed out")), publisher=publisher
        )
        async with Worker(scheduler):
            scheduler.submit(a_request(reason=EscalationReason.NEW_SALIENT_TRACK, tracks=tracks))
            await scheduler.drain()

        assert len(publisher.events) == 1
        event = publisher.events[0]
        assert event.description_unavailable is True
        assert "person" in event.description, "the fallback is built from the scene's labels"
        assert event.reason is EscalationReason.NEW_SALIENT_TRACK
        assert event.threat.value == pytest.approx(0.5)

    async def test_a_hanging_vlm_is_timed_out_and_still_publishes_a_fallback_event(self) -> None:
        """`vlm_timeout_seconds` is shipped at 30.0 and, before this test, nothing
        exercised it: a mutation sweep showed that deleting the `asyncio.timeout`
        wrapper — and independently hardcoding it to 1e9 — both survived the full
        353-test suite. A hung `describe` at concurrency 1 holds the only admission
        slot, so without the timeout the camera never escalates again for the lifetime
        of the process; §9's guarantee that the event survives depends on the wrapper.

        `timeout_seconds=0.0` costs no wall-clock time at all: `asyncio.timeout`
        schedules the deadline at `loop.time()`, which the loop reaches on its very
        next pass. The `wait_for` around `drain()` is the failure path only — against
        a scheduler with no timeout the worker never finishes this request, and this
        test must fail rather than hang the suite.
        """
        vlm = HangingVisionLLM()
        publisher = FakePublisher()
        admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
        scheduler = new_scheduler(
            vlm=vlm, publisher=publisher, admission=admission, timeout_seconds=0.0
        )

        async with Worker(scheduler):
            scheduler.submit(a_request())
            try:
                await asyncio.wait_for(scheduler.drain(), timeout=5.0)
            except TimeoutError:
                pytest.fail("the hung describe() was never timed out: vlm_timeout_seconds is dead")

        assert vlm.started.is_set(), "test setup: describe() must actually have been entered"
        assert vlm.cancelled is True, "the timeout must cancel the call, not merely abandon it"
        assert len(publisher.events) == 1, "§9: a hung VLM must not cost the event"
        assert publisher.events[0].description_unavailable is True
        assert publisher.events[0].threat.value == pytest.approx(0.5)
        assert admission.in_flight == 0, "a timed-out call must not leak the GPU slot"

    async def test_a_clip_that_fails_to_finish_still_publishes_with_no_uri(self) -> None:
        writer = FakeClipWriter(finish_error=OSError("minio unreachable"))
        handle = await writer.open("cam-1", uuid4(), fps=10.0)
        publisher = FakePublisher()
        async with Worker(new_scheduler(publisher=publisher)) as scheduler:
            scheduler.submit(a_request(clip=handle))
            await scheduler.drain()

        assert len(publisher.events) == 1
        assert publisher.events[0].clip_uri is None
        assert publisher.events[0].description_unavailable is False, (
            "a clip failure says nothing about whether the description succeeded"
        )
        assert handle.aborted, (
            "a failed finish() can leave the muxer's container open and its temp file "
            "on disk; abort() is idempotent and never raises, so the worker must call it"
        )

    async def test_an_out_of_range_threat_value_degrades_instead_of_dropping_the_event(
        self,
    ) -> None:
        """`SceneDescription.threat_value` is a bare unvalidated float and
        `ThreatScore.from_value` rejects anything outside [0, 1]. Task 13's Qwen adapter
        parses that number out of generated model text, so a model that emits `1.2` is
        an infrastructure failure like any other — and §9 does not exempt it.

        Fails against a `_describe` that builds the success-path `Event` outside the
        `try`: `from_value` then raises out of the success path, past `_process`, into
        the worker's top-level handler, which logs it and moves on. The event is gone —
        `publisher.events` stays empty.
        """
        publisher = FakePublisher()
        scheduler = new_scheduler(
            vlm=FakeVisionLLM(
                response=SceneDescription(
                    description="A person is standing near the door.",
                    threat_value=1.2,
                    suggested_action="Monitor.",
                )
            ),
            publisher=publisher,
        )
        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert len(publisher.events) == 1, "an unusable threat value must not lose the event"
        event = publisher.events[0]
        assert event.description_unavailable is True
        assert event.threat.value == pytest.approx(0.5), "the conservative mid-range placeholder"

    async def test_a_vlm_exception_other_than_timeout_still_publishes_and_frees_admission(
        self,
    ) -> None:
        """§9's rescue path must not narrow to `except TimeoutError` — any VLM failure
        mode (a provider outage, a CUDA OOM) must still yield a fallback event, and the
        admission slot it held must still be free for the next escalation.

        Fails against a `_describe` that only catches `TimeoutError`: the `RuntimeError`
        here would propagate out of `_process` uncaught by anything but the worker's own
        top-level `except Exception`, so no event would ever reach the publisher and
        `publisher.events` would stay empty after `drain()`.
        """
        admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
        publisher = FakePublisher()
        scheduler = new_scheduler(
            vlm=FakeVisionLLM(error=RuntimeError("cuda oom")),
            publisher=publisher,
            admission=admission,
        )
        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()
            assert len(publisher.events) == 1
            assert publisher.events[0].description_unavailable is True
            assert admission.in_flight == 0

            # Proof the gate is genuinely reusable, not merely reporting zero —
            # would hang forever if the first call ever leaked the slot.
            scheduler.submit(a_request())
            await scheduler.drain()

        assert len(publisher.events) == 2
        assert admission.in_flight == 0


class TestAPublisherFailureNeverLosesTheEvent:
    """Spec §9's last mile. `RabbitMQPublisher` spools when the *broker* is down and
    returns normally, so that path is already safe and must not be handled twice. The
    three paths that *raise* out of `publish()` all used to end with the event existing
    nowhere and every counter reading zero.

    Each case below runs the real `RabbitMQPublisher`, not a fake that raises: the
    point is that these are reachable through the shipped adapter.
    """

    @staticmethod
    def _publisher(spool_dir: Path) -> RabbitMQPublisher:
        return RabbitMQPublisher(
            url="amqp://unused", exchange="sentinel.events", spool_dir=spool_dir
        )

    async def test_the_ordinary_broker_outage_still_spools_and_is_not_double_handled(
        self, tmp_path: Path
    ) -> None:
        """The boundary. A spooling publish returns normally, so the dead-letter sink
        must stay untouched — otherwise every offline event would be recorded twice,
        once as recoverable and once as lost."""
        publisher = self._publisher(tmp_path / "spool")
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(publisher=publisher, dead_letter=dead_letter)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert len(list((tmp_path / "spool").glob("*.json"))) == 1
        assert dead_letter.stored == []
        assert scheduler.publish_failures == 0

    async def test_a_schema_invalid_event_is_kept_instead_of_vanishing(
        self, tmp_path: Path
    ) -> None:
        """`encode_event` validates *before* the transport guard, so a payload the
        schema rejects raises straight out of `publish()` — past the spool. Provoked
        here with an empty `camera_id`, which `Event` permits and the schema does not.

        Fails against the old scheduler with `dead_letter.stored == []` and no spool
        file either: the event was gone.
        """
        publisher = self._publisher(tmp_path / "spool")
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(publisher=publisher, dead_letter=dead_letter)
        request = replace(a_request(), camera_id="")

        async with Worker(scheduler):
            scheduler.submit(request)
            await scheduler.drain()

        assert list((tmp_path / "spool").glob("*.json")) == [], "the schema rejected it"
        assert [event.event_id for event in dead_letter.events] == [request.event_id]
        assert isinstance(dead_letter.stored[0][1], ValidationError)
        assert scheduler.publish_failures == 1

    async def test_a_spool_that_cannot_be_written_is_not_the_end_of_the_event(
        self, tmp_path: Path
    ) -> None:
        """`_spool` is called *from* the transport-error handler and from the
        `_exchange is None` branch, neither of which guards it, so an `OSError` there
        escapes `publish()` entirely — the disk-backed buffer failing takes the event
        with it."""
        spool = tmp_path / "spool"
        publisher = self._publisher(spool)
        spool.chmod(0o500)  # readable, not writable
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(publisher=publisher, dead_letter=dead_letter)
        try:
            async with Worker(scheduler):
                scheduler.submit(a_request())
                await scheduler.drain()
        finally:
            spool.chmod(0o700)

        assert isinstance(dead_letter.stored[0][1], OSError)
        assert scheduler.publish_failures == 1

    async def test_an_unexpected_publisher_error_is_kept_and_counted_apart_from_drops(
        self,
    ) -> None:
        """`dropped` counts the queue rejecting an escalation under load, which the
        governors make expected. This is an assembled event that could not be
        delivered, and it needs its own number: all three paths above previously left
        `dropped == 0`, so the loss appeared in no counter at all."""
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(
            publisher=FakePublisher(error=RuntimeError("transport exploded")),
            dead_letter=dead_letter,
        )

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

        assert scheduler.publish_failures == 1
        assert scheduler.dropped == 0
        assert len(dead_letter.stored) == 1

    async def test_a_dead_letter_sink_that_itself_fails_does_not_kill_the_worker(
        self,
    ) -> None:
        """The port says `store` must not raise. If one does anyway, the loss is
        already unavoidable — but it must not also stop every later escalation from
        being described, and it must not leak the admission slot."""
        admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
        publisher = FlakyPublisher(fail_first=1)
        scheduler = new_scheduler(
            publisher=publisher,
            admission=admission,
            dead_letter=FakeFailedEventSink(error=OSError("disk full")),
        )

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()
            scheduler.submit(a_request())
            await scheduler.drain()

        assert len(publisher.events) == 1, "the worker survived and published the second"
        assert admission.in_flight == 0


class TestAdmissionSlotIsNeverLeaked:
    async def test_a_publish_failure_still_releases_the_slot_and_the_worker_survives(
        self,
    ) -> None:
        """A leaked slot wedges the GPU permanently: at concurrency 1 the very next
        escalation would block on the semaphore forever.

        Fails against a `release()` that is not in a `finally` — `publish` raises before
        it would be reached — and against a worker loop that dies on the first failure,
        because the second request would then never be described at all.
        """
        admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
        publisher = FlakyPublisher(fail_first=1)
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(publisher=publisher, admission=admission, dead_letter=dead_letter)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

            assert publisher.attempts == 1
            assert publisher.events == [], "the first publish failed"
            # This assertion used to stop at the line above, which made the loss the
            # specification: spec §9 says the event survives an infrastructure
            # failure, and the publisher refusing it is one.
            assert len(dead_letter.stored) == 1, "the refused event is kept, not dropped"
            assert scheduler.publish_failures == 1
            assert admission.in_flight == 0, "the slot must be released even on failure"

            # Proof the gate is genuinely reusable, not merely reporting zero.
            scheduler.submit(a_request())
            await scheduler.drain()

        assert publisher.attempts == 2
        assert len(publisher.events) == 1
        assert admission.in_flight == 0


class TestEventTimestamps:
    """D1: `occurred_at` is Unix epoch seconds; the source timeline is kept beside it.

    `occurred_at` used to be `scene.timestamp` verbatim — `time.monotonic()` on RTSP.
    A live run published `"occurred_at": 51181.128868795`: a number that resets on
    every restart, shares no origin with a replay camera in the same process, and
    cannot be converted to a wall time, while `event_codec`'s docstring tells the Go
    consumer to sort by it.

    A first fix anchored the whole *process* once, at construction, to
    `(wall_clock(), clock())`. That is correct only when every camera's
    `scene.timestamp` shares `clock`'s timeline, which is true for `RtspSource`
    (`time.monotonic()`) and false for `FileSource` (container pts, starting at 0.0
    per file): a replay event came out ~one uptime in the past, and a replay camera
    and an RTSP camera in the same process landed hours apart for events observed
    seconds apart. The fix here anchors **per camera, lazily, from that camera's own
    first-seen `source_timestamp`** — not from `clock()` — so the anchor is always
    correct for whatever timeline that particular source counts on, at the cost of
    only being precise to when each camera's anchor was established (see
    `_to_epoch`'s docstring).
    """

    WALL_AT_ANCHOR = 1_700_000_000.0

    def _stepping_wall_clock(self) -> Callable[[], float]:
        """A wall clock that jumps 100s every time it is read.

        Any implementation that samples it per event rather than once per camera
        anchor gets caught twice over: the absolute value is wrong, and — the part
        that matters — the gap between two events from the *same* camera stops
        matching the gap between their source timestamps, which is exactly how an
        NTP step reorders two events relative to each other.
        """
        reads = iter(self.WALL_AT_ANCHOR + 100.0 * step for step in range(100))
        return lambda: next(reads)

    def _scheduler(self, publisher: FakePublisher) -> VlmScheduler:
        return new_scheduler(
            publisher=publisher,
            # A real elapsed-time reading standing in for a box with hours of
            # uptime. Fixed and unrelated to any `source_timestamp` below, so any
            # test failure that depends on it proves this value leaked into
            # `occurred_at` — which the per-camera anchor must never let happen.
            clock=lambda: 51_181.0,
            wall_clock=self._stepping_wall_clock(),
        )

    async def test_a_cameras_first_event_is_anchored_at_its_own_assembly(self) -> None:
        """The per-camera anchor: a camera's first event is stamped at the wall clock
        reading taken when *it* was assembled, regardless of what its own
        `source_timestamp` or the scheduler's `clock()` read."""
        publisher = FakePublisher()
        async with Worker(self._scheduler(publisher)) as scheduler:
            scheduler.submit(a_request(timestamp=12.5))
            await scheduler.drain()

        assert publisher.events[0].occurred_at == pytest.approx(self.WALL_AT_ANCHOR)
        assert publisher.events[0].occurred_at > 1_600_000_000.0, (
            "a plausible Unix epoch, not a monotonic reading"
        )

    async def test_two_events_are_exactly_their_source_delta_apart(self) -> None:
        """The property the anchor exists for, unchanged by the per-camera fix: two
        events from the *same* camera differ by exactly their source-timestamp delta.
        Fails against a `time.time()` read per event: the wall clock steps 100s
        between the two assemblies here, so the pair would come out 108s apart
        instead of 8s — and a backwards step would put them in the wrong order.
        Fails equally against anything that re-anchors on the *second* event instead
        of reusing the camera's first anchor."""
        publisher = FakePublisher()
        async with Worker(self._scheduler(publisher)) as scheduler:
            scheduler.submit(a_request(timestamp=12.5))
            await scheduler.drain()
            scheduler.submit(a_request(timestamp=20.5))
            await scheduler.drain()

        first, second = (event.occurred_at for event in publisher.events)
        assert second - first == pytest.approx(8.0), (
            "exactly the source delta — the anchor is read once per camera, not per event"
        )
        assert first == pytest.approx(self.WALL_AT_ANCHOR), (
            "the wall clock must be read once, for the first event of this camera, "
            "not again for the second"
        )

    async def test_the_raw_source_timeline_is_kept_alongside(self) -> None:
        """Rebasing must not throw the source timeline away: it is what correlates an
        event with its clip's pts and with `CameraTelemetry`. Fails against an
        implementation that only converts."""
        publisher = FakePublisher()
        async with Worker(self._scheduler(publisher)) as scheduler:
            scheduler.submit(a_request(timestamp=12.5))
            await scheduler.drain()

        assert publisher.events[0].source_timestamp == pytest.approx(12.5)

    async def test_an_abandoned_escalation_is_stamped_the_same_way(self) -> None:
        """`_assemble` is the single construction site (S14) precisely so no error path
        can produce a differently-shaped event. The §9 fallback must carry both
        timelines too, or a dead-lettered event is the one an operator cannot place in
        time."""
        dead_letter = FakeFailedEventSink()
        scheduler = new_scheduler(
            dead_letter=dead_letter,
            clock=lambda: 51_181.0,
            wall_clock=self._stepping_wall_clock(),
        )
        assert scheduler.submit(a_request(timestamp=12.5))
        assert await scheduler.abandon_pending(RuntimeError("shutdown")) == 1

        stored = dead_letter.events[0]
        assert stored.occurred_at == pytest.approx(self.WALL_AT_ANCHOR)
        assert stored.source_timestamp == pytest.approx(12.5)

    async def test_a_replay_source_starting_near_zero_lands_at_assembly_wall_time(self) -> None:
        """The defect this class exists to catch. `FileSource` stamps `scene.timestamp`
        from the container's own pts, starting at 0.0 per file. The process-wide
        anchor computed `occurred_at = wall_at_anchor + (source_timestamp -
        mono_at_anchor)` with `mono_at_anchor` a real `time.monotonic()` reading —
        i.e. seconds since boot, unrelated to this camera. On a box with ~14.2 hours
        of uptime (`clock() == 51_181.0`, the live run's own reading) a replay event
        with `source_timestamp == 0.02` used to land at `wall_at_anchor - 51_180.98`:
        about 14 hours in the past. Per-camera anchoring fixes it: the camera's
        first-seen `source_timestamp` *is* its own anchor, so the offset is always 0
        for that first event, however small `source_timestamp` is and however large
        `clock()` reads.

        Fails against the process-wide anchor with `occurred_at` off by ~51_181
        seconds (~14.2 hours in the past); passes with `occurred_at ==
        WALL_AT_ANCHOR`.
        """
        publisher = FakePublisher()
        async with Worker(self._scheduler(publisher)) as scheduler:
            scheduler.submit(a_request(timestamp=0.02))
            await scheduler.drain()

        occurred_at = publisher.events[0].occurred_at
        assert occurred_at == pytest.approx(self.WALL_AT_ANCHOR), (
            "must land at the wall clock reading taken at assembly, not one uptime in the past"
        )

    async def test_two_cameras_on_different_timelines_land_within_seconds(self) -> None:
        """The cross-source case D1 exists for. `FileSource` restarts its pts at 0.0
        per file; `RtspSource` stamps `scene.timestamp` from `time.monotonic()`,
        which on a long-uptime box is tens of thousands of seconds. Against the
        process-wide anchor these two cameras' first events land ~one uptime apart —
        see `test_a_replay_source_starting_near_zero_lands_at_assembly_wall_time` for
        that arithmetic in isolation. Per-camera anchoring fixes the pair: each
        camera anchors independently at its own first-seen `source_timestamp`, so
        both land at (approximately) the wall-clock reading taken when each was
        assembled — seconds apart if the two events were observed seconds apart,
        never hours.
        """
        wall_reads = iter([self.WALL_AT_ANCHOR, self.WALL_AT_ANCHOR + 0.4])
        publisher = FakePublisher()
        scheduler = new_scheduler(
            publisher=publisher,
            clock=lambda: 51_181.0,
            wall_clock=lambda: next(wall_reads),
        )
        file_camera = replace(a_request(timestamp=0.05), camera_id="file-cam")
        rtsp_camera = replace(a_request(timestamp=51_181.3), camera_id="rtsp-cam")

        async with Worker(scheduler) as running:
            running.submit(file_camera)
            await running.drain()
            running.submit(rtsp_camera)
            await running.drain()

        by_camera = {event.camera_id: event.occurred_at for event in publisher.events}
        assert abs(by_camera["file-cam"] - by_camera["rtsp-cam"]) < 5.0, (
            "two cameras observed moments apart must land moments apart, not ~14 "
            "hours apart (one uptime) as the process-wide anchor produced"
        )


class TestTheRecentEventHistory:
    """The console's bounded per-camera ring, written where the `Event` is built.

    `_assemble` is deliberately the one place in the system that constructs an
    `Event` (S14), and the ring hangs off it rather than off `_publish` precisely so
    that the events an operator most needs to see in the console — the §9 degraded
    one, and the one a cancelled shutdown dead-lettered — are in it. Two of the
    tests below are exactly that difference: they fail against a ring written on the
    publish path, because on that path no event was ever published.
    """

    async def test_a_published_event_is_retained_for_the_console(self) -> None:
        scheduler = new_scheduler()
        async with Worker(scheduler) as running:
            running.submit(a_request())
            await running.drain()

        history = scheduler.event_history("cam-1")
        assert len(history.events) == 1
        latest = history.latest
        assert latest is not None
        assert latest.description == "A person is standing near the door."
        assert latest.threat_score == pytest.approx(0.3)
        assert latest.severity.value == "low"
        assert latest.occurred_at == pytest.approx(wall_clock())
        assert latest.description_unavailable is False

    async def test_an_unknown_camera_has_an_empty_history_rather_than_raising(self) -> None:
        """The scheduler knows nothing about which camera ids are configured — that
        is `EngineService`'s job, and it is what turns an unknown id into a 404. Here
        the answer is simply an empty snapshot."""
        history = new_scheduler().event_history("never-heard-of-it")
        assert history.events == ()
        assert history.latest is None

    async def test_a_degraded_event_the_vlm_never_described_is_still_retained(self) -> None:
        """A §9 fallback event reaches the ring, and reaches it still flagged.

        The flag is the point. Drop it in the projection and the console shows a
        metadata stand-in — "periodic_summary: motion (...)" — as though the model
        had looked at the scene and said that.
        """
        vlm = FakeVisionLLM(error=RuntimeError("model not loaded"))
        scheduler = new_scheduler(vlm=vlm)
        async with Worker(scheduler) as running:
            running.submit(a_request())
            await running.drain()

        latest = scheduler.event_history("cam-1").latest
        assert latest is not None
        assert latest.description_unavailable is True
        assert latest.description == "periodic_summary: motion (initial scene summary)", (
            "the §9 stand-in built from cheap signals, not a scene description"
        )

    async def test_an_event_a_cancelled_shutdown_abandoned_is_still_retained(self) -> None:
        """`abandon_pending` dead-letters the escalations a shutdown cut short; they
        are never published. A ring written on the publish path holds nothing here,
        so this test fails against that design and passes against recording at
        assembly."""
        scheduler = new_scheduler()
        scheduler.submit(a_request())
        assert await scheduler.abandon_pending(RuntimeError("shutdown")) == 1

        history = scheduler.event_history("cam-1")
        assert len(history.events) == 1, "an abandoned event is exactly what the console needs"
        assert history.events[0].description_unavailable is True

    async def test_the_shipped_default_bound_is_enforced_end_to_end(self) -> None:
        """The ring the *production* scheduler builds, not a test-sized one.

        A capacity-3 ring test says nothing about whether the shipped default is
        wired in or whether it is enforced at all. This drives more events than the
        default bound through a scheduler constructed exactly as `main.compose`
        constructs it, and fails against an unbounded list.
        """
        overflow = 5
        total = RECENT_EVENTS_PER_CAMERA + overflow
        scheduler = new_scheduler(maxsize=total)
        async with Worker(scheduler) as running:
            for index in range(total):
                assert running.submit(a_request(timestamp=float(index))) is True
            await running.drain()

        history = scheduler.event_history("cam-1")
        assert history.capacity == RECENT_EVENTS_PER_CAMERA
        assert len(history.events) == RECENT_EVENTS_PER_CAMERA, (
            f"{total} events went in and the ring kept {len(history.events)}: the bound "
            "is not enforced, which is the leak this ring exists to avoid"
        )
        assert history.events[0].source_timestamp == pytest.approx(float(overflow)), (
            "the oldest entries are the ones evicted"
        )
        assert history.latest is not None
        assert history.latest.source_timestamp == pytest.approx(float(total - 1))

    async def test_a_busy_camera_cannot_evict_a_quiet_camera_s_history(self) -> None:
        """Through the real scheduler, not just the log: a global ring wired in here
        would drop `cam-quiet`'s single event long before `cam-busy` finished."""
        scheduler = new_scheduler(maxsize=32, recent_events_per_camera=3)
        async with Worker(scheduler) as running:
            running.submit(a_request(camera_id="cam-quiet", timestamp=0.0))
            for index in range(10):
                running.submit(a_request(camera_id="cam-busy", timestamp=float(index + 1)))
            await running.drain()

        assert len(scheduler.event_history("cam-quiet").events) == 1
        assert len(scheduler.event_history("cam-busy").events) == 3
