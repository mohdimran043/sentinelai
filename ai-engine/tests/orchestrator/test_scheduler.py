from __future__ import annotations

import asyncio
import contextlib
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


def a_request(
    clip: ClipHandle | None = None,
    reason: EscalationReason = EscalationReason.PERIODIC_SUMMARY,
    tracks: tuple[Track, ...] = (),
) -> EscalationRequest:
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=12.5,
        detections=(Detection("person", 0.9, BOX),),
        tracks=tracks,
        motion_energy=0.1,
        scene_signature=(1.0,),
    )
    return EscalationRequest(
        camera_id="cam-1",
        event_id=uuid4(),
        reason=reason,
        detail="initial scene summary",
        scene=scene,
        keyframe=FakeSource.make_frame("cam-1", 0, 0.0),
        profile=CameraProfile(camera_id="cam-1"),
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
    assert event.occurred_at == pytest.approx(12.5), "the scene's timestamp, not the clock's"
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
