from __future__ import annotations

import asyncio
import contextlib
from types import TracebackType
from uuid import uuid4

import pytest

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
from sentinel_ai.orchestrator.scheduler import EscalationRequest, VlmScheduler
from sentinel_ai.ports.clip_writer import ClipHandle
from sentinel_ai.ports.event_publisher import EventPublisher
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
from tests.fakes.io import FakeClipWriter, FakePublisher, FakeSource
from tests.fakes.models import FakeVisionLLM

BOX = BBox(0.0, 0.0, 10.0, 10.0)


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


def new_scheduler(
    vlm: VisionLanguageModel | None = None,
    publisher: EventPublisher | None = None,
    maxsize: int = 4,
    admission: AdmissionGate | None = None,
    timeout_seconds: float = 5.0,
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
        scheduler = new_scheduler(publisher=publisher, admission=admission)

        async with Worker(scheduler):
            scheduler.submit(a_request())
            await scheduler.drain()

            assert publisher.attempts == 1
            assert publisher.events == [], "the first publish failed"
            assert admission.in_flight == 0, "the slot must be released even on failure"

            # Proof the gate is genuinely reusable, not merely reporting zero.
            scheduler.submit(a_request())
            await scheduler.drain()

        assert publisher.attempts == 2
        assert len(publisher.events) == 1
        assert admission.in_flight == 0
