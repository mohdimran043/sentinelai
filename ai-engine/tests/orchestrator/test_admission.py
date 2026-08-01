from __future__ import annotations

import asyncio
from itertools import pairwise

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


class VirtualClock:
    """A clock that only advances when the gate sleeps.

    The existing interval tests replace `asyncio.sleep` with a recorder and then keep
    handing the gate a `now` of their own choosing — so the time the gate *believes* it
    spent waiting never feeds back into the next `now`. That is exactly the blind spot
    the under-spacing bug hides in, so this clock closes it: every second the gate sleeps
    off is a second the caller's next clock read observes.
    """

    def __init__(self) -> None:
        self.now = 0.0

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


async def test_successive_admissions_are_spaced_by_the_full_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is the only thing bounding *total* GPU load across cameras — Phase 1A's
    token bucket and cooldown are both per camera — so a gate that admits at twice its
    configured rate has no backstop behind it.

    Fails against a gate that records the pre-sleep clock read as the admission instant:
    that origin is stale by exactly the deficit it just slept off, so the next caller
    measures its own deficit as already elapsed and is admitted immediately. Four
    sequential acquires at a 2.0s floor then land at [0.0, 2.0, 2.0, 4.0] — gaps of
    [2.0, 0.0, 2.0], every second admission unspaced.
    """
    clock = VirtualClock()
    monkeypatch.setattr(admission_module, "_sleep", clock.sleep)
    gate = AdmissionGate(concurrency=1, min_interval_seconds=2.0)

    admitted: list[float] = []
    for _ in range(4):
        await gate.acquire(now=clock.now)
        admitted.append(clock.now)
        gate.release(now=clock.now)

    gaps = [later - earlier for earlier, later in pairwise(admitted)]
    assert gaps == pytest.approx([2.0, 2.0, 2.0]), (
        f"every admission must be spaced by the full interval; got {admitted}"
    )


async def test_the_first_acquire_of_a_gates_life_never_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is nothing to be spaced away from yet, so a cold start must be immediate —
    otherwise the very first escalation of a process pays the interval for nothing."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(admission_module, "_sleep", fake_sleep)
    gate = AdmissionGate(concurrency=1, min_interval_seconds=30.0)

    await gate.acquire(now=0.0)

    assert slept == []
