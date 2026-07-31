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
