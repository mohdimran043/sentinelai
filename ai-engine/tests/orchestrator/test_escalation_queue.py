"""The VLM work queue: order, eviction, and the accounting `join()` depends on."""

from __future__ import annotations

import asyncio

import pytest

from sentinel_ai.orchestrator.escalation_queue import PriorityWorkQueue


class TestOrder:
    async def test_the_most_urgent_item_is_served_first(self) -> None:
        """§16. A suspected fall queued behind four periodic summaries is thirteen
        seconds at the measured describe latency."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=10)
        queue.submit("summary", 0)
        queue.submit("fall", 3)
        queue.submit("crossing", 1)

        assert await queue.get() == "fall"
        assert await queue.get() == "crossing"
        assert await queue.get() == "summary"

    async def test_equal_priorities_keep_arrival_order(self) -> None:
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=10)
        for name in ("first", "second", "third"):
            queue.submit(name, 1)
        assert [await queue.get() for _ in range(3)] == ["first", "second", "third"]

    async def test_get_waits_rather_than_spinning_on_an_empty_queue(self) -> None:
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=4)
        task = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        assert not task.done()
        queue.submit("late", 1)
        assert await task == "late"


class TestEviction:
    def test_the_least_urgent_item_is_dropped_not_the_newest(self) -> None:
        """On a busy site the newest submission is disproportionately likely to be the
        interesting one — a fall is raised at the moment several cameras are also
        seeing unusual movement."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=2)
        queue.submit("summary", 0)
        queue.submit("crossing", 1)

        accepted, evicted = queue.submit("fall", 3)

        assert accepted is True
        assert evicted == "summary"
        assert len(queue) == 2

    def test_a_submission_that_is_itself_the_least_urgent_is_refused(self) -> None:
        """The caller's "was mine taken" question stays answerable without it needing
        to know the eviction rule."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=1)
        queue.submit("fall", 3)

        accepted, evicted = queue.submit("summary", 0)

        assert accepted is False
        assert evicted == "summary"

    def test_the_evicted_item_is_returned_rather_than_silently_eaten(self) -> None:
        """A queue that quietly discarded work would make a busy site
        indistinguishable from a quiet one."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=1)
        queue.submit("first", 1)
        _, evicted = queue.submit("second", 2)
        assert evicted == "first"

    def test_eviction_keeps_the_queue_at_its_bound(self) -> None:
        queue: PriorityWorkQueue[int] = PriorityWorkQueue(maxsize=3)
        for index in range(50):
            queue.submit(index, index % 4)
        assert len(queue) == 3

    def test_a_zero_capacity_queue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot hold one"):
            PriorityWorkQueue(maxsize=0)


class TestAccounting:
    async def test_join_returns_once_everything_is_finished(self) -> None:
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=4)
        queue.submit("one", 1)
        queue.submit("two", 1)

        waiter = asyncio.create_task(queue.join())
        await asyncio.sleep(0)
        assert not waiter.done()

        for _ in range(2):
            await queue.get()
            queue.task_done()
        await asyncio.wait_for(waiter, timeout=1.0)

    async def test_join_on_a_fresh_queue_returns_immediately(self) -> None:
        """An empty queue with nothing outstanding is already drained; a `join()` that
        hung here would wedge every shutdown."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=4)
        await asyncio.wait_for(queue.join(), timeout=1.0)

    def test_an_evicted_item_does_not_stay_outstanding(self) -> None:
        """Otherwise `join()` waits forever for work nobody will ever finish, and the
        shutdown that calls it never completes."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=1)
        queue.submit("first", 1)
        queue.submit("second", 2)
        assert queue.unfinished == 1

    def test_unmatched_task_done_raises(self) -> None:
        """It means the worker's accounting has drifted, and a `join()` that returned
        early because of it would let a shutdown drop escalations it believed were
        published."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=4)
        with pytest.raises(ValueError, match="more times"):
            queue.task_done()

    def test_drain_pending_empties_the_queue_most_urgent_first(self) -> None:
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=8)
        queue.submit("summary", 0)
        queue.submit("fall", 3)
        assert queue.drain_pending() == ("fall", "summary")
        assert len(queue) == 0

    def test_drained_items_are_still_the_callers_to_account_for(self) -> None:
        """`abandon_pending` has to record them: §9 forbids losing an event to a
        lifecycle failure."""
        queue: PriorityWorkQueue[str] = PriorityWorkQueue(maxsize=8)
        queue.submit("one", 1)
        queue.drain_pending()
        assert queue.unfinished == 1
        queue.task_done()
        assert queue.unfinished == 0
