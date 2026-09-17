"""The VLM work queue, ordered by consequence rather than arrival (spec §16).

The queue this replaces was a plain `asyncio.Queue` with drop-on-full, and both halves
of that were wrong once more than one kind of thing could be escalated.

**Order.** FIFO means a suspected fall waits behind whatever periodic summaries
happened to be queued first. At a measured ~3.3 s per describe and a queue of four,
that is up to thirteen seconds before anybody is asked to look at a person on the
floor. §16 asks for critical work to preempt, and preemption is only meaningful if the
queue is ordered by something other than luck.

**Eviction.** `asyncio.Queue.put_nowait` raises when full, so the *newest* submission
was dropped. On a busy site the newest submission is disproportionately likely to be
the interesting one — a fall is raised at the moment several cameras are also seeing
unusual movement. Dropping the least important item instead costs nothing that matters.

What this deliberately is not
------------------------------
It does not interrupt work already in flight. A describe that has started runs to
completion, because a half-finished generation is not a partial answer — it is nothing,
and cancelling it would waste the GPU time already spent without producing an event.
"Preempt" here means "goes to the front of the queue", not "displaces the running job".

Not thread-safe, and does not need to be: every caller is on the event loop. `submit`
is synchronous and never blocks, which is what keeps the camera pipeline free of any
back-pressure from a stalled VLM — the property `pipeline/runner.py`'s deadlock
analysis depends on.
"""

from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field

__all__ = ["PriorityWorkQueue"]


@dataclass(order=True)
class _Entry[T]:
    """One queued item. Ordered so that `heapq`'s min-heap yields the most urgent first.

    `sort_rank` is the **negated** priority, so a higher priority sorts lower and comes
    out first. `sequence` breaks ties in arrival order, giving FIFO within a priority
    band — and, just as importantly, guarantees two entries never compare equal, so
    `heapq` never falls through to comparing the payloads (which are not orderable and
    would raise).
    """

    sort_rank: int
    sequence: int
    item: T = field(compare=False)


class PriorityWorkQueue[T]:
    """A bounded queue that serves the most urgent item and drops the least urgent.

    Mirrors the slice of `asyncio.Queue` the scheduler actually used — `get`,
    `task_done`, `join` — so the worker loop around it is unchanged. `submit` replaces
    `put_nowait` and returns a bool rather than raising, because a drop here is an
    expected outcome the caller counts rather than an error it handles.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("a queue with no room for an item cannot hold one")
        self._maxsize = maxsize
        self._heap: list[_Entry[T]] = []
        self._sequence = 0
        self._unfinished = 0
        self._not_empty = asyncio.Event()
        # Starts *set*: an empty queue with nothing outstanding is already drained, and
        # a `join()` on a fresh queue must return immediately rather than hang.
        self._all_done = asyncio.Event()
        self._all_done.set()

    def submit(self, item: T, priority: int) -> tuple[bool, T | None]:
        """Queue `item`. Returns `(accepted, evicted)`.

        `accepted` is False only when `item` was itself the least urgent thing present
        and was therefore the one dropped — which keeps the caller's "was my submission
        taken" question answerable without it needing to know the eviction rule.

        `evicted` is whatever had to go to make room, if anything. Returned rather than
        silently discarded so the caller can count and log it: a queue that quietly ate
        work would make a busy site indistinguishable from a quiet one.
        """
        self._sequence += 1
        entry = _Entry(sort_rank=-priority, sequence=self._sequence, item=item)
        heapq.heappush(self._heap, entry)
        self._unfinished += 1
        self._all_done.clear()
        self._not_empty.set()

        if len(self._heap) <= self._maxsize:
            return (True, None)

        # Full. The least urgent item goes — which may be the one just submitted.
        # `max` over the heap rather than a second heap: the queue is bounded at a
        # handful of items, so a linear scan is cheaper than the bookkeeping a
        # double-ended structure would need, and far easier to be sure of.
        worst = max(self._heap)
        self._heap.remove(worst)
        heapq.heapify(self._heap)
        self._unfinished -= 1
        if not self._heap:
            self._not_empty.clear()
        return (worst is not entry, worst.item)

    async def get(self) -> T:
        """The most urgent queued item, waiting if the queue is empty."""
        while not self._heap:
            self._not_empty.clear()
            await self._not_empty.wait()
        entry = heapq.heappop(self._heap)
        if not self._heap:
            self._not_empty.clear()
        return entry.item

    def task_done(self) -> None:
        """Mark one retrieved item as finished, as `asyncio.Queue.task_done` does.

        Raises on an unmatched call for that method's reason: it means the worker's
        accounting has drifted, and a `join()` that returned early because of it would
        let a shutdown drop escalations it believed were published.
        """
        if self._unfinished <= 0:
            raise ValueError("task_done() called more times than there were items")
        self._unfinished -= 1
        if self._unfinished == 0:
            self._all_done.set()

    async def join(self) -> None:
        """Wait until everything submitted has been retrieved and finished."""
        await self._all_done.wait()

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def unfinished(self) -> int:
        """Queued plus in flight. What `join()` is waiting to reach zero."""
        return self._unfinished

    def drain_pending(self) -> tuple[T, ...]:
        """Remove and return everything still queued, most urgent first.

        For shutdown: `VlmScheduler.abandon_pending` has to account for escalations
        that will never be described, and §9 forbids losing an event to a lifecycle
        failure. Each drained item is still `task_done`'s responsibility, so the caller
        accounts for them exactly as the worker would have.
        """
        drained: list[T] = []
        while self._heap:
            drained.append(heapq.heappop(self._heap).item)
        self._not_empty.clear()
        return tuple(drained)
