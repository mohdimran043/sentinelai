"""Process-wide GPU admission gate (spec §5.4) — new in Phase 1B, not in the original layout.

Phase 1A's governors — the token bucket, cooldown, dedup — are all per camera, but the
GPU they protect is global. With one camera the two are equivalent, so this is not a
bug today; with N cameras, N independently-permitting buckets could each legitimately
allow a call and collectively saturate the GPU. Building this seam now, while it is
trivially testable with a single camera, avoids retrofitting a global limiter into the
hot path once a second camera exists.

`now` is an argument, not a clock read, matching the domain convention: the caller owns
the clock so the interval arithmetic is deterministic under test.
"""

from __future__ import annotations

import asyncio
from asyncio import sleep as _sleep

__all__ = ["AdmissionGate"]


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
        """Block until a global slot is free AND min_interval has elapsed.

        The interval is enforced *after* the semaphore, not before: a caller that has
        to queue for a slot has already been spaced out by whoever held it, and paying
        the interval before queueing would compound the two delays.
        """
        await self._semaphore.acquire()
        self._in_flight += 1
        admitted_at = now
        try:
            if self._last_acquired_at is not None:
                deficit = self._min_interval_seconds - (now - self._last_acquired_at)
                if deficit > 0:
                    await _sleep(deficit)
                    # The origin the *next* caller measures its deficit from must be the
                    # instant this call was actually admitted, not the clock read it took
                    # before paying the deficit off. Recording the stale `now` here makes
                    # the following caller's interval look already-elapsed, so every second
                    # admission slips through unspaced and the gate admits at ~2x its
                    # configured rate.
                    admitted_at = now + deficit
        except BaseException:
            # The slot is taken before the deficit is paid, so a cancellation inside that
            # sleep would otherwise leak it forever — and with a non-zero interval the
            # worker spends real time in there, so shutdown lands in this window routinely.
            # A leaked slot on a concurrency-1 gate wedges the GPU permanently.
            self._in_flight -= 1
            self._semaphore.release()
            raise
        self._last_acquired_at = admitted_at

    def release(self, now: float) -> None:
        del now  # no release-side interval policy today; kept for symmetry with acquire
        if self._in_flight == 0:
            # Over-releasing would raise the effective concurrency above the configured
            # limit and silently let two calls onto one GPU.
            raise RuntimeError("AdmissionGate.release() called without a matching acquire()")
        self._in_flight -= 1
        self._semaphore.release()

    @property
    def in_flight(self) -> int:
        return self._in_flight
