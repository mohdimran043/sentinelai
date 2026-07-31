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
