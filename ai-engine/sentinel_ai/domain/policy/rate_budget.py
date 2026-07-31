"""Per-camera VLM call budget (spec §4.1, 'The budget governor').

Immutable: every operation returns a new bucket, so the gate stays a pure
function of its inputs and is trivially testable at any simulated clock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class TokenBucket:
    capacity: int
    refill_seconds: float
    tokens: float
    updated_at: float

    @classmethod
    def full(cls, capacity: int, refill_seconds: float, now: float) -> TokenBucket:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_seconds <= 0.0:
            raise ValueError(f"refill_seconds must be > 0, got {refill_seconds}")
        return cls(
            capacity=capacity,
            refill_seconds=refill_seconds,
            tokens=float(capacity),
            updated_at=now,
        )

    @property
    def available(self) -> float:
        return self.tokens

    def refilled(self, now: float) -> TokenBucket:
        """Advance the bucket to `now`. A regressing clock is a no-op."""
        elapsed = now - self.updated_at
        if elapsed <= 0.0:
            return self
        gained = elapsed / self.refill_seconds
        return replace(
            self,
            tokens=min(float(self.capacity), self.tokens + gained),
            updated_at=now,
        )

    def try_consume(self, now: float) -> tuple[bool, TokenBucket]:
        """Attempt to spend one token. Returns (allowed, new_bucket)."""
        current = self.refilled(now)
        if current.tokens < 1.0:
            return False, current
        return True, replace(current, tokens=current.tokens - 1.0)
