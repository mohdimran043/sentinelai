"""Where the pure behaviour detectors keep their state (spec §6).

Every detector in `domain/behaviour/` is a function of the shape
`(observation, policy, state) -> (state, candidates)` with no clock and no I/O. That is
what makes "the person stayed down for 7.2 seconds" a test that runs in microseconds,
and it is not negotiable — see `domain/behaviour/__init__.py`.

The consequence is that *somebody* has to thread the state frame to frame, and that
somebody must live out here, in the impure layer, next to the loop that has the frames.
This class is that somebody, and it is deliberately the only mutable thing in the
behaviour subsystem.

Why explicit fields rather than a registry
-------------------------------------------
The tempting shape is a list of detectors behind a common interface, iterated
generically. It was not taken. Each detector's state and policy are different types —
`FallTracker` and `FallPolicy`, `ZoneTracker` and `ZonePolicy` — and a generic registry
either erases those to `Any`, which throws away the static checking that is most of the
point of a typed pure core, or needs a generic protocol whose variance would take
longer to read than the four explicit blocks below.

Four detectors is not enough to earn that. A tenth would be, and the `observe` method
is where it becomes obvious.

Each detector is independently optional: `None` means this camera did not ask for it,
and a policy that is present is a policy that runs. `enabled` says whether anything
would run at all, so the frame loop can skip the whole subsystem — including the pose
pass — on a camera that enabled none of it.
"""

from __future__ import annotations

from sentinel_ai.domain.behaviour.abandonment import (
    AbandonmentPolicy,
    AbandonmentTracker,
    observe_abandonment,
)
from sentinel_ai.domain.behaviour.candidate import BehaviourCandidate, BehaviourKind
from sentinel_ai.domain.behaviour.fall import FallPolicy, FallTracker, observe_falls
from sentinel_ai.domain.behaviour.observation import BehaviourObservation
from sentinel_ai.domain.behaviour.tamper import TamperPolicy, TamperState, observe_tamper
from sentinel_ai.domain.behaviour.zones import ZonePolicy, ZoneTracker, observe_zones

__all__ = ["BehaviourEngine"]

_PRIORITY: dict[BehaviourKind, int] = {
    BehaviourKind.FALL: 100,
    BehaviourKind.CAMERA_TAMPER: 90,
    BehaviourKind.ZONE_INTRUSION: 80,
    BehaviourKind.LINE_CROSSING: 70,
    BehaviourKind.ABANDONED_OBJECT: 60,
}
"""Which candidate wins when several complete on the same frame.

Only one escalation is raised per frame — the keyframe is shared, so a second would
describe the same image — so when two machines finish at once, one of them is not
reported. This table decides which, and the order is an argument about consequences
rather than about confidence:

* **A fall outranks everything.** Somebody may be hurt, and nothing else on this list
  is time-critical in the same way.
* **Tampering outranks the rest** because it is the only one that says the camera's
  *other* answers cannot be trusted. An intrusion alert from a camera that has just
  gone blind is an alert about a scene nobody can see.
* **Intrusion over crossing** — being inside a restricted area is a standing fact,
  where a crossing is an instant that has already passed.
* **An abandoned object is last.** It has by construction been sitting there for at
  least `unattended_seconds`, so it is the one finding that loses nothing by being
  reported on the next frame instead of this one.

A kind missing from this table raises `KeyError` in `observe` rather than sorting
arbitrarily, which is what makes adding a detector without ranking it a build failure
rather than a silent coin toss.
"""


class BehaviourEngine:
    """The mutable state of every behaviour detector enabled on one camera."""

    def __init__(
        self,
        *,
        fall: FallPolicy | None = None,
        abandonment: AbandonmentPolicy | None = None,
        tamper: TamperPolicy | None = None,
        zones: ZonePolicy | None = None,
    ) -> None:
        self._fall_policy = fall
        self._fall_state = FallTracker()

        self._abandonment_policy = abandonment
        self._abandonment_state = AbandonmentTracker()

        self._tamper_policy = tamper
        self._tamper_state = TamperState()

        # A zone policy with no geometry drawn could never fire, so it is dropped here
        # rather than carried: `enabled` would otherwise report a camera as running
        # zone monitoring when nothing had been drawn on it, and the frame loop would
        # pay for a pose pass to feed a detector with no boundaries to test against.
        self._zone_policy = zones if zones is not None and not zones.is_empty else None
        self._zone_state = ZoneTracker()

    @property
    def enabled(self) -> bool:
        """Whether anything here would run. False lets the frame loop skip the whole
        subsystem, including the pose forward pass."""
        return any(
            policy is not None
            for policy in (
                self._fall_policy,
                self._abandonment_policy,
                self._tamper_policy,
                self._zone_policy,
            )
        )

    @property
    def needs_pose(self) -> bool:
        """Whether a pose pass would tell any enabled detector something.

        Only the fall machine reads keypoints. Running the pose model for a camera that
        enabled only abandonment or tamper detection would be a second forward pass per
        frame whose output nothing looks at — the cost §13 exists to avoid, reappearing
        one layer down.
        """
        return self._fall_policy is not None

    def observe(self, observation: BehaviourObservation) -> tuple[BehaviourCandidate, ...]:
        """Advance every enabled detector by one frame, highest priority first.

        Returns all candidates rather than just the winner: the caller escalates one,
        but telemetry and the alert engine want to know what else completed, and
        throwing it away here would make that unrecoverable.
        """
        candidates: list[BehaviourCandidate] = []

        if self._fall_policy is not None:
            self._fall_state, evidence = observe_falls(
                observation, self._fall_policy, self._fall_state
            )
            candidates.extend(item.to_candidate() for item in evidence)

        if self._abandonment_policy is not None:
            self._abandonment_state, found = observe_abandonment(
                observation, self._abandonment_policy, self._abandonment_state
            )
            candidates.extend(found)

        if self._tamper_policy is not None:
            self._tamper_state, found = observe_tamper(
                observation, self._tamper_policy, self._tamper_state
            )
            candidates.extend(found)

        if self._zone_policy is not None:
            self._zone_state, found = observe_zones(
                observation, self._zone_policy, self._zone_state
            )
            candidates.extend(found)

        # `_PRIORITY[...]` rather than `.get(..., 0)`: an unranked kind should fail the
        # build, not sort arbitrarily among the ranked ones.
        candidates.sort(key=lambda candidate: -_PRIORITY[candidate.kind])
        return tuple(candidates)

    def reset(self) -> None:
        """Drop every detector's state — called on a stream discontinuity.

        For the tracker's reason one layer up: every phase these machines hold is keyed
        by a track id the reconnect has invalidated, and every timestamp in them is on a
        timeline that no longer exists. Carried across the break, a track that was
        upright before it would measure its first post-break frame against a centroid
        from the old stream — arbitrary pixels over an arbitrary interval, which is
        exactly how a reconnect becomes a reported fall that never happened.

        The policies are configuration and survive; only the state is dropped.
        """
        self._fall_state = FallTracker()
        self._abandonment_state = AbandonmentTracker()
        self._tamper_state = TamperState()
        self._zone_state = ZoneTracker()
