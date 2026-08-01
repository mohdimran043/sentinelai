"""A bounded, per-camera ring of recently assembled events — for the console only.

**This is not the event store.** RabbitMQ plus the Phase 1C Go consumer is the
durable record of what happened; this is a volatile in-memory cache the operator
console reads so it can draw a threat-over-time chart and show what the camera
saw a few minutes ago without a round trip to a service that does not exist yet.
Everything here is lost on process restart, and an entry is *silently dropped*
the moment the ring wraps. An operator who mistakes a structure with those two
properties for an audit trail will conclude an event never happened when it
merely aged out, and in a custodial setting that is not a cosmetic error — so
the distinction is repeated in the endpoint's own OpenAPI description, where the
person integrating against it will actually read it.

Bound
-----
`RECENT_EVENTS_PER_CAMERA` entries **per camera**, never a single global ring.
A global ring of the same total size lets one busy camera evict every other
camera's history — the quiet corridor that escalated once an hour is exactly the
history an operator wants, and it is the first thing a global ring throws away.
Per camera, a busy camera can only ever evict its own past.

The dict of rings is bounded too, by the number of configured cameras: `record()`
is only ever reached from `VlmScheduler._assemble`, whose `camera_id` comes from
a `CameraRunner` built from the camera file at composition time. Nothing in the
process invents a camera id at runtime, so there is no key-space to leak into.

Why events are recorded at assembly, and what that costs
--------------------------------------------------------
`VlmScheduler._assemble` is deliberately the one place in the system that
constructs an `Event` (S14), and the recording hangs off it rather than off the
publish path so that *every* event is in the ring: a good describe, a §9
degraded one, and one a cancelled shutdown abandoned to the dead-letter sink.
An event that could not be published is precisely the one an operator most needs
to see in the console.

The cost is that `clip_uri` is not carried here. A clip is attached *after*
assembly (`VlmScheduler._attach_clip`), so at record time it is always `None`.
Surfacing an always-null field would be worse than omitting it, and the
alternative — a second write path that back-fills the ring once the clip lands —
would put a second mutator on the state that the one-place-builds-an-Event rule
exists to keep singular. The console links to clips through the durable store.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from uuid import UUID

from sentinel_ai.domain.entities import EscalationReason, Event, Severity

__all__ = [
    "RECENT_EVENTS_PER_CAMERA",
    "CameraEventHistory",
    "RecentEvent",
    "RecentEventLog",
]

RECENT_EVENTS_PER_CAMERA = 200
"""How many recent events each camera keeps in memory.

Chosen against the rate the governors actually permit and the memory the box
actually has, not by feel:

* **Time covered.** `CameraProfile`'s token bucket (capacity 2, refilling every
  10 s) and its 5 s cooldown cap a camera at roughly one escalation per 10 s
  sustained. 200 entries is therefore ~33 minutes of a camera escalating flat
  out, and days of a camera behaving like a real one (a handful of escalations
  an hour). Both are comfortably longer than the window a console user is
  looking at.
* **Memory.** An entry is a slots dataclass holding a description bounded by
  `Settings.vlm_max_new_tokens` (256 tokens, so ~1 KiB of text worst case) plus
  a short action, a reason, a few labels and some floats — call it 2 KiB worst
  case, a few hundred bytes typical. 200 x 2 KiB is ~400 KiB per camera; even
  64 cameras is ~25 MiB against the ~7 GiB free on this box. The bound exists
  because an unbounded list in a process that runs for months is a leak, not
  because 200 is expensive.
* **Why not larger.** A bigger ring buys more of a history that is volatile and
  silently lossy anyway, and makes it likelier someone treats it as the record.
  Anything older than this belongs to the durable store by construction.
* **Why not smaller.** Below ~50 the chart is too sparse to read and a camera in
  a genuinely busy few minutes wraps while the operator is still looking at it.
"""


@dataclass(frozen=True, slots=True)
class RecentEvent:
    """The console's projection of an `Event` — no `clip_uri`, see the module docstring.

    A projection rather than the `Event` itself so that the fields the console
    depends on cannot silently change when the published event grows a field,
    and so nothing holds a reference to a domain object that may be replaced
    downstream.
    """

    event_id: UUID
    camera_id: str
    occurred_at: float
    source_timestamp: float | None
    reason: EscalationReason
    threat_score: float
    severity: Severity
    description: str
    suggested_action: str
    description_unavailable: bool
    labels: tuple[str, ...]
    track_ids: tuple[int, ...]

    @classmethod
    def from_event(cls, event: Event) -> RecentEvent:
        return cls(
            event_id=event.event_id,
            camera_id=event.camera_id,
            occurred_at=event.occurred_at,
            source_timestamp=event.source_timestamp,
            reason=event.reason,
            threat_score=event.threat.value,
            severity=event.threat.severity,
            description=event.description,
            suggested_action=event.suggested_action,
            description_unavailable=event.description_unavailable,
            labels=event.labels,
            track_ids=event.track_ids,
        )


@dataclass(frozen=True, slots=True)
class CameraEventHistory:
    """One camera's whole console view, taken as a single snapshot.

    The live panel's "current description" and the chart's last data point are read
    off the *same* tuple rather than fetched separately, so they can never disagree
    about which event is the most recent one.
    """

    camera_id: str
    capacity: int
    """The per-camera ring bound. Carried on the snapshot, and onto the wire, so a
    console can say how much history it is *not* showing rather than implying it
    has all of it."""

    events: tuple[RecentEvent, ...]
    """Oldest first — see `RecentEventLog.history`."""

    @property
    def latest(self) -> RecentEvent | None:
        """The most recent event, or None when nothing has been assembled for this
        camera in this process.

        None is emphatically not the same as "the VLM had nothing to say": that case
        is a `RecentEvent` carrying `description_unavailable=True`, and collapsing the
        two would tell an operator nothing happened when in fact the model failed.
        """
        return self.events[-1] if self.events else None


class RecentEventLog:
    """Per-camera bounded rings. Not thread-safe; single event loop, no locking needed."""

    def __init__(self, capacity: int = RECENT_EVENTS_PER_CAMERA) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._rings: dict[str, deque[RecentEvent]] = {}

    @property
    def capacity(self) -> int:
        """The per-camera bound. Exposed on the wire so a console can say how much
        history it is *not* showing rather than implying it has all of it."""
        return self._capacity

    def record(self, event: Event) -> None:
        """Append, evicting this camera's oldest entry once the ring is full.

        `deque(maxlen=...)` does the eviction, so the bound is a property of the
        container rather than of a length check someone can later delete.
        """
        ring = self._rings.get(event.camera_id)
        if ring is None:
            ring = deque(maxlen=self._capacity)
            self._rings[event.camera_id] = ring
        ring.append(RecentEvent.from_event(event))

    def history(self, camera_id: str) -> CameraEventHistory:
        """This camera's retained events, **oldest first**, as one snapshot.

        Chronological because the primary consumer plots threat against time; a
        list view reverses it in one line, whereas a chart handed newest-first
        data draws time backwards if the caller forgets to.

        A camera with no events yields an empty snapshot. Whether the id names a
        *configured* camera is not this object's business — `EngineService` owns
        that question and raises `UnknownCameraError` for it, so that an unknown
        camera is a 404 and a quiet one is an empty list.
        """
        ring = self._rings.get(camera_id)
        return CameraEventHistory(
            camera_id=camera_id,
            capacity=self._capacity,
            events=() if ring is None else tuple(ring),
        )
