"""A bounded, per-camera ring of recently assembled events, and the live feed over
it — for the console only.

Two readers, one structure: `GET /cameras/{camera_id}/events` takes a snapshot of one
camera's ring, and `GET /events/stream` takes a snapshot of every ring and then
follows it with each new write as it happens (`EventSubscription`). Deliberately the
same structure for both, so the two cannot disagree about what the engine has seen,
and so the stream inherits every caveat below rather than looking like a more
authoritative source than the endpoint it is derived from.

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

Why events are recorded at assembly, and how the clip catches up
----------------------------------------------------------------
`VlmScheduler._assemble` is deliberately the one place in the system that
constructs an `Event` (S14), and the recording hangs off it rather than off the
publish path so that *every* event is in the ring: a good describe, a §9
degraded one, and one a cancelled shutdown abandoned to the dead-letter sink.
An event that could not be published is precisely the one an operator most needs
to see in the console.

A clip, though, is attached *after* assembly (`VlmScheduler._attach_clip`), so at
record time `clip_uri` is always `None`. This module used to stop there and omit
the field, which left the camera page unable to link a notification to its
footage at all. It is now back-filled by `attach_clip()` once the clip lands.

The ordering is unchanged, and that is the point: recording still happens before
the clip is attempted, so spec §9's rule — an anomaly event is never lost to an
infrastructure failure — still holds exactly as it did. A clip that fails to
finish leaves the entry in place with `clip_uri=None`, which is the truth. Only
the success path writes a second time, and it rewrites one field of one entry
that is already there rather than appending anything, so the ring's bound, its
order and its one-Event-builder rule all survive. What a back-fill does change is
that an entry can now have two versions, which is why every entry carries a
`sequence` (below): a live subscriber that already holds the pre-clip copy has to
be able to tell an update from a duplicate.

Sequence numbers
----------------
Every write — a record, and every back-fill — takes the next value of one
process-wide counter. Two things depend on it:

* `subscribe()` hands a new stream client a backlog snapshot plus a watermark,
  and the client's queue drops anything at or below that watermark. That is what
  makes the backlog-to-live handover exact: no gap (the subscriber is registered
  *before* the snapshot is taken, so nothing recorded in between is missed) and
  no duplicate (anything already in the snapshot is filtered out of the queue).
* An update to an event the client has already seen arrives with a *higher*
  sequence than the copy it holds, so it is delivered rather than suppressed —
  a duplicate and a legitimate second version of the same `event_id` are
  distinguishable, which they would not be if de-duplication keyed on the id.

The counter is per process and means nothing across a restart, exactly like the
ring it orders.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, replace
from uuid import UUID

from sentinel_ai.domain.entities import EscalationReason, Event, Severity
from sentinel_ai.domain.welfare import WelfareConcern

__all__ = [
    "EVENT_STREAM_QUEUE_MAXSIZE",
    "RECENT_EVENTS_PER_CAMERA",
    "CameraEventHistory",
    "EventSubscription",
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

EVENT_STREAM_QUEUE_MAXSIZE = 256
"""How many events one live subscriber may fall behind by before it is cut off.

A bound rather than an unbounded list because the alternative is a browser tab
that stopped reading — a suspended laptop, a paused debugger, a socket whose
peer vanished without a FIN — growing the engine's heap for as long as it stays
open. The engine cannot tell that client apart from a slow one, so it must
survive both.

* **Why cutting off rather than dropping.** A subscriber past this bound is told
  so and its stream ends; it reconnects and opens with a fresh backlog, which is
  a *better* answer than the queue it lost — the backlog is the current window,
  while the queue holds what was current several minutes ago. Silently dropping
  the oldest queued entries instead would leave the client permanently, invisibly
  missing events with no way to notice.
* **Why 256.** Comfortably more than a client hiccup: at the roughly one
  escalation per 10 s per camera the governors permit, it is ~40 minutes of one
  camera or ~10 minutes of four, all without the client reading a byte. Anything
  that far behind is not hiccuping, it is gone.
* **Cost.** Entries are references to the same immutable `RecentEvent` objects
  the rings already hold, so a full queue is ~2 KiB of pointers plus, at worst,
  the few entries that have since been evicted from a ring.
"""


@dataclass(frozen=True, slots=True)
class RecentEvent:
    """The console's projection of an `Event`.

    A projection rather than the `Event` itself so that the fields the console
    depends on cannot silently change when the published event grows a field,
    and so nothing holds a reference to a domain object that may be replaced
    downstream.
    """

    event_id: UUID
    sequence: int
    """This version's position in the log's write order — see the module docstring.

    Distinct from `event_id`, which names the *event*: back-filling a clip URI
    produces a second `RecentEvent` with the same `event_id` and a higher
    `sequence`. Per process and monotonic; meaningless across a restart, so it is
    a de-duplication key and never a sort key for display (`occurred_at` is that).
    """

    clip_uri: str | None
    """Where the clip for this event was written, or None.

    None means one of three things and the console must not read it as any one of
    them alone: no clip was being recorded, the clip has not finished yet (it is
    attached shortly after the event is assembled), or the clip failed to write.
    """

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

    welfare_concerns: tuple[WelfareConcern, ...] = ()
    """Every welfare concern the event carried, unfiltered.

    **Not filtered by the camera's `notify_on`.** That setting decides which
    concerns are *pushed* to somebody who is not watching the console; it has no
    business deciding what an operator looking straight at the camera page may
    see. A camera muted for `medication` still detects, still records and still
    publishes — and the person reading that event needs the whole assessment,
    not the subset somebody chose to be paged about. This projection has no
    access to the camera's policy, which is what makes the filtering impossible
    to add here by accident. Contrast `WelfareNote.concerns`, which is
    deliberately narrowed, because that one leaves the building.

    Empty when the model reported nothing. The assessment's `basis` is not
    projected: it has exactly one value today, and a second one (spec: a
    multi-frame or pose-based source) is a deliberate change that rewrites the
    routing rule and the console's caveat together — see ADR 10.
    """

    @classmethod
    def from_event(cls, event: Event, sequence: int) -> RecentEvent:
        return cls(
            event_id=event.event_id,
            sequence=sequence,
            # Always None in practice at this point — the clip is attached after
            # assembly — but read from the event rather than hard-coded, so a future
            # caller that already has the URI is not silently overruled.
            clip_uri=event.clip_uri,
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
            welfare_concerns=event.welfare.concerns,
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


class EventSubscription:
    """One live client's view of the log: the backlog, then everything after it.

    Two steps, deliberately separable in time
    -----------------------------------------
    `RecentEventLog.subscribe()` registers this object, and from that instant every
    write reaches its queue. `open()` then takes the backlog snapshot and the
    watermark that goes with it. Production really does put an `await` between the
    two — the HTTP handler subscribes, and the response body generator runs later —
    and an event written in that window is genuinely written twice: once into the
    queue, once into the snapshot that comes after it.

    That is what the watermark is for, and why the order is register-then-snapshot
    and not the other way round:

    * **no gap** — registration precedes the snapshot, so nothing written after the
      client began subscribing can be missed by both;
    * **no duplicate** — `next_event()` discards anything whose `sequence` is at or
      below the watermark, because the snapshot already carried it.

    The reverse order would be a gap, and a gap is silent: the client would never
    learn the event existed. This ordering's failure mode is a duplicate, which is
    detectable and is then detected.

    A duplicate is not the same thing as a second version. An entry re-written by
    `attach_clip()` takes a *new* sequence above the watermark, so a client that
    already holds the pre-clip copy is sent the updated one and can replace it,
    keyed on `event_id`. De-duplicating on `event_id` instead would suppress exactly
    that update and leave every live client believing the event has no clip.

    Not thread-safe; one event loop, no locking. Everything except `next_event()` is
    synchronous and never yields, which is also why `offer()` cannot be wedged by a
    client that has stopped reading.
    """

    def __init__(self, log: RecentEventLog, *, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._log = log
        self._maxsize = maxsize
        self._pending: deque[RecentEvent] = deque()
        self._wakeup = asyncio.Event()
        self._watermark: int | None = None
        self._overflowed = False
        self._closed = False
        self._suppressed = 0

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def overflowed(self) -> bool:
        """True once this client fell further behind than `maxsize`.

        Its stream is over; it is expected to reconnect and open a fresh backlog,
        which is more current than the queue it lost."""
        return self._overflowed

    @property
    def suppressed_duplicates(self) -> int:
        """How many queued entries the watermark discarded because the backlog
        already carried them. Non-zero is normal and healthy — it is the handover
        working — and it is the counter that tells a test the filter is live."""
        return self._suppressed

    @property
    def pending(self) -> int:
        return len(self._pending)

    def open(self) -> tuple[RecentEvent, ...]:
        """The backlog this client starts from; fixes the watermark. Call once.

        Every camera's retained events in one list, ordered by `occurred_at` — the
        field the wire contract tells consumers to sort and plot on — with `sequence`
        breaking ties so the order is total and stable.
        """
        if self._watermark is not None:
            raise RuntimeError("this subscription has already been opened")
        backlog, self._watermark = self._log.snapshot()
        return backlog

    def offer(self, event: RecentEvent) -> None:
        """Queue one write for this client. Never awaits, never raises.

        Called by the log inside `record()`/`attach_clip()`, which are called from the
        scheduler's worker: a client that has stopped reading must not be able to slow
        an event's assembly down, let alone stop it.
        """
        if self._closed or self._overflowed:
            return
        if len(self._pending) >= self._maxsize:
            self._overflowed = True
            # Nothing queued will ever be sent now, so let go of it immediately rather
            # than holding a full queue's worth of references until the client's
            # response task is finally torn down.
            self._pending.clear()
            self._wakeup.set()
            return
        self._pending.append(event)
        self._wakeup.set()

    async def next_event(self, timeout: float | None) -> RecentEvent | None:
        """The next entry this client has not already been given.

        Returns None when the stream is over — `close()` (engine shutdown) or an
        overflow — which the two properties above distinguish. Raises `TimeoutError`
        when `timeout` elapses with nothing to send, which is the caller's cue to emit
        a keepalive; `None` as the timeout waits indefinitely.
        """
        if self._watermark is None:
            raise RuntimeError("open() this subscription before reading from it")
        watermark = self._watermark
        while True:
            # Cleared before the queue is drained, never after: a write landing
            # between the drain and the wait then leaves the flag set and the wait
            # returns at once, instead of the wakeup being cleared away underneath it.
            self._wakeup.clear()
            while self._pending:
                event = self._pending.popleft()
                if event.sequence > watermark:
                    return event
                self._suppressed += 1
            if self._closed or self._overflowed:
                return None
            await asyncio.wait_for(self._wakeup.wait(), timeout)

    def close(self) -> None:
        """End this client's stream and stop feeding it. Idempotent.

        Called from the response generator's `finally` (the client went away, or the
        server tore the response down) and from `RecentEventLog.close_all()` at engine
        shutdown, so a reader parked in `next_event()` is released rather than left
        waiting for an event that will never come.
        """
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._log.release(self)
        self._wakeup.set()


class RecentEventLog:
    """Per-camera bounded rings, plus the live fan-out over them.

    Not thread-safe; single event loop, no locking needed."""

    def __init__(self, capacity: int = RECENT_EVENTS_PER_CAMERA) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._rings: dict[str, deque[RecentEvent]] = {}
        self._sequence = 0
        self._subscribers: set[EventSubscription] = set()

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
        self._sequence += 1
        entry = RecentEvent.from_event(event, self._sequence)
        ring.append(entry)
        self._broadcast(entry)

    def attach_clip(self, camera_id: str, event_id: UUID, clip_uri: str) -> bool:
        """Back-fill a clip URI onto an entry already in the ring (T3).

        Returns False when the entry is not there, which is an ordinary outcome and
        not an error: the ring is bounded, so a camera busy enough to wrap between an
        event's assembly and its clip's upload has legitimately lost that entry. The
        caller must not resurrect it — an event re-appearing out of order, minutes
        after it aged out, is worse than a console link that is missing.

        The entry is replaced in place, so the ring's length, bound and chronological
        order are untouched; only `clip_uri` and `sequence` differ. Searched newest
        first because a clip belongs to an event assembled moments earlier, so the
        match is at or near the end in every realistic case.
        """
        ring = self._rings.get(camera_id)
        if ring is None:
            return False
        for offset in range(len(ring) - 1, -1, -1):
            entry = ring[offset]
            if entry.event_id != event_id:
                continue
            self._sequence += 1
            updated = replace(entry, clip_uri=clip_uri, sequence=self._sequence)
            ring[offset] = updated
            # Broadcast too: a client already streaming holds the pre-clip copy, and
            # without this it would keep it forever while a client that connected one
            # second later would see the clip. The higher sequence is what tells it
            # this is a newer version of an event it has, not a second event.
            self._broadcast(updated)
            return True
        return False

    # -- the live fan-out ---------------------------------------------------------

    def subscribe(self, *, maxsize: int = EVENT_STREAM_QUEUE_MAXSIZE) -> EventSubscription:
        """Register a live client. It receives every write from *now*, not from
        `open()` — see `EventSubscription` for why that order is the safe one."""
        subscription = EventSubscription(self, maxsize=maxsize)
        self._subscribers.add(subscription)
        return subscription

    def release(self, subscription: EventSubscription) -> None:
        """Stop feeding a subscription. Called by `EventSubscription.close()`, which
        is the method callers should use; unregistering without closing would leave a
        reader parked forever."""
        self._subscribers.discard(subscription)

    def close_all(self) -> int:
        """End every live stream and return how many there were.

        Shutdown calls this so a subscriber parked in `next_event()` is released
        rather than waiting on an engine that has stopped producing. Iterates a copy:
        `close()` unregisters, which mutates the set being walked.
        """
        subscriptions = tuple(self._subscribers)
        for subscription in subscriptions:
            subscription.close()
        return len(subscriptions)

    @property
    def subscribers(self) -> int:
        """How many live streams are attached. Bounded by open HTTP connections, and
        exposed so a test can prove a finished stream unregistered itself rather than
        accumulating a queue per request for the life of the process."""
        return len(self._subscribers)

    def snapshot(self) -> tuple[tuple[RecentEvent, ...], int]:
        """Every camera's retained events plus the current sequence watermark.

        One list across all cameras, since a stream client watches the site rather
        than one camera, ordered by `occurred_at` (the wire contract's sort key) with
        `sequence` as a tie-break so the order is total. The watermark is read in the
        same call, and nothing between them can yield, so it is exactly "the highest
        sequence this snapshot can contain".
        """
        events = sorted(
            (entry for ring in self._rings.values() for entry in ring),
            key=lambda entry: (entry.occurred_at, entry.sequence),
        )
        return tuple(events), self._sequence

    def _broadcast(self, entry: RecentEvent) -> None:
        """Offer one write to every live client. Synchronous and total: `offer()`
        neither awaits nor raises, so no subscriber can delay or break the assembly
        path it is called from."""
        for subscription in self._subscribers:
            subscription.offer(entry)

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
