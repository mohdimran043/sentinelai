"""How urgent an escalation is, before anything has looked at it (spec §16).

Two different questions get asked about how much an event matters, and conflating them
is the mistake this module exists to avoid.

**How unusual does the scene look?** That is `Event.threat.severity`, and it comes from
the vision-language model's reading of the keyframe. It is an *observation*, it is only
available after a describe has run, and on a camera without `scene_description` it never
exists at all.

**How bad would it be to get this wrong?** That is `EventPriority`, and it is a pure
function of the escalation reason. A suspected fall is urgent because somebody may be
hurt — before any model has been asked, and whatever the model eventually says. It is
known at the moment the escalation is *raised*, which is what makes it usable for the
one job the threat score cannot do: deciding which of several queued escalations gets
the GPU first (§16's "critical events should preempt lower-priority VLM jobs").

The two combine rather than compete. An alert's severity is the **worse** of the two
(`domain/policy/alerting.py`): a periodic summary the model found alarming is alarming,
and a suspected fall the model was unsure about is still a suspected fall.

Pure: no clock, no I/O, standard library only.
"""

from __future__ import annotations

from enum import StrEnum

from sentinel_ai.domain.entities import EscalationReason, Severity

__all__ = ["EventPriority", "priority_of", "severity_floor"]


class EventPriority(StrEnum):
    """§16's four bands, as an ordered vocabulary.

    A `StrEnum` rather than an int so it serialises and reads as itself, with
    `_RANK` below supplying the ordering — the same arrangement `Confidence.meets`
    uses, and for the same reason: `EventPriority.LOW > EventPriority.CRITICAL` is
    `True` by alphabet, which is the exact inversion of what anyone writing it means.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_RANK: dict[EventPriority, int] = {
    EventPriority.LOW: 0,
    EventPriority.MEDIUM: 1,
    EventPriority.HIGH: 2,
    EventPriority.CRITICAL: 3,
}


def outranks(left: EventPriority, right: EventPriority) -> bool:
    """Whether `left` should be served before `right`. The only ordering this type has."""
    return _RANK[left] > _RANK[right]


def rank(priority: EventPriority) -> int:
    """The numeric rank, for callers that need a sort key rather than a comparison."""
    return _RANK[priority]


_PRIORITIES: dict[EscalationReason, EventPriority] = {
    # Somebody may be hurt, and every second of queueing is a second nobody is coming.
    # Nothing else on this list is time-critical in the same way.
    EscalationReason.FALL_SUSPECTED: EventPriority.CRITICAL,
    # A blinded camera is high, not medium, because it invalidates every *other*
    # answer this camera gives. Its silence is indistinguishable from a quiet site.
    EscalationReason.CAMERA_TAMPER: EventPriority.HIGH,
    # Somebody is somewhere they were told not to be, and is still there.
    EscalationReason.ZONE_INTRUSION: EventPriority.HIGH,
    # §16 lists "unauthorized person" under CRITICAL. It is HIGH here, deliberately.
    # CRITICAL is what preempts the queue, and it is reserved for "somebody may be
    # hurt" — spending it on an access-control finding means a suspected collapse waits
    # behind one. An unauthorised person is urgent; a person on the floor is urgent in
    # a way that does not survive a delay.
    EscalationReason.UNAUTHORIZED_PERSON: EventPriority.HIGH,
    # Left long enough to be deliberate. §16 puts abandonment in HIGH and that is the
    # right band: the reason to look is not urgency but consequence.
    EscalationReason.ABANDONED_OBJECT: EventPriority.HIGH,
    # An instant that has already passed, where an intrusion is a standing fact.
    EscalationReason.LINE_CROSSING: EventPriority.MEDIUM,
    # §16's "loitering" — this is the reason `dwell_exceeded` raises.
    EscalationReason.DWELL_EXCEEDED: EventPriority.MEDIUM,
    # §16's "unusual movement", in the two forms this engine can actually observe.
    EscalationReason.SPEED_ANOMALY: EventPriority.MEDIUM,
    EscalationReason.TRACK_COUNT_SPIKE: EventPriority.MEDIUM,
    # §16's "general scene change".
    EscalationReason.SCENE_CHANGE: EventPriority.LOW,
    EscalationReason.NEW_SALIENT_TRACK: EventPriority.LOW,
    # Not an anomaly at all — the forced look that happens whether or not anything is
    # wrong. It must never displace something that is.
    EscalationReason.PERIODIC_SUMMARY: EventPriority.LOW,
    # A person asked for this and is waiting for the answer, which is worth more than
    # a routine summary and less than a fall. Deliberately not CRITICAL: an operator
    # clicking "describe now" on a quiet camera must not be able to push a suspected
    # collapse down the queue.
    EscalationReason.USER_REQUESTED: EventPriority.MEDIUM,
}
"""`EscalationReason` -> how urgently it should be served.

A dict rather than a method on the enum so `test_priority.py` can compare its keys
against `EscalationReason` directly and fail the build on a reason that forgot to
declare one. A missing entry would otherwise default to something plausible, and the
first anyone would know is a fall queued behind four periodic summaries.
"""


def priority_of(reason: EscalationReason) -> EventPriority:
    """How urgently this escalation should be served. Total by construction — a reason
    with no entry raises `KeyError` here rather than defaulting."""
    return _PRIORITIES[reason]


_SEVERITY_FLOORS: dict[EventPriority, Severity] = {
    EventPriority.CRITICAL: Severity.CRITICAL,
    EventPriority.HIGH: Severity.HIGH,
    EventPriority.MEDIUM: Severity.MEDIUM,
    EventPriority.LOW: Severity.INFO,
}
"""The lowest severity an alert of each priority may be shown at.

`LOW` floors at `INFO` rather than at `Severity.LOW`, deliberately: a periodic summary
of an empty corridor should sit at the bottom of the list, and giving every routine
look a non-`INFO` severity is how an alert list stops being scannable.
"""


def severity_floor(reason: EscalationReason) -> Severity:
    """The severity an alert for this reason cannot go below, whatever a model thought.

    A suspected fall the vision-language model read as unremarkable is still a
    suspected fall — the state machine watched a person go down and stay down, and a
    single frame's opinion does not overturn several seconds of geometry. Without a
    floor, a model that mis-described the keyframe could quietly demote the one event
    on the list that most needed attention.
    """
    return _SEVERITY_FLOORS[priority_of(reason)]
