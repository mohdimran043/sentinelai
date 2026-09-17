"""What every behaviour detector emits, and what it is worth (spec §6, §16).

`domain/behaviour/fall.py` shipped first and defined its own rich evidence record
(`FallEvidence`) carrying descent rate, settle duration and which signal read the body.
That is still the right shape for a detector's own findings — measurements, specific to
what was measured. But three more detectors arrived behind it, and the pipeline needs
one currency it can compare them in: which of two candidates raised on the same frame
is the more urgent, and what escalation reason each becomes.

So each detector keeps its own evidence type and converts to a `BehaviourCandidate`.
The conversion is where detector-specific measurement becomes a comparable claim, and
it is deliberately lossy — the summary is prose a human and a vision-language model can
both read, not a struct a consumer would be tempted to branch on.

Why there is no confidence score here either
---------------------------------------------
[ADR 10](../../../docs/decisions.md) argues that a float on a welfare opinion reads
downstream as a calibrated probability nothing has earned, and [ADR 12] applies the
same reasoning to the fall machine. It applies here unchanged and for every detector: a
state machine either completed its signature or it did not. What varies between kinds is
not how sure we are, it is **how bad it would be to ignore** — and that is `severity`,
which is a policy statement about consequences rather than a claim about evidence.

Pure: standard library only, no clock, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["BehaviourCandidate", "BehaviourKind"]


class BehaviourKind(StrEnum):
    """What a detector says happened.

    One member per *shipped* detector, never ahead of one — the same rule
    `domain/capabilities.py` follows, and for the same reason: a kind nothing can raise
    is a filter option in a console that never matches, and §40 forbids surfacing a
    capability the backend does not have.

    **Loitering is deliberately absent.** It is already implemented, as the
    `dwell_exceeded` escalation trigger (`domain/policy/triggers.py`): a salient track
    that stays within `dwell_radius_px` of its anchor for `dwell_seconds`. Adding a
    second loitering detector here would be the duplication §42 warns against, and the
    two would disagree the first time anyone tuned one of them.
    """

    FALL = "fall"
    ABANDONED_OBJECT = "abandoned_object"
    CAMERA_TAMPER = "camera_tamper"
    ZONE_INTRUSION = "zone_intrusion"
    LINE_CROSSING = "line_crossing"


@dataclass(frozen=True, slots=True)
class BehaviourCandidate:
    """One detector's completed finding, in the form the pipeline compares and escalates.

    Frozen and hashable so a caller can deduplicate a list without a defensive copy.
    """

    kind: BehaviourKind
    summary: str
    """One line of prose. Becomes the escalation detail and reaches the
    vision-language model as `VisionRequest.reason_detail`, so it is written to be read
    by a person and by a model — hedged, specific, and free of jargon neither would
    recognise. See `FallEvidence.summary` for the wording discipline §7 imposes."""

    track_ids: tuple[int, ...]
    """Whose behaviour this is. Empty for a detector that reasons about the whole frame
    rather than about a person — camera tampering being the case that exists: there is
    nobody to attribute a covered lens to, and inventing a track id would be worse than
    saying so."""

    started_at: float
    """When the behaviour began, on the camera's source timeline — not when the
    signature completed, which is later by however long the detector waited to be sure.

    This is what a clip should be centred on. A clip centred on the completion instant
    opens on a body already on the floor, or on a bag that has been sitting there for
    thirty seconds, missing the moment an investigator actually needs.
    """

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("a candidate must say what happened; summary is empty")
