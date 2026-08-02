"""Welfare concern types — a vision-language model's opinion, not a detection.

SentinelAI protects the watched person; it does not accuse them. That framing
forces a specific shape on this module, and the shape is deliberately not the
one a perimeter-security system would reach for.

`adapters/vision/qwen25vl.py`'s module docstring lays out why the VLM's harm
assessment already avoids `fall_detected: true`-style fields: a boolean reads
to any downstream consumer as a trained classifier's verdict, and there is no
fall detector anywhere in this system — only a language model's opinion about
a single still frame (spec §4, deferred to Phase 4; no pose estimation, no
action recognition, no per-limb tracking exists here or is planned for this
phase). Everything below is that same argument, applied to the richer,
multi-concern welfare signal a later task adds on top of the same VLM call:

* **No booleans.** `WelfareConcern` is `{kind, confidence, evidence}` — an
  opinion with its uncertainty and its supporting observation attached, never
  a yes/no. A boolean forces the model's judgment through a threshold before
  it reaches storage; keeping `confidence` as data lets every downstream
  consumer (a notification, a dashboard, an audit log) decide its own bar
  instead of inheriting one baked in here.
* **No `Confidence.CERTAIN`.** `Confidence` has exactly `POSSIBLE` and
  `LIKELY`. A single frame, sampled at most once every few seconds by the
  escalation gate (spec §4.1), cannot rule out an innocent explanation for
  what it shows — someone lying down is not necessarily someone who has
  collapsed, and two people standing close is not necessarily an altercation.
  Offering a `CERTAIN` tier would let a caller (or a future maintainer padding
  out an enum "for completeness") claim a certainty the evidence can never
  earn. Leaving it out is the honest option, not a missing one.
* **`basis` is a mandatory, single-valued marker.** Every `WelfareAssessment`
  carries `basis: Literal["single_frame_vlm"]`, always that value, so a
  consumer reading the payload alone — with no side channel, no tribal
  knowledge of which pipeline produced it — can see where the opinion came
  from and weigh it accordingly. If a future phase adds a second source
  (multi-frame reasoning, a purpose-built pose model), it gets its own basis
  value rather than silently widening what this one means.

This module is a welfare *concern*, not a threat: it exists to get a human to
look at someone who may need help, not to accuse anyone of wrongdoing. That is
why `evidence` is mandatory (a concern must point at what the model actually
saw, in prose a human can weigh) and why duplicate kinds collapse rather than
stack — a model that lists `collapse` twice across two prompt phrasings is
reporting one concern about one person, and counting it twice would only
distort whatever routing rule later reads `len(concerns)` or iterates by kind.

Pure, like the rest of `domain/`: no I/O, no clock reads, standard library
only (`test_architecture.py` enforces this — see its module docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class ConcernKind(StrEnum):
    """What the VLM's welfare-focused prompt asks the frame to be checked for."""

    COLLAPSE = "collapse"
    ALTERCATION = "altercation"
    SELF_HARM = "self_harm"
    MEDICATION = "medication"
    DISTRESS = "distress"
    OTHER = "other"


class Confidence(StrEnum):
    """How sure a single still frame can honestly make the model.

    Deliberately two members, not three: see the module docstring for why
    there is no `CERTAIN`.
    """

    POSSIBLE = "possible"
    LIKELY = "likely"


_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.POSSIBLE: 0,
    Confidence.LIKELY: 1,
}


@dataclass(frozen=True, slots=True)
class WelfareConcern:
    """One opinion about one kind of concern, with its supporting observation.

    Not a detection: `confidence` is the model's uncertainty, not a score
    thresholded into a verdict, and `evidence` is mandatory so the opinion
    always points at what was actually seen rather than standing alone.
    """

    kind: ConcernKind
    confidence: Confidence
    evidence: str

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("evidence must not be empty (or whitespace-only)")


@dataclass(frozen=True, slots=True)
class WelfareAssessment:
    """The full set of welfare concerns read from one keyframe.

    Duplicate `kind`s collapse to the highest-confidence concern of that kind
    instead of being stored twice: a model reporting `collapse` under two
    prompt phrasings is describing one person's one situation, and keeping
    both would double-count it in any routing rule built on `len(concerns)`
    or on iterating concerns by kind.
    """

    concerns: tuple[WelfareConcern, ...]
    basis: Literal["single_frame_vlm"] = "single_frame_vlm"

    def __post_init__(self) -> None:
        # `Literal["single_frame_vlm"]` is a mypy-only guarantee: nothing stops
        # `WelfareAssessment(concerns=(), basis="anything")` at runtime otherwise.
        # This type is reconstructed at untrusted-JSON boundaries (the event codec
        # today; a later task's VLM-response parser), so the check belongs here,
        # once, rather than relying on every such caller to remember it.
        if self.basis != "single_frame_vlm":
            raise ValueError(f"basis must be 'single_frame_vlm', got {self.basis!r}")
        best_by_kind: dict[ConcernKind, WelfareConcern] = {}
        for concern in self.concerns:
            existing = best_by_kind.get(concern.kind)
            if existing is None or (
                _CONFIDENCE_RANK[concern.confidence] > _CONFIDENCE_RANK[existing.confidence]
            ):
                best_by_kind[concern.kind] = concern
        if len(best_by_kind) != len(self.concerns):
            collapsed = tuple(
                concern for concern in self.concerns if best_by_kind.get(concern.kind) is concern
            )
            # `concerns` is frozen; go around it the same way a dataclass
            # setter would have to, but only once, only here.
            object.__setattr__(self, "concerns", collapsed)

    @classmethod
    def none(cls) -> WelfareAssessment:
        """The empty assessment: nothing of concern in this keyframe."""
        return cls(concerns=())

    def highest_confidence(self, kind: ConcernKind) -> Confidence | None:
        """The confidence of the concern matching `kind`, or `None` if absent.

        `__post_init__` already guarantees at most one concern per kind, so
        this is a lookup, not a reduction.
        """
        for concern in self.concerns:
            if concern.kind is kind:
                return concern.confidence
        return None
