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
* **`basis` is a mandatory marker naming the evidence behind the opinion.**
  Every `WelfareAssessment` carries one, so a consumer reading the payload
  alone — with no side channel, no tribal knowledge of which pipeline produced
  it — can see where the opinion came from and weigh it accordingly. It began
  single-valued (`single_frame_vlm`) when that was the only source there was,
  with the stated rule that a future second source would get **its own value
  rather than silently widening what the first one means**. That rule has now
  been exercised: `temporal_pose_vlm` (spec §7) is the same vision-language
  reading corroborated by a multi-second geometry state machine, and it is a
  second member precisely so that every opinion already stored under
  `single_frame_vlm` keeps meaning what it meant when it was written.

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
from typing import Literal, get_args


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

    def meets(self, minimum: Confidence) -> bool:
        """Whether this tier is at least `minimum`.

        A method rather than leaving every caller to compare members directly:
        `Confidence` is a `StrEnum`, so `Confidence.POSSIBLE >= Confidence.LIKELY`
        compiles, runs, and answers by *alphabetical* order of the values —
        `"possible" > "likely"` is True, which is the exact inversion of what a
        caller writing that comparison means. The rank table below is the only
        ordering this type has, and this is the only way to reach it.
        """
        return _CONFIDENCE_RANK[self] >= _CONFIDENCE_RANK[minimum]


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
    evidence_stated: bool = True
    """`False` means the model named this concern but described nothing —
    `evidence` then holds a fixed, honest placeholder
    (`adapters/vision/qwen25vl.py`'s `_EVIDENCE_UNSTATED`), not something the
    model actually said. `True` (the default) means `evidence` is the model's
    own text.

    This started as a private adapter-only distinction: the only way to tell
    "named but not described" from "actually evidenced" was to string-match
    `_EVIDENCE_UNSTATED`, a leading-underscore adapter constant, from whatever
    later reads `WelfareConcern` to route a notification. That forces a
    ports-boundary violation (importing an adapter internal from wherever
    routing lives) or, worse, a routing rule that just never makes the
    distinction and treats a placeholder the same as a real observation. Given
    a muted alarm is the worst outcome this system is built to avoid, the
    flag belongs here, in the domain, where every consumer can read it without
    knowing which adapter produced it or what string it used.

    Kept a plain `bool`, not folded into `Confidence`: whether evidence was
    given and how sure the model is are independent axes — a `likely` concern
    can arrive with no evidence text, and a `possible` one can arrive with a
    detailed observation. Collapsing them would lose information either
    reading could need.
    """

    def __post_init__(self) -> None:
        if not self.evidence.strip():
            raise ValueError("evidence must not be empty (or whitespace-only)")


AssessmentBasis = Literal["single_frame_vlm", "temporal_pose_vlm"]
"""What kind of evidence an assessment rests on.

* `single_frame_vlm` — a vision-language model's reading of one still frame, and
  nothing else. No motion, no history, no second opinion.
* `temporal_pose_vlm` — that same reading, corroborated by
  `domain/behaviour/fall.py`'s state machine having watched the person go from
  upright, through a rapid descent, to horizontal, and stay down. Strictly more
  evidence than the first.

A `Literal` union rather than a `StrEnum`, unlike every other closed vocabulary in
this package, because this field is *also* the thing a consumer branches on to decide
how much to trust the record, and the values are written into the published event
schema as a plain enum of strings. Keeping it a `Literal` keeps the type and the wire
form the same object with no encoder in between, which is the property that made the
original `const` safe to reason about.
"""

VALID_BASES: frozenset[str] = frozenset(get_args(AssessmentBasis))
"""The members, as data, for the runtime check below and for the tests that pin this
against the published schema. Derived from the `Literal` rather than repeated, so a
member added above cannot be forgotten here."""


@dataclass(frozen=True, slots=True)
class WelfareAssessment:
    """The full set of welfare concerns read from one keyframe.

    Duplicate `kind`s collapse to the highest-confidence concern of that kind
    instead of being stored twice: a model reporting `collapse` under two
    prompt phrasings is describing one person's one situation, and keeping
    both would double-count it in any routing rule built on `len(concerns)`
    or on iterating concerns by kind. At equal confidence, the concern with
    `evidence_stated=True` wins over one without — see `__post_init__` for
    why evidence, not just confidence, decides the tie.
    """

    concerns: tuple[WelfareConcern, ...]
    basis: AssessmentBasis = "single_frame_vlm"
    """Defaulted to the weaker of the two on purpose. An assessment that did not say
    otherwise was produced by looking at one frame, and a default of the stronger
    value would let a caller silently claim corroboration it never had."""

    def __post_init__(self) -> None:
        # The `Literal` is a mypy-only guarantee: nothing stops
        # `WelfareAssessment(concerns=(), basis="anything")` at runtime otherwise.
        # This type is reconstructed at untrusted-JSON boundaries (the event codec,
        # the VLM-response parser), so the check belongs here, once, rather than
        # relying on every such caller to remember it.
        if self.basis not in VALID_BASES:
            raise ValueError(f"basis must be one of {sorted(VALID_BASES)}, got {self.basis!r}")
        best_by_kind: dict[ConcernKind, WelfareConcern] = {}
        for concern in self.concerns:
            existing = best_by_kind.get(concern.kind)
            # Rank by (confidence, evidence_stated), in that order. Confidence
            # dominates: a `likely` concern with no evidence text still outranks
            # a `possible` one that has evidence, because confidence is the
            # model's own judgement of how sure it is, and a placeholder
            # evidence string must not be allowed to override that judgement.
            # `evidence_stated` only breaks a tie *within* the same confidence
            # tier — between two `likely` (or two `possible`) reports of the
            # same kind, the one a human can actually weigh against a real
            # observation should win over the one that only says "reported,
            # not described" (`_EVIDENCE_UNSTATED`). `bool` orders `False <
            # True` in Python, so this needs no extra rank table.
            if existing is None or (
                _CONFIDENCE_RANK[concern.confidence],
                concern.evidence_stated,
            ) > (
                _CONFIDENCE_RANK[existing.confidence],
                existing.evidence_stated,
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
