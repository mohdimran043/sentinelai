"""Authorised people, and the face measurements that identify them (spec §8-§12).

This is the most sensitive data in the system and the module is shaped by that rather
than by convenience. Three decisions are worth reading before anything else here is
used.

**An embedding is biometric data, not a hash.** It is not reversible to a photograph,
but it identifies a specific person across cameras and across time, and that is the
property that makes it sensitive. `FaceEmbedding` therefore carries no name, no person
id and no camera id — it is a bare vector with a dimension. Whatever binds it to a
person lives in the store (`ports/face.py`), where it can be encrypted and deleted as
one unit.

**There is no `matches()` returning a bool.** Similarity is a number and the threshold
is configuration (§12: "do not hard-code biometric thresholds"). A boolean baked in
here would put the most consequential number in the system somewhere nobody reviewing a
deployment would look.

**Quality is a precondition, not a tiebreak.** A low-quality face does not produce a
low-confidence match; it produces *no match attempt at all* (`FaceQuality.is_usable`).
A blurred, tiny or extremely side-on face embeds to a vector that sits near the middle
of the space and is therefore mildly similar to everybody — which reads as a weak match
against whichever enrolled person happens to be closest. Rejecting it up front is the
only honest handling, and it is why §10 asks for a minimum face size and quality
threshold rather than only a match threshold.

Pure: no clock, no I/O, no numpy. The vector is a plain tuple of floats, which keeps
this module importable in `domain/` and costs a conversion at the adapter boundary that
is irrelevant next to a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt
from uuid import UUID

__all__ = [
    "AuthorizedPerson",
    "FaceEmbedding",
    "FaceQuality",
    "PersonStatus",
    "cosine_similarity",
]


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Cosine similarity of two embeddings, in [-1, 1].

    Cosine rather than Euclidean distance because every face-embedding model this
    would plausibly use trains with an angular margin loss (ArcFace and its
    descendants), so the space is meaningful on the unit sphere and magnitude carries
    no identity information.

    Raises on a dimension mismatch rather than returning a low score. Two vectors of
    different lengths are not "dissimilar", they are incomparable — usually because the
    embedding model was swapped without re-enrolling anybody — and returning 0.0 would
    turn that migration error into a site where nobody is ever recognised and nothing
    says why.
    """
    if len(left) != len(right):
        raise ValueError(
            f"cannot compare embeddings of different dimensions ({len(left)} vs "
            f"{len(right)}) — this usually means the embedding model changed and the "
            f"enrolled faces need regenerating"
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        # A zero vector has no direction, so it has no similarity to anything. 0.0 is
        # the neutral answer rather than an error: a model can emit one for a
        # degenerate crop, and one bad crop must not take a pipeline down.
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class FaceEmbedding:
    """One face, as a vector. Carries no identity of its own — see the module docstring.

    `slots` and `frozen` so it is cheap and hashable: a camera comparing one face
    against every enrolled person does that comparison thousands of times an hour.
    """

    vector: tuple[float, ...]
    dimension: int

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("an embedding needs at least one dimension")
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"embedding declares {self.dimension} dimensions but carries "
                f"{len(self.vector)} values"
            )

    @classmethod
    def of(cls, values: tuple[float, ...]) -> FaceEmbedding:
        return cls(vector=values, dimension=len(values))

    def similarity_to(self, other: FaceEmbedding) -> float:
        return cosine_similarity(self.vector, other.vector)


@dataclass(frozen=True, slots=True)
class FaceQuality:
    """Whether a detected face is worth trying to identify at all (§10).

    Deliberately three independent measurements rather than one blended score. They
    fail for different reasons and an operator tuning a camera needs to know which:
    "faces are too small" is a lens and mounting problem, "faces are too side-on" is a
    camera-angle problem, and "the detector is unsure" is a lighting problem. A single
    number would tell them only that something is wrong.
    """

    box_pixels: int
    """The smaller side of the face box, in pixels. The dimension that actually limits
    what an embedder can read — a 200x20 sliver is a 20-pixel face."""

    detector_confidence: float
    """The face detector's own score for this being a face at all."""

    frontality: float
    """How face-on the subject is, in [0, 1], where 1 is looking at the camera.

    Estimated by the adapter from landmark geometry. Its own scale rather than an
    angle, because different detectors expose different landmarks and a normalised
    0-to-1 reading is the most any of them can honestly support.
    """

    def is_usable(
        self, *, min_box_pixels: int, min_confidence: float, min_frontality: float
    ) -> bool:
        """Whether this face clears every bar. **All three**, not a weighted blend.

        A blend lets an excellent frontality score compensate for a 12-pixel face, and
        it should not: the small face is unreadable whichever way it is pointing, and
        the resulting embedding will sit near the middle of the space and read as a
        mild match against everybody enrolled.
        """
        return (
            self.box_pixels >= min_box_pixels
            and self.detector_confidence >= min_confidence
            and self.frontality >= min_frontality
        )


class PersonStatus(StrEnum):
    """Whether an enrolled person's authorisation is currently in force.

    `DISABLED` rather than deletion is the point: §12 requires an audit trail and a
    deletion function, and those are different operations. Disabling revokes access
    while keeping the record of who was authorised and when — which is what an
    investigation of a past incident needs. Deleting removes the biometric data, which
    is what a person exercising a data right needs. Conflating them means one of those
    two obligations cannot be met.
    """

    ACTIVE = "active"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class AuthorizedPerson:
    """Somebody permitted to be somewhere, and where.

    Holds **no embeddings and no images**. Those live in the store behind
    `ports/face.py`, keyed by `person_id`, so that this record — the part a console
    lists, searches and displays — can be read and cached without biometric data
    travelling with it. §12's "avoid exposing biometric information unnecessarily" is
    a statement about defaults, and this is the default.
    """

    person_id: UUID
    display_name: str
    status: PersonStatus = PersonStatus.ACTIVE

    external_reference: str | None = None
    """A site's own identifier — staff number, badge id. Optional, opaque here, and
    never used for matching: it exists so an operator can reconcile this record with
    whatever system actually governs employment."""

    camera_ids: frozenset[str] = frozenset()
    """Which cameras this person is authorised on. **Empty means none**, not all.

    The dangerous default is the other way round, and it is the one to avoid: a person
    enrolled with no cameras assigned should be authorised nowhere until somebody says
    where, rather than everywhere until somebody says otherwise. §10 asks for
    camera-specific authorised lists, and this is what makes forgetting to set one a
    visible failure instead of a silent grant.
    """

    zones: frozenset[str] = frozenset()
    """Zone names this person is authorised in, as an alternative to naming cameras.
    Empty means none, for `camera_ids`' reason."""

    expires_at: float | None = None
    """Unix epoch seconds after which this authorisation lapses, or `None` for no
    expiry. §10's temporary authorisation — a contractor for an afternoon — without
    which every temporary grant becomes a permanent one somebody forgot to revoke."""

    notes: str = ""

    def is_authorized_on(self, *, camera_id: str, zone: str | None, now: float) -> bool:
        """Whether this person may be on this camera at this moment.

        Every clause must pass: active, unexpired, and named for this camera or its
        zone. Expressed as one function rather than left to callers so that a future
        clause cannot be added in one place and forgotten in another — which for an
        access rule means somebody being admitted somewhere nobody intended.
        """
        if self.status is not PersonStatus.ACTIVE:
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        if camera_id in self.camera_ids:
            return True
        return zone is not None and zone in self.zones


@dataclass(frozen=True, slots=True)
class EnrolledFace:
    """One reference face on a person's record — the vector, and what it looks like.

    Identified, because a person accumulates several (§10 asks for multiple references
    per person) and an operator looking at a bad one needs to be able to remove *that*
    one rather than the person.

    `has_image` rather than the image itself: a roster listing is read far more often
    than any single photograph is looked at, and a list endpoint that carried the
    pictures would move megabytes of biometric data to draw a table of names. The image
    is fetched deliberately, one at a time, by whoever actually needs to look.
    """

    face_id: UUID
    enrolled_at: float
    """Unix epoch seconds. Which reference is the oldest is the usual question when
    somebody has stopped being recognised."""

    has_image: bool
    """Whether a reference image was stored with this embedding.

    False for faces enrolled before images were kept, and for a deployment that turned
    them off. The distinction is worth carrying: an operator who sees no photograph
    should know whether none was kept or whether one failed to load."""


@dataclass(frozen=True, slots=True)
class MatchCandidate:
    """One enrolled person, and how closely this face resembled them.

    Carried rather than reduced to a winner because §10 asks an operator to be shown
    the near-match and its confidence. "Nobody matched" and "somebody matched at 0.38
    against a 0.45 threshold" are different situations, and the second is the one worth
    looking at a clip for.
    """

    person_id: UUID
    display_name: str
    similarity: float
    authorized_here: bool
    """Whether this person is authorised on *this* camera. Separate from the
    similarity, because a confident match against somebody authorised elsewhere is an
    unauthorised presence rather than a failure to recognise — and telling an operator
    "this is Employee B, who is not permitted in this wing" is far more use than
    "unknown person"."""


@dataclass(frozen=True, slots=True)
class FaceObservation:
    """One face seen on one frame, ready to be matched.

    `track_id` is what makes temporal confirmation possible: §11 requires that a
    single low-confidence frame never raises an alert, and "the same tracked person
    across N frames" is the only way to express that.
    """

    track_id: int
    embedding: FaceEmbedding
    quality: FaceQuality
    timestamp: float
    candidates: tuple[MatchCandidate, ...] = field(default_factory=tuple)

    @property
    def best(self) -> MatchCandidate | None:
        """The closest enrolled person, or `None` if nobody was compared.

        `max` over similarity — but note this says nothing about whether the match is
        *good enough*. That is the threshold's job, and it lives in
        `domain/policy/authorization.py` with the rest of the configuration.
        """
        return max(self.candidates, key=lambda c: c.similarity, default=None)
