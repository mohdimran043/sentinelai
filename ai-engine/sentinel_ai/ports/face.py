"""The face pipeline's seams (spec §9, §31).

Three ports rather than one, because the three things fail differently and are replaced
independently:

* `FaceDetector` finds faces in a frame and says how usable each one is.
* `FaceEmbedder` turns an aligned face into a vector.
* `FaceStore` holds enrolled people and their embeddings, and searches them.

Splitting detection from embedding is not ceremony. §9's enrolment flow is *detect →
quality check → align → embed*, with a rejection between every pair, and a combined
port would have to either run the whole chain or expose its middle — at which point it
is two ports with extra steps. It also lets the store be swapped for a vector database
later without the models noticing, which is the migration §29's search section implies.

Why the store is a port at all
-------------------------------
Because what it holds is biometric data, and §12 requires encryption at rest, deletion,
and an audit trail. Those are properties of a *storage implementation*, not of the
domain, and putting them behind a seam is what lets a deployment substitute a
compliance-approved store without touching the matching logic. The in-repo
implementation encrypts to disk; a site with a key-management service replaces it.

The ordering obligation, which is easy to get wrong
-----------------------------------------------------
`FaceDetector.detect` returns quality alongside geometry, and callers must consult it
**before** embedding. Embedding a face that failed its quality check is not merely
wasted GPU: the resulting vector sits near the middle of the space and is mildly
similar to everybody enrolled, which reads downstream as a weak match against whoever
happens to be closest. `domain/policy/authorization.py` discards such faces rather than
counting them, and it can only do that if the quality survives to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

from sentinel_ai.domain.entities import BBox
from sentinel_ai.domain.identity import (
    AuthorizedPerson,
    EnrolledFace,
    FaceEmbedding,
    FaceQuality,
    MatchCandidate,
)
from sentinel_ai.ports.frame_source import FrameData

__all__ = [
    "DetectedFace",
    "FaceDetector",
    "FaceEmbedder",
    "FaceStore",
]


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """One face found in a frame, with everything needed to decide whether to use it.

    `aligned` carries the cropped, pose-normalised face image as an opaque `object` for
    `FrameData.pixels`' reason — it is a numpy array at runtime, and typing it as one
    would drag numpy into `ports/` and fail the architecture fitness test. The embedder
    `isinstance`-checks it at the boundary.
    """

    box: BBox
    quality: FaceQuality
    aligned: object


class FaceDetector(ABC):
    @abstractmethod
    async def detect(self, frame: FrameData) -> tuple[DetectedFace, ...]:
        """Every face in the frame, with its quality, aligned and ready to embed.

        Returns **all** faces including unusable ones, with the quality attached rather
        than filtering them out. Filtering here would put the thresholds in the adapter,
        where §12 forbids them from being hard-coded and where an operator tuning a
        camera could not reach them. The caller applies the policy.

        A frame with no faces is an empty tuple, not an error — it is the overwhelmingly
        common case on any camera.
        """


class FaceEmbedder(ABC):
    @abstractmethod
    async def embed(self, faces: tuple[DetectedFace, ...]) -> tuple[FaceEmbedding, ...]:
        """One embedding per input face, in the same order.

        Batched deliberately: a frame with four people is four crops that a single
        forward pass handles roughly as cheaply as one, and this is the stage that runs
        per person rather than per frame.

        Order-preserving is a contract the caller depends on to pair embeddings back to
        their detections; an implementation that dropped a face it could not embed
        would silently misalign every subsequent pair. Implementations that cannot
        embed a face must raise rather than skip it.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The embedding size this model produces.

        Exposed so the store can refuse to compare vectors from a different model. A
        site that swaps embedders and does not re-enrol gets a loud dimension error
        rather than a system where nobody is ever recognised and nothing says why.
        """


class FaceStore(ABC):
    """Enrolled people, their embeddings, and search over them.

    Every method here touches biometric data, which is why the port exists — see the
    module docstring.
    """

    @abstractmethod
    async def add_person(self, person: AuthorizedPerson) -> None:
        """Create or replace a person record. Holds no biometric data of its own."""

    @abstractmethod
    async def get_person(self, person_id: UUID) -> AuthorizedPerson | None: ...

    @abstractmethod
    async def list_people(self) -> tuple[AuthorizedPerson, ...]: ...

    @abstractmethod
    async def add_embedding(
        self, person_id: UUID, embedding: FaceEmbedding, *, image: bytes | None = None
    ) -> UUID:
        """Enrol one more reference face for a person (§10: multiple images each).

        More than one matters: a single reference photograph taken face-on under
        office lighting matches poorly against a corridor camera at an angle, and the
        usual fix is not a better threshold but a second reference.

        Returns the new face's id, so a caller can immediately show or remove it.

        `image` is an optional JPEG of the face **as cropped by the detector**, not the
        photograph it came from. It exists so an operator can see who is on the roster
        and check an identification by eye. A store must hold it under the same
        protection as the embedding — see `EncryptedFaceStore` — and `None` must remain
        supported: a deployment that would rather keep no pictures keeps none, and
        everything except the pictures still works.
        """

    @abstractmethod
    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        """Every reference face on this person's record, oldest first.

        Metadata only. See `EnrolledFace.has_image` for why the pictures are not here.
        """

    @abstractmethod
    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        """The stored JPEG for one reference face, or `None` if there is not one.

        `None` covers both "no image was kept" and "no such face", deliberately: the
        caller is about to render a picture or a placeholder either way, and a store
        that distinguished them would invite an endpoint that lets somebody enumerate
        which of a person's faces exist.
        """

    @abstractmethod
    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        """Remove one reference face — its embedding and its image. False if unknown.

        Distinct from `delete_person`, and the finer grain is the point: a bad
        reference photograph makes somebody harder to recognise, and the fix is to drop
        that photograph, not to un-enrol the person and start again.
        """

    @abstractmethod
    async def reference_count(self, person_id: UUID) -> int:
        """How many reference faces are enrolled for this person.

        A count and never the faces. It is the one fact about somebody's biometric
        record that a console legitimately needs — §10 asks for multiple references per
        person, and "1" is usually the reason somebody is not recognised from an angle
        — and exposing it costs nothing, where exposing the vectors would cost
        everything §12 is protecting.

        Zero for a person who exists but has no faces, and zero for one who does not
        exist: the distinction is `get_person`'s to make, and duplicating it here would
        give two places to disagree.
        """

    @abstractmethod
    async def delete_person(self, person_id: UUID) -> bool:
        """Remove a person **and every embedding and image held for them**.

        §12 requires deletion to be available, and requires it to be real: a record
        that removed the name while leaving the vectors would keep exactly the part
        that identifies somebody. Returns False when there was nothing to delete, so a
        caller can tell a completed deletion from a no-op rather than reporting success
        for a person who was never there.
        """

    @abstractmethod
    async def search(
        self, embedding: FaceEmbedding, *, camera_id: str, zone: str | None, now: float
    ) -> tuple[MatchCandidate, ...]:
        """Every enrolled person, ranked by similarity to `embedding`.

        Returns **candidates with scores**, never a verdict. The threshold is
        configuration (§12) and lives in `domain/policy/authorization.py`; a store that
        returned only matches above some internal bar would put the most consequential
        number in the system somewhere nobody reviewing a deployment would look.

        `camera_id`, `zone` and `now` are taken so the store can mark each candidate
        `authorized_here` — which depends on the person's camera and zone lists and on
        whether their authorisation has expired. Computing that here rather than in the
        caller keeps `AuthorizedPerson.is_authorized_on`'s clauses in one place.
        """

    @abstractmethod
    async def close(self) -> None: ...
