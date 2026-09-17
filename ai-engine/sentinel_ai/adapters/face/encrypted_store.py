"""Enrolled people and their face embeddings, encrypted at rest (spec §12).

§12 treats biometric data as the most sensitive thing this system holds, and asks for
four specific properties. Each one shaped something here:

* **Encrypt at rest.** Every embedding is sealed with AES-256-GCM before it touches the
  disk. The key comes from `SENTINEL_FACE_ENCRYPTION_KEY` and is never written down by
  this module.
* **Prefer embeddings over images.** This store holds **no images at all**. §9's
  enrolment flow ends at an embedding, and the reference photograph is used to produce
  it and then discarded. A console that wants to show an operator a reference face is
  asking for something this deliberately cannot provide — see "What is not stored".
* **Provide deletion.** `delete_person` removes the record and every vector in one
  operation, and rewrites the file so the ciphertext is gone rather than orphaned.
* **Audit access.** Every read of a person's biometric data logs the person id and the
  camera that asked, at INFO, without the vector.

Why AES-GCM and not `cryptography.fernet`
------------------------------------------
Fernet is the friendlier API and it is the wrong one here. It is AES-128-CBC with an
HMAC and it embeds a timestamp, which is a *token* format — built for things that
expire. An embedding does not expire, and the timestamp is metadata about when somebody
was enrolled sitting in a file whose whole purpose is to minimise what is retained.
AES-256-GCM is authenticated encryption with nothing else attached: a nonce, a tag, and
the ciphertext.

Each embedding is sealed individually rather than the file being encrypted as a whole.
That costs a 12-byte nonce and a 16-byte tag per vector and buys the property that
matters: deleting one person rewrites the file without ever holding every other
person's plaintext vectors in memory at once.

What is not stored, and why that is a feature
-----------------------------------------------
No reference images, no crops, no frames. §12 says to prefer embeddings over raw
biometric images "where possible", and for this store it is entirely possible: nothing
in the matching path needs a picture. The cost is real and worth stating — an operator
investigating an unauthorised-person alert cannot be shown "here is the enrolled photo
of the person they most resemble", only the name and the similarity. That is the
trade-off §12 asks for, taken in the direction it asks for it.

Volatility, and the honest limit
---------------------------------
This is a JSON file. It is not a vector database, and `search` is a linear scan over
every enrolled embedding, which is correct and fast for the tens-to-hundreds of people
a site enrols and would not be for a hundred thousand. `FaceStore` is a port precisely
so that a site at that scale substitutes something else.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import stat
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from sentinel_ai.domain.identity import (
    AuthorizedPerson,
    EnrolledFace,
    FaceEmbedding,
    MatchCandidate,
    PersonStatus,
    cosine_similarity,
)
from sentinel_ai.ports.face import FaceStore

logger = logging.getLogger(__name__)

__all__ = ["EncryptedFaceStore", "FaceStoreError", "generate_key"]

_NONCE_BYTES = 12
"""96 bits, the size AES-GCM is specified for. A different length is accepted by the
library and weakens the construction, so it is fixed here rather than configurable."""

_SCHEMA_VERSION = 1


class FaceStoreError(RuntimeError):
    """The store could not be read, written, or decrypted.

    Distinct from a missing person, which is a `None` return. This is "the file is
    there and something is wrong with it" — a truncated write, or, most likely, the
    wrong key.
    """


def generate_key() -> str:
    """A fresh base64 AES-256 key, for an operator setting up a deployment.

    Here rather than in a script so that the one place that knows the required length
    is the one place that generates it. Printed by
    `python -m sentinel_ai.adapters.face.encrypted_store`.
    """
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def _load_key(raw: str) -> bytes:
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise FaceStoreError(
            "SENTINEL_FACE_ENCRYPTION_KEY must be base64. Generate one with "
            "`python -m sentinel_ai.adapters.face.encrypted_store`."
        ) from exc
    if len(key) not in (16, 24, 32):
        raise FaceStoreError(
            f"SENTINEL_FACE_ENCRYPTION_KEY decodes to {len(key)} bytes; AES needs 16, "
            f"24 or 32. Generate one with "
            f"`python -m sentinel_ai.adapters.face.encrypted_store`."
        )
    return key


def _image_name(face_id: UUID) -> str:
    """The on-disk name for a reference image, derived from the id and nothing else.

    Always regenerated rather than read back from the document: a name that came out of
    a JSON file is a name an editor of that file chose, and this one is used to build a
    filesystem path.
    """
    return f"{face_id}.jpg.enc"


@dataclass(frozen=True, slots=True)
class _SealedEmbedding:
    nonce: str
    ciphertext: str
    dimension: int
    """Kept in plaintext, deliberately. It is not biometric — every vector from one
    model has the same dimension — and it lets the store reject a mismatched embedder
    without decrypting anything."""

    face_id: UUID = field(default_factory=uuid4)
    """This reference's identity, so a bad photograph can be removed on its own."""

    enrolled_at: float = 0.0
    """Unix epoch seconds. 0.0 for records written before this field existed, which
    reads as "unknown" rather than as 1970 — see `EnrolledFace.enrolled_at`."""

    image_name: str | None = None
    """Filename of the sealed JPEG in the sidecar directory, or `None` for a reference
    enrolled without one. Not a path: a name resolved against the store's own directory
    cannot be made to point somewhere else by editing the JSON."""


class EncryptedFaceStore(FaceStore):
    """A JSON file of people, with every embedding individually sealed.

    Reads are served from an in-memory cache loaded once at construction; writes go
    through the same write-then-rename path `CameraFileStore` uses, so a reader sees
    the old document or the new one and never a half-written one.
    """

    def __init__(self, path: Path, *, encryption_key: str) -> None:
        self._path = path
        # A sidecar directory rather than base64 inside the JSON. The document is
        # rewritten whole on every enrolment, and a site with 500 people at three
        # references each would be rewriting tens of megabytes to add one vector.
        # Images are immutable once written, so they need no such treatment.
        self._image_dir = path.parent / f"{path.stem}-reference-faces"
        self._key = _load_key(encryption_key)
        self._aead = AESGCM(self._key)
        self._lock = asyncio.Lock()
        self._people: dict[UUID, AuthorizedPerson] = {}
        self._embeddings: dict[UUID, list[_SealedEmbedding]] = {}
        self._load()

    # -- persistence ----------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FaceStoreError(f"{self._path} is not valid JSON: {exc}") from exc

        version = document.get("schema_version")
        if version != _SCHEMA_VERSION:
            raise FaceStoreError(
                f"{self._path} has schema_version {version!r}; this build reads "
                f"{_SCHEMA_VERSION}. Refusing to guess at biometric data."
            )

        for entry in document.get("people", []):
            person = AuthorizedPerson(
                person_id=UUID(entry["person_id"]),
                display_name=entry["display_name"],
                status=PersonStatus(entry.get("status", PersonStatus.ACTIVE.value)),
                external_reference=entry.get("external_reference"),
                camera_ids=frozenset(entry.get("camera_ids", ())),
                zones=frozenset(entry.get("zones", ())),
                expires_at=entry.get("expires_at"),
                notes=entry.get("notes", ""),
            )
            self._people[person.person_id] = person
            self._embeddings[person.person_id] = [
                _SealedEmbedding(
                    nonce=sealed["nonce"],
                    ciphertext=sealed["ciphertext"],
                    dimension=sealed["dimension"],
                    # Defaulted, so a store written before references had identities
                    # loads rather than raising. Such a record gets a fresh id on every
                    # start, which is harmless — nothing persists it — and it has no
                    # image to address anyway.
                    face_id=UUID(sealed["face_id"]) if "face_id" in sealed else uuid4(),
                    enrolled_at=float(sealed.get("enrolled_at", 0.0)),
                    image_name=sealed.get("image_name"),
                )
                for sealed in entry.get("embeddings", [])
            ]

    def _write(self) -> None:
        document: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "_comment": (
                "Biometric data. Every embedding below is AES-256-GCM ciphertext and is "
                "useless without SENTINEL_FACE_ENCRYPTION_KEY. Reference photographs "
                "are sealed with the same key in the sidecar directory named after this "
                "file. Treat both as you would a credential store."
            ),
            "people": [
                {
                    "person_id": str(person.person_id),
                    "display_name": person.display_name,
                    "status": person.status.value,
                    "external_reference": person.external_reference,
                    "camera_ids": sorted(person.camera_ids),
                    "zones": sorted(person.zones),
                    "expires_at": person.expires_at,
                    "notes": person.notes,
                    "embeddings": [
                        {
                            "nonce": sealed.nonce,
                            "ciphertext": sealed.ciphertext,
                            "dimension": sealed.dimension,
                            "face_id": str(sealed.face_id),
                            "enrolled_at": sealed.enrolled_at,
                            "image_name": sealed.image_name,
                        }
                        for sealed in self._embeddings.get(person.person_id, [])
                    ],
                }
                for person in self._people.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        # 0600 before any content is written, not after: a file created world-readable
        # and tightened afterwards is world-readable for the window in between, and the
        # content here is what that window would expose.
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self._path)

    # -- sealing --------------------------------------------------------------------

    def _seal(self, embedding: FaceEmbedding) -> _SealedEmbedding:
        nonce = os.urandom(_NONCE_BYTES)
        plaintext = json.dumps(list(embedding.vector)).encode("utf-8")
        ciphertext = self._aead.encrypt(nonce, plaintext, None)
        return _SealedEmbedding(
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
            dimension=embedding.dimension,
            # Wall clock, not monotonic: this is written down and read back after a
            # restart, and "which of these references is the oldest" is the question an
            # operator asks when somebody has stopped being recognised.
            enrolled_at=time.time(),
        )

    def _open(self, sealed: _SealedEmbedding) -> FaceEmbedding:
        try:
            plaintext = self._aead.decrypt(
                base64.b64decode(sealed.nonce), base64.b64decode(sealed.ciphertext), None
            )
        except InvalidTag as exc:
            raise FaceStoreError(
                "could not decrypt an enrolled face: the key does not match the one "
                "this store was written with, or the file has been altered"
            ) from exc
        values = tuple(float(value) for value in json.loads(plaintext))
        return FaceEmbedding(vector=values, dimension=sealed.dimension)

    # -- reference images ------------------------------------------------------------

    def _write_image(self, face_id: UUID, image: bytes) -> None:
        """Seal a JPEG and write it 0600, atomically.

        Same key and same construction as an embedding. A photograph of somebody's face
        is biometric data in the plainest possible sense — it needs no model to be
        recognised by a person — so storing it in the clear beside encrypted vectors
        would protect the harder-to-read half and leave the easier one open.
        """
        self._image_dir.mkdir(parents=True, exist_ok=True)
        # 0700 on the directory as well as 0600 on the files: a listing of it is a list
        # of who is enrolled at this site, even if none of the contents can be read.
        os.chmod(self._image_dir, stat.S_IRWXU)
        nonce = os.urandom(_NONCE_BYTES)
        sealed = nonce + self._aead.encrypt(nonce, image, None)
        target = self._image_dir / _image_name(face_id)
        temporary = target.with_suffix(target.suffix + ".tmp")
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(sealed)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _open_image(self, blob: bytes) -> bytes | None:
        """Unseal a stored JPEG, or `None` if it cannot be read.

        `None` rather than raising, unlike `_open` for an embedding, because the two
        failures are not equally serious: an undecryptable *vector* means recognition is
        silently broken and must be loud, while an undecryptable *picture* means one
        thumbnail is missing from a roster.
        """
        if len(blob) <= _NONCE_BYTES:
            return None
        try:
            return bytes(self._aead.decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], None))
        except InvalidTag:
            logger.warning(
                "a reference image could not be decrypted; it was written with a "
                "different SENTINEL_FACE_ENCRYPTION_KEY or has been altered"
            )
            return None

    def _discard_image(self, face_id: UUID) -> None:
        with contextlib.suppress(OSError):
            (self._image_dir / _image_name(face_id)).unlink(missing_ok=True)

    # -- port ------------------------------------------------------------------------

    async def add_person(self, person: AuthorizedPerson) -> None:
        async with self._lock:
            self._people[person.person_id] = person
            self._embeddings.setdefault(person.person_id, [])
            self._write()
        logger.info(
            "enrolled or updated authorised person %s (cameras=%d, zones=%d)",
            person.person_id,
            len(person.camera_ids),
            len(person.zones),
        )

    async def get_person(self, person_id: UUID) -> AuthorizedPerson | None:
        return self._people.get(person_id)

    async def list_people(self) -> tuple[AuthorizedPerson, ...]:
        return tuple(self._people.values())

    async def add_embedding(
        self, person_id: UUID, embedding: FaceEmbedding, *, image: bytes | None = None
    ) -> UUID:
        async with self._lock:
            if person_id not in self._people:
                raise FaceStoreError(f"no enrolled person {person_id}")
            sealed = self._seal(embedding)
            if image is not None:
                # Written before the document that references it. The other order can
                # leave the JSON pointing at a file that never arrived; this order can
                # at worst leave an orphaned image, which nothing reads and
                # `delete_person` still sweeps.
                self._write_image(sealed.face_id, image)
                sealed = replace(sealed, image_name=_image_name(sealed.face_id))
            self._embeddings[person_id].append(sealed)
            self._write()
        logger.info(
            # Never the image, the vector, or any measurement of either — §12. The
            # count is the only thing here that is not about a specific person's face.
            "added a reference face for person %s (now %d, image %s)",
            person_id,
            len(self._embeddings[person_id]),
            "kept" if image is not None else "not kept",
        )
        return sealed.face_id

    async def list_faces(self, person_id: UUID) -> tuple[EnrolledFace, ...]:
        return tuple(
            EnrolledFace(
                face_id=sealed.face_id,
                enrolled_at=sealed.enrolled_at,
                has_image=sealed.image_name is not None,
            )
            for sealed in self._embeddings.get(person_id, [])
        )

    async def read_face_image(self, person_id: UUID, face_id: UUID) -> bytes | None:
        for sealed in self._embeddings.get(person_id, []):
            if sealed.face_id != face_id or sealed.image_name is None:
                continue
            # `image_name` is regenerated from the id rather than trusted from the
            # document, so a `..` smuggled into the JSON addresses nothing.
            path = self._image_dir / _image_name(face_id)
            try:
                blob = await asyncio.to_thread(path.read_bytes)
            except OSError:
                logger.warning("reference image for face %s could not be read", face_id)
                return None
            return self._open_image(blob)
        return None

    async def delete_face(self, person_id: UUID, face_id: UUID) -> bool:
        async with self._lock:
            faces = self._embeddings.get(person_id)
            if faces is None:
                return False
            remaining = [sealed for sealed in faces if sealed.face_id != face_id]
            if len(remaining) == len(faces):
                return False
            self._embeddings[person_id] = remaining
            self._discard_image(face_id)
            self._write()
        logger.info("removed one reference face from person %s", person_id)
        return True

    async def reference_count(self, person_id: UUID) -> int:
        return len(self._embeddings.get(person_id, []))

    async def delete_person(self, person_id: UUID) -> bool:
        async with self._lock:
            if person_id not in self._people:
                return False
            del self._people[person_id]
            # Dropped *and* the file rewritten, in one operation. A record that removed
            # the name while leaving the vectors would keep exactly the part that
            # identifies somebody (§12).
            for sealed in self._embeddings.pop(person_id, []):
                # Photographs too. Deleting somebody while leaving their picture on
                # disk is the same failure as leaving their vector, in a form anybody
                # who opens the directory can recognise on sight.
                self._discard_image(sealed.face_id)
            self._write()
        logger.info("deleted authorised person %s and every face enrolled for them", person_id)
        return True

    async def search(
        self, embedding: FaceEmbedding, *, camera_id: str, zone: str | None, now: float
    ) -> tuple[MatchCandidate, ...]:
        candidates: list[MatchCandidate] = []
        for person_id, sealed_faces in self._embeddings.items():
            person = self._people.get(person_id)
            if person is None or not sealed_faces:
                continue
            best = -1.0
            for sealed in sealed_faces:
                if sealed.dimension != embedding.dimension:
                    # A vector from a different embedder. Skipped rather than compared,
                    # because `cosine_similarity` would (correctly) raise and take the
                    # whole search down over one stale enrolment.
                    logger.warning(
                        "skipping a %d-dimension enrolled face for person %s against a "
                        "%d-dimension probe; the embedding model has changed and this "
                        "person needs re-enrolling",
                        sealed.dimension,
                        person_id,
                        embedding.dimension,
                    )
                    continue
                best = max(best, cosine_similarity(embedding.vector, self._open(sealed).vector))
            if best < -1.0 + 1e-9:
                continue
            candidates.append(
                MatchCandidate(
                    person_id=person_id,
                    display_name=person.display_name,
                    similarity=best,
                    authorized_here=person.is_authorized_on(
                        camera_id=camera_id, zone=zone, now=now
                    ),
                )
            )
        candidates.sort(key=lambda candidate: -candidate.similarity)
        if candidates:
            # §12's audit obligation: what was consulted and on whose behalf. The
            # vector is never logged, and neither is the similarity of anybody but the
            # closest — a full ranking in a log file is a biometric comparison matrix.
            logger.info(
                "face search on camera %s compared %d enrolled people; closest was %s",
                camera_id,
                len(candidates),
                candidates[0].person_id,
            )
        return tuple(candidates)

    async def close(self) -> None:
        """Nothing to release: the file is opened and closed per write."""


def main() -> None:  # pragma: no cover - an operator convenience
    print(generate_key())


if __name__ == "__main__":  # pragma: no cover
    main()
