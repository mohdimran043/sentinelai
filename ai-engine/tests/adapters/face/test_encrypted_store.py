"""The biometric store: encryption, deletion, and what it refuses to hold (spec §12)."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from uuid import uuid4

import pytest

from sentinel_ai.adapters.face.encrypted_store import (
    EncryptedFaceStore,
    FaceStoreError,
    generate_key,
)
from sentinel_ai.domain.identity import AuthorizedPerson, FaceEmbedding, PersonStatus

KEY = generate_key()


def a_person(**overrides: object) -> AuthorizedPerson:
    fields: dict[str, object] = {
        "person_id": uuid4(),
        "display_name": "Employee A",
        "camera_ids": frozenset({"cam-1"}),
    }
    fields.update(overrides)
    return AuthorizedPerson(**fields)  # type: ignore[arg-type]


def an_embedding(seed: float = 1.0) -> FaceEmbedding:
    return FaceEmbedding.of((seed, 0.0, 0.0))


def store(tmp_path: Path, key: str = KEY) -> EncryptedFaceStore:
    return EncryptedFaceStore(tmp_path / "faces.json", encryption_key=key)


class TestEncryption:
    async def test_no_embedding_appears_in_plaintext_on_disk(self, tmp_path: Path) -> None:
        """The property the whole module exists for. A distinctive value is enrolled
        and then looked for in the raw bytes."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, FaceEmbedding.of((0.123456789, 0.5, 0.25)))

        raw = (tmp_path / "faces.json").read_text(encoding="utf-8")
        assert "0.123456789" not in raw
        assert "0.5" not in raw or '"dimension"' in raw  # dimension is plaintext by design

    async def test_a_round_trip_returns_the_same_vector(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding(0.8))

        reopened = store(tmp_path)
        candidates = await reopened.search(an_embedding(0.8), camera_id="cam-1", zone=None, now=0.0)
        assert candidates[0].similarity == pytest.approx(1.0)

    async def test_the_wrong_key_fails_loudly_rather_than_returning_nothing(
        self, tmp_path: Path
    ) -> None:
        """Silently matching nobody would look exactly like a site where nobody is
        enrolled, and an operator would have no way to tell."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())

        wrong = store(tmp_path, key=generate_key())
        with pytest.raises(FaceStoreError, match="key does not match"):
            await wrong.search(an_embedding(), camera_id="cam-1", zone=None, now=0.0)

    def test_a_key_of_the_wrong_length_is_refused_at_construction(self, tmp_path: Path) -> None:
        import base64

        short = base64.b64encode(b"tooshort").decode("ascii")
        with pytest.raises(FaceStoreError, match="16, 24 or 32"):
            store(tmp_path, key=short)

    def test_a_non_base64_key_is_refused_with_a_usable_message(self, tmp_path: Path) -> None:
        with pytest.raises(FaceStoreError, match="base64"):
            store(tmp_path, key="not a key at all!!")

    async def test_the_file_is_not_world_readable(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        await subject.add_person(a_person())
        mode = (tmp_path / "faces.json").stat().st_mode
        assert not mode & stat.S_IROTH
        assert not mode & stat.S_IRGRP


class TestWhatIsNotStored:
    async def test_the_document_carries_no_biometric_plaintext(self, tmp_path: Path) -> None:
        """The record may name a person. It may not contain any readable part of them.

        This store used to hold no images at all, and the invariant was simply "no
        picture anywhere". Reference photographs are now kept, deliberately, so that an
        operator can see who is on the roster and check an identification by eye — so
        the invariant is narrower and has to be stated precisely: **nothing readable
        without the key**, and no picture inside this document at whatever size.
        """
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd8jpeg-ish")

        document = json.loads((tmp_path / "faces.json").read_text(encoding="utf-8"))
        record = document["people"][0]

        # The image lives beside the document, never in it: this file is rewritten whole
        # on every enrolment, and inlining pictures would make that cost grow with the
        # roster. `image_name` is a filename, and that is all it may be.
        assert record["embeddings"][0]["image_name"].endswith(".jpg.enc")
        assert "jpeg-ish" not in json.dumps(record)

        assert set(record) == {
            "person_id",
            "display_name",
            "status",
            "external_reference",
            "camera_ids",
            "zones",
            "expires_at",
            "notes",
            "embeddings",
        }

    async def test_a_reference_photograph_is_encrypted_on_disk(self, tmp_path: Path) -> None:
        """A photograph of a face is biometric data in the plainest sense — it needs no
        model to be recognised. Storing it in the clear next to encrypted vectors would
        protect the harder-to-read half and leave the easier one open."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd8secret")

        written = list((tmp_path / "faces-reference-faces").iterdir())
        assert len(written) == 1
        assert b"secret" not in written[0].read_bytes()
        assert stat.S_IMODE(written[0].stat().st_mode) == 0o600

    async def test_the_image_directory_is_not_world_listable(self, tmp_path: Path) -> None:
        """A listing of it is a list of who is enrolled at this site, even when none of
        the contents can be read."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd8x")

        directory = tmp_path / "faces-reference-faces"
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


class TestReferenceImages:
    async def test_a_stored_photograph_comes_back_exactly(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        face_id = await subject.add_embedding(
            person.person_id, an_embedding(), image=b"\xff\xd8original"
        )
        assert await subject.read_face_image(person.person_id, face_id) == b"\xff\xd8original"

    async def test_enrolling_without_a_photograph_still_works(self, tmp_path: Path) -> None:
        """A deployment that would rather keep no pictures keeps none, and everything
        except the pictures still works."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        face_id = await subject.add_embedding(person.person_id, an_embedding())

        (face,) = await subject.list_faces(person.person_id)
        assert face.has_image is False
        assert await subject.read_face_image(person.person_id, face_id) is None
        assert not (tmp_path / "faces-reference-faces").exists()

    async def test_several_references_are_listed_oldest_first(self, tmp_path: Path) -> None:
        """§10 asks for multiple references each. Which one is oldest is the usual
        question when somebody has stopped being recognised."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        first = await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd81")
        second = await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd82")

        faces = await subject.list_faces(person.person_id)
        assert [face.face_id for face in faces] == [first, second]

    async def test_removing_one_reference_leaves_the_others(self, tmp_path: Path) -> None:
        """The reason this is finer-grained than deleting the person: a bad photograph
        makes somebody *harder* to recognise, and the fix is to drop that photograph."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        doomed = await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd81")
        kept = await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd82")

        assert await subject.delete_face(person.person_id, doomed) is True
        assert [face.face_id for face in await subject.list_faces(person.person_id)] == [kept]
        assert await subject.read_face_image(person.person_id, doomed) is None
        assert await subject.read_face_image(person.person_id, kept) == b"\xff\xd82"
        assert len(list((tmp_path / "faces-reference-faces").iterdir())) == 1

    async def test_removing_an_unknown_reference_reports_it(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        assert await subject.delete_face(person.person_id, uuid4()) is False

    async def test_deleting_a_person_deletes_their_photographs(self, tmp_path: Path) -> None:
        """Deleting somebody while leaving their picture on disk is the same failure as
        leaving their vector, in a form anybody who opens the directory recognises."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd81")
        await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd82")

        assert await subject.delete_person(person.person_id) is True
        assert list((tmp_path / "faces-reference-faces").iterdir()) == []

    async def test_photographs_survive_a_reopen(self, tmp_path: Path) -> None:
        """The store is reloaded from disk on every start; a face id that changed across
        restarts would break every image URL an operator had open."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        face_id = await subject.add_embedding(
            person.person_id, an_embedding(), image=b"\xff\xd8kept"
        )

        reopened = store(tmp_path)
        (face,) = await reopened.list_faces(person.person_id)
        assert face.face_id == face_id
        assert await reopened.read_face_image(person.person_id, face_id) == b"\xff\xd8kept"

    async def test_a_photograph_written_with_another_key_is_not_fatal(self, tmp_path: Path) -> None:
        """An undecryptable *vector* means recognition is silently broken and must be
        loud. An undecryptable *picture* means one thumbnail is missing."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        face_id = await subject.add_embedding(person.person_id, an_embedding(), image=b"\xff\xd8x")

        target = tmp_path / "faces-reference-faces" / f"{face_id}.jpg.enc"
        target.write_bytes(b"\x00" * 64)

        assert await subject.read_face_image(person.person_id, face_id) is None


class TestDeletion:
    async def test_deleting_a_person_removes_their_vectors_too(self, tmp_path: Path) -> None:
        """A record that removed the name while leaving the vectors would keep exactly
        the part that identifies somebody."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())
        before = (tmp_path / "faces.json").read_text(encoding="utf-8")

        assert await subject.delete_person(person.person_id) is True

        after = (tmp_path / "faces.json").read_text(encoding="utf-8")
        assert str(person.person_id) not in after
        assert len(after) < len(before)
        assert await subject.get_person(person.person_id) is None

    async def test_deleting_someone_who_was_never_there_reports_false(self, tmp_path: Path) -> None:
        """So a caller can tell a completed deletion from a no-op, rather than
        reporting success for a person who never existed."""
        assert await store(tmp_path).delete_person(uuid4()) is False

    async def test_deletion_survives_a_reopen(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())
        await subject.delete_person(person.person_id)

        assert await store(tmp_path).list_people() == ()


class TestSearch:
    async def test_candidates_are_returned_with_scores_not_a_verdict(self, tmp_path: Path) -> None:
        """The threshold is configuration and lives in the policy. A store that
        returned only matches above some internal bar would put the most consequential
        number in the system where nobody reviewing a deployment would look."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, FaceEmbedding.of((1.0, 0.0, 0.0)))

        candidates = await subject.search(
            FaceEmbedding.of((0.0, 1.0, 0.0)), camera_id="cam-1", zone=None, now=0.0
        )
        assert len(candidates) == 1, "a poor match is still returned, with its score"
        assert candidates[0].similarity == pytest.approx(0.0)

    async def test_the_best_of_several_reference_faces_wins(self, tmp_path: Path) -> None:
        """§10: multiple reference images per person. A single face-on office
        photograph matches a corridor camera poorly, and the usual fix is a second
        reference rather than a lower threshold."""
        subject = store(tmp_path)
        person = a_person()
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, FaceEmbedding.of((1.0, 0.0, 0.0)))
        await subject.add_embedding(person.person_id, FaceEmbedding.of((0.0, 1.0, 0.0)))

        candidates = await subject.search(
            FaceEmbedding.of((0.0, 1.0, 0.0)), camera_id="cam-1", zone=None, now=0.0
        )
        assert candidates[0].similarity == pytest.approx(1.0)

    async def test_authorisation_is_resolved_per_camera(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person(camera_ids=frozenset({"cam-1"}))
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())

        here = await subject.search(an_embedding(), camera_id="cam-1", zone=None, now=0.0)
        elsewhere = await subject.search(an_embedding(), camera_id="cam-9", zone=None, now=0.0)
        assert here[0].authorized_here is True
        assert elsewhere[0].authorized_here is False

    async def test_a_disabled_person_is_not_authorised_anywhere(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        person = a_person(status=PersonStatus.DISABLED)
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())

        candidates = await subject.search(an_embedding(), camera_id="cam-1", zone=None, now=0.0)
        assert candidates[0].authorized_here is False

    async def test_an_expired_authorisation_lapses(self, tmp_path: Path) -> None:
        """§10's temporary authorisation, without which every temporary grant becomes a
        permanent one somebody forgot to revoke."""
        subject = store(tmp_path)
        person = a_person(expires_at=100.0)
        await subject.add_person(person)
        await subject.add_embedding(person.person_id, an_embedding())

        before = await subject.search(an_embedding(), camera_id="cam-1", zone=None, now=50.0)
        after = await subject.search(an_embedding(), camera_id="cam-1", zone=None, now=150.0)
        assert before[0].authorized_here is True
        assert after[0].authorized_here is False

    async def test_a_person_with_no_faces_enrolled_is_not_a_candidate(self, tmp_path: Path) -> None:
        subject = store(tmp_path)
        await subject.add_person(a_person())
        assert await subject.search(an_embedding(), camera_id="cam-1", zone=None, now=0.0) == ()

    async def test_a_stale_dimension_is_skipped_rather_than_taking_the_search_down(
        self, tmp_path: Path
    ) -> None:
        """One enrolment from an old embedding model must not stop every other person
        being recognised."""
        subject = store(tmp_path)
        stale = a_person(display_name="Old enrolment")
        current = a_person(display_name="Current")
        await subject.add_person(stale)
        await subject.add_person(current)
        await subject.add_embedding(stale.person_id, FaceEmbedding.of((1.0, 0.0)))
        await subject.add_embedding(current.person_id, FaceEmbedding.of((1.0, 0.0, 0.0)))

        candidates = await subject.search(
            FaceEmbedding.of((1.0, 0.0, 0.0)), camera_id="cam-1", zone=None, now=0.0
        )
        assert [c.display_name for c in candidates] == ["Current"]


class TestFileHandling:
    async def test_an_unreadable_schema_version_is_refused(self, tmp_path: Path) -> None:
        """Refusing to guess at biometric data."""
        path = tmp_path / "faces.json"
        path.write_text(json.dumps({"schema_version": 99, "people": []}), encoding="utf-8")
        with pytest.raises(FaceStoreError, match="schema_version"):
            EncryptedFaceStore(path, encryption_key=KEY)

    async def test_corrupt_json_is_refused_with_the_path(self, tmp_path: Path) -> None:
        path = tmp_path / "faces.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(FaceStoreError, match="not valid JSON"):
            EncryptedFaceStore(path, encryption_key=KEY)

    async def test_a_missing_file_is_an_empty_store_not_an_error(self, tmp_path: Path) -> None:
        """A site that has enrolled nobody yet is the normal starting state."""
        assert await store(tmp_path).list_people() == ()


def test_generated_keys_are_usable_and_distinct() -> None:
    first, second = generate_key(), generate_key()
    assert first != second
    # Round-trips through the loader without raising.
    EncryptedFaceStore(Path("/nonexistent/faces.json"), encryption_key=first)
