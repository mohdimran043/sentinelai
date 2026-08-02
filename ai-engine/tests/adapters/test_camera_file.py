"""`adapters/config/camera_file.py` — the reader's invariants, and the writer's.

The reader's own cases live in `tests/test_main.py`, where they were written and
where `load_cameras` is still imported from; this file is about what changed when
the file stopped being read-only.

Four properties are worth more than the rest, and each of them is a defect this
project has paid for in a neighbouring form:

* **nothing this engine does not understand is lost.** A camera file holds
  comment keys, other cameras, and fields a later version will add. A writer that
  round-trips through the parsed record deletes all of it, and nothing complains.
* **nothing is half-applied.** An edit that would produce an unloadable document
  is refused *before* the write, so a bad edit can never cost a restart.
* **an omitted field and a null field are different instructions.** `zone: null`
  ungroups; an absent `zone` leaves the grouping alone.
* **the write is atomic.** A crash between the two halves leaves the previous
  document intact, never a truncated one.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from sentinel_ai.adapters.config.camera_file import (
    UNSET,
    CameraConfigError,
    CameraEdit,
    CameraFileStore,
    edited_document,
    load_cameras,
)
from sentinel_ai.domain.zone import Zone


def write(path: Path, document: Any) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def a_file(tmp_path: Path, *entries: Any, **extra: Any) -> Path:
    return write(tmp_path / "cameras.json", {"cameras": list(entries), **extra})


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


CAM = {"id": "cam-1", "url": "rtsp://host/one", "label": "Front door", "zone": "corridor"}


class TestTheEditIsPartial:
    async def test_a_label_edit_leaves_the_zone_alone(self, tmp_path: Path) -> None:
        path = a_file(tmp_path, CAM)
        store = CameraFileStore(path)

        record = await store.apply("cam-1", CameraEdit(label="Back door"))

        assert record.label == "Back door"
        assert record.zone is Zone.CORRIDOR
        assert read(path)["cameras"][0]["zone"] == "corridor"

    async def test_a_zone_edit_leaves_the_label_alone(self, tmp_path: Path) -> None:
        path = a_file(tmp_path, CAM)
        record = await CameraFileStore(path).apply("cam-1", CameraEdit(zone=Zone.DAYROOM))

        assert record.label == "Front door"
        assert record.zone is Zone.DAYROOM

    async def test_a_null_zone_ungroups_the_camera(self, tmp_path: Path) -> None:
        """`None` is an instruction, not an absence — and the file records that
        someone chose it rather than dropping the key and making it look like a
        camera nobody got round to."""
        path = a_file(tmp_path, CAM)
        record = await CameraFileStore(path).apply("cam-1", CameraEdit(zone=None))

        assert record.zone is None
        assert read(path)["cameras"][0]["zone"] is None
        assert "zone" in read(path)["cameras"][0]

    async def test_an_omitted_zone_is_not_an_ungrouping(self, tmp_path: Path) -> None:
        """The distinction the `UNSET` sentinel exists for. A console changing only a
        label must not silently ungroup the camera it is renaming — which is exactly
        what a `zone: Zone | None = None` default would do."""
        path = a_file(tmp_path, CAM)
        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert read(path)["cameras"][0]["zone"] == "corridor"

    async def test_both_fields_can_change_in_one_write(self, tmp_path: Path) -> None:
        path = a_file(tmp_path, CAM)
        record = await CameraFileStore(path).apply(
            "cam-1", CameraEdit(label="Wing B door", zone=Zone.ROOM)
        )

        assert (record.label, record.zone) == ("Wing B door", Zone.ROOM)
        stored = read(path)["cameras"][0]
        assert (stored["label"], stored["zone"]) == ("Wing B door", "room")

    async def test_an_edit_that_names_nothing_is_refused(self, tmp_path: Path) -> None:
        """Rewriting the file for a request that changes nothing is churn on the one
        artefact this system cannot afford to churn."""
        path = a_file(tmp_path, CAM)
        before = path.read_text(encoding="utf-8")

        with pytest.raises(CameraConfigError, match="names no fields"):
            await CameraFileStore(path).apply("cam-1", CameraEdit())

        assert path.read_text(encoding="utf-8") == before


class TestNothingElseIsTouched:
    """The defect class this whole module is shaped around: a config writer that
    silently drops a field it did not understand."""

    async def test_keys_this_engine_has_no_model_of_survive(self, tmp_path: Path) -> None:
        """`cameras.example.json` ships `_note` on every camera and `_comment` at the
        top level. Round-tripping through `CameraConfig` would delete both and no
        test that only checked `label` would notice."""
        path = a_file(
            tmp_path,
            {**CAM, "_note": "pushed in with ffmpeg", "future_field": {"nested": [1, 2]}},
            _comment=["read me first"],
        )

        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        document = read(path)
        assert document["_comment"] == ["read me first"]
        assert document["cameras"][0]["_note"] == "pushed in with ffmpeg"
        assert document["cameras"][0]["future_field"] == {"nested": [1, 2]}

    async def test_the_profile_survives_an_edit_untouched(self, tmp_path: Path) -> None:
        """`profile` is restart-only, which must mean "left exactly as it was", not
        "regenerated from defaults" — a profile silently reset to the shipped
        cooldowns would change when every camera escalates."""
        profile = {"cooldown_seconds": 12.5, "summary_interval_seconds": 45.0}
        path = a_file(tmp_path, {**CAM, "profile": profile})

        await CameraFileStore(path).apply("cam-1", CameraEdit(zone=Zone.ROOM))

        assert read(path)["cameras"][0]["profile"] == profile

    async def test_the_url_survives_an_edit_untouched(self, tmp_path: Path) -> None:
        path = a_file(tmp_path, {**CAM, "url": "rtsp://user:secret@host:8554/one"})

        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert read(path)["cameras"][0]["url"] == "rtsp://user:secret@host:8554/one"

    async def test_other_cameras_are_untouched_and_keep_their_order(self, tmp_path: Path) -> None:
        path = a_file(
            tmp_path,
            {"id": "cam-0", "url": "rtsp://host/zero", "label": "Zero"},
            CAM,
            {"id": "cam-2", "url": "rtsp://host/two", "label": "Two", "zone": "room"},
        )

        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed", zone=None))

        cameras = read(path)["cameras"]
        assert [camera["id"] for camera in cameras] == ["cam-0", "cam-1", "cam-2"]
        assert cameras[0] == {"id": "cam-0", "url": "rtsp://host/zero", "label": "Zero"}
        assert cameras[2]["zone"] == "room"

    async def test_a_camera_with_no_label_does_not_gain_one_from_the_default(
        self, tmp_path: Path
    ) -> None:
        """`load_cameras` defaults an absent label to the id. That default belongs to
        the *reader*: writing it into the file would turn "no label configured" into
        "labelled with its own id", and the operator's next hand-edit would be
        arguing with a value they never set."""
        path = a_file(tmp_path, {"id": "cam-1", "url": "rtsp://host/one"})

        record = await CameraFileStore(path).apply("cam-1", CameraEdit(zone=Zone.ROOM))

        assert record.label == "cam-1"
        assert "label" not in read(path)["cameras"][0]


class TestABadEditIsRefusedWhole:
    async def test_an_unknown_camera_is_refused_rather_than_appended(self, tmp_path: Path) -> None:
        """The engine knowing about a camera the file does not means the file has been
        edited since startup. Synthesising an entry would overwrite whoever did that."""
        path = a_file(tmp_path, CAM)
        before = path.read_text(encoding="utf-8")

        with pytest.raises(CameraConfigError, match="no camera 'cam-9'"):
            await CameraFileStore(path).apply("cam-9", CameraEdit(label="New"))

        assert path.read_text(encoding="utf-8") == before

    async def test_an_empty_label_is_refused_and_nothing_is_written(self, tmp_path: Path) -> None:
        """The re-parse before the write is what catches this. Without it the file
        would hold `"label": ""`, load_cameras would reject it, and the engine would
        refuse to start — a console edit that bricks the next restart."""
        path = a_file(tmp_path, CAM)
        before = path.read_text(encoding="utf-8")

        with pytest.raises(CameraConfigError, match="non-empty"):
            await CameraFileStore(path).apply("cam-1", CameraEdit(label=""))

        assert path.read_text(encoding="utf-8") == before

    async def test_an_edit_to_a_file_that_is_already_invalid_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Not "fixed silently": the duplicate id below is a real problem someone
        introduced by hand, and writing over it would hide it and lose a camera."""
        path = a_file(tmp_path, CAM, {**CAM, "label": "Duplicate"})

        with pytest.raises(CameraConfigError, match="duplicate camera id"):
            await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

    async def test_a_file_that_is_no_longer_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "cameras.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(CameraConfigError, match="not valid JSON"):
            await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

    async def test_a_missing_file_is_refused_rather_than_created(self, tmp_path: Path) -> None:
        path = tmp_path / "cameras.json"

        with pytest.raises(CameraConfigError, match="no camera configuration"):
            await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert not path.exists()

    async def test_what_is_written_always_loads(self, tmp_path: Path) -> None:
        """The property the re-parse buys, stated directly: whatever this store
        writes, the shipped reader accepts. A writer that can produce a file its own
        reader rejects turns an edit into a failed restart hours later."""
        path = a_file(
            tmp_path,
            CAM,
            {"id": "cam-2", "url": "./clip.mp4", "profile": {"cooldown_seconds": 2.0}},
        )

        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Ünïcode ✓", zone=None))
        await CameraFileStore(path).apply("cam-2", CameraEdit(zone=Zone.DAYROOM))

        cameras = load_cameras(path)
        assert [(c.camera_id, c.label, c.zone) for c in cameras] == [
            ("cam-1", "Ünïcode ✓", None),
            ("cam-2", "cam-2", Zone.DAYROOM),
        ]
        assert cameras[1].profile.cooldown_seconds == 2.0


class TestTheWriteIsAtomic:
    async def test_the_temporary_file_is_gone_afterwards(self, tmp_path: Path) -> None:
        """A leftover `.tmp` sibling would be a second, stale copy of the camera
        record sitting next to the real one."""
        path = a_file(tmp_path, CAM)
        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["cameras.json"]

    async def test_a_failure_during_the_write_leaves_the_previous_document_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rename is last, so a write that dies part-way leaves the old file
        whole. Simulated by making the rename itself fail, which is the last moment
        at which anything can go wrong."""
        path = a_file(tmp_path, CAM)
        before = path.read_text(encoding="utf-8")

        def explode(self: Path, target: Any) -> Path:
            raise OSError("no space left on device")

        monkeypatch.setattr(Path, "replace", explode)

        with pytest.raises(OSError, match="no space left"):
            await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert path.read_text(encoding="utf-8") == before
        # The stated invariant of the test above ("the temporary file is gone
        # afterwards") has to hold here too, on the one path where the temp file
        # actually holds a document nobody else will ever clean up: it carries the
        # full camera record, RTSP credentials included, and a stray `.tmp` next to
        # `cameras.json` is exactly the kind of file a careless `git add -A` picks up.
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ["cameras.json"]

    async def test_the_temp_file_is_never_group_or_world_readable_while_it_holds_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The write used to happen in two visible steps: the temp file was created,
        the full document — RTSP credentials included — was written into it and
        `fsync`ed (the slowest step in `_write`), and only *then* was its mode
        narrowed to match `cameras.json`. On a host with an ordinary umask that left
        `cameras.json.tmp` sitting group- or world-readable, holding every camera's
        `rtsp://user:pass@host/stream`, for as long as the fsync took. The temp
        filename is fixed, so a local user watching the directory with inotify could
        read it on every edit — and the edit endpoint has no authentication, so they
        can trigger one on demand.

        Reproduced by monkeypatching `os.chmod`: `_write` calls it exactly once per
        write (to land the file at its final, preserved mode), and the instant just
        *before* that call is the worst case the old code ever produced — the file
        fully written and fsynced, and, pre-fix, still sitting at whatever the umask
        left it at. Capturing the file's mode and content at that instant captures
        the exposure directly rather than inferring it from the end state, which was
        always correct even when the window in the middle was not.
        """
        path = a_file(tmp_path, {**CAM, "url": "rtsp://user:s3cret@host/one"})
        path.chmod(0o600)
        old_umask = os.umask(0o022)
        captured: dict[str, Any] = {}
        original_chmod = os.chmod

        def spy_chmod(target: Any, mode: int, *args: Any, **kwargs: Any) -> None:
            if not captured:
                captured["mode"] = stat.S_IMODE(os.stat(target).st_mode)
                captured["content"] = Path(target).read_text(encoding="utf-8")
            return original_chmod(target, mode, *args, **kwargs)

        monkeypatch.setattr(os, "chmod", spy_chmod)
        try:
            await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))
        finally:
            os.umask(old_umask)

        assert captured, "os.chmod was never called — this test needs a previous file to exist"
        assert "user:s3cret" in captured["content"]
        assert captured["mode"] & (stat.S_IRWXG | stat.S_IRWXO) == 0, (
            f"temp file was {oct(captured['mode'])} while it held credentials — "
            f"group or other could read it"
        )

    async def test_the_files_permissions_are_preserved_across_a_write(self, tmp_path: Path) -> None:
        """`cameras.json` holds RTSP credentials, so its mode is part of what
        protects them. `Path.open("w")` on a *new* file — which the write-then-rename
        temp file always is — gets `0o666 & ~umask`, and the rename carries that mode
        onto `cameras.json` in place of whatever an operator had set. An edit made
        through the console must not be the thing that quietly widens a `0600` file
        to group- or world-readable."""
        path = a_file(tmp_path, CAM)
        path.chmod(0o600)

        await CameraFileStore(path).apply("cam-1", CameraEdit(label="Renamed"))

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    async def test_concurrent_edits_to_different_fields_do_not_lose_one(
        self, tmp_path: Path
    ) -> None:
        """Two operators, two windows, one file. Without the lock both would read the
        pre-edit document and the second rename would silently discard the first
        edit — the classic lost update, and invisible to whoever made it."""
        path = a_file(tmp_path, CAM)
        store = CameraFileStore(path)

        await asyncio.gather(
            store.apply("cam-1", CameraEdit(label="Renamed")),
            store.apply("cam-1", CameraEdit(zone=Zone.ROOM)),
        )

        stored = read(path)["cameras"][0]
        assert stored["label"] == "Renamed"
        assert stored["zone"] == "room"

    async def test_a_second_edit_waits_for_the_first_to_finish(self, tmp_path: Path) -> None:
        """The mutual exclusion itself, asserted directly.

        The test above cannot see it: `apply` has no await inside the lock, so two
        `gather`ed calls run to completion one after the other on this loop whether
        or not a lock is held — a store with the lock deleted passes it. Holding the
        lock from outside and watching a second `apply` fail to progress is the only
        way to tell "serialised" from "happens not to interleave today", and today's
        arrangement is not a property anyone promised to keep.
        """
        path = a_file(tmp_path, CAM)
        store = CameraFileStore(path)

        async with store._lock:
            blocked = asyncio.create_task(store.apply("cam-1", CameraEdit(label="Renamed")))
            await asyncio.sleep(0)
            assert not blocked.done(), "a second edit ran while another held the file"
            assert read(path)["cameras"][0]["label"] == "Front door"

        await blocked
        assert read(path)["cameras"][0]["label"] == "Renamed"


class TestEditedDocumentIsPure:
    """`edited_document` is the part with no I/O in it, so it is the part that can
    be checked for not mutating its input."""

    def test_the_original_document_is_not_mutated(self) -> None:
        document = {"cameras": [dict(CAM)]}
        edited_document(document, "cam-1", CameraEdit(label="Renamed"), "origin")

        assert document["cameras"][0]["label"] == "Front door"

    def test_an_unset_field_is_not_written_as_the_sentinel(self) -> None:
        """A sentinel that leaks into the document would serialise as something no
        reader understands — the failure mode a bare `object()` sentinel invites."""
        result = edited_document({"cameras": [dict(CAM)]}, "cam-1", CameraEdit(zone=None), "o")

        assert result["cameras"][0]["label"] == "Front door"
        assert UNSET not in result["cameras"][0].values()
