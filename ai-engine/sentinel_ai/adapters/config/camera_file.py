"""The camera record on disk: read at startup, and edited at runtime.

This module used to live inside `sentinel_ai/main.py`, which is still the only
place that *composes* anything from it — `main` re-exports `CameraConfig`,
`CameraConfigError` and `load_cameras` unchanged, so nothing that imported them
from there has to move. It is a module of its own now because the file stopped
being read-only: `CameraFileStore` writes it back, and a writer belongs beside
the reader whose invariants it has to keep, not in a composition root.

The document shape is unchanged and documented in `main`'s own docstring and in
`cameras.example.json`.

What may be edited, and what may not
------------------------------------
`label` and `zone` only. Both are metadata: nothing in the pipeline branches on
either (`CameraRunner` carries them so `telemetry()` and the assembled event can
report them), so changing one is an attribute write on a live object and the next
frame is unaffected.

`url` and `profile` are deliberately **not** editable through this store.

* `url` names the source. Changing it means tearing down a running
  `CameraRunner` — its producer task, its `_LatestSlot`, its pre-roll buffer, any
  clip mid-recording — and building a new `FrameSource` in its place. That is a
  camera restart wearing an edit's clothes, and doing it under an HTTP request
  would drop frames and possibly an in-flight escalation. A second reason is
  narrower and just as decisive: an RTSP URL routinely carries credentials
  (`rtsp://user:pass@host/stream`), and this engine has no authentication, so the
  URL is never put on the wire in either direction. Edit `cameras.json` and
  restart.
* `profile` is the escalation policy — cooldowns, thresholds, the token bucket
  the gate is mid-way through spending. Swapping it under a running gate would
  reset or contradict `GateState` in ways that have no obvious right answer.
  Restart.

Neither is silently dropped: `api.schemas.CameraEdit` forbids unknown fields, so
a request carrying `url` is rejected with a message naming the restart, and this
store never sees it.

Atomicity
---------
Write-then-rename, the same technique `RabbitMQPublisher._spool` uses: the new
document goes to a `.tmp` sibling and is moved into place with `Path.replace`,
an atomic rename on the same filesystem. A reader therefore sees the old
document or the new one and never a half-written one.

Two things are added on top of the spool's version, because the stakes differ. A
spool file is a duplicate of an event the process still holds, so losing one to a
crash mid-write costs a retry; `cameras.json` is the only copy of the camera
record, so the temporary file is `fsync`ed before the rename and the directory is
`fsync`ed after it. And the document is **re-parsed in full before it is
written** — see `CameraFileStore.apply`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final, Literal

from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.zone import Zone

logger = logging.getLogger(__name__)

__all__ = [
    "EDITABLE_FIELDS",
    "RESTART_REQUIRED_FIELDS",
    "UNSET",
    "CameraConfig",
    "CameraConfigError",
    "CameraEdit",
    "CameraFileStore",
    "Unset",
    "edited_document",
    "load_cameras",
    "parse_cameras",
]

_PROFILE_FIELDS = frozenset(field.name for field in fields(CameraProfile)) - {"camera_id"}

EDITABLE_FIELDS: Final = ("label", "zone")
"""The fields `CameraFileStore.apply` will change. Published so the API contract
and this module cannot drift about which those are."""

RESTART_REQUIRED_FIELDS: Final = ("url", "profile")
"""Stored in the file, honoured at startup, and not editable at runtime. See this
module's docstring for why each one is here."""


class CameraConfigError(ValueError):
    """The camera file is missing, unparseable, or describes something unbuildable.

    Fail-loud at startup on purpose: every alternative (skip the bad camera, fall
    back to defaults) produces a process that runs while silently watching fewer
    cameras than the operator configured.

    The same type is raised by `CameraFileStore.apply`, and for the same reason —
    an edit that would produce a file the loader rejects is refused whole rather
    than written and discovered at the next restart.
    """


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    label: str
    url: str
    profile: CameraProfile
    zone: Zone | None = None
    """Which space this camera watches, or None when nobody has grouped it.

    Optional, and its absence is *not* a configuration error: every camera file
    written before zones existed keeps loading, and an ungrouped camera is a true
    statement about a deployment rather than a broken one. A zone that is *present and
    unknown* is a different thing — see `_zone_from`."""


class _Unset(Enum):
    """Single-member enum rather than `object()`: mypy narrows `X | Unset` on an
    `is UNSET` test only for a literal type, and every edit here has to keep
    "the caller did not mention this field" apart from "the caller set it to
    null"."""

    TOKEN = auto()


UNSET: Final = _Unset.TOKEN
Unset = Literal[_Unset.TOKEN]


@dataclass(frozen=True, slots=True)
class CameraEdit:
    """A partial change to one camera's record.

    `zone` is the field that forces the sentinel: `None` is a real, requestable
    value meaning "ungroup this camera", so a plain `Zone | None = None` default
    could not tell an ungrouping from a request that never mentioned the zone at
    all — and silently re-grouping (or silently ungrouping) a camera an operator
    did not ask about is precisely the kind of half-applied edit this whole path
    exists to make impossible. `label` uses the same sentinel for symmetry.
    """

    label: str | Unset = UNSET
    zone: Zone | Unset | None = UNSET

    @property
    def is_empty(self) -> bool:
        """True when nothing was asked for. The API rejects this rather than
        reporting a successful write that changed nothing."""
        return self.label is UNSET and self.zone is UNSET


# -- reading ------------------------------------------------------------------------


def _profile_from(camera_id: str, raw: Mapping[str, Any]) -> CameraProfile:
    unknown = sorted(set(raw) - _PROFILE_FIELDS)
    if unknown:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown profile field(s) {unknown}; "
            f"valid fields are {sorted(_PROFILE_FIELDS)}"
        )
    overrides = dict(raw)
    if "salient_classes" in overrides:
        overrides["salient_classes"] = frozenset(overrides["salient_classes"])
    try:
        return CameraProfile(camera_id=camera_id, **overrides)
    except (TypeError, ValueError) as exc:
        # CameraProfile.__post_init__ enforces its own invariants; surfacing them as
        # a CameraConfigError keeps every startup configuration failure one type.
        raise CameraConfigError(f"camera {camera_id!r}: invalid profile: {exc}") from exc


def _zone_from(camera_id: str, raw: Any) -> Zone | None:
    """Absent means ungrouped; present-but-unknown means the file is wrong.

    Keeping those two apart is the whole contract of this field. Silently downgrading
    `"hallway"` to ungrouped would produce a console that looks right and quietly
    leaves a camera out of the group the operator put it in — the same class of
    failure `_profile_from`'s unknown-field check exists to prevent — while treating
    the absent case as an error would make every pre-zone camera file unloadable for
    no gain.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'zone' must be a string or absent, got {type(raw).__name__}"
        )
    try:
        return Zone(raw)
    except ValueError:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown zone {raw!r}; "
            f"valid zones are {[zone.value for zone in Zone]}, or omit the field to "
            f"leave the camera ungrouped"
        ) from None


def parse_cameras(document: Any, origin: str) -> tuple[CameraConfig, ...]:
    """Validate an already-decoded camera document.

    Split out of `load_cameras` so that `CameraFileStore.apply` can run the exact
    same validation over the document it is *about to* write. Two parsers would be
    two vocabularies of what a valid camera file is, and the writer's would be the
    one nobody tested against a real startup.

    `origin` only names the thing being validated in error messages — a path when
    a file is being read, the same path when a file is about to be written.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries: list[Any] = document["cameras"]
    if not entries:
        # An engine with no cameras starts, serves /health, and watches nothing. That
        # is never what an operator meant.
        raise CameraConfigError(f"{origin} configures no cameras")

    configs: list[CameraConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CameraConfigError(f"{origin}: cameras[{index}] must be an object")
        camera_id = entry.get("id")
        url = entry.get("url")
        if not isinstance(camera_id, str) or not camera_id:
            raise CameraConfigError(f"{origin}: cameras[{index}] needs a non-empty string 'id'")
        if not isinstance(url, str) or not url:
            raise CameraConfigError(
                f"{origin}: camera {camera_id!r} needs a non-empty string 'url'"
            )
        if camera_id in seen:
            # Duplicates are silently survivable — `EngineService` keys cameras by id,
            # so the second would simply replace the first and one camera would never
            # be watched.
            raise CameraConfigError(f"{origin}: duplicate camera id {camera_id!r}")
        seen.add(camera_id)

        label = entry.get("label", camera_id)
        if not isinstance(label, str) or not label:
            raise CameraConfigError(
                f"{origin}: camera {camera_id!r} 'label' must be a non-empty str"
            )
        raw_profile = entry.get("profile", {})
        if not isinstance(raw_profile, dict):
            raise CameraConfigError(f"{origin}: camera {camera_id!r} 'profile' must be an object")

        configs.append(
            CameraConfig(
                camera_id=camera_id,
                label=label,
                url=url,
                profile=_profile_from(camera_id, raw_profile),
                zone=_zone_from(camera_id, entry.get("zone")),
            )
        )
    return tuple(configs)


def _read_document(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CameraConfigError(
            f"no camera configuration at {path} — copy cameras.example.json, or point "
            f"SENTINEL_CAMERAS_FILE somewhere else"
        ) from None
    except json.JSONDecodeError as exc:
        raise CameraConfigError(f"{path} is not valid JSON: {exc}") from exc


def load_cameras(path: Path) -> tuple[CameraConfig, ...]:
    """Read and validate the camera list. See `sentinel_ai.main` for the shape."""
    return parse_cameras(_read_document(path), origin=str(path))


# -- writing ------------------------------------------------------------------------


def edited_document(document: Any, camera_id: str, edit: CameraEdit, origin: str) -> Any:
    """The whole document, with only this camera's edited keys changed.

    Pure, and takes the *whole* document rather than the one entry, because the
    property that matters is about everything it does not touch. The file an
    operator hand-wrote holds things this engine has no model of — the
    `_comment` and `_note` keys `cameras.example.json` ships, whatever a future
    field turns out to be, and every other camera in the list. Round-tripping
    through `CameraConfig` and re-serialising the parsed shape would silently
    delete all of it, which is the specific defect this function is written to
    avoid: a config writer that drops a field it did not understand.

    So the original mappings are copied and only the named keys are replaced.
    Ordering survives too (`dict` preserves insertion order and `cameras` is
    already present, so the spread below leaves it where it was).

    An unknown `camera_id` raises rather than appending: the engine knowing about
    a camera the file does not means the file has been edited since startup, and
    synthesising an entry from memory would overwrite whoever did that.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries: list[Any] = document["cameras"]
    updated: list[Any] = []
    found = False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") != camera_id:
            updated.append(entry)
            continue
        found = True
        changed = dict(entry)
        if edit.label is not UNSET:
            changed["label"] = edit.label
        if edit.zone is not UNSET:
            # An explicit `null` rather than deleting the key: `parse_cameras`
            # already reads null as ungrouped, and leaving the field visible
            # records that someone chose this, where a deleted key looks like a
            # camera nobody has got round to grouping yet.
            changed["zone"] = None if edit.zone is None else edit.zone.value
        updated.append(changed)

    if not found:
        raise CameraConfigError(
            f"{origin} has no camera {camera_id!r} — the file has been edited since the "
            f"engine started, so this edit was not applied; reconcile the file and "
            f"restart rather than letting the console overwrite it"
        )
    return {**document, "cameras": updated}


class CameraFileStore:
    """Read-modify-write over `cameras.json`, one edit at a time.

    Serialised by an `asyncio.Lock` because the sequence is a read, a decision and
    a write: two concurrent `PATCH`es without it would both read the pre-edit
    document and the second rename would drop the first operator's change. The
    file I/O inside the lock is synchronous rather than pushed to a thread — the
    document is a kilobyte and this is not a hot path, and a thread would buy an
    interleaving hazard in exchange for nothing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def apply(self, camera_id: str, edit: CameraEdit) -> CameraConfig:
        """Persist `edit` and return the camera's record as the file now holds it.

        The order is the contract:

          1. read and decode the file as it is *now*, not as it was at startup;
          2. build the edited document, preserving everything not named;
          3. **re-parse the whole document** — if the result would not load, raise
             and write nothing. Nothing is half-applied, and it is impossible for
             this path to leave behind a file that fails the next startup;
          4. only then write, atomically.

        The returned record is read back out of the re-parsed document rather than
        assembled from the request, so a caller that then applies it to a running
        camera is applying what is on disk. Memory and file cannot disagree about
        an edit that succeeded.

        Raises `CameraConfigError` for anything about the file or the edit, and
        `OSError` if the write itself fails — in which case, because the write is
        last and is a rename, the previous document is still intact.
        """
        if edit.is_empty:
            # Cheap, but it is the store's own invariant and not only the API's:
            # rewriting the file for an edit that names nothing is churn on the one
            # artefact this system cannot afford to churn, and any future caller
            # that reaches here directly deserves the same answer the API gives.
            raise CameraConfigError(
                f"camera {camera_id!r}: the edit names no fields; editable fields are "
                f"{list(EDITABLE_FIELDS)}"
            )
        async with self._lock:
            origin = str(self._path)
            document = _read_document(self._path)
            candidate = edited_document(document, camera_id, edit, origin)
            cameras = parse_cameras(candidate, origin)
            record = next(camera for camera in cameras if camera.camera_id == camera_id)
            self._write(candidate)
            return record

    def _write(self, document: Any) -> None:
        """Write-then-rename, `fsync`ed at both ends.

        `RabbitMQPublisher._spool` does the rename without the syncs, which is the
        right trade there: a spool file duplicates an event the process still
        holds, so a crash that loses one costs a replay. This file is the only copy
        of the camera record, so the cost of a rename that lands ahead of the data
        is an engine that will not start.

        Three things this does that a plain write-then-rename would not:

        * **the temp file is created owner-only, and only ever widened after it
          already holds nothing readable.** `Path.open("w")` on a *new* file gets
          `0o666 & ~umask` — group- or world-readable by default on an ordinary
          host — and the file holds the *complete* document, RTSP credentials
          included, from the first byte written until the trailing `fsync` below
          returns, which is the slowest step in this whole function. Creating the
          file with that default and narrowing it with `chmod` afterwards — the
          previous shape of this code — leaves exactly that window open: a fixed,
          predictable filename, readable by anyone in the file's group or (with a
          permissive umask) anyone on the host, for as long as the write and fsync
          take. So the inode is created with `os.open(tmp_path, ..., 0o600)`
          instead: `os.open`'s mode argument is itself masked by umask same as
          `open(2)`'s always is, but umask can only *clear* bits, never add them, so
          asking for `0o600` guarantees the file is owner-only from the instant it
          exists no matter what the host's umask happens to be — there is no
          window to catch it in, because it is never open. Only once the body is
          written and fsynced is the mode changed — to the previous file's mode if
          one existed (this is the read-modify-write path, so almost always), or
          left at `0o600` on the very first write, which is the correct resting
          mode for a file nothing has told to be more open than that and strictly
          narrower than the historical `0o666 & ~umask` default.
        * **the mode (and, where possible, the owner) of the old file survive the
          rename.** Because the rename replaces the old file with the new inode,
          whatever mode the temp file has when it lands becomes `cameras.json`'s
          mode permanently and silently. So the old file's mode is read *before*
          anything is written and stamped onto the temp file *before* the rename,
          the same way `install(1)` and every other tool that has to replace a
          sensitive file in place does it. Ownership is restored on a best-effort
          basis: an unprivileged process (the common case — the engine does not run
          as root) cannot `chown` at all, and failing the edit over that would make
          every edit fail on exactly the deployments most likely to run this way —
          logged at debug when it fails, because a process that runs as a different
          user than the one that owns `cameras.json` will fail this silently on
          *every* edit, quietly handing the file's ownership to itself one rename
          at a time, and that is worth being able to diagnose.
        * **a temp file never survives a failed write.** Anything that goes wrong
          between creating the temp file and completing the rename — a full disk on
          write, a rename that fails because the target is on another filesystem —
          is re-raised after the temp file is removed, not before. Without that, the
          failure path leaves `cameras.json.tmp` behind holding a full copy of the
          document, credentials included, next to a `.json` ignore rule that does
          not cover `.json.tmp` — a second, unprotected copy of the same secret. The
          cleanup unlink is itself best-effort: if it fails (a read-only directory,
          an `EIO`), that failure is swallowed rather than raised in place of the
          real one — an operator staring at an `ENOSPC` from the write needs to see
          `ENOSPC`, not an unrelated `PermissionError` from the cleanup that ran
          while handling it.
        """
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        try:
            previous = self._path.stat()
        except FileNotFoundError:
            # Nothing to preserve: this is the never-existed-before-now case, and
            # `_read_document` above would already have refused an `apply` against a
            # missing file, so in practice this only happens if the file is removed
            # out from under a caller that reaches `_write` directly.
            previous = None
        try:
            body = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            # See the docstring above: created owner-only via `os.open` with an
            # explicit mode, not `Path.open("w")` followed by a chmod, so the
            # credential-bearing content is never sitting at the umask default.
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            mode = stat.S_IMODE(previous.st_mode) if previous is not None else 0o600
            os.chmod(tmp_path, mode)
            if previous is not None:
                # Best-effort: an unprivileged process ordinarily cannot chown to an
                # arbitrary owner, and that must not turn a metadata edit into a
                # failed one — the mode, restored above, is the part that actually
                # gates who can read the credentials.
                try:
                    os.chown(tmp_path, previous.st_uid, previous.st_gid)
                except OSError:
                    logger.debug(
                        "%s: could not restore owner uid=%s gid=%s on the "
                        "replacement file; it will be owned by whichever user this "
                        "process runs as instead, which can lock the original "
                        "owner out of their own cameras.json",
                        tmp_path,
                        previous.st_uid,
                        previous.st_gid,
                        exc_info=True,
                    )
            tmp_path.replace(self._path)  # atomic rename on the same filesystem
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise

        # The rename above already landed: the document on disk is the new one, and
        # a reader opening `cameras.json` right now sees it. What is not guaranteed
        # on every filesystem without an fsync of the directory *entry* is that it
        # survives a crash the instant after — a durability guarantee about the
        # rename's visibility, not about the data. Everything below, including
        # opening the directory in the first place, is folded into one
        # `except OSError`: `os.open` here can fail too (EMFILE/ENFILE under fd
        # pressure is the realistic case, and it runs after the rename just the same
        # as the fsync does), and a failure to open is no less "already landed" than
        # a failure to fsync. Raising in either case would tell the caller the edit
        # failed when it did not: `update_camera` would refuse to apply it to the
        # running camera, the API would answer 500, and the next restart would
        # silently pick up the very change the operator was told never happened —
        # the file and memory disagreeing being exactly what `docs/operations.md`
        # promises cannot happen. So every failure here is logged, not raised —
        # including `os.close` in the `finally`, which can itself raise `EIO` on
        # some filesystems even after the fd was opened and fsynced successfully.
        try:
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            logger.warning(
                "%s: directory entry fsync failed after cameras.json was written; "
                "the edit is committed to disk, but its durability against a crash "
                "right now is not guaranteed",
                self._path.parent,
                exc_info=True,
            )
