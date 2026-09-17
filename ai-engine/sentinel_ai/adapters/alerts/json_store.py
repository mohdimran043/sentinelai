"""A JSON file holding operator triage state (spec §17).

A file, for `ADR 9`'s reason turned the other way round: that decision says a file is
right for a small set that a human edits and environment variables are right for
scalars. This set is small but *not* human-edited — so it gets a file for a third
reason, which is that it is the only durable store this engine owns. Postgres arrives
with Phase 1C and will replace this adapter and nothing else, which is the point of the
port being there.

Written the same way `EncryptedFaceStore` writes: serialise, `fsync`, rename over the
target. A rename within a directory is atomic on POSIX, so a reader — this engine on
its next start — sees either the previous complete set or the new complete set, never
a half-written one. The failure that matters here is not corruption in the abstract; it
is an engine that will not start because its alert file was truncated by a power cut,
and the atomic swap is what makes that unreachable.

No encryption, unlike the face store, and the difference is worth stating. An alert
holds a camera id, a description the VLM wrote, and a name somebody typed into an
acknowledge box. That is operational data about a site. A face embedding is biometric
data about a person. Treating them the same would either leave biometrics
under-protected or imply a key-management obligation on a file that does not need one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from sentinel_ai.adapters.serialization.alert_codec import decode_register, encode_register
from sentinel_ai.domain.alert import Alert
from sentinel_ai.ports.alert_store import AlertStore

logger = logging.getLogger(__name__)

__all__ = ["JsonAlertStore"]


class JsonAlertStore(AlertStore):
    def __init__(self, path: Path) -> None:
        self._path = path
        # One writer at a time. Two coalesced saves racing would interleave their
        # temp-file creation harmlessly but could rename out of order, leaving the
        # older set on disk — which is exactly the lost acknowledgement this class
        # exists to prevent.
        self._lock = asyncio.Lock()

    async def load(self) -> tuple[Alert, ...]:
        """Never raises. See `AlertStore.load` on why that is the contract."""
        try:
            raw = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return ()
        except OSError as error:
            logger.warning("could not read the alert store at %s: %s", self._path, error)
            return ()
        try:
            return decode_register(json.loads(raw))
        except (ValueError, KeyError, TypeError) as error:
            # Kept, not deleted. The next save overwrites it anyway, and a file that
            # would not parse is the one artefact somebody debugging this needs.
            logger.warning(
                "the alert store at %s could not be read and no triage state was "
                "restored; every alert it held will show as unseen: %s",
                self._path,
                error,
            )
            return ()

    async def save(self, alerts: Sequence[Alert]) -> None:
        document = encode_register(alerts)
        async with self._lock:
            await asyncio.to_thread(self._write, document)

    def _write(self, document: object) -> None:
        """Serialise, fsync, rename. `EncryptedFaceStore._persist`'s shape, minus the
        0600 mode — see the module docstring on why these two files differ there."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, ensure_ascii=False)
                handle.flush()
                # The rename is atomic, but only relative to data the kernel already
                # has. An fsync-less rename can leave a correctly-named *empty* file
                # after a power cut, which is the one outcome this path exists to
                # avoid — and an empty alert store reads as "nobody has seen anything".
                os.fsync(handle.fileno())
            temporary.replace(self._path)
        except BaseException:
            # A half-written temp file left behind would be picked up by nothing, but
            # it would also never be cleaned up, and a directory slowly filling with
            # them is how a disk-full error becomes somebody else's problem.
            temporary.unlink(missing_ok=True)
            raise

    async def close(self) -> None:
        """Nothing to release: the file is opened and closed per write."""
