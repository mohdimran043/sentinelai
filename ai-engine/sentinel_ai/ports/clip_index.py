"""What clips exist, and what they are costing.

Separate from `ClipReader`, which answers "give me these bytes". This answers "what is
there" — the two have different shapes (an inventory versus a blob), different failure
modes (a slow list versus a missing object) and different consumers (a browsing screen
versus a player). One adapter implements both; keeping the ports apart is what stops a
screen that only needs a list from being handed something that can stream evidence.

Nothing here is a durable index. It is a listing of the object store as it is right
now, so retention has already removed whatever it has removed — which is the honest
answer for a footage browser, and is why neither type carries a "deleted" state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID

__all__ = ["CameraStorage", "ClipIndex", "ClipRecord", "StorageUsage"]


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """One recorded clip, as the object store holds it."""

    event_id: UUID
    """The event this clip was recorded for. The clip's identity, not a separate one —
    `MinioClipWriter` names every object after its event, which is what lets a clip be
    found again from an event or an alert without an index."""

    size_bytes: int
    modified_at: float
    """Unix epoch seconds, from the object store's own record of when the upload
    finished. **Not when the incident happened** — it is later by the post-roll plus
    the upload, and on a busy engine that gap is seconds. Close enough to sort by,
    wrong to present as the time of the event."""

    has_short_copy: bool
    """Whether a `SENTINEL_NOTIFY_CLIP_SECONDS` trim was written beside it."""


@dataclass(frozen=True, slots=True)
class CameraStorage:
    camera_id: str
    clips: int
    bytes_used: int


@dataclass(frozen=True, slots=True)
class StorageUsage:
    """What the clip bucket is holding, totalled and broken down by camera."""

    bucket: str
    clips: int
    """Recordings, counting a clip and its short copy as one. An operator asking how
    many clips there are means incidents, not objects."""

    objects: int
    """Objects, which is the number the store bills for and roughly twice `clips`."""

    bytes_used: int
    retention_days: int
    """`0` means the lifecycle rule is off and nothing expires on its own."""

    per_camera: tuple[CameraStorage, ...]
    reachable: bool = True
    """`False` when the object store could not be listed. Every count is then zero and
    means nothing — a storage screen must say "could not read this" rather than draw an
    empty bucket, which is the one reading that would let an operator believe their
    evidence had been deleted."""


class ClipIndex(ABC):
    @abstractmethod
    async def usage(self) -> StorageUsage:
        """What the bucket holds. Never raises — an unreachable store comes back as
        `reachable=False`, because "storage is down" is a thing this screen exists to
        show and not a reason for it to fail."""

    @abstractmethod
    def clip_uri(self, camera_id: str, event_id: UUID, *, short: bool) -> str:
        """Where a given event's clip lives, without asking the store.

        Here rather than in the caller because the object naming is the implementation's
        business. A route that wants to play a clip names a camera and an event; only
        the adapter turns that into a path, which is also what keeps a caller from
        assembling one out of strings it was handed.

        Says where it *would* be, not that anything is there — retention may have taken
        it. `ClipReader.read` answers the second question.
        """

    @abstractmethod
    async def list_clips(self, camera_id: str, *, limit: int) -> tuple[ClipRecord, ...]:
        """One camera's clips, newest first, bounded.

        Bounded because a camera left running for a week holds thousands and a browser
        renders the first screenful; `limit` is the promise that this never reads more
        than it was asked for.
        """
