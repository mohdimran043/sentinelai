"""Reading a clip back out (spec §23, §26).

The counterpart to `ClipWriter`, and a separate port rather than a method on it because
the two have different lifetimes and different callers. A writer is held by the pipeline
and used once per event, as packets arrive. A reader is held by the API and used when an
operator opens a row, which may be days later and is not on any pipeline's path.

Why this exists at all
----------------------
`ClipHandle.finish()` returns a URI, and until now nothing could open one. `s3://` is
not a scheme a browser can follow, so a clip URI on an alert was a string an operator
could read and not a recording they could watch. This is the seam that turns one into
the other.

What an implementation must refuse
----------------------------------
A URI reaches `read()` having come from an alert this engine wrote, never from a
request. That is the design, and it is still not enough on its own: the day somebody
adds an endpoint that takes a URI from a caller, this port is the last thing standing
between that and the object store. So an implementation **must** verify that the URI
names its own bucket, and reject anything else, rather than trusting its caller to have
checked. See `MinioClipReader.read`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["ClipReader"]


class ClipReader(ABC):
    @abstractmethod
    async def read(self, uri: str) -> bytes | None:
        """The bytes of the clip at `uri`, or `None` if there is no such clip.

        `None` rather than an exception for a missing object, because missing is
        ordinary: retention deletes clips on a schedule, so an alert older than the
        retention window names a recording that is legitimately gone. An operator
        should be told the clip has expired, which is a 404, not that the engine
        failed, which is a 500.

        Raises only when the object store itself could not be reached. Whole bytes
        rather than a stream: these are seconds of video, a few hundred kilobytes, and
        a streaming variant would buy nothing but a longer-lived connection.
        """
