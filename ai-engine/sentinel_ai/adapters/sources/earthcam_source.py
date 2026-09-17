"""An EarthCam page URL as a `FrameSource`.

`RtspSource` with one thing changed: what to open. An RTSP camera has a fixed URL that
stays valid; an EarthCam page hands out a signed playlist URL that expires, so this
resolves a fresh one immediately before every connection attempt.

That single difference is why this subclasses rather than copies. The reconnect loop,
the demux fan-out into decoded frames *and* still-encoded packets, the discontinuity
signal that resets the tracker on reconnect, the bounded queues, the shutdown — all of
it is the same problem and all of it is inherited. `RtspSource._open_container` and
`_stream_label` exist as the two seams this needs.

Why re-resolving is the whole recovery strategy
-------------------------------------------------
There is no separate "403 handler" that refreshes a cached token, because nothing is
cached. Every connection begins by fetching the page again, so a signature that expired
mid-stream is repaired by the reconnect the expiry itself triggers. A 403 is still
recognised and logged distinctly — it is the one failure that means "the stream is fine,
your URL is old", and an operator should not have to infer that from a generic ffmpeg
error.

Backoff is inherited too, and matters more here than for a camera on the same LAN: the
thing being retried is somebody else's public website.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import av

from sentinel_ai.adapters.sources.earthcam import (
    EarthCamError,
    EarthCamStream,
    browser_headers,
    redact_url,
    resolve_earthcam,
)
from sentinel_ai.adapters.sources.rtsp import RtspSource

__all__ = ["EarthCamSource"]

logger = logging.getLogger(__name__)

_HTTP_OPEN_TIMEOUT_MICROSECONDS = "10000000"
"""10s for ffmpeg's HTTP layer. Without it a CDN that accepts the connection and then
says nothing holds a worker thread open for as long as it likes."""


class EarthCamSource(RtspSource):
    """A public EarthCam camera, resolved fresh on every connection."""

    def __init__(
        self,
        camera_id: str,
        page_url: str,
        *,
        reconnect_initial_seconds: float,
        reconnect_max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        hwaccel_device: str | None = None,
        max_fps: float = 0.0,
    ) -> None:
        self._page_url = page_url
        self._resolved: EarthCamStream | None = None
        super().__init__(
            camera_id=camera_id,
            # The page URL stands in until the first resolution. `RtspSource` reads
            # `_url` only inside `_open_container` and `_stream_label`, both overridden
            # here, so it is never opened as though it were a stream.
            url=page_url,
            reconnect_initial_seconds=reconnect_initial_seconds,
            reconnect_max_seconds=reconnect_max_seconds,
            clock=clock,
            hwaccel_device=hwaccel_device,
            max_fps=max_fps,
        )

    async def _connect_once(self) -> None:
        """Resolve, then hand the rest to `RtspSource`.

        On the event loop rather than the worker thread, because resolution is async
        HTTP while the pump is blocking PyAV. The resolved stream is stashed for
        `_open_container`, which runs a moment later on the executor.
        """
        try:
            self._resolved = await resolve_earthcam(self._page_url)
        except EarthCamError as error:
            # Raised, not swallowed: `_ReconnectLoop` decides to wait and try again, and
            # it is the only thing that knows how long to wait.
            raise ConnectionError(f"could not resolve {self._page_url}: {error}") from error

        logger.info(
            "resolved EarthCam camera %s (%s) to %s",
            self._camera_id,
            self._resolved.title,
            self._resolved.redacted_url,
        )
        await super()._connect_once()

    def _open_container(self) -> Any:
        resolved = self._resolved
        if resolved is None:  # pragma: no cover - `_connect_once` always sets it first
            raise ConnectionError("EarthCamSource opened before resolving a stream")

        # The same request context the page's own player uses. ffmpeg wants one blob of
        # CRLF-terminated header lines; `User-Agent` goes through its own option because
        # ffmpeg sets a default one that would otherwise be sent twice.
        headers = browser_headers(resolved.page_url)
        user_agent = headers.pop("User-Agent")
        options = {
            "headers": "".join(f"{name}: {value}\r\n" for name, value in headers.items()),
            "user_agent": user_agent,
            "timeout": _HTTP_OPEN_TIMEOUT_MICROSECONDS,
            "follow_redirects": "1",
        }
        try:
            return (
                av.open(resolved.stream_url, options=options, hwaccel=self._hwaccel)
                if self._hwaccel is not None
                else av.open(resolved.stream_url, options=options)
            )
        except av.HTTPForbiddenError as error:
            # The one failure worth naming. The stream is healthy and the signature is
            # stale, which the next reconnect fixes by fetching the page again — so this
            # says so, rather than leaving an operator to read "Server returned 403" and
            # wonder whether the camera has gone private.
            logger.info(
                "EarthCam stream expired; refreshing stream metadata (%s)",
                redact_url(resolved.stream_url),
            )
            self._resolved = None
            raise ConnectionError("EarthCam signature expired") from error

    def _stream_label(self) -> str:
        """Never the signed URL — this reaches a `ConnectionError` message that the
        reconnect loop logs every time the stream ends."""
        return self._page_url

    async def close(self) -> None:
        await super().close()
        self._resolved = None
