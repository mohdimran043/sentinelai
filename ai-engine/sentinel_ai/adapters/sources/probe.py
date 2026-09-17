"""Look at a stream before committing to it (the console's "preview").

Adding a camera whose URL is wrong is easy and the failure is quiet: the camera appears
in the list, the runner retries forever behind exponential backoff, and the only sign is
`frames_seen` that never moves. This opens the stream once, decodes one frame, and
reports what it found — so an operator sees the picture before saving rather than a row
that may or may not be a camera.

Deliberately not a `FrameSource`. A probe is a single question answered once and closed:
no reconnect loop, no packet fan-out, no pre-roll. Reusing the source machinery would
mean starting a `CameraRunner` for a camera nobody has agreed to add yet.
"""

from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Any

from sentinel_ai.adapters.sources.earthcam import EarthCamError, is_earthcam_url, resolve_earthcam

__all__ = ["ProbeResult", "probe_source"]

logger = logging.getLogger(__name__)

_OPEN_TIMEOUT_SECONDS = 20.0
_THUMBNAIL_MAX_EDGE = 640


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What one look at a stream found."""

    ok: bool
    detail: str
    """Why it failed, or what it is. Shown to the operator verbatim — the useful part of
    a bad URL is almost always the transport's own error."""

    width: int = 0
    height: int = 0
    codec: str = ""
    fps: float = 0.0
    source_kind: str = ""
    """`earthcam`, `rtsp` or `file` — which source the engine would build for this URL,
    so an operator can see that a page URL was understood as a page."""

    title: str = ""
    """What the source calls itself, where it says. An EarthCam page names its camera,
    which is a better label than anything a person would type."""

    thumbnail_jpeg: bytes | None = None
    """One decoded frame, JPEG, at most 640px on its longest edge. The whole point of a
    preview: a stream that opens and decodes green is indistinguishable from a working
    one in every field above."""


def _kind_for(url: str) -> str:
    if is_earthcam_url(url):
        return "earthcam"
    if url.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    return "file"


async def probe_source(url: str) -> ProbeResult:
    """Open `url`, decode one frame, and close. Never raises.

    Never raises because every caller is an HTTP handler rendering a result either way,
    and a probe that threw would make "this URL is bad" indistinguishable from "the probe
    itself is broken".
    """
    kind = _kind_for(url)
    title = ""
    target = url

    if kind == "earthcam":
        try:
            resolved = await resolve_earthcam(url)
        except EarthCamError as error:
            return ProbeResult(ok=False, detail=str(error), source_kind=kind)
        target, title = resolved.stream_url, resolved.title

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_probe_blocking, target, url, kind, title),
            timeout=_OPEN_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return ProbeResult(
            ok=False,
            detail=f"nothing decoded within {_OPEN_TIMEOUT_SECONDS:.0f}s",
            source_kind=kind,
            title=title,
        )
    except Exception as error:
        return ProbeResult(ok=False, detail=f"{type(error).__name__}: {error}", source_kind=kind)


def _probe_blocking(target: str, original: str, kind: str, title: str) -> ProbeResult:
    import av

    from sentinel_ai.adapters.sources.earthcam import browser_headers

    options: dict[str, str] = {}
    if kind == "earthcam":
        headers = browser_headers(original)
        options = {
            "headers": "".join(f"{name}: {value}\r\n" for name, value in headers.items()),
            "user_agent": headers.get("User-Agent", ""),
        }
    elif kind == "rtsp":
        options = {"rtsp_transport": "tcp", "stimeout": "10000000"}

    container = av.open(target, options=options) if options else av.open(target)
    try:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            for frame in packet.decode():
                return ProbeResult(
                    ok=True,
                    detail="decoded a frame",
                    width=frame.width,
                    height=frame.height,
                    codec=stream.codec_context.name,
                    fps=float(stream.average_rate) if stream.average_rate else 0.0,
                    source_kind=kind,
                    title=title,
                    thumbnail_jpeg=_thumbnail(frame),
                )
        return ProbeResult(
            ok=False,
            detail="the stream opened but decoded no frames",
            source_kind=kind,
            title=title,
        )
    finally:
        container.close()


def _thumbnail(frame: Any) -> bytes | None:
    """A small JPEG of one frame, or `None` if it cannot be made.

    `None` rather than raising: a probe that found a playable stream has answered the
    question, and failing the whole preview because the thumbnail encoder complained
    would turn a working camera into an unaddable one.
    """
    try:
        from PIL import Image

        picture = Image.fromarray(frame.to_ndarray(format="rgb24"))
        picture.thumbnail((_THUMBNAIL_MAX_EDGE, _THUMBNAIL_MAX_EDGE))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()
    except Exception:
        logger.debug("could not encode a preview thumbnail", exc_info=True)
        return None
