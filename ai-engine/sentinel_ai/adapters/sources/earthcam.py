"""Resolving an EarthCam page URL to the stream its own player would use.

EarthCam publishes free, public, no-login cameras. The page hands every visitor a
short-lived signed playlist URL, and the player then fetches it with the page as its
referrer. This module does the same two things — fetch the page, use the URL the page
gave it — so that a camera can be configured as

    "url": "https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/"

rather than as an `.m3u8` somebody pasted out of a browser an hour ago.

**It resolves; it does not circumvent.** There is no login, paywall or DRM here. The
signature is issued by the page to anonymous visitors and is not an access control this
code defeats — it is a freshness token, and the entire point of this module is to keep
getting a current one instead of clinging to a stale one. Nothing here forges a token,
extends one, or retries past a refusal; a 403 is answered by asking the page again, and
by backing off so the asking stays polite.

Why the whole page, every time
--------------------------------
`html5_streampath` arrives with `?t=...&td=...` already attached and both change.
Caching the resolved URL and reusing it is exactly the failure this module exists to
prevent, so `EarthCamStream` carries `resolved_at` and callers are expected to
re-resolve rather than persist one.

Redaction
-----------
The signed URL is the one value here worth keeping out of logs and tracebacks, so
`EarthCamStream.__repr__` omits it. Log `stream.redacted_url`, never `stream.stream_url`.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse, urlsplit, urlunsplit

__all__ = [
    "EARTHCAM_USER_AGENT",
    "EarthCamError",
    "EarthCamStream",
    "browser_headers",
    "is_earthcam_url",
    "redact_url",
    "resolve_earthcam",
    "select_variant",
]

logger = logging.getLogger(__name__)

EARTHCAM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
)
"""What a current desktop Chrome sends.

A default `python-httpx/0.27` user agent is not dishonest, it is simply not what the CDN
serves media to, and the resulting 403 is indistinguishable from a real refusal. This is
the player's context, not a disguise for a different actor.
"""

_PAGE_ORIGIN = "https://www.earthcam.com"

_ALLOWED_HOST_SUFFIXES = ("earthcam.com",)
"""Only EarthCam. `build_source` hands this module a URL out of `cameras.json`, and a
resolver that fetched whatever it was given would turn a camera config into a
server-side request forgery primitive. Checked on the page URL *and* on every media URL
the page points at, because the second is page-controlled if the first ever is."""

_JSON_BASE = re.compile(r"json_base\s*=\s*(\{.*?\})\s*;\s*$", re.S | re.M)
_CURRENT_NAME = re.compile(r'currentName\s*=\s*"([^"]+)"')
_NUMERIC_ID = re.compile(r"/(\d+)\.flv\b")

_HTTP_TIMEOUT_SECONDS = 15.0


class EarthCamError(RuntimeError):
    """The page could not be turned into a playable stream.

    One exception type rather than several: every cause — page unreachable, layout
    changed, camera not on the page, playlist refused — leads the caller to the same
    place, which is to back off and ask the page again later.
    """


def is_earthcam_url(url: str) -> bool:
    """Whether `url` is an EarthCam page or media URL this module may fetch."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in _ALLOWED_HOST_SUFFIXES)


def redact_url(url: str) -> str:
    """`.../playlist.m3u8?REDACTED` — safe to log, still identifies the stream.

    The path is kept because it names the camera and is how somebody reading a log works
    out which stream misbehaved. The query is dropped whole rather than field by field:
    `t` is the signature today, and a parameter nobody has thought about yet should not
    have to be added to a denylist to stay out of a log.
    """
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "REDACTED" if parts.query else "", "")
    )


def browser_headers(page_url: str) -> dict[str, str]:
    """The request context EarthCam's own player sends for media.

    `Referer` is the page rather than the site root: it is what the player sends, and it
    is the truthful statement of where this request came from.
    """
    return {
        "User-Agent": EARTHCAM_USER_AGENT,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": _PAGE_ORIGIN,
        "Referer": page_url,
    }


@dataclass(frozen=True, slots=True)
class EarthCamStream:
    """A resolved, currently-signed stream. Short-lived by construction."""

    page_url: str
    camera_name: str
    """EarthCam's own key for the camera on the page, e.g. `linkoudistrict`."""

    camera_id: str
    """The numeric id in the stream path, e.g. `29321`. Empty when the path carries
    none — several cameras share a page and not all of their paths are `.flv`."""

    title: str
    streaming_host: str
    stream_path: str
    """Path only, signature stripped. Stable across resolutions, so it is the part worth
    logging and comparing."""

    stream_url: str = field(repr=False)
    """The signed URL. **Never logged** — `__repr__` omits it and `redacted_url` is what
    to print. Kept out of `repr` rather than merely out of log statements, because the
    way a token reaches a log is usually a traceback nobody wrote."""

    resolution: str = ""
    resolved_at: float = 0.0

    @property
    def redacted_url(self) -> str:
        return redact_url(self.stream_url)

    def __str__(self) -> str:
        return f"{self.title or self.camera_name} <{self.redacted_url}>"


def _decode_json_base(html: str) -> dict[str, Any]:
    match = _JSON_BASE.search(html)
    if match is None:
        raise EarthCamError(
            "no json_base configuration on the page; EarthCam's page layout may have "
            "changed, or this URL is not a camera page"
        )
    try:
        decoded: dict[str, Any] = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise EarthCamError(f"json_base on the page is not valid JSON: {exc}") from exc
    return decoded


def _choose_camera(cameras: dict[str, Any], html: str, page_url: str) -> tuple[str, dict[str, Any]]:
    """Which camera on the page to use.

    Pages carry several. Bourbon Street's has `catsmeow2`, `bourbonstreet` and more, and
    which one a visitor sees comes from `?cam=` and otherwise from the page's own
    `currentName`. Both are honoured, in that order, before falling back to the first —
    so a URL that names a camera always gets that camera.
    """
    if not cameras:
        raise EarthCamError("json_base carries no cameras")

    requested = parse_qs(urlparse(page_url).query).get("cam", [None])[0]
    if requested is not None:
        if requested in cameras:
            return requested, cameras[requested]
        raise EarthCamError(
            f"the page does not carry a camera called {requested!r}; it has: "
            f"{', '.join(sorted(cameras))}"
        )

    current = _CURRENT_NAME.search(html)
    if current is not None and current.group(1) in cameras:
        return current.group(1), cameras[current.group(1)]

    name = next(iter(cameras))
    return name, cameras[name]


def _stream_url_for(camera: dict[str, Any]) -> str:
    """The signed playlist URL, preferring what the page already built.

    `stream` is the page's own fully-formed URL. `html5_streamingdomain` +
    `html5_streampath` is the same thing in two halves and is the documented fallback.
    Joined with `urljoin` rather than concatenated, so a domain with a trailing slash or
    a path without a leading one cannot produce `https://host//path` — which some CDNs
    serve and others refuse.
    """
    direct = camera.get("stream")
    if isinstance(direct, str) and ".m3u8" in direct:
        return direct

    domain = camera.get("html5_streamingdomain")
    path = camera.get("html5_streampath")
    if isinstance(domain, str) and isinstance(path, str) and path:
        return urljoin(domain if domain.endswith("/") else domain + "/", path.lstrip("/"))

    raise EarthCamError(
        "the camera entry has neither a 'stream' URL nor "
        "'html5_streamingdomain' + 'html5_streampath'"
    )


def select_variant(playlist: str, playlist_url: str) -> tuple[str, str]:
    """`(url, resolution)` of the best variant in a master playlist.

    Returns the playlist's own URL when it is already a media playlist — a master is
    identified by carrying `#EXT-X-STREAM-INF`, and treating a media playlist as a
    master would pick its first `.ts` segment as a "variant".

    Highest `BANDWIDTH` wins. Resolution is reported but never selected on: it is absent
    from some variants, while `BANDWIDTH` is required by the HLS specification.

    Relative targets are resolved with `urljoin` against the playlist's own URL, which is
    what carries the signature down from `playlist.m3u8?t=...` onto
    `chunklist_x.m3u8?t=...`, and what makes an absolute-path variant work too.
    """
    best: tuple[int, str, str] | None = None
    lines = [line.strip() for line in playlist.splitlines()]
    for index, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        target = next((lines[j] for j in range(index + 1, len(lines)) if lines[j]), "")
        if not target or target.startswith("#"):
            continue
        bandwidth_match = re.search(r"BANDWIDTH=(\d+)", line)
        resolution_match = re.search(r"RESOLUTION=([0-9x]+)", line)
        bandwidth = int(bandwidth_match.group(1)) if bandwidth_match else 0
        resolution = resolution_match.group(1) if resolution_match else ""
        if best is None or bandwidth > best[0]:
            best = (bandwidth, urljoin(playlist_url, target), resolution)

    if best is None:
        return playlist_url, ""
    return best[1], best[2]


async def resolve_earthcam(page_url: str, *, client: Any | None = None) -> EarthCamStream:
    """Fetch an EarthCam page and return the stream its player would use, signed and now.

    One client for the page and every playlist under it, so the cookies the page sets are
    present on the media requests — which is how a browser does it and, on some of these
    cameras, the difference between 200 and 403.

    Raises `EarthCamError` for every failure. The caller's answer is always the same:
    back off, ask the page again later.
    """
    if not is_earthcam_url(page_url):
        raise EarthCamError(f"{redact_url(page_url)} is not an earthcam.com URL")

    import httpx

    owned = client is None
    session: Any = client or httpx.AsyncClient(
        follow_redirects=True,
        timeout=_HTTP_TIMEOUT_SECONDS,
        headers={
            "User-Agent": EARTHCAM_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    try:
        return await _resolve_with(session, page_url)
    finally:
        if owned:
            await session.aclose()


async def _resolve_with(session: Any, page_url: str) -> EarthCamStream:
    import httpx

    try:
        page = await session.get(page_url)
        page.raise_for_status()
    except httpx.HTTPError as exc:
        raise EarthCamError(f"could not fetch {redact_url(page_url)}: {exc}") from exc

    html = page.text
    cameras = _decode_json_base(html).get("cam")
    if not isinstance(cameras, dict):
        raise EarthCamError("json_base has no 'cam' object")

    name, camera = _choose_camera(cameras, html, page_url)
    stream_url = _stream_url_for(camera)
    if not is_earthcam_url(stream_url):
        # The page decides this URL. A resolver that followed wherever it pointed would
        # fetch arbitrary hosts from inside the engine's network.
        raise EarthCamError("the page points its stream at a host outside earthcam.com")

    parts = urlsplit(stream_url)
    headers = browser_headers(page_url)

    try:
        master = await session.get(stream_url, headers=headers)
        master.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise EarthCamError(
            f"{exc.response.status_code} for {redact_url(stream_url)}; the page's "
            "signature was refused"
        ) from exc
    except httpx.HTTPError as exc:
        raise EarthCamError(f"could not fetch {redact_url(stream_url)}: {exc}") from exc

    variant_url, resolution = select_variant(master.text, str(master.url))
    if not is_earthcam_url(variant_url):
        raise EarthCamError("the master playlist points outside earthcam.com")

    numeric = _NUMERIC_ID.search(parts.path)
    return EarthCamStream(
        page_url=page_url,
        camera_name=name,
        camera_id=numeric.group(1) if numeric else "",
        title=str(camera.get("title") or name),
        streaming_host=parts.netloc,
        stream_path=parts.path,
        # The variant, not the master: ffmpeg takes either, and handing it the one
        # already chosen keeps variant selection here, where it is tested, rather than
        # inside ffmpeg where it is not.
        stream_url=variant_url,
        resolution=resolution,
        resolved_at=time.time(),
    )
