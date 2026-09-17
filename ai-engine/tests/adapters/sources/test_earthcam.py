"""The EarthCam resolver: parsing, URL joining, redaction, and the allowlist.

Everything here runs offline against fixture HTML shaped like the real pages. The two
tests that actually reach earthcam.com are marked `network` and are the only ones that
can fail because somebody else's website changed — deselect them with
`-m 'not network'`.

The fixtures are the real structure, reduced: `json_base = {...};` carrying a `cam`
object, plus a `currentName`. Tokens in them are obvious fakes, because a fixture
carrying a real signature would be a real signature committed to a repository.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sentinel_ai.adapters.sources.earthcam import (
    EarthCamError,
    EarthCamStream,
    browser_headers,
    is_earthcam_url,
    redact_url,
    resolve_earthcam,
    select_variant,
)

LINKOU_PAGE = "https://www.earthcam.com/world/taiwan/newtaipeicity/linkoudistrict/"
BOURBON_PAGE = "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/?cam=bourbonstreet"

FAKE_TOKEN = "FAKETOKENFAKETOKEN"


def page_html(cameras: dict[str, Any], current: str | None = None) -> str:
    """A page shaped like EarthCam's: a `json_base` assignment and a `currentName`."""
    body = f"var json_base\t\t\t= {json.dumps({'cam': cameras})};\n"
    if current is not None:
        body += f'var currentName\t\t\t= "{current}";\n'
    return f"<html><head><script>\n{body}</script></head><body></body></html>"


def a_camera(
    name: str = "linkoudistrict",
    numeric: str = "29321",
    *,
    stream: str | None = None,
    domain: str | None = "https://videos-3.earthcam.com",
    path: str | None = None,
    title: str = "Linkou Old Street Cam",
) -> dict[str, Any]:
    default_path = f"/fecnetwork/{numeric}.flv/playlist.m3u8?t={FAKE_TOKEN}&td=202609170013"
    entry: dict[str, Any] = {"cam_name": name, "title": title}
    if stream is not None:
        entry["stream"] = stream
    if domain is not None:
        entry["html5_streamingdomain"] = domain
    entry["html5_streampath"] = default_path if path is None else path
    return entry


class FakeResponse:
    def __init__(self, text: str, url: str, status: int = 200) -> None:
        self.text = text
        self.url = url
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            request = httpx.Request("GET", self.url)
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


class FakeClient:
    """Stands in for the shared `httpx.AsyncClient`, recording what was asked for."""

    def __init__(self, routes: dict[str, FakeResponse]) -> None:
        self._routes = routes
        self.requested: list[str] = []
        self.headers_seen: list[dict[str, str]] = []

    async def get(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
        self.requested.append(url)
        self.headers_seen.append(dict(headers or {}))
        for pattern, response in self._routes.items():
            if url.startswith(pattern):
                return response
        raise AssertionError(f"unexpected request: {url}")

    async def aclose(self) -> None: ...


MASTER = (
    "#EXTM3U\n"
    "#EXT-X-VERSION:4\n"
    '#EXT-X-STREAM-INF:BANDWIDTH=540013,CODECS="avc1.4d0028",RESOLUTION=1920x1080\n'
    f"chunklist_w1489386422.m3u8?t={FAKE_TOKEN}&td=202609170013\n"
)


class TestTheAllowlist:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.earthcam.com/anything",
            "https://earthcam.com/anything",
            "https://videos-3.earthcam.com/fecnetwork/1.flv/playlist.m3u8",
        ],
    )
    def test_earthcam_hosts_are_allowed(self, url: str) -> None:
        assert is_earthcam_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/",
            "http://127.0.0.1:8000/cameras",
            "https://earthcam.com.evil.test/",
            "https://notearthcam.com/",
            "file:///etc/passwd",
        ],
    )
    def test_everything_else_is_refused(self, url: str) -> None:
        """`build_source` hands this a URL out of `cameras.json`. Without the check, a
        camera entry is a server-side request forgery primitive — and
        `earthcam.com.evil.test` is the suffix-match mistake that makes one."""
        assert not is_earthcam_url(url)

    @pytest.mark.asyncio
    async def test_resolving_a_foreign_url_refuses_before_fetching(self) -> None:
        client = FakeClient({})
        with pytest.raises(EarthCamError, match=r"not an earthcam\.com URL"):
            await resolve_earthcam("https://example.com/cam", client=client)
        assert client.requested == []


class TestRedaction:
    def test_the_query_is_dropped_whole(self) -> None:
        url = (
            f"https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8?t={FAKE_TOKEN}&td=1"
        )
        assert redact_url(url) == (
            "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8?REDACTED"
        )
        assert FAKE_TOKEN not in redact_url(url)

    def test_a_url_with_no_query_is_unchanged(self) -> None:
        url = "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8"
        assert redact_url(url) == url

    def test_the_signed_url_is_absent_from_repr(self) -> None:
        """Print statements can be audited; a traceback cannot. The field is kept out of
        `repr` so a token cannot reach a log through an exception nobody wrote."""
        stream = EarthCamStream(
            page_url=LINKOU_PAGE,
            camera_name="linkoudistrict",
            camera_id="29321",
            title="Linkou Old Street Cam",
            streaming_host="videos-3.earthcam.com",
            stream_path="/fecnetwork/29321.flv/playlist.m3u8",
            stream_url=f"https://videos-3.earthcam.com/x.m3u8?t={FAKE_TOKEN}",
        )
        assert FAKE_TOKEN not in repr(stream)
        assert FAKE_TOKEN not in str(stream)
        assert "REDACTED" in stream.redacted_url


class TestVariantSelection:
    def test_the_highest_bandwidth_variant_wins(self) -> None:
        playlist = (
            "#EXTM3U\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=100000,RESOLUTION=640x360\nlow.m3u8\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=900000,RESOLUTION=1920x1080\nhigh.m3u8\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=400000,RESOLUTION=1280x720\nmid.m3u8\n"
        )
        url, resolution = select_variant(playlist, "https://videos-3.earthcam.com/a/b/x.m3u8")
        assert url == "https://videos-3.earthcam.com/a/b/high.m3u8"
        assert resolution == "1920x1080"

    def test_bandwidth_decides_even_when_resolution_is_missing(self) -> None:
        """RESOLUTION is optional in HLS; BANDWIDTH is not. Selecting on the optional one
        would silently prefer whichever variant happened to declare it."""
        playlist = (
            "#EXTM3U\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=100000,RESOLUTION=1920x1080\nlow.m3u8\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=900000\nhigh.m3u8\n"
        )
        url, resolution = select_variant(playlist, "https://videos-3.earthcam.com/a/x.m3u8")
        assert url.endswith("high.m3u8")
        assert resolution == ""

    def test_a_media_playlist_resolves_to_itself(self) -> None:
        """No `#EXT-X-STREAM-INF` means this is already the media playlist. Treating it as
        a master would pick its first `.ts` segment as a "variant"."""
        playlist = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXTINF:2.0,\nmedia_w1_1090.ts\n"
        source = "https://videos-3.earthcam.com/a/chunklist.m3u8?t=x"
        assert select_variant(playlist, source) == (source, "")

    def test_a_relative_variant_keeps_the_signature(self) -> None:
        """The whole reason for `urljoin`: the master is signed, the variant line is
        relative, and the signature has to survive the join onto it."""
        url, _ = select_variant(
            MASTER,
            f"https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8?t={FAKE_TOKEN}",
        )
        assert url == (
            "https://videos-3.earthcam.com/fecnetwork/29321.flv/"
            f"chunklist_w1489386422.m3u8?t={FAKE_TOKEN}&td=202609170013"
        )

    def test_an_absolute_path_variant_resolves_against_the_host(self) -> None:
        playlist = "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n/other/chunklist.m3u8?t=z\n"
        url, _ = select_variant(playlist, "https://videos-3.earthcam.com/a/b/playlist.m3u8")
        assert url == "https://videos-3.earthcam.com/other/chunklist.m3u8?t=z"

    def test_a_fully_qualified_variant_is_left_alone(self) -> None:
        playlist = (
            "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n"
            "https://videos-9.earthcam.com/x/chunklist.m3u8?t=z\n"
        )
        url, _ = select_variant(playlist, "https://videos-3.earthcam.com/a/playlist.m3u8")
        assert url == "https://videos-9.earthcam.com/x/chunklist.m3u8?t=z"


class TestPageParsing:
    @pytest.mark.asyncio
    async def test_a_single_camera_page_resolves(self) -> None:
        html = page_html({"linkoudistrict": a_camera()}, current="linkoudistrict")
        client = FakeClient(
            {
                LINKOU_PAGE: FakeResponse(html, LINKOU_PAGE),
                "https://videos-3.earthcam.com/": FakeResponse(
                    MASTER,
                    "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8"
                    f"?t={FAKE_TOKEN}&td=202609170013",
                ),
            }
        )

        stream = await resolve_earthcam(LINKOU_PAGE, client=client)

        assert stream.camera_name == "linkoudistrict"
        assert stream.camera_id == "29321"
        assert stream.title == "Linkou Old Street Cam"
        assert stream.streaming_host == "videos-3.earthcam.com"
        assert stream.stream_path == "/fecnetwork/29321.flv/playlist.m3u8"
        assert stream.resolution == "1920x1080"
        assert stream.stream_url.endswith(
            f"chunklist_w1489386422.m3u8?t={FAKE_TOKEN}&td=202609170013"
        )

    @pytest.mark.asyncio
    async def test_the_cam_query_parameter_picks_the_camera(self) -> None:
        """Bourbon Street's page carries several cameras. `?cam=` is how a visitor picks
        one, so it is how this picks one."""
        html = page_html(
            {
                "catsmeow2": a_camera("catsmeow2", "4282", title="Cat's Meow"),
                "bourbonstreet": a_camera("bourbonstreet", "4280", title="Bourbon Street"),
            },
            current="catsmeow2",
        )
        client = FakeClient(
            {
                BOURBON_PAGE: FakeResponse(html, BOURBON_PAGE),
                "https://videos-3.earthcam.com/": FakeResponse(
                    MASTER, "https://videos-3.earthcam.com/fecnetwork/4280.flv/playlist.m3u8"
                ),
            }
        )

        stream = await resolve_earthcam(BOURBON_PAGE, client=client)

        assert stream.camera_name == "bourbonstreet"
        assert stream.camera_id == "4280"

    @pytest.mark.asyncio
    async def test_current_name_is_used_when_no_camera_is_requested(self) -> None:
        page = "https://www.earthcam.com/usa/louisiana/neworleans/bourbonstreet/"
        html = page_html(
            {
                "catsmeow2": a_camera("catsmeow2", "4282"),
                "bourbonstreet": a_camera("bourbonstreet", "4280"),
            },
            current="bourbonstreet",
        )
        client = FakeClient(
            {
                page: FakeResponse(html, page),
                "https://videos-3.earthcam.com/": FakeResponse(
                    MASTER, "https://videos-3.earthcam.com/fecnetwork/4280.flv/playlist.m3u8"
                ),
            }
        )

        stream = await resolve_earthcam(page, client=client)

        assert stream.camera_name == "bourbonstreet"

    @pytest.mark.asyncio
    async def test_a_camera_the_page_does_not_have_is_named_in_the_error(self) -> None:
        """Silently falling back to a different camera would give somebody a working
        stream of the wrong street."""
        page = "https://www.earthcam.com/x/?cam=nosuchcam"
        html = page_html({"bourbonstreet": a_camera("bourbonstreet", "4280")})
        client = FakeClient({page: FakeResponse(html, page)})

        with pytest.raises(EarthCamError, match="nosuchcam"):
            await resolve_earthcam(page, client=client)

    @pytest.mark.asyncio
    async def test_the_two_part_form_is_joined_when_there_is_no_stream_url(self) -> None:
        html = page_html({"linkoudistrict": a_camera(stream=None)})
        client = FakeClient(
            {
                LINKOU_PAGE: FakeResponse(html, LINKOU_PAGE),
                "https://videos-3.earthcam.com/": FakeResponse(
                    MASTER, "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8"
                ),
            }
        )

        stream = await resolve_earthcam(LINKOU_PAGE, client=client)

        assert stream.stream_path == "/fecnetwork/29321.flv/playlist.m3u8"
        # Joined, not concatenated: no `//` between host and path.
        assert "earthcam.com//" not in client.requested[1]

    @pytest.mark.asyncio
    async def test_a_page_with_no_configuration_says_so(self) -> None:
        client = FakeClient({LINKOU_PAGE: FakeResponse("<html>nothing</html>", LINKOU_PAGE)})
        with pytest.raises(EarthCamError, match="no json_base"):
            await resolve_earthcam(LINKOU_PAGE, client=client)

    @pytest.mark.asyncio
    async def test_a_camera_pointing_off_site_is_refused(self) -> None:
        """The page decides this URL. A resolver that followed it anywhere would fetch
        arbitrary hosts from inside the engine's network."""
        html = page_html({"linkoudistrict": a_camera(stream="https://evil.test/steal.m3u8")})
        client = FakeClient({LINKOU_PAGE: FakeResponse(html, LINKOU_PAGE)})

        with pytest.raises(EarthCamError, match=r"outside earthcam\.com"):
            await resolve_earthcam(LINKOU_PAGE, client=client)


class TestStaleSignatures:
    @pytest.mark.asyncio
    async def test_a_403_on_the_playlist_is_reported_without_the_token(self) -> None:
        """Test 3's first half: a stale signature surfaces as an error the caller can act
        on, and the error text carries no token."""
        html = page_html({"linkoudistrict": a_camera()})
        client = FakeClient(
            {
                LINKOU_PAGE: FakeResponse(html, LINKOU_PAGE),
                "https://videos-3.earthcam.com/": FakeResponse(
                    "", "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8", 403
                ),
            }
        )

        with pytest.raises(EarthCamError) as caught:
            await resolve_earthcam(LINKOU_PAGE, client=client)

        assert "403" in str(caught.value)
        assert FAKE_TOKEN not in str(caught.value)

    @pytest.mark.asyncio
    async def test_resolving_again_fetches_the_page_again(self) -> None:
        """Test 3's second half, and the whole recovery strategy: nothing is cached, so a
        second resolution asks the page for a current signature rather than reusing the
        one that was refused."""
        first = page_html(
            {"linkoudistrict": a_camera(path="/fecnetwork/29321.flv/playlist.m3u8?t=OLD")}
        )
        second = page_html(
            {"linkoudistrict": a_camera(path="/fecnetwork/29321.flv/playlist.m3u8?t=NEW")}
        )
        pages = [FakeResponse(first, LINKOU_PAGE), FakeResponse(second, LINKOU_PAGE)]

        class TwoPageClient(FakeClient):
            async def get(self, url: str, headers: dict[str, str] | None = None) -> FakeResponse:
                self.requested.append(url)
                if url == LINKOU_PAGE:
                    return pages.pop(0)
                return FakeResponse("#EXTM3U\n#EXT-X-TARGETDURATION:2\n", url)

        client = TwoPageClient({})
        one = await resolve_earthcam(LINKOU_PAGE, client=client)
        two = await resolve_earthcam(LINKOU_PAGE, client=client)

        assert "t=OLD" in one.stream_url
        assert "t=NEW" in two.stream_url
        assert client.requested.count(LINKOU_PAGE) == 2


class TestRequestContext:
    def test_the_referer_is_the_page_itself(self) -> None:
        headers = browser_headers(LINKOU_PAGE)
        assert headers["Referer"] == LINKOU_PAGE
        assert headers["Origin"] == "https://www.earthcam.com"
        assert "Chrome" in headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_media_requests_carry_the_page_as_referrer(self) -> None:
        html = page_html({"linkoudistrict": a_camera()})
        client = FakeClient(
            {
                LINKOU_PAGE: FakeResponse(html, LINKOU_PAGE),
                "https://videos-3.earthcam.com/": FakeResponse(
                    MASTER, "https://videos-3.earthcam.com/fecnetwork/29321.flv/playlist.m3u8"
                ),
            }
        )

        await resolve_earthcam(LINKOU_PAGE, client=client)

        # The page request carries the client's default headers; the media request adds
        # the player's context. That second one is what the CDN serves on.
        assert client.headers_seen[1]["Referer"] == LINKOU_PAGE


@pytest.mark.network
class TestAgainstTheRealPages:
    """The two cameras this was built for. Marked `network` — these fail when somebody
    else's website changes, which is information, but not a reason to fail CI."""

    @pytest.mark.asyncio
    async def test_linkou_old_street_resolves(self) -> None:
        stream = await resolve_earthcam(LINKOU_PAGE)

        assert stream.camera_name == "linkoudistrict"
        assert stream.camera_id == "29321"
        assert stream.streaming_host.endswith("earthcam.com")
        assert ".m3u8" in stream.stream_url
        assert stream.resolved_at > 0

    @pytest.mark.asyncio
    async def test_bourbon_street_resolves_and_is_a_different_camera(self) -> None:
        """The resolver must not have learned anything camera-specific from Linkou."""
        stream = await resolve_earthcam(BOURBON_PAGE)

        assert stream.camera_name == "bourbonstreet"
        assert stream.camera_id
        assert stream.camera_id != "29321"
        assert stream.streaming_host.endswith("earthcam.com")
