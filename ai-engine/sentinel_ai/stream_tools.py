"""`python -m sentinel_ai.stream_tools` — resolve and test a stream without the engine.

Two subcommands, both answering questions an operator has before adding a camera to
`cameras.json`:

    python -m sentinel_ai.stream_tools resolve --url "<EARTHCAM PAGE URL>"
    python -m sentinel_ai.stream_tools test    --url "<EARTHCAM PAGE URL>"

`resolve` asks the page and reports what it found. `test` goes further and actually
decodes frames, which is the only way to tell a page that parses from a camera that
plays.

**Neither ever prints the signature.** Every URL here goes through `redact_url`, and the
one field that holds a token is kept out of `EarthCamStream`'s repr as well, so a
traceback cannot leak what the print statements are careful about.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from sentinel_ai.adapters.sources.earthcam import EarthCamError, resolve_earthcam

__all__ = ["main"]

_TEST_SECONDS = 8.0
_TEST_FRAME_TARGET = 50


async def _resolve(url: str) -> int:
    try:
        stream = await resolve_earthcam(url)
    except EarthCamError as error:
        print(f"Status: FAILED\n  {error}", file=sys.stderr)
        return 1

    print("EarthCam Stream Resolver")
    print("------------------------")
    print(f"Page:             {stream.title}")
    print(f"Camera:           {stream.camera_name}")
    print(f"Camera ID:        {stream.camera_id or 'not in the stream path'}")
    print(f"Streaming Domain: {stream.streaming_host}")
    print("Protocol:         HLS")
    print(f"Resolution:       {stream.resolution or 'not advertised'}")
    print("Status:           RESOLVED")
    print(f"Playlist:         {stream.redacted_url}")
    return 0


async def _test(url: str) -> int:
    import av

    from sentinel_ai.adapters.sources.earthcam import browser_headers

    try:
        stream = await resolve_earthcam(url)
    except EarthCamError as error:
        print(f"Status: FAILED to resolve\n  {error}", file=sys.stderr)
        return 1

    print(f"Resolved {stream.title} -> {stream.redacted_url}")
    headers = browser_headers(stream.page_url)
    user_agent = headers.pop("User-Agent")
    options = {
        "headers": "".join(f"{name}: {value}\r\n" for name, value in headers.items()),
        "user_agent": user_agent,
        "timeout": "10000000",
    }

    def decode() -> tuple[int, int, int, float]:
        container = av.open(stream.stream_url, options=options)
        try:
            video = container.streams.video[0]
            frames = 0
            width = height = 0
            started = time.perf_counter()
            for packet in container.demux(video):
                for frame in packet.decode():
                    frames += 1
                    width, height = frame.width, frame.height
                    if frames >= _TEST_FRAME_TARGET:
                        return frames, width, height, time.perf_counter() - started
                if time.perf_counter() - started > _TEST_SECONDS:
                    break
            return frames, width, height, time.perf_counter() - started
        finally:
            container.close()

    try:
        frames, width, height, elapsed = await asyncio.to_thread(decode)
    except Exception as error:
        print(f"Status: FAILED to play\n  {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    if frames == 0:
        print("Status: NO FRAMES — the playlist opened but decoded nothing", file=sys.stderr)
        return 1

    print(f"Decoded:          {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} fps)")
    print(f"Frame size:       {width}x{height}")
    print("Status:           PLAYING")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_ai.stream_tools",
        description="Resolve and test a camera stream without starting the engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("resolve", "fetch the page and report the stream it points at"),
        ("test", "resolve, then decode frames and report what played"),
    ):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--url", required=True, help="the EarthCam camera page URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "resolve":
        return asyncio.run(_resolve(args.url))
    return asyncio.run(_test(args.url))


if __name__ == "__main__":
    raise SystemExit(main())
