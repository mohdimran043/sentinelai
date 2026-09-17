"""`python -m sentinel_ai.benchmark` — the §32 measurement entry point.

Deliberately a module entry rather than a console script: it is a developer and
release-engineering tool, not something a deployment installs onto its PATH, and
adding it to `[project.scripts]` would put a benchmark next to the engine in every
container image.

    # The default: every capability, the most expensive configuration there is.
    python -m sentinel_ai.benchmark --video datasets/avenue/avenue_01.mp4 \\
        --cameras 1,5,10,20,30 --seconds 30

    # The cheap tier, for the camera count a triggers-only site could reach.
    python -m sentinel_ai.benchmark --video clip.mp4 --cameras 1,10,30,50 \\
        --capabilities anomaly_detection
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sentinel_ai.benchmark.harness import default_capabilities, parse_capabilities, run_sweep
from sentinel_ai.config import Settings


def _counts(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("--cameras must be a comma-separated list of positives")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_ai.benchmark",
        description=(
            "Measure the real pipeline at N cameras. See harness.py for what these "
            "numbers do and do not cover."
        ),
    )
    parser.add_argument("--video", type=Path, required=True, help="clip every camera replays")
    parser.add_argument(
        "--cameras",
        type=_counts,
        default=(1, 5, 10, 20),
        help="comma-separated camera counts to sweep, e.g. 1,5,10,20,30",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help=(
            "wall-clock duration of each point. Short runs make the tail meaningless; "
            "30s is the floor at which p95 stops being one unlucky frame."
        ),
    )
    parser.add_argument(
        "--capabilities",
        type=str,
        default=None,
        help=(
            "comma-separated capability names, or 'none'. Defaults to all of them, so "
            "the headline number is the expensive configuration rather than the cheap one."
        ),
    )
    parser.add_argument(
        "--detect-every-n",
        type=int,
        default=1,
        help=(
            "run detection on every Nth decoded frame. The single most effective lever "
            "on camera count: one shared detector serves every camera (ADR 4), so the "
            "GPU sees arrival_rate x cameras / N detections per second."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2

    capabilities = (
        default_capabilities()
        if args.capabilities is None
        else parse_capabilities(args.capabilities)
    )
    # Realtime, so the sources pace themselves the way a camera does. Measuring with
    # an unthrottled decoder reports decoder-versus-detector thread contention instead
    # of capacity — see `harness.py` for the sweep that showed it.
    settings = Settings(
        source_realtime=True,
        detect_every_n_frames=args.detect_every_n,
    )

    print(f"video={args.video}  capabilities={capabilities.names() or ['none']}")
    results = asyncio.run(
        run_sweep(
            args.video,
            args.cameras,
            seconds=args.seconds,
            capabilities=capabilities,
            settings=settings,
        )
    )
    for result in results:
        print()
        print(result.summary())

    print()
    print("cameras   fps   drop%   peak VRAM MiB")
    for result in results:
        print(
            f"{result.cameras:>7}  {result.frames_per_second:>5.1f}  "
            f"{result.drop_rate:>5.1%}  {result.peak_vram_mib:>13}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
