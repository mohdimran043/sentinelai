"""Does the fall detector detect real falls? (spec §7, ADR 12)

Until this module existed, `docs/performance.md` said plainly that the fall state
machine had "**not** been measured against real fall footage, because none is in this
repository. Its false-positive and false-negative rates on real video are unknown."
That was true and it was the largest gap in the project: a critical-severity detector
validated only against scripted geometry.

The dataset
-------------
**URFall** (Kwolek & Kepski, University of Rzeszow): 30 clips each containing one fall,
and 40 clips of activities of daily living containing none — sitting down, bending,
lying down deliberately, picking things up. The ADL clips are the important half. A
detector that fires on somebody sitting heavily is a detector an operator switches off,
and posture alone cannot tell the two apart, which is the whole argument of ADR 12.

`cam0` is the camera parallel to the floor — the view a wall-mounted security camera
has. `cam1` is ceiling-mounted and is not used: nothing in this system is designed for
an overhead view, and scoring against one would flatter or damn the detector for the
wrong reason.

    datasets/urfd/fetch.sh          # 70 clips, about 90 MB

What is scored, and why two numbers rather than one
-----------------------------------------------------
**Raised** is what an operator receives: the machine completed and produced a candidate.
That is the number that matters, and on this dataset it is dominated by an awkward fact
— URFall clips end about a second after the fall, and the shipped policy waits
`settle_seconds` (3.0) of stillness before it will say anything. Most clips stop before
that window closes.

**Signature seen** is the machine reaching its `DOWN` phase: upright, then a descent
fast enough in body heights per second, then horizontal. It is the detection itself,
separated from the confirmation delay that follows it. Reporting only "raised" would
blame the detector for the dataset's clip length; reporting only "signature seen" would
quietly drop the requirement that makes this a fall detector rather than a posture
detector.

Both are scored on the same clips, and `--settle-seconds` re-runs with a different
confirmation delay so the trade between the two is visible rather than argued about.

A clip scored as detected is not scored as correctly *timed* — see the limitations the
run prints.

This composes the real detector, the real tracker, the real pose model and the real
`observe_falls`, through `BehaviourEngine`, in the order `CameraRunner` uses. It is not
a reimplementation: a bug in the shipped path is a bug in these numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector
from sentinel_ai.adapters.pose.yolo11_pose import Yolo11PoseEstimator
from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.config import Settings
from sentinel_ai.domain.behaviour.fall import (
    FallPolicy,
    FallTracker,
    _Phase,
    observe_falls,
)
from sentinel_ai.domain.behaviour.observation import BehaviourObservation, PersonPose
from sentinel_ai.domain.entities import SceneState
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from sentinel_ai.validation.report import BinaryOutcome, format_confusion

__all__ = ["ClipResult", "run_clip", "score"]

_SIGNATURE_PHASES = frozenset({_Phase.DOWN, _Phase.REPORTED})
"""The machine has seen the whole signature: upright, a fast descent, then horizontal.

Reaching into a private enum, deliberately and only here. `_Phase` is private because
nothing in the *product* should branch on it — a consumer that did would be reacting to
a fall before the confirmation delay that makes it a fall rather than a posture. A
validation harness is the one caller allowed to look, because separating "did it see
it" from "did it finish waiting" is the whole point of this measurement."""


@dataclass(frozen=True, slots=True)
class ClipResult:
    name: str
    expected_fall: bool
    raised: bool
    """The machine completed and produced a candidate — what an operator would get."""

    signature_seen: bool
    """The machine reached `down`: upright, a fast descent, then horizontal. The
    detection proper, before the confirmation delay that follows it."""

    first_signature_seconds: float | None
    first_raised_seconds: float | None
    duration_seconds: float
    frames: int
    used_pose_fraction: float
    """How often the reading came from a skeleton rather than a bounding box.

    Reported because ADR 12 claims pose is an enhancement and geometry is the fallback.
    If this is near zero on real footage the claim is technically true and practically
    hollow, and the recall number below is a geometry-only number wearing a pose label.
    """


async def run_clip(
    path: Path,
    *,
    detector: Yolo11Detector,
    pose: Yolo11PoseEstimator,
    policy: FallPolicy,
    expected_fall: bool,
) -> ClipResult:
    """Replay one clip through the real pipeline and report whether a fall was raised.

    `realtime=False`: this is an accuracy measurement, and pacing 70 clips at 25 fps
    would take half an hour to produce numbers that do not depend on wall time. The
    state machine reads `SceneState.timestamp`, which is the clip's own presentation
    timeline either way — the same values it would see live.
    """
    source = FileSource(str(path), camera_id=path.stem, realtime=False)
    tracker = ByteTrackTracker()
    motion = MotionAnalyzer()
    state = FallTracker()

    frames = 0
    pose_frames = 0
    last_timestamp = 0.0
    first_raised: float | None = None
    first_signature: float | None = None
    try:
        async for frame in source:
            frames += 1
            last_timestamp = frame.timestamp
            detections = await detector.detect(frame)
            tracks = tracker.update(detections, frame.timestamp)
            signals = motion.analyze(frame)
            poses: dict[int, PersonPose] = {}
            if tracks:
                poses = dict(await pose.estimate(frame, tracks))
                if poses:
                    pose_frames += 1
            scene = SceneState(
                camera_id=path.stem,
                frame_index=frame.frame_index,
                timestamp=frame.timestamp,
                detections=detections,
                tracks=tracks,
                motion_energy=signals.motion_energy,
                scene_signature=signals.scene_signature,
            )
            state, candidates = observe_falls(
                BehaviourObservation(
                    scene=scene,
                    poses=poses,
                    frame_width=frame.width,
                    frame_height=frame.height,
                ),
                policy,
                state,
            )
            # `observe_falls` rather than `BehaviourEngine.observe`: the same shipped
            # function the engine calls, but the tracker it threads is visible here, and
            # the phase is the second of the two numbers this module reports.
            if first_signature is None and any(
                track.phase in _SIGNATURE_PHASES for track in state.tracks.values()
            ):
                first_signature = frame.timestamp
            if first_raised is None and candidates:
                first_raised = frame.timestamp
    finally:
        await source.close()

    return ClipResult(
        name=path.stem,
        expected_fall=expected_fall,
        raised=first_raised is not None,
        signature_seen=first_signature is not None,
        first_signature_seconds=first_signature,
        first_raised_seconds=first_raised,
        duration_seconds=last_timestamp,
        frames=frames,
        used_pose_fraction=pose_frames / frames if frames else 0.0,
    )


def score(results: Sequence[ClipResult], *, signal: str = "raised") -> BinaryOutcome:
    """Confusion matrix over clips. `signal` is `raised` or `signature_seen`."""
    flagged = [bool(getattr(r, signal)) for r in results]
    return BinaryOutcome(
        true_positives=sum(
            1 for r, f in zip(results, flagged, strict=True) if r.expected_fall and f
        ),
        false_negatives=sum(
            1 for r, f in zip(results, flagged, strict=True) if r.expected_fall and not f
        ),
        true_negatives=sum(
            1 for r, f in zip(results, flagged, strict=True) if not r.expected_fall and not f
        ),
        false_positives=sum(
            1 for r, f in zip(results, flagged, strict=True) if not r.expected_fall and f
        ),
    )


def _clips(dataset: Path, limit: int | None) -> tuple[list[Path], list[Path]]:
    falls = sorted(dataset.glob("fall-*-cam0.mp4"))
    adls = sorted(dataset.glob("adl-*-cam0.mp4"))
    if limit is not None:
        falls, adls = falls[:limit], adls[:limit]
    return falls, adls


async def _main(args: argparse.Namespace) -> int:
    falls, adls = _clips(args.dataset, args.limit)
    if not falls and not adls:
        print(
            f"no URFall clips under {args.dataset}. Run datasets/urfd/fetch.sh first.",
            file=sys.stderr,
        )
        return 2

    settings = Settings()
    detector = Yolo11Detector(
        model_id=settings.detector_model_id,
        conf=settings.detector_conf_threshold,
        iou=settings.detector_iou_threshold,
        imgsz=settings.detector_imgsz,
        device=args.device,
    )
    pose = Yolo11PoseEstimator(
        model_id=settings.pose_model_id,
        conf=settings.pose_conf_threshold,
        imgsz=settings.pose_imgsz,
        device=args.device,
    )
    await detector.initialize()
    await detector.warmup()
    await pose.initialize()
    await pose.warmup()

    policy = (
        FallPolicy()
        if args.settle_seconds is None
        else FallPolicy(settle_seconds=args.settle_seconds)
    )
    print(f"{len(falls)} fall clips, {len(adls)} ADL clips; policy = {policy}\n")

    results: list[ClipResult] = []
    for path, expected in [(p, True) for p in falls] + [(p, False) for p in adls]:
        result = await run_clip(
            path, detector=detector, pose=pose, policy=policy, expected_fall=expected
        )
        results.append(result)
        sig = (
            f"signature {result.first_signature_seconds:4.1f}s"
            if result.first_signature_seconds is not None
            else "signature   -  "
        )
        raised = (
            f"raised {result.first_raised_seconds:4.1f}s"
            if result.first_raised_seconds is not None
            else "raised   -  "
        )
        print(
            f"  {result.name:18s} {'fall' if expected else 'adl ':4s} {sig}  {raised}"
            f"  clip {result.duration_seconds:4.1f}s  pose {result.used_pose_fraction:5.1%}"
        )

    print("\nRAISED — the candidate an operator would actually receive")
    print(format_confusion(score(results), positive="falls", negative="daily activities"))
    print("\nSIGNATURE SEEN — upright, fast descent, horizontal, before the settle wait")
    print(
        format_confusion(
            score(results, signal="signature_seen"),
            positive="falls",
            negative="daily activities",
        )
    )
    short = [
        r
        for r in results
        if r.expected_fall
        and r.signature_seen
        and not r.raised
        and r.first_signature_seconds is not None
        and r.duration_seconds - r.first_signature_seconds < policy.settle_seconds
    ]
    if short:
        print(
            f"\n  {len(short)} fall clip(s) saw the signature and ended before "
            f"settle_seconds ({policy.settle_seconds}s) could elapse. On this dataset "
            f"\n  that is the gap between the two tables, not a detector that missed."
        )
    pose_seen = sum(r.used_pose_fraction * r.frames for r in results)
    total = sum(r.frames for r in results)
    print(f"\n  pose available on {pose_seen / total:.1%} of frames overall")
    print(
        "\n  Not measured here: detection *timing* (a clip is scored on whether a fall "
        "\n  was raised at all), behaviour on falls that begin off-camera, and any "
        "\n  population other than this dataset's actors."
    )
    await detector.shutdown()
    await pose.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_ai.validation.falls",
        description="Score the fall state machine against the URFall dataset.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("../datasets/urfd"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=None,
        help=(
            "override FallPolicy.settle_seconds, to see the trade between raising "
            "early and raising on a person who merely sat down"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="clips per class, for a quick smoke run"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
