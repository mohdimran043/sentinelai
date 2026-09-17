"""Does the face pipeline identify the right person, and equally well for everybody?

Until this module existed, `docs/operations.md` said of person authorization that
"**demographic accuracy is unmeasured**. Face recognition error rates are well
documented as uneven across skin tone, age and gender, and nothing in this repository
measures that for this model on your population." The latency and VRAM figures were
real; nothing said whether the thing worked.

Two questions, two datasets, because no public dataset answers both
--------------------------------------------------------------------
**Is it accurate?** LFW's pairs protocol: matched pairs of the same person and of
different people, scored at the threshold this engine actually ships
(`AuthorizationPolicy.match_threshold`, 0.42). The threshold is *reported against*, not
tuned on, the split being scored — `--tune` prints the best threshold for a split so it
can be chosen on one and reported on another, and the run says which it did.

**Is it even-handed?** FairFace, which carries race, gender and age labels. It has no
identity labels, so same-person pairs cannot be built from it and verification accuracy
per group is out of reach. What *is* reachable is the thing this pipeline gates on:

* **detection rate per group** — a face the detector never finds is a person the system
  never checks;
* **quality-bar pass rate per group** — `min_box_pixels`, `min_frontality` and
  `detector_confidence` decide whose face is allowed into a comparison at all, and a
  group failing them more often is a group more often treated as unverifiable;
* **impostor similarity per group** — every FairFace pair is two different people, so
  the distribution of their similarity is the distribution of near-misses. A group whose
  impostors sit closer together has a higher false-match rate at any fixed threshold,
  and this engine uses a fixed threshold.

That is a real fairness measurement of *this pipeline's* decisions. It is not a
verification-accuracy-by-group measurement, and this module says so rather than letting
the stronger claim be inferred.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sentinel_ai.adapters.face.insightface_pipeline import InsightFacePipeline
from sentinel_ai.domain.identity import FaceQuality, cosine_similarity
from sentinel_ai.domain.policy.authorization import AuthorizationPolicy
from sentinel_ai.ports.face import DetectedFace
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.validation.report import BinaryOutcome, format_confusion

__all__ = ["PairScore", "score_pairs"]


@dataclass(frozen=True, slots=True)
class PairScore:
    same_person: bool
    similarity: float | None
    """`None` when a face could not be found in one of the two images.

    Kept distinct from a low similarity throughout. A pipeline that cannot see a face
    has not decided that two people differ — and counting a non-detection as a non-match
    is how a detector that fails on a whole group scores as merely strict.
    """


def _decode_image(payload: bytes) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as handle:
        rgb = handle.convert("RGB")
        return np.asarray(rgb)[:, :, ::-1].copy()  # PIL is RGB; the pipeline wants BGR


def _as_frame(pixels: Any, index: int) -> FrameData:
    return FrameData(
        camera_id="validation",
        frame_index=index,
        timestamp=float(index),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        pixels=pixels,
    )


def _largest(faces: Sequence[DetectedFace]) -> DetectedFace | None:
    """The biggest face in the image, which for a portrait is the subject.

    The same rule the enrolment endpoint uses, so this measures what the product does.
    """
    return max(faces, key=lambda f: f.box.area, default=None)


def _passes_quality(quality: FaceQuality, policy: AuthorizationPolicy) -> bool:
    return quality.is_usable(
        min_box_pixels=policy.min_box_pixels,
        min_frontality=policy.min_frontality,
        min_confidence=policy.min_detector_confidence,
    )


async def _embed_largest(
    pipeline: InsightFacePipeline, pixels: Any, index: int
) -> tuple[Any | None, FaceQuality | None]:
    faces = await pipeline.detect(_as_frame(pixels, index))
    face = _largest(faces)
    if face is None:
        return None, None
    (embedding,) = await pipeline.embed((face,))
    return embedding, face.quality


async def score_pairs(
    pipeline: InsightFacePipeline, rows: Sequence[tuple[bytes, bytes, bool]]
) -> list[PairScore]:
    scores: list[PairScore] = []
    for index, (left, right, same) in enumerate(rows):
        a, _ = await _embed_largest(pipeline, _decode_image(left), index * 2)
        b, _ = await _embed_largest(pipeline, _decode_image(right), index * 2 + 1)
        similarity = None if a is None or b is None else cosine_similarity(a.vector, b.vector)
        scores.append(PairScore(same_person=same, similarity=similarity))
    return scores


def confusion_at(scores: Sequence[PairScore], threshold: float) -> BinaryOutcome:
    """A pair with no detectable face counts as *not matched*, which is what the engine
    would do: no usable face means no observation, and no observation means no match."""
    matched = [(s.similarity is not None and s.similarity >= threshold) for s in scores]
    return BinaryOutcome(
        true_positives=sum(1 for s, m in zip(scores, matched, strict=True) if s.same_person and m),
        false_negatives=sum(
            1 for s, m in zip(scores, matched, strict=True) if s.same_person and not m
        ),
        true_negatives=sum(
            1 for s, m in zip(scores, matched, strict=True) if not s.same_person and not m
        ),
        false_positives=sum(
            1 for s, m in zip(scores, matched, strict=True) if not s.same_person and m
        ),
    )


def best_threshold(scores: Sequence[PairScore]) -> tuple[float, float]:
    """The accuracy-maximising threshold and its accuracy. **Only for `--tune`.**

    Reporting a threshold chosen on the same pairs it is scored against is the oldest
    way to publish a number that does not survive contact with a different set.
    """
    candidates = sorted({round(s.similarity, 3) for s in scores if s.similarity is not None})
    best = (0.0, 0.0)
    for threshold in candidates:
        accuracy = confusion_at(scores, threshold).accuracy
        if accuracy > best[1]:
            best = (threshold, accuracy)
    return best


def _load_pairs(path: Path, limit: int | None) -> list[tuple[bytes, bytes, bool]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    labels = table.column("pair").to_pylist()
    left = table.column("img_0").to_pylist()
    right = table.column("img_1").to_pylist()
    rows = [(a["bytes"], b["bytes"], bool(y)) for a, b, y in zip(left, right, labels, strict=True)]
    if limit is None:
        return rows
    # Interleaved, never the first N. The file is grouped by label, so a plain slice is
    # 100% same-person pairs — a smoke run against which any threshold scores perfectly
    # and no false-match can exist.
    same = [r for r in rows if r[2]]
    different = [r for r in rows if not r[2]]
    half = limit // 2
    return same[:half] + different[: limit - half]


async def _run_pairs(args: argparse.Namespace, pipeline: InsightFacePipeline) -> None:
    rows = _load_pairs(args.pairs, args.limit)
    same = sum(1 for _, _, y in rows if y)
    print(f"LFW pairs: {len(rows)} ({same} same-person, {len(rows) - same} different)")

    scores = await score_pairs(pipeline, rows)
    undetected = sum(1 for s in scores if s.similarity is None)
    policy = AuthorizationPolicy()

    print(f"\nat the shipped threshold, match_threshold = {policy.match_threshold}")
    print(
        format_confusion(
            confusion_at(scores, policy.match_threshold),
            positive="same person",
            negative="different people",
        )
    )
    if undetected:
        print(f"\n  {undetected} pair(s) had no detectable face and counted as no-match")

    genuine = [s.similarity for s in scores if s.same_person and s.similarity is not None]
    impostor = [s.similarity for s in scores if not s.same_person and s.similarity is not None]
    if genuine and impostor:
        print(
            f"\n  same-person similarity   median {statistics.median(genuine):.3f}"
            f"   5th pct {np.percentile(genuine, 5):.3f}"
        )
        print(
            f"  different-person         median {statistics.median(impostor):.3f}"
            f"  95th pct {np.percentile(impostor, 95):.3f}"
        )

    if args.tune:
        threshold, accuracy = best_threshold(scores)
        print(
            f"\n  best threshold on THIS split: {threshold:.3f} ({accuracy:.1%} accuracy)."
            "\n  Choose it on one split and report it on another; the table above is the"
            "\n  shipped value, reported without tuning."
        )


async def _run_fairness(args: argparse.Namespace, pipeline: InsightFacePipeline) -> None:
    from datasets import load_dataset

    policy = AuthorizationPolicy()
    # "1.25" is the padding=1.25 crop — more context around the face than "0.25", and
    # closer to what a detector sees on a real frame than a tight crop would be.
    data = load_dataset("HuggingFaceM4/FairFace", "1.25", split=f"validation[:{args.fairness}]")
    groups: dict[str, list[tuple[bool, bool, Any]]] = defaultdict(list)
    race_names = data.features["race"].names

    for index, row in enumerate(data):
        pixels = np.asarray(row["image"].convert("RGB"))[:, :, ::-1].copy()
        faces = await pipeline.detect(_as_frame(pixels, index))
        face = _largest(faces)
        group = race_names[row["race"]]
        if face is None:
            groups[group].append((False, False, None))
            continue
        (embedding,) = await pipeline.embed((face,))
        groups[group].append((True, _passes_quality(face.quality, policy), embedding))

    print(f"\nFairFace, {sum(len(v) for v in groups.values())} images, by labelled race")
    print(f"  {'group':<20} {'n':>5} {'detected':>9} {'passes bars':>12} {'impostor p99':>13}")
    rng = np.random.default_rng(0)
    for group in sorted(groups):
        rows = groups[group]
        embeddings = [e for _, ok, e in rows if ok and e is not None]
        detected = sum(1 for d, _, _ in rows if d)
        passed = sum(1 for _, ok, _ in rows if ok)
        impostor_p99 = float("nan")
        if len(embeddings) >= 20:
            sims = []
            for _ in range(2000):
                i, j = rng.choice(len(embeddings), size=2, replace=False)
                sims.append(cosine_similarity(embeddings[i].vector, embeddings[j].vector))
            impostor_p99 = float(np.percentile(sims, 99))
        print(
            f"  {group:<20} {len(rows):5d} {detected / len(rows):8.1%} "
            f"{passed / len(rows):11.1%} {impostor_p99:13.3f}"
        )
    print(
        f"\n  Every pair above is two *different* people, so the last column is how close"
        f"\n  near-misses get within a group. The shipped threshold is"
        f" {policy.match_threshold}: a group"
        f"\n  whose p99 approaches it has a higher false-match rate at that fixed value."
        f"\n  This is not verification accuracy by group — FairFace has no identity"
        f"\n  labels, so same-person pairs cannot be built from it."
    )


async def _main(args: argparse.Namespace) -> int:
    if args.pairs is None and args.fairness is None:
        print("nothing to do: pass --pairs and/or --fairness", file=sys.stderr)
        return 2
    pipeline = InsightFacePipeline(device=args.device)
    await pipeline.initialize()
    await pipeline.warmup()
    print(f"{pipeline.version()}\n")
    try:
        if args.pairs is not None:
            await _run_pairs(args, pipeline)
        if args.fairness is not None:
            await _run_fairness(args, pipeline)
    finally:
        await pipeline.shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sentinel_ai.validation.faces",
        description="Score the face pipeline for accuracy (LFW) and even-handedness (FairFace).",
    )
    parser.add_argument("--pairs", type=Path, default=None, help="an LFW pairs parquet")
    parser.add_argument("--limit", type=int, default=None, help="pairs to score, for a smoke run")
    parser.add_argument(
        "--fairness", type=int, default=None, help="FairFace validation images to score"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--tune", action="store_true", help="also print the best threshold on this split"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
