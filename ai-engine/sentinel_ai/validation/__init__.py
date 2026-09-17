"""Does it work, rather than what does it cost.

`sentinel_ai/benchmark/` answers the performance question: latency, VRAM, camera count.
It says nothing at all about whether a detector is *right*, and for two of this engine's
capabilities that was the honest gap in the documentation — a fall detector that had
never seen a real fall, and a face pipeline whose accuracy on any population was
unmeasured.

This package closes it, on public datasets, with the real models. Every number it
produces is reproducible by anybody with the same clips:

    python -m sentinel_ai.validation.falls  --dataset ../datasets/urfd
    python -m sentinel_ai.validation.faces  --pairs   lfw_pairs_test.parquet

What it deliberately does *not* do is turn into a training loop. Nothing here tunes a
threshold against the set it reports on, and where a threshold is chosen from data the
module says which split it came from.
"""
