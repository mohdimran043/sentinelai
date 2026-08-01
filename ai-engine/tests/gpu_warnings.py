"""The one warning this suite tolerates, and the reasoning for it.

`pyproject.toml` sets `filterwarnings = ["error"]`, so every warning anywhere in the
suite fails a test. One third-party warning genuinely cannot be fixed from here:
loading `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit NF4 makes bitsandbytes report, per
matmul, that a weight matrix's inner dimension is not a multiple of its kernel block
size, so it falls back to a slower path:

    UserWarning: inner dimension (3420) is not aligned for fast kernel with
    blocksize=64, falling back to slower implementation.
      bitsandbytes/backends/cuda/ops.py:944

It is emitted by a dependency, about that dependency's own performance, and is
determined entirely by the checkpoint's layer shapes — nothing in this repo can
change 3420 or 64. Treating it as an error makes the three GPU tests unrunnable
without buying anything.

The exemption is scoped as narrowly as it can be:

  * a `@pytest.mark.filterwarnings` on the individual tests that load the VLM, not a
    `filterwarnings` entry in `pyproject.toml`, so it cannot silence anything in the
    ~430 CPU tests CI actually runs;
  * matched on the message text, so any *other* `UserWarning` from bitsandbytes — a
    deprecation, a misconfiguration, a fallback with a different cause — still fails;
  * `UserWarning` only.

Anything wider would have re-opened the hole this exists inside the fix for.
"""

from __future__ import annotations

BITSANDBYTES_UNALIGNED_KERNEL = (
    "ignore:.*is not aligned for fast kernel with blocksize.*:UserWarning"
)
"""Use as `@pytest.mark.filterwarnings(BITSANDBYTES_UNALIGNED_KERNEL)`."""
