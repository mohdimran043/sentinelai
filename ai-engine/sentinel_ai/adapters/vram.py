"""How much VRAM one model actually costs.

The bug this module exists to fix
----------------------------------
Every model adapter used to measure itself like this, in `warmup()`:

    reserved = torch.cuda.memory_reserved(device) // MiB
    self._vram_mib = reserved + _CUDA_CONTEXT_OVERHEAD_MIB

`torch.cuda.memory_reserved()` is **process-wide**. It is not "memory this model
reserved", it is "memory this process has reserved", so each model measured itself as
the cumulative total of everything loaded before it. With two models the error was easy
to miss — the detector loads first and measured roughly correctly, and the VLM's
over-report looked like the documented conservatism. With three it became obvious: a
live `/health` reported

    yolo11s.pt        452 MiB
    Qwen2.5-VL-3B    2816 MiB
    yolo11n-pose.pt  2816 MiB   <- measured at 462 MiB standalone

The pose model, which costs about 460 MiB on its own, claimed the VLM's footprint
because it warmed up afterwards.

This matters beyond a cosmetic number. `domain/policy/vram_budget.plan_residency()`
does hard admission arithmetic against `capabilities().vram_mib`: a model that
over-reports by 2.3 GiB makes the planner evict things it did not need to evict, and on
a smaller card makes it raise `InsufficientVram` for a set that would have fitted.

The fix, and the second bug inside it
--------------------------------------
Measure the **delta**: sample before the model loads anything, again after its warmup
pass, and subtract. That is this model's contribution and nobody else's.

Measuring that delta on `memory_reserved()` alone is not enough, and the live engine
said so immediately — the pose model, loading third, reported **0 MiB**. The caching
allocator had enough slack freed by the VLM's warmup activations to place an entire
model without ever asking the driver for more, so the reserved pool did not grow.

Zero is a far worse answer than the over-count it replaced. An over-reporting model
makes `plan_residency()` evict something it did not need to; an under-reporting one
makes it **admit a set that does not fit**, which is an OOM mid-escalation.

So three readings are taken and the largest delta wins:

* `memory_allocated()` tracks live tensors — the weights, precisely, and nothing else.
  It is immune to allocator slack, and it is the absolute floor.
* `memory_reserved()` is what the driver actually handed over, so it captures workspace
  the allocator had to go and get. The better number when it grows at all.
* `max_memory_allocated()`, with the peak counter **reset at the start of this model's
  load**, is the high-water mark this model actually drove — weights *plus* the
  activations its warmup pass needed. It is the only one of the three that sees
  transient workspace, and it is usually the largest.

The third was not there at first, and the gap showed: with only the first two, the pose
model reported **11 MiB** — its weights exactly, and nothing for the workspace a forward
pass needs. Against a standalone measurement of 462 MiB that is an under-report of
40x, in the direction that makes `plan_residency()` admit a set that does not fit.

Resetting the process-wide peak counter is safe here because `ResidentSet.ensure()`
loads models **sequentially**; two concurrent loads would clobber each other's
measurement, and if that ever changes this is the code that has to change with it.

The CUDA context — created on first kernel launch, roughly 150-300 MiB, held for the
life of the process — is then added to **whichever model created it**, identified as
the one that found an empty pool. Adding it to every model would re-introduce a smaller
version of the same over-count (three models, three contexts, one real). Attributing it
to the first loader is not strictly true either: the context belongs to the process, and
if that model is later evicted its context does not go with it. But the planner needs
every byte accounted to *something* or it will admit a set that does not fit, and the
first loader is the least wrong owner available — it is the one whose load the context
actually accompanied, and in this engine it is the always-resident detector, which is
never evicted anyway.

The estimate stays deliberately generous for `adapters/detectors/yolo11.py`'s stated
reason: the planner evicting one model too early is recoverable, and an OOM
mid-escalation is not.

What this still does not capture
---------------------------------
Measured on an RTX 4090 with all three models resident, `/health` reports 452 + 2364 +
11 = **2827 MiB**, while `nvidia-smi` attributes roughly **4.1 GiB** to the process.
The ~1.2 GiB difference is the CUDA context (larger on this driver than the 300 MiB
charged above), cuDNN/cuBLAS workspaces, and allocator fragmentation — none of which
any torch-level API attributes to a model.

Two consequences worth being precise about:

* **The per-model figures are now marginal costs, and that is the right thing for
  them to be.** Pose's 11 MiB is not wrong: loading it alongside an already-resident
  VLM genuinely costs about that, because it reuses slack the allocator already holds
  and pays nothing toward a context that exists. The 462 MiB it measures *standalone*
  is mostly the context, which in a real deployment the detector has already paid.
  `plan_residency()` asks "can I fit this model as well", so marginal is the question
  it is asking.
* **The process total is still under-reported**, and `SENTINEL_VRAM_RESERVED_MIB`
  (2048 by default) is what covers the gap. That headroom is not a rounding allowance;
  it is load-bearing, and lowering it on the assumption that these figures are complete
  is how a deployment OOMs. See `docs/operations.md`.
"""

from __future__ import annotations

_MIB = 1024 * 1024

CUDA_CONTEXT_OVERHEAD_MIB = 300
"""Charged once, to the model that created the context.

A documented over-estimate rather than a measured value: the CUDA context and the
cuDNN/cuBLAS workspace buffers are real and visible to `nvidia-smi`, but no torch-level
API surfaces the context's true size. Measured on an RTX 4060 with YOLO11s at
imgsz=640: `memory_allocated()` 68 MiB, `memory_reserved()` 132 MiB, real `nvidia-smi`
delta ~281 MiB — so even `memory_reserved()` alone undershoots by ~150 MiB, which is
what this covers.
"""


def reserved_mib(device: str | None = None) -> int:
    """The process's reserved pool and live-tensor total, in MiB, as one reading.

    Returns `(reserved, allocated)` collapsed into the reserved figure for callers that
    only want the headline; `sample_pools` is the one that returns both. Kept because
    `warmup()` call sites and tests read it directly, and because 0 is the truthful
    answer on a box with no VRAM rather than something to raise about.
    """
    return sample_pools(device)[0]


def _cuda_device(device: str | None) -> str | None:
    """`device` if it names a CUDA device, else `None` meaning "do not ask CUDA".

    `Settings.device` can legitimately be `"cpu"` on a box that *has* a visible GPU —
    forcing CPU is a supported configuration — and `torch.cuda.memory_reserved("cpu")`
    raises `ValueError` rather than answering zero. Guarding on
    `torch.cuda.is_available()` alone is therefore not enough, which a test on exactly
    that configuration found.
    """
    if device is None:
        return None
    return device if device.startswith("cuda") else _NOT_CUDA


_NOT_CUDA = "\0not-cuda"
"""Sentinel distinguishing "no device specified, ask about the default" (`None`) from
"a device that is explicitly not CUDA". A plain `None` for both would make `device=cpu`
report the default GPU's memory as though it were this model's."""


def sample_pools(device: str | None = None) -> tuple[int, int]:
    """`(reserved_mib, allocated_mib)`, or `(0, 0)` without CUDA.

    Both, because neither alone is trustworthy — see the module docstring. `allocated`
    is live tensors (weights); `reserved` is what the driver handed this process.

    **Also resets the peak-allocated counter**, so that `measure_model_vram_mib` can
    read the high-water mark this model drives rather than one an earlier model left
    behind. Sampling and resetting belong together precisely because forgetting the
    reset is invisible: the measurement still returns a plausible number, just one
    about the wrong model.
    """
    try:
        import torch
    except ImportError:
        return (0, 0)
    resolved = _cuda_device(device)
    if not torch.cuda.is_available() or resolved is _NOT_CUDA:
        return (0, 0)
    torch.cuda.reset_peak_memory_stats(resolved)
    return (
        int(torch.cuda.memory_reserved(resolved) // _MIB),
        int(torch.cuda.memory_allocated(resolved) // _MIB),
    )


def peak_allocated_mib(device: str | None = None) -> int:
    """The high-water mark since the last reset, in MiB, or 0 without CUDA."""
    try:
        import torch
    except ImportError:
        return 0
    resolved = _cuda_device(device)
    if not torch.cuda.is_available() or resolved is _NOT_CUDA:
        return 0
    return int(torch.cuda.max_memory_allocated(resolved) // _MIB)


def measure_model_vram_mib(
    baseline_mib: int, device: str | None = None, allocated_baseline_mib: int = 0
) -> int:
    """This model's own footprint: the growth in the reserved pool since `baseline_mib`.

    `baseline_mib` is what `reserved_mib()` returned **before this model loaded
    anything** — captured at the top of `initialize()`, not in `warmup()`, because the
    weights are placed by the former and the activations by the latter, and both are
    this model's cost.

    A baseline of 0 means this model found an empty pool and therefore created the CUDA
    context, so the context overhead is charged here and to no one else. See the module
    docstring for why that attribution is the least wrong one available.

    Clamped at zero: a delta can come out negative when another model was evicted
    between the two samples and its memory returned to the allocator, and a negative
    VRAM figure would make `plan_residency()` believe a model frees memory by existing.
    """
    reserved_now, allocated_now = sample_pools(device)
    if reserved_now == 0:
        # No CUDA at all. Not "a model that costs nothing" — there is nothing to cost.
        return 0
    # The larger of the two deltas. Reserved can be 0 for a model placed entirely
    # inside slack the allocator already held; allocated cannot, because the weights
    # are live tensors either way. See the module docstring.
    delta = max(
        0,
        reserved_now - baseline_mib,
        allocated_now - allocated_baseline_mib,
        # The peak this model drove during its own load and warmup, against the live
        # total it started from. The only term that sees transient workspace.
        peak_allocated_mib(device) - allocated_baseline_mib,
    )
    context = CUDA_CONTEXT_OVERHEAD_MIB if baseline_mib == 0 else 0
    return delta + context


def device_used_mib(device: str | None = None) -> int:
    """VRAM the **driver** reports as in use on the device, in MiB, or 0 without CUDA.

    Everything above this function measures torch's caching allocator, which is the
    right instrument for a torch model and blind to everything else. ONNX Runtime —
    which `adapters/face/insightface_pipeline.py` runs under, deliberately, so the face
    models are not a second torch graph — allocates through CUDA directly. Measured
    with `sample_pools`, a face pipeline holding 608 MiB of weights and workspace
    reports **0 MiB**, which is precisely the failure this module's docstring calls
    worse than the over-count it replaced: `plan_residency()` admits a model set that
    does not fit and the first OOM lands somewhere unrelated.

    `torch.cuda.mem_get_info` asks the driver, so it sees every allocator on the card.
    """
    resolved = _cuda_device(device)
    if resolved is _NOT_CUDA:
        return 0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, total = torch.cuda.mem_get_info(resolved)
    except (ImportError, RuntimeError, ValueError, AssertionError):
        return 0
    return int(total - free) // _MIB


def measure_foreign_vram_mib(baseline_mib: int, device: str | None = None) -> int:
    """A non-torch model's footprint: the growth in **device-wide** use since a baseline.

    `baseline_mib` is what `device_used_mib()` returned before this model loaded
    anything.

    **Device-wide is the compromise, and it is a real one.** This reading cannot
    attribute memory to a process, so anything else that allocates on the same card
    between the two samples is charged to this model: another process, or a torch model
    warming up concurrently in this one. The engine's own composition root loads models
    sequentially, so the second case does not arise in practice, but a second engine
    sharing a GPU would inflate this number.

    That was still the right trade. The alternative reading is a guaranteed 0 for every
    ONNX model, and an over-report makes `plan_residency()` cautious while an
    under-report makes it wrong. Clamped at zero for the same reason as
    `measure_model_vram_mib`: a model cannot free memory by existing.
    """
    now = device_used_mib(device)
    if now == 0:
        return 0
    return max(0, now - baseline_mib)
