"""Hardware-accelerated decode, and an honest account of when it helps (ADR 2).

`SENTINEL_DECODE_HWACCEL` used to be inert — a documented placeholder. This module is
what it now drives, and the measurements that justify its shape are worth stating
because they contradict the obvious expectation.

Measured on an RTX 4090, one 1080p H.264 clip, PyAV 18.1:

| path                                   |   fps | CPU cores |
|----------------------------------------|------:|----------:|
| software decode only                   |   580 |      1.00 |
| **CUDA decode only**                   | **801** |  **0.26** |
| software decode + BGR conversion       |   336 |      1.60 |
| CUDA decode + BGR conversion           |   260 |      1.84 |

The decode itself gets nearly four times cheaper on CPU. The *pipeline* gets more
expensive, because a decoded frame has to reach a numpy array in host memory, and
pulling an NV12 surface back off the GPU and converting it there costs more than
decoding straight to YUV420P in host memory ever did.

So this is off by default and will stay off until a frame can go from NVDEC into the
detector without a round trip through host memory — which is a change to what
`FrameData` carries, not a configuration flag. What it is genuinely for today is the
case where **CPU, not total throughput, is the binding resource**: a box with many
cameras and few cores, where 0.26 cores of decode against 1.00 buys camera count even
though each frame costs more to deliver. `docs/performance.md` has the numbers to
decide with.

Never silently: an unavailable device type raises at source construction rather than
falling back, because a deployment that asked for hardware decode and quietly got
software is a deployment whose capacity planning is wrong and whose logs do not say so.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["available_hwaccel_devices", "build_hwaccel"]


def available_hwaccel_devices() -> tuple[str, ...]:
    """Device types this FFmpeg build can open, e.g. `("cuda", "qsv", "drm", "amf")`."""
    from av.codec.hwaccel import hwdevices_available

    return tuple(sorted(hwdevices_available()))


def build_hwaccel(device_type: str | None) -> Any | None:
    """An `av.codec.hwaccel.HWAccel` for `device_type`, or `None` for software decode.

    Raises `ValueError` if the name is not a device type this FFmpeg build can open.
    The error lists what is available, because the usual cause is a correct intention
    and a build without the right support compiled in — and "cuda is not in ('drm',)"
    is a much faster diagnosis than a decode that is merely slower than expected.

    `allow_software_fallback=False` for the same reason. FFmpeg will happily fall back
    per stream, which produces a camera that is quietly on the CPU path while `/health`
    and the configuration both say otherwise.
    """
    if device_type is None:
        return None

    from av.codec.hwaccel import HWAccel

    available = available_hwaccel_devices()
    if device_type not in available:
        raise ValueError(
            f"SENTINEL_DECODE_HWACCEL={device_type!r} is not available in this FFmpeg "
            f"build; it offers {available or '()'}. Leave it unset for software decode."
        )
    logger.info(
        "hardware decode enabled (%s). Decode costs about a quarter of the CPU and "
        "each delivered frame costs more — see docs/performance.md before assuming "
        "this raises throughput.",
        device_type,
    )
    return HWAccel(device_type=device_type, allow_software_fallback=False)
