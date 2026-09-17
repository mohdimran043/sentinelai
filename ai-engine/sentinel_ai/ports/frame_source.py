"""Video input port. Every source (RTSP, file, USB) reduces to this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass


class DeferredPixels:
    """A frame's pixels, converted on first use and then remembered.

    Decoding H.264 and turning the result into a BGR array are two costs, and only the
    first is unavoidable. Measured on a 1080p clip: decoding alone runs at 580 fps and
    one core; decoding *and* converting every frame runs at 334 fps and 1.51 cores. The
    conversion is a third of the work.

    Most of that third is wasted. Frames land in `_LatestSlot`, a single-slot mailbox
    that overwrites, so a camera whose consumer is busy drops almost everything it
    decodes — 90% at twenty cameras, under measurement. Converting a frame and then
    discarding it is the single largest piece of avoidable work in the pipeline, and
    converting one frame in ten instead of all of them measured **1.51 -> 1.09 cores**
    on the same clip.

    It cannot be fixed by dropping earlier. The mailbox keeps the *newest* frame, so at
    the moment the pump has a frame in hand there is no way to know whether the next one
    will arrive before the consumer wakes. The only thing that can know is the consumer,
    and by then the conversion has already happened — unless it is deferred to here.

    **Not thread-safe, deliberately.** One camera's frame is resolved by one task: the
    runner materialises it before handing it to any model, and a keyframe kept for a
    later VLM call is already resolved by then. A lock would cost every frame to protect
    against a caller that does not exist.

    The converter is released after the first call. It closes over the decoder's own
    `VideoFrame`, and a retained keyframe would otherwise pin a decode buffer for as long
    as the escalation takes.
    """

    __slots__ = ("_convert", "_value")

    def __init__(self, convert: Callable[[], object]) -> None:
        self._convert: Callable[[], object] | None = convert
        self._value: object = None

    def get(self) -> object:
        convert = self._convert
        if convert is not None:
            self._value = convert()
            self._convert = None
        return self._value

    @property
    def resolved(self) -> bool:
        """Whether the conversion has already been paid for. For tests and telemetry."""
        return self._convert is None


@dataclass(frozen=True, slots=True)
class FrameData:
    """A decoded frame.

    `pixels` is typed `object` deliberately: it carries a numpy array at
    runtime, but typing it as such would drag numpy into the ports layer and
    the architecture fitness test would (correctly) fail the build.

    It may also carry a `DeferredPixels`, which is what a real source produces.
    **Read it through `pixel_array()`, never directly** — a direct read is how a
    consumer ends up handing a `DeferredPixels` to numpy. The five places in this
    codebase that need pixels all do, and each still type-checks the result.
    """

    camera_id: str
    frame_index: int
    timestamp: float
    width: int
    height: int
    pixels: object

    def pixel_array(self) -> object:
        """The pixels, converting them now if nobody has yet.

        Accepts an eagerly-built frame unchanged, which is what keeps a test able to
        write `FrameData(..., pixels=np.zeros(...))` without knowing any of this exists.
        """
        pixels = self.pixels
        return pixels.get() if isinstance(pixels, DeferredPixels) else pixels


@dataclass(frozen=True, slots=True)
class EncodedPacket:
    """A demuxed, still-encoded packet. `data` is bytes so ports stay codec-agnostic.

    `pts` is float seconds on the same monotonic timeline as `FrameData.timestamp` — the
    pre-roll buffer and the clip writer line packets and frames up on it.
    """

    camera_id: str
    data: bytes
    pts: float
    is_keyframe: bool
    codec: str
    """Container short codec name, lowercase — "h264", "hevc".

    A plain `str`, deliberately: the clip writer needs to know what it is remuxing, but a
    library enum here would drag a codec dependency into the pure ports layer and the
    architecture fitness test would reject it. Without this field the clip writer has to
    *assume* H.264, and a non-H.264 source produces an empty clip instead of an error.
    """


class FrameSource(ABC):
    @abstractmethod
    def __aiter__(self) -> AsyncIterator[FrameData]: ...

    @abstractmethod
    def packets(self) -> AsyncIterator[EncodedPacket]:
        """The same demux pass as `__aiter__`, fanned out as still-encoded packets.

        Feeds the pre-roll buffer and the clip writer (spec §4.2). Every demuxed packet is
        offered here even though only the sampled ones are decoded into frames.
        """

    @abstractmethod
    async def close(self) -> None: ...
