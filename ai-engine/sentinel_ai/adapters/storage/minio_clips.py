"""MinIO evidence clip writer (spec §5.5). A remux, never a re-encode: the
packets that arrive are already H.264, the pre-roll buffer already holds them
encoded, and the output is an MP4 container around the same bytes — so there
is no decode and no encode anywhere in this path, and the evidence stays
bit-identical to what the camera sent.

The crux this module resolves: `ClipHandle.open()` receives only
`camera_id`, `event_id` and `fps` (S4); `append()` receives an
`EncodedPacket` — bytes, a pts, a keyframe flag and a codec name (S2).
Nothing carries width, height, or the SPS/PPS an MP4 muxer needs to write a
valid `stsd` box, because the port is deliberately codec-agnostic beyond
that one string. Those are recovered from the bytes themselves in two
separate steps, both driven by the codec name:

* Width/height come from `av.CodecContext.create(codec, "r")` — an
  elementary-stream *parser*, not a decoder. Feeding it each packet's raw
  bytes via `.parse()` reassembles NAL units into access units and, once it
  has seen a keyframe's SPS, populates `.width`/`.height`.
* Extradata (the SPS/PPS pair the muxer's `avcC`/`hvcC` box needs) comes
  from a separate `extract_extradata` bitstream filter run over the same
  bytes. `CodecContext.parse()` alone does not populate `.extradata` for
  H.264/HEVC in PyAV: that field mirrors a decoder's parsed configuration
  record, and a parser context never opens a decoder. `extract_extradata`
  is ffmpeg's own filter for exactly this: it scans Annex-B packets for
  in-band parameter sets and attaches them to the packet as `new_extradata`
  side data — the same mechanism ffmpeg's own mov muxer relies on when you
  run `ffmpeg -i rtsp://camera -c copy clip.mp4`, since that muxer detects
  extradata that is not already in AVCC/HVCC form (first byte != 1) and
  converts it — and the Annex-B packets alongside it — on the fly while
  writing the `stsd` box. Passing the extracted (still Annex-B-shaped)
  extradata straight through to `codec_context.extradata` relies on that
  same auto-conversion, which is exactly why no manual avcC construction is
  needed here.

H.264 over RTSP/RTP (RFC 6184, the deployment source in scope) carries
SPS/PPS in-band on every keyframe, unlike an MP4 file's out-of-band `avcC`
record — which is why both of the above see a complete parameter set on the
very first keyframe, rather than needing to wait for one to be assembled
across several packets.

PTS/DTS rebasing: `CodecContext.parse()` buffers by exactly one access unit
(H.264/HEVC parsers reorder-detect by looking for the *next* access unit's
start), so the access unit it yields on a given call was assembled from an
*earlier* packet, not the one just fed. A FIFO of the fed packets' `pts`
values, popped once per yielded access unit, re-pairs each output access
unit with the input packet it actually came from. The first popped `pts`
becomes the clip's own zero point; every later timestamp is expressed as
`(packet.pts - start_pts)` in the output's timebase, so the written MP4
always starts at pts 0 regardless of where the camera's monotonic clock
happened to be — which is what lets a player seek and start the clip
immediately instead of showing a long presentation delay.

`SUPPORTED_CODECS` gates this explicitly. A codec outside it raises
`UnsupportedCodecError` on the very first packet rather than parsing
forever and finalising an empty MP4 — a clip writer that fails silently is
worse than one that fails, because the operator only discovers it when they
go looking for evidence that was never written. A mid-clip codec change
(a camera renegotiating its stream) is rejected the same way: one `_RemuxSession`
can only ever back one MP4 video track.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import av
from minio import Minio

from sentinel_ai.ports.clip_writer import ClipHandle, ClipWriter
from sentinel_ai.ports.frame_source import EncodedPacket

logger = logging.getLogger(__name__)

_OUTPUT_TIME_BASE = Fraction(1, 90_000)
"""A conventional high-resolution MP4 video timebase, independent of fps —
gives sub-frame pts precision regardless of the source framerate."""

SUPPORTED_CODECS: frozenset[str] = frozenset({"h264", "hevc"})
"""Codecs that can be remuxed straight into MP4 and that carry parameter
sets in-band. Everything mediamtx emits and every RTSP camera in scope is
one of these."""


class UnsupportedCodecError(RuntimeError):
    """Raised when a source delivers a codec the clip writer cannot remux."""


def _create_parser(codec: str) -> av.VideoCodecContext:
    """`av.CodecContext.create` is overloaded on a `Literal` codec-name type;
    dispatching on the two supported literals (rather than the caller's
    plain `str`) is what lets the return type be `VideoCodecContext` (with
    `.width`/`.height`) instead of the codec-agnostic base `CodecContext`.
    """
    if codec == "h264":
        return av.CodecContext.create("h264", "r")
    if codec == "hevc":
        return av.CodecContext.create("hevc", "r")
    raise UnsupportedCodecError(
        f"cannot remux codec {codec!r}; supported: {sorted(SUPPORTED_CODECS)}"
    )


class _RemuxSession:
    """Copies `EncodedPacket` bytes into a local MP4 file. No MinIO, no
    network — isolated from `MinioClipHandle` so the remux, the part with
    real logic, is unit-testable without a running MinIO server.
    """

    def __init__(self, temp_path: Path, fps: float) -> None:
        self._temp_path = temp_path
        self._fps = fps
        self._container = av.open(str(temp_path), mode="w")
        self._codec: str | None = None
        self._parser: av.VideoCodecContext | None = None
        self._extradata_bsf: av.BitStreamFilterContext | None = None
        self._extradata: bytes | None = None
        self._out_stream: av.VideoStream | None = None
        self._pts_queue: deque[float] = deque()
        self._start_pts: float | None = None
        self._closed = False

    @property
    def temp_path(self) -> Path:
        return self._temp_path

    def append(self, packet: EncodedPacket) -> None:
        if self._closed:
            return

        if self._codec is None:
            # Checked before the keyframe gate below: an unsupported codec must raise
            # on the very first packet, even if that packet happens not to be a
            # keyframe, rather than being silently dropped forever.
            if packet.codec not in SUPPORTED_CODECS:
                raise UnsupportedCodecError(
                    f"cannot remux codec {packet.codec!r}; supported: {sorted(SUPPORTED_CODECS)}"
                )
            if not packet.is_keyframe:
                # Cannot start a clip mid-GOP. `PreRollBuffer.flush()` guarantees the
                # first packet handed to a real `ClipHandle` is a keyframe, so this only
                # guards a caller that violates that contract.
                logger.debug("dropping packet before the first keyframe")
                return
            self._codec = packet.codec
            self._parser = _create_parser(packet.codec)
            self._extradata_bsf = av.BitStreamFilterContext("extract_extradata", packet.codec)
        elif packet.codec != self._codec:
            # A mid-clip codec change (e.g. a camera renegotiating its stream) cannot be
            # remuxed into the one MP4 track already opened for `self._codec`.
            raise UnsupportedCodecError(
                f"codec changed mid-clip: {self._codec!r} -> {packet.codec!r}"
            )

        assert self._parser is not None
        assert self._extradata_bsf is not None

        self._pts_queue.append(packet.pts)

        if self._extradata is None:
            probe = av.Packet(packet.data)
            probe.is_keyframe = packet.is_keyframe
            for extracted in self._extradata_bsf.filter(probe):
                side_data = extracted.get_sidedata("new_extradata")
                if side_data:
                    self._extradata = bytes(side_data)

        for parsed in self._parser.parse(packet.data):
            self._mux(parsed)

    def _add_output_stream(self, codec: str, rate: int) -> av.VideoStream:
        """Same literal-dispatch trick as `_create_parser`: `add_stream` is
        overloaded on a `Literal` codec name, and only that overload returns
        `VideoStream` (with a typed `.codec_context`) rather than the
        `VideoStream | AudioStream | SubtitleStream` union the generic
        `str` overload returns."""
        if codec == "h264":
            return self._container.add_stream("h264", rate=rate)
        if codec == "hevc":
            return self._container.add_stream("hevc", rate=rate)
        raise UnsupportedCodecError(f"cannot remux codec {codec!r}")  # unreachable: pre-validated

    def _mux(self, parsed: av.Packet[av.VideoStream]) -> None:
        assert self._parser is not None
        assert self._codec is not None
        au_pts = self._pts_queue.popleft()

        if self._out_stream is None:
            if not (self._parser.width and self._parser.height and self._extradata):
                return  # SPS/PPS not fully resolved yet; drop this access unit
            out_stream = self._add_output_stream(self._codec, max(1, round(self._fps)))
            out_stream.codec_context.width = self._parser.width
            out_stream.codec_context.height = self._parser.height
            out_stream.codec_context.extradata = self._extradata
            out_stream.codec_context.pix_fmt = "yuv420p"
            out_stream.time_base = _OUTPUT_TIME_BASE
            self._out_stream = out_stream
            self._start_pts = au_pts

        assert self._start_pts is not None
        # PTS/DTS rebasing: the clip's own clock starts at 0 regardless of where the
        # camera's monotonic pts happened to be.
        ticks = round((au_pts - self._start_pts) / float(_OUTPUT_TIME_BASE))
        parsed.pts = ticks
        parsed.dts = ticks  # no B-frames on the low-latency IPPP profiles in scope
        parsed.time_base = _OUTPUT_TIME_BASE
        parsed.stream = self._out_stream
        self._container.mux(parsed)

    def finish(self) -> Path:
        if self._closed:
            return self._temp_path
        if self._parser is not None:
            for parsed in self._parser.parse(None):  # flush the parser's trailing access unit
                self._mux(parsed)
        self._container.close()
        self._closed = True
        return self._temp_path

    def abort(self) -> None:
        """Discard a partial clip. Must not raise — this runs on shutdown paths."""
        if self._closed:
            return
        self._closed = True
        try:
            self._container.close()
        except Exception:
            logger.warning("error closing partial clip %s", self._temp_path, exc_info=True)
        try:
            self._temp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to remove temp clip file %s", self._temp_path, exc_info=True)


class MinioClipHandle(ClipHandle):
    def __init__(
        self, session: _RemuxSession, client: Minio, bucket: str, camera_id: str, event_id: UUID
    ) -> None:
        self._session = session
        self._client = client
        self._bucket = bucket
        self._camera_id = camera_id
        self._event_id = event_id
        self._done = False

    async def append(self, packet: EncodedPacket) -> None:
        await asyncio.to_thread(self._session.append, packet)

    async def finish(self) -> str:
        if self._done:
            raise RuntimeError("clip handle already finished or aborted")
        self._done = True
        temp_path = await asyncio.to_thread(self._session.finish)
        object_name = f"{self._camera_id}/{self._event_id}.mp4"
        try:
            await asyncio.to_thread(
                self._client.fput_object, self._bucket, object_name, str(temp_path)
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return f"s3://{self._bucket}/{object_name}"

    async def abort(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            await asyncio.to_thread(self._session.abort)
        except Exception:
            logger.warning(
                "error aborting clip %s/%s", self._camera_id, self._event_id, exc_info=True
            )


class MinioClipWriter(ClipWriter):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        temp_dir: Path,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._temp_dir = temp_dir
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._bucket_ready = False

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle:
        if not self._bucket_ready:
            await asyncio.to_thread(self._ensure_bucket)
            self._bucket_ready = True
        temp_path = self._temp_dir / f"{camera_id}-{event_id}.mp4"
        session = await asyncio.to_thread(_RemuxSession, temp_path, fps)
        return MinioClipHandle(session, self._client, self._bucket, camera_id, event_id)

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
