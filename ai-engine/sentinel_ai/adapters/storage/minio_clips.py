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
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import UUID

import av
from minio import Minio
from minio.error import S3Error

from sentinel_ai.ports.clip_index import CameraStorage, ClipIndex, ClipRecord, StorageUsage
from sentinel_ai.ports.clip_reader import ClipReader
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
        self._last_ticks: int | None = None
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
        if self._last_ticks is not None and ticks <= self._last_ticks:
            # A non-increasing DTS makes ffmpeg's mov muxer return EINVAL, which reaches
            # here as `av.error.ArgumentError: ... returned 22` — the 1-in-5 clip loss.
            # It is not a rare race: on RTSP `au_pts` is packet *arrival* time
            # (`rtsp.py`), and a TCP-interleaved socket delivers a burst after any
            # stall, so two access units routinely land inside one 11.1 us tick and
            # `round()` maps them to the same integer. Arrival order regressing
            # outright does the same thing.
            #
            # Nudging to `last + 1` rather than dropping the access unit: one tick is
            # 1/90 000 s, so the displacement is three orders of magnitude below a frame
            # at any framerate in scope and invisible on playback, while dropping would
            # silently delete evidence — and a burst is exactly the moment something is
            # happening. It only ever moves a timestamp forward, so the rebased zero
            # point and the clip's overall span are untouched.
            ticks = self._last_ticks + 1
        self._last_ticks = ticks
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


def _restamped(packet: Any, out_stream: Any, index: int, rate: float, time_base: Any) -> Any:
    """Re-time a packet onto a clean zero-based timeline.

    The source's timestamps carry the camera's arrival clock, gaps and all. Copying them
    into a three-second excerpt produces a file whose player scrubber claims a length it
    does not have, and whose first frame sits minutes in. Renumbering is safe precisely
    because this is an excerpt and not the evidence: the evidence clip keeps the real
    timeline, and this one only has to play.
    """
    step = 1.0 / rate
    stamp = int(index * step / float(time_base)) if time_base else index
    packet.stream = out_stream
    packet.pts = stamp
    packet.dts = stamp
    return packet


def _trim(source: Path, target: Path, seconds: float) -> None:
    """Copy the first `seconds` of `source` into `target`, remuxing only.

    Packet-level copy: the encoded bytes are the camera's own, exactly as ADR 3 requires
    of the evidence clip, so the short copy is a true excerpt rather than a re-rendering
    of one.

    Cut on the presentation timestamp of the *input*, and stop at the first packet past
    the limit rather than trying to land exactly on it. An mp4 cannot start anywhere but
    a keyframe, and chasing an exact duration here would mean either re-encoding or
    emitting a clip that opens on a grey block.
    """
    import av

    # Counted in frames, not measured on the timeline, and the difference is the whole
    # reason this is not two lines.
    #
    # An evidence clip's timestamps are the camera's own arrival clock, and that clock
    # has holes in it: the pre-roll ring is flushed in one go, a reconnect resets the
    # source, a busy camera drops frames. "The first three seconds of the timeline" is
    # therefore not three seconds of footage — measured on a live run it was between
    # 0.4s and 1.6s of it, varying per clip, which is the worst possible behaviour for
    # something nobody watches twice.
    #
    # Frames are what a person actually sees, so frames are what gets counted. `fps`
    # comes from the stream itself rather than from the writer's estimate, because a
    # remuxed clip carries the rate the camera really delivered.
    with av.open(str(source)) as inbound, av.open(str(target), mode="w") as outbound:
        stream = inbound.streams.video[0]
        out_stream = outbound.add_stream_from_template(stream)
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        if rate <= 0:
            raise ValueError("clip declares no frame rate; cannot cut it by duration")
        wanted = max(1, int(seconds * rate))

        kept = 0
        for packet in inbound.demux(stream):
            if packet.dts is None:
                continue
            outbound.mux(_restamped(packet, out_stream, kept, rate, stream.time_base))
            kept += 1
            if kept >= wanted:
                break


class MinioClipHandle(ClipHandle):
    def __init__(
        self,
        session: _RemuxSession,
        client: Minio,
        bucket: str,
        camera_id: str,
        event_id: UUID,
        notify_seconds: float = 0.0,
    ) -> None:
        self._session = session
        self._client = client
        self._bucket = bucket
        self._camera_id = camera_id
        self._event_id = event_id
        self._notify_seconds = notify_seconds
        self._notify_uri: str | None = None
        self._done = False

    async def append(self, packet: EncodedPacket) -> None:
        await asyncio.to_thread(self._session.append, packet)

    async def finish(self) -> str:
        if self._done:
            raise RuntimeError("clip handle already finished or aborted")
        # `_done` is set only once the remux itself has actually completed. If
        # `self._session.finish()` raises (e.g. a mux() failure while flushing
        # trailing packets), the handle must NOT be marked done — otherwise a
        # caller's subsequent `abort()` would return immediately as a no-op,
        # leaking the still-open PyAV container and the temp file.
        temp_path = await asyncio.to_thread(self._session.finish)
        self._done = True
        object_name = f"{self._camera_id}/{self._event_id}.mp4"
        try:
            await asyncio.to_thread(
                self._client.fput_object, self._bucket, object_name, str(temp_path)
            )
        except Exception:
            # Spec §9: an anomaly event is never lost to an infrastructure failure.
            # A transient MinIO failure (network blip, disk full, timeout) must not
            # destroy the only copy of the evidence — retain the temp file and log
            # its path so an operator can recover it. Deleting it here would trade
            # "lost to MinIO" for "lost to us", which is strictly worse.
            logger.warning(
                "clip upload failed for camera %s event %s; retaining local file at %s "
                "for manual recovery",
                self._camera_id,
                self._event_id,
                temp_path,
                exc_info=True,
            )
            raise
        # Before the temp file goes: the short clip is cut from it, and once it is
        # deleted there is nothing left to cut. Best-effort — a notification without a
        # short clip still carries the full one, while a failed evidence upload is a
        # lost anomaly, so this must never be able to fail that.
        if self._notify_seconds > 0:
            self._notify_uri = await self._write_short(temp_path)

        # Only reached on a genuine upload success — the temp file's job is done.
        temp_path.unlink(missing_ok=True)
        return f"s3://{self._bucket}/{object_name}"

    @property
    def notify_uri(self) -> str | None:
        return self._notify_uri

    async def _write_short(self, source: Path) -> str | None:
        """The first `notify_seconds` of the finished clip, as a second object.

        A remux, never a re-encode — [ADR 3](../../../docs/decisions.md)'s rule applies
        here for the same reasons, and one of them is sharper: this copy is the one a
        person actually looks at on a phone, so degrading it would degrade the only
        version most notifications ever get watched as.

        Starts at the beginning rather than at the keyframe, which is the pre-roll: the
        seconds *before* the event are what make a notification legible — somebody
        walking in, then falling — and a clip that opens on the fall shows a person
        already on the floor.
        """
        target = source.with_name(f"{source.stem}-notify.mp4")
        object_name = f"{self._camera_id}/{self._event_id}-notify.mp4"
        try:
            await asyncio.to_thread(_trim, source, target, self._notify_seconds)
            await asyncio.to_thread(
                self._client.fput_object, self._bucket, object_name, str(target)
            )
        except Exception:
            logger.warning(
                "could not make a short notification clip for camera %s event %s; the "
                "notification will carry the full clip instead",
                self._camera_id,
                self._event_id,
                exc_info=True,
            )
            return None
        finally:
            target.unlink(missing_ok=True)
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
        retention_days: int = 0,
        notify_seconds: float = 0.0,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._temp_dir = temp_dir
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._notify_seconds = notify_seconds
        self._bucket_ready = False

    async def open(self, camera_id: str, event_id: UUID, fps: float) -> ClipHandle:
        if not self._bucket_ready:
            await asyncio.to_thread(self._ensure_bucket)
            self._bucket_ready = True
        temp_path = self._temp_dir / f"{camera_id}-{event_id}.mp4"
        session = await asyncio.to_thread(_RemuxSession, temp_path, fps)
        return MinioClipHandle(
            session,
            self._client,
            self._bucket,
            camera_id,
            event_id,
            notify_seconds=self._notify_seconds,
        )

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
        self._ensure_lifecycle()

    def _ensure_lifecycle(self) -> None:
        """Expire clips after `retention_days`, enforced by the object store itself.

        A lifecycle rule rather than a sweeper task in this process, because the two
        fail very differently. A sweeper deletes nothing while the engine is down, and
        an engine that has been down for a week comes back to a week of clips it was
        supposed to have expired. The rule is evaluated by MinIO whether or not anything
        is running.

        `retention_days <= 0` removes the rule and keeps clips forever, which is the
        supported way to say "we handle retention elsewhere" — plenty of deployments
        have a bucket policy of their own, and silently layering a second one on top of
        it is how evidence disappears a week before somebody expected.
        """
        from minio.commonconfig import ENABLED, Filter
        from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

        if self._retention_days <= 0:
            try:
                self._client.delete_bucket_lifecycle(self._bucket)
            except Exception:
                logger.debug("no lifecycle rule to remove from %s", self._bucket)
            return

        config = LifecycleConfig(
            [
                Rule(
                    ENABLED,
                    rule_id="sentinel-clip-retention",
                    # Every object in the bucket. The bucket holds clips and nothing
                    # else, so a prefix would only be a place for a future key layout to
                    # quietly fall outside the rule.
                    rule_filter=Filter(prefix=""),
                    expiration=Expiration(days=self._retention_days),
                )
            ]
        )
        try:
            self._client.set_bucket_lifecycle(self._bucket, config)
        except Exception:
            # Logged, never fatal. A store that refuses lifecycle configuration — an S3
            # implementation without it, or credentials without the permission — is
            # still a store that takes clips, and refusing to record evidence because it
            # cannot be scheduled for deletion is the wrong way round.
            logger.warning(
                "could not set a %d-day retention rule on %s; clips will be kept until "
                "something else removes them",
                self._retention_days,
                self._bucket,
                exc_info=True,
            )


_SHORT_SUFFIX = "-notify.mp4"
"""How `MinioClipHandle._write_short` names the trimmed copy. Named once so the writer
and the index cannot drift about which objects are which."""

_MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "NoSuchBucket"})
"""What the object store says when a clip is simply not there.

Retention deletes clips on a schedule, so an alert that outlives its clip is ordinary
rather than exceptional — `NoSuchBucket` joins it because a bucket whose last object
expired can be reaped too, and "the recording is gone" is the same answer either way.
Any other `S3Error` is a real failure and is raised.
"""


class MinioClipReader(ClipReader, ClipIndex):
    """Reads back what `MinioClipWriter` wrote, and says what is there.

    Separate from the writer rather than a second method on it, because the API holds
    this and the pipeline holds that — and because a reader configured with read-only
    credentials is a thing a deployment should be able to have.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        retention_days: int = 0,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        # Reported, not enforced, by this class: the expiry is a bucket lifecycle rule
        # the object store applies (see `MinioClipWriter._ensure_lifecycle`). Carried
        # here so a storage screen can say how long clips last without a second call.
        self._retention_days = retention_days

    async def read(self, uri: str) -> bytes | None:
        object_name = self._object_name(uri)
        if object_name is None:
            # Not an exception: the caller passed a URI this reader does not serve, and
            # the honest answer to "give me that clip" is that there is no such clip
            # here. An exception would turn a rejected URI into a 500, which reads as
            # an engine fault rather than as the refusal it is.
            logger.warning("refusing to read a clip uri outside this reader's bucket")
            return None
        try:
            return await asyncio.to_thread(self._fetch, object_name)
        except S3Error as error:
            if error.code in _MISSING_OBJECT_CODES:
                return None
            raise

    def _fetch(self, object_name: str) -> bytes:
        response = None
        try:
            response = self._client.get_object(self._bucket, object_name)
            data: bytes = response.read()
            return data
        finally:
            # `get_object` hands back a live urllib3 stream. Without both calls the
            # connection is never returned to the pool, and a console polling a wall of
            # alerts exhausts it.
            if response is not None:
                response.close()
                response.release_conn()

    def _object_name(self, uri: str) -> str | None:
        """The object this URI names, or `None` if this reader must not serve it.

        The bucket check is the point. A clip URI reaches this having come off an alert
        that this engine wrote, so in the current wiring it is already trustworthy —
        and this refuses to depend on that, because the check costs nothing and the
        assumption is one endpoint away from being false. `..` is rejected for the same
        reason: S3 keys are opaque and `..` has no meaning to MinIO, but this string
        also becomes a filesystem path in `_trim`'s sibling code paths, and a rule that
        holds everywhere is worth more than one that holds where it was first written.
        """
        prefix = f"s3://{self._bucket}/"
        if not uri.startswith(prefix):
            return None
        object_name = uri[len(prefix) :]
        if not object_name or ".." in object_name.split("/"):
            return None
        return object_name

    def clip_uri(self, camera_id: str, event_id: UUID, *, short: bool) -> str:
        """Where a given event's clip lives, without asking the store.

        The naming convention is `MinioClipWriter`'s, and this is the one place outside
        it that knows the convention — so a caller that wants to play a clip names the
        camera and the event rather than assembling an object path of its own.
        """
        suffix = "-notify" if short else ""
        return f"s3://{self._bucket}/{camera_id}/{event_id}{suffix}.mp4"

    async def usage(self) -> StorageUsage:
        try:
            return await asyncio.to_thread(self._usage)
        except Exception:
            logger.warning("could not read clip storage usage", exc_info=True)
            # Zeroes with `reachable=False`, never zeroes on their own: a storage screen
            # drawing an empty bucket is how an operator concludes their evidence has
            # been deleted when in fact nobody could ask.
            return StorageUsage(
                bucket=self._bucket,
                clips=0,
                objects=0,
                bytes_used=0,
                retention_days=self._retention_days,
                per_camera=(),
                reachable=False,
            )

    def _usage(self) -> StorageUsage:
        per_camera: dict[str, list[int]] = {}
        objects = 0
        total = 0
        for item in self._client.list_objects(self._bucket, recursive=True):
            name = item.object_name or ""
            size = item.size or 0
            objects += 1
            total += size
            camera_id = name.split("/", 1)[0] if "/" in name else ""
            counts = per_camera.setdefault(camera_id, [0, 0])
            counts[1] += size
            # Only the full clip increments the count. An operator asking how many
            # clips a camera has means incidents, and counting the short copy would
            # double every one of them.
            if not name.endswith(_SHORT_SUFFIX):
                counts[0] += 1
        return StorageUsage(
            bucket=self._bucket,
            clips=sum(counts[0] for counts in per_camera.values()),
            objects=objects,
            bytes_used=total,
            retention_days=self._retention_days,
            per_camera=tuple(
                CameraStorage(camera_id=camera_id, clips=counts[0], bytes_used=counts[1])
                for camera_id, counts in sorted(per_camera.items())
            ),
        )

    async def list_clips(self, camera_id: str, *, limit: int) -> tuple[ClipRecord, ...]:
        try:
            return await asyncio.to_thread(self._list_clips, camera_id, limit)
        except Exception:
            logger.warning("could not list clips for camera %s", camera_id, exc_info=True)
            return ()

    def _list_clips(self, camera_id: str, limit: int) -> tuple[ClipRecord, ...]:
        full: dict[UUID, ClipRecord] = {}
        short: set[UUID] = set()
        # `prefix` rather than filtering client-side: a bucket holding every camera's
        # clips would otherwise be read whole to answer a question about one of them.
        for item in self._client.list_objects(
            self._bucket, prefix=f"{camera_id}/", recursive=True
        ):
            name = item.object_name or ""
            stem = name.rsplit("/", 1)[-1]
            is_short = stem.endswith(_SHORT_SUFFIX)
            raw = stem[: -len(_SHORT_SUFFIX)] if is_short else stem.removesuffix(".mp4")
            try:
                event_id = UUID(raw)
            except ValueError:
                # Something else put an object here. Skipped rather than raised: one
                # stray file must not make a camera's whole footage list unreadable.
                continue
            if is_short:
                short.add(event_id)
                continue
            full[event_id] = ClipRecord(
                event_id=event_id,
                size_bytes=item.size or 0,
                modified_at=item.last_modified.timestamp() if item.last_modified else 0.0,
                has_short_copy=False,
            )
        ordered = sorted(full.values(), key=lambda clip: clip.modified_at, reverse=True)
        return tuple(
            replace(clip, has_short_copy=clip.event_id in short) for clip in ordered[:limit]
        )
