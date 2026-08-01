"""Remux clip-writer tests (spec §5.5). The remux itself is pure PyAV and is
tested here with no MinIO server involved; `MinioClipHandle`/`MinioClipWriter`
are tested with the `minio.Minio` client monkeypatched, still with no network
access. The live-MinIO round trip lives in
`test_minio_clips_integration.py`, marked `@pytest.mark.integration` and
excluded from CI.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import av
import pytest

from sentinel_ai.adapters.storage.minio_clips import (
    MinioClipHandle,
    MinioClipWriter,
    UnsupportedCodecError,
    _RemuxSession,
)
from sentinel_ai.ports.frame_source import EncodedPacket

FIXTURE = Path(__file__).resolve().parents[2] / "assets" / "synthetic_clip.mp4"


def _annexb_packets_from_fixture(camera_id: str) -> list[EncodedPacket]:
    """Read the committed fixture and re-wrap it as Annex-B `EncodedPacket`s.

    `FileSource` (Task 4) demuxes this same fixture for the pipeline; this
    independently re-derives what `RtspSource` hands a `ClipWriter` in
    production. RTP H.264 (RFC 6184) carries SPS/PPS in-band, unlike the MP4
    container's out-of-band `avcC` record — so this converts the fixture's
    AVCC packets to Annex-B with the `h264_mp4toannexb` bitstream filter,
    which is exactly the ffmpeg-side transform that already happens for free
    on an RTSP source. `_RemuxSession` depends on packets looking like this.
    """
    container = av.open(str(FIXTURE))
    in_stream = container.streams.video[0]
    bsf = av.BitStreamFilterContext("h264_mp4toannexb", in_stream)
    packets: list[EncodedPacket] = []
    for packet in container.demux(in_stream):
        if packet.dts is None:
            continue
        for filtered in bsf.filter(packet):
            if filtered.pts is None:
                continue
            packets.append(
                EncodedPacket(
                    camera_id=camera_id,
                    data=bytes(filtered),
                    pts=float(filtered.pts * filtered.time_base),
                    is_keyframe=bool(filtered.is_keyframe),
                    codec=in_stream.codec_context.name,
                )
            )
    container.close()
    assert packets, "fixture produced no packets — is FIXTURE the right file?"
    assert packets[0].is_keyframe, "fixture must start on a keyframe for this test to be valid"
    # The fixture is a fixed-GOP-size encode: a keyframe roughly every 10 packets, giving
    # the mid-GOP test below a real non-keyframe packet to start from and a real later
    # keyframe to land on.
    assert any(p.is_keyframe for p in packets[1:]), "fixture needs a second keyframe"
    return packets


class TestRemuxSession:
    def test_a_written_clip_is_a_valid_mp4_starting_on_a_keyframe(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"

        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()

        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        out_packets = [p for p in out.demux(out_stream) if p.dts is not None]
        out.close()

        assert out_packets, "no packets were muxed into the clip"
        assert out_packets[0].is_keyframe
        assert len(out_packets) == len(packets)

    def test_dropped_leading_packets_still_yield_a_clip_starting_on_a_keyframe(
        self, tmp_path: Path
    ) -> None:
        """The first packet handed to a real `ClipHandle` is always a keyframe
        (`PreRollBuffer.flush()` guarantees it) — but this defends the case
        where a caller violates that contract, so it must be exercised
        directly: feed the session a non-keyframe packet first, along with
        several more before the next real keyframe, and confirm it drops all
        of them rather than starting the MP4 mid-GOP.

        The fixture already starts on a keyframe, so a test that appends every
        packet (as above) would pass even if the drop-until-keyframe logic
        were missing entirely. This test starts the feed mid-GOP specifically
        to exercise that logic.
        """
        packets = _annexb_packets_from_fixture("cam-1")
        first_keyframe_index = next(i for i, p in enumerate(packets) if p.is_keyframe)
        second_keyframe_index = next(
            i for i, p in enumerate(packets) if i > first_keyframe_index and p.is_keyframe
        )
        # Start the feed a few packets before the second keyframe, all non-keyframes.
        mid_gop_packets = packets[second_keyframe_index - 3 :]
        assert not mid_gop_packets[0].is_keyframe, "test setup: feed must start mid-GOP"
        assert mid_gop_packets[3].is_keyframe, "test setup: a keyframe must follow"

        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        for packet in mid_gop_packets:
            session.append(packet)
        session.finish()

        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        out_packets = [p for p in out.demux(out_stream) if p.dts is not None]
        out.close()

        assert out_packets, "no packets were muxed into the clip"
        assert out_packets[0].is_keyframe
        # Only the packets from the second keyframe onward should have been kept.
        assert len(out_packets) == len(mid_gop_packets) - 3

    def test_clip_duration_matches_the_appended_packets(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"

        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()

        # The muxed track's own duration also carries the last frame's presentation
        # duration (a real sample needs a length), so it is slightly longer than the
        # raw pts span between the first and last appended packet — hence the
        # two-frame tolerance rather than a tight one.
        expected_span = packets[-1].pts - packets[0].pts
        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        assert out_stream.duration is not None
        assert out_stream.time_base is not None
        actual_span = float(out_stream.duration * out_stream.time_base)
        out.close()

        assert actual_span == pytest.approx(expected_span, abs=2.0 / 25.0)

    def test_first_muxed_packet_starts_at_pts_zero(self, tmp_path: Path) -> None:
        """PTS/DTS rebasing: the clip's own clock starts at 0 regardless of where
        the camera's monotonic pts happened to be — otherwise players that assume
        a clip starts at t=0 would show a long black lead-in or refuse to seek.
        """
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"

        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()

        out = av.open(str(temp_path))
        out_stream = out.streams.video[0]
        first = next(p for p in out.demux(out_stream) if p.dts is not None)
        out.close()

        assert first.pts == 0

    def test_unsupported_codec_raises_on_the_first_packet(self, tmp_path: Path) -> None:
        """A codec outside `SUPPORTED_CODECS` must raise immediately rather than
        being parsed forever and finalising a valid-but-empty MP4 — the operator
        must find out now, not months later when they go looking for evidence
        that was never written.
        """
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        bad_packet = EncodedPacket(
            camera_id="cam-1",
            data=b"\x00\x00\x00\x01\x65",
            pts=0.0,
            is_keyframe=True,
            codec="mjpeg",
        )

        with pytest.raises(UnsupportedCodecError):
            session.append(bad_packet)

        session.abort()

    def test_codec_change_mid_clip_raises(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        session.append(packets[0])

        changed_codec_packet = EncodedPacket(
            camera_id="cam-1",
            data=packets[1].data,
            pts=packets[1].pts,
            is_keyframe=packets[1].is_keyframe,
            codec="hevc",
        )

        with pytest.raises(UnsupportedCodecError):
            session.append(changed_codec_packet)

        session.abort()

    def test_abort_without_finish_deletes_the_temp_file(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        # PyAV's output container does not actually create the file on disk until a
        # stream has been added and at least one packet muxed — and the h264 parser
        # only emits its first access unit on the *second* fed packet (see the pts
        # rebasing note above `_mux`). Appending only `packets[0]` would make this
        # test pass even against an `abort()` that never unlinks anything, because
        # there would be no file to leak yet. Feeding two packets is the minimum
        # needed to make the "must actually delete something" assertion meaningful.
        session.append(packets[0])
        session.append(packets[1])
        assert temp_path.exists(), "test setup: the session must have written a real file"

        session.abort()
        assert not temp_path.exists()

        session.abort()  # idempotent

    def test_abort_after_finish_never_raises(self, tmp_path: Path) -> None:
        packets = _annexb_packets_from_fixture("cam-1")
        temp_path = tmp_path / "clip.mp4"
        session = _RemuxSession(temp_path, fps=25.0)
        for packet in packets:
            session.append(packet)
        session.finish()
        assert temp_path.exists(), "finish() leaves the local file for the caller to upload"

        session.abort()  # after finish(): must be a safe no-op, not raise
        session.abort()  # and idempotent on top of that
        assert temp_path.exists(), "abort() after finish() must not delete a delivered clip"


class TestMinioClipHandle:
    async def test_finish_uploads_then_deletes_the_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)
        uploaded: list[tuple[str, str, str]] = []
        monkeypatch.setattr(
            writer._client,
            "fput_object",
            lambda bucket, name, path: uploaded.append((bucket, name, path)),
        )

        camera_id, event_id = "cam-1", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        for packet in _annexb_packets_from_fixture(camera_id):
            await handle.append(packet)
        uri = await handle.finish()

        assert uri == f"s3://sentinel-clips/{camera_id}/{event_id}.mp4"
        assert len(uploaded) == 1
        bucket, object_name, local_path = uploaded[0]
        assert bucket == "sentinel-clips"
        assert object_name == f"{camera_id}/{event_id}.mp4"
        assert not Path(local_path).exists(), "the temp file is deleted after upload"

    async def test_abort_deletes_the_temp_file_and_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)

        camera_id, event_id = "cam-1", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        packets = _annexb_packets_from_fixture(camera_id)
        # Two packets, not one: the h264 parser only emits its first access unit on
        # the second fed packet, and PyAV never creates the output file on disk until
        # a real access unit has been muxed. One packet would make this assertion
        # pass even against an `abort()` that never deletes anything.
        await handle.append(packets[0])
        await handle.append(packets[1])
        temp_path = tmp_path / f"{camera_id}-{event_id}.mp4"
        assert temp_path.exists(), "test setup: the handle must have written a real file"

        await handle.abort()
        assert not temp_path.exists()
        await handle.abort()  # idempotent, must not raise

    async def test_finish_retains_temp_file_when_upload_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spec §9: an anomaly event is never lost to an infrastructure failure. A
        transient MinIO failure (network blip, disk full, timeout) must not destroy
        the only copy of the evidence — the temp file must survive so an operator
        can recover it, even though the upload itself failed.
        """
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)

        def _raise_upload_failure(*_a: object, **_kw: object) -> None:
            raise OSError("simulated MinIO upload failure")

        monkeypatch.setattr(writer._client, "fput_object", _raise_upload_failure)

        camera_id, event_id = "cam-1", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        for packet in _annexb_packets_from_fixture(camera_id):
            await handle.append(packet)
        temp_path = tmp_path / f"{camera_id}-{event_id}.mp4"
        assert temp_path.exists(), "test setup: the handle must have written a real file"

        with pytest.raises(OSError, match="simulated MinIO upload failure"):
            await handle.finish()

        assert temp_path.exists(), "the only copy of the clip must survive an upload failure"

    async def test_abort_after_failed_finish_still_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the session's flush-and-close raises (e.g. a mux() failure while
        flushing trailing packets), the handle must not already be marked done —
        otherwise a subsequent `abort()` is a permanent no-op that leaks both the
        open PyAV container and the temp file.
        """
        writer = MinioClipWriter(
            endpoint="localhost:9000",
            access_key="x",
            secret_key="x",
            bucket="sentinel-clips",
            secure=False,
            temp_dir=tmp_path,
        )
        monkeypatch.setattr(writer._client, "bucket_exists", lambda *_a, **_kw: True)
        monkeypatch.setattr(writer._client, "make_bucket", lambda *_a, **_kw: None)

        camera_id, event_id = "cam-1", uuid4()
        handle = await writer.open(camera_id, event_id, fps=25.0)
        packets = _annexb_packets_from_fixture(camera_id)
        # Two packets, not one: same PyAV-file-creation-timing reasoning as the abort
        # tests above — one packet never produces a muxed access unit, so no file
        # would exist yet to leak regardless of what abort() does.
        await handle.append(packets[0])
        await handle.append(packets[1])
        temp_path = tmp_path / f"{camera_id}-{event_id}.mp4"
        assert temp_path.exists(), "test setup: the handle must have written a real file"

        def _raise_flush_failure() -> Path:
            raise RuntimeError("simulated mux() failure while flushing trailing packets")

        assert isinstance(handle, MinioClipHandle), "test setup: writer.open() must return this"
        monkeypatch.setattr(handle._session, "finish", _raise_flush_failure)

        with pytest.raises(RuntimeError, match="simulated mux"):
            await handle.finish()

        await handle.abort()
        assert not temp_path.exists(), "abort() must still clean up after a failed finish()"
