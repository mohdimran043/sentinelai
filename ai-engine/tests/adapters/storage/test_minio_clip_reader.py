"""`MinioClipReader` — reading a clip back out, and refusing to read anything else.

No network: the `minio.Minio` client is replaced, as it is in `test_minio_clips.py`.
What is under test here is not MinIO's behaviour but this class's two jobs — turning a
missing object into `None` rather than an exception, and refusing a URI it must not
serve.
"""

from __future__ import annotations

import pytest
from minio.error import S3Error

from sentinel_ai.adapters.storage.minio_clips import MinioClipReader

BUCKET = "sentinel-clips"


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class _FakeClient:
    """Records every object name it is asked for, so a test can assert that a rejected
    URI never reached the store at all — not merely that its bytes were discarded."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.requested: list[str] = []
        self.responses: list[_FakeResponse] = []

    def get_object(self, bucket: str, object_name: str) -> _FakeResponse:
        self.requested.append(object_name)
        if bucket != BUCKET or object_name not in self.objects:
            raise S3Error(
                code="NoSuchKey",
                message="Object does not exist",
                resource=f"/{bucket}/{object_name}",
                request_id="test",
                host_id="test",
                response=None,
            )
        response = _FakeResponse(self.objects[object_name])
        self.responses.append(response)
        return response


def reader(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> MinioClipReader:
    monkeypatch.setattr(
        "sentinel_ai.adapters.storage.minio_clips.Minio",
        lambda *args, **kwargs: client,
    )
    return MinioClipReader(
        endpoint="minio:9000",
        access_key="key",
        secret_key="secret",
        bucket=BUCKET,
        secure=False,
    )


class TestReadingAClip:
    async def test_returns_the_object_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _FakeClient({"cam-1/abc-notify.mp4": b"mp4-bytes"})
        got = await reader(monkeypatch, client).read(f"s3://{BUCKET}/cam-1/abc-notify.mp4")

        assert got == b"mp4-bytes"
        assert client.requested == ["cam-1/abc-notify.mp4"]

    async def test_a_missing_object_is_none_rather_than_a_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retention deletes clips on a schedule. An alert that outlived its clip is the
        ordinary state of an old alert, and the API turns this `None` into a 404 saying
        the footage has expired — not a 500 saying the engine broke."""
        client = _FakeClient()
        got = await reader(monkeypatch, client).read(f"s3://{BUCKET}/cam-1/gone.mp4")

        assert got is None

    async def test_the_connection_is_always_returned_to_the_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`get_object` hands back a live stream. Both calls, or a console polling a
        wall of alerts exhausts the pool and the engine stops answering."""
        client = _FakeClient({"cam-1/abc.mp4": b"bytes"})
        await reader(monkeypatch, client).read(f"s3://{BUCKET}/cam-1/abc.mp4")

        assert [(r.closed, r.released) for r in client.responses] == [(True, True)]

    async def test_a_real_store_failure_still_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only "it is not there" is swallowed. An unreachable or refusing store is a
        fault, and reporting it as a missing clip would tell an operator their evidence
        had expired when the truth is that nobody can read any of it."""

        class _BrokenClient(_FakeClient):
            def get_object(self, bucket: str, object_name: str) -> _FakeResponse:
                raise S3Error(
                    code="AccessDenied",
                    message="nope",
                    resource="/x",
                    request_id="t",
                    host_id="t",
                    response=None,
                )

        with pytest.raises(S3Error):
            await reader(monkeypatch, _BrokenClient()).read(f"s3://{BUCKET}/cam-1/abc.mp4")


class TestRefusingAUriItMustNotServe:
    """The reader is the last thing between a URI and the object store.

    In the current wiring every URI it sees came off an alert this engine wrote, so
    these refusals are redundant today. They exist because that is one endpoint away
    from being false, and the check is cheaper than the audit that would otherwise be
    needed every time someone adds a route.
    """

    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("s3://other-bucket/cam-1/abc.mp4", id="another bucket"),
            pytest.param("s3://sentinel-clips-evil/cam-1/abc.mp4", id="a bucket sharing a prefix"),
            pytest.param("https://example.com/abc.mp4", id="a different scheme entirely"),
            pytest.param("file:///etc/passwd", id="a local file"),
            pytest.param(f"s3://{BUCKET}/../secrets/abc.mp4", id="traversal out of the bucket"),
            pytest.param(f"s3://{BUCKET}/cam-1/../../abc.mp4", id="traversal inside a key"),
            pytest.param(f"s3://{BUCKET}/", id="the bucket itself"),
            pytest.param("", id="nothing at all"),
        ],
    )
    async def test_is_refused_without_reaching_the_store(
        self, monkeypatch: pytest.MonkeyPatch, uri: str
    ) -> None:
        client = _FakeClient({"cam-1/abc.mp4": b"bytes"})
        got = await reader(monkeypatch, client).read(uri)

        assert got is None
        # The point of the assertion: not merely that the caller got nothing, but that
        # the object store was never asked.
        assert client.requested == []

    async def test_the_refusal_never_logs_the_uri(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A rejected URI is the one most likely to be interesting, and an engine log is
        the wrong place for it to end up."""
        client = _FakeClient()
        with caplog.at_level("WARNING"):
            await reader(monkeypatch, client).read("s3://other-bucket/secret/path.mp4")

        assert "other-bucket" not in caplog.text
        assert "secret/path" not in caplog.text


def test_the_reader_does_not_expose_a_write_path() -> None:
    """A reader that could delete or overwrite a clip would make evidence editable
    through the console. It holds a client that can, so the guarantee is that this class
    offers no method that does.

    Asserted as "nothing writes" rather than as an exact list of methods, because the
    exact list is not the property worth protecting — this class implements two read
    ports (`ClipReader` and `ClipIndex`) and may implement a third, and a test that
    failed on every addition would be edited into agreement rather than consulted.
    """
    surface = {name for name in vars(MinioClipReader) if not name.startswith("_")}

    assert surface == {"read", "usage", "list_clips", "clip_uri"}
    forbidden = ("write", "put", "delete", "remove", "upload", "set", "abort")
    assert not [name for name in surface if any(verb in name for verb in forbidden)]
