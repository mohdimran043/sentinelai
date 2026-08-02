"""Tests for `WebhookNotifier` (Task 6).

No real sockets: every test drives `httpx.MockTransport` instead of a live
endpoint, and every backoff sleep is a fake that records its argument rather
than actually waiting — `pyproject.toml`'s `filterwarnings = ["error"]` means
even an accidental `RuntimeWarning` from a real coroutine would fail the
suite, and a suite that really slept through exponential backoff would also
just be slow for no reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from sentinel_ai.adapters.notifiers.webhook import WebhookNotifier
from sentinel_ai.adapters.publishers.dead_letter import DeadLetterSpool
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.ports.notifier import Notifier, WelfareNote


def a_note(**overrides: object) -> WelfareNote:
    defaults: dict[str, object] = {
        "event_id": uuid4(),
        "camera_id": "cam-1",
        "label": "Front Door",
        "zone": "entrance",
        "occurred_at": 1_700_000_000.0,
        "severity": "high",
        "description": "A person is lying motionless on the floor.",
        "concerns": (
            WelfareConcern(
                kind=ConcernKind.COLLAPSE,
                confidence=Confidence.LIKELY,
                evidence="Person is prone and not moving.",
            ),
        ),
        "clip_uri": "http://minio.local/sentinel-clips/cam-1/evt.mp4",
    }
    defaults.update(overrides)
    return WelfareNote(**defaults)  # type: ignore[arg-type]


class _RecordingSleep:
    """An injectable `sleep` that records its argument instead of waiting,
    matching `adapters.sources.rtsp`'s `_ReconnectLoop` test convention."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class _RaisingDeadLetter:
    """A `_DeadLetterSink` double whose `store` always raises, used to prove
    `notify()` swallows even a failure in the last-resort sink itself."""

    async def store(self, event: WelfareNote, error: BaseException) -> None:
        raise RuntimeError("disk full")


def test_it_is_a_notifier() -> None:
    assert issubclass(WebhookNotifier, Notifier)


async def test_it_posts_the_note_as_json_to_the_configured_url(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(handle),
    )
    note = a_note(camera_id="cam-42")

    await notifier.notify(note)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://example.invalid/hook"
    body = json.loads(request.content)
    assert body["camera_id"] == "cam-42"
    assert body["event_id"] == str(note.event_id)
    # A successful delivery must not also dead-letter the note.
    assert list(tmp_path.glob("*.json")) == []


async def test_the_payload_carries_clip_uri_and_a_credential_warning(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(handle),
    )
    note = a_note(clip_uri="http://minio.local/bucket/evt.mp4")

    await notifier.notify(note)

    assert captured["clip_uri"] == "http://minio.local/bucket/evt.mp4"
    # Literal, explicit: this is not a UI hint, it's the wire payload itself.
    note_text = str(captured.get("clip_uri_note", ""))
    assert "credential" in note_text.lower()


async def test_evidence_stated_false_is_represented_in_the_payload_not_dropped(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(handle),
    )
    unevidenced = WelfareConcern(
        kind=ConcernKind.DISTRESS,
        confidence=Confidence.POSSIBLE,
        evidence="reported, not described",
        evidence_stated=False,
    )
    evidenced = WelfareConcern(
        kind=ConcernKind.COLLAPSE,
        confidence=Confidence.LIKELY,
        evidence="Person is prone and not moving.",
        evidence_stated=True,
    )
    note = a_note(concerns=(unevidenced, evidenced))

    await notifier.notify(note)

    concerns = captured["concerns"]
    assert isinstance(concerns, list)
    by_kind = {c["kind"]: c for c in concerns}
    assert by_kind["distress"]["evidence_stated"] is False
    assert by_kind["collapse"]["evidence_stated"] is True


async def test_default_timeout_is_5_seconds(tmp_path: Path) -> None:
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    assert notifier.timeout_seconds == 5.0


async def test_a_configured_timeout_overrides_the_default(tmp_path: Path) -> None:
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        timeout_seconds=1.5,
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    assert notifier.timeout_seconds == 1.5


async def test_a_timeout_does_not_raise_and_the_note_is_dead_lettered(tmp_path: Path) -> None:
    """The MockTransport handler raises the timeout exception directly —
    proving the code path is exercised without ever actually waiting out a
    real timeout."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout")

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=2,
        initial_backoff_seconds=0.01,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )
    note = a_note()

    await notifier.notify(note)  # must not raise

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "ConnectTimeout"
    assert record["event"]["camera_id"] == "cam-1"


async def test_connection_errors_are_retried_with_backoff_then_succeed(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("simulated connection failure")
        return httpx.Response(200)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=10.0,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert attempts["n"] == 3
    assert sleep.calls == pytest.approx([1.0, 2.0])  # doubling
    assert notifier.backoff_log == pytest.approx([1.0, 2.0])
    assert list(tmp_path.glob("*.json")) == []  # eventual success, no dead-letter


async def test_5xx_responses_are_retried_then_dead_lettered_on_exhaustion(
    tmp_path: Path,
) -> None:
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=3,
        initial_backoff_seconds=0.5,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )
    note = a_note()

    await notifier.notify(note)

    assert attempts["n"] == 3  # retried up to max_attempts, not dropped early
    assert sleep.calls == pytest.approx([0.5, 1.0])  # 2 sleeps between 3 attempts
    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "_WebhookStatusError"
    assert "503" in record["error"]
    assert record["event"]["event_id"] == str(note.event_id)


async def test_4xx_responses_are_not_retried(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=5,  # would retry 5 times if 4xx were treated as retryable
        initial_backoff_seconds=1.0,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert attempts["n"] == 1  # one attempt only — a 400 will be a 400 again
    assert sleep.calls == []  # no backoff sleep for a non-retryable failure
    assert notifier.backoff_log == []
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_a_401_is_not_retried_either(tmp_path: Path) -> None:
    """Distinct from the generic 4xx test: 401 specifically is the case the
    brief calls out — retrying it just replays a rejected credential."""
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=4,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert attempts["n"] == 1


async def test_the_webhook_url_never_appears_in_a_log_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The URL may carry a per-recipient token (ntfy/Slack put it in the
    path) — same discipline this codebase already applies to RTSP URLs."""
    secret_url = "https://example.invalid/webhook/super-secret-token-xyz"

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        secret_url,
        DeadLetterSpool(tmp_path),
        max_attempts=2,
        initial_backoff_seconds=0.01,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    import logging

    with caplog.at_level(logging.DEBUG):
        await notifier.notify(a_note())

    assert "super-secret-token-xyz" not in caplog.text


async def test_notify_never_raises_even_when_the_dead_letter_write_itself_fails() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        _RaisingDeadLetter(),
        max_attempts=1,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())  # must not raise


async def test_backoff_is_capped_at_max_backoff_seconds(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=3.0,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    # Doubling (1, 2, 4, 8...) would exceed max_backoff_seconds=3.0 by the
    # third sleep; it must be capped there instead.
    assert sleep.calls == pytest.approx([1.0, 2.0, 3.0, 3.0])
