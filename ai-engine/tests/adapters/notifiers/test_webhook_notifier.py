"""Tests for `WebhookNotifier` (Task 6).

No real sockets: every test drives `httpx.MockTransport` instead of a live
endpoint, and every backoff sleep is a fake that records its argument rather
than actually waiting — `pyproject.toml`'s `filterwarnings = ["error"]` means
even an accidental `RuntimeWarning` from a real coroutine would fail the
suite, and a suite that really slept through exponential backoff would also
just be slow for no reason.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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
    """Asserting only `notifier.timeout_seconds == 5.0` would pass even if the
    constructor never wired the value into the `httpx.AsyncClient` at all —
    httpx's own built-in default also happens to be 5.0. The requirement
    under test is that the figure actually reaches the client, so the
    assertion has to live inside the transport handler, on the request httpx
    actually sent."""
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(handle),
    )
    assert notifier.timeout_seconds == 5.0

    await notifier.notify(a_note())

    assert seen["timeout"] == {"connect": 5.0, "read": 5.0, "write": 5.0, "pool": 5.0}


async def test_a_configured_timeout_overrides_the_default(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["timeout"] = request.extensions["timeout"]
        return httpx.Response(200)

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        timeout_seconds=1.5,
        transport=httpx.MockTransport(handle),
    )
    assert notifier.timeout_seconds == 1.5

    await notifier.notify(a_note())

    assert seen["timeout"] == {"connect": 1.5, "read": 1.5, "write": 1.5, "pool": 1.5}


async def test_timeout_seconds_bounds_wall_clock_time_not_just_per_phase_gaps(
    tmp_path: Path,
) -> None:
    """`httpx.Timeout` alone is per-*phase*: its `read` clock resets on every
    byte received, so a handler that keeps producing output — however
    slowly — never trips it. `MockTransport` makes this concrete: it enforces
    no timeout of its own at all, so before the fix this test's handler would
    run to completion regardless of `timeout_seconds`. Wrapping the attempt in
    `asyncio.timeout` is what turns `timeout_seconds` into an actual
    wall-clock bound, which is what `notify`'s docstring claims."""

    async def handle(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)  # far longer than timeout_seconds below
        return httpx.Response(200)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        timeout_seconds=0.02,
        max_attempts=1,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    started = time.monotonic()
    await notifier.notify(a_note())  # must not raise
    elapsed = time.monotonic() - started

    assert elapsed < 0.2  # bounded by timeout_seconds, not the handler's 0.3s sleep
    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "TimeoutError"


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


async def test_a_3xx_response_is_not_treated_as_success(tmp_path: Path) -> None:
    """`httpx.AsyncClient` defaults to `follow_redirects=False` (and this
    module deliberately never turns that on — see the module docstring: a
    redirect target could exfiltrate the token-bearing URL), so a 301/302/307/
    308 comes back as a normal `Response`, not an exception. Treating
    anything under 400 as success — the bug this test guards against — means
    an ntfy host that 308s to its canonical URL, or an auth proxy that 302s
    to a login page, silently discards every welfare alert from that moment:
    no retry, no dead-letter, no log line. A 3xx must fail like any other
    non-2xx response and must not be retried (it will be the same redirect
    next attempt)."""
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(302, headers={"Location": "https://example.invalid/elsewhere"})

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=3,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())  # must not raise

    assert attempts["n"] == 1
    assert sleep.calls == []
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_the_webhook_url_never_appears_in_a_log_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The URL may carry a per-recipient token (ntfy/Slack put it in the
    path) — same discipline this codebase already applies to RTSP URLs.
    httpx logs its own "HTTP Request: POST <url> ..." line at INFO on the
    process-wide `httpx` logger (see `_client.py`), independent of anything
    this module logs itself, so redaction has to actually intercept that
    line rather than merely avoid emitting the URL from this module's own
    calls.

    `install_httpx_log_redaction` is explicit, not a constructor side
    effect (see its docstring for why), so this test calls it — and removes
    it again in `finally`, so the filter does not leak onto the process-wide
    `httpx` logger for every test after this one."""
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

    notifier.install_httpx_log_redaction()
    try:
        with caplog.at_level(logging.DEBUG):
            await notifier.notify(a_note())
            # Positive control: a targeted filter must leave every other
            # httpx diagnostic alone — this is the whole reason the fix is a
            # `Filter` and not the blanket `setLevel(WARNING)` it replaces,
            # which would have thrown this line away along with the URL.
            logging.getLogger("httpx").info("connection pool created")
    finally:
        notifier.remove_httpx_log_redaction()

    assert "super-secret-token-xyz" not in caplog.text
    assert "<webhook url redacted>" in caplog.text
    assert "connection pool created" in caplog.text


async def test_removing_the_redaction_filter_lets_the_url_through_again(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Proves `remove_httpx_log_redaction` actually detaches the filter,
    rather than merely existing as an unused method — install, log a
    matching line, remove, log another matching line, and only the first
    one should be redacted."""
    secret_url = "https://example.invalid/webhook/super-secret-token-xyz"
    notifier = WebhookNotifier(
        secret_url,
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    httpx_logger = logging.getLogger("httpx")

    notifier.install_httpx_log_redaction()
    with caplog.at_level(logging.DEBUG):
        httpx_logger.info("first line mentions %s", secret_url)
    notifier.remove_httpx_log_redaction()
    with caplog.at_level(logging.DEBUG):
        httpx_logger.info("second line mentions %s", secret_url)

    records = [r.getMessage() for r in caplog.records]
    assert any("redacted" in m and secret_url not in m for m in records)
    assert any(secret_url in m for m in records)


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


async def test_a_closed_client_dead_letters_the_note_instead_of_dropping_it(
    tmp_path: Path,
) -> None:
    """`_deliver` only converts `httpx.RequestError` into a dead-letter, so
    anything else escaping `self._client.post` used to bypass `store()`
    entirely and vanish. Calling `notify()` after `aclose()` is not a
    contrived case: it is exactly the ordering the composition root produces
    on shutdown if an escalation worker still has a note queued when
    `aclose()` runs — the same race `orchestrator.service._spill_to_dead_letter`
    exists to close for events. httpx raises a plain `RuntimeError` (not an
    `httpx.RequestError`) for a post-close request, so this exercises the
    exact gap."""
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    await notifier.aclose()
    note = a_note()

    await notifier.notify(note)  # must not raise

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "RuntimeError"
    assert record["event"]["camera_id"] == "cam-1"


async def test_a_non_httpx_exception_from_the_transport_is_dead_lettered_not_dropped(
    tmp_path: Path,
) -> None:
    """A `ValueError` (or any exception that is not an `httpx.RequestError`)
    raised out of the transport is exactly as capable of losing a note as
    the closed-client case above — this proves the fix is general, not a
    special case for `RuntimeError` alone."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise ValueError("simulated non-httpx transport failure")

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        transport=httpx.MockTransport(handle),
    )
    note = a_note()

    await notifier.notify(note)  # must not raise

    (path,) = sorted(tmp_path.glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["error_type"] == "ValueError"
    assert record["event"]["camera_id"] == "cam-1"


async def test_a_non_httpx_exception_is_dead_lettered_even_if_the_store_itself_then_fails() -> None:
    """The nested dead-letter attempt in `notify()`'s catch-all must itself
    never raise — mirrors `test_notify_never_raises_even_when_the_dead_letter_write_itself_fails`
    but for the non-`RequestError` path specifically, since that path is a
    separate `except` branch in the implementation."""

    def handle(request: httpx.Request) -> httpx.Response:
        raise ValueError("simulated non-httpx transport failure")

    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        _RaisingDeadLetter(),
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())  # must not raise


async def test_429_is_retried_honouring_a_parseable_retry_after(tmp_path: Path) -> None:
    """ntfy and Slack rate-limit with 429, and a burst of simultaneous
    welfare alerts is exactly the moment a 429 is most likely — and the
    moment dropping the note to the dead letter instead of retrying matters
    most. The configured `initial_backoff_seconds` is set to an obviously
    wrong value (5.0) so the test fails loudly if `Retry-After` is ignored
    in favour of normal backoff."""
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(429, headers={"Retry-After": "0.25"})
        return httpx.Response(200)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=3,
        initial_backoff_seconds=5.0,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert attempts["n"] == 2
    assert sleep.calls == pytest.approx([0.25])
    assert list(tmp_path.glob("*.json")) == []


async def test_429_without_a_parseable_retry_after_falls_back_to_normal_backoff(
    tmp_path: Path,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)  # no Retry-After header at all

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=3,
        initial_backoff_seconds=0.5,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert sleep.calls == pytest.approx([0.5, 1.0])  # normal doubling, unaffected
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_a_non_numeric_retry_after_also_falls_back_to_normal_backoff(
    tmp_path: Path,
) -> None:
    """`Retry-After` may be an HTTP-date (`Wed, 21 Oct 2026 07:28:00 GMT`)
    rather than a plain second count; this module only honours the numeric
    form (see `_parse_retry_after`'s docstring), so a date string must fall
    back exactly like a missing header."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=2,
        initial_backoff_seconds=0.5,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert sleep.calls == pytest.approx([0.5])


async def test_a_hostile_retry_after_is_capped_at_max_backoff_seconds(tmp_path: Path) -> None:
    """A huge (or malicious) `Retry-After` must not be able to pin a worker
    indefinitely — capped at the same ceiling normal backoff already
    respects."""

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "999999"})

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=2,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=3.0,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert sleep.calls == pytest.approx([3.0])


async def test_408_is_retried_rather_than_dead_lettered_on_the_first_attempt(
    tmp_path: Path,
) -> None:
    """408 (Request Timeout) is the other deliberate exception to "4xx never
    retries": a server-side request timeout is transient in the same way a
    connection timeout is."""
    attempts = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 2:
            return httpx.Response(408)
        return httpx.Response(200)

    sleep = _RecordingSleep()
    notifier = WebhookNotifier(
        "https://example.invalid/hook",
        DeadLetterSpool(tmp_path),
        max_attempts=3,
        initial_backoff_seconds=0.1,
        sleep=sleep,
        transport=httpx.MockTransport(handle),
    )

    await notifier.notify(a_note())

    assert attempts["n"] == 2
    assert list(tmp_path.glob("*.json")) == []


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
