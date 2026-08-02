"""`WebhookNotifier` — POSTs a `WelfareNote` to a configured HTTP endpoint (Task 6).

`LoggingNotifier` (Task 5) is an observable trail for a console nobody is
watching. This is the adapter that reaches a human who is not looking at
one: a webhook is the one shape that reaches Signal (via signal-cli-rest),
ntfy, Slack, Home Assistant, or a custom endpoint without this codebase
committing to a provider the spec explicitly deferred choosing.

Three failure modes are handled deliberately, not incidentally, because a
welfare notifier's whole job is to survive them:

* **A hanging endpoint must not hold anything up.** `httpx.Timeout` bounds
  every attempt (`timeout_seconds`, default 5s) covering connect/write/read/
  pool as one figure — this codebase does not need per-phase tuning for a
  small JSON POST, and one number is one thing to get wrong.
* **Retry belongs only where retrying can help.** A 5xx or a connection
  error (DNS failure, refused connection, timeout) may be transient, so
  those retry with exponential backoff, capped and bounded (see `notify`'s
  docstring for the worst-case duration). A 4xx will be the same 4xx on the
  next attempt — a malformed payload stays malformed, a rejected token
  stays rejected — so those fail on the first response, matching this
  module's `logging.py` sibling in spirit: don't manufacture retries a
  human would have to explain.
* **A failed delivery must not vanish.** `DeadLetterSpool` (Task 5's sibling
  `adapters/publishers/dead_letter.py`) already writes a disk-backed,
  write-then-rename JSON record for exactly this situation; this module
  reuses it rather than inventing a second spool, via the `_DeadLetterSink`
  protocol below (mirrors `main.BrokerConnection`'s pattern of depending on
  the slice of a concrete class actually needed, not the class itself, so
  tests can substitute a double without touching a real directory).

Backoff sequencing (`initial_backoff_seconds` doubling, capped at
`max_backoff_seconds`) and the injectable `sleep` are the same shape as
`adapters.sources.rtsp._ReconnectLoop` and `main.BrokerLink` — this module
does not invent a third retry style. `sleep` defaults to `asyncio.sleep`
(a cooperative yield, not a blocking wait) so a slow or dead endpoint stalls
only this one `notify()` call, never the event loop the camera pipelines
share it with.

The webhook `url` itself commonly carries a credential in the path (an ntfy
topic token, a Slack incoming-webhook token) — the same reason
`adapters/config/camera_file.py` treats an RTSP URL as sensitive. Nothing in
this module logs `url`, and no wire exception's `str()` (which for
`httpx.HTTPStatusError`-style formatting would embed the URL) is logged
either — `_WebhookStatusError` carries only the status code, and connection/
timeout failures are logged by exception *type* only.

`notify()` never raises — the port's own contract for every adapter here,
restated in this module's docstring rather than assumed: a notifier that
threw would take down the pipeline it exists to observe.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from sentinel_ai.ports.notifier import Notifier, WelfareNote

logger = logging.getLogger(__name__)

__all__ = ["WebhookNotifier"]

_CLIP_URI_CREDENTIAL_NOTE = (
    "clip_uri, if present, may require credentials this recipient does not have — "
    "clip storage in this deployment is not presigned."
)


class _DeadLetterSink(Protocol):
    """The slice of `DeadLetterSpool` this notifier depends on.

    Mirrors `main.BrokerConnection`'s reason for existing: decouples this
    adapter (and its tests) from the concrete `DeadLetterSpool` class, so a
    test can inject a double that fails on `store` without needing a real,
    made-unwritable directory. `DeadLetterSpool.store` accepts
    `Event | WelfareNote` (see `adapters/publishers/dead_letter.py`), a
    supertype of this protocol's `WelfareNote`-only parameter, so it
    satisfies this structurally with no adapter-side glue.
    """

    async def store(self, event: WelfareNote, error: BaseException) -> None: ...


class _WebhookStatusError(RuntimeError):
    """A non-2xx HTTP response, carrying only the status code.

    Deliberately not the response body and not built from
    `httpx.Response.raise_for_status()`: that method's message embeds the
    request URL, and the URL may carry a per-recipient credential (see the
    module docstring). The status code alone is enough to log, dead-letter,
    and decide retryability.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"webhook responded with HTTP {status_code}")
        self.status_code = status_code


def _build_payload(note: WelfareNote) -> dict[str, Any]:
    """The wire body. Fields are named explicitly, the same discipline
    `logging.py`'s module docstring lays out for its log line: never
    `asdict(note)` wholesale, so a field added to `WelfareNote` later needs a
    deliberate edit here before it starts leaving the process over HTTP.

    `evidence_stated` is carried per concern, not flattened away — folding
    it into `evidence`'s text (or dropping it) would let a model that named
    a concern without describing it read, on the receiving end, as though it
    had (see `domain/welfare.py`'s `WelfareConcern.evidence_stated`
    docstring). `clip_uri_note` is always present, not only when `clip_uri`
    is set: a fixed, honest caveat costs nothing extra and needs no branch to
    get wrong.
    """
    return {
        "event_id": str(note.event_id),
        "camera_id": note.camera_id,
        "label": note.label,
        "zone": note.zone,
        "occurred_at": note.occurred_at,
        "severity": note.severity,
        "description": note.description,
        "concerns": [
            {
                "kind": str(concern.kind),
                "confidence": str(concern.confidence),
                "evidence": concern.evidence,
                "evidence_stated": concern.evidence_stated,
            }
            for concern in note.concerns
        ],
        "clip_uri": note.clip_uri,
        "clip_uri_note": _CLIP_URI_CREDENTIAL_NOTE,
    }


class WebhookNotifier(Notifier):
    def __init__(
        self,
        url: str,
        dead_letter: _DeadLetterSink,
        *,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 10.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = url
        self._dead_letter = dead_letter
        self.timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._sleep = sleep
        # Populated as attempts back off; read by tests the same way
        # `_ReconnectLoop.backoff_log`/`BrokerLink.backoff_log` are, rather
        # than asserting on wall-clock time spent sleeping.
        self.backoff_log: list[float] = []
        # `transport` is the httpx.MockTransport seam in tests; `None` in
        # production uses httpx's real transport. One `AsyncClient` per
        # notifier instance, not per call, so pooled connections are reused
        # across notifications the way a long-lived adapter should.
        self._client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(timeout_seconds)
        )
        # httpx logs "HTTP Request: POST <url> ..." at INFO by default on its own
        # `httpx` logger — a line this module never emits itself, but one that would
        # still put a URL-embedded credential (an ntfy/Slack token in the path) into
        # the log stream underneath it. Raising the threshold is the only way to
        # suppress a log line this module does not own. Idempotent, so constructing
        # more than one `WebhookNotifier` is harmless.
        logging.getLogger("httpx").setLevel(logging.WARNING)

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Not part of the `Notifier`
        port — same as `RabbitMQPublisher.close()`, this is the composition
        root's business to call during shutdown, not something every
        `Notifier` implementation needs (`LoggingNotifier` has no resource to
        release at all)."""
        await self._client.aclose()

    async def notify(self, note: WelfareNote) -> None:
        """Deliver `note`, retrying transient failures, dead-lettering a
        final failure, and never raising.

        Worst-case duration of one call: `max_attempts` request timeouts
        plus the backoff slept between them, i.e. (with the defaults)
        `3 * 5.0 + (1.0 + 2.0) = 18.0` seconds — bounded because
        `max_attempts` is finite and backoff is capped at
        `max_backoff_seconds`, not because either shrinks over time.
        """
        try:
            await self._deliver(note)
        except Exception:
            # Never raise: reaching here means a bug in this adapter itself
            # (`_deliver` already turns every wire/HTTP failure into a
            # dead-letter write, not an exception) — the same one exception
            # this port's docstring carves out for every notifier, matching
            # `logging.py`'s `notify`.
            logger.error(
                "webhook notifier failed unexpectedly for event_id=%s",
                getattr(note, "event_id", "<unknown>"),
                exc_info=True,
            )

    async def _deliver(self, note: WelfareNote) -> None:
        payload = _build_payload(note)
        backoff = self._initial_backoff
        error: BaseException = RuntimeError("webhook delivery did not run")
        attempt = 0
        for attempt in range(1, self._max_attempts + 1):
            retryable = True
            try:
                response = await self._client.post(self._url, json=payload)
            except httpx.RequestError as exc:
                # Covers both connection errors (DNS failure, refused
                # connection, reset) and timeouts — `httpx.TimeoutException`
                # is itself a `RequestError` subclass, so a hanging endpoint
                # and a dead one are retried the same way.
                error = exc
            else:
                if response.status_code < 400:
                    return
                error = _WebhookStatusError(response.status_code)
                # No retry on 4xx: a 400 will be 400 again, and retrying a
                # 401 just replays a rejected credential.
                retryable = response.status_code >= 500

            if not retryable or attempt == self._max_attempts:
                break

            logger.warning(
                "webhook notification attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                self._max_attempts,
                type(error).__name__,
                backoff,
            )
            self.backoff_log.append(backoff)
            await self._sleep(backoff)
            backoff = min(backoff * 2.0, self._max_backoff)

        logger.error(
            "webhook notification for event_id=%s could not be delivered after "
            "%d attempt(s) (%s); spooling to dead-letter",
            note.event_id,
            attempt,
            type(error).__name__,
        )
        await self._dead_letter.store(note, error)
