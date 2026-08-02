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
  small JSON POST, and one number is one thing to get wrong. That figure
  alone is not a wall-clock bound, though: `httpx.Timeout`'s `read` clock
  resets on every byte received, so a server dribbling output slowly enough
  never trips it. `_deliver` additionally wraps each attempt in
  `asyncio.timeout(self.timeout_seconds)`, which *is* wall-clock, so the
  worst-case duration `notify`'s docstring documents is actually true.
* **Retry belongs only where retrying can help.** A 5xx, a 429 or 408
  (rate-limited or a server-side request timeout — both transient, and for
  a welfare system a burst of simultaneous alerts is exactly when a 429 is
  most likely and when dropping the note matters most), or a connection
  error (DNS failure, refused connection, timeout) may be transient, so
  those retry with exponential backoff, capped and bounded (see `notify`'s
  docstring for the worst-case duration) — honouring a numeric `Retry-After`
  on a 429/408 when the endpoint sends one, itself capped at
  `max_backoff_seconds` so a hostile or mistaken header cannot pin a worker.
  Any other 4xx will be the same 4xx on the next attempt — a malformed
  payload stays malformed, a rejected token stays rejected — so those fail
  on the first response, matching this module's `logging.py` sibling in
  spirit: don't manufacture retries a human would have to explain. A 3xx is
  not success (`httpx.AsyncClient` defaults to `follow_redirects=False`, and
  this module deliberately never overrides that — see below — so a
  301/302/307/308 comes back as an ordinary `Response`, not an exception);
  it fails like any other non-2xx and is not retried either, for the same
  "will be the same next time" reason.
* **A failed delivery must not vanish.** `DeadLetterSpool` (Task 5's sibling
  `adapters/publishers/dead_letter.py`) already writes a disk-backed,
  write-then-rename JSON record for exactly this situation; this module
  reuses it rather than inventing a second spool, via the `_DeadLetterSink`
  protocol below (mirrors `main.BrokerConnection`'s pattern of depending on
  the slice of a concrete class actually needed, not the class itself, so
  tests can substitute a double without touching a real directory). This
  covers more than the wire failures `_deliver` anticipates: `notify`'s own
  catch-all also makes a best-effort dead-letter attempt for anything else
  that escapes delivery (a bug in this adapter, or the composition root
  calling `aclose()` while a note is still in flight), because the
  alternative — the note just vanishing — is the one failure this whole
  module exists to prevent.

Backoff sequencing (`initial_backoff_seconds` doubling, capped at
`max_backoff_seconds`) and the injectable `sleep` are the same shape as
`adapters.sources.rtsp._ReconnectLoop` and `main.BrokerLink` — this module
does not invent a third retry style. `sleep` defaults to `asyncio.sleep`
(a cooperative yield, not a blocking wait) so a slow or dead endpoint stalls
only this one `notify()` call, never the event loop the camera pipelines
share it with.

The webhook `url` itself commonly carries a credential in the path (an ntfy
topic token, a Slack incoming-webhook token) — the same reason
`adapters/config/camera_file.py` treats an RTSP URL as sensitive, and the
same reason `follow_redirects` is never turned on above: a redirect target
could name any host, and following one would re-send the credential-bearing
URL wherever the response's `Location` header points. Nothing in this
module logs `url`, and no wire exception's `str()` (which for
`httpx.HTTPStatusError`-style formatting would embed the URL) is logged
either — `_WebhookStatusError` carries only the status code, and connection/
timeout failures are logged by exception *type* only.

httpx logs its own "HTTP Request: POST <url> ..." line at INFO on the
process-wide `httpx` logger, independent of anything this module logs
itself, which would put the same credential into the log stream underneath
it. `install_httpx_log_redaction` attaches a `logging.Filter` scoped to this
one URL rather than raising that logger's level — a level raise is global
(every `WebhookNotifier` in the process, plus anything else using httpx,
e.g. `transformers` fetching a model over HTTP) and would silently override
an operator who raised httpx's level to DEBUG to diagnose the very delivery
problem this module exists to survive.

`notify()` never raises — the port's own contract for every adapter here,
restated in this module's docstring rather than assumed: a notifier that
threw would take down the pipeline it exists to observe.
"""

from __future__ import annotations

import asyncio
import logging
import math
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


class _RedactWebhookUrl(logging.Filter):
    """Scrubs one specific webhook URL out of any record on the logger it is
    attached to.

    Deliberately a `Filter`, not a level change: a `Filter` inspects and can
    rewrite each record but never decides whether the record is emitted at
    all, so it leaves every other `httpx` diagnostic — including ones this
    module would never think to anticipate — reaching wherever the operator
    configured logging to send them. `filter()` always returns `True`: this
    is a redactor, not a gate, and a `Filter` that dropped records would be
    the same disproportionate cure this module rejected once already (see
    `WebhookNotifier.install_httpx_log_redaction`).
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._url in message:
            # `record.getMessage()` already interpolates `record.args` into
            # `record.msg`; writing the redacted text back to `record.msg`
            # and clearing `record.args` is what makes the *formatted*
            # record redacted too, not just this one inspection of it —
            # leaving `args` in place would let a handler that re-formats
            # `record.msg % record.args` reproduce the URL right back.
            record.msg = message.replace(self._url, "<webhook url redacted>")
            record.args = ()
        return True


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a `Retry-After` header as a plain count of seconds.

    Deliberately not the HTTP-date form (`Retry-After: Wed, 21 Oct 2026
    07:28:00 GMT`) that the header also permits: parsing that would need
    clock-skew handling this module has no other use for, and both ntfy and
    Slack send the numeric form in practice. Anything that is not a plain,
    finite, non-negative number — including the date form, garbage, or a
    missing header — returns `None`, which `_deliver` treats as "fall back
    to this notifier's own backoff", always a safe default since it never
    depends on anything the far end sent.
    """
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


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
        # Set by `install_httpx_log_redaction`, unset by
        # `remove_httpx_log_redaction`; `None` means "not installed" so the
        # latter is a safe no-op if the former was never called.
        self._httpx_redaction_filter: logging.Filter | None = None

    def install_httpx_log_redaction(self) -> logging.Filter:
        """Attach a `_RedactWebhookUrl` filter to the process-wide `httpx`
        logger so this notifier's `url` — which commonly carries a
        per-recipient credential, an ntfy topic token or a Slack
        incoming-webhook token in the path — never reaches a log handler via
        httpx's own request-line logging (see the module docstring), without
        touching that logger's level and so without hiding any other httpx
        diagnostic.

        Deliberately not called from `__init__`. Mutating a logger anyone
        else in the process shares is a global, process-wide side effect
        (`transformers` uses httpx for model fetches and shares this same
        `httpx` logger, for one), and an adapter's constructor running that
        invisibly — on every test that merely constructs a
        `WebhookNotifier`, not only the ones that care — is the wrong place
        to hide it. The composition root (Task 10) calls this explicitly,
        once, when a real webhook URL is actually configured; a caller that
        wants it undone again (a test, or a composition root tearing an
        instance down) calls `remove_httpx_log_redaction`.

        Returns the installed filter for convenience — it round-trips back
        into `logging.getLogger("httpx").removeFilter(...)` if a caller
        wants to manage removal itself instead of using
        `remove_httpx_log_redaction`.
        """
        self._httpx_redaction_filter = _RedactWebhookUrl(self._url)
        logging.getLogger("httpx").addFilter(self._httpx_redaction_filter)
        return self._httpx_redaction_filter

    def remove_httpx_log_redaction(self) -> None:
        """Undo `install_httpx_log_redaction`. A no-op if it was never
        called — so a caller (a test's `finally`, a composition root's
        shutdown path) can always call this unconditionally rather than
        tracking whether installation happened."""
        if self._httpx_redaction_filter is not None:
            logging.getLogger("httpx").removeFilter(self._httpx_redaction_filter)
            self._httpx_redaction_filter = None

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

        Worst-case duration of one call: `max_attempts` request attempts,
        each wall-clock-bounded at `timeout_seconds` by the
        `asyncio.timeout` wrapping it in `_deliver` (not merely
        `httpx.Timeout`, which is per-phase and so does not by itself bound
        an attempt against an endpoint that dribbles bytes slowly — see the
        module docstring), plus the backoff slept between them, i.e. (with
        the defaults) `3 * 5.0 + (1.0 + 2.0) = 18.0` seconds — bounded
        because `max_attempts` is finite and backoff (including any
        `Retry-After` this notifier honours) is capped at
        `max_backoff_seconds`, not because either shrinks over time.
        """
        try:
            await self._deliver(note)
        except Exception as exc:
            # Never raise: reaching here means either a bug in this adapter
            # itself, or a failure `_deliver` cannot convert into a
            # dead-letter write because it never reaches `_deliver`'s own
            # `except httpx.RequestError` at all — a `RuntimeError` from
            # `self._client.post` after `aclose()` (the composition root
            # tearing this notifier down while an escalation worker still
            # has a note queued — the same race
            # `orchestrator.service._spill_to_dead_letter` exists to close
            # for events, with nothing upstream protecting this one) is the
            # case that matters, but any other exception type is just as
            # capable of losing the note. So this is still the one exception
            # this port's docstring carves out for every notifier, matching
            # `logging.py`'s `notify`, but it now also makes a best-effort
            # attempt to spool the note before giving up — losing it here
            # instead would be exactly the failure this whole module exists
            # to prevent.
            logger.error(
                "webhook notifier failed unexpectedly for event_id=%s",
                getattr(note, "event_id", "<unknown>"),
                exc_info=True,
            )
            try:
                await self._dead_letter.store(note, exc)
            except Exception:
                # Nested and never re-raised: this is the last resort's last
                # resort. Reaching here means both delivery and the spill
                # attempt above failed, and there is nothing further to fall
                # back to but the log line — same posture as
                # `DeadLetterSpool.store`'s own unwritable-directory case.
                logger.error(
                    "dead-letter write also failed for event_id=%s after an "
                    "unexpected webhook notifier failure; the note is lost",
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
            retry_after_seconds: float | None = None
            try:
                # `asyncio.timeout` wraps the attempt in an actual wall-clock
                # deadline; `httpx.Timeout` above it is per-phase (its `read`
                # clock resets on every byte received) and so, alone, would
                # not bound an endpoint that dribbles output slowly — see the
                # module docstring and `notify`'s docstring for why the two
                # figures are meant to be the same number doing different jobs.
                async with asyncio.timeout(self.timeout_seconds):
                    response = await self._client.post(self._url, json=payload)
            except (httpx.RequestError, TimeoutError) as exc:
                # Covers connection errors (DNS failure, refused connection,
                # reset), httpx's own per-phase timeout (`httpx.TimeoutException`
                # is itself a `RequestError` subclass), and the wall-clock
                # `TimeoutError` `asyncio.timeout` raises above — a hanging
                # endpoint and a dead one are retried the same way regardless
                # of which layer noticed. `asyncio.timeout` only converts its
                # *own* deadline into `TimeoutError`; a cancellation delivered
                # from outside this call (e.g. process shutdown) still
                # surfaces as `asyncio.CancelledError`, which is not caught
                # here and propagates, per this module's contract.
                error = exc
            else:
                if 200 <= response.status_code < 300:
                    return
                error = _WebhookStatusError(response.status_code)
                # No retry on most 4xx: a 400 will be 400 again, and
                # retrying a 401 just replays a rejected credential. 429 and
                # 408 are the deliberate exceptions (see the module
                # docstring for why). A 3xx also lands here as non-retryable:
                # it is not success (see the module docstring for why
                # `follow_redirects` stays off), and it will be the same
                # redirect on the next attempt.
                retryable = response.status_code >= 500 or response.status_code in (429, 408)
                if response.status_code in (429, 408):
                    retry_after_seconds = _parse_retry_after(response.headers.get("Retry-After"))

            if not retryable or attempt == self._max_attempts:
                break

            sleep_seconds = backoff
            if retry_after_seconds is not None:
                # Honour the server's stated wait, but never past this
                # notifier's own ceiling — a hostile or simply enormous
                # `Retry-After` must not be able to pin a worker indefinitely.
                sleep_seconds = min(retry_after_seconds, self._max_backoff)

            logger.warning(
                "webhook notification attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                self._max_attempts,
                type(error).__name__,
                sleep_seconds,
            )
            self.backoff_log.append(sleep_seconds)
            await self._sleep(sleep_seconds)
            backoff = min(backoff * 2.0, self._max_backoff)

        logger.error(
            "webhook notification for event_id=%s could not be delivered after "
            "%d attempt(s) (%s); spooling to dead-letter",
            note.event_id,
            attempt,
            type(error).__name__,
        )
        await self._dead_letter.store(note, error)
