"""Which welfare concerns are worth a human's attention (spec §5, Task 10).

The rule itself is two clauses and a corroboration gate:

* the concern's `kind` is one this camera is configured to notify about
  (`notify_on`), and
* its `confidence` meets this camera's `notify_min_confidence`, and
* if the concern is only `possible`, the event's threat score must already be
  in the caution band or above.

The third clause is why this module exists rather than a two-line comprehension
at the call site. `possible` is the model's own admission that a single still
frame cannot rule out an innocent explanation (see `domain/welfare.py` on why
there is no `CERTAIN` tier). Acting on it alone, with nothing else in the
system agreeing, is how a welfare notifier becomes noise an operator learns to
ignore — and an ignored notifier is a muted one. Requiring the threat score to
agree means a `possible` concern reaches a human when *something else about the
scene was already unusual*, and stays quiet when it was not.

Pure, like the rest of `domain/policy/`: no clock, no I/O, no `config`. What a
camera's `notify_on` and `notify_min_confidence` are is configuration, read by
`pipeline/runner.py` and carried to `orchestrator/scheduler.py`; what they
*mean* is here.
"""

from __future__ import annotations

from sentinel_ai.domain.entities import Severity
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareAssessment, WelfareConcern

__all__ = ["NOTIFYING_SEVERITIES", "concerns_to_notify"]

NOTIFYING_SEVERITIES: frozenset[Severity] = frozenset(
    {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}
)
"""Spec §5's "caution band or above", named once here rather than spelled as a
bare `0.4` or an inline tuple at the one place that reads it.

*Caution band* is the console's vocabulary, not this package's: `Severity` has no
member of that name and nothing in Python defines the phrase. `web/src/lib/
severity.ts` is where it is defined — it maps `info`/`low` onto the nominal tone,
`medium` onto `caution`, and `high`/`critical` onto `breach` — so "the caution
band or above" is exactly these three members. Against `_SEVERITY_BANDS` in
`domain/entities.py` that is a threat score of 0.4 or more, and
`test_notification.py` pins the two together so a change to either band table
that broke the correspondence fails a test rather than silently moving the
threshold a person gets woken at.

Expressed in `Severity` rather than as a float because that is the form the
routed event already carries (`Event.threat.severity`), and because a second
independent copy of the 0.4 boundary is the thing most likely to drift.
"""


def concerns_to_notify(
    welfare: WelfareAssessment,
    *,
    severity: Severity,
    notify_on: frozenset[ConcernKind],
    min_confidence: Confidence,
) -> tuple[WelfareConcern, ...]:
    """The subset of `welfare.concerns` this camera should notify a human about.

    Returns the concerns themselves rather than a bool, because the note that
    goes out carries them: a caller that only learned *whether* to notify would
    have to re-derive *what about*, and could then send a note naming a kind the
    operator had muted — leaking precisely what `notify_on` exists to suppress.
    An empty result means "send nothing", which is the only sense in which this
    is a predicate.

    `evidence_stated` is deliberately not consulted. A concern the model named
    but never described (`domain/welfare.py`) is still a concern; withholding it
    would trade a possible false alarm for a possible missed collapse, and this
    system is built to prefer the first. The flag rides along on the note so the
    reader can tell the difference for themselves.

    `severity` is the whole event's, not the concern's — concerns have no
    severity, only a confidence. It is read only by the `possible` clause.
    """
    corroborated = severity in NOTIFYING_SEVERITIES
    return tuple(
        concern
        for concern in welfare.concerns
        if concern.kind in notify_on
        and concern.confidence.meets(min_confidence)
        # `possible` alone is not enough, whatever the camera's own bar is: a
        # camera that lowered `notify_min_confidence` to `possible` asked to be
        # told about maybes, not to be told about every maybe in an otherwise
        # unremarkable scene.
        and (concern.confidence is not Confidence.POSSIBLE or corroborated)
    )
