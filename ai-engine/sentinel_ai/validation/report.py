"""Shared shapes for a validation run, so two very different subjects report alike."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BinaryOutcome", "format_confusion"]


@dataclass(frozen=True, slots=True)
class BinaryOutcome:
    """A confusion matrix and the rates that matter for a surveillance detector.

    Named for the operator's question, not the statistician's. `recall` is "of the
    things that happened, how many did it tell me about"; `false_alarm_rate` is "of the
    times nothing happened, how often did it interrupt somebody". Precision is reported
    too but is the least useful of the three here, because it moves with how often the
    event actually occurs and a validation set's base rate is not a site's.
    """

    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int

    @property
    def positives(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def negatives(self) -> int:
        return self.true_negatives + self.false_positives

    @property
    def recall(self) -> float:
        return self.true_positives / self.positives if self.positives else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_positives / self.negatives if self.negatives else 0.0

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def accuracy(self) -> float:
        total = self.positives + self.negatives
        return (self.true_positives + self.true_negatives) / total if total else 0.0


def format_confusion(outcome: BinaryOutcome, *, positive: str, negative: str) -> str:
    return "\n".join(
        [
            f"  {positive:<22} {outcome.positives:4d}   detected {outcome.true_positives:4d}"
            f"   missed {outcome.false_negatives:4d}",
            f"  {negative:<22} {outcome.negatives:4d}   quiet    {outcome.true_negatives:4d}"
            f"   flagged {outcome.false_positives:4d}",
            "",
            f"  recall           {outcome.recall:6.1%}"
            f"   ({outcome.true_positives}/{outcome.positives})",
            f"  false alarm rate {outcome.false_alarm_rate:6.1%}"
            f"   ({outcome.false_positives}/{outcome.negatives})",
            f"  precision        {outcome.precision:6.1%}",
            f"  accuracy         {outcome.accuracy:6.1%}",
        ]
    )
