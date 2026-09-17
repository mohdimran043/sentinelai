"""The arithmetic behind every accuracy number this project publishes.

Small, but not trivial to get right, and wrong here means wrong in `docs/performance.md`
— which is where somebody decides whether to rely on a fall detector.
"""

from __future__ import annotations

from sentinel_ai.validation.report import BinaryOutcome, format_confusion


class TestRates:
    def test_the_worked_example_from_the_urfall_run(self) -> None:
        """The real numbers, so a refactor that swaps two fields is caught by the case
        the documentation quotes."""
        outcome = BinaryOutcome(
            true_positives=29, false_negatives=1, true_negatives=17, false_positives=23
        )
        assert outcome.positives == 30
        assert outcome.negatives == 40
        assert round(outcome.recall, 3) == 0.967
        assert round(outcome.false_alarm_rate, 3) == 0.575
        assert round(outcome.precision, 3) == 0.558
        assert round(outcome.accuracy, 3) == 0.657

    def test_a_detector_that_never_fires_scores_zero_recall_not_an_error(self) -> None:
        """The shipped fall detector's actual result on URFall. A harness that divided
        by zero here would have hidden the finding."""
        outcome = BinaryOutcome(
            true_positives=0, false_negatives=30, true_negatives=40, false_positives=0
        )
        assert outcome.recall == 0.0
        assert outcome.false_alarm_rate == 0.0
        assert outcome.precision == 0.0

    def test_no_negatives_at_all_does_not_divide_by_zero(self) -> None:
        outcome = BinaryOutcome(
            true_positives=5, false_negatives=1, true_negatives=0, false_positives=0
        )
        assert outcome.false_alarm_rate == 0.0
        assert outcome.accuracy == 5 / 6

    def test_an_empty_run_is_all_zeros_rather_than_a_crash(self) -> None:
        outcome = BinaryOutcome(0, 0, 0, 0)
        assert (outcome.recall, outcome.precision, outcome.accuracy) == (0.0, 0.0, 0.0)


class TestFormatting:
    def test_the_report_names_both_classes_it_is_counting(self) -> None:
        """A confusion matrix whose rows are unlabelled is one somebody reads backwards."""
        text = format_confusion(
            BinaryOutcome(29, 1, 17, 23), positive="falls", negative="daily activities"
        )
        assert "falls" in text
        assert "daily activities" in text
        assert "recall" in text
        assert "false alarm rate" in text
