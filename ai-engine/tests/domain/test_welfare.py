from __future__ import annotations

import pytest

from sentinel_ai.domain.welfare import (
    ConcernKind,
    Confidence,
    WelfareAssessment,
    WelfareConcern,
)


def _concern(
    kind: ConcernKind = ConcernKind.COLLAPSE,
    confidence: Confidence = Confidence.POSSIBLE,
    evidence: str = "person lying motionless on the floor",
) -> WelfareConcern:
    return WelfareConcern(kind=kind, confidence=confidence, evidence=evidence)


class TestWelfareConcern:
    def test_construction(self) -> None:
        concern = _concern()
        assert concern.kind is ConcernKind.COLLAPSE
        assert concern.confidence is Confidence.POSSIBLE
        assert concern.evidence == "person lying motionless on the floor"

    def test_is_immutable(self) -> None:
        concern = _concern()
        with pytest.raises(AttributeError):
            concern.evidence = "changed"  # type: ignore[misc]

    def test_empty_evidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence"):
            _concern(evidence="")

    def test_whitespace_only_evidence_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="evidence"):
            _concern(evidence="   \n\t  ")


class TestConcernKind:
    def test_covers_the_expected_kinds(self) -> None:
        assert {k.value for k in ConcernKind} == {
            "collapse",
            "altercation",
            "self_harm",
            "medication",
            "distress",
            "other",
        }


class TestConfidence:
    def test_has_exactly_possible_and_likely(self) -> None:
        """No `CERTAIN`: a single still frame cannot honestly support it."""
        assert {c.value for c in Confidence} == {"possible", "likely"}


class TestWelfareAssessment:
    def test_construction(self) -> None:
        concern = _concern()
        assessment = WelfareAssessment(concerns=(concern,))
        assert assessment.concerns == (concern,)

    def test_basis_is_always_single_frame_vlm(self) -> None:
        assessment = WelfareAssessment(concerns=())
        assert assessment.basis == "single_frame_vlm"

    def test_a_basis_other_than_single_frame_vlm_is_rejected_at_runtime(self) -> None:
        """`basis: Literal["single_frame_vlm"]` is enforced by mypy only — nothing
        stops `WelfareAssessment(concerns=(), basis="anything")` at runtime unless
        `__post_init__` checks it too. This module deserializes into untrusted-JSON
        boundaries (the event codec, later the VLM's own response), so the guard
        earns its keep here rather than relying on every caller to remember."""
        with pytest.raises(ValueError, match="basis"):
            WelfareAssessment(concerns=(), basis="anything")  # type: ignore[arg-type]

    def test_is_immutable(self) -> None:
        assessment = WelfareAssessment(concerns=())
        with pytest.raises(AttributeError):
            assessment.concerns = (_concern(),)  # type: ignore[misc]

    def test_none_has_no_concerns(self) -> None:
        assessment = WelfareAssessment.none()
        assert assessment.concerns == ()
        assert assessment.basis == "single_frame_vlm"

    def test_duplicate_kinds_collapse_to_the_highest_confidence(self) -> None:
        low = _concern(
            kind=ConcernKind.COLLAPSE,
            confidence=Confidence.POSSIBLE,
            evidence="lying still near the doorway",
        )
        high = _concern(
            kind=ConcernKind.COLLAPSE,
            confidence=Confidence.LIKELY,
            evidence="not moving after apparent fall",
        )
        assessment = WelfareAssessment(concerns=(low, high))
        assert len(assessment.concerns) == 1
        assert assessment.concerns[0].confidence is Confidence.LIKELY

    def test_duplicate_kinds_collapse_regardless_of_order(self) -> None:
        low = _concern(kind=ConcernKind.DISTRESS, confidence=Confidence.POSSIBLE)
        high = _concern(kind=ConcernKind.DISTRESS, confidence=Confidence.LIKELY)
        assessment = WelfareAssessment(concerns=(high, low))
        assert len(assessment.concerns) == 1
        assert assessment.concerns[0].confidence is Confidence.LIKELY

    def test_highest_confidence_returns_none_for_an_absent_kind(self) -> None:
        assessment = WelfareAssessment.none()
        assert assessment.highest_confidence(ConcernKind.COLLAPSE) is None

    def test_highest_confidence_returns_the_matching_concerns_confidence(self) -> None:
        concern = _concern(kind=ConcernKind.SELF_HARM, confidence=Confidence.LIKELY)
        assessment = WelfareAssessment(concerns=(concern,))
        assert assessment.highest_confidence(ConcernKind.SELF_HARM) is Confidence.LIKELY
        assert assessment.highest_confidence(ConcernKind.MEDICATION) is None
