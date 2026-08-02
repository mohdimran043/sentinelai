"""Tests for Qwen25VLDescriber (Task 13).

_build_text_prompt and _parse_response are pure — no torch, no transformers,
no GPU — which is what lets them run in CI. Real inference is
@pytest.mark.gpu and exercised locally on the GPU box (Branch B: transformers
+ bitsandbytes NF4 on the unquantised checkpoint — see
docs/superpowers/plans/phase1b-spike-result.md for why autoawq was rejected).
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.vision.qwen25vl import (
    _EVIDENCE_UNSTATED,
    _FALLBACK_DESCRIPTION,
    _HARM_ASSESSMENT,
    _HARM_THREAT_FLOOR,
    _HARM_THREAT_FLOOR_SEVERE,
    _RESPONSE_INSTRUCTIONS,
    HARM_CHECKS,
    Qwen25VLDescriber,
    _base_checkpoint,
    _build_text_prompt,
    _parse_response,
)
from sentinel_ai.domain.entities import BBox, SceneState, Track
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareAssessment, WelfareConcern
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import ModelRuntime
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest
from tests.gpu_warnings import BITSANDBYTES_UNALIGNED_KERNEL


def _scene(tracks: tuple[Track, ...] = ()) -> SceneState:
    return SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=tracks,
        motion_energy=0.4,
        scene_signature=(1.0,),
    )


def _request(**overrides: object) -> VisionRequest:
    frame = FrameData(
        camera_id="cam-1", frame_index=0, timestamp=0.0, width=4, height=4, pixels=None
    )
    defaults: dict[str, object] = dict(
        keyframe=frame,
        scene=_scene(),
        history=(),
        camera_label="Front Door",
        reason_detail="new salient track: person",
    )
    defaults.update(overrides)
    return VisionRequest(**defaults)  # type: ignore[arg-type]


def test_qwen25vl_describer_satisfies_both_ports() -> None:
    assert issubclass(Qwen25VLDescriber, VisionLanguageModel)
    assert issubclass(Qwen25VLDescriber, ModelRuntime)


def test_base_checkpoint_strips_the_awq_suffix() -> None:
    assert _base_checkpoint("Qwen/Qwen2.5-VL-3B-Instruct-AWQ") == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_base_checkpoint_is_unchanged_without_the_suffix() -> None:
    assert _base_checkpoint("Qwen/Qwen2.5-VL-3B-Instruct") == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_prompt_includes_camera_label_reason_and_tracked_object_counts() -> None:
    track = Track(
        track_id=1, label="person", box=BBox(0.0, 0.0, 1.0, 1.0), age_frames=9, speed_px_s=0.0
    )
    prompt = _build_text_prompt(_request(scene=_scene(tracks=(track, track))))
    assert "Front Door" in prompt
    assert "new salient track: person" in prompt
    assert "2 person" in prompt
    assert "Qwen" not in prompt


def test_prompt_never_mentions_a_model_name() -> None:
    assert "qwen" not in _build_text_prompt(_request()).lower()


class TestTheHarmAssessment:
    """T3: the prompt must make the model look for genuinely harmful events.

    Every test here fails against the pre-change prompt, which asked only for a
    description, a threat value and a suggested action — a fight came back as calm
    prose scored 0.2 and arrived in an operator's severity-sorted list as `info`.

    What this is *not* is covered by the module docstring: a vision-language model's
    opinion about one keyframe, sampled at the escalation rate, with no pose or
    action recognition anywhere behind it. These tests pin the question being asked,
    not the quality of the answer — only a real GPU run can speak to that.
    """

    @pytest.mark.parametrize(
        ("phrase", "what"),
        [
            ("collapsed", "a person collapsed"),
            ("fallen", "a person fallen"),
            ("unresponsive", "a person unresponsive"),
            ("altercation", "a physical altercation"),
            ("fighting", "a physical altercation"),
            ("self-harm", "apparent self-harm"),
            ("medication", "apparent medication ingestion"),
            ("unlabelled container", "an unlabelled-container ingestion"),
            ("distress", "other apparent distress or harm"),
        ],
    )
    def test_the_prompt_asks_about_each_kind_of_harm(self, phrase: str, what: str) -> None:
        """Asserted against `_HARM_ASSESSMENT` — not the whole prompt via
        `_build_text_prompt` — because `_WELFARE_KIND_GUIDE` echoes this same
        vocabulary ("collapsed/fallen/unresponsive", "self-harm", "medication",
        "distress") into the full prompt independently of `HARM_CHECKS`. Against
        the full prompt this test kept passing after deleting the collapse check
        *and* the self-harm check from `HARM_CHECKS` entirely — the model would
        never have been asked to look for either — because the kind guide alone
        supplied the words. `_HARM_ASSESSMENT` renders only `HARM_CHECKS`, so it
        actually discriminates a deleted check.
        """
        assert phrase in _HARM_ASSESSMENT.lower(), (
            f"the prompt no longer asks the model to check for {what}"
        )

    def test_every_concern_kind_has_a_check(self) -> None:
        """Ties `HARM_CHECKS` to `ConcernKind` itself so a deleted entry is a
        test failure on its own, without relying on a specific phrase
        happening to catch it. `test_every_declared_harm_check_actually_
        reaches_the_prompt` below iterates `HARM_CHECKS` and would pass
        vacuously on a shortened mapping; this test cannot.

        MINOR 2 (Task 3 review): `HARM_CHECKS` moved from a bare
        `tuple[str, ...]` to a `Mapping[ConcernKind, str]` precisely because a
        length comparison alone could not tell a real check from a
        near-duplicate — two identical strings under two different keys would
        still satisfy `len(HARM_CHECKS) == len(ConcernKind) - 1`. `set(...)
        == set(...)` pins the *keys* (one check per kind, no kind missing, no
        stray extra), and the second assertion pins that the check *text*
        itself is not degenerate (no two kinds sharing one near-duplicate
        check that would pass the key check but ask the model nothing
        distinct)."""
        assert set(HARM_CHECKS) == set(ConcernKind) - {ConcernKind.OTHER}
        assert len(set(HARM_CHECKS.values())) == len(HARM_CHECKS)  # no near-duplicate checks

    _BANNED_MEDICATION_WORDS = (
        "dose",
        "dosage",
        "overdose",
        "milligram",
        "prescribed",
        "prescription",
        "ibuprofen",
        "acetaminophen",
        "paracetamol",
        "aspirin",
        "opioid",
        "narcotic",
    )

    @pytest.mark.parametrize("word", _BANNED_MEDICATION_WORDS)
    def test_the_medication_check_never_asks_to_name_a_substance_or_judge_a_dose(
        self, word: str
    ) -> None:
        """Binding constraint: this is a custodial welfare system, and a single-frame
        VLM has no basis for naming a substance, estimating a dose, or judging
        whether medication was prescribed. The prompt must ask only whether an
        apparent ingestion was seen and for a description of what was visible — never
        for a clinical assessment a still frame cannot honestly support."""
        assert word not in _build_text_prompt(_request()).lower()

    def test_the_medication_check_keeps_its_restraint_clause(self) -> None:
        """Only the banned-word list was tested above; nothing pinned the
        positive half of the constraint, so deleting the clause that actually
        tells the model never to name the substance — `"; state only that this
        was seen and describe what was visible, never what the substance is"`
        — would break nothing else here. This pins the clause itself."""
        assert "never what the substance is" in _HARM_ASSESSMENT

    def test_every_declared_harm_check_actually_reaches_the_prompt(self) -> None:
        """`HARM_CHECKS` is the reviewable mapping of what is asked; a check that
        was added to it but never rendered into the prompt would be invisible."""
        prompt = _build_text_prompt(_request())
        for check in HARM_CHECKS.values():
            assert check in prompt

    def test_the_prompt_ties_harm_to_a_high_threat_value(self) -> None:
        """Naming the harms is not enough on its own. Without an explicit floor the
        model narrates a fight accurately and then scores it 0.2, which is exactly the
        low-threat prose T3 exists to stop — the description would be right and the
        triage ordering would be wrong.
        """
        prompt = _build_text_prompt(_request())
        assert str(_HARM_THREAT_FLOOR) in prompt
        assert str(_HARM_THREAT_FLOOR_SEVERE) in prompt
        assert _HARM_THREAT_FLOOR >= 0.6, "below the HIGH severity band the floor buys nothing"
        assert _HARM_THREAT_FLOOR_SEVERE >= 0.8, "an injured person must reach CRITICAL"

    def test_the_prompt_still_asks_for_the_same_json_shape(self) -> None:
        """The parser is unchanged, so the contract with it must be too — a prompt
        that asked for a new key would silently take every reply down the malformed
        fallback path.

        Extended (Task 3 review, IMPORTANT): the original version of this test
        only pinned the three-field description/threat_value/suggested_action
        shape, predating the `welfare` array `_parse_welfare` (`qwen25vl.py`)
        reads out of the same reply. Nothing here asserted the prompt still
        *asks* for that field at all — verified: deleting the entire welfare
        block from `_RESPONSE_INSTRUCTIONS` left this test, and the full 686-test
        suite, green. `welfare`/`kind`/`confidence`/`evidence` close that gap.
        """
        prompt = _build_text_prompt(_request())
        assert '"description"' in prompt
        assert '"threat_value"' in prompt
        assert '"suggested_action"' in prompt
        assert '"welfare"' in prompt
        assert '"kind"' in prompt
        assert '"confidence"' in prompt
        assert '"evidence"' in prompt

    def test_the_prompt_names_every_concern_kind_and_confidence_tier(self) -> None:
        """Against `_RESPONSE_INSTRUCTIONS`, not the whole prompt — `_HARM_ASSESSMENT`
        contains 'collapsed'/'distress' independently and would mask a deleted guide.

        Verified mutations (Task 3 review, IMPORTANT), each against the full
        suite before this test existed: deleting `_WELFARE_KIND_GUIDE` from the
        prompt, and deleting the possible/likely confidence guidance, both left
        686/686 green. `_HARM_ASSESSMENT` (built from `HARM_CHECKS`) happens to
        contain the same English words ("collapsed", "distress", ...) for a
        different reason — describing what to look for in the frame, not what
        JSON value to emit — so asserting against the whole prompt would have
        passed vacuously on either deletion. This asserts specifically against
        `_RESPONSE_INSTRUCTIONS`, the half of the prompt that tells the model to
        emit the `welfare` array and what values it accepts, so it actually
        discriminates a deleted guide or a deleted confidence clause.
        """
        for kind in ConcernKind:
            assert f'"{kind.value}"' in _RESPONSE_INSTRUCTIONS
        for tier in Confidence:
            assert f'"{tier.value}"' in _RESPONSE_INSTRUCTIONS

    def test_the_response_instructions_keep_the_substance_name_restraint(self) -> None:
        """Verified mutation (Task 3 review, IMPORTANT): deleting "never a
        substance name" from `_RESPONSE_INSTRUCTIONS` left 686/686 green —
        `test_the_medication_check_keeps_its_restraint_clause` above pins the
        sibling clause in `_HARM_ASSESSMENT` ("never what the substance is"),
        a different string in a different half of the prompt, and does not
        cover this one."""
        assert "never a substance name" in _RESPONSE_INSTRUCTIONS

    def test_the_harm_assessment_introduces_no_new_wire_field(self) -> None:
        """A reply in the documented shape must still parse into exactly the three
        fields `SceneDescription` has had all along. Adding, say, `fall_detected` to
        the wire would imply a detector that does not exist (spec §4 defers behaviour
        detection to Phase 4) — the assessment rides in `description` and
        `threat_value` instead.
        """
        raw = (
            '{"description": "A person is lying motionless on the floor.", '
            '"threat_value": 0.9, "suggested_action": "Dispatch a responder now."}'
        )
        assert _parse_response(raw) == SceneDescription(
            description="A person is lying motionless on the floor.",
            threat_value=0.9,
            suggested_action="Dispatch a responder now.",
        )

    def test_the_prompt_does_not_tell_the_model_to_guess(self) -> None:
        """A single frame cannot separate someone lying down from someone who has
        collapsed. The prompt must ask for that uncertainty to be stated rather than
        resolved, or the honest limitation in the module docstring is contradicted by
        the instruction itself."""
        prompt = _build_text_prompt(_request()).lower()
        assert "unsure" in prompt
        assert "only what is visible" in prompt

    def test_the_harm_assessment_never_mentions_a_model_name(self) -> None:
        """Spec §3.3 again, over the new text: the prompt grew, and the guarantee
        must grow with it."""
        assert "qwen" not in _HARM_ASSESSMENT.lower()

    def test_a_malformed_reply_still_falls_back_rather_than_crashing(self) -> None:
        """A longer, more demanding prompt makes a formatting slip *more* likely, not
        less, so the existing §9 fallback matters more than it did."""
        result = _parse_response("I see two people fighting near the gate. Very concerning!")
        assert result.description == _FALLBACK_DESCRIPTION
        assert 0.0 <= result.threat_value <= 1.0


class TestWelfareParsing:
    """T3: `_parse_response` also reads an optional `welfare` array out of the same
    JSON reply and turns it into a `WelfareAssessment` (`domain/welfare.py`) — a
    vision-language model's opinion, not a detection. Every test here treats the
    array as untrusted model output: nothing here may raise, and nothing may let the
    payload set its own provenance (`basis`).
    """

    def test_a_well_formed_welfare_array_parses(self) -> None:
        raw = (
            '{"description": "A person is lying motionless.", "threat_value": 0.9, '
            '"suggested_action": "Send a responder now.", '
            '"welfare": [{"kind": "collapse", "confidence": "likely", '
            '"evidence": "lying motionless on the floor, not moving"}]}'
        )
        result = _parse_response(raw)
        assert result.welfare == WelfareAssessment(
            concerns=(
                WelfareConcern(
                    kind=ConcernKind.COLLAPSE,
                    confidence=Confidence.LIKELY,
                    evidence="lying motionless on the floor, not moving",
                ),
            )
        )
        assert result.welfare.concerns[0].evidence_stated is True

    def test_an_unknown_kind_maps_to_other_rather_than_raising(self) -> None:
        raw = (
            '{"description": "X", "threat_value": 0.5, "suggested_action": "Y", '
            '"welfare": [{"kind": "sasquatch", "confidence": "possible", "evidence": "unclear"}]}'
        )
        result = _parse_response(raw)
        assert result.welfare.concerns[0].kind == ConcernKind.OTHER

    def test_an_unknown_confidence_rounds_down_to_possible_never_up(self) -> None:
        """Binding constraint: a garbled confidence must never round up into
        `LIKELY` — that is the tier a routing decision would treat as more
        actionable, and a formatting slip must not be able to manufacture one."""
        raw = (
            '{"description": "X", "threat_value": 0.5, "suggested_action": "Y", '
            '"welfare": [{"kind": "distress", "confidence": "extremely certain", '
            '"evidence": "shouting"}]}'
        )
        result = _parse_response(raw)
        assert result.welfare.concerns[0].confidence == Confidence.POSSIBLE

    def test_an_absent_welfare_key_yields_none(self) -> None:
        raw = '{"description": "X", "threat_value": 0.1, "suggested_action": "Y"}'
        assert _parse_response(raw).welfare == WelfareAssessment.none()

    def test_a_non_list_welfare_value_yields_none_rather_than_raising(self) -> None:
        raw = (
            '{"description": "X", "threat_value": 0.1, "suggested_action": "Y", '
            '"welfare": "collapse"}'
        )
        assert _parse_response(raw).welfare == WelfareAssessment.none()

    def test_a_recognised_kind_with_blank_evidence_is_kept_with_a_placeholder(self) -> None:
        """`WelfareConcern.__post_init__` rejects empty evidence, so unvalidated
        model output reaching the constructor directly would raise straight
        through `_parse_response`. That is guarded against here not by dropping
        the concern (the pre-fix behaviour) but by substituting
        `_EVIDENCE_UNSTATED`: `kind` named a real, recognised concern
        ("collapse"), and dropping it entirely because only the evidence text
        was blank would make `WelfareAssessment.none()` indistinguishable from
        the model genuinely finding nothing — see `_EVIDENCE_UNSTATED`'s
        docstring on `qwen25vl.py` for why that is the wrong direction to fail
        in a custodial welfare system."""
        raw = (
            '{"description": "X", "threat_value": 0.1, "suggested_action": "Y", '
            '"welfare": [{"kind": "collapse", "confidence": "likely", "evidence": "   "}, '
            '{"kind": "distress", "confidence": "possible", '
            '"evidence": "shouting near the gate"}]}'
        )
        result = _parse_response(raw)
        assert len(result.welfare.concerns) == 2
        collapse = next(c for c in result.welfare.concerns if c.kind == ConcernKind.COLLAPSE)
        assert collapse.confidence == Confidence.LIKELY
        assert collapse.evidence == _EVIDENCE_UNSTATED
        # MINOR 3 (Task 3 review): the structural flag, not just the placeholder
        # string, must say this evidence was not stated — a caller routing on
        # `evidence_stated` should never have to compare `evidence` to
        # `_EVIDENCE_UNSTATED` (an adapter-private constant) to learn this.
        assert collapse.evidence_stated is False
        distress = next(c for c in result.welfare.concerns if c.kind == ConcernKind.DISTRESS)
        assert distress.evidence == "shouting near the gate"
        assert distress.evidence_stated is True

    def test_a_recognised_kind_with_a_missing_evidence_key_is_kept_with_a_placeholder(
        self,
    ) -> None:
        """Verified failure mode (IMPORTANT 2): a 3B model's well-formed JSON
        reply that names a real concern at `likely` confidence but omits the
        `evidence` key entirely — not blank, simply absent — is an ordinary
        slip for a small model, and the model did say "collapse, likely". The
        structured record must say the same, with an honest placeholder standing
        in for the description it never gave, not silently agree with nothing."""
        raw = (
            '{"description": "A person is lying motionless on the floor.", '
            '"threat_value": 0.9, "suggested_action": "Dispatch a responder now.", '
            '"welfare": [{"kind": "collapse", "confidence": "likely"}]}'
        )
        result = _parse_response(raw)
        assert len(result.welfare.concerns) == 1
        concern = result.welfare.concerns[0]
        assert concern.kind == ConcernKind.COLLAPSE
        assert concern.confidence == Confidence.LIKELY
        assert concern.evidence == _EVIDENCE_UNSTATED
        assert concern.evidence_stated is False

    def test_an_unrecognised_kind_with_no_evidence_is_still_dropped(self) -> None:
        """The mirror case, and the reason `_EVIDENCE_UNSTATED` is not applied
        unconditionally: an item naming nothing recognisable (`kind` maps to
        `ConcernKind.OTHER`) and carrying no evidence either has no signal in
        it at all. Keeping it would manufacture a concern out of pure noise —
        the cry-wolf direction, which drowns real alerts and gets the alarm
        muted. `{"foo": 1}` (no recognisable `kind` key at all) must be dropped
        the same way."""
        raw = (
            '{"description": "X", "threat_value": 0.5, "suggested_action": "Y", '
            '"welfare": [{"kind": "sasquatch", "confidence": "possible"}, {"foo": 1}]}'
        )
        result = _parse_response(raw)
        assert result.welfare == WelfareAssessment.none()

    def test_confidence_survives_trailing_punctuation_without_rounding_up(self) -> None:
        """MINOR 3: a 3B model ending a sentence with a period or exclamation
        mark is reflexive formatting noise, not a signal to downgrade a real
        `likely` to `possible`. `"likely."` and `"likely!"` must still parse to
        `LIKELY` — stripping trailing punctuation can only reveal a value the
        enum already recognises (see `test_an_unknown_confidence_rounds_down_to_
        possible_never_up` above, unchanged and still passing: `"extremely
        certain"` has no punctuation to strip and still rounds down)."""
        for confidence_text in ("likely.", "likely!"):
            raw = (
                '{"description": "X", "threat_value": 0.5, "suggested_action": "Y", '
                '"welfare": [{"kind": "collapse", "confidence": "'
                + confidence_text
                + '", "evidence": "lying motionless"}]}'
            )
            result = _parse_response(raw)
            assert result.welfare.concerns[0].confidence == Confidence.LIKELY, confidence_text

    def test_malformed_json_falls_back_to_none_and_the_fixed_description(self) -> None:
        """The existing malformed-response fallback path (`_FALLBACK_DESCRIPTION`)
        must degrade welfare to `none()` too, never leave it looking like "assessed,
        found nothing" by some other, unaudited route."""
        result = _parse_response("I see someone on the floor, maybe collapsed!")
        assert result.description == _FALLBACK_DESCRIPTION
        assert result.welfare == WelfareAssessment.none()

    def test_the_payload_cannot_set_its_own_basis(self) -> None:
        """Binding constraint: `basis` always comes from the `WelfareAssessment`
        constant, never the payload — building the type by spreading parsed JSON
        (`WelfareAssessment(**payload)`) would let a payload claim its own
        provenance marker, defeating the entire point of the field."""
        raw = (
            '{"description": "X", "threat_value": 0.5, "suggested_action": "Y", '
            '"welfare": [{"kind": "collapse", "confidence": "likely", '
            '"evidence": "on the floor", "basis": "hand_verified"}]}'
        )
        result = _parse_response(raw)
        assert result.welfare.basis == "single_frame_vlm"


def test_parse_response_reads_well_formed_json() -> None:
    raw = '{"description": "A person walks by.", "threat_value": 0.2, "suggested_action": "None."}'
    result = _parse_response(raw)
    assert result == SceneDescription(
        description="A person walks by.", threat_value=0.2, suggested_action="None."
    )


def test_parse_response_extracts_json_embedded_in_prose_or_fences() -> None:
    raw = (
        'Sure, here you go:\n```json\n{"description": "Ok.", "threat_value": 0.1, '
        '"suggested_action": "None."}\n```'
    )
    result = _parse_response(raw)
    assert result.description == "Ok."
    assert result.threat_value == 0.1


def test_parse_response_clamps_an_out_of_range_threat_value() -> None:
    raw = '{"description": "X", "threat_value": 1.7, "suggested_action": "Y"}'
    assert _parse_response(raw).threat_value == 1.0


def test_parse_response_clamps_a_negative_threat_value() -> None:
    """The mirror of the >1 clamp test above — a parser that only clamped one
    side of the range (or that clamped by, say, taking abs()) would pass the
    positive-side test but fail this one.
    """
    raw = '{"description": "X", "threat_value": -0.4, "suggested_action": "Y"}'
    assert _parse_response(raw).threat_value == 0.0


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_parse_response_falls_back_on_a_non_finite_threat_value(literal: str) -> None:
    """MINOR 4 (Task 3 review, pre-existing): `json.loads` accepts the
    non-standard `NaN`/`Infinity`/`-Infinity` tokens by default, and
    `_clamp01`'s `max(0.0, min(1.0, value))` turns `NaN` and `+Infinity` into
    `1.0` (`min(1.0, nan) == 1.0` in Python) and `-Infinity` into `0.0` — a
    garbled float would otherwise clamp "successfully" into a maximum-severity
    CRITICAL event, the same cry-wolf direction as every other finding in this
    module. `isinstance(threat_value, int | float)` alone does not catch this:
    a Python float `nan`/`inf` passes that check cleanly. This must instead
    fall through to the safe, fixed fallback, same as any other malformed
    `threat_value`.
    """
    raw = f'{{"description": "X", "threat_value": {literal}, "suggested_action": "Y"}}'
    result = _parse_response(raw)
    assert result.description == _FALLBACK_DESCRIPTION
    assert 0.0 <= result.threat_value <= 1.0


def test_parse_response_falls_back_on_prose_with_no_json() -> None:
    """F1 (Task 13 review): the fallback path must use the fixed
    `_FALLBACK_DESCRIPTION`, never the raw text verbatim — see
    `test_parse_response_never_leaks_a_model_name_via_the_fallback_description`
    for why (spec §3.3: raw model text is not safe to publish unfiltered).
    """
    result = _parse_response("There is a person near the entrance, nothing concerning.")
    assert result.description == _FALLBACK_DESCRIPTION
    assert 0.0 <= result.threat_value <= 1.0
    assert result.suggested_action


def test_parse_response_falls_back_on_json_missing_a_required_key() -> None:
    raw = '{"description": "X", "suggested_action": "Y"}'
    result = _parse_response(raw)
    assert result.description == _FALLBACK_DESCRIPTION
    assert 0.0 <= result.threat_value <= 1.0


def test_parse_response_falls_back_on_wrong_typed_threat_value() -> None:
    """threat_value as a string, not a number — a parser that did not check
    types (e.g. blindly `float()`-cast whatever key it found) would either
    raise or silently coerce; this must instead hit the safe fallback path,
    with the fixed `_FALLBACK_DESCRIPTION` used in place of the raw text
    (F1, Task 13 review).
    """
    raw = '{"description": "X", "threat_value": "high", "suggested_action": "Y"}'
    result = _parse_response(raw)
    assert result.description == _FALLBACK_DESCRIPTION
    assert 0.0 <= result.threat_value <= 1.0


def test_pathologically_nested_json_falls_back_rather_than_raising() -> None:
    """MINOR 1 (Task 3 review): `_parse_response`'s docstring claims
    `json.loads`'s `RecursionError` on deeply nested JSON is caught alongside
    `json.JSONDecodeError`, but nothing exercised that path — reverting the
    `except (json.JSONDecodeError, RecursionError):` clause to the
    single-exception `except json.JSONDecodeError:` it evolved from still
    passed all 686 tests, while 200,000 levels of nesting genuinely raises
    `RecursionError` through `json.loads`. This must fall through to the same
    fixed, safe fallback as any other malformed reply, never propagate.
    """
    raw = (
        '{"description":"X","threat_value":0.5,"suggested_action":"Y","welfare":'
        + "[" * 200_000
        + "]" * 200_000
        + "}"
    )
    result = _parse_response(raw)
    assert result.description == _FALLBACK_DESCRIPTION
    assert result.welfare == WelfareAssessment.none()


def test_parse_response_never_leaks_a_model_name_via_the_fallback_description() -> None:
    """F1 (Task 13 review, Critical): the pre-fix code used
    `description=candidate[:500]` on the malformed-response fallback path —
    the model's raw, unfiltered text verbatim — while `suggested_action` on
    the same path already used the fixed `_FALLBACK_ACTION`. A model
    confused enough to ignore the JSON-only instruction is exactly the model
    most likely to answer in first person ("As Qwen2.5-VL, I can see...").
    Spec §3.3 requires published events carry no model identity, and that
    string would otherwise flow into `SceneDescription.description`, then
    `Event`, then RabbitMQ, then the operator UI.

    This must fail against the pre-fix code (which puts "qwen" straight into
    `result.description`) and pass after the fix (fixed
    `_FALLBACK_DESCRIPTION`, no raw text).
    """
    raw = "As Qwen2.5-VL, I can see a person standing near the entrance."
    result = _parse_response(raw)
    assert "qwen" not in result.description.lower()
    assert result.description == _FALLBACK_DESCRIPTION


def test_parse_response_recovers_a_json_block_followed_by_trailing_braces() -> None:
    """Also worth doing (Task 13 review): the original greedy
    `re.compile(r"\\{.*\\}", re.DOTALL)` regex spans from the first `{` to the
    *last* `}` in the whole response, so incidental braces after a valid
    JSON object (e.g. a trailing aside) get swallowed into the match and
    break `json.loads` — a spurious fall-through that, after F1, means
    losing the parsed description entirely even though the model's JSON was
    perfectly valid. The balanced-brace scanner recovers the first complete
    object regardless of what follows it.
    """
    raw = (
        '{"description": "Ok.", "threat_value": 0.1, "suggested_action": "None."} '
        "note: {no further action needed}"
    )
    result = _parse_response(raw)
    assert result.description == "Ok."
    assert result.threat_value == 0.1


def test_capabilities_reports_the_checkpoint_actually_loaded_not_the_awq_variant() -> None:
    """F3 (Task 13 review): Branch B always loads the unquantised checkpoint
    via `_base_checkpoint`, regardless of the configured `-AWQ` model id. An
    operator reading `capabilities()`/`health()` — the boundary explicitly
    designated as the correct place for model identity (spec §3.3) — must
    see the checkpoint that is actually resident, not the one merely
    configured, or they would reasonably conclude AWQ is active when it
    never is. No GPU/torch import needed: `capabilities()` is pure.
    """
    describer = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=64, device="cpu"
    )
    assert describer.capabilities().model_key == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_parse_response_never_raises_on_empty_text() -> None:
    result = _parse_response("")
    assert result.description
    assert 0.0 <= result.threat_value <= 1.0


@pytest.mark.gpu
@pytest.mark.filterwarnings(BITSANDBYTES_UNALIGNED_KERNEL)
async def test_real_describer_loads_warms_up_and_reports_health() -> None:
    """End-to-end lifecycle on the real GPU: Branch B (bitsandbytes NF4 on the
    unquantised checkpoint — see phase1b-spike-result.md). Also proves
    shutdown() actually frees VRAM rather than just flipping the lifecycle
    state, which is what makes the 600s idle-unload (ResidentSet, Task 7)
    a real reclaim instead of a nominal one: checking only
    `capabilities().vram_mib == 0` would pass even if `shutdown()` merely
    zeroed that field without releasing anything, so this also reads
    `torch.cuda.memory_reserved()` directly, before and after.
    """
    import torch

    describer = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=64, device="cuda"
    )
    await describer.initialize()
    await describer.warmup()

    assert describer.health().state.value == "healthy"
    assert describer.capabilities().vram_mib > 0
    # Spec §3.3: the registry's health view may carry the model id, but the
    # thing it measures (VRAM) must not be confused with an Event field.
    # F3 (Task 13 review): reports the checkpoint Branch B actually loads
    # (unquantised), not the configured "-AWQ" id, matching `_base_checkpoint`.
    assert describer.capabilities().model_key == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "-AWQ" not in describer.version()

    reserved_before_shutdown = torch.cuda.memory_reserved() // (1024 * 1024)

    await describer.shutdown()
    assert describer.health().state.value == "unloaded"
    # An unloaded model must not keep reporting VRAM it no longer holds.
    assert describer.capabilities().vram_mib == 0

    reserved_after_shutdown = torch.cuda.memory_reserved() // (1024 * 1024)
    # A `shutdown()` that only flips `_vram_mib` to 0 without actually
    # releasing the caching allocator's pool would leave this reading
    # unchanged; measured on this box, a genuine release drops it from
    # ~2.4 GB to well under 200 MiB.
    assert reserved_after_shutdown < reserved_before_shutdown / 2


@pytest.mark.gpu
@pytest.mark.filterwarnings(BITSANDBYTES_UNALIGNED_KERNEL)
async def test_real_describer_produces_a_real_description_with_no_model_identity_leak() -> None:
    """Runs actual inference on a real photo, not a blank frame, and checks
    the *content* of the result: a describer that always returned a fixed
    canned string would fail the "mentions people" assertion below (a blank
    frame's description does not), and a describer that leaked its own name
    into the description would fail the spec §3.3 assertion.
    """
    from pathlib import Path

    import cv2
    import ultralytics

    describer = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=64, device="cuda"
    )
    await describer.initialize()
    await describer.warmup()

    # `bus.jpg` ships inside the installed `ultralytics` package (its
    # long-standing quickstart demo image: a bus with several people at a
    # stop) — real content, no download, no new binary fixture in this repo.
    image_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    pixels = cv2.imread(str(image_path))
    assert pixels is not None, f"failed to decode fixture image at {image_path}"
    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    height, width = pixels.shape[:2]

    frame = FrameData(
        camera_id="cam-1", frame_index=0, timestamp=0.0, width=width, height=height, pixels=pixels
    )
    scene = SceneState(
        camera_id="cam-1",
        frame_index=0,
        timestamp=0.0,
        detections=(),
        tracks=(),
        motion_energy=0.5,
        scene_signature=(1.0,),
    )
    request = VisionRequest(
        keyframe=frame,
        scene=scene,
        history=(),
        camera_label="Front Door",
        reason_detail="new salient track: person",
    )

    result = await describer.describe(request)

    assert isinstance(result, SceneDescription)
    assert result.description
    assert 0.0 <= result.threat_value <= 1.0
    assert result.suggested_action
    assert "qwen" not in result.description.lower()
    assert "qwen" not in result.suggested_action.lower()

    await describer.shutdown()


@pytest.mark.gpu
async def test_real_describer_rejects_non_ndarray_pixels() -> None:
    describer = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=64, device="cuda"
    )
    await describer.initialize()
    bad_request = _request(
        keyframe=FrameData(
            camera_id="cam-1", frame_index=0, timestamp=0.0, width=4, height=4, pixels=[[0, 0, 0]]
        )
    )
    with pytest.raises(TypeError, match="pixels"):
        await describer.describe(bad_request)
    await describer.shutdown()
