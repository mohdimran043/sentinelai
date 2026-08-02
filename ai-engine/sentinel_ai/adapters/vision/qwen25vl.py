"""Qwen2.5-VL vision-language describer (spec §5; ports/vision_llm.py, ports/model_runtime.py).

Loading path is Branch B — settled by Task 1's spike
(`docs/superpowers/plans/phase1b-spike-result.md`): `autoawq` installed and
loaded weights, but inference died inside its own bundled Triton GEMM kernel
against the installed Triton 3.7.1 (four distinct failures across three cheap
fixes, the last one landing at `lm_head` with a nonsensical fp16/fp16 type
error internal to that kernel). `initialize()` below instead loads the
unquantised `Qwen/Qwen2.5-VL-3B-Instruct` checkpoint through plain
`transformers` with `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)` — verified
end to end on the target GPU (694 MiB baseline -> 3359 MiB loaded -> 3517 MiB
inference peak, real caption produced). Every other method here is identical
regardless of which branch loads the weights — that is the entire reason the
`VisionLanguageModel` port abstraction exists (spec §7).

Heavy ML dependencies (`torch`, `transformers`, `PIL`) are imported lazily
inside the methods that need them, never at module scope — see
`adapters/detectors/yolo11.py`'s module docstring for why: it is what makes
`_build_text_prompt` and `_parse_response` unit-testable on CPU in CI with no
`gpu` extra installed.

Spec §3.3: the event carries no model identity. `_build_text_prompt` and
`_parse_response` never mention a model name, and neither does the
`SceneDescription` this class returns; `version()` and
`Capabilities.model_key` are orchestrator-internal (the registry's health
view) — a different boundary from the published event.

Harm assessment: what it is, and what it is not
-----------------------------------------------
`_build_text_prompt` asks the model to check the keyframe specifically for a
person who is collapsed, fallen or unresponsive, for a physical altercation
between people, and for other apparent distress or harm, and to weight
`threat_value` upwards when it sees one. That is worth having — without it a
fight comes back as calm prose scored 0.2, which is worse than useless for
triage. But it is **a vision-language model's opinion about one still frame**,
and nothing downstream may treat it as a trained classifier:

  * **Sampled, not continuous.** The frame the model sees is the keyframe of an
    escalation the gate already chose to spend a GPU slot on (spec §4.1's seven
    triggers), and `CameraProfile`'s token bucket and cooldown bound how often
    that happens — roughly once per 10 s per camera at most. A fall that begins
    and ends between two escalations is never looked at by anything.
  * **No pose or action recognition anywhere in this phase.**
    `adapters/detectors/yolo11.py` is an object detector: it reports that a
    `person` box exists, never what that person is doing. Nothing in Phase 1B
    estimates pose, tracks limbs, or classifies actions. Spec §4 defers
    behaviour detection and the 18 anomaly detectors to Phase 4, and this
    prompt is not a down payment on them.
  * **Fallible in both directions.** A single frame cannot reliably separate
    someone lying down from someone who has collapsed, or horseplay from an
    assault, and the model will confidently assert either. Treat a high
    `threat_score` as a reason to look at the clip, never as a finding.
  * **No new event reason.** The seven `EscalationReason` values are unchanged.
    There is deliberately no `fight_detected` reason, because a reason of that
    name would imply a detector that does not exist.

The assessment reaches the wire only through the fields that already exist:
prose in `description`, weighting in `threat_score`/`severity`. Nothing was
added to `contracts/events/anomaly_event.schema.json` for it, precisely because
a dedicated structured field (`fall_detected: true`) would read to the Phase 1C
consumer as a detector output with a detector's reliability.

T3 extends this same prompt to ask about two more things by name — apparent
self-harm and apparent medication or unlabelled-container ingestion — and
parses an optional structured `welfare` array out of the same JSON reply into
a `WelfareAssessment` (`domain/welfare.py`, `ports/vision_llm.py`'s
`SceneDescription.welfare`). Everything above still applies unchanged: this is
still one still frame, still no pose or action recognition, still fallible in
both directions, and `WelfareConcern.confidence` is still `possible`/`likely`
because a single frame can never honestly be `CERTAIN` — see `domain/welfare.py`
for why that tier does not exist.

The medication check is deliberately narrower than every other check here.
The prompt does not ask the model to name a substance, estimate a dose, or
judge whether medication was prescribed — a single-frame VLM has no clinical
basis for any of those, and a system in a custodial welfare setting that
records one invites a reader to act on it as if it were a clinical finding.
It asks only whether an apparent ingestion was seen, and for a description of
what was visible: the container, the action, nothing more.

`_parse_welfare` treats the `welfare` array exactly the way `_parse_response`
already treats the rest of the reply: untrusted text, never trusted structure.
An unrecognised `kind` becomes `ConcernKind.OTHER`, an unrecognised
`confidence` becomes `Confidence.POSSIBLE` — the weaker tier, always rounded
down, never up into a routing decision that wakes someone at 3am or, just as
bad, suppresses a real concern under a false `LIKELY`. A concern whose
`evidence` is absent, null, non-string or blank is handled two different ways
depending on whether `kind` named something real: if `kind` parsed to
`ConcernKind.OTHER` — nothing recognisable was named either — the item carries
no signal at all and is dropped, same as before. But if `kind` parsed to a
*recognised* kind, dropping it would be the wrong direction to fail in a
custodial welfare system: a reply of `{"kind": "collapse", "confidence":
"likely"}` with no `evidence` key is the model naming a real concern and
merely omitting the description of what it saw, an ordinary slip for a small
model — dropping it silently degrades to `WelfareAssessment.none()`, which the
event codec then omits from the wire entirely, and the schema's own words say
an omitted welfare object reads as "not assessed". "Collapse, likely" becoming
indistinguishable from "nothing found" fails toward false reassurance, which
is worse than the alternative of raising a concern whose evidence field
honestly says the model did not describe what it saw. So that case is kept,
with `_EVIDENCE_UNSTATED` substituted for the missing text — a fixed, honest
placeholder that can never leak model text, matching the discipline
`_FALLBACK_DESCRIPTION` already keeps for the description path.

A missing or malformed `welfare` key yields `WelfareAssessment.none()`
without ever failing the surrounding description. `WelfareAssessment` is built
field by field from validated values, never `WelfareAssessment(**payload)`:
`basis` always comes from the type's own constant, never the payload, or a
garbled or adversarial reply could set its own provenance marker and defeat
the entire point of the field. The malformed-response fallback path
(`_FALLBACK_DESCRIPTION`) degrades welfare to `none()` too, for the same
reason F1 fixed the description on that path — an assessment that silently
vanished into "found nothing" would be indistinguishable from one that
genuinely found nothing.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from sentinel_ai.domain.entities import SceneState
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareAssessment, WelfareConcern
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import (
    Capabilities,
    HealthReport,
    LifecycleState,
    ModelRuntime,
)
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest

logger = logging.getLogger(__name__)

_AWQ_SUFFIX = "-AWQ"
_FALLBACK_ACTION = "Review the clip manually — automated threat assessment unavailable."
_FALLBACK_DESCRIPTION = (
    "Automated description unavailable — the vision model's reply did not match "
    "the expected format."
)
"""Fixed stand-in for `SceneDescription.description` on the malformed-response
fallback path (Task 13 review, finding F1).

The pre-fix code used the model's raw text verbatim here (truncated to 500
chars) while `suggested_action` on the same path already used the fixed
`_FALLBACK_ACTION` — an asymmetry that let raw, unfiltered model output reach
`SceneDescription`, then `Event`, then RabbitMQ, then the operator UI. Spec
§3.3 requires published events carry no model identity, and a model confused
enough to ignore the JSON instruction is exactly the model most likely to
answer in first person ("As Qwen2.5-VL, I can see..."). Sanitising the raw
text instead of replacing it was considered and rejected: it would require
enumerating every way a model might name itself, an open-ended problem,
whereas a fixed fallback cannot leak by construction. The cost is losing
whatever real scene detail the raw text might have carried on this
(malformed-response) path only; the raw text is still logged at DEBUG level
in `_parse_response` for operators — logs are operator-facing infrastructure,
not the published event, so that is not a spec §3.3 concern.
"""

HARM_CHECKS: Mapping[ConcernKind, str] = MappingProxyType(
    {
        ConcernKind.COLLAPSE: (
            "a person who is collapsed, fallen, lying on the ground, or appears unresponsive"
        ),
        ConcernKind.ALTERCATION: (
            "a physical altercation between people — fighting, striking, grappling, pushing"
        ),
        ConcernKind.SELF_HARM: (
            "apparent self-harm — a person cutting, striking, or otherwise deliberately "
            "injuring themselves"
        ),
        ConcernKind.MEDICATION: (
            "a person appearing to swallow pills, liquid, or the contents of an unlabelled "
            "container — an apparent medication or unlabelled-container ingestion; state "
            "only that this was seen and describe what was visible, never what the "
            "substance is"
        ),
        ConcernKind.DISTRESS: (
            "any other apparent distress or harm to a person — someone being restrained or "
            "dragged, someone clutching an injury, someone fleeing, a weapon held or raised"
        ),
    }
)
"""The five things the prompt makes the model look for by name, one per
`ConcernKind` except `OTHER` (which is not a check to look for — it is what
`_parse_concern_kind` falls back to when the model names something none of
these five cover).

Named and enumerated rather than buried in one long paragraph so the set is
reviewable, testable, and extendable without rewriting the prompt around it.
Public (no underscore) because `tests/adapters/vision/test_qwen25vl.py` asserts
every entry actually reaches the prompt: a check that silently stopped being
asked for would be invisible otherwise, and this is the whole of T3's behaviour.

Keyed by `ConcernKind` rather than a bare tuple (Task 3 review, MINOR 2): a
`tuple[str, ...]` has no structural link to the enum it is meant to cover, so
`len(HARM_CHECKS) == len(ConcernKind) - 1` could not tell a real, distinct
check from an accidental duplicate — two identical strings would still pass
that length assertion. A mapping keyed by `ConcernKind` makes "one check per
kind" a property of the type itself, checkable as `set(HARM_CHECKS) ==
set(ConcernKind) - {ConcernKind.OTHER}`, and `_WELFARE_KIND_GUIDE` below is
now *built from this mapping* rather than hand-typed a second time — that
second hand-typed list was the root cause of the review finding this module's
own history records: a deleted `HARM_CHECKS` entry left the guide's echo of
the same vocabulary in place, so a test asserting against the *whole prompt*
kept passing even though the model was no longer being asked to look for that
harm at all.

These are *questions put to a vision-language model about one frame*, not
detector outputs. See the module docstring for exactly what that does and does
not buy, and for why the medication check is worded the way it is: no
substance name, no dose, no judgement about whether it was prescribed.
"""


def _format_kind_guide() -> str:
    """Build the `kind` vocabulary sentence in `_RESPONSE_INSTRUCTIONS` from
    `HARM_CHECKS` itself, not a second hand-typed list of the same five
    concepts — see `HARM_CHECKS`'s docstring for why that duplication was the
    root cause of an earlier review finding. Reusing each check's own text
    (rather than a hand-authored paraphrase) means a check deleted from
    `HARM_CHECKS` disappears from the guide too, in the same edit, instead of
    needing a second, easy-to-forget deletion.
    """
    named = [f'"{kind.value}" for {text}' for kind, text in HARM_CHECKS.items()]
    return ", ".join(named) + f', and "{ConcernKind.OTHER.value}" only if none of those fit'


_WELFARE_KIND_GUIDE = _format_kind_guide()
"""Derived from `HARM_CHECKS`, not hand-typed strings, so the prompt's
vocabulary can never drift from `domain/welfare.py`'s enum — a hand-typed "self_harm"
here that the enum later renamed would silently stop round-tripping through
`_parse_welfare`, which maps anything it does not recognise to `OTHER`."""

_HARM_THREAT_FLOOR = 0.7
_HARM_THREAT_FLOOR_SEVERE = 0.85
"""Floors asked for, not floors enforced. 0.7 lands in `Severity.HIGH` and 0.85
in `Severity.CRITICAL` (`domain/entities.py`'s bands), which is the point:
without them the model narrates a fight accurately and then scores it 0.3, and
the event arrives as `low` in an operator's list sorted by severity.

Only the *published* severity moves. The escalation gate decides whether to
describe a frame at all long before a threat value exists (spec §4.1), so this
cannot make a camera escalate more often.

Measured on this box (RTX 4060, Qwen2.5-VL-3B NF4, greedy decode, same frames
through the old prompt and this one):

    two people grappling      0.3 "possibly a sport" -> 0.9 "a physical altercation"
    a person lying motionless 0.2                    -> 0.6
    a person seated on grass  0.2                    -> 0.0
    two men shouting, no contact  0.3                -> 0.3
    an ordinary concourse     0.2                    -> 0.2
    a stretcher being carried 0.2                    -> 0.3   (missed)

Read that honestly. The two clearest harms move from `low` to `high`/`critical`
and none of the ordinary scenes moves at all — but the lying-down frame came
back at 0.6, *under* the 0.7 the prompt asks for, and the stretcher was missed
outright. A 3B model treats these numbers as suggestions. They are worth stating
because they shift the distribution the right way; they are not a guarantee, and
nothing downstream may be written as though they were.
"""


def _format_harm_checks() -> str:
    return "\n".join(f"- {check};" for check in HARM_CHECKS.values())


_HARM_ASSESSMENT = (
    "Before you answer, look at the frame specifically for each of the following, "
    "and state plainly in the description whether you see it:\n"
    f"{_format_harm_checks()}\n"
    f"If any of these is present, treat the situation as serious: set threat_value to "
    f"at least {_HARM_THREAT_FLOOR}, and to at least {_HARM_THREAT_FLOOR_SEVERE} when a "
    "person appears injured, unresponsive, or under attack. Report only what is visible "
    "in this frame — if you are unsure whether someone has fallen or is simply sitting "
    "or crouching, say which you think it is and why, rather than asserting either. "
    "If none of these is present, say so, and score the frame on ordinary security "
    "grounds instead."
)

_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON object of this exact shape, no other text:\n"
    '{"description": "<one or two plain sentences describing what is visible>", '
    '"threat_value": <float between 0.0 and 1.0>, '
    '"suggested_action": "<one short, concrete sentence for a human reviewer>", '
    '"welfare": [{"kind": "<concern kind>", "confidence": "<possible|likely>", '
    '"evidence": "<what you actually saw>"}]}\n'
    "Include one welfare entry for each concern from the checks above that you "
    "actually observed in this frame; use an empty array when none apply. For "
    f"kind use {_WELFARE_KIND_GUIDE}. For confidence use "
    f'"{Confidence.LIKELY.value}" when what you see clearly supports it, and '
    f'"{Confidence.POSSIBLE.value}" when an innocent explanation is equally '
    "consistent. For the medication kind, evidence must describe only what "
    "was visible — never a substance name."
)
"""`Confidence.LIKELY`'s bar used to be worded as being *sure* — but
`domain/welfare.py` has no `Confidence.CERTAIN` precisely because a single
frame can never earn that, so asking the model to be "sure" before it will
say `likely` set a bar the prompt's own domain model says is unreachable, and
could only push genuine concerns down into `possible` or out of the array
altogether. "Clearly supports it" vs. "an innocent explanation is equally
consistent" asks the model to compare explanations instead of asserting
certainty it cannot honestly have."""

_CUDA_CONTEXT_OVERHEAD_MIB = 300
"""Fixed tax added on top of `torch.cuda.memory_reserved()` in `warmup()`.

Same rationale as `adapters/detectors/yolo11.py`'s constant of the same name
(Task 11's review finding, corrected there): `memory_allocated()` only counts
live tensors and excludes the caching allocator's reserved pool; even
`memory_reserved()` still misses the CUDA context itself (~150-300 MiB,
created on first kernel launch and held for the process lifetime) and
cuDNN/cuBLAS workspace buffers, both of which are real and visible to
`nvidia-smi` but surfaced by no torch-level API.

`capabilities().vram_mib` feeds `plan_residency()`
(`sentinel_ai/domain/policy/vram_budget.py`), which does hard
admission-control arithmetic against the total VRAM budget: this constant is
a deliberate, documented over-estimate because the planner evicting a model
too early is recoverable, an OOM mid-escalation from under-reporting is not.
Measured on an RTX 4060 with the desktop session live (Task 1's spike, NF4
4-bit): baseline 694 MiB -> loaded 3359 MiB -> inference peak 3517 MiB
(`nvidia-smi` deltas); `torch.cuda.max_memory_allocated()` only 2564 MiB over
the same run, understating the real footprint the same way Task 11 found for
YOLO11s.
"""


def _base_checkpoint(model_id: str) -> str:
    """Branch B loads the unquantised checkpoint; the `-AWQ` suffix only names
    the AWQ-prequantised repo, which this branch never touches. Stripping it
    lets `Settings.vlm_model_id`'s default
    ("Qwen/Qwen2.5-VL-3B-Instruct-AWQ") keep naming the model family without
    this class ever trying to load the AWQ repo itself.
    """
    return model_id[: -len(_AWQ_SUFFIX)] if model_id.endswith(_AWQ_SUFFIX) else model_id


def _format_detections(scene: SceneState) -> str:
    if not scene.tracks:
        return "no tracked objects"
    counts: dict[str, int] = {}
    for track in scene.tracks:
        counts[track.label] = counts.get(track.label, 0) + 1
    return ", ".join(f"{count} {label}" for label, count in sorted(counts.items()))


def _format_history(history: tuple[str, ...]) -> str:
    if not history:
        return "(none yet)"
    return "\n".join(f"- {line}" for line in history)


def _build_text_prompt(request: VisionRequest) -> str:
    """Pure string construction — no image, no model.

    Carries the harm assessment (`_HARM_ASSESSMENT`) as well as the description
    request. Read the module docstring before treating what comes back as a
    detection: this asks a vision-language model what it thinks of one keyframe,
    and there is no pose or action recognition behind it.
    """
    return (
        "You are a security monitoring assistant describing a single still frame "
        f"from a fixed security camera named '{request.camera_label}'.\n"
        f"This frame was captured because: {request.reason_detail}\n"
        f"Objects currently tracked in view: {_format_detections(request.scene)}\n"
        f"Recent motion energy (0=static, 1=high motion): {request.scene.motion_energy:.2f}\n"
        "Prior descriptions for this camera, most recent last:\n"
        f"{_format_history(request.history)}\n\n"
        "Describe what is visible, assess how concerning it is, and suggest one "
        "action for a human reviewer.\n"
        f"{_HARM_ASSESSMENT}\n"
        f"{_RESPONSE_INSTRUCTIONS}"
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_json_block(text: str) -> str | None:
    """Find the first balanced `{...}` block in `text`, or `None`.

    Task 13 review ("also worth doing"): the original approach used
    `re.compile(r"\\{.*\\}", re.DOTALL).search(...)`, which is greedy across
    the *entire* response — any incidental `{`/`}` appearing after a
    perfectly valid JSON object (e.g. the model tacking on a trailing aside
    like "note: {see above}") gets swallowed into the match, breaking
    `json.loads` and causing a spurious fall-through to the fallback path.
    Scanning for the first *balanced* pair starting at the first `{` is
    precise regardless of what follows it.

    Known limitation: this is a brace counter, not a JSON-string-aware
    scanner, so a `{` or `}` appearing inside a quoted string value (e.g.
    `{"description": "a sign reading {DANGER}"}`) would miscount and either
    truncate the match or fail to close it. Not currently exercised by any
    observed model output on this box; a real occurrence would still fall
    through safely to the fixed fallback rather than crash or leak, per
    `_parse_response`'s contract.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_concern_kind(raw: object) -> ConcernKind:
    """Unknown or malformed `kind` maps to `ConcernKind.OTHER`, never raises.

    Model output is untrusted text: `ConcernKind(raw)` raises `ValueError` on
    anything it does not recognise, so that has to be caught here rather than
    left to propagate — an unrecognised kind is not evidence of nothing, it is
    evidence of *something* the model could not name from the fixed vocabulary,
    which is exactly what `OTHER` is for.
    """
    if isinstance(raw, str):
        try:
            return ConcernKind(raw.strip().lower())
        except ValueError:
            pass
    return ConcernKind.OTHER


def _parse_confidence(raw: object) -> Confidence:
    """Unknown or malformed `confidence` maps to `Confidence.POSSIBLE` — the
    weaker tier — and never to `LIKELY`.

    This is the one direction that matters: rounding a garbled reply *up* into
    `LIKELY` would let a formatting slip manufacture the confidence a routing
    decision treats as more actionable (waking someone at 3am), while rounding
    down at worst under-states a real concern that a human still sees in the
    description text and `threat_value`. `POSSIBLE` is always the safe default.

    Trailing punctuation (`"likely."`, `"likely!"`) is stripped alongside
    whitespace before the lookup — the same class of sentence-ending noise a
    small model tacks on regardless of instruction, and indistinguishable in
    intent from `"likely"` on its own. This can only ever *reveal* a value the
    enum already recognises; it cannot invent one, so `"extremely certain."`
    still falls through to `POSSIBLE` exactly as `"extremely certain"` did —
    the round-down guarantee above is unchanged by this.
    """
    if isinstance(raw, str):
        normalized = raw.strip().strip(".!,;").strip().lower()
        try:
            return Confidence(normalized)
        except ValueError:
            pass
    return Confidence.POSSIBLE


_EVIDENCE_UNSTATED = "The model reported this concern without describing what it saw."
"""Fixed, honest stand-in for `WelfareConcern.evidence` when a *recognised*
`kind` arrives with no usable evidence text (`_parse_welfare` below).

Verified failure mode this exists for: a 3B model can reply with otherwise
well-formed JSON — `{"kind": "collapse", "confidence": "likely"}` — and simply
omit the `evidence` key, an ordinary slip. The pre-fix code dropped that item
outright because `WelfareConcern.__post_init__` rejects blank evidence and the
old code never gave it anything else to construct with; with the array then
empty, `_parse_response` returns `WelfareAssessment.none()`, which the event
codec omits from the wire entirely — and the schema's own words say an
omitted welfare object reads as "not assessed". The model said "collapse,
likely" and the structured record said nothing was found: silent, and in a
custodial welfare system, silent in exactly the direction (false reassurance)
that gets someone hurt. A fixed placeholder — never the raw payload, so it
cannot leak model text any more than `_FALLBACK_DESCRIPTION` can — keeps the
concern visible while being honest that the *evidence itself* is missing;
nothing about `kind` or `confidence` is invented, only the description of
what was seen is filled in with the truth that it was not stated. This is
still not extended to `ConcernKind.OTHER`: an item that names nothing
recognisable *and* has no evidence carries no signal at all, and manufacturing
a concern out of it would be the opposite failure — noise dressed up as a
finding, which drowns real alerts and gets the alarm muted.

Whenever this placeholder is substituted, `_parse_welfare` also sets
`WelfareConcern.evidence_stated=False` (Task 3 review, MINOR 3). This string
used to be the *only* way to tell "named but not described" from a real,
evidenced concern — which meant a caller outside this module (a notification
router, say) would have to string-match a leading-underscore adapter constant
across the ports boundary to make that distinction, or simply never make it.
`evidence_stated` moved that signal into `domain/welfare.py` where it belongs;
this placeholder text still exists for a human reading the event, but routing
logic should read the flag, not this string."""


def _parse_welfare(raw: object) -> WelfareAssessment:
    """Defensively parse the optional `welfare` array into a `WelfareAssessment`.

    Untrusted model output end to end: anything other than a list yields
    `WelfareAssessment.none()` rather than raising, and each item is read field
    by field — `kind` through `_parse_concern_kind`, `confidence` through
    `_parse_confidence`. `evidence` absent, null, non-string or blank is
    handled two different ways depending on `kind`: a *recognised* kind keeps
    the concern with `_EVIDENCE_UNSTATED` substituted in and
    `evidence_stated=False` (see that constant's docstring for why dropping it
    was the wrong direction to fail, and why the flag, not the placeholder
    string, is what a caller outside this module should read), while
    `ConcernKind.OTHER` — nothing recognisable named, and no evidence either —
    is dropped, same as before this change. Either way
    `WelfareConcern.__post_init__`'s ban on blank evidence is never hit here:
    every concern this function constructs already carries either the model's
    own text (with `evidence_stated=True`) or the fixed placeholder (with
    `evidence_stated=False`).

    Deliberately never `WelfareAssessment(**item)` or `WelfareConcern(**item)`:
    spreading the payload into the constructor would let it set fields it must
    never control, `basis` above all — this function never reads a `basis` key
    from anywhere, `WelfareAssessment`'s own default is the only source of it.
    A payload key literally named `evidence_stated` is read no differently:
    this function decides that flag itself from whether real evidence text was
    found, never by trusting a same-named key the payload might contain.
    """
    if not isinstance(raw, list):
        return WelfareAssessment.none()
    concerns: list[WelfareConcern] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = _parse_concern_kind(item.get("kind"))
        evidence = item.get("evidence")
        if isinstance(evidence, str) and evidence.strip():
            evidence_text = evidence.strip()
            evidence_stated = True
        elif kind is ConcernKind.OTHER:
            # Nothing recognisable named, and no evidence either — there is no
            # signal here to keep. Manufacturing an OTHER concern out of this
            # would be the cry-wolf direction: noise reported as a finding.
            continue
        else:
            # A recognised kind with unusable evidence is still a concern the
            # model actually raised; see `_EVIDENCE_UNSTATED`'s docstring for
            # why dropping it, not keeping it, was the finding here.
            # `evidence_stated=False` records the same fact structurally, so a
            # caller does not have to compare `evidence` against this fixed
            # string to learn it.
            evidence_text = _EVIDENCE_UNSTATED
            evidence_stated = False
        concerns.append(
            WelfareConcern(
                kind=kind,
                confidence=_parse_confidence(item.get("confidence")),
                evidence=evidence_text,
                evidence_stated=evidence_stated,
            )
        )
    return WelfareAssessment(concerns=tuple(concerns))


def _parse_response(raw_text: str) -> SceneDescription:
    """Parse the model's reply, tolerating one that ignores the JSON
    instruction — a formatting slip must never crash the pipeline (spec §9).

    Any failure — no JSON block found, invalid JSON, wrong types, a missing
    required key — falls through to a fixed, safe `SceneDescription` built
    from `_FALLBACK_DESCRIPTION`/`_FALLBACK_ACTION`, never the raw text and
    never an exception (see `_FALLBACK_DESCRIPTION`'s docstring for why the
    raw text is not used here — spec §3.3, Task 13 review finding F1). The
    raw text is logged at DEBUG so the failure stays diagnosable to an
    operator without ever reaching the published `Event`. This whole path is
    a different failure mode from the scheduler's `description_unavailable=True`
    path (S14): that one fires on timeout/OOM (infrastructure failure); this
    one fires on a malformed *success* (a formatting slip), and callers must
    not conflate the two.

    `json.loads` raises `RecursionError`, not `json.JSONDecodeError`, on JSON
    nested deep enough to blow the interpreter's recursion limit — a model
    can produce that by looping on a malformed structure the same way it can
    produce any other malformed reply, and this docstring's "never an
    exception" has to cover that case too, so both exception types fall
    through to the same fixed fallback below.

    `json.loads` also accepts the non-standard `NaN`, `Infinity` and
    `-Infinity` tokens by default — a Python extension to JSON, not something
    a spec-compliant model reply should contain, but nothing stops a garbled
    generation from emitting one. Without an explicit finiteness check,
    `threat_value: NaN` reaches `_clamp01`'s `min(1.0, value)`, and `min` with
    a `NaN` operand returns whichever argument happened to be its first
    positional operand by Python's `<` semantics — here that resolves to
    `1.0`, so a garbled float silently becomes a maximum-severity CRITICAL
    event (Task 3 review, MINOR 4). `+Infinity`/`-Infinity` clamp "correctly"
    to 1.0/0.0 under the plain `min`/`max` arithmetic, which is exactly the
    problem: a value that was never a real model opinion reads as a
    confident, in-range score. `math.isfinite` rejects all three before
    `_clamp01` ever sees them, sending the reply down the same safe fallback
    as any other malformed `threat_value`.
    """
    candidate = raw_text.strip()
    json_block = _extract_json_block(candidate)
    if json_block is not None:
        try:
            payload = json.loads(json_block)
        except (json.JSONDecodeError, RecursionError):
            payload = None
        if isinstance(payload, dict):
            description = payload.get("description")
            threat_value = payload.get("threat_value")
            suggested_action = payload.get("suggested_action")
            if (
                isinstance(description, str)
                and description.strip()
                and isinstance(threat_value, int | float)
                and not isinstance(threat_value, bool)
                and math.isfinite(threat_value)
                and isinstance(suggested_action, str)
                and suggested_action.strip()
            ):
                return SceneDescription(
                    description=description.strip(),
                    threat_value=_clamp01(float(threat_value)),
                    suggested_action=suggested_action.strip(),
                    welfare=_parse_welfare(payload.get("welfare")),
                )
    logger.warning(
        "vision model reply did not match the expected JSON schema; falling back to a "
        "fixed description (raw reply logged at DEBUG, never published — spec §3.3)"
    )
    logger.debug("raw vision model reply that failed to parse: %r", candidate)
    return SceneDescription(
        description=_FALLBACK_DESCRIPTION,
        threat_value=0.5,
        suggested_action=_FALLBACK_ACTION,
        # Explicit, not just `SceneDescription`'s own default: a malformed reply has
        # no welfare opinion to carry, and this must never look like "assessed,
        # found nothing" via some other, unaudited route.
        welfare=WelfareAssessment.none(),
    )


class Qwen25VLDescriber(VisionLanguageModel, ModelRuntime):
    def __init__(self, model_id: str, max_new_tokens: int, device: str) -> None:
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._model: object | None = None
        self._processor: object | None = None
        self._state = LifecycleState.UNLOADED
        self._health_detail = ""
        self._vram_mib = 0

    async def initialize(self) -> None:
        """Branch B: transformers + bitsandbytes 4-bit NF4 on the unquantised
        checkpoint. `autoawq` (Branch A) installed and loaded weights on this
        GPU but crashed inside its own bundled Triton GEMM kernel at
        inference time against the installed Triton 3.7.1 — a hard,
        unmaintained-library incompatibility, not a config or pin issue (see
        `docs/superpowers/plans/phase1b-spike-result.md`). This path measured
        ~3.5 GB peak VRAM, comparable to spec §7's AWQ estimate, and is
        actively maintained.
        """
        self._state = LifecycleState.DOWNLOADING
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            checkpoint = _base_checkpoint(self._model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint, quantization_config=quantization_config, device_map=self._device
            )
            model.eval()
            self._model = model
            self._processor = AutoProcessor.from_pretrained(checkpoint)
            self._state = LifecycleState.LOADED
        except Exception as exc:
            self._state = LifecycleState.UNHEALTHY
            self._health_detail = str(exc)
            raise

    async def warmup(self) -> None:
        if self._model is None:
            raise RuntimeError("Qwen25VLDescriber.warmup called before initialize()")
        dummy_frame = FrameData(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            width=64,
            height=64,
            pixels=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        dummy_scene = SceneState(
            camera_id="warmup",
            frame_index=0,
            timestamp=0.0,
            detections=(),
            tracks=(),
            motion_energy=0.0,
            scene_signature=(1.0,),
        )
        request = VisionRequest(
            keyframe=dummy_frame,
            scene=dummy_scene,
            history=(),
            camera_label="warmup",
            reason_detail="warmup",
        )
        await self.describe(request)
        import torch

        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(self._device) // (1024 * 1024)
            self._vram_mib = int(reserved) + _CUDA_CONTEXT_OVERHEAD_MIB
        self._state = LifecycleState.HEALTHY

    async def describe(self, request: VisionRequest) -> SceneDescription:
        if self._model is None or self._processor is None:
            raise RuntimeError("Qwen25VLDescriber.describe called before initialize()")
        if not isinstance(request.keyframe.pixels, np.ndarray):
            raise TypeError(
                f"FrameData.pixels must be a numpy array, got {type(request.keyframe.pixels)!r}"
            )
        import asyncio

        from PIL import Image

        pixels: npt.NDArray[np.uint8] = request.keyframe.pixels
        image = Image.fromarray(pixels.astype(np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _build_text_prompt(request)},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(  # type: ignore[attr-defined]
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)  # type: ignore[attr-defined]

        loop = asyncio.get_running_loop()
        generated_ids: Any = await loop.run_in_executor(None, self._generate_sync, inputs)
        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        raw_text = self._processor.batch_decode(  # type: ignore[attr-defined]
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return _parse_response(raw_text)

    def _generate_sync(self, inputs: Any) -> Any:
        import torch

        assert self._model is not None
        with torch.inference_mode():
            return self._model.generate(  # type: ignore[attr-defined]
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False, num_beams=1
            )

    async def predict(self, request: object) -> object:
        if not isinstance(request, VisionRequest):
            raise TypeError(
                f"Qwen25VLDescriber.predict expects VisionRequest, got {type(request)!r}"
            )
        return await self.describe(request)

    async def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._vram_mib = 0
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._release_vram_sync)
        self._state = LifecycleState.UNLOADED

    def _release_vram_sync(self) -> None:
        """`gc.collect()` before `empty_cache()` is not decorative: a
        `nn.Module` graph (the multi-billion-parameter Qwen2.5-VL model
        included) is full of reference cycles, so dropping `self._model`'s
        refcount to zero does not free it immediately under CPython's
        refcounting alone — it needs a GC cycle to be collected before the
        caching allocator has anything to return to the driver. Measured on
        this box: `empty_cache()` alone left ~2.4 GB reserved after
        shutdown; adding `gc.collect()` first dropped that to ~54 MiB. The
        600s idle-unload (ResidentSet, Task 7) depends on this being a real
        reclaim, not a nominal one — a stale ~2.4 GB reservation would starve
        `plan_residency()`'s admission control of the VRAM it thinks it just
        got back.

        Task 13 review finding F2: this runs on a worker thread via
        `shutdown()`'s `run_in_executor`, matching the pattern `describe()`
        already establishes for `_generate_sync`. A full `gc.collect()` pass
        blocks whatever thread calls it for its entire duration; running it
        directly in the coroutine body would stall every other camera's
        pipeline sharing this event loop for that long, and `ResidentSet`'s
        600s idle-unload can fire while those pipelines are still running.
        """
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def mark_unhealthy(self, detail: str) -> None:
        self._state = LifecycleState.UNHEALTHY
        self._health_detail = detail

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        """Reports the checkpoint Branch B actually loads (Task 13 review
        finding F3), not the configured `-AWQ` id — `_base_checkpoint` is the
        same stripping `initialize()` uses, so this can never drift from
        what is really resident.
        """
        import transformers

        return f"transformers=={transformers.__version__} model={_base_checkpoint(self._model_id)}"

    def capabilities(self) -> Capabilities:
        """`model_key` reports the checkpoint actually loaded (Task 13
        review finding F3): Branch B always loads the unquantised checkpoint
        via `_base_checkpoint`, never the configured `-AWQ` repo, so
        reporting the raw `self._model_id` here would let an operator
        reading `health()`/`capabilities()` reasonably — and wrongly —
        conclude AWQ is active. This is orchestrator-internal (the
        registry's health view), a different boundary from the published
        `Event` (spec §3.3), so it does not conflict with F1.
        """
        return Capabilities(
            model_key=_base_checkpoint(self._model_id),
            kind="vision",
            labels=frozenset(),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
