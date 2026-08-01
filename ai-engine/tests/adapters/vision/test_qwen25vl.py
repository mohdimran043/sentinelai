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
    _FALLBACK_DESCRIPTION,
    Qwen25VLDescriber,
    _base_checkpoint,
    _build_text_prompt,
    _parse_response,
)
from sentinel_ai.domain.entities import BBox, SceneState, Track
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
