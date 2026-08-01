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
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from sentinel_ai.domain.entities import SceneState
from sentinel_ai.ports.frame_source import FrameData
from sentinel_ai.ports.model_runtime import (
    Capabilities,
    HealthReport,
    LifecycleState,
    ModelRuntime,
)
from sentinel_ai.ports.vision_llm import SceneDescription, VisionLanguageModel, VisionRequest

_AWQ_SUFFIX = "-AWQ"
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK_ACTION = "Review the clip manually — automated threat assessment unavailable."

_RESPONSE_INSTRUCTIONS = (
    "Respond with ONLY a JSON object of this exact shape, no other text:\n"
    '{"description": "<one or two plain sentences describing what is visible>", '
    '"threat_value": <float between 0.0 and 1.0>, '
    '"suggested_action": "<one short, concrete sentence for a human reviewer>"}'
)

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
    """Pure string construction — no image, no model."""
    return (
        "You are a security monitoring assistant describing a single still frame "
        f"from a fixed security camera named '{request.camera_label}'.\n"
        f"This frame was captured because: {request.reason_detail}\n"
        f"Objects currently tracked in view: {_format_detections(request.scene)}\n"
        f"Recent motion energy (0=static, 1=high motion): {request.scene.motion_energy:.2f}\n"
        "Prior descriptions for this camera, most recent last:\n"
        f"{_format_history(request.history)}\n\n"
        "Describe what is visible, assess how concerning it is, and suggest one "
        f"action for a human reviewer.\n{_RESPONSE_INSTRUCTIONS}"
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_response(raw_text: str) -> SceneDescription:
    """Parse the model's reply, tolerating one that ignores the JSON
    instruction — a formatting slip must never crash the pipeline (spec §9).

    Any failure — no JSON block found, invalid JSON, wrong types, a missing
    required key — falls through to a fixed, safe `SceneDescription` built
    from the raw text verbatim, never an exception. This is a different
    failure mode from the scheduler's `description_unavailable=True` path
    (S14): that one fires on timeout/OOM (infrastructure failure); this one
    fires on a malformed *success* (a formatting slip), and callers must not
    conflate the two.
    """
    candidate = raw_text.strip()
    match = _JSON_BLOCK.search(candidate)
    if match is not None:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
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
                and isinstance(suggested_action, str)
                and suggested_action.strip()
            ):
                return SceneDescription(
                    description=description.strip(),
                    threat_value=_clamp01(float(threat_value)),
                    suggested_action=suggested_action.strip(),
                )
    fallback_description = candidate[:500] if candidate else "The vision model returned no text."
    return SceneDescription(
        description=fallback_description, threat_value=0.5, suggested_action=_FALLBACK_ACTION
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
        import gc

        import torch

        # `gc.collect()` before `empty_cache()` is not decorative: a
        # `nn.Module` graph (the multi-billion-parameter Qwen2.5-VL model
        # included) is full of reference cycles, so dropping `self._model`'s
        # refcount to zero does not free it immediately under CPython's
        # refcounting alone — it needs a GC cycle to be collected before the
        # caching allocator has anything to return to the driver. Measured on
        # this box: `empty_cache()` alone left ~2.4 GB reserved after
        # shutdown; adding `gc.collect()` first dropped that to ~54 MiB. The
        # 600s idle-unload (ResidentSet, Task 7) depends on this being a real
        # reclaim, not a nominal one — a stale ~2.4 GB reservation would starve
        # `plan_residency()`'s admission control of the VRAM it thinks it just
        # got back.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._state = LifecycleState.UNLOADED

    def health(self) -> HealthReport:
        return HealthReport(state=self._state, detail=self._health_detail, vram_mib=self._vram_mib)

    def version(self) -> str:
        import transformers

        return f"transformers=={transformers.__version__} model={self._model_id}"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            model_key=self._model_id,
            kind="vision",
            labels=frozenset(),
            vram_mib=self._vram_mib,
            batch_max=1,
        )
