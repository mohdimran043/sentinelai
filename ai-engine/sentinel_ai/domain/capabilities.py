"""What a camera is asked to do, and what that costs in loaded models (spec §13).

Every camera in this engine used to do the same thing: detect, track, and escalate to
a vision-language model when the gate allowed it. That is the right default and it is
still the default, but it is the wrong *requirement*. A corridor camera that only needs
to know when someone lingers should not oblige the process to hold 2.7 GiB of VLM
weights, and a camera watching a room where face recognition would be indefensible must
be able to say so in a way the engine enforces rather than merely respects.

Two types, and the split between them is the whole point of this module:

* `Capability` is what an **operator** asks for — "describe scenes here", "tell me
  about anomalies here". It is the vocabulary of `cameras.json` and of the console.
* `ModelRole` is what that **costs** — a detector, a vision-language model. It is the
  vocabulary the composition root builds from.

`required_roles` is the function between them, and it is the reason §13's "automatically
avoid loading unnecessary models for disabled capabilities" is achievable at all: the
composition root can ask, over every configured camera and *before constructing
anything*, which roles any camera actually wants. A role nobody names is never built,
so it never downloads weights, never takes VRAM, and never appears in `/health` as a
model that is loaded and idle for the life of the process.

Why roles rather than model keys
--------------------------------
A `ModelSpec.model_key` is a configured model id — `"yolo11s.pt"`,
`"Qwen/Qwen2.5-VL-3B-Instruct-AWQ"` — and it lives in `Settings`. This module is in
`domain/`, which may not import `config` (see `tests/test_architecture.py`, and
`architecture.md` on why `config` is on the forbidden list: `Settings()` reads `.env`
from disk, and a policy that depended on process environment would stop being a pure
function). So the pure layer names the *role* a capability needs filling and the
composition root decides which checkpoint fills it. Swapping YOLO11s for YOLO11n, or
Qwen for Moondream, changes a setting and touches nothing here.

Why `PERSON_DETECTION` is not a member
--------------------------------------
§13's example list has one, and it is deliberately absent. Object detection is not a
capability sitting alongside the others; it is the substrate all of them are computed
from. The escalation triggers are pure predicates over `Detection` and `Track`, the
behaviour detectors read the same tracks, and the VLM is handed the detections as
prompt context. A camera with detection "off" is a camera with nothing on, which is
already expressible as `CameraCapabilities.none()`. Offering it as a separate toggle
would invite a configuration — anomaly detection on, person detection off — that
cannot mean anything, and the engine would have to either ignore it or fail, both of
which are worse than not offering it.

Members are added when the thing they name works
------------------------------------------------
This enum grows one member per shipped capability, never ahead of one. A member whose
implementation is not wired would be a checkbox in the console that turns nothing on,
which is exactly what §40 forbids. `test_capabilities.py` requires every member to
declare its roles, so a member added without a cost entry fails the build rather than
silently requiring nothing.

Pure, like the rest of `domain/`: standard library only, no clock, no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DEFAULT_CAPABILITIES",
    "CameraCapabilities",
    "Capability",
    "ModelRole",
    "required_roles",
]


class ModelRole(StrEnum):
    """A job some model must do, named independently of which checkpoint does it.

    The composition root binds each role to a configured model id. Nothing here knows
    what a "detector" *is*, only that something has to be one.
    """

    DETECTOR = "detector"
    VLM = "vlm"
    FACE = "face"
    """Face detection plus embedding. One model in practice (`buffalo_l`), because the
    pipeline that does both cannot usefully be split — see
    `adapters/face/insightface_pipeline.py`."""

    POSE = "pose"
    """A per-person keypoint model. A second forward pass per frame on every person,
    so it is loaded only where a capability genuinely needs it — see
    `domain/behaviour/fall.py` on why the detectors that use it must still work
    without it."""


class Capability(StrEnum):
    """What an operator turns on for one camera.

    `StrEnum` so it serialises as its own wire value with no encoder — the same reason
    `Zone` and `ConcernKind` are.
    """

    SCENE_DESCRIPTION = "scene_description"
    """Ask a vision-language model what is happening, when the gate allows it.

    Turning it off leaves the camera detecting, tracking and escalating — the
    escalation is simply never described, so the event carries the metadata-derived
    stand-in the scheduler already produces when no description exists, marked in
    `Event.metadata` as configuration rather than a model failure (see
    `orchestrator/scheduler.py`'s `DESCRIPTION_SKIPPED_METADATA_KEY`).
    """

    ANOMALY_DETECTION = "anomaly_detection"
    """Run the automatic escalation triggers (`domain/policy/triggers.py`).

    Off leaves only `user_requested` — a camera an operator can ask about on demand
    but that raises nothing by itself. Useful for a camera being watched by a human
    anyway, where automatic escalations would be noise rather than information.
    """

    FALL_DETECTION = "fall_detection"
    """Watch for the temporal signature of a person falling and not getting up.

    Independent of `anomaly_detection`, deliberately: the six automatic triggers ask
    "is this scene worth a look", while this asks one specific question about one
    person over several seconds. A camera can want either without the other — a
    dayroom that should raise a fall but not a track-count spike is an ordinary
    configuration, not a contradiction.

    Requires `ModelRole.POSE`, which is what makes the reading robust enough to be
    worth acting on. The state machine degrades to bounding-box geometry per frame
    when a skeleton cannot answer, and records which it used, but a camera that asked
    for fall detection gets the pose model loaded rather than the fallback by default.
    """

    ABANDONED_OBJECT = "abandoned_object"
    """Watch for a bag, case or backpack a person was with being left behind.

    Reads the detector's existing output — the three COCO luggage classes are already
    in `DEFAULT_SALIENT_CLASSES` — so it needs no model of its own beyond the one every
    capability here already implies.
    """

    CAMERA_TAMPER = "camera_tamper"
    """Watch for this camera going blind — lens covered, view obstructed, scene unlit.

    **The only capability that requires no model at all.** It reads the luma histogram
    `MotionAnalyzer` already computes for the escalation gate, so the marginal cost is
    a comparison per frame and it can be switched on across an entire site for nothing.

    That matters more than it sounds. Every other detector goes *silent* when a camera
    is obstructed, and silence is exactly what a healthy camera watching an empty
    corridor looks like — so without this, the failure mode is a wall of green tiles
    and nobody watching anything.
    """

    PERSON_AUTHORIZATION = "person_authorization"
    """Check faces against the enrolled roster and raise when somebody is not on it.

    **Off by default and enabled per camera, which §8 states twice and means.** Face
    recognition is the most invasive thing this system can do and the most consequential
    when it is wrong: a false result is an accusation about a specific identifiable
    person. A camera watching a space where it would be indefensible must be able to
    say so, and the way it says so is by not enabling this.

    Requires `ModelRole.FACE`, and a `SENTINEL_FACE_ENCRYPTION_KEY` — the composition
    root refuses to start a deployment that enables this without one, rather than
    silently writing biometric data in the clear.
    """

    ZONE_MONITORING = "zone_monitoring"
    """Watch for people entering restricted areas or crossing virtual boundaries.

    One capability rather than two, because both read the same per-camera geometry and
    the geometry is already the switch: a camera with zones drawn and no lines raises
    no crossings. Splitting them would let an operator enable a capability whose
    configuration says it should do nothing, and then wonder which of the two settings
    was wrong.

    A camera that enables this and draws no geometry is a checkbox that does nothing,
    which `ZonePolicy.is_empty` exists to let the composition root refuse rather than
    silently accept.
    """


_ROLES: dict[Capability, frozenset[ModelRole]] = {
    Capability.SCENE_DESCRIPTION: frozenset({ModelRole.DETECTOR, ModelRole.VLM}),
    # The VLM is handed the frame's detections as prompt context (`VisionRequest.scene`),
    # so describing a scene needs the detector too — not only the model that speaks.
    Capability.ANOMALY_DETECTION: frozenset({ModelRole.DETECTOR}),
    Capability.FALL_DETECTION: frozenset({ModelRole.DETECTOR, ModelRole.POSE}),
    Capability.ABANDONED_OBJECT: frozenset({ModelRole.DETECTOR}),
    # Genuinely nothing. Tamper detection reads the luma histogram the motion stage
    # already computes, so a site can switch it on everywhere for free — see the
    # member's docstring for why that is worth more than it sounds.
    Capability.CAMERA_TAMPER: frozenset(),
    Capability.ZONE_MONITORING: frozenset({ModelRole.DETECTOR}),
    # The object detector as well as the face model: authorisation reasons about a
    # *tracked person* over several frames (§11), and the track comes from the detector.
    Capability.PERSON_AUTHORIZATION: frozenset({ModelRole.DETECTOR, ModelRole.FACE}),
}
"""`Capability` -> the roles it obliges the process to load.

A dict rather than a method on the enum so `test_capabilities.py` can compare its keys
against `Capability` directly and fail the build on a member that forgot to declare its
cost. A missing entry raises `KeyError` here rather than defaulting to "needs nothing",
because defaulting would produce a camera running a capability with no model behind it.
"""


@dataclass(frozen=True, slots=True)
class CameraCapabilities:
    """The set of capabilities enabled on one camera.

    A wrapper around `frozenset[Capability]` rather than the bare frozenset, for the
    reason `Zone` is an enum rather than a string: this value is parsed from an
    untrusted file and rendered back into one, and the parsing and the rendering
    belong with the type instead of being open-coded at both ends. `frozen` and
    `slots` make it hashable and cheap, so it can key a cache or ride on a frozen
    telemetry record without a defensive copy.
    """

    enabled_set: frozenset[Capability]

    @classmethod
    def of(cls, *capabilities: Capability) -> CameraCapabilities:
        """From members. Order-insensitive and deduplicating, being a set."""
        return cls(enabled_set=frozenset(capabilities))

    @classmethod
    def none(cls) -> CameraCapabilities:
        """Nothing enabled — a watched camera with no AI running on it.

        A real, storable instruction rather than an unconfigured state: the engine
        still decodes it, still reports its liveness, and still lets an operator force
        a description by hand. It simply raises nothing on its own.
        """
        return cls(enabled_set=frozenset())

    @classmethod
    def from_names(cls, names: Iterable[str]) -> CameraCapabilities:
        """From the wire/file vocabulary, rejecting anything unrecognised.

        Present-but-unknown is an error, not a value to drop, and for a sharper
        version of `domain/zone.py`'s reason: a mistyped zone mis-groups a camera in a
        console, while a mistyped capability leaves an operator believing monitoring
        is running that was never switched on. Fail-loud at startup is the only
        option that cannot produce a quietly under-watched site.

        An empty iterable is *not* an error: `[]` is the explicit "run nothing here"
        instruction, and it has to stay distinguishable from an absent field (which
        means `DEFAULT_CAPABILITIES`). The caller owns that distinction — this method
        is only reached once the field is known to be present — exactly as
        `notify_on`'s absent-versus-empty handling works.
        """
        members: set[Capability] = set()
        valid = sorted(member.value for member in Capability)
        for name in names:
            try:
                members.add(Capability(name))
            except ValueError as exc:
                raise ValueError(
                    f"unknown capability {name!r}; valid capabilities are {valid}"
                ) from exc
        return cls(enabled_set=frozenset(members))

    def enabled(self, capability: Capability) -> bool:
        return capability in self.enabled_set

    def names(self) -> list[str]:
        """The wire form, sorted.

        Sorted because `CameraFileStore` writes this document back on every edit, and
        an iteration-order-dependent list would rewrite an unrelated camera's
        capabilities line on every unrelated save — churning the diff of a file whose
        reviewability is the reason it is a file at all (ADR 9).
        """
        return sorted(capability.value for capability in self.enabled_set)


DEFAULT_CAPABILITIES: CameraCapabilities = CameraCapabilities.of(
    Capability.SCENE_DESCRIPTION,
    Capability.ANOMALY_DETECTION,
)
"""What a camera that does not mention `capabilities` gets.

Exactly what every camera does today, so upgrading an existing deployment changes no
behaviour: the field's absence has to keep meaning what the absence of the field meant
before it existed. Anything narrower would silently stop monitoring a site at the
moment its engine was upgraded, which is the one upgrade outcome a surveillance system
must never produce.
"""


def required_roles(*cameras: CameraCapabilities) -> frozenset[ModelRole]:
    """The union of every role any of these cameras needs.

    The **union**, never the intersection: the engine holds one instance of each model
    for every camera (ADR 4 — one shared detector behind a lock), so the question is
    "does anybody need this", and one camera asking is enough to make the process pay.
    An intersection would leave a camera configured for a capability the process
    declined to load a model for, which is a silently unserved camera.

    Variadic with no cameras answering `frozenset()`: an engine configured with zero
    cameras loads zero models, and that falls out of the fold rather than needing a
    special case.
    """
    roles: set[ModelRole] = set()
    for camera in cameras:
        for capability in camera.enabled_set:
            roles |= _ROLES[capability]
    return frozenset(roles)
