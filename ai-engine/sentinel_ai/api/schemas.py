"""Pydantic wire models. Domain dataclasses never cross the API boundary
directly — these mirror the relevant fields so the wire shape and the
orchestrator's internal shape can change independently."""

from __future__ import annotations

from typing import Annotated, Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sentinel_ai.adapters.config.camera_file import (
    EDITABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
    UNSET,
    CameraConfig,
    CameraEdit,
)
from sentinel_ai.domain.alert import Alert, AlertState
from sentinel_ai.domain.capabilities import CameraCapabilities, Capability
from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.domain.identity import PersonStatus
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.welfare import ConcernKind, Confidence, WelfareConcern
from sentinel_ai.domain.zone import Zone, ZoneKind
from sentinel_ai.orchestrator.event_history import CameraEventHistory, RecentEvent
from sentinel_ai.pipeline.runner import CameraTelemetry

CameraLabel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
"""A camera's display name. Trimmed, non-empty after trimming, and bounded.

`load_cameras` already rejects an empty label, but it cannot see the difference
between `""` and `"   "` — both are falsy to an operator and only one is falsy to
Python. Trimming first makes them the same answer. The 120-character bound is
about the console: a label is a nav item, and there is no length at which a
longer one is more useful than a truncated one.
"""

NonNegativeSeconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
"""A duration that may be zero — currently only `clip_preroll_seconds`, where no
lead-in at all is a real choice."""

PositiveSeconds = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
"""A duration that must be greater than zero: a clip that ends where it starts,
or a summary that runs every no-seconds, is never what anyone meant.

Both aliases mirror the bounds `load_cameras` enforces, and they exist so the
rejection happens *here*. `CameraFileStore.apply` re-parses the whole document
before writing and would refuse these too, but as a `CameraConfigError` — which
this API answers 409, i.e. "the file will not take your edit". For a value that
was simply out of range that is a lie, and it sends an operator to inspect a file
that is perfectly fine.

`allow_inf_nan=False` completes the mirror, and is the part most likely to be
dropped as noise. `1e999` and `NaN` are valid JSON that `json.loads` decodes
happily — Python's `json.dumps` emits them by default, so a client does not have
to be hand-rolled to send one — and `inf` clears every `gt`/`ge` check above,
which makes it the one out-of-range value that reaches the store looking in
range. Rejecting it here is what keeps "out of range is a 422" true without
exception.

The reason this needs saying: the 422 body FastAPI builds echoes the offending
value back, and Starlette renders it with `allow_nan=False`, so *any* 422
carrying a non-finite input fails to serialise and becomes a 500. That is not a
reason to stop rejecting non-finite values — `-inf` and `NaN` already fail
`ge`/`gt` and hit the same crash whatever this line says. It is handled where it
happens instead, by `api/app.py`'s `RequestValidationError` handler, which nulls
non-finite floats out of the error body before rendering it.
"""


class ModelHealth(BaseModel):
    key: str
    state: str
    detail: str
    vram_mib: int


class HealthResponse(BaseModel):
    status: str
    models: list[ModelHealth]


class CameraStatus(BaseModel):
    camera_id: str
    label: str = Field(
        description=(
            "The camera's display name, from `cameras.json`; defaults to `camera_id` "
            "when the file gives none. Editable at runtime via "
            "`PATCH /cameras/{camera_id}` — read it back from here after a write to "
            "see what the engine is actually using."
        )
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Whether the engine is watching this camera at all. `false` is a camera an "
            "operator deliberately stopped: the record is still in `cameras.json`, it "
            "still has a label, a zone and a capability set, and no source connection "
            "is open. Distinct from a camera with no capabilities, which still pulls "
            "and decodes every frame. **The counters on a disabled camera are frozen "
            "at the moment it was stopped, not zeroed** — except after a restart, when "
            "they are genuinely zero because nothing has been watched this run."
        ),
    )
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    falls_suspected: int = Field(
        default=0,
        description=(
            "How many fall signatures have completed on this camera since the engine "
            "started. Always 0 on a camera without the `fall_detection` capability, "
            "and 0 is not the same claim as absent — it means the machine ran and saw "
            "nothing. Cumulative and monotonic within one process run; it resets on "
            "restart, like every other counter here. **Not a count of falls**: it is a "
            "count of times a geometry state machine's signature completed and a "
            "vision-language model was asked to confirm. See the camera's events for "
            "what the model then said."
        ),
    )
    last_frame_at: float | None = Field(
        description=(
            "When the last frame arrived, on **this camera's own source timeline** — "
            "`time.monotonic()` for a live RTSP camera, seconds-from-start-of-file for "
            "a replayed one. It is what correlates telemetry with a clip's pts, and it "
            "is **not** a wall-clock time: it resets on restart and two cameras do not "
            "share an origin. Never render it as an age — use `last_frame_epoch`."
        )
    )
    last_frame_epoch: float | None = Field(
        default=None,
        description=(
            "When this camera was last **observed** to have delivered a frame, in Unix "
            "epoch seconds. This is the field a liveness indicator reads.\n\n"
            "Separate from `last_frame_at` because that one cannot answer the question: "
            "reading a source timeline as an epoch put every camera at '20712d ago' and "
            "reported '0 of 3 delivering' while all three were. Observational — it is "
            "stamped when a read notices the source timeline has advanced — so it is as "
            "fresh as the last time somebody asked, which for a polling console is "
            "every few seconds. Null when this camera has not been seen to deliver "
            "anything yet."
        ),
    )
    last_escalation_at: float | None
    zone: Zone | None = Field(
        default=None,
        description=(
            "Which space this camera watches — the field cameras are grouped by. Null "
            "means ungrouped: nobody has assigned this camera a zone. Null is not a "
            "group; do not render it as one alongside the real zones."
        ),
    )
    zone_kind: ZoneKind | None = Field(
        default=None,
        description=(
            "The coarse grouping `zone` falls into, derived from it and never stored "
            "separately, so the two cannot disagree. Null exactly when `zone` is null."
        ),
    )
    capabilities: list[Capability] = Field(
        description=(
            "Which AI capabilities are running on this camera, as the engine has them "
            "— always the full list, never a diff, and sorted so two reads of the same "
            "record compare equal. `[]` is a stored choice, not an unset field: the "
            "camera is watched and decoded, and nothing is run on it. **Render this as "
            "what the engine is doing, not as what the file asked for** — it is read "
            "back off the running camera, which is what makes a capability checkbox "
            "honest rather than decorative. A capability absent here is one no model "
            "was loaded for."
        )
    )
    notify_on: list[ConcernKind] = Field(
        description=(
            "The concern kinds this camera will notify a human about, as stored — "
            "always the full list, never a diff, and sorted so two reads of the same "
            "record compare equal. A camera whose file says nothing about `notify_on` "
            "lists every kind, because that is what saying nothing means. `[]` means "
            "this camera notifies nobody, which is a stored choice rather than an "
            "unset field: **render it as muted, not as unconfigured.** Identical in "
            "meaning to `CameraEditResponse.notify_on`, so a console can compare what "
            "it wrote against what it later reads here."
        )
    )
    notify_min_confidence: Confidence = Field(
        description="The stored confidence threshold. Never null: a camera always has one."
    )
    clip_preroll_seconds: float | None = Field(
        description=(
            "The stored per-camera override, or null when this camera follows the "
            "engine-wide default. **Null is the answer to 'what is stored', not a "
            "report of the effective value** — the same meaning `CameraEditResponse` "
            "gives it. A console that resolved it locally and then submitted what it "
            "showed would pin the camera to a number nobody chose and opt it out of "
            "any later change to the default."
        )
    )
    clip_postroll_seconds: float | None = Field(
        description="As `clip_preroll_seconds`: the stored override, or null for the default."
    )
    summary_interval_seconds: float | None = Field(
        description=(
            "The stored override, or null when this camera falls back to its "
            "profile's `summary_interval_seconds`."
        )
    )

    @classmethod
    def from_telemetry(
        cls, telemetry: CameraTelemetry, *, last_frame_epoch: float | None = None
    ) -> CameraStatus:
        return cls(
            last_frame_epoch=last_frame_epoch,
            label=telemetry.label,
            enabled=telemetry.enabled,
            capabilities=[Capability(name) for name in telemetry.capabilities.names()],
            falls_suspected=telemetry.falls_suspected,
            zone=telemetry.zone,
            # Derived here rather than carried, so no configuration can make the fine
            # and coarse groupings contradict each other on the wire.
            zone_kind=None if telemetry.zone is None else telemetry.zone.kind,
            camera_id=telemetry.camera_id,
            frames_seen=telemetry.frames_seen,
            frames_dropped=telemetry.frames_dropped,
            detections_run=telemetry.detections_run,
            escalations=telemetry.escalations,
            escalations_dropped=telemetry.escalations_dropped,
            discontinuities=telemetry.discontinuities,
            last_frame_at=telemetry.last_frame_at,
            last_escalation_at=telemetry.last_escalation_at,
            # Sorted for the reason `CameraEditResponse.from_config` sorts it: the
            # source is a frozenset, whose iteration order varies with the process's
            # hash seed, and a console diffing two reads should not see a change that
            # is not one.
            notify_on=sorted(telemetry.notify_on),
            notify_min_confidence=telemetry.notify_min_confidence,
            clip_preroll_seconds=telemetry.clip_preroll_seconds,
            clip_postroll_seconds=telemetry.clip_postroll_seconds,
            summary_interval_seconds=telemetry.summary_interval_seconds,
        )


class CamerasResponse(BaseModel):
    cameras: list[CameraStatus]
    config_writable: bool = Field(
        description=(
            "Whether `PATCH /cameras/{camera_id}` will do anything on this "
            "deployment. False — the default — means the write endpoint answers 403 "
            "and the camera record can only be changed by editing `cameras.json` and "
            "restarting. This engine has **no authentication**, so writes are opt-in "
            "per deployment (`SENTINEL_ENABLE_CAMERA_WRITES`); see the PATCH "
            "operation's description. Published here so a console can render the "
            "record read-only rather than offering controls that will 403."
        )
    )


class DescribeResponse(BaseModel):
    event_id: UUID


_NULL_IS_NOT_AN_INSTRUCTION: Final = {
    "label": (
        "'label' must be a non-empty string; omit the field to leave the label "
        "unchanged. A camera always has a label — an unset one falls back to its id "
        "at load, which is not the same as no label."
    ),
    "capabilities": (
        "'capabilities' must be an array of capability names; send [] to run nothing "
        "on this camera, or omit the field to leave it unchanged. Null would have to "
        "mean one of those two, and guessing which is how a camera someone switched "
        "off starts running models again."
    ),
    "notify_on": (
        "'notify_on' must be an array of concern kinds; send [] to stop this camera "
        "notifying anyone, or omit the field to leave its routing unchanged. Null "
        "would have to mean one of those two, and guessing which is how a camera "
        "someone muted starts alerting again."
    ),
    "notify_min_confidence": (
        "'notify_min_confidence' must be a confidence tier; omit the field to leave "
        "the threshold unchanged. There is no 'no threshold' state — every concern "
        "arrives with a confidence, so something always has to be compared against."
    ),
}
"""The editable fields where `None` is pydantic's "not mentioned" default and
never an instruction, mapped to what to send instead.

Each of these has to stay `X | None` on the model, because pydantic cannot
represent "omitted" in the type at all — `model_fields_set` is the only thing
that knows. So a mentioned-but-null is caught in the validator instead, and the
message says what the caller should have sent: a 422 reading only "expected str"
tells an operator their console is broken rather than which of two real
instructions they meant. `zone` and the three duration overrides are deliberately
absent from this map — for them null *is* an instruction.
"""


class CameraEditRequest(BaseModel):
    """A partial edit to one camera's record: its label, its zone, and its welfare
    notification policy, in any combination.

    Partial on purpose — a console changing a label must not have to restate a zone
    it is not touching, because restating it is how one operator's window silently
    reverts another's change.

    **Only the runtime-editable fields are accepted, and the rest are rejected
    rather than ignored.** `extra="forbid"` means a body carrying `url` or
    `profile` is a 422 naming the field, not a 200 that quietly dropped it. That
    is the whole reason the model is strict: the failure this endpoint must never
    have is an operator re-pointing a camera at a new stream, being told it
    worked, and watching the old stream for a week.
    """

    model_config = ConfigDict(extra="forbid")

    label: CameraLabel | None = Field(
        default=None,
        description=(
            "The camera's new display name. Trimmed; must be non-empty after "
            "trimming. Omit the field to leave the label alone — `null` is not a "
            "label and is rejected."
        ),
    )
    zone: Zone | None = Field(
        default=None,
        description=(
            "The camera's new zone, or `null` to ungroup it. Unlike `label`, `null` "
            "here is a real instruction, so **omitting the field and sending null "
            "mean different things**: omit to leave the grouping alone, send null to "
            "remove it. A value outside the enum is a 422 — the same fail-loud "
            "`load_cameras` applies at startup, because a typo'd zone is a camera "
            "the operator meant to group and silently did not."
        ),
    )
    capabilities: list[Capability] | None = Field(
        default=None,
        description=(
            "Which AI capabilities run on this camera, replacing whatever is "
            "configured now — a whole new list, not an addition to the old one. "
            "**`[]` means run nothing on this camera**, which is a real instruction "
            "and not an empty edit: the camera stays watched and decoded, and no "
            "model is run against it. Omit the field to leave the capabilities alone; "
            "`null` is rejected, because `[]` already says the only thing it could "
            "mean.\n\n"
            "**A capability can only be enabled if this process already loaded the "
            "model it needs.** Which models exist is decided once, at startup, from "
            "the union over every configured camera (see `GET /cameras`), because "
            "loading a 3B vision model is a multi-second download-and-place that "
            "cannot happen under an HTTP request without stalling every camera "
            "sharing the GPU. Enabling one whose model is absent is a 409 naming the "
            "restart, never a 200 that quietly did nothing — the same fail-loud this "
            "endpoint applies to `url`. Disabling is always allowed."
        ),
    )
    notify_on: list[ConcernKind] | None = Field(
        default=None,
        description=(
            "Which welfare concern kinds this camera notifies a human about, "
            "replacing whatever is configured now — this is a whole new list, not an "
            "addition to the old one. **`[]` means never notify from this camera**, "
            "which is a real instruction and not an empty edit: it silences one "
            "camera without turning welfare monitoring off anywhere else. Omit the "
            "field to leave the routing alone; `null` is rejected, because `[]` "
            "already covers the only thing it could have meant. A kind outside the "
            "enum is a 422 rather than a silent narrowing of the list — a typo'd "
            "kind is a concern the operator meant to be told about and would not be."
        ),
    )
    notify_min_confidence: Confidence | None = Field(
        default=None,
        description=(
            "The lowest confidence tier that may notify. `likely` is the engine's "
            "default; `possible` widens it to everything the model flags at all, "
            "which on an ordinary day is most of its opinions. Omit to leave the "
            "threshold alone; `null` is rejected, because a camera has no "
            "'no threshold' state. There are exactly two tiers and there is no "
            "`certain` — see `domain/welfare.py`: one still frame cannot earn it."
        ),
    )
    clip_preroll_seconds: NonNegativeSeconds | None = Field(
        default=None,
        description=(
            "Seconds of buffered video to keep before a notified concern's keyframe, "
            "for this camera only. `null` reverts it to the engine-wide default; "
            "omitting the field leaves it as configured — **the two are different "
            "instructions**, the same way `zone`'s are. `0` is allowed and means no "
            "lead-in at all, which is why this bound is `>= 0` where the other two "
            "durations are `> 0`."
        ),
    )
    clip_postroll_seconds: PositiveSeconds | None = Field(
        default=None,
        description=(
            "Seconds of video to keep recording after a notified concern's keyframe, "
            "for this camera only. `null` reverts it to the engine-wide default; "
            "omitting the field leaves it as configured. Must be greater than zero — "
            "a clip that ends where it begins is not a shorter clip, it is no clip."
        ),
    )
    summary_interval_seconds: PositiveSeconds | None = Field(
        default=None,
        description=(
            "How often this camera's periodic summary runs. `null` reverts it to the "
            "camera profile's interval; omitting the field leaves it as configured. "
            "Distinct from `profile.summary_interval_seconds`, which is restart-only "
            "— this is the runtime-editable override of it, and it wins where both "
            "are set."
        ),
    )

    @model_validator(mode="after")
    def _reject_an_edit_that_asks_for_nothing(self) -> CameraEditRequest:
        """Two ways of naming no change, both rejected here rather than downstream.

        `{}` most often means a console lost track of which fields it was
        submitting; `{"label": null}` means it sent a cleared input. Either one
        answered 200 would report a successful save for a request that changed
        nothing, and the operator would believe their edit landed.

        The nulls rejected here are the ones listed in
        `_NULL_IS_NOT_AN_INSTRUCTION`, and they are caught by a validator rather
        than by the type because the type has to stay nullable: `None` is also each
        field's "not mentioned" default, and pydantic cannot distinguish an omitted
        optional from an explicit null without `model_fields_set`. `zone` and the
        three duration overrides are the opposite case and deliberately so — a null
        there is a real instruction to ungroup, or to fall back to the default.
        """
        for field_name, what_to_send_instead in _NULL_IS_NOT_AN_INSTRUCTION.items():
            if field_name in self.model_fields_set and getattr(self, field_name) is None:
                raise ValueError(what_to_send_instead)
        if not self.model_fields_set:
            raise ValueError(
                f"name at least one field to change; editable fields are {list(EDITABLE_FIELDS)}"
            )
        return self

    def to_edit(self) -> CameraEdit:
        """Convert to the store's shape, preserving *which fields were mentioned*.

        `model_fields_set` is what carries that: pydantic cannot otherwise tell a
        `zone` the caller explicitly set to null from a `zone` the caller never
        mentioned, and those are the two instructions this endpoint most needs to
        keep apart. The same applies to each duration override below.

        The two shapes are visible in the lines: a field whose null the validator
        already rejected is guarded with `is not None` (which is therefore a
        narrowing for the type checker, not a decision), and a field whose null is
        an instruction is passed straight through.
        """
        mentioned = self.model_fields_set
        return CameraEdit(
            label=self.label if "label" in mentioned and self.label is not None else UNSET,
            zone=self.zone if "zone" in mentioned else UNSET,
            capabilities=(
                CameraCapabilities.of(*self.capabilities)
                if "capabilities" in mentioned and self.capabilities is not None
                else UNSET
            ),
            notify_on=(
                frozenset(self.notify_on)
                if "notify_on" in mentioned and self.notify_on is not None
                else UNSET
            ),
            notify_min_confidence=(
                self.notify_min_confidence
                if "notify_min_confidence" in mentioned and self.notify_min_confidence is not None
                else UNSET
            ),
            clip_preroll_seconds=(
                self.clip_preroll_seconds if "clip_preroll_seconds" in mentioned else UNSET
            ),
            clip_postroll_seconds=(
                self.clip_postroll_seconds if "clip_postroll_seconds" in mentioned else UNSET
            ),
            summary_interval_seconds=(
                self.summary_interval_seconds if "summary_interval_seconds" in mentioned else UNSET
            ),
        )


class CameraEditResponse(BaseModel):
    """The camera's record as `cameras.json` now holds it, after a successful edit.

    The whole record rather than an acknowledgement: a console that has just
    written should render what was stored, not what it hoped was stored.
    """

    camera_id: str
    label: str
    zone: Zone | None = Field(description="The stored zone, or null when the camera is ungrouped.")
    zone_kind: ZoneKind | None = Field(
        description="Derived from `zone`, exactly as on `CameraStatus`. Null when `zone` is."
    )
    capabilities: list[Capability] = Field(
        description=(
            "The capabilities now stored for this camera, sorted — always the full "
            "list, never a diff. `[]` means nothing runs on this camera. Identical in "
            "meaning to `CameraStatus.capabilities`, so a console can compare what it "
            "wrote against what it later reads there."
        )
    )
    notify_on: list[ConcernKind] = Field(
        description=(
            "The concern kinds this camera will notify on, as stored — always the "
            "full list, never a diff, and sorted so two reads of the same record "
            "compare equal. A camera whose file says nothing about `notify_on` lists "
            "every kind here, because that is what saying nothing means. `[]` means "
            "this camera notifies nobody."
        )
    )
    notify_min_confidence: Confidence = Field(
        description="The stored confidence threshold. Never null: a camera always has one."
    )
    clip_preroll_seconds: float | None = Field(
        description=(
            "The stored per-camera override, or null when this camera uses the "
            "engine-wide default. Null here is the answer to 'what is stored', not "
            "a report of the effective value — the default in force is not this "
            "endpoint's to state."
        )
    )
    clip_postroll_seconds: float | None = Field(
        description="As `clip_preroll_seconds`: the stored override, or null for the default."
    )
    summary_interval_seconds: float | None = Field(
        description=(
            "The stored override, or null when this camera falls back to its "
            "profile's `summary_interval_seconds`."
        )
    )
    persisted: Literal[True] = Field(
        default=True,
        description=(
            "Always true, and present so it cannot be overlooked: this edit was "
            "written to the engine's `cameras.json` before this response was sent, "
            "and survives a restart. An edit that could not be written is an error "
            "response, never a 200 with this set to false."
        ),
    )
    restart_required_fields: list[str] = Field(
        default_factory=lambda: list(RESTART_REQUIRED_FIELDS),
        description=(
            "Fields of the camera record that this endpoint will not change at all: "
            "they are stored in `cameras.json`, honoured at startup, and require "
            "editing that file and restarting the engine. `url` because changing it "
            "means tearing down the running camera and building a new source (and "
            "because an RTSP URL routinely carries credentials, which an "
            "unauthenticated API must not move in either direction); `profile` "
            "because it is the escalation policy the gate is part-way through "
            "applying. Sending either one is a 422, not a silent drop — a console "
            "should show them as read-only and say why, not offer a control that "
            "does nothing."
        ),
    )

    @classmethod
    def from_config(cls, config: CameraConfig) -> CameraEditResponse:
        return cls(
            camera_id=config.camera_id,
            label=config.label,
            zone=config.zone,
            zone_kind=None if config.zone is None else config.zone.kind,
            # Sorted for the same reason the file's copy is: `notify_on` is a
            # frozenset, whose iteration order varies with the process's hash seed,
            # and a console diffing two reads should not see a change that is not one.
            capabilities=[Capability(name) for name in config.capabilities.names()],
            notify_on=sorted(config.notify_on),
            notify_min_confidence=config.notify_min_confidence,
            clip_preroll_seconds=config.clip_preroll_seconds,
            clip_postroll_seconds=config.clip_postroll_seconds,
            summary_interval_seconds=config.summary_interval_seconds,
        )


class EnrolledFaceEntry(BaseModel):
    """One reference face on a person's record.

    Metadata only — the picture is fetched one at a time from its own endpoint. A list
    that inlined the images would move every enrolled person's biometric data across the
    network to draw a table of names, which is the opposite of §12's "do not expose
    biometric information unnecessarily".
    """

    face_id: UUID
    enrolled_at: float | None = Field(
        default=None,
        description=(
            "Unix epoch seconds, or null for a face enrolled before this was recorded. "
            "Null means unknown, never 1970."
        ),
    )
    has_image: bool = Field(
        description=(
            "Whether a reference photograph was kept. False is a real answer, not a "
            "loading state: enrolment can keep the embedding and no picture."
        )
    )


class FacesResponse(BaseModel):
    faces: list[EnrolledFaceEntry]


class CameraCreateRequest(BaseModel):
    """A new camera, as the console describes one.

    `url` is here and is absent from `CameraEditRequest`, and that asymmetry is the
    design: changing a running camera's source means tearing down its runner, its
    pre-roll and any clip mid-recording, while adding one destroys nothing.

    `profile` is absent from both. The escalation policy has live state — cooldowns, a
    part-filled token bucket — so a new camera starts on the defaults and tuning it is
    still `cameras.json` and a restart.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    url: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)] = (
        Field(
            description=(
                "An rtsp(s) URL, an earthcam.com **page** URL, or a local file path. "
                "Which source gets built is decided from this and nothing else."
            )
        )
    )
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] = ""
    zone: Zone | None = Field(
        default=None,
        description="The group this camera belongs to. Null leaves it ungrouped.",
    )
    capabilities: list[Capability] | None = Field(
        default=None,
        description=(
            "Null takes the engine's defaults. A capability whose model this process "
            "never loaded is refused with a 409 naming the restart — which models exist "
            "is decided at startup."
        ),
    )


class ProbeRequest(BaseModel):
    """Look at a stream before committing to it."""

    model_config = ConfigDict(extra="forbid")

    url: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]


class ProbeResponse(BaseModel):
    """What one look at a stream found.

    `ok` false is a normal answer, not an error status: the caller is rendering a
    preview either way, and a 4xx would make "this URL is wrong" indistinguishable from
    "the request was malformed".
    """

    ok: bool
    detail: str
    source_kind: str = Field(
        description="`earthcam`, `rtsp` or `file` — which source the engine would build."
    )
    title: str = ""
    width: int = 0
    height: int = 0
    codec: str = ""
    fps: float = 0.0
    thumbnail: str | None = Field(
        default=None,
        description=(
            "One decoded frame as a `data:image/jpeg;base64,…` URL, or null. The point "
            "of a preview: a stream that opens and decodes green looks identical to a "
            "working one in every other field here."
        ),
    )


class PersonEntry(BaseModel):
    """One enrolled person, as a console lists them (spec §23).

    **Carries no biometric data.** No embedding, no vector, no image — those live in
    the encrypted store and never travel on this API. What a console shows is a name, a
    status and where somebody is authorised, and that is deliberately all it can show:
    §12 asks that biometric information is not exposed unnecessarily, and the necessary
    amount here is none.

    `reference_faces` is a count rather than the faces themselves, for the same reason.
    The faces have their own endpoint, and their photographs one each — see
    `EnrolledFaceEntry`.
    It is worth showing because §10 asks for multiple references per person and "1" is
    usually the reason somebody is not being recognised.
    """

    person_id: UUID
    display_name: str
    status: PersonStatus
    external_reference: str | None = None
    camera_ids: list[str] = Field(
        description=(
            "Cameras this person is authorised on. **Empty means none, not all** — a "
            "person enrolled with no cameras assigned is authorised nowhere until "
            "somebody says where. The opposite default would make forgetting to set "
            "this a silent grant."
        )
    )
    zones: list[str] = Field(description="Zone names authorised, as an alternative to cameras.")
    expires_at: float | None = Field(
        default=None,
        description=(
            "Unix epoch seconds after which this authorisation lapses, or null for no "
            "expiry. §10's temporary authorisation — without it every temporary grant "
            "becomes a permanent one somebody forgot to revoke."
        ),
    )
    reference_faces: int = Field(
        description=(
            "How many reference faces are enrolled. A count, never the faces. One is "
            "usually the reason somebody is not being recognised from an angle."
        )
    )
    notes: str = ""


class PeopleResponse(BaseModel):
    people: list[PersonEntry]


class PersonRequest(BaseModel):
    """Create or replace an authorised person. Carries no biometric data either —
    faces are enrolled separately, against an existing person."""

    model_config = ConfigDict(extra="forbid")

    display_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]
    status: PersonStatus = PersonStatus.ACTIVE
    external_reference: str | None = Field(
        default=None,
        max_length=120,
        description=(
            "A site's own identifier — staff number, badge id. Opaque here and never "
            "used for matching; it exists so an operator can reconcile this record "
            "with whatever system actually governs employment."
        ),
    )
    camera_ids: list[str] = Field(
        default_factory=list,
        description="Cameras to authorise on. Empty authorises nowhere.",
    )
    zones: list[str] = Field(default_factory=list)
    expires_at: float | None = None
    notes: str = Field(default="", max_length=2000)


class AlertEntry(BaseModel):
    """One ongoing situation an operator is asked to act on (spec §17).

    **Not an event.** An event is what happened; an alert is an *episode* that events
    accumulate into. Measured on real footage, forty-five seconds of one corridor
    produced eleven events and would produce eleven rows — `occurrences` is what turns
    that back into one row saying how many times it happened.
    """

    alert_id: UUID
    camera_id: str
    camera_label: str
    zone: Zone | None = Field(
        default=None, description="Where the camera watches, or null when ungrouped."
    )
    reason: EscalationReason = Field(
        description="What kind of situation this is. The same vocabulary as an event's."
    )
    subject: str = Field(
        description=(
            "Which tracked identity this episode is about, as a comma-separated list "
            'of track ids, or `""` for a camera-level finding with nobody to '
            "attribute it to (camera tampering). Two events with the same camera, "
            "reason and subject are the same episode; a different subject is a "
            "different person and a different alert."
        )
    )
    state: AlertState = Field(
        description=(
            "`active` — nobody has looked. `acknowledged` — a person has seen it and "
            "is dealing with it. `resolved` — a person has said it is finished. "
            "**Nothing here changes on its own:** no alert ages out, because an alert "
            "that expired quietly would leave no trace that nobody ever went to look."
        )
    )
    severity: Severity = Field(
        description=(
            "The worse of what the model saw and what the reason structurally implies, "
            "and it **rises across an episode and never falls** — one calm frame must "
            "not drop an escalating situation down the list."
        )
    )
    priority: EventPriority = Field(
        description=(
            "How urgently this should be served, known from the reason alone before "
            "any model has looked. Distinct from `severity`: severity is how bad the "
            "scene appears, priority is how bad it would be to get this one wrong."
        )
    )
    first_seen: float
    last_seen: float = Field(
        description="Unix epoch seconds, like `occurred_at` — sortable across cameras."
    )
    occurrences: int = Field(
        description=(
            "How many events have folded into this episode. Always exact, even once "
            "`event_ids` stops being complete."
        )
    )
    description: str = Field(
        description=(
            "The most recent contributing event's description. Most recent rather than "
            "first because an episode develops — what is happening now is more use to "
            "somebody deciding whether to go than what was happening a minute ago."
        )
    )
    event_ids: list[UUID] = Field(
        description=(
            "The **first** contributing events, bounded. First rather than latest "
            "because an investigator works backwards from the start of an episode, and "
            "the oldest event is the one whose clip shows how it began."
        )
    )
    clip_uri: str | None = Field(
        default=None,
        description=(
            "The first contributing clip that finished, for the same reason. Null when "
            "no clip has finished yet, which is normal early in an episode — clips "
            "complete after their event is assembled. A storage URI, not a URL: it is "
            "here so an operator can say which object an alert refers to. To *watch* "
            "it, GET /alerts/{alert_id}/clip, which is the only route by which this "
            "engine will serve a recording."
        ),
    )
    notify_clip_uri: str | None = Field(
        default=None,
        description=(
            "The same recording trimmed to SENTINEL_NOTIFY_CLIP_SECONDS, when the "
            "writer made one — what a notification carries, and what "
            "GET /alerts/{alert_id}/clip returns by default. Null is ordinary: the "
            "engine may be configured to make no short copy, and the trim can fail "
            "without costing the full clip. Null here while `clip_uri` is set means "
            "the short request falls back to the full recording, never to nothing."
        ),
    )
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None

    @classmethod
    def from_alert(cls, alert: Alert) -> AlertEntry:
        return cls(
            alert_id=alert.alert_id,
            camera_id=alert.key.camera_id,
            camera_label=alert.camera_label,
            zone=alert.zone,
            reason=alert.key.reason,
            subject=alert.key.subject,
            state=alert.state,
            severity=alert.severity,
            priority=alert.priority,
            first_seen=alert.first_seen,
            last_seen=alert.last_seen,
            occurrences=alert.occurrences,
            description=alert.description,
            event_ids=list(alert.event_ids),
            clip_uri=alert.clip_uri,
            notify_clip_uri=alert.notify_clip_uri,
            acknowledged_by=alert.acknowledged_by,
            acknowledged_at=alert.acknowledged_at,
        )


class AlertsResponse(BaseModel):
    alerts: list[AlertEntry] = Field(
        description=(
            "Worst first, then most recent first. Ordered by the engine rather than "
            "left to the client so that every reader agrees about what is at the top "
            "of the list — §27's requirement that the critical thing is visible "
            "immediately is a property of this ordering."
        )
    )
    open_count: int = Field(
        description="How many are not yet resolved — the number worth putting on a badge."
    )


class CameraEnabledRequest(BaseModel):
    """Which way to move the switch.

    A body rather than two verbs (`/enable`, `/disable`) so the request is idempotent
    in the literal sense: it states the state it wants, not the transition, and sending
    it twice asks for the same thing twice.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(description="`true` to start watching this camera, `false` to stop.")


class SettingEntryModel(BaseModel):
    """One configuration value, as the running process holds it."""

    name: str = Field(description="The environment variable an operator would set.")
    value: str = Field(
        description=(
            "The effective value, rendered for reading. For a credential this is "
            "`(set)` or `(unset)` and never the value — see `config_report.SECRET_FIELDS`."
        )
    )
    is_default: bool = Field(
        description="Whether this is the value the engine ships with. False means this deployment moved it."
    )
    secret: bool
    group: str


class SettingsResponse(BaseModel):
    """What this engine is configured to do.

    Read-only, and it is worth saying why: almost nothing here can change without a
    restart — which models to load, how much VRAM to budget, where the broker is — and
    an endpoint that accepted a write would have to either lie about taking effect or
    restart the engine under an operator who asked for a settings change. The two
    things that *are* live are already editable where they belong, per camera.
    """

    settings: list[SettingEntryModel]
    changed: int = Field(
        description="How many settings this deployment has moved off their default. The number worth reading first."
    )


class CameraStorageModel(BaseModel):
    camera_id: str
    clips: int
    bytes_used: int


class StorageResponse(BaseModel):
    """What the clip bucket holds."""

    bucket: str
    reachable: bool = Field(
        description=(
            "False when the object store could not be listed. Every count is then zero "
            "and means nothing — a console must say so rather than draw an empty "
            "bucket, which reads as evidence having been deleted."
        )
    )
    clips: int = Field(description="Recordings, counting a clip and its short copy as one.")
    objects: int = Field(description="Objects, which is what the store bills for — roughly twice `clips`.")
    bytes_used: int
    retention_days: int = Field(
        description="How long a clip survives, enforced by the object store's own lifecycle rule. `0` means nothing expires."
    )
    per_camera: list[CameraStorageModel]


class ClipRecordModel(BaseModel):
    event_id: UUID
    size_bytes: int
    modified_at: float = Field(
        description=(
            "Unix epoch seconds, from the object store's record of when the upload "
            "finished — **later than the incident** by the post-roll plus the upload. "
            "Sort by it; do not present it as the time something happened."
        )
    )
    has_short_copy: bool


class ClipsResponse(BaseModel):
    camera_id: str
    clips: list[ClipRecordModel]


class AcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "Who is acknowledging. **This engine has no authentication**, so this is a "
            "self-declared label and not an identity — it records what somebody typed, "
            "which is worth having and is not an audit trail. See "
            "`docs/operations.md`."
        ),
    )


LatestDescriptionState = Literal["none", "available", "unavailable"]
"""Which of the three genuinely different things the live panel is looking at.

* `none` — nothing has been assembled for this camera **in this process**. Either
  the camera has not escalated yet, or the engine restarted and the volatile ring
  went with it. Show "no description yet", not an error.
* `available` — the most recent event carries a real vision-model description of
  the scene.
* `unavailable` — the most recent event has `description_unavailable: true`: the
  vision model did not answer (a timeout, an out-of-memory, an unloaded model, or
  a shutdown that abandoned the escalation). `latest.description` on that event is
  a stand-in assembled from the escalation reason and the tracked object labels —
  it is **not** a description of the scene, and a console that renders it as one is
  telling the operator the model said something it never said. This state exists so
  that "nothing happened" and "we could not tell you what happened" are never the
  same pixel.
"""


class WelfareConcernEntry(BaseModel):
    """One welfare concern as the console reads it.

    The wire form of `domain.welfare.WelfareConcern`, minus the assessment's
    `basis` — which has exactly one value today, so repeating it on every concern
    would be noise. A second basis is a deliberate change that rewrites the
    routing rule and the console's wording together (ADR 10), not something a
    consumer should be silently reading for.
    """

    kind: ConcernKind = Field(
        description="Which of the five things the prompt asks about, or `other`."
    )
    confidence: Confidence = Field(
        description=(
            "The model's own uncertainty, not a thresholded score. There are exactly "
            "two tiers and there is no `certain`: one still frame cannot earn it."
        )
    )
    evidence: str = Field(
        description=(
            "What the model says it actually saw, in its own words — **unless** "
            "`evidence_stated` is false, in which case this is a fixed placeholder "
            "and the model described nothing. Never render one as the other."
        )
    )
    evidence_stated: bool = Field(
        description=(
            "False when the model named this concern but described nothing. The "
            "concern still routes and still displays — a concern without a stated "
            "reason is not a concern that did not happen — but a reader weighing it "
            "needs to know the difference."
        )
    )

    @classmethod
    def from_concern(cls, concern: WelfareConcern) -> WelfareConcernEntry:
        return cls(
            kind=concern.kind,
            confidence=concern.confidence,
            evidence=concern.evidence,
            evidence_stated=concern.evidence_stated,
        )


class RecentEventEntry(BaseModel):
    """One event as the console sees it — a projection of the published anomaly event,
    not the event itself.

    The same shape is served three ways: in `GET /cameras/{camera_id}/events`, in the
    `backlog` array that opens `GET /events/stream`, and in each live frame on that
    stream. One shape on purpose, so a console can merge all three into one list
    keyed by `event_id`, keeping the copy with the highest `sequence`.
    """

    event_id: UUID
    camera_id: str = Field(
        description=(
            "Which camera produced this. Redundant on the per-camera endpoint, and "
            "essential on `/events/stream`, which carries every camera's events."
        )
    )
    sequence: int = Field(
        description=(
            "This version's position in the engine's write order for the recent-event "
            "ring — a de-duplication key, never a sort key. An event can be sent more "
            "than once with the same `event_id`: it is re-sent when something about it "
            "changes, currently when its clip finishes uploading and `clip_uri` "
            "appears. The copy with the higher `sequence` is the newer one; two copies "
            "with the same `sequence` are the same copy. Per process and monotonic — "
            "it restarts from zero when the engine does, exactly like the ring itself, "
            "so never persist it or compare it across a restart."
        )
    )
    clip_uri: str | None = Field(
        description=(
            "Where this event's clip was written, or null. Null covers three different "
            "situations and is not by itself evidence of any one of them: no clip was "
            "being recorded, the clip has not finished uploading yet (it lands shortly "
            "after the event, and the event is then re-sent on the stream with a higher "
            "`sequence`), or the clip failed to write. An event is never withheld "
            "because its clip failed — spec §9 — so a null here says nothing at all "
            "about whether the event happened."
        )
    )
    occurred_at: float = Field(
        description=(
            "Unix epoch seconds (UTC, fractional). Sort and plot on this — it is the "
            "same field, with the same meaning, as the published event's `occurred_at`."
        )
    )
    source_timestamp: float | None = Field(
        description=(
            "The camera source's own timeline for the same instant. Correlates with a "
            "clip's pts; meaningful only within one process run for one camera, so "
            "never sort or display on it."
        )
    )
    reason: str = Field(description="Which of the seven escalation triggers fired.")
    threat_score: float = Field(ge=0.0, le=1.0, description="The threat value, 0.0 to 1.0.")
    severity: str = Field(description="The band `threat_score` falls in.")
    description: str = Field(
        description=(
            "The scene description — unless `description_unavailable` is true, in which "
            "case this is a metadata-derived stand-in and not a description of the scene."
        )
    )
    suggested_action: str
    description_unavailable: bool = Field(
        description=(
            "True when the vision model could not answer and this event was assembled "
            "from cheap signals alone. Never conflate it with the absence of an event."
        )
    )
    labels: list[str]
    track_ids: list[int]
    welfare_concerns: list[WelfareConcernEntry] = Field(
        description=(
            "What the vision model said about a person's wellbeing in this frame — "
            "**an opinion about one still frame, not a detector's finding.** Empty "
            "means it reported nothing, which is not evidence that nothing happened; "
            "when `description_unavailable` is true it means the model could not "
            "answer at all. Always a list, never absent.\n\n"
            "**Not filtered by the camera's `notify_on`.** That policy decides which "
            "concerns are pushed to somebody who is *not* watching the console; every "
            "concern the model reported appears here, so muting a camera silences its "
            "notifications without also blinding the operator reading its events."
        ),
    )

    @classmethod
    def from_recent_event(cls, event: RecentEvent) -> RecentEventEntry:
        return cls(
            event_id=event.event_id,
            camera_id=event.camera_id,
            sequence=event.sequence,
            clip_uri=event.clip_uri,
            occurred_at=event.occurred_at,
            source_timestamp=event.source_timestamp,
            reason=event.reason.value,
            threat_score=event.threat_score,
            severity=event.severity.value,
            description=event.description,
            suggested_action=event.suggested_action,
            description_unavailable=event.description_unavailable,
            labels=list(event.labels),
            track_ids=list(event.track_ids),
            welfare_concerns=[
                WelfareConcernEntry.from_concern(concern) for concern in event.welfare_concerns
            ],
        )


class CameraEventsResponse(BaseModel):
    """A **volatile, bounded, in-memory view for the operator console. Not the event
    store, and not an audit trail.**

    The durable record of what happened is RabbitMQ plus the Phase 1C consumer that
    writes it down. This endpoint reads an in-process ring that (a) is emptied by any
    restart of the engine, and (b) silently discards a camera's oldest entry once that
    camera has produced more than `capacity` events. An absent event here therefore
    means "not in the last `capacity` events of this process run" and never "did not
    happen". Do not reconcile against it, do not report from it, and do not use it to
    establish that nothing occurred.
    """

    camera_id: str
    capacity: int = Field(
        description=(
            "The per-camera ring bound. Once `returned` reaches it, every new event "
            "silently evicts the oldest — published here so a console can say how much "
            "history it is not showing."
        )
    )
    returned: int = Field(description="How many events this response carries.")
    volatile: Literal[True] = Field(
        default=True,
        description=(
            "Always true, and present so it cannot be overlooked: this history is held "
            "in engine memory only and does not survive a restart. The event store is "
            "elsewhere."
        ),
    )
    latest_description_state: LatestDescriptionState = Field(
        description=(
            "Whether the live panel has a real description ('available'), a stand-in "
            "because the vision model could not answer ('unavailable'), or nothing at "
            "all yet ('none')."
        )
    )
    latest: RecentEventEntry | None = Field(
        description=(
            "The most recent event, i.e. the last element of `events`, lifted out for "
            "the live scene panel. Read from the same snapshot as `events`, so the "
            "panel and the chart can never disagree about which event is newest. Null "
            "exactly when `latest_description_state` is 'none'."
        )
    )
    events: list[RecentEventEntry] = Field(
        description="Retained events, oldest first — plot them in this order."
    )

    @classmethod
    def from_history(cls, history: CameraEventHistory) -> CameraEventsResponse:
        latest = history.latest
        return cls(
            camera_id=history.camera_id,
            capacity=history.capacity,
            returned=len(history.events),
            latest_description_state=_latest_description_state(latest),
            latest=None if latest is None else RecentEventEntry.from_recent_event(latest),
            events=[RecentEventEntry.from_recent_event(event) for event in history.events],
        )


def _latest_description_state(latest: RecentEvent | None) -> LatestDescriptionState:
    """The three-way discriminator, derived rather than stored so it cannot drift
    from the event it describes."""
    if latest is None:
        return "none"
    return "unavailable" if latest.description_unavailable else "available"
