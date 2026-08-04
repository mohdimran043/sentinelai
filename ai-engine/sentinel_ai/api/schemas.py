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
from sentinel_ai.domain.welfare import ConcernKind, Confidence
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
    frames_seen: int
    frames_dropped: int
    detections_run: int
    escalations: int
    escalations_dropped: int
    discontinuities: int
    last_frame_at: float | None
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
    def from_telemetry(cls, telemetry: CameraTelemetry) -> CameraStatus:
        return cls(
            label=telemetry.label,
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
            notify_on=sorted(config.notify_on),
            notify_min_confidence=config.notify_min_confidence,
            clip_preroll_seconds=config.clip_preroll_seconds,
            clip_postroll_seconds=config.clip_postroll_seconds,
            summary_interval_seconds=config.summary_interval_seconds,
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
