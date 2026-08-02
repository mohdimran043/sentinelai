"""Pydantic wire models. Domain dataclasses never cross the API boundary
directly — these mirror the relevant fields so the wire shape and the
orchestrator's internal shape can change independently."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from sentinel_ai.adapters.config.camera_file import (
    EDITABLE_FIELDS,
    RESTART_REQUIRED_FIELDS,
    UNSET,
    CameraConfig,
    CameraEdit,
)
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


class CameraEditRequest(BaseModel):
    """A partial edit to one camera's record: `label`, `zone`, or both.

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

    @model_validator(mode="after")
    def _reject_an_edit_that_asks_for_nothing(self) -> CameraEditRequest:
        """Two ways of naming no change, both rejected here rather than downstream.

        `{}` most often means a console lost track of which fields it was
        submitting; `{"label": null}` means it sent a cleared input. Either one
        answered 200 would report a successful save for a request that changed
        nothing, and the operator would believe their edit landed.

        `label: null` is caught here rather than by the type because the type has to
        stay nullable: `None` is `label`'s "not mentioned" default, and pydantic
        cannot distinguish an omitted optional from an explicit null without
        `model_fields_set`. `zone` is the opposite case and deliberately so — a null
        zone is a real instruction to ungroup.
        """
        if "label" in self.model_fields_set and self.label is None:
            raise ValueError(
                "'label' must be a non-empty string; omit the field to leave the "
                "label unchanged. A camera always has a label — an unset one falls "
                "back to its id at load, which is not the same as no label."
            )
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
        keep apart. The validator above has already ruled out a mentioned-but-null
        `label`, so the narrowing below cannot silently drop one.
        """
        mentioned = self.model_fields_set
        return CameraEdit(
            label=self.label if "label" in mentioned and self.label is not None else UNSET,
            zone=self.zone if "zone" in mentioned else UNSET,
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
