"""The camera record on disk: read at startup, and edited at runtime.

This module used to live inside `sentinel_ai/main.py`, which is still the only
place that *composes* anything from it — `main` re-exports `CameraConfig`,
`CameraConfigError` and `load_cameras` unchanged, so nothing that imported them
from there has to move. It is a module of its own now because the file stopped
being read-only: `CameraFileStore` writes it back, and a writer belongs beside
the reader whose invariants it has to keep, not in a composition root.

The document shape is unchanged and documented in `main`'s own docstring and in
`cameras.example.json`.

What may be edited, and what may not
------------------------------------
`EDITABLE_FIELDS` is the list: `label`, `zone`, and the five per-camera welfare
policy fields. What they have in common is not that they are inert — three of them
are genuinely policy — but that changing one can never invalidate a decision the
pipeline has already taken:

* `label` and `zone` are metadata that `CameraRunner` carries so `telemetry()` and
  the assembled event can report them. The loop never branches on either, so
  changing one is an attribute write on a live object and the next frame is
  unaffected.
* `notify_on` and `notify_min_confidence` decide what a *notification* does with
  an already-assembled concern, which is downstream of the pipeline entirely.
* `clip_preroll_seconds`, `clip_postroll_seconds` and `summary_interval_seconds`
  **are** read by the running camera — the gate compares the interval every frame
  and the clip lengths shape every recording. They are still editable because each
  is read afresh at the point it is used and nothing is derived from an earlier
  reading: a clip already recording keeps the deadline it opened with, and the new
  values apply from the next escalation. `CameraRunner.apply_metadata` states that
  argument field by field, and is the place to check it if any of these grows a
  second reader.

`profile` — the policy the escalation gate is part-way through applying — is the
case where that does not hold, which is why it is not editable.

`url` and `profile` are deliberately **not** editable through this store.

* `url` names the source. Changing it means tearing down a running
  `CameraRunner` — its producer task, its `_LatestSlot`, its pre-roll buffer, any
  clip mid-recording — and building a new `FrameSource` in its place. That is a
  camera restart wearing an edit's clothes, and doing it under an HTTP request
  would drop frames and possibly an in-flight escalation. A second reason is
  narrower and just as decisive: an RTSP URL routinely carries credentials
  (`rtsp://user:pass@host/stream`), and this engine has no authentication, so the
  URL is never put on the wire in either direction. Edit `cameras.json` and
  restart.
* `profile` is the escalation policy — cooldowns, thresholds, the token bucket
  the gate is mid-way through spending. Swapping it under a running gate would
  reset or contradict `GateState` in ways that have no obvious right answer.
  Restart.

Neither is silently dropped: `api.schemas.CameraEdit` forbids unknown fields, so
a request carrying `url` is rejected with a message naming the restart, and this
store never sees it.

Atomicity
---------
Write-then-rename, the same technique `RabbitMQPublisher._spool` uses: the new
document goes to a `.tmp` sibling and is moved into place with `Path.replace`,
an atomic rename on the same filesystem. A reader therefore sees the old
document or the new one and never a half-written one.

Two things are added on top of the spool's version, because the stakes differ. A
spool file is a duplicate of an event the process still holds, so losing one to a
crash mid-write costs a retry; `cameras.json` is the only copy of the camera
record, so the temporary file is `fsync`ed before the rename and the directory is
`fsync`ed after it. And the document is **re-parsed in full before it is
written** — see `CameraFileStore.apply`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Any, Final, Literal

from sentinel_ai.domain.behaviour.abandonment import AbandonmentPolicy
from sentinel_ai.domain.behaviour.fall import FallPolicy
from sentinel_ai.domain.behaviour.tamper import TamperPolicy
from sentinel_ai.domain.behaviour.zones import (
    CrossingDirection,
    CrossingLine,
    RestrictedZone,
    ZonePolicy,
)
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.domain.capabilities import DEFAULT_CAPABILITIES, CameraCapabilities, Capability
from sentinel_ai.domain.policy.authorization import AuthorizationPolicy
from sentinel_ai.domain.welfare import ConcernKind, Confidence
from sentinel_ai.domain.zone import Zone

logger = logging.getLogger(__name__)

__all__ = [
    "EDITABLE_FIELDS",
    "RESTART_REQUIRED_FIELDS",
    "UNSET",
    "CameraConfig",
    "CameraConfigError",
    "CameraEdit",
    "CameraFileStore",
    "Unset",
    "edited_document",
    "enabled_document",
    "load_cameras",
    "parse_cameras",
]

_PROFILE_FIELDS = frozenset(field.name for field in fields(CameraProfile)) - {"camera_id"}

EDITABLE_FIELDS: Final = (
    "label",
    "zone",
    "capabilities",
    "notify_on",
    "notify_min_confidence",
    "clip_preroll_seconds",
    "clip_postroll_seconds",
    "summary_interval_seconds",
)
"""The fields `CameraFileStore.apply` will change. Published so the API contract
and this module cannot drift about which those are."""

_NULLABLE_SECONDS_FIELDS: Final = (
    "clip_preroll_seconds",
    "clip_postroll_seconds",
    "summary_interval_seconds",
)
"""The editable fields whose `None` means "fall back to the global or profile
default". Named once because the *writer* treats all three identically — their
bounds differ, but that is the reader's business (`_optional_seconds_from`) and
the API's, not this list's."""

RESTART_REQUIRED_FIELDS: Final = ("url", "profile")
"""Stored in the file, honoured at startup, and not editable at runtime. See this
module's docstring for why each one is here."""


class CameraConfigError(ValueError):
    """The camera file is missing, unparseable, or describes something unbuildable.

    Fail-loud at startup on purpose: every alternative (skip the bad camera, fall
    back to defaults) produces a process that runs while silently watching fewer
    cameras than the operator configured.

    The same type is raised by `CameraFileStore.apply`, and for the same reason —
    an edit that would produce a file the loader rejects is refused whole rather
    than written and discovered at the next restart.
    """


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_id: str
    label: str
    url: str
    profile: CameraProfile
    zone: Zone | None = None
    """Which space this camera watches, or None when nobody has grouped it.

    Optional, and its absence is *not* a configuration error: every camera file
    written before zones existed keeps loading, and an ungrouped camera is a true
    statement about a deployment rather than a broken one. A zone that is *present and
    unknown* is a different thing — see `_zone_from`."""

    enabled: bool = True
    """Whether the engine builds a `CameraRunner` for this camera at all.

    `False` is a camera an operator deliberately stopped: the record stays here, whole,
    and nothing watches the source. It is **not** a way of saying "watch it but run
    nothing on it" — that is `capabilities: []`, and the difference is the source
    connection. A disabled camera opens no socket, decodes nothing and costs nothing;
    a camera with no capabilities still pulls and decodes every frame.

    Absent means enabled, so every camera file written before this field existed keeps
    describing exactly what it used to.

    Not in `EDITABLE_FIELDS`. Everything on that list is metadata a running camera can
    absorb between frames; this one starts or stops the runner, which is a lifecycle
    change with its own path (`CameraFileStore.set_enabled`) precisely so it can never
    be confused for one.
    """

    notify_on: frozenset[ConcernKind] = frozenset(ConcernKind)
    """Which welfare concern kinds this camera notifies a human about. Not read
    here — a later task's routing reads it — this module only loads and
    validates it.

    Absent means "every kind": a freshly configured camera notifies on
    anything the VLM flags until an operator narrows it. An explicit empty
    list means the opposite: never notify this camera, without turning
    welfare monitoring off entirely. Collapsing that distinction either
    starts alerting on a camera an operator silenced, or silences one they
    never touched — see `_notify_on_from`, which mirrors `_zone_from`'s
    absent-vs-unknown handling for the same reason."""

    notify_min_confidence: Confidence = Confidence.LIKELY
    """The minimum `domain.welfare.Confidence` tier a concern must clear before
    it notifies. Defaults to the stronger tier: notifying on every `possible`
    guess would fire on the majority of an ordinary day's VLM opinions, and an
    operator who wants that lower bar can ask for it explicitly."""

    clip_preroll_seconds: float | None = None
    """Seconds of buffered video to include before a notified concern's
    keyframe. `None` means use the process-wide default
    (`Settings.clip_preroll_seconds`); `CameraRunner` resolves that.

    `0.0` is a real value meaning "no lead-in", not a disabled field — see
    `adapters/sources/preroll.py`, which has to accept it because this edge and
    the API edge both do."""

    clip_postroll_seconds: float | None = None
    """Seconds of video to keep recording after a notified concern's keyframe.
    `None` means use the process-wide default
    (`Settings.clip_postroll_seconds`); `CameraRunner` resolves that."""

    summary_interval_seconds: float | None = None
    """How often this camera's periodic summary runs, in seconds. `None` means
    use `profile.summary_interval_seconds`.

    Deliberately a field of its own rather than a replacement for
    `profile.summary_interval_seconds` — the two look redundant, but they are
    not: `profile` is in `RESTART_REQUIRED_FIELDS` and cannot change without
    tearing down the camera runner (see this module's docstring), while this
    field is runtime-editable. `CameraRunner` composes the two by copying the
    profile with this value substituted, so the override wins where both are
    set and reverting it to `None` restores the profile's own. Do not fold them
    together; that would take away the one thing this duplication buys."""

    fall_policy: FallPolicy = field(default_factory=FallPolicy)
    """Thresholds for this camera's fall detection (spec §7, §12).

    Read only when `capabilities` includes `fall_detection`; carried regardless, so a
    camera can be configured ahead of being switched on. `default_factory` rather than
    a bare `FallPolicy()` default for `Event.welfare`'s reason — a bare default is
    evaluated once at class-definition time and shared by every record that does not
    set its own. `FallPolicy` is frozen so aliasing would not corrupt anything today,
    but the factory is the correct mechanism rather than a shortcut that happens to
    work.

    Per camera because the thresholds are genuinely per camera: a corridor watched
    head-on and a dayroom watched from a high corner disagree about what a descent
    looks like, and §12 forbids hard-coding any of it."""

    abandonment_policy: AbandonmentPolicy = field(default_factory=AbandonmentPolicy)
    """Thresholds for this camera's abandoned-object detection. Read only when
    `capabilities` includes `abandoned_object`; `default_factory` for `fall_policy`'s
    reason."""

    tamper_policy: TamperPolicy = field(default_factory=TamperPolicy)
    """Thresholds for this camera's tamper detection. Read only when `capabilities`
    includes `camera_tamper`."""

    zone_policy: ZonePolicy = field(default_factory=ZonePolicy)
    """The restricted areas and virtual boundaries drawn on this camera, in normalised
    frame coordinates. Read only when `capabilities` includes `zone_monitoring`.

    Empty by default, and an empty policy is one the composition root drops rather than
    attaches: a camera with the capability on and no geometry drawn would otherwise be
    a checkbox that does nothing. See `ZonePolicy.is_empty`."""

    authorization_policy: AuthorizationPolicy = field(default_factory=AuthorizationPolicy)
    """Thresholds for this camera's person authorisation (§10, §11, §12).

    Read only when `capabilities` includes `person_authorization`. Per camera because
    the numbers genuinely are: a camera two metres from a doorway sees faces four times
    the size of one across a lobby, so `min_box_pixels` cannot be one site-wide value.
    §12 forbids hard-coding any of it."""

    capabilities: CameraCapabilities = DEFAULT_CAPABILITIES
    """Which AI capabilities run on this camera (spec §13).

    Absent means `DEFAULT_CAPABILITIES` — scene description plus the automatic
    triggers, which is exactly what every camera did before this field existed, so
    upgrading an engine changes no behaviour. An explicit `[]` is the opposite and
    is a real instruction: watch this camera, run nothing on it. That absent-versus-
    empty distinction is `notify_on`'s exactly, and it has to survive the same round
    trip through the PATCH body and back into the file — see `_capabilities_from`.

    Read by `main.build_models`, which loads only the model roles the union of every
    camera's capabilities actually requires, and by `main.compose`, which derives
    `CameraProfile.auto_escalation_enabled` from it rather than letting the two disagree."""


class _Unset(Enum):
    """Single-member enum rather than `object()`: mypy narrows `X | Unset` on an
    `is UNSET` test only for a literal type, and every edit here has to keep
    "the caller did not mention this field" apart from "the caller set it to
    null"."""

    TOKEN = auto()


UNSET: Final = _Unset.TOKEN
Unset = Literal[_Unset.TOKEN]


@dataclass(frozen=True, slots=True)
class CameraEdit:
    """A partial change to one camera's record.

    `zone` is the field that forces the sentinel: `None` is a real, requestable
    value meaning "ungroup this camera", so a plain `Zone | None = None` default
    could not tell an ungrouping from a request that never mentioned the zone at
    all — and silently re-grouping (or silently ungrouping) a camera an operator
    did not ask about is precisely the kind of half-applied edit this whole path
    exists to make impossible. `label` uses the same sentinel for symmetry.

    The three duration overrides are `zone`'s case exactly: `None` means "revert
    to the global or profile default", so it has to stay distinguishable from
    "not mentioned". `notify_on` and `notify_min_confidence` are not nullable at
    all — `notify_on=frozenset()` is already the "never notify" instruction and a
    confidence threshold has no absent state, so `None` would be a third meaning
    neither field has room for.
    """

    label: str | Unset = UNSET
    zone: Zone | Unset | None = UNSET
    capabilities: CameraCapabilities | Unset = UNSET
    """The whole replacement set, or `UNSET` to leave it alone.

    Not nullable, for `notify_on`'s reason exactly: `CameraCapabilities.none()` is
    already the "run nothing on this camera" instruction, so `null` would be a third
    meaning the field has no room for.
    """
    notify_on: frozenset[ConcernKind] | Unset = UNSET
    notify_min_confidence: Confidence | Unset = UNSET
    clip_preroll_seconds: float | Unset | None = UNSET
    clip_postroll_seconds: float | Unset | None = UNSET
    summary_interval_seconds: float | Unset | None = UNSET

    @property
    def is_empty(self) -> bool:
        """True when nothing was asked for. The API rejects this rather than
        reporting a successful write that changed nothing.

        Read off the dataclass's own fields rather than a list repeated here: a
        field added above and forgotten in this property would make an edit that
        names only that field look like an edit that names nothing, and be
        refused.
        """
        return all(getattr(self, field.name) is UNSET for field in fields(self))


# -- reading ------------------------------------------------------------------------


def _profile_from(camera_id: str, raw: Mapping[str, Any]) -> CameraProfile:
    unknown = sorted(set(raw) - _PROFILE_FIELDS)
    if unknown:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown profile field(s) {unknown}; "
            f"valid fields are {sorted(_PROFILE_FIELDS)}"
        )
    overrides = dict(raw)
    if "salient_classes" in overrides:
        overrides["salient_classes"] = frozenset(overrides["salient_classes"])
    try:
        return CameraProfile(camera_id=camera_id, **overrides)
    except (TypeError, ValueError) as exc:
        # CameraProfile.__post_init__ enforces its own invariants; surfacing them as
        # a CameraConfigError keeps every startup configuration failure one type.
        raise CameraConfigError(f"camera {camera_id!r}: invalid profile: {exc}") from exc


def _enabled_from(camera_id: str, raw: Any) -> bool:
    """Absent means enabled, and anything that is not a bool is refused.

    Strict rather than truthy: `"false"` and `0` are exactly the values somebody writes
    meaning "off", and Python would read both as on. A camera an operator believes they
    stopped, quietly watching, is the one failure this field must not have.
    """
    if raw is None:
        return True
    if not isinstance(raw, bool):
        raise CameraConfigError(
            f"camera {camera_id!r} 'enabled' must be true or false, not {raw!r}"
        )
    return raw


def _zone_from(camera_id: str, raw: Any) -> Zone | None:
    """Absent means ungrouped; present-but-unknown means the file is wrong.

    Keeping those two apart is the whole contract of this field. Silently downgrading
    `"hallway"` to ungrouped would produce a console that looks right and quietly
    leaves a camera out of the group the operator put it in — the same class of
    failure `_profile_from`'s unknown-field check exists to prevent — while treating
    the absent case as an error would make every pre-zone camera file unloadable for
    no gain.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'zone' must be a string or absent, got {type(raw).__name__}"
        )
    try:
        return Zone(raw)
    except ValueError:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown zone {raw!r}; "
            f"valid zones are {[zone.value for zone in Zone]}, or omit the field to "
            f"leave the camera ungrouped"
        ) from None


def _notify_on_from(camera_id: str, raw: Any) -> frozenset[ConcernKind]:
    """Absent means every kind; present-but-unknown means the file is wrong.

    The same shape as `_zone_from`, with one distinction that field does not
    need: `notify_on: []` is not "absent" and must not fall back to "every
    kind" — it is an operator's explicit instruction to mute this camera's
    welfare notifications, and treating it the same as an unconfigured field
    would either silence a camera nobody asked to silence or keep alerting on
    one an operator deliberately quieted.
    """
    if raw is None:
        return frozenset(ConcernKind)
    if not isinstance(raw, list):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'notify_on' must be an array of strings or absent, "
            f"got {type(raw).__name__}"
        )
    kinds: set[ConcernKind] = set()
    for item in raw:
        if not isinstance(item, str):
            raise CameraConfigError(
                f"camera {camera_id!r}: 'notify_on' entries must be strings, got "
                f"{item!r} ({type(item).__name__})"
            )
        try:
            kinds.add(ConcernKind(item))
        except ValueError:
            raise CameraConfigError(
                f"camera {camera_id!r}: unknown concern kind {item!r} in 'notify_on'; "
                f"valid kinds are {[kind.value for kind in ConcernKind]}, or omit the "
                f"field to notify on every kind"
            ) from None
    return frozenset(kinds)


def _notify_min_confidence_from(camera_id: str, raw: Any) -> Confidence:
    """Same unknown-value contract as `_zone_from` and `_notify_on_from`: a
    typo'd confidence is a threshold the operator meant to set and did not."""
    if raw is None:
        return Confidence.LIKELY
    if not isinstance(raw, str):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'notify_min_confidence' must be a string or absent, "
            f"got {type(raw).__name__}"
        )
    try:
        return Confidence(raw)
    except ValueError:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown confidence {raw!r} in 'notify_min_confidence'; "
            f"valid values are {[confidence.value for confidence in Confidence]}, or omit "
            f"the field to default to {Confidence.LIKELY.value!r}"
        ) from None


def _capabilities_from(camera_id: str, raw: Any) -> CameraCapabilities:
    """Absent means the default set; present-but-unknown means the file is wrong.

    `_notify_on_from`'s shape exactly, and for a sharper reason. A mistyped concern
    kind narrows which notifications go out from a camera that is still being
    watched; a mistyped capability leaves an operator believing a capability is
    running on a camera where nothing was ever switched on. Both fail loud, and this
    one has the stronger case for it.

    `[]` is not absent: it is the explicit "run nothing here" instruction, and
    collapsing it into the default would silently re-enable monitoring on a camera
    someone deliberately quieted — the same absent-versus-empty trap `notify_on`
    documents, with the polarity that matters more.
    """
    if raw is None:
        return DEFAULT_CAPABILITIES
    if not isinstance(raw, list):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'capabilities' must be an array of strings or absent, "
            f"got {type(raw).__name__}"
        )
    for item in raw:
        if not isinstance(item, str):
            raise CameraConfigError(
                f"camera {camera_id!r}: 'capabilities' entries must be strings, got "
                f"{item!r} ({type(item).__name__})"
            )
    try:
        return CameraCapabilities.from_names(raw)
    except ValueError as exc:
        # `CameraCapabilities.from_names` raises a bare ValueError naming the valid
        # vocabulary; re-raising as CameraConfigError keeps every startup
        # configuration failure one type, as `_profile_from` does for CameraProfile.
        raise CameraConfigError(f"camera {camera_id!r}: {exc}") from exc


_FALL_POLICY_FIELDS = frozenset(f.name for f in fields(FallPolicy))
_ABANDONMENT_POLICY_FIELDS = frozenset(f.name for f in fields(AbandonmentPolicy))
_TAMPER_POLICY_FIELDS = frozenset(f.name for f in fields(TamperPolicy))
_AUTHORIZATION_POLICY_FIELDS = frozenset(f.name for f in fields(AuthorizationPolicy))


def _policy_from[PolicyT](
    camera_id: str,
    key: str,
    raw: Any,
    *,
    factory: Callable[..., PolicyT],
    valid_fields: frozenset[str],
) -> PolicyT:
    """A flat threshold object, validated against its own dataclass field names.

    `_profile_from`'s shape exactly, generalised over the three behaviour policies that
    are plain bags of numbers (`fall_policy`, `abandonment_policy`, `tamper_policy`).
    `zone_policy` is not one of them — it carries nested geometry and has its own
    parser below.

    A typo'd threshold name that was silently ignored would leave an operator believing
    they had tuned a camera that is still running the defaults, and here the defaults
    decide whether anyone is told a person is on the floor. So unknown fields, wrong
    types and out-of-range values all fail at startup rather than at the first incident.
    """
    if raw is None:
        return factory()
    if not isinstance(raw, dict):
        raise CameraConfigError(
            f"camera {camera_id!r}: {key!r} must be an object or absent, got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - valid_fields)
    if unknown:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown {key} field(s) {unknown}; "
            f"valid fields are {sorted(valid_fields)}"
        )
    try:
        return factory(**raw)
    except (TypeError, ValueError) as exc:
        raise CameraConfigError(f"camera {camera_id!r}: invalid {key}: {exc}") from exc


def _point_from(camera_id: str, where: str, raw: Any) -> tuple[float, float]:
    """One `[x, y]` pair of frame fractions.

    Fractions rather than pixels so a zone survives an RTSP resolution renegotiation —
    see `domain/behaviour/zones.py`. `bool` is rejected for `_optional_seconds_from`'s
    reason: `isinstance(True, int)` is `True`, so `[true, 0.5]` would otherwise parse
    as the perfectly plausible point `(1.0, 0.5)`.
    """
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise CameraConfigError(
            f"camera {camera_id!r}: {where} must be a two-element [x, y] array of "
            f"frame fractions, got {raw!r}"
        )
    values: list[float] = []
    for value in raw:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise CameraConfigError(
                f"camera {camera_id!r}: {where} coordinates must be numbers, got {value!r}"
            )
        if not math.isfinite(value):
            raise CameraConfigError(
                f"camera {camera_id!r}: {where} coordinates must be finite, got {value!r}"
            )
        values.append(float(value))
    return (values[0], values[1])


def _zone_policy_from(camera_id: str, raw: Any) -> ZonePolicy:
    """The restricted areas and tripwires drawn on one camera.

    Nested, unlike the other three policies, so it gets its own parser rather than a
    generic one. Every error names the camera and the zone, because an operator reading
    a startup failure has a file with several cameras in it and needs to know which
    polygon they mistyped.
    """
    if raw is None:
        return ZonePolicy()
    if not isinstance(raw, dict):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'zone_policy' must be an object or absent, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - {"zones", "lines", "min_frames_inside"})
    if unknown:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown zone_policy field(s) {unknown}; "
            f"valid fields are ['lines', 'min_frames_inside', 'zones']"
        )

    zones: list[RestrictedZone] = []
    for index, entry in enumerate(raw.get("zones") or []):
        if not isinstance(entry, dict):
            raise CameraConfigError(f"camera {camera_id!r}: zones[{index}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str):
            raise CameraConfigError(
                f"camera {camera_id!r}: zones[{index}] needs a string 'name' — it is "
                f"what an operator reads in the alert"
            )
        polygon_raw = entry.get("polygon")
        if not isinstance(polygon_raw, list):
            raise CameraConfigError(
                f"camera {camera_id!r}: zone {name!r} needs a 'polygon' array of [x, y] points"
            )
        polygon = tuple(
            _point_from(camera_id, f"zone {name!r} polygon point {point_index}", point)
            for point_index, point in enumerate(polygon_raw)
        )
        try:
            zones.append(RestrictedZone(name=name, polygon=polygon))
        except ValueError as exc:
            raise CameraConfigError(f"camera {camera_id!r}: {exc}") from exc

    lines: list[CrossingLine] = []
    for index, entry in enumerate(raw.get("lines") or []):
        if not isinstance(entry, dict):
            raise CameraConfigError(f"camera {camera_id!r}: lines[{index}] must be an object")
        name = entry.get("name")
        if not isinstance(name, str):
            raise CameraConfigError(f"camera {camera_id!r}: lines[{index}] needs a string 'name'")
        direction_raw = entry.get("direction", CrossingDirection.BOTH.value)
        try:
            direction = CrossingDirection(direction_raw)
        except ValueError:
            raise CameraConfigError(
                f"camera {camera_id!r}: line {name!r} has unknown direction "
                f"{direction_raw!r}; valid values are "
                f"{[member.value for member in CrossingDirection]}"
            ) from None
        try:
            lines.append(
                CrossingLine(
                    name=name,
                    start=_point_from(camera_id, f"line {name!r} start", entry.get("start")),
                    end=_point_from(camera_id, f"line {name!r} end", entry.get("end")),
                    direction=direction,
                )
            )
        except ValueError as exc:
            raise CameraConfigError(f"camera {camera_id!r}: {exc}") from exc

    minimum = raw.get("min_frames_inside", 3)
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise CameraConfigError(
            f"camera {camera_id!r}: zone_policy.min_frames_inside must be an integer, "
            f"got {minimum!r}"
        )
    try:
        return ZonePolicy(zones=tuple(zones), lines=tuple(lines), min_frames_inside=minimum)
    except ValueError as exc:
        raise CameraConfigError(f"camera {camera_id!r}: invalid zone_policy: {exc}") from exc


def _fall_policy_from(camera_id: str, raw: Any) -> FallPolicy:
    """Per-camera fall thresholds, validated against `FallPolicy`'s own field names.

    `_profile_from`'s shape exactly, and for its reason: a typo'd threshold name that
    was silently ignored would leave an operator believing they had tuned a camera
    that is still running the defaults — and here the defaults decide whether anyone
    is told that a person is on the floor.

    Absent means every default. Present-but-misspelled, or present-and-out-of-range,
    fails at startup rather than at the first fall.
    """
    if raw is None:
        return FallPolicy()
    if not isinstance(raw, dict):
        raise CameraConfigError(
            f"camera {camera_id!r}: 'fall_policy' must be an object or absent, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(set(raw) - _FALL_POLICY_FIELDS)
    if unknown:
        raise CameraConfigError(
            f"camera {camera_id!r}: unknown fall_policy field(s) {unknown}; "
            f"valid fields are {sorted(_FALL_POLICY_FIELDS)}"
        )
    try:
        return FallPolicy(**raw)
    except (TypeError, ValueError) as exc:
        # `FallPolicy.__post_init__` enforces its own invariants (the aspect and torso
        # bands must not overlap, every duration must be positive); surfacing them as
        # a `CameraConfigError` keeps every startup configuration failure one type.
        raise CameraConfigError(f"camera {camera_id!r}: invalid fall_policy: {exc}") from exc


def _optional_seconds_from(
    camera_id: str, field_name: str, raw: Any, *, minimum: float, inclusive: bool
) -> float | None:
    """A finite number within bounds, or absent meaning "use the default
    elsewhere". Shared by the three duration fields below instead of three
    near-identical bodies.

    `bool` is rejected on purpose: `isinstance(True, int)` is `True` in
    Python, so an unguarded numeric check would accept
    `"clip_preroll_seconds": true` as `1.0` — a JSON typo silently becoming a
    plausible-looking duration instead of the load error it should be.
    `math.isfinite` rejects `NaN`/`Infinity`, which `json.loads` accepts as a
    non-standard extension of the JSON it reads; without this check a file
    that loads with `"clip_postroll_seconds": NaN` would produce a clip with
    impossible bounds rather than fail at startup where it can be noticed
    (the same reasoning `adapters/vision/qwen25vl.py`'s `threat_value` parsing
    applies to a model-supplied float).
    """
    if raw is None:
        return None
    if not isinstance(raw, int | float) or isinstance(raw, bool) or not math.isfinite(raw):
        raise CameraConfigError(
            f"camera {camera_id!r}: {field_name!r} must be a finite number or absent, got {raw!r}"
        )
    value = float(raw)
    if inclusive:
        if value < minimum:
            raise CameraConfigError(
                f"camera {camera_id!r}: {field_name!r} must be >= {minimum}, got {value}"
            )
    elif value <= minimum:
        raise CameraConfigError(
            f"camera {camera_id!r}: {field_name!r} must be > {minimum}, got {value}"
        )
    return value


def parse_cameras(document: Any, origin: str) -> tuple[CameraConfig, ...]:
    """Validate an already-decoded camera document.

    Split out of `load_cameras` so that `CameraFileStore.apply` can run the exact
    same validation over the document it is *about to* write. Two parsers would be
    two vocabularies of what a valid camera file is, and the writer's would be the
    one nobody tested against a real startup.

    `origin` only names the thing being validated in error messages — a path when
    a file is being read, the same path when a file is about to be written.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries: list[Any] = document["cameras"]
    if not entries:
        # An engine with no cameras starts, serves /health, and watches nothing. That
        # is never what an operator meant.
        raise CameraConfigError(f"{origin} configures no cameras")

    configs: list[CameraConfig] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CameraConfigError(f"{origin}: cameras[{index}] must be an object")
        camera_id = entry.get("id")
        url = entry.get("url")
        if not isinstance(camera_id, str) or not camera_id:
            raise CameraConfigError(f"{origin}: cameras[{index}] needs a non-empty string 'id'")
        if not isinstance(url, str) or not url:
            raise CameraConfigError(
                f"{origin}: camera {camera_id!r} needs a non-empty string 'url'"
            )
        if camera_id in seen:
            # Duplicates are silently survivable — `EngineService` keys cameras by id,
            # so the second would simply replace the first and one camera would never
            # be watched.
            raise CameraConfigError(f"{origin}: duplicate camera id {camera_id!r}")
        seen.add(camera_id)

        label = entry.get("label", camera_id)
        if not isinstance(label, str) or not label:
            raise CameraConfigError(
                f"{origin}: camera {camera_id!r} 'label' must be a non-empty str"
            )
        raw_profile = entry.get("profile", {})
        if not isinstance(raw_profile, dict):
            raise CameraConfigError(f"{origin}: camera {camera_id!r} 'profile' must be an object")

        configs.append(
            CameraConfig(
                camera_id=camera_id,
                label=label,
                url=url,
                profile=_profile_from(camera_id, raw_profile),
                zone=_zone_from(camera_id, entry.get("zone")),
                enabled=_enabled_from(camera_id, entry.get("enabled")),
                notify_on=_notify_on_from(camera_id, entry.get("notify_on")),
                notify_min_confidence=_notify_min_confidence_from(
                    camera_id, entry.get("notify_min_confidence")
                ),
                clip_preroll_seconds=_optional_seconds_from(
                    camera_id,
                    "clip_preroll_seconds",
                    entry.get("clip_preroll_seconds"),
                    minimum=0.0,
                    inclusive=True,
                ),
                clip_postroll_seconds=_optional_seconds_from(
                    camera_id,
                    "clip_postroll_seconds",
                    entry.get("clip_postroll_seconds"),
                    minimum=0.0,
                    inclusive=False,
                ),
                summary_interval_seconds=_optional_seconds_from(
                    camera_id,
                    "summary_interval_seconds",
                    entry.get("summary_interval_seconds"),
                    minimum=0.0,
                    inclusive=False,
                ),
                capabilities=_capabilities_from(camera_id, entry.get("capabilities")),
                fall_policy=_policy_from(
                    camera_id,
                    "fall_policy",
                    entry.get("fall_policy"),
                    factory=FallPolicy,
                    valid_fields=_FALL_POLICY_FIELDS,
                ),
                abandonment_policy=_policy_from(
                    camera_id,
                    "abandonment_policy",
                    entry.get("abandonment_policy"),
                    factory=AbandonmentPolicy,
                    valid_fields=_ABANDONMENT_POLICY_FIELDS,
                ),
                tamper_policy=_policy_from(
                    camera_id,
                    "tamper_policy",
                    entry.get("tamper_policy"),
                    factory=TamperPolicy,
                    valid_fields=_TAMPER_POLICY_FIELDS,
                ),
                zone_policy=_zone_policy_from(camera_id, entry.get("zone_policy")),
                authorization_policy=_policy_from(
                    camera_id,
                    "authorization_policy",
                    entry.get("authorization_policy"),
                    factory=AuthorizationPolicy,
                    valid_fields=_AUTHORIZATION_POLICY_FIELDS,
                ),
            )
        )
    return tuple(configs)


def _read_document(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise CameraConfigError(
            f"no camera configuration at {path} — copy cameras.example.json, or point "
            f"SENTINEL_CAMERAS_FILE somewhere else"
        ) from None
    except json.JSONDecodeError as exc:
        raise CameraConfigError(f"{path} is not valid JSON: {exc}") from exc


def load_cameras(path: Path) -> tuple[CameraConfig, ...]:
    """Read and validate the camera list. See `sentinel_ai.main` for the shape."""
    return parse_cameras(_read_document(path), origin=str(path))


# -- writing ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraCreate:
    """A new camera, as an operator describes one.

    `url` is here and is *not* on `CameraEdit`, and the asymmetry is the point.
    Changing a running camera's source means tearing down its `CameraRunner`, its
    pre-roll ring and any clip mid-recording — which is why `PATCH` refuses it. Adding
    one destroys nothing: there is no runner yet, so the whole objection disappears.

    `profile` is deliberately absent. The escalation policy has live state — cooldowns, a
    token bucket part-way through refilling — and a new camera starts on the defaults;
    tuning it is `cameras.json` and a restart, exactly as it was.
    """

    camera_id: str
    url: str
    label: str = ""
    zone: Zone | None = None
    capabilities: CameraCapabilities | None = None

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise CameraConfigError("a camera needs an id")
        if not self.url.strip():
            raise CameraConfigError(f"camera {self.camera_id!r}: a camera needs a url")


def created_document(document: Any, create: CameraCreate, origin: str) -> Any:
    """The whole document with one camera appended.

    Same contract as `edited_document`: everything this version has no model of is
    copied through untouched, and the new entry is *appended* so an operator's ordering
    survives.

    A duplicate id raises rather than replacing. Two cameras with one id is a file whose
    second entry silently wins at startup, and an operator who meant to edit has a PATCH
    for that.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries = list(document["cameras"])
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id")) == create.camera_id:
            raise CameraConfigError(
                f"{origin} already has a camera called {create.camera_id!r}; "
                "edit it rather than adding it twice"
            )

    fresh: dict[str, Any] = {"id": create.camera_id, "url": create.url}
    if create.label:
        fresh["label"] = create.label
    if create.zone is not None:
        fresh["zone"] = create.zone.value
    if create.capabilities is not None:
        fresh["capabilities"] = sorted(
            capability.value for capability in Capability if create.capabilities.enabled(capability)
        )
    return {**document, "cameras": [*entries, fresh]}


def removed_document(document: Any, camera_id: str, origin: str) -> Any:
    """The whole document with one camera dropped.

    An unknown id raises, for `edited_document`'s reason: the engine knowing about a
    camera the file does not means somebody edited the file, and quietly succeeding
    would report a deletion that deleted nothing.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries = list(document["cameras"])
    remaining = [
        entry
        for entry in entries
        if not (isinstance(entry, dict) and str(entry.get("id")) == camera_id)
    ]
    if len(remaining) == len(entries):
        raise CameraConfigError(f"{origin} has no camera called {camera_id!r}")
    return {**document, "cameras": remaining}


def enabled_document(document: Any, camera_id: str, enabled: bool, origin: str) -> Any:
    """The whole document with one camera's `enabled` flag set.

    Separate from `edited_document` rather than a field on `CameraEdit`, because the two
    are different kinds of change and sharing a path would blur that. Everything
    `edited_document` writes is metadata a running camera absorbs between frames; this
    decides whether the camera runs at all. Keeping them apart means the metadata path
    can never start or stop anything, and this one can never touch policy.

    Written explicitly even when it matches the default, for `zone`'s reason: a visible
    `"enabled": true` records that somebody turned this back on, where a deleted key
    says only that nobody ever touched it.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    updated: list[Any] = []
    found = False
    for entry in document["cameras"]:
        if not isinstance(entry, dict) or entry.get("id") != camera_id:
            updated.append(entry)
            continue
        found = True
        updated.append({**entry, "enabled": enabled})

    if not found:
        raise CameraConfigError(
            f"{origin} has no camera {camera_id!r} — the file has been edited since the "
            f"engine started, so this change was not applied; reconcile the file and "
            f"restart rather than letting the console overwrite it"
        )
    return {**document, "cameras": updated}


def edited_document(document: Any, camera_id: str, edit: CameraEdit, origin: str) -> Any:
    """The whole document, with only this camera's edited keys changed.

    Pure, and takes the *whole* document rather than the one entry, because the
    property that matters is about everything it does not touch. The file an
    operator hand-wrote holds things this engine has no model of — the
    `_comment` and `_note` keys `cameras.example.json` ships, whatever a future
    field turns out to be, and every other camera in the list. Round-tripping
    through `CameraConfig` and re-serialising the parsed shape would silently
    delete all of it, which is the specific defect this function is written to
    avoid: a config writer that drops a field it did not understand.

    So the original mappings are copied and only the named keys are replaced.
    Ordering survives too (`dict` preserves insertion order and `cameras` is
    already present, so the spread below leaves it where it was).

    An unknown `camera_id` raises rather than appending: the engine knowing about
    a camera the file does not means the file has been edited since startup, and
    synthesising an entry from memory would overwrite whoever did that.
    """
    if not isinstance(document, dict) or not isinstance(document.get("cameras"), list):
        raise CameraConfigError(f"{origin} must be an object with a 'cameras' array")

    entries: list[Any] = document["cameras"]
    updated: list[Any] = []
    found = False
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("id") != camera_id:
            updated.append(entry)
            continue
        found = True
        changed = dict(entry)
        if edit.label is not UNSET:
            changed["label"] = edit.label
        if edit.zone is not UNSET:
            # An explicit `null` rather than deleting the key: `parse_cameras`
            # already reads null as ungrouped, and leaving the field visible
            # records that someone chose this, where a deleted key looks like a
            # camera nobody has got round to grouping yet.
            changed["zone"] = None if edit.zone is None else edit.zone.value
        if edit.capabilities is not UNSET:
            # `names()` is already sorted, for the same stable-diff reason
            # `notify_on` sorts below.
            changed["capabilities"] = edit.capabilities.names()
        if edit.notify_on is not UNSET:
            # Sorted, not just listed: `frozenset` iterates in an order that
            # depends on the process's string hash seed, so writing it raw would
            # make the same edit produce a different file from one run to the
            # next — a spurious diff on the one artefact an operator hand-edits.
            changed["notify_on"] = sorted(kind.value for kind in edit.notify_on)
        if edit.notify_min_confidence is not UNSET:
            changed["notify_min_confidence"] = edit.notify_min_confidence.value
        for field_name in _NULLABLE_SECONDS_FIELDS:
            value = getattr(edit, field_name)
            if value is not UNSET:
                # `None` written through as an explicit null, for `zone`'s reason
                # above: `parse_cameras` reads it as "use the default", and a
                # visible null says an operator chose that where a deleted key
                # says nobody ever looked.
                changed[field_name] = value
        updated.append(changed)

    if not found:
        raise CameraConfigError(
            f"{origin} has no camera {camera_id!r} — the file has been edited since the "
            f"engine started, so this edit was not applied; reconcile the file and "
            f"restart rather than letting the console overwrite it"
        )
    return {**document, "cameras": updated}


class CameraFileStore:
    """Read-modify-write over `cameras.json`, one edit at a time.

    Serialised by an `asyncio.Lock` because the sequence is a read, a decision and
    a write: two concurrent `PATCH`es without it would both read the pre-edit
    document and the second rename would drop the first operator's change. The
    file I/O inside the lock is synchronous rather than pushed to a thread — the
    document is a kilobyte and this is not a hot path, and a thread would buy an
    interleaving hazard in exchange for nothing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    async def apply(self, camera_id: str, edit: CameraEdit) -> CameraConfig:
        """Persist `edit` and return the camera's record as the file now holds it.

        The order is the contract:

          1. read and decode the file as it is *now*, not as it was at startup;
          2. build the edited document, preserving everything not named;
          3. **re-parse the whole document** — if the result would not load, raise
             and write nothing. Nothing is half-applied, and it is impossible for
             this path to leave behind a file that fails the next startup;
          4. only then write, atomically.

        The returned record is read back out of the re-parsed document rather than
        assembled from the request, so a caller that then applies it to a running
        camera is applying what is on disk. Memory and file cannot disagree about
        an edit that succeeded.

        Raises `CameraConfigError` for anything about the file or the edit, and
        `OSError` if the write itself fails — in which case, because the write is
        last and is a rename, the previous document is still intact.
        """
        if edit.is_empty:
            # Cheap, but it is the store's own invariant and not only the API's:
            # rewriting the file for an edit that names nothing is churn on the one
            # artefact this system cannot afford to churn, and any future caller
            # that reaches here directly deserves the same answer the API gives.
            raise CameraConfigError(
                f"camera {camera_id!r}: the edit names no fields; editable fields are "
                f"{list(EDITABLE_FIELDS)}"
            )
        async with self._lock:
            origin = str(self._path)
            document = _read_document(self._path)
            candidate = edited_document(document, camera_id, edit, origin)
            cameras = parse_cameras(candidate, origin)
            record = next(camera for camera in cameras if camera.camera_id == camera_id)
            self._write(candidate)
            return record

    async def set_enabled(self, camera_id: str, enabled: bool) -> CameraConfig:
        """Start or stop watching a camera, persistently. Returns the record as stored.

        `apply`'s order — read, transform, re-parse the whole document, then write — so
        a file this build could not load back is refused before it reaches the disk.
        """
        async with self._lock:
            origin = str(self._path)
            document = _read_document(self._path)
            candidate = enabled_document(document, camera_id, enabled, origin)
            cameras = parse_cameras(candidate, origin)
            record = next(camera for camera in cameras if camera.camera_id == camera_id)
            self._write(candidate)
            return record

    async def create(self, create: CameraCreate) -> CameraConfig:
        """Persist a new camera and return it as the file now holds it.

        `apply`'s order exactly — read, build, **re-parse the whole document**, then
        write — so a camera whose url or capabilities the loader would reject is
        refused before anything reaches the disk rather than after.
        """
        async with self._lock:
            origin = str(self._path)
            document = _read_document(self._path)
            candidate = created_document(document, create, origin)
            cameras = parse_cameras(candidate, origin)
            record = next(camera for camera in cameras if camera.camera_id == create.camera_id)
            self._write(candidate)
            return record

    async def remove(self, camera_id: str) -> None:
        """Drop a camera from the file. Re-parsed before writing, like every other path
        here, so removing one cannot leave a document the next startup refuses."""
        async with self._lock:
            origin = str(self._path)
            document = _read_document(self._path)
            candidate = removed_document(document, camera_id, origin)
            parse_cameras(candidate, origin)
            self._write(candidate)

    def _write(self, document: Any) -> None:
        """Write-then-rename, `fsync`ed at both ends.

        `RabbitMQPublisher._spool` does the rename without the syncs, which is the
        right trade there: a spool file duplicates an event the process still
        holds, so a crash that loses one costs a replay. This file is the only copy
        of the camera record, so the cost of a rename that lands ahead of the data
        is an engine that will not start.

        Three things this does that a plain write-then-rename would not:

        * **the temp file is created owner-only, and only ever widened after it
          already holds nothing readable.** `Path.open("w")` on a *new* file gets
          `0o666 & ~umask` — group- or world-readable by default on an ordinary
          host — and the file holds the *complete* document, RTSP credentials
          included, from the first byte written until the trailing `fsync` below
          returns, which is the slowest step in this whole function. Creating the
          file with that default and narrowing it with `chmod` afterwards — the
          previous shape of this code — leaves exactly that window open: a fixed,
          predictable filename, readable by anyone in the file's group or (with a
          permissive umask) anyone on the host, for as long as the write and fsync
          take. So the inode is created with `os.open(tmp_path, ..., 0o600)`
          instead: `os.open`'s mode argument is itself masked by umask same as
          `open(2)`'s always is, but umask can only *clear* bits, never add them, so
          asking for `0o600` guarantees the file is owner-only from the instant it
          exists no matter what the host's umask happens to be — there is no
          window to catch it in, because it is never open. Only once the body is
          written and fsynced is the mode changed — to the previous file's mode if
          one existed (this is the read-modify-write path, so almost always), or
          left at `0o600` on the very first write, which is the correct resting
          mode for a file nothing has told to be more open than that and strictly
          narrower than the historical `0o666 & ~umask` default.
        * **the mode (and, where possible, the owner) of the old file survive the
          rename.** Because the rename replaces the old file with the new inode,
          whatever mode the temp file has when it lands becomes `cameras.json`'s
          mode permanently and silently. So the old file's mode is read *before*
          anything is written and stamped onto the temp file *before* the rename,
          the same way `install(1)` and every other tool that has to replace a
          sensitive file in place does it. Ownership is restored on a best-effort
          basis: an unprivileged process (the common case — the engine does not run
          as root) cannot `chown` at all, and failing the edit over that would make
          every edit fail on exactly the deployments most likely to run this way —
          logged at debug when it fails, because a process that runs as a different
          user than the one that owns `cameras.json` will fail this silently on
          *every* edit, quietly handing the file's ownership to itself one rename
          at a time, and that is worth being able to diagnose.
        * **a temp file never survives a failed write.** Anything that goes wrong
          between creating the temp file and completing the rename — a full disk on
          write, a rename that fails because the target is on another filesystem —
          is re-raised after the temp file is removed, not before. Without that, the
          failure path leaves `cameras.json.tmp` behind holding a full copy of the
          document, credentials included, next to a `.json` ignore rule that does
          not cover `.json.tmp` — a second, unprotected copy of the same secret. The
          cleanup unlink is itself best-effort: if it fails (a read-only directory,
          an `EIO`), that failure is swallowed rather than raised in place of the
          real one — an operator staring at an `ENOSPC` from the write needs to see
          `ENOSPC`, not an unrelated `PermissionError` from the cleanup that ran
          while handling it.
        """
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        try:
            previous = self._path.stat()
        except FileNotFoundError:
            # Nothing to preserve: this is the never-existed-before-now case, and
            # `_read_document` above would already have refused an `apply` against a
            # missing file, so in practice this only happens if the file is removed
            # out from under a caller that reaches `_write` directly.
            previous = None
        try:
            body = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            # See the docstring above: created owner-only via `os.open` with an
            # explicit mode, not `Path.open("w")` followed by a chmod, so the
            # credential-bearing content is never sitting at the umask default.
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            mode = stat.S_IMODE(previous.st_mode) if previous is not None else 0o600
            os.chmod(tmp_path, mode)
            if previous is not None:
                # Best-effort: an unprivileged process ordinarily cannot chown to an
                # arbitrary owner, and that must not turn a metadata edit into a
                # failed one — the mode, restored above, is the part that actually
                # gates who can read the credentials.
                try:
                    os.chown(tmp_path, previous.st_uid, previous.st_gid)
                except OSError:
                    logger.debug(
                        "%s: could not restore owner uid=%s gid=%s on the "
                        "replacement file; it will be owned by whichever user this "
                        "process runs as instead, which can lock the original "
                        "owner out of their own cameras.json",
                        tmp_path,
                        previous.st_uid,
                        previous.st_gid,
                        exc_info=True,
                    )
            tmp_path.replace(self._path)  # atomic rename on the same filesystem
        except BaseException:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise

        # The rename above already landed: the document on disk is the new one, and
        # a reader opening `cameras.json` right now sees it. What is not guaranteed
        # on every filesystem without an fsync of the directory *entry* is that it
        # survives a crash the instant after — a durability guarantee about the
        # rename's visibility, not about the data. Everything below, including
        # opening the directory in the first place, is folded into one
        # `except OSError`: `os.open` here can fail too (EMFILE/ENFILE under fd
        # pressure is the realistic case, and it runs after the rename just the same
        # as the fsync does), and a failure to open is no less "already landed" than
        # a failure to fsync. Raising in either case would tell the caller the edit
        # failed when it did not: `update_camera` would refuse to apply it to the
        # running camera, the API would answer 500, and the next restart would
        # silently pick up the very change the operator was told never happened —
        # the file and memory disagreeing being exactly what `docs/operations.md`
        # promises cannot happen. So every failure here is logged, not raised —
        # including `os.close` in the `finally`, which can itself raise `EIO` on
        # some filesystems even after the fd was opened and fsynced successfully.
        try:
            directory = os.open(self._path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            logger.warning(
                "%s: directory entry fsync failed after cameras.json was written; "
                "the edit is committed to disk, but its durability against a crash "
                "right now is not guaranteed",
                self._path.parent,
                exc_info=True,
            )
