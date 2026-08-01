"""Where a camera is — the one field cameras are grouped by.

The recorder next door carries two fields on every camera: `mode`
(`room` / `common_area`) and `space_type` (`room`, `corridor`, `dayroom`). Across all
four of its real cameras the first is a function of the second — `room` -> `room`,
`corridor` -> `common_area`, `dayroom` -> `common_area` — so the pair is one fact
written down twice.

**One stored field, constrained; the coarse grouping derived.** Storing both would
let a camera file say `mode: room, space_type: corridor` and nothing in the process
could say which half was the lie; a console grouping by one and colouring by the other
would then disagree with itself. So `Zone` is the stored field and takes the finer
vocabulary (it is the one that cannot be reconstructed from the other), and `ZoneKind`
is computed from it. A config can be wrong about a camera's zone, but it can no longer
be *inconsistent* about it, and there is no migration in which the two drift apart.

**Constrained rather than free-form**, for the same reason `CameraProfile` validates
its field names: grouping is only useful if two operators typing the same idea produce
the same group. Free-form text gives `Corridor`, `corridor `, `hallway` and `Corridor
2` as four zones with one meaning, and a console cannot render a semantics it cannot
predict — `common_area` is the distinction that actually changes how a scene should be
read (a person alone in a room at 03:00 is not the story a person alone in a corridor
at 03:00 is). A camera whose real zone is not in this vocabulary is a request to widen
the vocabulary here, where the derivation is updated in the same commit, rather than
a string nothing downstream understands.

`None` is not a member of this enum and is not modelled here: an *ungrouped* camera is
the loader's concern (`sentinel_ai.main.load_cameras`), and it is a legitimate state —
a camera file written before zones existed keeps loading. An `UNKNOWN` member would
force every consumer to decide whether it is a group; absence says plainly that it is
not.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Zone", "ZoneKind"]


class ZoneKind(StrEnum):
    """The coarse privacy/behaviour split a `Zone` falls into.

    Derived from `Zone`, never stored: see this module's docstring. `StrEnum` so it
    serialises as `"room"` / `"common_area"` with no encoder of its own.
    """

    ROOM = "room"
    COMMON_AREA = "common_area"


class Zone(StrEnum):
    """The space a camera watches. The recorder's `space_type` vocabulary, exactly.

    Deliberately the recorder's four-camera vocabulary rather than a new one: the two
    systems watch the same building, and a console that shows both must not have to
    translate. Extending it is a one-line addition here plus its entry in `_KINDS`,
    which `tests/domain/test_zone.py` requires for every member.
    """

    ROOM = "room"
    CORRIDOR = "corridor"
    DAYROOM = "dayroom"

    @property
    def kind(self) -> ZoneKind:
        """The coarse grouping. Total by construction — a member with no entry in
        `_KINDS` raises here rather than defaulting to a plausible answer, and the
        exhaustiveness test makes that unreachable."""
        return _KINDS[self]


_KINDS: dict[Zone, ZoneKind] = {
    Zone.ROOM: ZoneKind.ROOM,
    Zone.CORRIDOR: ZoneKind.COMMON_AREA,
    Zone.DAYROOM: ZoneKind.COMMON_AREA,
}
"""`Zone` -> `ZoneKind`, the recorder's own mapping. A dict rather than a `match`
so the exhaustiveness test can compare its keys against `Zone` directly."""
