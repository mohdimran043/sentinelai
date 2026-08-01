"""The camera zone (T1): one stored field, a derived coarse grouping."""

from __future__ import annotations

import json

import pytest

from sentinel_ai.domain.zone import Zone, ZoneKind


def test_the_vocabulary_is_the_recorders_space_types() -> None:
    """The recorder next door models the same building with these four cameras:
    room_4b/room, corridor_1/corridor, room_2a/room, dayroom_1/dayroom. A console
    showing both systems must not have to translate between two vocabularies."""
    assert {zone.value for zone in Zone} == {"room", "corridor", "dayroom"}


@pytest.mark.parametrize(
    ("zone", "expected"),
    [
        (Zone.ROOM, ZoneKind.ROOM),
        (Zone.CORRIDOR, ZoneKind.COMMON_AREA),
        (Zone.DAYROOM, ZoneKind.COMMON_AREA),
    ],
)
def test_the_coarse_grouping_matches_the_recorders_mode(zone: Zone, expected: ZoneKind) -> None:
    """`mode` on the recorder's cameras is exactly this function of `space_type`;
    deriving it is what stops the two from ever disagreeing here."""
    assert zone.kind is expected


def test_every_zone_has_a_kind() -> None:
    """The derivation is total. A zone added without its grouping would otherwise
    raise a KeyError at request time, on the one endpoint a console groups by."""
    for zone in Zone:
        assert isinstance(zone.kind, ZoneKind)


def test_an_unknown_zone_is_rejected_rather_than_becoming_its_own_group() -> None:
    """Free-form text would make `hallway`, `Corridor` and `corridor ` three groups
    with one meaning, and no console can render a semantics it cannot predict."""
    with pytest.raises(ValueError, match="hallway"):
        Zone("hallway")


def test_a_zone_serialises_as_its_own_string() -> None:
    """`StrEnum`, so the wire value needs no encoder and cannot drift from the member
    name it was built from."""
    assert str(Zone.DAYROOM) == "dayroom"
    assert Zone.DAYROOM.value == "dayroom"
    assert json.dumps({"zone": Zone.DAYROOM}) == '{"zone": "dayroom"}'
    assert str(ZoneKind.COMMON_AREA) == "common_area"
