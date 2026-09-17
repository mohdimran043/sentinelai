"""What a camera's enabled AI capabilities imply, and what they must never imply.

The load-bearing property here is the one §13 asks for and that nothing else in the
engine can provide: a capability nobody enabled must not cause a model to be loaded.
`required_roles` is the pure function the composition root asks before it builds
anything, so these tests are what stop a disabled VLM from still costing 2.7 GiB.
"""

from __future__ import annotations

import pytest

from sentinel_ai.domain.capabilities import (
    DEFAULT_CAPABILITIES,
    CameraCapabilities,
    Capability,
    ModelRole,
    required_roles,
)


def test_every_capability_declares_the_roles_it_needs() -> None:
    """No member may be added without saying what it costs.

    A capability missing from the role table would silently require nothing, so a
    camera that enabled it would run with no model behind it — the exact shape of
    §40's "UI claims a feature the backend does not have", introduced one enum
    member at a time.
    """
    for capability in Capability:
        # Raises KeyError if the member has no entry, which is the failure we want.
        roles = required_roles(CameraCapabilities.of(capability))
        assert isinstance(roles, frozenset)


def test_no_capabilities_requires_no_models() -> None:
    assert required_roles(CameraCapabilities.none()) == frozenset()


def test_scene_description_requires_the_vlm() -> None:
    caps = CameraCapabilities.of(Capability.SCENE_DESCRIPTION)
    assert ModelRole.VLM in required_roles(caps)


def test_anomaly_detection_does_not_require_the_vlm() -> None:
    """The six automatic triggers are pure predicates over detector output.

    They decide *whether to look*, which is upstream of looking. A deployment that
    wants cheap anomaly flagging without a 3B model paying for every escalation is a
    real configuration, and this is what makes it one.
    """
    caps = CameraCapabilities.of(Capability.ANOMALY_DETECTION)
    assert required_roles(caps) == frozenset({ModelRole.DETECTOR})


def test_roles_are_the_union_over_enabled_capabilities() -> None:
    caps = CameraCapabilities.of(Capability.SCENE_DESCRIPTION, Capability.ANOMALY_DETECTION)
    assert required_roles(caps) == frozenset({ModelRole.DETECTOR, ModelRole.VLM})


def test_roles_across_cameras_is_the_union_not_the_intersection() -> None:
    """One camera wanting the VLM is enough to make the process load it.

    The engine holds one set of weights for every camera (ADR 4), so the question
    the composition root asks is "does *anybody* need this", never "does everybody".
    An intersection here would leave a camera silently unserved.
    """
    quiet = CameraCapabilities.of(Capability.ANOMALY_DETECTION)
    loud = CameraCapabilities.of(Capability.SCENE_DESCRIPTION)
    assert required_roles(quiet, loud) == frozenset({ModelRole.DETECTOR, ModelRole.VLM})


def test_roles_over_no_cameras_at_all_is_empty() -> None:
    """An engine with zero cameras configured loads zero models."""
    assert required_roles() == frozenset()


def test_default_capabilities_preserve_todays_behaviour() -> None:
    """A camera file written before capabilities existed must keep working.

    Today every camera describes scenes and runs the automatic triggers, so that is
    what an unspecified `capabilities` has to mean. Anything narrower would silently
    turn off monitoring on an existing deployment at upgrade time.
    """
    assert DEFAULT_CAPABILITIES.enabled(Capability.SCENE_DESCRIPTION)
    assert DEFAULT_CAPABILITIES.enabled(Capability.ANOMALY_DETECTION)


def test_capabilities_are_order_insensitive_and_deduplicating() -> None:
    a = CameraCapabilities.of(Capability.SCENE_DESCRIPTION, Capability.ANOMALY_DETECTION)
    b = CameraCapabilities.of(
        Capability.ANOMALY_DETECTION,
        Capability.SCENE_DESCRIPTION,
        Capability.SCENE_DESCRIPTION,
    )
    assert a == b


def test_enabled_is_false_for_a_capability_not_granted() -> None:
    caps = CameraCapabilities.of(Capability.ANOMALY_DETECTION)
    assert not caps.enabled(Capability.SCENE_DESCRIPTION)


def test_capabilities_are_hashable_so_they_can_key_a_registry() -> None:
    caps = CameraCapabilities.of(Capability.ANOMALY_DETECTION)
    assert len({caps, CameraCapabilities.of(Capability.ANOMALY_DETECTION)}) == 1


def test_from_names_accepts_the_wire_vocabulary() -> None:
    caps = CameraCapabilities.from_names(["scene_description", "anomaly_detection"])
    assert caps == CameraCapabilities.of(Capability.SCENE_DESCRIPTION, Capability.ANOMALY_DETECTION)


def test_from_names_rejects_an_unknown_capability_by_name() -> None:
    """Present-but-unknown is a configuration error, exactly as an unknown `zone` is.

    Silently dropping it would give an operator a camera they believe is running a
    capability that was never enabled — the failure mode `domain/zone.py` already
    argues about at length, and it is worse here because the field turns monitoring
    on rather than merely grouping it.
    """
    with pytest.raises(ValueError, match="unknown capability"):
        CameraCapabilities.from_names(["scene_description", "xray_vision"])


def test_from_names_error_names_the_valid_vocabulary() -> None:
    with pytest.raises(ValueError) as excinfo:
        CameraCapabilities.from_names(["nope"])
    assert "scene_description" in str(excinfo.value)


def test_from_names_of_an_empty_list_is_no_capabilities() -> None:
    """`[]` means "watch this camera, run nothing on it" — a real, storable choice.

    Distinct from an absent field, which means the default. Same absent-vs-empty
    distinction `notify_on` carries, and it must survive the same round trip.
    """
    assert CameraCapabilities.from_names([]) == CameraCapabilities.none()


def test_names_round_trips_through_from_names() -> None:
    caps = CameraCapabilities.of(Capability.SCENE_DESCRIPTION, Capability.ANOMALY_DETECTION)
    assert CameraCapabilities.from_names(caps.names()) == caps


def test_names_is_sorted_so_a_written_file_is_stable() -> None:
    """`CameraFileStore` rewrites this document; an unstable order would churn the
    diff on every unrelated edit."""
    caps = CameraCapabilities.of(Capability.SCENE_DESCRIPTION, Capability.ANOMALY_DETECTION)
    assert caps.names() == sorted(caps.names())
