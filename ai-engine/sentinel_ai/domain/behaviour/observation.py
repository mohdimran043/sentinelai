"""What a behaviour detector is allowed to look at.

`SceneState` is what the escalation gate sees — detections, tracks, motion energy, a
scene signature, and deliberately **no pixels** (see `domain/entities.py`). Behaviour
detectors see the same thing plus one optional extra: pose keypoints, for the cameras
that enabled a capability needing them.

Optional is the load-bearing word. A pose model is a second forward pass per frame, so
it is only run where a capability asks for it (`domain/capabilities.py`). Every
detector here therefore has to work from geometry alone and merely *improve* when pose
is present — never require it. A detector that silently did nothing without pose would
be a capability that appears enabled and detects nothing, which is exactly what §40
forbids.

Pure, like the rest of `domain/`: no numpy (keypoints are plain floats), no clock, no
I/O. Time arrives as `SceneState.timestamp`, on the camera's own source timeline, the
same way the gate receives it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import atan2, degrees, hypot

from sentinel_ai.domain.entities import SceneState

__all__ = [
    "BehaviourObservation",
    "Keypoint",
    "KeypointName",
    "PersonPose",
]


class KeypointName(StrEnum):
    """The COCO-17 skeleton, which is what every pose model this project would
    plausibly use emits (YOLO-pose, RTMPose, MoveNet, OpenPose's COCO output).

    A `StrEnum` rather than positional indices because a detector reading
    `keypoints[5]` is a detector that breaks silently the day a model with a different
    ordering is substituted — and the point of `ports/` is that models are
    substitutable. The adapter maps its own model's ordering onto these names once, at
    the boundary, where the mapping is visible and testable.

    Only the members this codebase actually reads are listed. The full seventeen are
    not here for completeness' sake: an unused member is a name nothing pins, and the
    first detector to reach for it would have no test telling it whether the adapter
    ever populates it.
    """

    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"


@dataclass(frozen=True, slots=True)
class Keypoint:
    """One joint, in the same pixel coordinates as `BBox`.

    `confidence` is the model's own per-joint score. It matters more here than a
    detection score does: a pose model asked about a partly-occluded person returns a
    full skeleton regardless, with low confidence on the joints it guessed. A detector
    that ignored the score would read a hallucinated hip as a real one and call a
    standing person horizontal.
    """

    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonPose:
    """The skeleton for one tracked person.

    Keyed by `track_id`, not by detection index, so a pose survives the same
    association the rest of the pipeline uses. A pose whose track the tracker dropped
    simply does not appear.

    `keypoints` is a mapping rather than a tuple because it is routinely partial: a
    model may emit nothing usable for an occluded joint, and callers must be able to
    ask "is there a left hip" without a sentinel value that could be mistaken for a
    coordinate.
    """

    track_id: int
    keypoints: Mapping[KeypointName, Keypoint] = field(default_factory=dict)

    def _confident(self, name: KeypointName, minimum: float) -> Keypoint | None:
        point = self.keypoints.get(name)
        return point if point is not None and point.confidence >= minimum else None

    def torso_angle_degrees(self, *, min_keypoint_confidence: float) -> float | None:
        """Angle of the shoulders→hips axis away from vertical, in degrees, or `None`.

        0 is upright, 90 is horizontal. Unsigned: a person lying head-left and one
        lying head-right are the same posture, and distinguishing them would only
        invite a detector to treat one of them as upright.

        **The single most useful signal a pose model adds**, and the reason this
        method exists rather than callers reading joints directly. Bounding-box aspect
        ratio — the geometry-only fallback — is confounded by everything: a person
        crouching, a person carrying something wide, a tracker box that grew to
        include a chair. The torso axis is confounded by much less.

        `None` when the skeleton cannot support the answer: fewer than one shoulder
        and one hip above `min_keypoint_confidence`, or a torso of zero length (both
        ends resolving to the same point, which a model sometimes emits for a person
        seen exactly end-on). Returning `None` rather than a plausible number is the
        whole contract — every caller then falls back to geometry, and a guessed
        angle would be worse than no angle because it would silently outrank the
        fallback it replaced.

        Midpoints are taken over whichever side is confident, so a person half out of
        frame still yields an angle from one shoulder and one hip.
        """
        shoulders = [
            point
            for point in (
                self._confident(KeypointName.LEFT_SHOULDER, min_keypoint_confidence),
                self._confident(KeypointName.RIGHT_SHOULDER, min_keypoint_confidence),
            )
            if point is not None
        ]
        hips = [
            point
            for point in (
                self._confident(KeypointName.LEFT_HIP, min_keypoint_confidence),
                self._confident(KeypointName.RIGHT_HIP, min_keypoint_confidence),
            )
            if point is not None
        ]
        if not shoulders or not hips:
            return None

        shoulder_x = sum(point.x for point in shoulders) / len(shoulders)
        shoulder_y = sum(point.y for point in shoulders) / len(shoulders)
        hip_x = sum(point.x for point in hips) / len(hips)
        hip_y = sum(point.y for point in hips) / len(hips)

        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y
        if hypot(dx, dy) == 0.0:
            return None

        # `abs` on both components folds the four quadrants into one: the torso axis
        # is a line, not a direction, so an upside-down person is still upright-ish
        # and a person lying either way round is horizontal. Using atan2 on signed
        # values would make "head at the top" and "head at the bottom" differ by 180
        # degrees for no behavioural reason.
        return degrees(atan2(abs(dx), abs(dy)))


@dataclass(frozen=True, slots=True)
class BehaviourObservation:
    """One frame, as a behaviour detector sees it.

    Deliberately a wrapper around `SceneState` rather than extra fields on it: the
    escalation gate must keep seeing exactly what it sees today, and adding an
    optional `poses` to `SceneState` would put a field on the gate's input that the
    gate must never read. Keeping the two types apart is what makes that a compile-
    time fact rather than a convention.
    """

    scene: SceneState
    poses: Mapping[int, PersonPose] = field(default_factory=dict)
    """Pose by `track_id`. Empty whenever no pose model ran for this camera, which is
    the common case and must stay a supported one."""

    frame_width: int = 0
    frame_height: int = 0
    """The decoded frame's dimensions, needed to turn a normalised zone or line into
    pixels (`domain/behaviour/zones.py`).

    Here rather than on `SceneState` deliberately, for the reason this whole type is a
    wrapper: the escalation gate must keep seeing exactly what it sees today, and a
    field on its input that it must never read is a field somebody eventually reads.

    Defaulted to 0 so every existing construction site keeps working, and 0 is treated
    by `zones.py` as "cannot place a zone" rather than as a degenerate frame — a
    detector that silently mapped every zone onto a single pixel would report
    intrusions nobody could explain.
    """

    @property
    def timestamp(self) -> float:
        """The camera's own source timeline, as `SceneState` carries it. No clock is
        read here or anywhere else in `domain/`."""
        return self.scene.timestamp

    def pose_for(self, track_id: int) -> PersonPose | None:
        return self.poses.get(track_id)
