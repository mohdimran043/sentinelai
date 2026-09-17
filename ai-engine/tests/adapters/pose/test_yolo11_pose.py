"""The pure half of the pose adapter: attribution and keypoint mapping.

Both are reachable without weights, a GPU or `ultralytics`, and both are the parts
most able to be quietly wrong. Attribution especially: a skeleton bound to the wrong
track does not add noise, it makes one person's posture read as another's for as long
as the confusion lasts — which is exactly how `domain/behaviour/fall.py` would report
a fall that did not happen.

The forward pass itself is a `gpu`-marked test elsewhere; there is nothing to learn
from mocking `YOLO.predict` and asserting it was called.
"""

from __future__ import annotations

from sentinel_ai.adapters.pose.yolo11_pose import (
    COCO_KEYPOINT_ORDER,
    attribute_poses,
    keypoints_from_row,
)
from sentinel_ai.domain.behaviour.observation import Keypoint, KeypointName
from sentinel_ai.domain.entities import BBox, Track


def track(track_id: int, box: BBox) -> Track:
    return Track(track_id=track_id, label="person", box=box, age_frames=5, speed_px_s=0.0)


def skeleton(x: float) -> dict[KeypointName, Keypoint]:
    """A minimal distinguishable skeleton, tagged by its x so a test can tell two apart."""
    return {KeypointName.LEFT_SHOULDER: Keypoint(x=x, y=0.0, confidence=0.9)}


class TestAttribution:
    def test_an_overlapping_pose_is_bound_to_its_track(self) -> None:
        box = BBox(0.0, 0.0, 100.0, 200.0)
        result = attribute_poses((track(7, box),), ((box, skeleton(1.0)),))
        assert set(result) == {7}
        assert result[7].track_id == 7

    def test_a_pose_nowhere_near_a_track_is_dropped_rather_than_guessed(self) -> None:
        """Partial results are the documented normal. A best-effort match on a
        non-overlapping box would be inventing the association."""
        result = attribute_poses(
            (track(1, BBox(0.0, 0.0, 100.0, 200.0)),),
            ((BBox(500.0, 500.0, 600.0, 700.0), skeleton(1.0)),),
        )
        assert result == {}

    def test_a_partially_overlapping_pose_below_the_floor_is_dropped(self) -> None:
        # Half-overlapping boxes give IoU 1/3, under the 0.45 floor.
        result = attribute_poses(
            (track(1, BBox(0.0, 0.0, 100.0, 200.0)),),
            ((BBox(50.0, 0.0, 150.0, 200.0), skeleton(1.0)),),
        )
        assert result == {}

    def test_two_people_get_their_own_skeletons_not_the_same_one(self) -> None:
        """The crowd case, and the whole reason attribution is greedy-by-overlap with
        both sides consumed. Sharing one skeleton between two adjacent people is how a
        fall gets read off the wrong body."""
        left = BBox(0.0, 0.0, 100.0, 200.0)
        right = BBox(300.0, 0.0, 400.0, 200.0)
        result = attribute_poses(
            (track(1, left), track(2, right)),
            ((right, skeleton(9.0)), (left, skeleton(1.0))),
        )
        assert result[1].keypoints[KeypointName.LEFT_SHOULDER].x == 1.0
        assert result[2].keypoints[KeypointName.LEFT_SHOULDER].x == 9.0

    def test_one_pose_among_two_overlapping_tracks_goes_to_the_better_match(self) -> None:
        exact = BBox(0.0, 0.0, 100.0, 200.0)
        looser = BBox(10.0, 10.0, 105.0, 205.0)
        result = attribute_poses((track(1, looser), track(2, exact)), ((exact, skeleton(1.0)),))
        assert set(result) == {2}

    def test_a_track_with_no_pose_is_simply_absent(self) -> None:
        result = attribute_poses(
            (track(1, BBox(0.0, 0.0, 100.0, 200.0)), track(2, BBox(300.0, 0.0, 400.0, 200.0))),
            ((BBox(0.0, 0.0, 100.0, 200.0), skeleton(1.0)),),
        )
        assert set(result) == {1}

    def test_no_tracks_or_no_poses_yields_nothing(self) -> None:
        box = BBox(0.0, 0.0, 100.0, 200.0)
        assert attribute_poses((), ((box, skeleton(1.0)),)) == {}
        assert attribute_poses((track(1, box),), ()) == {}

    def test_attribution_is_deterministic_across_detection_orderings(self) -> None:
        """A model may return its detections in a different order run to run. Two
        equally-good matches resolving differently would make a fall episode's pose
        source flicker frame to frame."""
        a = BBox(0.0, 0.0, 100.0, 200.0)
        b = BBox(0.0, 0.0, 100.0, 200.0)
        tracks = (track(1, a), track(2, b))
        forward = attribute_poses(tracks, ((a, skeleton(1.0)), (b, skeleton(2.0))))
        backward = attribute_poses(tracks, ((b, skeleton(2.0)), (a, skeleton(1.0))))
        assert set(forward) == set(backward)


class TestKeypointMapping:
    def test_the_named_joints_are_read_from_their_coco_positions(self) -> None:
        xy = [[float(i), float(i * 2)] for i in range(17)]
        conf = [0.9] * 17
        keypoints = keypoints_from_row(xy, conf)

        left_shoulder_index = COCO_KEYPOINT_ORDER.index("left_shoulder")
        assert keypoints[KeypointName.LEFT_SHOULDER].x == float(left_shoulder_index)
        right_hip_index = COCO_KEYPOINT_ORDER.index("right_hip")
        assert keypoints[KeypointName.RIGHT_HIP].y == float(right_hip_index * 2)

    def test_only_the_joints_anything_reads_are_kept(self) -> None:
        """A skeleton carrying all seventeen would be thirteen joints per person per
        frame that no consumer has a test for."""
        keypoints = keypoints_from_row([[1.0, 1.0]] * 17, [0.9] * 17)
        assert set(keypoints) == set(KeypointName)

    def test_a_short_row_yields_what_it_has_rather_than_raising(self) -> None:
        """A model variant emitting fewer joints must not raise inside a frame loop
        that has a working geometry fallback."""
        keypoints = keypoints_from_row([[1.0, 1.0]] * 7, [0.9] * 7)
        assert KeypointName.LEFT_SHOULDER in keypoints
        assert KeypointName.LEFT_HIP not in keypoints

    def test_a_missing_confidence_reads_as_zero_not_as_certain(self) -> None:
        """The dangerous default. A checkpoint exported without keypoint scores that
        was read as 1.0 would clear `min_keypoint_confidence` on every guessed joint,
        and a hallucinated hip would then decide a person's posture."""
        keypoints = keypoints_from_row([[1.0, 1.0]] * 17, [])
        assert keypoints[KeypointName.LEFT_SHOULDER].confidence == 0.0

    def test_the_coco_order_is_the_full_seventeen(self) -> None:
        """Pinned because every positional index in this codebase lives in that tuple
        and nowhere else; a member dropped from it silently shifts every joint after
        it onto the wrong name."""
        assert len(COCO_KEYPOINT_ORDER) == 17
        assert COCO_KEYPOINT_ORDER[0] == "nose"
        assert COCO_KEYPOINT_ORDER[-1] == "right_ankle"
        assert {name.value for name in KeypointName} <= set(COCO_KEYPOINT_ORDER)
