"""Frontality estimation: the pure half of the face adapter.

Needs no model and no ONNX Runtime, which is the point — this is the part most able to
be quietly wrong, and a wrong frontality either lets unreadable faces into the matcher
or excludes readable ones.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from sentinel_ai.adapters.face.insightface_pipeline import estimate_frontality


def landmarks(*, left_eye_x: float, right_eye_x: float, nose_x: float) -> npt.NDArray[np.float32]:
    """SCRFD's five points: left eye, right eye, nose, left mouth, right mouth.

    Only the x coordinates matter to this estimate; the rest are filled plausibly so
    the shape is the one the adapter actually receives.
    """
    return np.array(
        [
            [left_eye_x, 100.0],
            [right_eye_x, 100.0],
            [nose_x, 130.0],
            [left_eye_x, 160.0],
            [right_eye_x, 160.0],
        ],
        dtype=np.float32,
    )


def test_a_face_on_subject_scores_one() -> None:
    """Nose exactly between the eyes."""
    assert estimate_frontality(
        landmarks(left_eye_x=80.0, right_eye_x=120.0, nose_x=100.0)
    ) == pytest.approx(1.0)


def test_a_slightly_turned_head_scores_below_one_but_is_still_usable() -> None:
    score = estimate_frontality(landmarks(left_eye_x=80.0, right_eye_x=120.0, nose_x=105.0))
    assert 0.5 < score < 1.0


def test_a_head_turned_far_enough_scores_zero() -> None:
    """A nose halfway to one eye is a head turned far enough that the embedding is not
    worth trusting — which is why the measure is doubled rather than linear."""
    assert estimate_frontality(
        landmarks(left_eye_x=80.0, right_eye_x=120.0, nose_x=120.0)
    ) == pytest.approx(0.0)


def test_the_measure_is_symmetric() -> None:
    """Turned left and turned right are the same amount of turned."""
    left = estimate_frontality(landmarks(left_eye_x=80.0, right_eye_x=120.0, nose_x=92.0))
    right = estimate_frontality(landmarks(left_eye_x=80.0, right_eye_x=120.0, nose_x=108.0))
    assert left == pytest.approx(right)


def test_the_measure_is_scale_invariant() -> None:
    """A face near the camera and the same face far away are equally face-on. Measuring
    the offset in pixels rather than as a fraction of eye separation would make
    distance look like pose."""
    near = estimate_frontality(landmarks(left_eye_x=0.0, right_eye_x=200.0, nose_x=125.0))
    far = estimate_frontality(landmarks(left_eye_x=0.0, right_eye_x=20.0, nose_x=12.5))
    assert near == pytest.approx(far)


def test_missing_landmarks_score_zero_rather_than_one() -> None:
    """An unknown pose must fail a frontality floor, not sail through it. Defaulting to
    1.0 would make every face the detector could not landmark into a 'perfectly
    face-on' one."""
    assert estimate_frontality(None) == 0.0
    assert estimate_frontality(np.zeros((2, 2), dtype=np.float32)) == 0.0


def test_a_profile_view_with_coincident_eyes_scores_zero() -> None:
    """Both eyes at the same x is a face seen edge-on, or a degenerate detection.
    Either way there is nothing usable, and the division it would otherwise cause is
    guarded rather than allowed to raise inside a frame loop."""
    assert estimate_frontality(landmarks(left_eye_x=100.0, right_eye_x=100.0, nose_x=100.0)) == 0.0
