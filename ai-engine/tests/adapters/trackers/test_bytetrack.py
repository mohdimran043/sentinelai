"""Tests for ByteTrackTracker (Task 12). Runs on CPU: supervision's ByteTrack
has no GPU/torch dependency, so — unlike the model tasks either side of this
one — these tests belong in CI.

A quirk of the underlying `sv.ByteTrack` (present across the whole
`>=0.24,<0.28` pin, verified empirically, not just at the top of the range):
a brand-new track is only reported starting on the *second* consecutive
frame it is matched on — except tracks first seen on the tracker's very
first ever call, which are reported immediately (there being no prior frame
for a "second consecutive match" to refer to). This is `minimum_consecutive_frames`
(default 1) doing its job of not reporting single-frame noise. Several tests
below thread an extra confirmation frame through for exactly this reason —
see the comment at each one.
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.domain.entities import BBox, Detection
from sentinel_ai.ports.tracker import Tracker


def _det(label: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Detection:
    return Detection(label=label, confidence=conf, box=BBox(x1, y1, x2, y2))


def test_bytetrack_tracker_satisfies_its_port() -> None:
    assert issubclass(ByteTrackTracker, Tracker)


def test_stable_ids_across_frames() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    first = tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    second = tracker.update((_det("person", 1.0, 0.0, 11.0, 10.0),), timestamp=0.1)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_age_frames_increments_once_per_sighting() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    tracker.update((_det("person", 1.0, 0.0, 11.0, 10.0),), timestamp=0.1)
    third = tracker.update((_det("person", 2.0, 0.0, 12.0, 10.0),), timestamp=0.2)
    assert third[0].age_frames == 3


def test_age_counts_sightings_not_elapsed_calls() -> None:
    """Two objects start together (both on the tracker's bootstrap frame, so
    both are reported immediately). One goes briefly undetected — ByteTrack's
    `lost_track_buffer` keeps its identity alive rather than discarding it —
    then reappears. Its age must reflect two *sightings*, not three elapsed
    calls; an adapter that ages every known track_id on every call (instead
    of only the ids ByteTrack actually hands back) would report 3 here.
    """
    tracker = ByteTrackTracker(frame_rate=10)
    near = _det("person", 0.0, 0.0, 10.0, 10.0)
    far = _det("person", 200.0, 0.0, 210.0, 10.0)

    tracker.update((near, far), timestamp=0.0)  # both sighted: age 1 each
    only_near = tracker.update((near,), timestamp=0.1)  # far goes undetected
    assert {t.label for t in only_near} == {"person"}
    assert len(only_near) == 1, "far must not be hallucinated while undetected"

    reappeared = tracker.update((near, far), timestamp=0.2)  # far returns
    far_track = next(t for t in reappeared if t.box.x1 > 100.0)
    assert far_track.age_frames == 2, "far was sighted twice, not three times"


def test_speed_from_known_displacement_and_elapsed_time() -> None:
    tracker = ByteTrackTracker(frame_rate=10)
    tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=0.0)
    # A 2px shift keeps enough IoU overlap with the previous box for ByteTrack
    # to treat this as the *same* track rather than spawning (and provisionally
    # withholding) a new one — see the module docstring.
    first_move = tracker.update((_det("person", 2.0, 0.0, 12.0, 10.0),), timestamp=0.1)
    assert first_move[0].speed_px_s == pytest.approx(20.0, rel=0.05)

    # A second, differently-sized displacement: a hardcoded 20.0 (or any other
    # fixed constant) would fail this second assertion.
    second_move = tracker.update((_det("person", 2.0, 5.0, 12.0, 15.0),), timestamp=0.2)
    assert second_move[0].speed_px_s == pytest.approx(50.0, rel=0.05)


def test_reset_clears_state_and_a_re_sighted_object_gets_a_fresh_id() -> None:
    """Re-sight `far` after reset — an id or age that merely happens to look
    right is not enough (the Phase 1A review found exactly this gap in
    FakeTracker's original reset test)."""
    tracker = ByteTrackTracker(frame_rate=10)
    near = _det("person", 0.0, 0.0, 10.0, 10.0)
    far = _det("person", 500.0, 500.0, 510.0, 510.0)

    tracker.update((near,), timestamp=0.0)  # bootstrap frame: near -> id 1
    # far is new on this (non-bootstrap) frame, so ByteTrack withholds it for
    # one confirmation frame — see the module docstring.
    tracker.update((near, far), timestamp=0.1)
    before = tracker.update((near, far), timestamp=0.2)
    far_before = next(t for t in before if t.box.x1 > 100.0)
    assert far_before.track_id != 1, "the id counter has advanced past its starting value"

    tracker.reset()
    after = tracker.update((far,), timestamp=1.0)

    assert after[0].track_id != far_before.track_id, "the association must be forgotten"
    assert after[0].track_id == 1, "the id counter must be rewound"
    assert after[0].age_frames == 1, "a re-sighted object is new, not aged"


def test_a_non_advancing_timestamp_yields_zero_speed_rather_than_dividing_by_zero() -> None:
    """A repeated or regressing timestamp must not raise.

    Duplicate timestamps are reachable in practice: a source can emit two packets
    with the same pts, and an RTSP reconnect can hand the pipeline a timestamp
    behind the previous one. Without the `elapsed > 0` guard this is a
    ZeroDivisionError (equal) or a negative speed (regressing) — and a negative
    speed silently disarms the SpeedAnomaly trigger, which only ever compares
    upward against a threshold.
    """
    tracker = ByteTrackTracker(frame_rate=10)
    tracker.update((_det("person", 0.0, 0.0, 10.0, 10.0),), timestamp=1.0)

    same_ts = tracker.update((_det("person", 2.0, 0.0, 12.0, 10.0),), timestamp=1.0)
    assert same_ts[0].speed_px_s == 0.0

    earlier_ts = tracker.update((_det("person", 4.0, 0.0, 14.0, 10.0),), timestamp=0.5)
    assert earlier_ts[0].speed_px_s == 0.0
    assert earlier_ts[0].speed_px_s >= 0.0, "a negative speed would disarm SpeedAnomaly"
