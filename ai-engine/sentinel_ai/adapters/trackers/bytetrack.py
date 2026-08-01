"""Supervision ByteTrack tracker (spec §5; ports/tracker.py).

Runs entirely on CPU — supervision's ByteTrack is a numpy/scipy association
algorithm with no torch dependency — which is why, unlike the model adapters
either side of it in the task list, its tests belong in CI.

ByteTrack hands back `tracker_id`, `xyxy`, `confidence`, `class_id` per call;
it does not hand back how long a track has existed or how fast it is moving,
both of which the escalation gate needs (`min_track_frames`,
`speed_fallback_px_s`). This adapter maintains both itself, keyed by
`tracker_id`.
"""

from __future__ import annotations

from math import hypot

import numpy as np
import supervision as sv

from sentinel_ai.domain.entities import BBox, Detection, Track
from sentinel_ai.ports.tracker import Tracker


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: int = 30) -> None:
        self._tracker = sv.ByteTrack(frame_rate=frame_rate)
        self._age: dict[int, int] = {}
        self._last_position: dict[int, tuple[BBox, float]] = {}
        self._label_ids: dict[str, int] = {}
        self._id_labels: dict[int, str] = {}

    def _class_id_for(self, label: str) -> int:
        if label not in self._label_ids:
            new_id = len(self._label_ids)
            self._label_ids[label] = new_id
            self._id_labels[new_id] = label
        return self._label_ids[label]

    def _to_sv_detections(self, detections: tuple[Detection, ...]) -> sv.Detections:
        if not detections:
            return sv.Detections.empty()
        xyxy = np.array(
            [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections], dtype=np.float32
        )
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([self._class_id_for(d.label) for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: tuple[Detection, ...], timestamp: float) -> tuple[Track, ...]:
        sv_detections = self._to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_detections)
        if len(tracked) == 0:
            return ()
        # `update_with_detections` only ever returns detections it has assigned
        # a tracker_id and class_id to (see `_to_sv_detections` — every
        # detection this adapter constructs has a class_id); both are
        # `ndarray | None` in the stubs only to cover the *unfiltered*
        # `Detections` shape.
        assert tracked.tracker_id is not None
        assert tracked.class_id is not None

        results: list[Track] = []
        for xyxy, tracker_id, class_id in zip(
            tracked.xyxy, tracked.tracker_id, tracked.class_id, strict=True
        ):
            track_id = int(tracker_id)
            label = self._id_labels[int(class_id)]
            box = BBox(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]))

            age = self._age.get(track_id, 0) + 1
            self._age[track_id] = age

            previous = self._last_position.get(track_id)
            if previous is not None:
                prev_box, prev_ts = previous
                elapsed = timestamp - prev_ts
                speed = (
                    hypot(box.cx - prev_box.cx, box.cy - prev_box.cy) / elapsed
                    if elapsed > 0
                    else 0.0
                )
            else:
                speed = 0.0
            self._last_position[track_id] = (box, timestamp)

            results.append(
                Track(track_id=track_id, label=label, box=box, age_frames=age, speed_px_s=speed)
            )
        return tuple(results)

    def reset(self) -> None:
        self._tracker.reset()
        self._age.clear()
        self._last_position.clear()
        self._label_ids.clear()
        self._id_labels.clear()
