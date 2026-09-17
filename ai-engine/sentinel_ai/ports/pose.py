"""Human pose estimation port (spec §7, §31).

The ninth seam, and the first one that is genuinely **optional at runtime**. Every
other port in this package is either always present (`FrameSource`, `ObjectDetector`)
or has a default that is always built (`Notifier`). A pose model is loaded only where
some camera enabled a capability that asks for it (`domain/capabilities.py`), because
it is a second forward pass over every person in every sampled frame — the difference
between running thirty cameras and running eight.

That optionality is a contract on the *consumers*, not only on the composition root:
`domain/behaviour/fall.py` reads a torso angle when one is available and bounding-box
geometry when one is not, and records which it used. A behaviour detector that
required pose would turn a camera without it into a capability that looks enabled and
detects nothing, which is what §40 forbids.

Why `estimate` takes tracks rather than a frame alone
-----------------------------------------------------
The pipeline has already detected and associated people by the time pose runs, and
that association is the only reason a skeleton is useful over time: a fall is a
statement about one person across several seconds, so a keypoint set that could not be
attributed to a track could not participate in one. Passing the tracks in also lets an
adapter crop to them (top-down pose) or simply attribute whole-image results back to
them (bottom-up), without the port choosing between those two model families.

`async`, like `ObjectDetector` and unlike `Tracker`: this is a GPU forward pass hiding
a blocking C-extension call, so the adapter offloads it and the loop stays free. See
`ports/tracker.py` for the other half of that argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel_ai.domain.behaviour.observation import PersonPose
from sentinel_ai.domain.entities import Track
from sentinel_ai.ports.frame_source import FrameData


class PoseEstimator(ABC):
    @abstractmethod
    async def estimate(self, frame: FrameData, tracks: tuple[Track, ...]) -> dict[int, PersonPose]:
        """Skeletons for the people in `tracks`, keyed by `Track.track_id`.

        **Partial results are normal and must not raise.** A person too small, too
        occluded or too close to the frame edge for the model to place joints on
        simply does not appear in the mapping, and a consumer asking for a track that
        is absent gets `None` (`BehaviourObservation.pose_for`). Returning a skeleton
        of zero-confidence keypoints instead would push the "is this usable" decision
        into every caller, and the callers do not have the model's own scores to make
        it with.

        Implementations map their model's own keypoint ordering onto `KeypointName`
        at this boundary, once. Nothing downstream indexes keypoints positionally, so
        substituting a model with a different skeleton order is an adapter change and
        touches no detector — which is the entire point of the seam.

        A track whose `label` is not a person is not the caller's to filter: an
        implementation may be handed every track and returns entries only for the ones
        it could place a human skeleton on.
        """
