"""Alert aggregation: §17's requirement, as executable prose.

The specification states it plainly — the same person detected repeatedly for twenty
seconds must produce one alert with an occurrence count, not a hundred alerts. That is
the first test below, written as close to the spec's own worked example as a test can
get, because it is the requirement everything else here exists to serve.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from sentinel_ai.domain.alert import MAX_EVENT_IDS, Alert, AlertKey, AlertState
from sentinel_ai.domain.entities import EscalationReason, Event, Severity, ThreatScore
from sentinel_ai.domain.policy.alerting import (
    DEFAULT_MERGE_WINDOW_SECONDS,
    alert_severity,
    is_alertable,
    merged,
    opened,
    should_merge,
)
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.zone import Zone


def event(
    *,
    reason: EscalationReason = EscalationReason.ZONE_INTRUSION,
    threat: float = 0.5,
    track_ids: tuple[int, ...] = (7,),
    occurred_at: float = 0.0,
    camera_id: str = "cam-1",
    description: str = "a person is in the stairwell",
    clip_uri: str | None = None,
    event_id: UUID | None = None,
) -> Event:
    return Event(
        event_id=event_id or uuid4(),
        camera_id=camera_id,
        occurred_at=occurred_at,
        reason=reason,
        threat=ThreatScore.from_value(threat),
        description=description,
        suggested_action="Go and look.",
        track_ids=track_ids,
        # The subject, which is what an alert keys on — `track_ids` is the whole scene.
        subject_track_ids=track_ids,
        clip_uri=clip_uri,
    )


def open_alert(first: Event, *, now: float = 0.0) -> Alert:
    return opened(first, alert_id=uuid4(), camera_label="Corridor 1", zone=Zone.CORRIDOR, now=now)


class TestTheSpecExample:
    """§17, verbatim: the same unauthorised person detected repeatedly for 20 seconds
    must be one alert with an occurrence count — not a hundred alerts."""

    def test_seventeen_sightings_in_twenty_seconds_are_one_alert(self) -> None:
        first = event(occurred_at=0.0)
        alert = open_alert(first, now=0.0)

        for index in range(1, 17):
            recurrence = event(occurred_at=index * 1.25)
            assert should_merge(alert, recurrence, now=recurrence.occurred_at)
            alert = merged(alert, recurrence, now=recurrence.occurred_at)

        assert alert.occurrences == 17
        assert alert.first_seen == 0.0
        assert alert.last_seen == pytest.approx(20.0)

    def test_a_different_person_is_a_different_alert(self) -> None:
        """Ten people entering a restricted area is ten people, not one vague alert
        saying somebody is in the stairwell."""
        alert = open_alert(event(track_ids=(7,)))
        other = event(track_ids=(8,))
        assert not should_merge(alert, other, now=1.0)

    def test_the_same_person_four_minutes_later_is_a_new_episode(self) -> None:
        """Without a window an alert has no end, and the console shows one row with
        `occurrences: 340` spanning a week."""
        alert = open_alert(event(occurred_at=0.0))
        later = event(occurred_at=DEFAULT_MERGE_WINDOW_SECONDS + 1.0)
        assert not should_merge(alert, later, now=later.occurred_at)


class TestKeying:
    def test_track_ids_are_order_insensitive(self) -> None:
        assert AlertKey.from_event(event(track_ids=(3, 9))) == AlertKey.from_event(
            event(track_ids=(9, 3))
        )

    def test_a_camera_level_finding_has_no_subject(self) -> None:
        """Nobody to attribute a covered lens to, so every tamper report from one
        camera is one episode."""
        key = AlertKey.from_event(event(reason=EscalationReason.CAMERA_TAMPER, track_ids=()))
        assert key.subject == ""

    def test_the_same_person_on_two_cameras_is_two_alerts(self) -> None:
        alert = open_alert(event(camera_id="cam-1"))
        assert not should_merge(alert, event(camera_id="cam-2"), now=1.0)

    def test_two_reasons_about_one_person_are_two_alerts(self) -> None:
        """A person who fell *and* is in a restricted area needs both said."""
        alert = open_alert(event(reason=EscalationReason.ZONE_INTRUSION))
        assert not should_merge(alert, event(reason=EscalationReason.FALL_SUSPECTED), now=1.0)


class TestSeverity:
    def test_a_structurally_serious_reason_cannot_be_demoted_by_a_calm_description(
        self,
    ) -> None:
        """The floor's whole job. Several seconds of geometry watched a person go down
        and stay down; one frame's opinion does not overturn that."""
        calm_fall = event(reason=EscalationReason.FALL_SUSPECTED, threat=0.0)
        assert alert_severity(calm_fall) is Severity.CRITICAL

    def test_an_alarming_description_promotes_a_routine_reason(self) -> None:
        """The only thing that can notice a fire in a periodic summary."""
        alarming = event(reason=EscalationReason.PERIODIC_SUMMARY, threat=0.95)
        assert alert_severity(alarming) is Severity.CRITICAL

    def test_severity_rises_and_never_falls_across_an_episode(self) -> None:
        """One calm frame must not drop an escalating situation down the list at the
        moment it most needs to be at the top."""
        alert = open_alert(event(reason=EscalationReason.SPEED_ANOMALY, threat=0.85))
        assert alert.severity is Severity.CRITICAL
        calmer = event(reason=EscalationReason.SPEED_ANOMALY, threat=0.1)
        assert merged(alert, calmer, now=1.0).severity is Severity.CRITICAL

    def test_a_quiet_periodic_summary_is_not_an_alert(self) -> None:
        """Most of what the gate escalates. A list that includes them is a list an
        operator scrolls past."""
        assert not is_alertable(event(reason=EscalationReason.PERIODIC_SUMMARY, threat=0.05))

    def test_every_behaviour_finding_is_alertable_however_it_was_described(self) -> None:
        for reason in (
            EscalationReason.FALL_SUSPECTED,
            EscalationReason.CAMERA_TAMPER,
            EscalationReason.ZONE_INTRUSION,
            EscalationReason.ABANDONED_OBJECT,
            EscalationReason.LINE_CROSSING,
        ):
            assert is_alertable(event(reason=reason, threat=0.0)), reason


class TestMerging:
    def test_the_description_becomes_the_most_recent(self) -> None:
        """An episode develops. "A person is lying on the floor" is more use than
        "a person appears to have fallen" was thirty seconds ago."""
        alert = open_alert(event(description="a person appears to have fallen"))
        alert = merged(alert, event(description="a person is lying on the floor"), now=5.0)
        assert alert.description == "a person is lying on the floor"

    def test_the_first_clip_is_kept_not_the_latest(self) -> None:
        """A later clip shows the middle of an episode; the first shows how it began."""
        alert = open_alert(event(clip_uri="s3://clips/first.mp4"))
        alert = merged(alert, event(clip_uri="s3://clips/second.mp4"), now=5.0)
        assert alert.clip_uri == "s3://clips/first.mp4"

    def test_a_clip_arrives_late_when_the_first_event_had_none(self) -> None:
        """Clips finish after their event is assembled, so the first contributing event
        routinely has `clip_uri=None`."""
        alert = open_alert(event(clip_uri=None))
        alert = merged(alert, event(clip_uri="s3://clips/second.mp4"), now=5.0)
        assert alert.clip_uri == "s3://clips/second.mp4"

    def test_event_ids_are_bounded_and_keep_the_oldest(self) -> None:
        alert = open_alert(event())
        for index in range(MAX_EVENT_IDS + 20):
            alert = merged(alert, event(), now=float(index))
        assert len(alert.event_ids) == MAX_EVENT_IDS
        assert alert.occurrences == MAX_EVENT_IDS + 21, "the count stays exact"

    def test_an_acknowledged_alert_that_recurs_stays_acknowledged(self) -> None:
        """A person already knows. Re-raising it would put it back in front of somebody
        in the middle of dealing with it."""
        alert = open_alert(event()).acknowledged(by="operator-1", at=1.0)
        alert = merged(alert, event(), now=5.0)
        assert alert.state is AlertState.ACKNOWLEDGED
        assert alert.occurrences == 2

    def test_a_resolved_alert_does_not_absorb_a_recurrence(self) -> None:
        """A person said it was finished. Quietly reopening would erase that
        judgement; it happening again is a new episode."""
        alert = open_alert(event()).resolved()
        assert not should_merge(alert, event(), now=1.0)


class TestStateMachine:
    def test_acknowledging_records_who_and_when(self) -> None:
        alert = open_alert(event()).acknowledged(by="operator-1", at=42.0)
        assert alert.state is AlertState.ACKNOWLEDGED
        assert alert.acknowledged_by == "operator-1"
        assert alert.acknowledged_at == 42.0

    def test_acknowledging_a_resolved_alert_is_refused(self) -> None:
        """It means the operator is looking at a stale list. Silently accepting would
        tell them they had done something they had not."""
        with pytest.raises(ValueError, match="already resolved"):
            open_alert(event()).resolved().acknowledged(by="operator-1", at=1.0)

    def test_resolving_twice_is_not_an_error(self) -> None:
        """Two operators closing the same row is an ordinary race, not a mistake
        either of them made."""
        assert open_alert(event()).resolved().resolved().state is AlertState.RESOLVED

    def test_nothing_resolves_itself(self) -> None:
        """A fall alert that aged out on its own would leave no trace that nobody ever
        went to look. There is deliberately no function here that could do it."""
        import sentinel_ai.domain.policy.alerting as alerting

        assert not [name for name in dir(alerting) if "expire" in name or "auto" in name]


class TestInvariants:
    def test_an_alert_always_has_at_least_one_occurrence(self) -> None:
        with pytest.raises(ValueError, match="occurrences"):
            Alert(
                alert_id=uuid4(),
                key=AlertKey("cam-1", EscalationReason.ZONE_INTRUSION, "7"),
                state=AlertState.ACTIVE,
                severity=Severity.HIGH,
                priority=EventPriority.HIGH,
                first_seen=0.0,
                last_seen=0.0,
                occurrences=0,
                description="",
                camera_label="Corridor 1",
                zone=None,
                event_ids=(),
            )

    def test_time_cannot_run_backwards_within_an_alert(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            Alert(
                alert_id=uuid4(),
                key=AlertKey("cam-1", EscalationReason.ZONE_INTRUSION, "7"),
                state=AlertState.ACTIVE,
                severity=Severity.HIGH,
                priority=EventPriority.HIGH,
                first_seen=10.0,
                last_seen=5.0,
                occurrences=1,
                description="",
                camera_label="Corridor 1",
                zone=None,
                event_ids=(),
            )


class TestTheSubjectIsNotTheScene:
    """The bug a live run found, pinned.

    Alerts originally keyed on `Event.track_ids` — everybody in shot. On a busy camera
    that is most of the frame and it changes every frame, so eleven intrusions by the
    same handful of people produced subjects like `3,4,43,120,130,133,146,160` and
    eleven separate alerts. §17's deduplication requirement was silently unmet.
    """

    def test_a_crowded_scene_does_not_change_the_subject(self) -> None:
        first = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=0.0,
            reason=EscalationReason.ZONE_INTRUSION,
            threat=ThreatScore.from_value(0.5),
            description="a person is in the stairwell",
            suggested_action="Go and look.",
            track_ids=(1, 2, 3, 4, 7),
            subject_track_ids=(7,),
        )
        crowded = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=5.0,
            reason=EscalationReason.ZONE_INTRUSION,
            threat=ThreatScore.from_value(0.5),
            description="a person is in the stairwell",
            suggested_action="Go and look.",
            # Eight more people wandered into shot. Same trespasser.
            track_ids=(1, 2, 3, 4, 7, 43, 120, 130, 133, 146),
            subject_track_ids=(7,),
        )
        alert = open_alert(first)
        assert should_merge(alert, crowded, now=5.0), "the scene changed; the subject did not"

    def test_an_unattributed_event_does_not_borrow_the_scene_as_its_subject(self) -> None:
        """Empty must stay empty. Falling back to `track_ids` is precisely the bug."""
        unattributed = Event(
            event_id=uuid4(),
            camera_id="cam-1",
            occurred_at=0.0,
            reason=EscalationReason.CAMERA_TAMPER,
            threat=ThreatScore.from_value(0.5),
            description="this camera appears to have stopped seeing",
            suggested_action="Go and look.",
            track_ids=(1, 2, 3),
            subject_track_ids=(),
        )
        assert AlertKey.from_event(unattributed).subject == ""


class TestTheShortClip:
    """The notification-length copy of the same recording (`Alert.notify_clip_uri`).

    It travels beside `clip_uri` rather than instead of it because the two answer
    different questions: a console triaging a wall of rows wants three seconds that play
    inline, and somebody who has chosen a row wants the seconds before and after. The
    rules below exist so the short one can never describe a different moment than the
    long one.
    """

    def test_an_opened_alert_keeps_the_short_clip_beside_the_full_one(self) -> None:
        alert = opened(
            event(clip_uri="s3://clips/cam-1/a.mp4"),
            alert_id=uuid4(),
            camera_label="Corridor 1",
            zone=Zone.CORRIDOR,
            now=0.0,
            notify_clip_uri="s3://clips/cam-1/a-notify.mp4",
        )
        assert alert.clip_uri == "s3://clips/cam-1/a.mp4"
        assert alert.notify_clip_uri == "s3://clips/cam-1/a-notify.mp4"

    def test_a_short_clip_without_a_full_one_is_dropped(self) -> None:
        """An alert whose `clip_uri` is null is saying no recording landed. Keeping a
        short clip alongside that would offer evidence the same row says does not
        exist."""
        alert = opened(
            event(clip_uri=None),
            alert_id=uuid4(),
            camera_label="Corridor 1",
            zone=Zone.CORRIDOR,
            now=0.0,
            notify_clip_uri="s3://clips/cam-1/a-notify.mp4",
        )
        assert alert.notify_clip_uri is None

    def test_the_short_clip_is_absent_when_the_writer_made_none(self) -> None:
        """The ordinary case on an engine configured not to trim, and after a trim that
        failed. The full clip is unaffected either way."""
        alert = open_alert(event(clip_uri="s3://clips/cam-1/a.mp4"))
        assert alert.clip_uri == "s3://clips/cam-1/a.mp4"
        assert alert.notify_clip_uri is None

    def test_a_later_event_does_not_replace_a_short_clip_already_held(self) -> None:
        """`clip_uri` keeps the first clip that finished, so this has to keep the first
        short one. A rule that let the newest win would pair three seconds of the middle
        of an episode with a full clip showing its beginning."""
        alert = opened(
            event(clip_uri="s3://clips/cam-1/first.mp4"),
            alert_id=uuid4(),
            camera_label="Corridor 1",
            zone=Zone.CORRIDOR,
            now=0.0,
            notify_clip_uri="s3://clips/cam-1/first-notify.mp4",
        )
        later = merged(
            alert,
            event(clip_uri="s3://clips/cam-1/second.mp4", occurred_at=5.0),
            now=5.0,
            notify_clip_uri="s3://clips/cam-1/second-notify.mp4",
        )
        assert later.clip_uri == "s3://clips/cam-1/first.mp4"
        assert later.notify_clip_uri == "s3://clips/cam-1/first-notify.mp4"

    def test_a_later_clip_brings_its_own_short_copy_with_it(self) -> None:
        """An episode that opened before any clip had finished. When one finally does,
        both halves of it arrive together — the pair is what must stay consistent, not
        the presence of either one."""
        alert = open_alert(event(clip_uri=None))
        assert alert.clip_uri is None

        later = merged(
            alert,
            event(clip_uri="s3://clips/cam-1/late.mp4", occurred_at=5.0),
            now=5.0,
            notify_clip_uri="s3://clips/cam-1/late-notify.mp4",
        )
        assert later.clip_uri == "s3://clips/cam-1/late.mp4"
        assert later.notify_clip_uri == "s3://clips/cam-1/late-notify.mp4"

    def test_a_short_clip_cannot_arrive_without_the_full_one_on_a_merge(self) -> None:
        """The merge-time half of `test_a_short_clip_without_a_full_one_is_dropped`."""
        alert = open_alert(event(clip_uri=None))
        later = merged(
            alert,
            event(clip_uri=None, occurred_at=5.0),
            now=5.0,
            notify_clip_uri="s3://clips/cam-1/orphan-notify.mp4",
        )
        assert later.clip_uri is None
        assert later.notify_clip_uri is None
