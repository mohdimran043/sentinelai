"""The alert store: does an acknowledgement actually survive a restart?

Everything here is about the one promise the store makes. The register was always
correct in memory; what was missing was that a `kill` threw away the only record of
what a human had decided.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from sentinel_ai.adapters.alerts.json_store import JsonAlertStore
from sentinel_ai.adapters.serialization.alert_codec import decode_register, encode_register
from sentinel_ai.domain.alert import Alert, AlertKey, AlertState
from sentinel_ai.domain.entities import EscalationReason, Severity
from sentinel_ai.domain.policy.priority import EventPriority
from sentinel_ai.domain.zone import Zone


def an_alert(**overrides: Any) -> Alert:
    """A fully-populated alert: every optional field set, so a codec that drops one is
    caught by the `==` in the round-trip tests rather than by luck."""
    base = Alert(
        alert_id=uuid4(),
        key=AlertKey(camera_id="corridor-1", reason=EscalationReason.FALL_SUSPECTED, subject="7"),
        state=AlertState.ACKNOWLEDGED,
        severity=Severity.CRITICAL,
        priority=EventPriority.CRITICAL,
        first_seen=1_700_000_000.0,
        last_seen=1_700_000_042.5,
        occurrences=17,
        description="a person appears to have fallen",
        camera_label="Corridor 1",
        zone=Zone.CORRIDOR,
        event_ids=(uuid4(), uuid4()),
        clip_uri="s3://clips/one.mp4",
        acknowledged_by="night shift",
        acknowledged_at=1_700_000_050.0,
    )
    return replace(base, **overrides) if overrides else base


class TestRoundTrip:
    @pytest.mark.asyncio
    async def test_an_acknowledgement_survives_a_restart(self, tmp_path: Path) -> None:
        """The whole point, in one test. A second `JsonAlertStore` over the same path is
        what a restarted engine has."""
        alert = an_alert()
        await JsonAlertStore(tmp_path / "alerts.json").save([alert])

        (restored,) = await JsonAlertStore(tmp_path / "alerts.json").load()

        assert restored == alert
        assert restored.state is AlertState.ACKNOWLEDGED
        assert restored.acknowledged_by == "night shift"

    @pytest.mark.asyncio
    async def test_every_field_survives_not_just_the_state(self, tmp_path: Path) -> None:
        """`==` on a frozen dataclass compares all of them, so this pins that the codec
        does not quietly drop one that `Alert.__eq__` would notice."""
        alerts = [
            an_alert(occurrences=1, zone=None, clip_uri=None, acknowledged_by=None),
            an_alert(state=AlertState.RESOLVED),
            an_alert(state=AlertState.ACTIVE, acknowledged_by=None, acknowledged_at=None),
        ]
        store = JsonAlertStore(tmp_path / "alerts.json")
        await store.save(alerts)
        assert list(await store.load()) == alerts

    @pytest.mark.asyncio
    async def test_saving_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        """The register is the source of truth and a save mirrors it whole. A store that
        accumulated would resurrect an evicted alert on the next restart."""
        store = JsonAlertStore(tmp_path / "alerts.json")
        await store.save([an_alert(), an_alert()])
        await store.save([an_alert()])
        assert len(await store.load()) == 1


class TestFailureIsNeverFatal:
    @pytest.mark.asyncio
    async def test_a_missing_file_is_a_fresh_site_not_an_error(self, tmp_path: Path) -> None:
        assert await JsonAlertStore(tmp_path / "nothing-here.json").load() == ()

    @pytest.mark.asyncio
    async def test_unreadable_json_loses_triage_state_and_keeps_the_engine_up(
        self, tmp_path: Path
    ) -> None:
        """Surveillance must not stop because a bookkeeping file was truncated."""
        path = tmp_path / "alerts.json"
        path.write_text("{ this is not json", encoding="utf-8")
        assert await JsonAlertStore(path).load() == ()
        # Kept rather than deleted: the next save overwrites it, and it is the one
        # artefact somebody debugging this needs.
        assert path.exists()

    @pytest.mark.asyncio
    async def test_a_future_schema_version_is_refused_rather_than_guessed_at(
        self, tmp_path: Path
    ) -> None:
        """A build reading a newer file would be guessing at the fields that decide
        whether an operator is shown an alert as unseen."""
        path = tmp_path / "alerts.json"
        document = encode_register([an_alert()])
        document["schema_version"] = 99
        path.write_text(json.dumps(document), encoding="utf-8")
        assert await JsonAlertStore(path).load() == ()

    @pytest.mark.asyncio
    async def test_a_directory_that_does_not_exist_yet_is_created(self, tmp_path: Path) -> None:
        """`./var/alerts/alerts.json` is the default and nothing else creates `var/`."""
        store = JsonAlertStore(tmp_path / "var" / "alerts" / "alerts.json")
        await store.save([an_alert()])
        assert len(await store.load()) == 1


class TestAtomicity:
    @pytest.mark.asyncio
    async def test_a_failed_write_leaves_the_previous_set_intact(self, tmp_path: Path) -> None:
        """The property the rename buys. An in-place write that failed half way would
        leave an alert list that parses as fewer alerts than there are."""
        path = tmp_path / "alerts.json"
        keep = an_alert()
        store = JsonAlertStore(path)
        await store.save([keep])

        # A description that encodes fine and serialises not at all, so the failure
        # lands inside `json.dump` — i.e. on the write path this test is about, rather
        # than before it.
        with pytest.raises(TypeError):
            await store.save([an_alert(description=object())])

        assert list(await store.load()) == [keep]

    @pytest.mark.asyncio
    async def test_a_failed_write_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        """A directory slowly filling with orphaned temp files is how a disk-full error
        becomes somebody else's problem."""
        path = tmp_path / "alerts.json"
        store = JsonAlertStore(path)
        await store.save([an_alert()])

        with pytest.raises(TypeError):
            await store.save([an_alert(description=object())])

        assert [p.name for p in tmp_path.iterdir()] == ["alerts.json"]


class TestCodec:
    def test_enums_are_stored_by_value_never_by_position(self) -> None:
        """Reordering `AlertState` or inserting a `Severity` must not reinterpret a
        stored alert. A name or an ordinal would let it."""
        document = encode_register([an_alert()])
        stored = document["alerts"][0]
        assert stored["state"] == "acknowledged"
        assert stored["severity"] == "critical"
        assert stored["key"]["reason"] == "fall_suspected"

    def test_the_document_says_what_it_is_for_whoever_finds_it(self) -> None:
        """Somebody will find this file on a server and need to know whether it is the
        record of what happened. It is not."""
        document = encode_register([])
        assert "not the record of what happened" in str(document["_comment"])

    def test_decoding_a_document_with_no_version_refuses(self) -> None:
        with pytest.raises(ValueError, match="refusing to guess"):
            decode_register({"alerts": []})
