"""What this engine is actually configured to do, as a readable inventory.

Built from the live `Settings` object rather than from the environment, so what it
reports is what the process is running on — including defaults nobody set, which are
the ones an operator is most likely to be wrong about.

Secrecy is decided here and only here
--------------------------------------
`Settings` holds credentials: the MinIO keys, the broker URL with its password in it,
the biometric encryption key, the webhook URL with whatever token is embedded in its
path. None of those may cross the API, and this module is the single place that decides
which is which — a per-field flag scattered through `config.py` would be one missed
annotation away from publishing a key.

The rule is deliberately the conservative one: a field is secret if its *name* matches,
not if its value looks sensitive. A URL that happens to have no password in it today is
still redacted, because whether it does is a deployment's business and can change
without this file being touched.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentinel_ai.config import Settings

__all__ = ["SECRET_FIELDS", "SettingEntry", "describe_settings", "group_of"]

SECRET_FIELDS = frozenset(
    {
        "face_encryption_key",
        "minio_access_key",
        "minio_secret_key",
        "rabbitmq_url",
        "notifier_webhook_url",
    }
)
"""Never reported by value.

`rabbitmq_url` and `notifier_webhook_url` are here as whole values rather than being
parsed and partially shown: a URL's password is not the only secret in it — a webhook
path is frequently the credential — and a redactor that understood URLs would be a
thing to get subtly wrong for no gain.
"""

_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Models", ("model_", "vlm_", "detector_", "yolo", "pose_", "face_", "device", "quantization")),
    ("Cameras & sources", ("camera", "source_", "reconnect_", "hwaccel", "rtsp_")),
    ("Detection & escalation", ("escalation_", "motion_", "threat_", "summary_", "queue_")),
    ("Alerts & notifications", ("alert_", "notify_", "notifier_")),
    ("Clips & storage", ("clip_", "minio_")),
    ("Messaging", ("rabbitmq_", "publish", "dead_letter")),
)
"""Prefix to heading, first match wins. Ordered so the specific prefixes are tried
before the general ones — `face_encryption_key` is a model setting by prefix and a
credential by name, and only the second matters for redaction."""

_DEFAULT_GROUP = "Engine"


def group_of(name: str) -> str:
    """Which heading a setting belongs under. Presentation, kept beside the settings
    rather than in the console, so both agree and neither has to enumerate 30 names."""
    for heading, prefixes in _GROUPS:
        if any(name.startswith(prefix) or prefix in name for prefix in prefixes):
            return heading
    return _DEFAULT_GROUP


@dataclass(frozen=True, slots=True)
class SettingEntry:
    name: str
    """The environment variable, `SENTINEL_`-prefixed and upper-cased — what an operator
    would set, not the Python attribute they cannot."""

    value: str
    """Rendered for reading, and `"(set)"` or `"(unset)"` for a secret. A string rather
    than the typed value because the consumer is a table: `None`, `3.0` and `"high"` all
    have to render, and a union of every settable type buys the console nothing."""

    is_default: bool
    """Whether this is the value shipped with the engine. The column an operator scans
    to find what this deployment has actually changed."""

    secret: bool
    group: str


def describe_settings(settings: Settings) -> tuple[SettingEntry, ...]:
    """Every setting, with its effective value and whether anyone moved it."""
    defaults = Settings.model_fields
    entries: list[SettingEntry] = []
    for name in sorted(defaults):
        value = getattr(settings, name)
        default = defaults[name].default
        secret = name in SECRET_FIELDS
        entries.append(
            SettingEntry(
                name=f"SENTINEL_{name.upper()}",
                # A secret still reports *whether* it is set, which is the fact an
                # operator actually needs from it — "the engine has no encryption key"
                # is a diagnosis, and the key itself is never one.
                value=("(set)" if value not in (None, "") else "(unset)")
                if secret
                else _render(value),
                is_default=value == default,
                secret=secret,
                group=group_of(name),
            )
        )
    return tuple(entries)


def _render(value: object) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, bool):
        # Before the str branch and before the enum one: `True` renders as "True"
        # otherwise, which is not what anyone writes in an env file.
        return "true" if value else "false"
    return str(getattr(value, "value", value))
