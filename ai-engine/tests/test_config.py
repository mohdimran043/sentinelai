from __future__ import annotations

import pytest

from sentinel_ai.config import Mode, Settings


def defaults() -> Settings:
    """Settings with no `.env` applied.

    `Settings()` honours `env_file=".env"`, so a developer with an
    `ai-engine/.env` that sets SENTINEL_MODE would otherwise fail these tests.
    These assertions pin the *code* defaults, not the local environment.
    """
    # `_env_file` is a pydantic-settings runtime override; it is not part of the
    # model's generated __init__ signature, hence the ignore.
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_development_is_the_default_mode() -> None:
    assert defaults().mode is Mode.DEVELOPMENT


def test_default_vlm_is_the_3b_awq_build() -> None:
    assert defaults().vlm_model_id == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


def test_vram_defaults_match_the_8gb_budget() -> None:
    settings = defaults()
    assert settings.vram_total_mib == 8192
    assert settings.vram_reserved_mib == 2048


def test_the_vlm_unloads_after_ten_idle_minutes() -> None:
    assert defaults().vlm_idle_unload_seconds == 600.0


def test_clips_carry_three_seconds_of_pre_roll() -> None:
    assert defaults().clip_preroll_seconds == 3.0


def test_env_prefix_overrides_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_MODE", "production")
    assert Settings().mode is Mode.PRODUCTION
