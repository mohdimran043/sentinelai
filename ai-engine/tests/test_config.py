from __future__ import annotations

from sentinel_ai.config import Mode, Settings


def test_development_is_the_default_mode() -> None:
    assert Settings().mode is Mode.DEVELOPMENT


def test_default_vlm_is_the_3b_awq_build() -> None:
    assert Settings().vlm_model_id == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"


def test_vram_defaults_match_the_8gb_budget() -> None:
    settings = Settings()
    assert settings.vram_total_mib == 8192
    assert settings.vram_reserved_mib == 2048


def test_env_prefix_overrides_mode(monkeypatch) -> None:
    monkeypatch.setenv("SENTINEL_MODE", "production")
    assert Settings().mode is Mode.PRODUCTION
