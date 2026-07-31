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


def test_clips_carry_five_seconds_of_post_roll() -> None:
    assert defaults().clip_postroll_seconds == 5.0


def test_detector_thresholds_match_the_yolo11s_defaults() -> None:
    settings = defaults()
    assert settings.detector_conf_threshold == 0.35
    assert settings.detector_iou_threshold == 0.45
    assert settings.detector_imgsz == 640


def test_vlm_queue_defaults_bound_the_scheduler() -> None:
    settings = defaults()
    assert settings.vlm_queue_maxsize == 4
    assert settings.vlm_timeout_seconds == 30.0
    assert settings.vlm_max_new_tokens == 256
    assert settings.vlm_global_concurrency == 1
    assert settings.vlm_global_min_interval_seconds == 2.0


def test_detect_every_n_frames_defaults_to_every_frame() -> None:
    assert defaults().detect_every_n_frames == 1


def test_source_realtime_defaults_to_true() -> None:
    assert defaults().source_realtime is True


def test_rtsp_reconnect_backoff_defaults() -> None:
    settings = defaults()
    assert settings.rtsp_reconnect_initial_seconds == 1.0
    assert settings.rtsp_reconnect_max_seconds == 30.0


def test_clip_temp_dir_default() -> None:
    assert defaults().clip_temp_dir == "./var/clips"


def test_env_prefix_overrides_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_MODE", "production")
    assert Settings().mode is Mode.PRODUCTION
