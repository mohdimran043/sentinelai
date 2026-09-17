"""`SENTINEL_DECODE_HWACCEL` stopped being a placeholder. These pin what it now does.

The behaviour that matters is the refusal. FFmpeg will fall back to software decode per
stream without saying so, and a deployment that asked for hardware decode and quietly
got software is one whose capacity planning is wrong and whose logs agree with it.
"""

from __future__ import annotations

import pytest

from sentinel_ai.adapters.sources.hwaccel import available_hwaccel_devices, build_hwaccel


class TestBuildHwaccel:
    def test_unset_means_software_decode(self) -> None:
        """The default, and the configuration every measurement in docs/performance.md
        except one was taken under."""
        assert build_hwaccel(None) is None

    def test_an_unavailable_device_is_refused_rather_than_fallen_back_from(self) -> None:
        name = "videotoolbox"  # macOS-only; never present in a Linux CI image
        if name in available_hwaccel_devices():  # pragma: no cover - not on Linux
            pytest.skip(f"{name} is available on this box, so it cannot stand in")
        with pytest.raises(ValueError, match="not available in this FFmpeg build"):
            build_hwaccel(name)

    def test_the_refusal_names_what_is_available(self) -> None:
        """The usual cause is a correct intention and a build without the support
        compiled in, and "cuda is not in ('drm',)" is a much faster diagnosis than a
        decode that is merely slower than expected."""
        with pytest.raises(ValueError) as caught:
            build_hwaccel("definitely-not-a-device")
        for device in available_hwaccel_devices():
            assert device in str(caught.value)

    def test_an_available_device_produces_an_hwaccel(self) -> None:
        available = available_hwaccel_devices()
        if not available:  # pragma: no cover - a build with no hwaccel at all
            pytest.skip("this FFmpeg build offers no hardware devices")
        accel = build_hwaccel(available[0])
        assert accel is not None

    def test_software_fallback_is_disabled_on_the_accelerator_itself(self) -> None:
        """Belt and braces: the name check above cannot see a device that opens and then
        fails per stream, which is exactly when FFmpeg falls back silently."""
        available = available_hwaccel_devices()
        if not available:  # pragma: no cover
            pytest.skip("this FFmpeg build offers no hardware devices")
        accel = build_hwaccel(available[0])
        assert accel is not None
        assert accel.allow_software_fallback is False
