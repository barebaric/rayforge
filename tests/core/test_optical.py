"""Tests for the physical optical model (absorption + burn response)."""

import pytest

from rayforge.core.material import Material, MaterialAppearance
from rayforge.core.optical import (
    DEFAULT_ABSORPTION,
    DEFAULT_BAND_ABSORPTION,
    absorption_for,
    burn_response_for,
    material_absorption,
    material_burn_response,
    wavelength_to_band,
)


class TestWavelengthToBand:
    def test_blue_band(self):
        assert wavelength_to_band(455.0) == "blue"
        assert wavelength_to_band(445.0) == "blue"
        assert wavelength_to_band(400.0) == "blue"

    def test_ir_band(self):
        assert wavelength_to_band(1064.0) == "ir"

    def test_co2_band(self):
        assert wavelength_to_band(10600.0) == "co2"
        assert wavelength_to_band(9300.0) == "co2"

    def test_zero_or_negative_falls_back_to_blue(self):
        assert wavelength_to_band(0.0) == "blue"
        assert wavelength_to_band(-1.0) == "blue"


class TestAbsorptionFor:
    def test_none_dict_returns_full_absorption(self):
        assert absorption_for(455.0, None) == DEFAULT_ABSORPTION

    def test_missing_band_returns_neutral(self):
        assert absorption_for(455.0, {"ir": 0.2}) == DEFAULT_BAND_ABSORPTION

    def test_present_band_returns_value(self):
        assert absorption_for(455.0, {"blue": 0.9}) == 0.9

    def test_clamps_to_unit_range(self):
        assert absorption_for(455.0, {"blue": 2.0}) == 1.0
        assert absorption_for(455.0, {"blue": -0.5}) == 0.0

    def test_non_numeric_falls_back_to_neutral(self):
        bad: dict[str, float] = {"blue": "high"}  # type: ignore[dict-item]
        assert absorption_for(455.0, bad) == DEFAULT_BAND_ABSORPTION

    def test_co2_wavelength_maps_to_co2_band(self):
        assert absorption_for(10600.0, {"blue": 0.1, "co2": 0.92}) == 0.92


class TestBurnResponseFor:
    def test_none_returns_defaults(self):
        br = burn_response_for(None)
        assert br["char_threshold"] == 35.0
        assert br["char_saturation"] == 125.0
        assert br["char_color_low"] == (0.04, 0.03, 0.02)

    def test_partial_override(self):
        br = burn_response_for({"char_threshold": 10.0})
        assert br["char_threshold"] == 10.0
        assert br["char_saturation"] == 125.0  # default kept

    def test_clamps_thresholds(self):
        br = burn_response_for(
            {"char_threshold": -1.0, "char_saturation": 500.0}
        )
        assert br["char_threshold"] == 0.0
        assert br["char_saturation"] == 500.0

    def test_non_degenerate_ramp_enforced(self):
        br = burn_response_for(
            {"char_threshold": 35.0, "char_saturation": 40.0}
        )
        assert br["char_saturation"] > br["char_threshold"]

    def test_color_from_hex(self):
        br = burn_response_for({"char_color_low": "#ff0000"})
        assert br["char_color_low"] == (1.0, 0.0, 0.0)

    def test_color_from_floats(self):
        br = burn_response_for({"char_color_low": [0.1, 0.2, 0.3]})
        assert br["char_color_low"] == (0.1, 0.2, 0.3)

    def test_color_from_0_255_ints_normalized(self):
        br = burn_response_for({"char_color_low": [128, 64, 32]})
        assert br["char_color_low"] == pytest.approx(
            (128 / 255, 64 / 255, 32 / 255)
        )

    def test_invalid_color_falls_back_to_grey(self):
        br = burn_response_for({"char_color_low": "not-a-color"})
        assert br["char_color_low"] == (0.5, 0.5, 0.5)


class TestMaterialOptical:
    def _material(self, extra=None):
        appearance = MaterialAppearance()
        appearance.extra = extra or {}
        return Material(uid="t", appearance=appearance)

    def test_material_without_optical_data_uses_full_absorption(self):
        m = self._material()
        assert material_absorption(m, 455.0) == DEFAULT_ABSORPTION

    def test_material_with_absorption(self):
        m = self._material({"absorption": {"blue": 0.3, "co2": 0.9}})
        assert material_absorption(m, 455.0) == 0.3
        assert material_absorption(m, 10600.0) == 0.9
        # Missing ir band → neutral default
        assert material_absorption(m, 1064.0) == DEFAULT_BAND_ABSORPTION

    def test_material_burn_response_defaults(self):
        m = self._material()
        br = material_burn_response(m)
        assert "char_threshold" in br
        assert "char_color_low" in br

    def test_material_burn_response_override(self):
        m = self._material(
            {"burn_response": {"char_threshold": 0.3, "char_saturation": 0.9}}
        )
        br = material_burn_response(m)
        assert br["char_threshold"] == 0.3
        assert br["char_saturation"] == 0.9
