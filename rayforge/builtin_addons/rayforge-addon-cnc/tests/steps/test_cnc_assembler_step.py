"""Tests for the base CNC assembler step setters."""

from unittest.mock import MagicMock

import pytest
from cnc_essentials.steps import (
    AdaptiveClearStep,
    ProfileOuterStep,
    ToroidalClearStep,
)
from cnc_essentials.steps.cnc_assembler_step import CncAssemblerStep

from rayforge.core.step import Step


@pytest.fixture
def cnc_step():
    return ProfileOuterStep(name="profile_outer")


@pytest.mark.parametrize(
    "attr, value, expected",
    [
        ("tool_diameter", 8.0, 8.0),
        ("spindle_rpm", 15000, 15000),
        ("plunge_speed", 300, 300),
        ("target_depth", -3.5, -3.5),
        ("depth_per_pass", 0.5, 0.5),
        ("safe_z", 5.0, 5.0),
    ],
)
def test_setters_update_attribute_and_signal(cnc_step, attr, value, expected):
    handler = MagicMock()
    cnc_step.updated.connect(handler)

    setter = getattr(cnc_step, f"set_{attr}")
    setter(value)

    assert getattr(cnc_step, attr) == expected
    handler.assert_called_once_with(cnc_step)


@pytest.mark.parametrize(
    "attr",
    [
        "tool_diameter",
        "spindle_rpm",
        "plunge_speed",
        "target_depth",
        "depth_per_pass",
        "safe_z",
    ],
)
def test_setters_no_signal_on_same_value(cnc_step, attr):
    handler = MagicMock()
    cnc_step.updated.connect(handler)

    getattr(cnc_step, f"set_{attr}")(getattr(cnc_step, attr))

    handler.assert_not_called()


def test_int_setters_coerce_values(cnc_step):
    cnc_step.set_spindle_rpm(15000.9)
    cnc_step.set_plunge_speed(300.9)
    assert cnc_step.spindle_rpm == 15000
    assert cnc_step.plunge_speed == 300


def test_float_setters_coerce_values(cnc_step):
    cnc_step.set_tool_diameter(8)
    assert isinstance(cnc_step.tool_diameter, float)
    assert cnc_step.tool_diameter == 8.0


def test_all_recipe_keys_have_setters(cnc_step):
    for var in CncAssemblerStep.recipe_varset():
        assert hasattr(cnc_step, f"set_{var.key}"), var.key


BASE_CNC_KEYS = (
    "tool_diameter",
    "spindle_rpm",
    "plunge_speed",
    "target_depth",
    "depth_per_pass",
    "safe_z",
)

PROFILE_KEYS = ("step_over", "step_length", "wall_margin")


class TestCncSerialization:
    def test_base_cnc_attrs_round_trip(self):
        step = ProfileOuterStep(name="profile_outer")
        step.tool_diameter = 8.0
        step.spindle_rpm = 15000
        step.plunge_speed = 300
        step.target_depth = -3.5
        step.depth_per_pass = 0.5
        step.safe_z = 5.0

        data = step.to_dict()
        restored = ProfileOuterStep.from_dict(data)

        assert restored.tool_diameter == 8.0
        assert restored.spindle_rpm == 15000
        assert restored.plunge_speed == 300
        assert restored.target_depth == -3.5
        assert restored.depth_per_pass == 0.5
        assert restored.safe_z == 5.0

    def test_step_specific_attrs_round_trip(self):
        step = ProfileOuterStep(name="profile_outer")
        step.step_over = 3.0
        step.step_length = 0.8
        step.wall_margin = 0.5

        data = step.to_dict()
        restored = ProfileOuterStep.from_dict(data)

        assert restored.step_over == 3.0
        assert restored.step_length == 0.8
        assert restored.wall_margin == 0.5

    def test_adaptive_clear_attrs_round_trip(self):
        step = AdaptiveClearStep(name="adaptive_clear")
        step.step_over = 2.5
        step.step_length = 0.7
        step.max_deflection_deg = 25.0
        step.wall_margin = 0.2
        step.area_tolerance = 0.5

        data = step.to_dict()
        restored = AdaptiveClearStep.from_dict(data)

        assert restored.step_over == 2.5
        assert restored.step_length == 0.7
        assert restored.max_deflection_deg == 25.0
        assert restored.wall_margin == 0.2
        assert restored.area_tolerance == 0.5

    def test_toroidal_clear_attrs_round_trip(self):
        step = ToroidalClearStep(name="toroidal_clear")
        step.step_over = 3.5

        data = step.to_dict()
        restored = ToroidalClearStep.from_dict(data)

        assert restored.step_over == 3.5

    def test_cnc_attrs_not_stashed_in_extra(self):
        step = ProfileOuterStep(name="profile_outer")
        data = step.to_dict()
        restored = ProfileOuterStep.from_dict(data)

        for key in BASE_CNC_KEYS + PROFILE_KEYS:
            assert key not in restored.extra

    def test_old_files_without_cnc_keys_load_defaults(self):
        step = ProfileOuterStep(name="profile_outer")
        data = step.to_dict()
        for key in BASE_CNC_KEYS:
            data.pop(key)

        restored = ProfileOuterStep.from_dict(data)

        assert restored.tool_diameter == 6.0
        assert restored.spindle_rpm == 12000
        assert restored.plunge_speed == 200
        assert restored.target_depth == -5.0
        assert restored.depth_per_pass == 1.0
        assert restored.safe_z == 2.0

    def test_unknown_keys_preserved_in_extra(self):
        step = ProfileOuterStep(name="profile_outer")
        data = step.to_dict()
        data["future_field"] = "future value"

        restored = ProfileOuterStep.from_dict(data)

        assert restored.extra["future_field"] == "future value"
        re_serialized = restored.to_dict()
        assert re_serialized["future_field"] == "future value"

    def test_dispatch_via_base_step_from_dict(self):
        step = ProfileOuterStep(name="profile_outer")
        step.step_over = 3.0
        data = step.to_dict()

        restored = Step.from_dict(data)

        assert isinstance(restored, ProfileOuterStep)
        assert restored.step_over == 3.0
        assert restored.tool_diameter == step.tool_diameter
