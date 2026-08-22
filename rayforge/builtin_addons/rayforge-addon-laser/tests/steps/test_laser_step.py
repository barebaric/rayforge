"""Tests for the LaserStep domain base class."""

from unittest.mock import MagicMock

import pytest
from laser_essentials.steps import ContourStep, EngraveStep, LaserStep

from rayforge.core.step import Step
from rayforge.machine.driver.driver import PWMParams, pwm_varset
from rayforge.machine.models.laser import (
    MIN_SPOT_SIZE_MM,
    LaserHead,
)
from rayforge.machine.models.spindle import SpindleHead


def test_contour_defaults_preserved():
    s = ContourStep(name="t")
    assert s.power == 0.8, s.power
    assert s.offset_mm == 0.0, s.offset_mm
    assert s.cut_speed == 500, s.cut_speed
    assert s.air_assist is False
    assert isinstance(s, LaserStep)


def test_engrave_defaults():
    s = EngraveStep(name="t")
    assert s.power == 0.2, s.power
    assert s.cut_speed == 500, s.cut_speed
    assert isinstance(s, LaserStep)


def test_engrave_create_derives_cut_speed_from_machine():
    """EngraveStep.create() derives the operating feed from the machine.

    The machine only exposes its ceiling, so the default is that
    ceiling, bounded by engraving's typical feed rate."""
    context = MagicMock()
    machine = MagicMock()
    machine.max_cut_speed = 600
    machine.max_travel_speed = 10000
    machine.acceleration = 3000
    head = MagicMock()
    head.uid = "laser-1"
    head.spot_size_mm = (0.1, 0.1)
    head.get_defaults.return_value = {}
    machine.get_default_laser_head.return_value = head
    context.machine = machine

    s = EngraveStep.create(context, name="t")
    assert s.cut_speed == 600  # machine ceiling, below the 4000 bound

    machine.max_cut_speed = 8000
    s2 = EngraveStep.create(context, name="t")
    assert s2.cut_speed == 4000  # bounded by engraving's typical feed


def test_laser_step_serialization_roundtrip():
    s = ContourStep(name="t")
    s.power = 0.6
    s.offset_mm = 0.2
    s.air_assist = True
    s.frequency = 2000
    s.pulse_width = 100
    data = s.to_dict()

    r = Step.from_dict(data)
    assert type(r) is ContourStep
    assert r.power == 0.6
    assert r.offset_mm == 0.2
    assert r.air_assist is True
    assert r.frequency == 2000
    assert r.pulse_width == 100


def test_laser_step_summary():
    s = ContourStep(name="t")
    assert "% power" in s.get_summary()


def test_laser_step_get_selected_laser():
    s = ContourStep(name="t")
    machine = MagicMock()
    laser = MagicMock(spec=LaserHead)
    laser.uid = "laser-1"
    spindle = SpindleHead()
    spindle.uid = "spindle-1"
    machine.heads = [laser, spindle]
    assert s.get_selected_laser(machine) is laser
    s.selected_head_uid = "spindle-1"
    assert s.get_selected_laser(machine) is None


def test_laser_get_spot_size_uses_head_spot_size():
    s = ContourStep(name="t")
    machine = MagicMock()
    head = MagicMock(spec=LaserHead)
    head.uid = "laser-1"
    head.spot_size_mm = (0.08, 0.15)
    machine.heads = [head]
    assert LaserHead.get_spot_size(s.get_selected_laser(machine)) == (
        0.08,
        0.15,
    )


def test_laser_get_spot_size_falls_back_without_head():
    """With no laser head the spot falls back to a sane minimum."""
    assert LaserHead.get_spot_size(None) == (
        MIN_SPOT_SIZE_MM,
        MIN_SPOT_SIZE_MM,
    )


def test_laser_get_spot_size_clamps_zero():
    """A zero spot size (unconfigured head) is clamped so the raster
    pipeline never divides by zero."""
    head = LaserHead()
    head.spot_size_mm = (0.0, 0.0)
    spot_x, spot_y = LaserHead.get_spot_size(head)
    assert spot_x > 0
    assert spot_y > 0


def test_laser_get_spot_size_clamps_negative():
    head = LaserHead()
    head.spot_size_mm = (-0.1, 0.2)
    spot_x, spot_y = LaserHead.get_spot_size(head)
    assert spot_x > 0
    assert spot_y == 0.2


def test_laser_step_base_defaults():
    """LaserStep declares the laser domain defaults explicitly."""
    s = LaserStep(typelabel="test")
    assert s.power == 1.0
    assert s.max_power == 1000
    assert s.air_assist is False
    assert s.tab_power == 0.0
    assert s.frequency == 0
    assert s.pulse_width == 0


def test_set_power_validation():
    """set_power raises ValueError for out-of-range values."""
    s = ContourStep(name="t")
    with pytest.raises(ValueError):
        s.set_power(-0.1)
    with pytest.raises(ValueError):
        s.set_power(1.1)


def test_laser_setters_and_signals():
    """Laser setters update the value and fire the 'updated' signal."""
    s = ContourStep(name="t")
    handler = MagicMock()
    s.updated.connect(handler)

    s.set_power(0.75)
    assert s.power == 0.75
    handler.assert_called_once_with(s)
    handler.reset_mock()

    s.set_air_assist(True)
    assert s.air_assist is True
    handler.assert_called_once_with(s)
    handler.reset_mock()

    s.set_tab_power(0.15)
    assert s.tab_power == 0.15
    handler.assert_called_once_with(s)


def test_set_tab_power_validation():
    """set_tab_power raises ValueError for out-of-range values."""
    s = ContourStep(name="t")
    with pytest.raises(ValueError):
        s.set_tab_power(-0.1)
    with pytest.raises(ValueError):
        s.set_tab_power(1.1)


def test_frequency_and_pulse_width_defaults():
    s = ContourStep(name="t")
    assert s.frequency == 0
    assert s.pulse_width == 0


def test_set_frequency():
    s = ContourStep(name="t")
    handler = MagicMock()
    s.updated.connect(handler)
    s.set_frequency(1000)
    assert s.frequency == 1000
    handler.assert_called_once_with(s)


def test_set_pulse_width():
    s = ContourStep(name="t")
    handler = MagicMock()
    s.updated.connect(handler)
    s.set_pulse_width(50)
    assert s.pulse_width == 50
    handler.assert_called_once_with(s)


def test_setters_no_signal_on_same_value():
    s = ContourStep(name="t")
    handler = MagicMock()
    s.updated.connect(handler)
    s.set_frequency(0)
    s.set_pulse_width(0)
    handler.assert_not_called()


def test_frequency_pulse_width_serialization_roundtrip():
    s = ContourStep(name="t")
    s.set_frequency(2000)
    s.set_pulse_width(100)
    data = s.to_dict()
    assert data["frequency"] == 2000
    assert data["pulse_width"] == 100

    restored = ContourStep.from_dict(data)
    assert restored.frequency == 2000
    assert restored.pulse_width == 100


def test_frequency_pulse_width_missing_defaults():
    data = {
        "uid": "step-min",
        "type": "step",
        "typelabel": "MinimalType",
        "visible": True,
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "per_workpiece_transformers_dicts": [],
        "per_step_transformers_dicts": [],
    }
    restored = ContourStep.from_dict(data)
    assert restored.frequency == 0
    assert restored.pulse_width == 0


def test_get_settings_includes_frequency_and_pulse_width():
    s = ContourStep(name="t")
    s.set_frequency(1000)
    s.set_pulse_width(50)
    settings = s.get_settings()
    assert settings["frequency"] == 1000
    assert settings["pulse_width"] == 50


def test_machine_reports_pwm_settings():
    """A machine's PWM settings expose the driver's PWM defaults."""
    params = PWMParams(1000, 5000, 50, 5, 500)
    machine = MagicMock()
    machine.get_pwm_settings.return_value = pwm_varset(params)

    vs = machine.get_pwm_settings(None)
    assert vs["frequency"].default == 1000
    assert vs["pulse_width"].default == 50


def test_machine_reports_no_pwm_without_support():
    machine = MagicMock()
    machine.get_pwm_settings.return_value = None
    assert machine.get_pwm_settings(None) is None


def test_create_applies_head_pwm_defaults():
    """create() adopts the default head's PWM defaults."""
    context = MagicMock()
    machine = MagicMock()
    machine.max_cut_speed = 600
    machine.max_travel_speed = 10000
    machine.acceleration = 3000
    head = MagicMock(spec=LaserHead)
    head.uid = "laser-1"
    head.spot_size_mm = (0.1, 0.1)
    machine.get_default_laser_head.return_value = head
    machine.get_pwm_params.return_value = PWMParams(1000, 5000, 50, 5, 500)
    context.machine = machine

    s = ContourStep.create(context, name="t")
    assert s.frequency == 1000
    assert s.pulse_width == 50


def test_laser_step_uses_head_color():
    """A laser step reports the head's color."""
    head = MagicMock(spec=LaserHead)
    head.cut_color = "#112233"

    s = ContourStep(name="t")
    assert s.get_operation_color(head) == "#112233"


def test_engrave_step_uses_head_color():
    """An engraving step reports the head's color."""
    head = MagicMock(spec=LaserHead)
    head.cut_color = "#112233"

    s = EngraveStep(name="t")
    assert s.get_operation_color(head) == "#112233"


def test_laser_step_color_none_for_non_laser_head():
    """A laser step reports no color for a non-laser head."""
    head = SpindleHead()
    s = ContourStep(name="t")
    assert s.get_operation_color(head) is None
