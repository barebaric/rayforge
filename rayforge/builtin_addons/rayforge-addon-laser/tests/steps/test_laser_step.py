"""Tests for the LaserStep domain base class."""

from unittest.mock import MagicMock

from laser_essentials.steps import ContourStep, EngraveStep, LaserStep

from rayforge.core.step import Step
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.spindle import SpindleHead


def test_contour_defaults_preserved():
    s = ContourStep(name="t")
    assert s.power == 0.8, s.power
    assert s.kerf_mm == 0.1, s.kerf_mm
    assert s.cut_speed == 500, s.cut_speed
    assert s.air_assist is False
    assert isinstance(s, LaserStep)


def test_engrave_defaults():
    s = EngraveStep(name="t")
    assert s.power == 0.2, s.power
    assert s.cut_speed == 4000, s.cut_speed
    assert isinstance(s, LaserStep)


def test_laser_step_serialization_roundtrip():
    s = ContourStep(name="t")
    s.power = 0.6
    s.kerf_mm = 0.2
    s.air_assist = True
    s.frequency = 2000
    s.pulse_width = 100
    data = s.to_dict()

    r = Step.from_dict(data)
    assert type(r) is ContourStep
    assert r.power == 0.6
    assert r.kerf_mm == 0.2
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
