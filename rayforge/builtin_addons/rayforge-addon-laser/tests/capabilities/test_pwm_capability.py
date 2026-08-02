"""Tests for the laser addon's PWMCapability."""

from laser_essentials.capabilities import PWMCapability

from rayforge.core.capability import Capability
from rayforge.core.varset import IntVar, VarSet
from rayforge.machine.driver.driver import PWMParams


def test_construction():
    cap = PWMCapability(
        PWMParams(
            frequency=1000,
            max_frequency=5000,
            pulse_width=50,
            min_pulse_width=1,
            max_pulse_width=100,
        )
    )
    assert cap.name == "PWM"
    assert cap.label == "PWM"
    assert isinstance(cap, Capability)


def test_varset_keys():
    cap = PWMCapability(PWMParams(1000, 5000, 50, 1, 100))
    varset = cap.varset
    assert isinstance(varset, VarSet)
    keys = [v.key for v in varset]
    assert "frequency" in keys
    assert "pulse_width" in keys


def test_varset_defaults():
    cap = PWMCapability(PWMParams(1000, 5000, 50, 1, 100))
    varset = cap.varset
    freq_var = varset["frequency"]
    assert isinstance(freq_var, IntVar)
    assert freq_var.default == 1000
    pw_var = varset["pulse_width"]
    assert isinstance(pw_var, IntVar)
    assert pw_var.default == 50


def test_varset_bounds():
    cap = PWMCapability(PWMParams(1000, 5000, 50, 1, 100))
    varset = cap.varset
    freq_var = varset["frequency"]
    assert isinstance(freq_var, IntVar)
    assert freq_var.min_val == 1
    assert freq_var.max_val == 5000
    pw_var = varset["pulse_width"]
    assert isinstance(pw_var, IntVar)
    assert pw_var.min_val == 1
    assert pw_var.max_val == 100
