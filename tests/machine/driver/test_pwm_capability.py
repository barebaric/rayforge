"""Tests for the core PWM settings varset."""

from rayforge.core.varset import IntVar, VarSet
from rayforge.machine.driver.driver import PWMParams, pwm_varset


def test_construction():
    vs = pwm_varset(
        PWMParams(
            frequency=1000,
            max_frequency=5000,
            pulse_width=50,
            min_pulse_width=1,
            max_pulse_width=100,
        )
    )
    assert isinstance(vs, VarSet)
    keys = [v.key for v in vs]
    assert "frequency" in keys
    assert "pulse_width" in keys


def test_defaults():
    vs = pwm_varset(PWMParams(1000, 5000, 50, 1, 100))
    freq_var = vs["frequency"]
    assert isinstance(freq_var, IntVar)
    assert freq_var.default == 1000
    pulse_var = vs["pulse_width"]
    assert isinstance(pulse_var, IntVar)
    assert pulse_var.default == 50


def test_bounds():
    vs = pwm_varset(PWMParams(1000, 5000, 50, 1, 100))
    freq_var = vs["frequency"]
    assert isinstance(freq_var, IntVar)
    assert freq_var.min_val == 1
    assert freq_var.max_val == 5000
    pulse_var = vs["pulse_width"]
    assert isinstance(pulse_var, IntVar)
    assert pulse_var.min_val == 1
    assert pulse_var.max_val == 100
