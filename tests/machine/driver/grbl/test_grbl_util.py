import pytest

from rayforge.core.varset import Var, VarSet
from rayforge.machine.driver.grbl.grbl_util import (
    apply_setting_to_varset,
    get_grbl_setting_varsets,
)


class TestGrblHALMaskSettings:
    """Regression tests for grblHAL bitmask settings (#401).

    GrblHAL redefines $4 (step enable invert) and $5 (limit pins
    invert) as per-axis masks, so values like 15 are legitimate.
    Classic Grbl only reports 0/1, which coerce cleanly to int too.
    """

    def test_stepper_enable_var_is_int_mask(self):
        varsets = get_grbl_setting_varsets()
        var = next(v for vs in varsets for v in vs if v.key == "4")
        assert var.var_type is int

    def test_limit_pins_var_is_int_mask(self):
        varsets = get_grbl_setting_varsets()
        var = next(v for vs in varsets for v in vs if v.key == "5")
        assert var.var_type is int

    def test_mask_value_from_device_is_accepted(self):
        varsets = get_grbl_setting_varsets()
        stepper_varset = next(
            vs
            for vs in varsets
            if "4" in vs.keys()  # noqa: SIM118
        )
        apply_setting_to_varset(stepper_varset, "4", "15")
        assert stepper_varset["4"].value == 15
        assert stepper_varset["4"].var_type is int

    def test_boolean_value_still_coerces(self):
        varsets = get_grbl_setting_varsets()
        stepper_varset = next(
            vs
            for vs in varsets
            if "4" in vs.keys()  # noqa: SIM118
        )
        apply_setting_to_varset(stepper_varset, "4", "1")
        assert stepper_varset["4"].value == 1


class TestApplySettingToVarset:
    """Tests for the apply_setting_to_varset fallback behavior."""

    def _make_varset(self, var_type) -> VarSet:
        return VarSet(vars=[Var(key="42", label="$42", var_type=var_type)])

    def test_assigns_coercible_value(self):
        varset = self._make_varset(int)
        apply_setting_to_varset(varset, "42", "7")
        assert varset["42"].value == 7
        assert varset["42"].var_type is int

    def test_widens_to_str_on_incompatible_value(self):
        varset = self._make_varset(bool)
        apply_setting_to_varset(varset, "42", "15")
        assert varset["42"].value == "15"
        assert varset["42"].var_type is str

    def test_raises_for_unknown_key(self):
        varset = self._make_varset(int)
        with pytest.raises(KeyError):
            apply_setting_to_varset(varset, "99", "1")
