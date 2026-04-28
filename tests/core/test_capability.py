from rayforge.core.varset import VarSet, SliderFloatVar, IntVar, BoolVar
from rayforge.core.capability import (
    Capability,
    CutCapability,
    EngraveCapability,
    KerfCapability,
    MaterialTestCapability,
    LaserHeadVar,
    PWMCapability,
    ScoreCapability,
    _CombinedCapability,
    CUT,
    ENGRAVE,
    SCORE,
    WITH_KERF,
    MATERIAL_TEST,
    ALL_CAPABILITIES,
    CAPABILITIES_BY_NAME,
)


def test_cut_capability(mocker):
    """Tests the properties of the CutCapability singleton."""
    mocker.patch("rayforge.core.capability.get_context")
    assert CUT.name == "CUT"
    assert CUT.label == "Cut"
    assert isinstance(CUT, CutCapability)
    assert isinstance(CUT, Capability)

    varset = CUT.varset
    assert isinstance(varset, VarSet)
    var_keys = [v.key for v in varset]
    assert "power" in var_keys
    assert "cut_speed" in var_keys
    assert "air_assist" in var_keys
    assert "selected_laser_uid" in var_keys

    power_var = varset["power"]
    assert isinstance(power_var, SliderFloatVar)
    assert power_var.label == "Power"
    assert power_var.default == 0.8

    air_var = varset["air_assist"]
    assert isinstance(air_var, BoolVar)
    assert air_var.default is False

    laser_var = varset["selected_laser_uid"]
    assert isinstance(laser_var, LaserHeadVar)


def test_engrave_capability(mocker):
    """Tests the properties of the EngraveCapability singleton."""
    mocker.patch("rayforge.core.capability.get_context")
    assert ENGRAVE.name == "ENGRAVE"
    assert ENGRAVE.label == "Engrave"
    assert isinstance(ENGRAVE, EngraveCapability)

    varset = ENGRAVE.varset
    var_keys = [v.key for v in varset]
    assert "power" in var_keys
    assert "cut_speed" in var_keys
    assert "air_assist" in var_keys
    assert "selected_laser_uid" in var_keys

    speed_var = varset["cut_speed"]
    assert isinstance(speed_var, IntVar)
    assert speed_var.label == "Engrave Speed"
    assert speed_var.default == 4000


def test_score_capability(mocker):
    """Tests the properties of the ScoreCapability singleton."""
    mocker.patch("rayforge.core.capability.get_context")
    assert SCORE.name == "SCORE"
    assert SCORE.label == "Score"
    assert isinstance(SCORE, ScoreCapability)

    varset = SCORE.varset
    speed_var = varset["cut_speed"]
    assert speed_var.label == "Score Speed"
    assert speed_var.default == 5000


def test_collections():
    """Tests the global collections of capabilities."""
    assert len(ALL_CAPABILITIES) == 5
    assert CUT in ALL_CAPABILITIES
    assert ENGRAVE in ALL_CAPABILITIES
    assert SCORE in ALL_CAPABILITIES
    assert WITH_KERF in ALL_CAPABILITIES
    assert MATERIAL_TEST in ALL_CAPABILITIES

    assert len(CAPABILITIES_BY_NAME) == 5
    assert CAPABILITIES_BY_NAME["CUT"] is CUT
    assert CAPABILITIES_BY_NAME["ENGRAVE"] is ENGRAVE
    assert CAPABILITIES_BY_NAME["SCORE"] is SCORE
    assert CAPABILITIES_BY_NAME["WITH_KERF"] is WITH_KERF
    assert CAPABILITIES_BY_NAME["MATERIAL_TEST"] is MATERIAL_TEST


def test_kerf_capability():
    varset = WITH_KERF.varset
    assert isinstance(WITH_KERF, KerfCapability)
    var_keys = [v.key for v in varset]
    assert "kerf_mm" in var_keys
    assert "power" not in var_keys
    assert "cut_speed" not in var_keys


def test_material_test_capability():
    assert isinstance(MATERIAL_TEST, MaterialTestCapability)
    assert MATERIAL_TEST.name == "MATERIAL_TEST"
    assert list(MATERIAL_TEST.varset.vars) == []


def test_capability_or_operator():
    combined = CUT | WITH_KERF
    assert isinstance(combined, _CombinedCapability)
    var_keys = [v.key for v in combined.varset]
    assert "power" in var_keys
    assert "kerf_mm" in var_keys
    assert "selected_laser_uid" in var_keys

    triple = CUT | SCORE | WITH_KERF
    triple_keys = [v.key for v in triple.varset]
    assert "power" in triple_keys
    assert "kerf_mm" in triple_keys
    assert "tab_power" in triple_keys


def test_capability_or_right_overrides():
    """Right operand's vars override left for shared keys."""
    combined = CUT | ENGRAVE
    power_var = combined.varset["power"]
    assert power_var.default == 0.2  # ENGRAVE default, not CUT's 0.8


class TestPWMCapability:
    def test_construction(self):
        cap = PWMCapability(
            frequency=1000,
            max_frequency=5000,
            pulse_width=50,
            min_pulse_width=1,
            max_pulse_width=100,
        )
        assert cap.name == "PWM"
        assert cap.label == "PWM"
        assert isinstance(cap, Capability)

    def test_varset_keys(self):
        cap = PWMCapability(1000, 5000, 50, 1, 100)
        varset = cap.varset
        assert isinstance(varset, VarSet)
        keys = [v.key for v in varset]
        assert "frequency" in keys
        assert "pulse_width" in keys

    def test_varset_defaults(self):
        cap = PWMCapability(1000, 5000, 50, 1, 100)
        varset = cap.varset
        freq_var = varset["frequency"]
        assert isinstance(freq_var, IntVar)
        assert freq_var.default == 1000
        pw_var = varset["pulse_width"]
        assert isinstance(pw_var, IntVar)
        assert pw_var.default == 50

    def test_varset_bounds(self):
        cap = PWMCapability(1000, 5000, 50, 1, 100)
        varset = cap.varset
        freq_var = varset["frequency"]
        assert isinstance(freq_var, IntVar)
        assert freq_var.min_val == 1
        assert freq_var.max_val == 5000
        pw_var = varset["pulse_width"]
        assert isinstance(pw_var, IntVar)
        assert pw_var.min_val == 1
        assert pw_var.max_val == 100

    def test_not_singleton(self):
        cap1 = PWMCapability(100, 1000, 10, 1, 50)
        cap2 = PWMCapability(2000, 5000, 50, 1, 100)
        assert cap1 is not cap2
        assert cap1.varset["frequency"].default == 100
        assert cap2.varset["frequency"].default == 2000
