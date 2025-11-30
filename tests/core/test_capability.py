from rayforge.core.varset import VarSet, SliderFloatVar, IntVar, BoolVar
from rayforge.core.capability import (
    Capability,
    CutCapability,
    EngraveCapability,
    LaserHeadVar,
    ScoreCapability,
    CUT,
    ENGRAVE,
    SCORE,
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
    assert air_var.default is True

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
    assert len(ALL_CAPABILITIES) == 3
    assert CUT in ALL_CAPABILITIES
    assert ENGRAVE in ALL_CAPABILITIES
    assert SCORE in ALL_CAPABILITIES

    assert len(CAPABILITIES_BY_NAME) == 3
    assert CAPABILITIES_BY_NAME["CUT"] is CUT
    assert CAPABILITIES_BY_NAME["ENGRAVE"] is ENGRAVE
    assert CAPABILITIES_BY_NAME["SCORE"] is SCORE
