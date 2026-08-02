from unittest.mock import MagicMock

from laser_essentials.capabilities import (
    CUT,
    ENGRAVE,
    MATERIAL_TEST,
    SCORE,
    CutCapability,
    EngraveCapability,
    LaserHeadVar,
    MaterialTestCapability,
    ScoreCapability,
)

from rayforge.core.capability import (
    MachineCapability,
    StepCapability,
    _CombinedCapability,
)
from rayforge.core.varset import BoolVar, IntVar, SliderFloatVar, VarSet


def test_cut_capability(mocker):
    """Tests the properties of the CutCapability singleton."""
    mocker.patch("laser_essentials.capabilities.get_context")
    assert CUT.name == "CUT"
    assert CUT.label == "Cut"
    assert isinstance(CUT, CutCapability)
    assert isinstance(CUT, StepCapability)

    varset = CUT.varset
    assert isinstance(varset, VarSet)
    var_keys = [v.key for v in varset]
    assert "power" in var_keys
    assert "cut_speed" in var_keys
    assert "air_assist" in var_keys
    assert "selected_head_uid" in var_keys

    power_var = varset["power"]
    assert isinstance(power_var, SliderFloatVar)
    assert power_var.label == "Power"
    assert power_var.default == 0.8

    air_var = varset["air_assist"]
    assert isinstance(air_var, BoolVar)
    assert air_var.default is False

    laser_var = varset["selected_head_uid"]
    assert isinstance(laser_var, LaserHeadVar)


def test_engrave_capability(mocker):
    """Tests the properties of the EngraveCapability singleton."""
    mocker.patch("laser_essentials.capabilities.get_context")
    assert ENGRAVE.name == "ENGRAVE"
    assert ENGRAVE.label == "Engrave"
    assert isinstance(ENGRAVE, EngraveCapability)

    varset = ENGRAVE.varset
    var_keys = [v.key for v in varset]
    assert "power" in var_keys
    assert "cut_speed" in var_keys
    assert "air_assist" in var_keys
    assert "selected_head_uid" in var_keys

    speed_var = varset["cut_speed"]
    assert isinstance(speed_var, IntVar)
    assert speed_var.label == "Engrave Speed"
    assert speed_var.default == 4000


def test_score_capability(mocker):
    """Tests the properties of the ScoreCapability singleton."""
    mocker.patch("laser_essentials.capabilities.get_context")
    assert SCORE.name == "SCORE"
    assert SCORE.label == "Score"
    assert isinstance(SCORE, ScoreCapability)

    varset = SCORE.varset
    speed_var = varset["cut_speed"]
    assert speed_var.label == "Score Speed"
    assert speed_var.default == 5000


def test_material_test_capability():
    assert isinstance(MATERIAL_TEST, MaterialTestCapability)
    assert MATERIAL_TEST.name == "MATERIAL_TEST"
    var_keys = [v.key for v in MATERIAL_TEST.varset]
    assert "air_assist" in var_keys
    air_var = MATERIAL_TEST.varset["air_assist"]
    assert isinstance(air_var, BoolVar)
    assert air_var.default is False


def test_capability_or_operator():
    combined = CUT | SCORE
    assert isinstance(combined, _CombinedCapability)
    var_keys = [v.key for v in combined.varset]
    assert "power" in var_keys
    assert "selected_head_uid" in var_keys

    triple = CUT | SCORE | ENGRAVE
    triple_keys = [v.key for v in triple.varset]
    assert "power" in triple_keys
    assert "tab_power" in triple_keys


def test_capability_or_right_overrides():
    """Right operand's vars override left for shared keys."""
    combined = CUT | ENGRAVE
    power_var = combined.varset["power"]
    assert power_var.default == 0.2  # ENGRAVE default, not CUT's 0.8


def test_laser_head_var_populates_from_machine(mocker):
    """LaserHeadVar pulls head choices from the active machine."""
    head_a = MagicMock()
    head_a.name = "Head A"
    head_a.uid = "uid-a"
    head_a.machine_capability = MachineCapability.LASER
    head_b = MagicMock()
    head_b.name = "Head B"
    head_b.uid = "uid-b"
    head_b.machine_capability = MachineCapability.LASER
    machine = MagicMock()
    machine.heads = [head_a, head_b]
    mock_context = mocker.patch("laser_essentials.capabilities.get_context")
    mock_context.return_value.machine = machine

    var = LaserHeadVar()
    assert var.choices == ["Head A", "Head B"]
    assert var.get_value_for_display("Head A") == "uid-a"
    assert var.get_display_for_value("uid-b") == "Head B"


def test_laser_head_var_ignores_non_laser_heads(mocker):
    """LaserHeadVar only lists heads with the LASER machine capability."""
    spindle = MagicMock()
    spindle.name = "Spindle"
    spindle.uid = "spindle-1"
    spindle.machine_capability = MachineCapability.MILL
    machine = MagicMock()
    machine.heads = [spindle]
    mock_context = mocker.patch("laser_essentials.capabilities.get_context")
    mock_context.return_value.machine = machine

    var = LaserHeadVar()
    assert var.choices == []
