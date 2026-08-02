from cnc_essentials.capabilities import MILL, MillCapability

from rayforge.core.capability import StepCapability
from rayforge.core.varset import FloatVar, IntVar, SpeedVar, VarSet


def test_mill_capability():
    """Tests the properties of the MillCapability singleton."""
    assert MILL.name == "MILL"
    assert MILL.label == "Mill"
    assert isinstance(MILL, MillCapability)
    assert isinstance(MILL, StepCapability)

    varset = MILL.varset
    assert isinstance(varset, VarSet)
    var_keys = [v.key for v in varset]
    assert "tool_diameter" in var_keys
    assert "spindle_rpm" in var_keys
    assert "cut_speed" in var_keys
    assert "plunge_speed" in var_keys
    assert "travel_speed" in var_keys
    assert "target_depth" in var_keys
    assert "depth_per_pass" in var_keys
    assert "safe_z" in var_keys

    assert isinstance(varset["tool_diameter"], FloatVar)
    assert isinstance(varset["spindle_rpm"], IntVar)
    assert isinstance(varset["cut_speed"], SpeedVar)
