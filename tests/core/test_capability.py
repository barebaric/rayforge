import pytest

from rayforge.core.capability import (
    MachineCapability,
    StepCapability,
    _CombinedCapability,
)
from rayforge.core.capability_registry import step_capability_registry
from rayforge.core.varset import BoolVar, VarSet


class _TestCapability(StepCapability):
    """A concrete capability used to exercise the core Capability ABC."""

    def __init__(self, name: str, label: str, varset: VarSet):
        self._name = name
        self._label = label
        self._varset = varset

    @property
    def name(self) -> str:
        return self._name

    @property
    def label(self) -> str:
        return self._label

    @property
    def varset(self) -> VarSet:
        return self._varset


def _make_cap(name: str, label: str, defaults: dict) -> _TestCapability:
    varset = VarSet(
        vars=[
            BoolVar(key=key, label=label, default=default)
            for key, default in defaults.items()
        ]
    )
    return _TestCapability(name, label, varset)


CUT = _make_cap("CUT", "Cut", {"power": True, "cut_speed": True})
SCORE = _make_cap("SCORE", "Score", {"power": True, "kerf": False})


class TestMachineCapability:
    def test_labels(self):
        assert MachineCapability.LASER.label == "Laser"
        assert MachineCapability.MILL.label == "Mill"

    def test_descriptions(self):
        assert "laser" in MachineCapability.LASER.description
        assert "spindle" in MachineCapability.MILL.description


class TestCapability:
    def test_abstract(self):
        with pytest.raises(TypeError):
            StepCapability()  # type: ignore[abstract]

    def test_properties(self):
        assert CUT.name == "CUT"
        assert CUT.label == "Cut"
        assert str(CUT) == "Cut"

    def test_icon_name_defaults_to_name(self):
        assert CUT.icon_name == "cut-symbolic"
        assert SCORE.icon_name == "score-symbolic"

    def test_icon_name_is_overridable(self):
        class CustomIcon(_TestCapability):
            @property
            def icon_name(self) -> str:
                return "custom-icon"

        cap = CustomIcon("CUSTOM", "Custom", VarSet(vars=[]))
        assert cap.icon_name == "custom-icon"

    def test_get_setting_keys(self):
        assert CUT.get_setting_keys() == ["power", "cut_speed"]

    def test_or_operator(self):
        combined = CUT | SCORE
        assert isinstance(combined, _CombinedCapability)
        assert combined.name == "CUT|SCORE"
        var_keys = [v.key for v in combined.varset]
        assert "power" in var_keys
        assert "cut_speed" in var_keys
        assert "kerf" in var_keys

    def test_or_operator_flattens(self):
        triple = CUT | SCORE | SCORE
        assert triple.name == "CUT|SCORE|SCORE"
        var_keys = [v.key for v in triple.varset]
        assert "power" in var_keys

    def test_or_operator_non_capability(self):
        with pytest.raises(TypeError):
            _ = CUT | "not-a-capability"  # type: ignore[operator]


class TestCapabilityRegistry:
    def test_register_and_get(self):
        cap = _make_cap("TEST", "Test", {"value": True})
        step_capability_registry.register(cap, addon_name="test_addon")
        try:
            assert step_capability_registry.get("TEST") is cap
        finally:
            step_capability_registry.unregister("TEST")

    def test_get_unknown_returns_none(self):
        assert step_capability_registry.get("UNREGISTERED_CAP") is None

    def test_all_capabilities_returns_registration_order(self):
        cap_a = _make_cap("CAP_A", "A", {})
        cap_b = _make_cap("CAP_B", "B", {})
        step_capability_registry.register(cap_a, addon_name="test_addon")
        step_capability_registry.register(cap_b, addon_name="test_addon")
        try:
            all_caps = step_capability_registry.all_capabilities()
            assert all_caps[-2:] == [cap_a, cap_b]
        finally:
            step_capability_registry.unregister("CAP_A")
            step_capability_registry.unregister("CAP_B")

    def test_register_overwrites_same_name(self):
        cap_a = _make_cap("SAME", "A", {})
        cap_b = _make_cap("SAME", "B", {})
        step_capability_registry.register(cap_a, addon_name="test_addon")
        step_capability_registry.register(cap_b, addon_name="test_addon")
        try:
            assert step_capability_registry.get("SAME") is cap_b
        finally:
            step_capability_registry.unregister("SAME")

    def test_unregister_unknown_returns_false(self):
        assert step_capability_registry.unregister("NEVER_REGISTERED") is False

    def test_unregister_all_from_addon(self):
        cap_a = _make_cap("ADDON_A", "A", {})
        cap_b = _make_cap("ADDON_B", "B", {})
        step_capability_registry.register(cap_a, addon_name="addon_1")
        step_capability_registry.register(cap_b, addon_name="addon_2")
        try:
            count = step_capability_registry.unregister_all_from_addon(
                "addon_1"
            )
            assert count == 1
            assert step_capability_registry.get("ADDON_A") is None
            assert step_capability_registry.get("ADDON_B") is cap_b
        finally:
            step_capability_registry.unregister("ADDON_B")

    def test_unregister_all_from_unknown_addon_returns_zero(self):
        assert (
            step_capability_registry.unregister_all_from_addon("no_such") == 0
        )


@pytest.mark.asyncio
async def test_registry_populated_via_addon_hook(context_initializer):
    """After addons load, the registry holds all domain capabilities."""
    names = {c.name for c in step_capability_registry.all_capabilities()}
    assert {
        "CUT",
        "ENGRAVE",
        "SCORE",
        "WITH_KERF",
        "MATERIAL_TEST",
        "MILL",
    } <= names
    cut_cap = step_capability_registry.get("CUT")
    assert cut_cap is not None
    assert cut_cap.label == "Cut"
    assert step_capability_registry.get("UNREGISTERED") is None
