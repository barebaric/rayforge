"""
Tests for the CommandStep model.

The step carries a multi-line machine-code text block; each
non-empty line becomes one raygeo Custom command via the CustomSpec
assembler at the step's workflow position.
"""

from automation.steps import CommandStep
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.custom import CustomSpec

from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry


def _make_step(text: str = "") -> CommandStep:
    step = CommandStep(name="cmd")
    step.command_text = text
    return step


def test_registration():
    assert step_registry.get("CommandStep") is CommandStep


def test_class_flags():
    assert CommandStep.needs_workpieces is False
    assert CommandStep.ASSEMBLER_NAME == "custom"
    assert CommandStep.REQUIRED_MACHINE_CAPS == frozenset()
    assert CommandStep.TYPELABEL


def test_command_lines_skip_empty_lines():
    step = _make_step("M101\n\n   \nG4 P2\n")
    assert step.command_lines == ["M101", "G4 P2"]


def test_set_command_text_notifies():
    step = _make_step()
    events = []

    def on_updated(sender):
        events.append(sender.command_text)

    step.updated.connect(on_updated)
    step.set_command_text("M101")
    step.set_command_text("M101")  # Unchanged: no duplicate signal.
    assert events == ["M101"]


def test_build_compute_payload_uses_custom_spec(machine):
    step = _make_step("M101\nG4 P2")
    part, payload = step.build_command_payload(machine)
    assert part.size_mm == (0.0, 0.0)
    assembler = payload.assembler
    spec = assembler.spec
    assert isinstance(spec, CustomSpec)
    assert spec.lines == ["M101", "G4 P2"]
    assert Assembler(CustomSpec(["M101"])) is not None


def test_get_cache_params_include_text():
    step_a = _make_step("M101")
    step_b = _make_step("M102")
    assert step_a.get_cache_params() != step_b.get_cache_params()
    assert step_a.get_cache_params()["command_text"] == "M101"


def test_dict_round_trip():
    step = _make_step("M101\n; {machine.name}")
    step.visible = False

    restored = Step.from_dict(step.to_dict())
    assert isinstance(restored, CommandStep)
    assert restored.uid == step.uid
    assert restored.command_text == "M101\n; {machine.name}"
    assert restored.visible is False


def test_from_dict_legacy_producer_params():
    data = {
        "uid": "u1",
        "type": "step",
        "step_type": "CommandStep",
        "name": "cmd",
        "matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "typelabel": "Command",
        "visible": True,
        "opsproducer_dict": {"params": {"command_text": "M101"}},
        "children": [],
    }
    restored = Step.from_dict(data)
    assert isinstance(restored, CommandStep)
    assert restored.command_text == "M101"


def test_check_warns_on_empty_text(machine):
    step = _make_step("")
    warnings = step.check(machine)
    assert any("no machine code" in w for w in warnings)


def test_check_warns_on_non_gcode_driver(machine):
    machine.driver_name = "RuidaRPAAdapter"
    step = _make_step("M101")
    warnings = step.check(machine)
    assert any("does not support" in w for w in warnings)


def test_check_no_warnings_for_gcode_machine_with_text(machine):
    step = _make_step("M101")
    assert step.check(machine) == []


def test_get_summary():
    assert _make_step("").get_summary() == "No machine code"
    assert _make_step("M101").get_summary() == "M101"
    assert "2" in _make_step("M101\nM102").get_summary()


def test_show_general_settings_disabled():
    assert _make_step().show_general_settings is False


def test_recipe_varset_exposes_command_text():
    varset = CommandStep.recipe_varset()
    keys = [var.key for var in varset]
    assert "command_text" in keys
    assert CommandStep.recipe_keys() == tuple(keys)


def test_recipe_round_trip_via_set_recipe_value():
    step = _make_step("M101")
    assert step.get_recipe_setter_name("command_text") == "set_command_text"

    step.set_recipe_value("command_text", "M102")

    assert step.command_text == "M102"
