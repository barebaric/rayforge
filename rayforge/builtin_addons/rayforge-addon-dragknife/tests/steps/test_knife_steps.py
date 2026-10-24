"""Tests for the knife cut steps."""

from unittest.mock import MagicMock

import pytest
from dragknife.steps import DragKnifeCutStep, TangentialKnifeCutStep
from dragknife.steps.knife_cut_step import KnifeCutStep
from raygeo.ops.types import CommandType

from rayforge.core.capability import MachineCapability
from rayforge.machine.models.knife import DragKnifeHead


@pytest.fixture
def context(machine):
    ctx = MagicMock()
    ctx.machine = machine
    return ctx


@pytest.fixture
def machine():
    m = MagicMock()
    knife_head = DragKnifeHead()
    knife_head.uid = "knife-uid"
    knife_head.offset_mm = 0.8
    other_head = MagicMock()
    other_head.machine_capability = MachineCapability.LASER
    other_head.uid = "laser-uid"
    m.heads = [other_head, knife_head]
    return m


@pytest.fixture
def drag_step():
    return DragKnifeCutStep(name="drag")


@pytest.fixture
def tan_step():
    return TangentialKnifeCutStep(name="tan")


class TestDragKnifeCutStep:
    def test_capabilities(self, drag_step):
        assert drag_step.REQUIRED_MACHINE_CAPS == frozenset(
            {MachineCapability.DRAG_KNIFE}
        )
        assert drag_step.ASSEMBLER_NAME == "drag_knife"
        assert drag_step.TYPELABEL

    def test_setters_update_attribute_and_signal(self, drag_step):
        for attr, value in [
            ("offset_mm", 1.2),
            ("safe_z", 5.0),
            ("swivel_angle_deg", 90.0),
        ]:
            handler = MagicMock()
            drag_step.updated.connect(handler)
            getattr(drag_step, f"set_{attr}")(value)
            assert getattr(drag_step, attr) == value
            handler.assert_called_once_with(drag_step)

    def test_setters_no_signal_on_same_value(self, drag_step):
        for attr in ("offset_mm", "safe_z", "swivel_angle_deg"):
            handler = MagicMock()
            drag_step.updated.connect(handler)
            getattr(drag_step, f"set_{attr}")(getattr(drag_step, attr))
            handler.assert_not_called()

    def test_setter_syncs_transformer_dict(self, drag_step):
        drag_step.set_offset_mm(1.5)
        t_dict = drag_step.per_workpiece_transformers_dicts[0]
        assert t_dict["name"] == "DragKnifeTransformer"
        assert t_dict["offset_mm"] == pytest.approx(1.5)
        drag_step.set_swivel_angle_deg(120.0)
        t_dict = drag_step.per_workpiece_transformers_dicts[0]
        assert t_dict["swivel_angle_deg"] == pytest.approx(120.0)

    def test_serialization_roundtrip(self, drag_step):
        drag_step.offset_mm = 0.7
        drag_step.swivel_angle_deg = 60.0
        data = drag_step.to_dict()
        restored = DragKnifeCutStep.from_dict(data)
        assert restored.offset_mm == pytest.approx(0.7)
        assert restored.swivel_angle_deg == pytest.approx(60.0)
        assert restored.name == drag_step.name

    def test_create_uses_knife_head(self, context):
        step = DragKnifeCutStep.create(context)
        assert step.selected_head_uid == "knife-uid"
        assert step.offset_mm == pytest.approx(0.8)
        t_dict = step.per_workpiece_transformers_dicts[0]
        assert t_dict["offset_mm"] == pytest.approx(0.8)

    def test_all_recipe_keys_have_setters(self, drag_step):
        for var in DragKnifeCutStep.recipe_varset():
            assert hasattr(drag_step, f"set_{var.key}"), var.key


class TestTangentialKnifeCutStep:
    def test_capabilities(self, tan_step):
        assert tan_step.REQUIRED_MACHINE_CAPS == frozenset(
            {MachineCapability.TANGENTIAL_KNIFE}
        )
        assert tan_step.ASSEMBLER_NAME == "tangential_knife"
        assert tan_step.TYPELABEL

    def test_setter_syncs_transformer_dict(self, tan_step):
        tan_step.set_angle_tolerance_deg(15.0)
        tan_step.set_radius_tolerance_mm(2.0)
        tan_step.set_safe_z(4.0)
        t_dict = tan_step.per_workpiece_transformers_dicts[0]
        assert t_dict["name"] == "TangentialKnifeTransformer"
        assert t_dict["angle_tolerance_deg"] == pytest.approx(15.0)
        assert t_dict["radius_tolerance_mm"] == pytest.approx(2.0)
        assert t_dict["safe_z"] == pytest.approx(4.0)

    def test_serialization_roundtrip(self, tan_step):
        tan_step.angle_tolerance_deg = 20.0
        tan_step.radius_tolerance_mm = 3.0
        tan_step.spindle_rpm = 8000
        data = tan_step.to_dict()
        restored = TangentialKnifeCutStep.from_dict(data)
        assert restored.angle_tolerance_deg == pytest.approx(20.0)
        assert restored.radius_tolerance_mm == pytest.approx(3.0)
        assert restored.spindle_rpm == 8000

    def test_create_initial_ops_sets_spindle(self, tan_step):
        ops = tan_step.create_initial_ops()
        spindle_cmds = [
            i
            for i in range(ops.len())
            if ops.command_type(i) == CommandType.SET_SPINDLE_RPM
        ]
        assert len(spindle_cmds) == 1

    def test_all_recipe_keys_have_setters(self, tan_step):
        for var in TangentialKnifeCutStep.recipe_varset():
            assert hasattr(tan_step, f"set_{var.key}"), var.key


class TestKnifeCutStepBase:
    def test_steps_share_knife_base(self):
        assert issubclass(DragKnifeCutStep, KnifeCutStep)
        assert issubclass(TangentialKnifeCutStep, KnifeCutStep)
