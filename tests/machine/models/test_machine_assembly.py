import math

from rayforge.context import RayforgeContext
from rayforge.core.layer import Layer
from rayforge.machine.assembly import LinkRole
from rayforge.machine.driver.driver import Axis
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryModule
from rayforge.simulator.machine_state import MachineState


def _make_machine():
    ctx = RayforgeContext()
    return Machine(ctx)


def _make_layer(
    rotary_enabled=False, rotary_module_uid=None, rotary_diameter=25.0
):
    layer = Layer("Test")
    layer.rotary_enabled = rotary_enabled
    layer.rotary_module_uid = rotary_module_uid
    layer.rotary_diameter = rotary_diameter
    return layer


class TestAssemblyCaching:
    def test_assembly_cached_on_repeated_access(self):
        machine = _make_machine()
        asm1 = machine.assembly
        asm2 = machine.assembly
        assert asm1 is asm2

    def test_invalidate_forces_rebuild(self):
        machine = _make_machine()
        asm1 = machine.assembly
        machine.invalidate_assembly()
        asm2 = machine.assembly
        assert asm1 is not asm2

    def test_initial_assembly_is_cartesian(self):
        machine = _make_machine()
        assert not machine.assembly.has_rotary


class TestConfigureForLayer:
    def test_configure_none_keeps_cartesian(self):
        machine = _make_machine()
        module = RotaryModule()
        machine.add_rotary_module(module)
        machine.configure_for_layer(None)
        assert not machine.assembly.has_rotary

    def test_configure_layer_with_rotary(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 40.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=40.0,
        )
        machine.configure_for_layer(layer)
        asm = machine.assembly
        assert asm.has_rotary
        assert asm.rotary_diameter == 40.0
        chucks = asm.get_links_by_role(LinkRole.CHUCK)
        assert len(chucks) == 1
        assert chucks[0].driver_axis == Axis.Y

    def test_configure_layer_without_rotary(self):
        machine = _make_machine()
        module = RotaryModule()
        machine.add_rotary_module(module)
        layer = _make_layer(rotary_enabled=False)
        machine.configure_for_layer(layer)
        assert not machine.assembly.has_rotary

    def test_configure_switches_between_layers(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 30.0
        machine.add_rotary_module(module)
        rotary_layer = _make_layer(
            rotary_enabled=True, rotary_module_uid=module.uid
        )
        flat_layer = _make_layer(rotary_enabled=False)

        machine.configure_for_layer(rotary_layer)
        assert machine.assembly.has_rotary

        machine.configure_for_layer(flat_layer)
        assert not machine.assembly.has_rotary

    def test_same_layer_no_rebuild(self):
        machine = _make_machine()
        module = RotaryModule()
        machine.add_rotary_module(module)
        layer = _make_layer(rotary_enabled=True, rotary_module_uid=module.uid)
        machine.configure_for_layer(layer)
        asm1 = machine.assembly

        machine.configure_for_layer(layer)
        asm2 = machine.assembly
        assert asm1 is asm2

    def test_unknown_module_uid_no_rotary(self):
        machine = _make_machine()
        layer = _make_layer(
            rotary_enabled=True, rotary_module_uid="nonexistent"
        )
        machine.configure_for_layer(layer)
        assert not machine.assembly.has_rotary


class TestAssemblyHeadLinks:
    def test_single_head(self):
        machine = _make_machine()
        asm = machine.assembly
        heads = asm.get_links_by_role(LinkRole.HEAD)
        assert len(heads) == 1
        assert heads[0].name == "head_0"

    def test_head_positions(self):
        machine = _make_machine()
        state = MachineState()
        state.axes[Axis.X] = 10.0
        state.axes[Axis.Y] = 20.0
        state.axes[Axis.Z] = 5.0
        positions = machine.assembly.head_positions(state)
        assert "head_0" in positions
        assert positions["head_0"] == (10.0, 20.0, 5.0)


class TestAssemblyInvalidation:
    def test_add_rotary_module_invalidates(self):
        machine = _make_machine()
        asm_before = machine.assembly
        assert not asm_before.has_rotary

        module = RotaryModule()
        module.default_diameter = 30.0
        machine.add_rotary_module(module)

        asm_after = machine.assembly
        assert asm_after.has_rotary
        assert asm_after is not asm_before

    def test_remove_rotary_module_invalidates(self):
        machine = _make_machine()
        module = RotaryModule()
        machine.add_rotary_module(module)
        asm_before = machine.assembly
        assert asm_before.has_rotary

        machine.remove_rotary_module(module)
        asm_after = machine.assembly
        assert not asm_after.has_rotary
        assert asm_after is not asm_before

    def test_rotary_module_change_invalidates_if_mounted(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 30.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=30.0,
        )
        machine.configure_for_layer(layer)
        asm_before = machine.assembly
        assert asm_before.rotary_diameter == 30.0

        layer.rotary_diameter = 50.0
        machine.configure_for_layer(layer)
        asm_after = machine.assembly
        assert asm_after.rotary_diameter == 50.0
        assert asm_after is not asm_before

    def test_rotary_module_change_invalidates(self):
        machine = _make_machine()
        module = RotaryModule()
        machine.add_rotary_module(module)
        machine.configure_for_layer(None)
        asm_before = machine.assembly
        module.set_default_diameter(50.0)
        asm_after = machine.assembly
        assert asm_after is not asm_before

    def test_add_head_invalidates(self):
        machine = _make_machine()
        asm_before = machine.assembly

        machine.add_head(Laser())
        asm_after = machine.assembly
        assert asm_after is not asm_before
        heads = asm_after.get_links_by_role(LinkRole.HEAD)
        assert len(heads) == 2

    def test_remove_head_invalidates(self):
        machine = _make_machine()

        extra = Laser()
        machine.add_head(extra)
        asm_before = machine.assembly
        machine.remove_head(extra)
        asm_after = machine.assembly
        assert asm_after is not asm_before
        heads = asm_after.get_links_by_role(LinkRole.HEAD)
        assert len(heads) == 1


class TestAssemblyRotarySpecs:
    def test_rotary_uses_module_axis(self):
        machine = _make_machine()
        module = RotaryModule()
        module.axis = Axis.A
        module.default_diameter = 25.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=25.0,
        )
        machine.configure_for_layer(layer)
        chucks = machine.assembly.get_links_by_role(LinkRole.CHUCK)
        assert chucks[0].driver_axis == Axis.Y

    def test_rotary_uses_module_model_id(self):
        machine = _make_machine()
        module = RotaryModule()
        module.model_id = "models/chuck.glb"
        module.default_diameter = 25.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=25.0,
        )
        machine.configure_for_layer(layer)
        chucks = machine.assembly.get_links_by_role(LinkRole.CHUCK)
        assert chucks[0].model_id == "models/chuck.glb"

    def test_rotary_uses_module_transform(self):
        import numpy as np

        machine = _make_machine()
        module = RotaryModule()
        module.set_position(10.0, 20.0, 0.0)
        module.default_diameter = 25.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=25.0,
        )
        machine.configure_for_layer(layer)
        bases = [
            link
            for link in machine.assembly._links
            if link.name.startswith("rotary_base_")
        ]
        assert len(bases) == 1
        expected_pos = np.array([10.0, 20.0, 0.0])
        actual_pos = bases[0].local_transform[:3, 3]
        np.testing.assert_array_almost_equal(actual_pos, expected_pos)

    def test_rotary_diameter_from_layer(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 42.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=42.0,
        )
        machine.configure_for_layer(layer)
        state = MachineState()
        state.axes[Axis.Y] = 42.0 * math.pi
        angles = machine.assembly.chuck_angles(state)
        chuck_names = list(angles.keys())
        assert len(chuck_names) == 1
        assert abs(angles[chuck_names[0]] - 2 * math.pi) < 1e-9

    def test_layer_diameter_overrides_module_default(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 30.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=50.0,
        )
        machine.configure_for_layer(layer)
        assert machine.assembly.rotary_diameter == 50.0
        assert machine.assembly.chuck_diameters == {"rotary_chuck_0": 50.0}

    def test_diameter_change_triggers_rebuild(self):
        machine = _make_machine()
        module = RotaryModule()
        module.default_diameter = 30.0
        machine.add_rotary_module(module)
        layer = _make_layer(
            rotary_enabled=True,
            rotary_module_uid=module.uid,
            rotary_diameter=30.0,
        )
        machine.configure_for_layer(layer)
        asm1 = machine.assembly
        assert asm1.rotary_diameter == 30.0

        layer.rotary_diameter = 40.0
        machine.configure_for_layer(layer)
        asm2 = machine.assembly
        assert asm2.rotary_diameter == 40.0
        assert asm2 is not asm1
