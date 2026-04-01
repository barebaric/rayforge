import numpy as np
import pytest
from numpy.testing import assert_array_equal

from rayforge.machine.driver.driver import Axis
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryModule


@pytest.mark.usefixtures("lite_context")
class TestRotaryModule:
    def test_initialization(self):
        rm = RotaryModule()
        assert rm.uid is not None
        assert rm.name == "Rotary Module"
        assert rm.axis == Axis.A
        assert rm.default_diameter == 25.0
        assert rm.model_id is None
        assert_array_equal(rm.transform, np.eye(4))
        assert rm.extra == {}

    def test_setters_emit_changed(self):
        rm = RotaryModule()
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)

        rm.set_name("My Rotary")
        assert rm.name == "My Rotary"
        assert len(signals) == 1

        rm.set_axis(Axis.B)
        assert rm.axis == Axis.B
        assert len(signals) == 2

        rm.set_position(10, 20, 30)
        assert rm.transform[0, 3] == 10.0
        assert rm.transform[1, 3] == 20.0
        assert rm.transform[2, 3] == 30.0
        assert len(signals) == 3

        rm.set_model_id("rotary/test.glb")
        assert rm.model_id == "rotary/test.glb"
        assert len(signals) == 4

        rm.set_default_diameter(50.0)
        assert rm.default_diameter == 50.0
        assert len(signals) == 5

    def test_set_position_no_signal_if_same(self):
        rm = RotaryModule()
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)
        rm.set_position(0, 0, 0)
        assert len(signals) == 0

    def test_set_axis_no_signal_if_same(self):
        rm = RotaryModule()
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)
        rm.set_axis(Axis.A)
        assert len(signals) == 0

    def test_set_model_id_no_signal_if_same(self):
        rm = RotaryModule()
        rm.model_id = "rotary/test.glb"
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)
        rm.set_model_id("rotary/test.glb")
        assert len(signals) == 0

    def test_set_model_id_to_none_no_signal_if_already_none(self):
        rm = RotaryModule()
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)
        rm.set_model_id(None)
        assert len(signals) == 0

    def test_serialization_roundtrip(self):
        rm = RotaryModule()
        rm.name = "Test Rotary"
        rm.axis = Axis.B
        rm.set_position(10.0, 20.0, 5.0)
        rm.set_model_id("rotary/standard.glb")

        data = rm.to_dict()
        rm2 = RotaryModule.from_dict(data)

        assert rm2.uid == rm.uid
        assert rm2.name == "Test Rotary"
        assert rm2.axis == Axis.B
        assert_array_equal(rm2.transform, rm.transform)
        assert rm2.model_id == "rotary/standard.glb"

    def test_deserialization_defaults(self):
        data = {"uid": "test-uid"}
        rm = RotaryModule.from_dict(data)
        assert rm.uid == "test-uid"
        assert rm.axis == Axis.A
        assert rm.model_id is None
        assert_array_equal(rm.transform, np.eye(4))

    def test_deserialization_extra_keys_preserved(self):
        data = {"uid": "test", "future_field": "value"}
        rm = RotaryModule.from_dict(data)
        assert rm.extra["future_field"] == "value"
        assert "future_field" in rm.to_dict()

    def test_deserialization_legacy_position(self):
        data = {"uid": "test", "x": 10.0, "y": 20.0, "z": 5.0}
        rm = RotaryModule.from_dict(data)
        assert rm.transform[0, 3] == 10.0
        assert rm.transform[1, 3] == 20.0
        assert rm.transform[2, 3] == 5.0

    def test_deserialization_legacy_model_path(self):
        data = {"uid": "test", "model_path": "rotary/old.glb"}
        rm = RotaryModule.from_dict(data)
        assert rm.model_id == "rotary/old.glb"

    def test_deserialization_legacy_keys_not_in_extra(self):
        data = {
            "uid": "test",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "length": 300.0,
            "chuck_diameter": 50.0,
            "model_path": "test.glb",
            "viz_mode": "parametric",
            "unknown_future": 42,
        }
        rm = RotaryModule.from_dict(data)
        assert "unknown_future" not in rm.__dict__
        assert rm.extra["unknown_future"] == 42
        assert "x" not in rm.extra
        assert "length" not in rm.extra
        assert "model_path" not in rm.extra

    def test_transform_serialization(self):
        rm = RotaryModule()
        rm.transform[0, 3] = 10.0
        rm.transform[1, 3] = 20.0
        rm.transform[2, 3] = 30.0
        rm.transform[0, 0] = 2.0

        data = rm.to_dict()
        assert len(data["transform"]) == 16

        rm2 = RotaryModule.from_dict(data)
        assert_array_equal(rm2.transform, rm.transform)

    def test_pickling(self):
        rm = RotaryModule()
        rm.set_position(10, 20, 30)

        state = rm.__getstate__()
        assert "changed" not in state

        rm2 = RotaryModule()
        rm2.__setstate__(state)
        assert_array_equal(rm2.transform, rm.transform)
        assert hasattr(rm2, "changed")

    def test_collision_bbox_returns_none(self):
        rm = RotaryModule()
        assert rm.get_collision_bbox() is None

    def test_collision_bbox_returns_none_with_model(self):
        rm = RotaryModule()
        rm.set_model_id("rotary/test.glb")
        assert rm.get_collision_bbox() is None


@pytest.mark.usefixtures("lite_context")
class TestMachineRotaryModules:
    def test_machine_initially_has_no_rotary_modules(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine.rotary_modules == {}

    def test_add_rotary_module(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        assert rm.uid in machine.rotary_modules

    def test_add_multiple_rotary_modules(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm1 = RotaryModule()
        rm2 = RotaryModule()
        machine.add_rotary_module(rm1)
        machine.add_rotary_module(rm2)
        assert len(machine.rotary_modules) == 2
        assert rm1.uid in machine.rotary_modules
        assert rm2.uid in machine.rotary_modules

    def test_remove_rotary_module(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        machine.remove_rotary_module(rm)
        assert machine.rotary_modules == {}

    def test_get_rotary_module_by_uid(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        assert machine.get_rotary_module_by_uid(rm.uid) is rm
        assert machine.get_rotary_module_by_uid("nonexistent") is None

    def test_get_default_rotary_module_returns_none_initially(
        self, lite_context
    ):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine.get_default_rotary_module() is None

    def test_get_default_rotary_module_returns_set_default(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        machine.set_default_rotary_module_uid(rm.uid)
        assert machine.get_default_rotary_module() is rm

    def test_set_default_rotary_module_uid_no_signal_if_same(
        self, lite_context
    ):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        machine.set_default_rotary_module_uid(rm.uid)

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        machine.changed.connect(on_changed)
        machine.set_default_rotary_module_uid(rm.uid)
        assert len(signals) == 0

    def test_rotary_module_changed_relays_to_machine(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        machine.changed.connect(on_changed)
        rm.set_name("New Name")
        assert len(signals) == 1

    def test_removing_module_disconnects_signal(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        machine.remove_rotary_module(rm)

        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        machine.changed.connect(on_changed)
        rm.set_name("Should not emit")
        assert len(signals) == 0

    def test_removing_default_picks_another(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm1 = RotaryModule()
        rm2 = RotaryModule()
        machine.add_rotary_module(rm1)
        machine.add_rotary_module(rm2)
        machine.set_default_rotary_module_uid(rm1.uid)

        machine.remove_rotary_module(rm1)
        assert machine.default_rotary_module_uid == rm2.uid
        assert machine.get_default_rotary_module() is rm2

    def test_removing_last_default_clears(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        machine.add_rotary_module(rm)
        machine.set_default_rotary_module_uid(rm.uid)

        machine.remove_rotary_module(rm)
        assert machine.default_rotary_module_uid is None
        assert machine.get_default_rotary_module() is None

    def test_serialization_roundtrip(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        rm.name = "My Module"
        rm.set_position(10, 20, 5)
        machine.add_rotary_module(rm)

        data = machine.to_dict()
        machine2 = Machine.from_dict(data)

        assert len(machine2.rotary_modules) == 1
        rm2 = machine2.rotary_modules[rm.uid]
        assert rm2.name == "My Module"
        assert rm2.transform[0, 3] == 10.0
        assert rm2.transform[1, 3] == 20.0
        assert rm2.transform[2, 3] == 5.0

    def test_serialization_without_modules(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        data = machine.to_dict()
        assert data["machine"]["rotary_modules"] == []

        machine2 = Machine.from_dict(data)
        assert machine2.rotary_modules == {}

    def test_serialization_multiple_modules(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm1 = RotaryModule()
        rm1.name = "Module A"
        rm2 = RotaryModule()
        rm2.name = "Module B"
        machine.add_rotary_module(rm1)
        machine.add_rotary_module(rm2)

        data = machine.to_dict()
        machine2 = Machine.from_dict(data)

        assert len(machine2.rotary_modules) == 2
        assert machine2.rotary_modules[rm1.uid].name == "Module A"
        assert machine2.rotary_modules[rm2.uid].name == "Module B"

    def test_serialization_roundtrip_with_model_id(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        rm.name = "Chuck Module"
        rm.set_model_id("rotary/custom.glb")
        machine.add_rotary_module(rm)

        data = machine.to_dict()
        machine2 = Machine.from_dict(data)

        assert len(machine2.rotary_modules) == 1
        rm2 = machine2.rotary_modules[rm.uid]
        assert rm2.name == "Chuck Module"
        assert rm2.model_id == "rotary/custom.glb"

    def test_serialization_model_id_none(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        rm = RotaryModule()
        rm.model_id = None
        machine.add_rotary_module(rm)

        data = machine.to_dict()
        assert data["machine"]["rotary_modules"][0]["model_id"] is None

        machine2 = Machine.from_dict(data)
        rm2 = list(machine2.rotary_modules.values())[0]
        assert rm2.model_id is None
