import pytest

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
        assert rm.x == 0.0
        assert rm.y == 0.0
        assert rm.z == 0.0
        assert rm.length == 200.0
        assert rm.chuck_diameter == 40.0
        assert rm.tailstock_diameter == 20.0
        assert rm.max_height == 60.0
        assert rm.default_diameter == 25.0

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
        assert rm.x == 10.0
        assert rm.y == 20.0
        assert rm.z == 30.0
        assert len(signals) == 3

        rm.set_length(300)
        assert rm.length == 300
        assert len(signals) == 4

        rm.set_chuck_diameter(50)
        assert rm.chuck_diameter == 50
        assert len(signals) == 5

        rm.set_tailstock_diameter(25)
        assert rm.tailstock_diameter == 25
        assert len(signals) == 6

        rm.set_max_height(80)
        assert rm.max_height == 80
        assert len(signals) == 7

    def test_set_axis_no_signal_if_same(self):
        rm = RotaryModule()
        signals = []

        def on_changed(sender, **kwargs):
            signals.append(sender)

        rm.changed.connect(on_changed)
        rm.set_axis(Axis.A)
        assert len(signals) == 0

    def test_serialization_roundtrip(self):
        rm = RotaryModule()
        rm.name = "Test Rotary"
        rm.axis = Axis.B
        rm.x = 10.0
        rm.y = 20.0
        rm.z = 5.0
        rm.length = 300.0
        rm.chuck_diameter = 50.0
        rm.tailstock_diameter = 30.0
        rm.max_height = 80.0

        data = rm.to_dict()
        rm2 = RotaryModule.from_dict(data)

        assert rm2.uid == rm.uid
        assert rm2.name == "Test Rotary"
        assert rm2.axis == Axis.B
        assert rm2.x == 10.0
        assert rm2.y == 20.0
        assert rm2.z == 5.0
        assert rm2.length == 300.0
        assert rm2.chuck_diameter == 50.0
        assert rm2.tailstock_diameter == 30.0
        assert rm2.max_height == 80.0

    def test_deserialization_defaults(self):
        data = {"uid": "test-uid"}
        rm = RotaryModule.from_dict(data)
        assert rm.uid == "test-uid"
        assert rm.axis == Axis.A
        assert rm.length == 200.0

    def test_deserialization_extra_keys_preserved(self):
        data = {"uid": "test", "future_field": "value"}
        rm = RotaryModule.from_dict(data)
        assert rm.extra["future_field"] == "value"
        assert "future_field" in rm.to_dict()

    def test_pickling(self):
        rm = RotaryModule()
        rm.set_position(10, 20, 30)

        state = rm.__getstate__()
        assert "changed" not in state

        rm2 = RotaryModule()
        rm2.__setstate__(state)
        assert rm2.x == 10.0
        assert hasattr(rm2, "changed")


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
        rm.set_length(500)
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
        rm.set_length(500)
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
        rm.set_length(300)
        machine.add_rotary_module(rm)

        data = machine.to_dict()
        machine2 = Machine.from_dict(data)

        assert len(machine2.rotary_modules) == 1
        rm2 = machine2.rotary_modules[rm.uid]
        assert rm2.name == "My Module"
        assert rm2.x == 10.0
        assert rm2.y == 20.0
        assert rm2.z == 5.0
        assert rm2.length == 300.0

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
