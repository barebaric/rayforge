"""Tests for machine capabilities inferred from heads (1a-1d)."""

from typing import Iterable, cast

from raygeo.ops.state import CoolantMode

from rayforge.core.capability import MachineCapability
from rayforge.machine.models.head import head_from_dict
from rayforge.machine.models.laser import LaserHead
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead


def _configure(machine: Machine):
    """Give a real Machine its default heads."""
    machine.heads = [LaserHead()]
    return machine


class TestCapabilitiesFromHeads:
    def test_only_laser_head_infers_laser(self, isolated_machine):
        _configure(isolated_machine)
        assert isolated_machine.get_capabilities() == {MachineCapability.LASER}

    def test_only_spindle_head_infers_mill(self, isolated_machine):
        _configure(isolated_machine)
        isolated_machine.heads = [SpindleHead()]
        assert isolated_machine.get_capabilities() == {MachineCapability.MILL}

    def test_laser_and_spindle_infers_both(self, isolated_machine):
        _configure(isolated_machine)
        isolated_machine.heads = [LaserHead(), SpindleHead()]
        assert isolated_machine.get_capabilities() == {
            MachineCapability.LASER,
            MachineCapability.MILL,
        }

    def test_no_heads_infers_nothing(self, isolated_machine):
        _configure(isolated_machine)
        isolated_machine.heads = []
        assert isolated_machine.get_capabilities() == frozenset()

    def test_explicit_capabilities_merge_with_inference(
        self, isolated_machine
    ):
        _configure(isolated_machine)
        isolated_machine.heads = [SpindleHead()]
        isolated_machine.set_explicit_capabilities(
            frozenset({MachineCapability.LASER})
        )
        # Explicit caps are merged with caps inferred from the heads.
        assert isolated_machine.get_capabilities() == {
            MachineCapability.LASER,
            MachineCapability.MILL,
        }

    def test_explicit_capabilities_only(self, isolated_machine):
        _configure(isolated_machine)
        isolated_machine.heads = []
        isolated_machine.set_explicit_capabilities(
            frozenset({MachineCapability.MILL})
        )
        assert isolated_machine.get_capabilities() == {MachineCapability.MILL}


class TestHeadDispatch:
    def test_laser_head_from_dict(self):
        data = {"type": "LaserHead", "name": "Diode", "max_power": 500}
        head = head_from_dict(data)
        assert isinstance(head, LaserHead)
        assert head.name == "Diode"
        assert head.max_power == 500

    def test_spindle_head_from_dict(self):
        data = {
            "type": "SpindleHead",
            "name": "Router",
            "max_rpm": 24000,
            "min_rpm": 500,
            "cooling_methods": ["FLOOD", "MIST"],
        }
        head = head_from_dict(data)
        assert isinstance(head, SpindleHead)
        assert head.max_rpm == 24000
        assert head.min_rpm == 500
        assert head.cooling_methods == (CoolantMode.FLOOD, CoolantMode.MIST)

    def test_spindle_cooling_methods_default_to_none(self):
        head = SpindleHead()
        assert head.cooling_methods == ()

    def test_set_cooling_methods_strips_off(self):
        head = SpindleHead()
        head.set_cooling_methods([CoolantMode.FLOOD, CoolantMode.OFF])
        assert head.cooling_methods == (CoolantMode.FLOOD,)

    def test_set_cooling_methods_ignores_non_modes(self):
        head = SpindleHead()
        raw = cast(
            Iterable[CoolantMode],
            [CoolantMode.MIST, "not-a-mode"],
        )
        head.set_cooling_methods(raw)
        assert head.cooling_methods == (CoolantMode.MIST,)

    def test_cooling_methods_from_dict_ignores_unknown(self):
        data = {
            "type": "SpindleHead",
            "name": "Router",
            "cooling_methods": ["FLOOD", "MIST", "VAPOR", "OFF"],
        }
        head = head_from_dict(data)
        assert isinstance(head, SpindleHead)
        assert head.cooling_methods == (CoolantMode.FLOOD, CoolantMode.MIST)

    def test_legacy_collet_key_is_ignored(self):
        data = {
            "type": "SpindleHead",
            "name": "Router",
            "max_rpm": 24000,
            "min_rpm": 500,
            "collet_type": "er20",
        }
        head = head_from_dict(data)
        assert isinstance(head, SpindleHead)
        assert not hasattr(head, "collet_type")

    def test_legacy_dict_without_type_defaults_to_laser(self):
        data = {"name": "Legacy", "max_power": 1000}
        head = head_from_dict(data)
        assert isinstance(head, LaserHead)
        assert head.max_power == 1000

    def test_spindle_roundtrip(self):
        head = SpindleHead()
        head.max_rpm = 18000
        head.min_rpm = 800
        head.cooling_methods = (CoolantMode.FLOOD, CoolantMode.MIST)
        restored = head_from_dict(head.to_dict())
        assert isinstance(restored, SpindleHead)
        assert restored.max_rpm == 18000
        assert restored.min_rpm == 800
        assert restored.cooling_methods == head.cooling_methods
