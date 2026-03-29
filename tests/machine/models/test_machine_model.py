"""
Tests for the Machine data model.

This module tests the Machine class as a data model, focusing on:
- Machine properties and attributes
- Coordinate transformations
- Machine state management

The Machine class delegates driver lifecycle and command logic to
MachineController, so this module focuses on the data model aspects.
"""

import math

import pytest

from rayforge.core.doc import Doc
from rayforge.core.ops import Ops, MoveToCommand, LineToCommand
from rayforge.machine.driver.driver import Axis
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.transport import TransportStatus


@pytest.mark.usefixtures("lite_context")
class TestMachineModel:
    """Test suite for the Machine data model."""

    def test_machine_initialization(self, lite_context):
        """Test that a Machine can be initialized with a context."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine is not None
        assert machine.context == lite_context
        assert machine.controller is not None
        assert machine.driver is not None

    def test_machine_id_generation(self, lite_context):
        """Test that each machine gets a unique ID."""
        machine1 = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine1)
        machine2 = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine2)
        assert machine1.id != machine2.id
        assert isinstance(machine1.id, str)

    def test_machine_origin(self, lite_context):
        """Test setting and getting machine origin."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        machine.set_origin(Origin.TOP_LEFT)
        assert machine.origin == Origin.TOP_LEFT

        machine.set_origin(Origin.BOTTOM_LEFT)
        assert machine.origin == Origin.BOTTOM_LEFT

    def test_machine_driver_property(self, lite_context):
        """Test that the driver property delegates to controller."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine.driver == machine.controller.driver

    def test_machine_connection_status(self, lite_context):
        """Test machine connection status property."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine.connection_status == TransportStatus.DISCONNECTED

    def test_machine_wcs_properties(self, lite_context):
        """Test machine WCS (Work Coordinate System) properties."""
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        assert machine.machine_space_wcs is not None
        assert isinstance(machine.machine_space_wcs_display_name, str)

    def test_machine_signals_exist(self, lite_context):
        """Test that machine has all required signals."""
        machine = Machine(lite_context)
        assert hasattr(machine, "connection_status_changed")
        assert hasattr(machine, "state_changed")
        assert hasattr(machine, "job_finished")
        assert hasattr(machine, "command_status_changed")
        assert hasattr(machine, "wcs_updated")


@pytest.mark.usefixtures("lite_context")
class TestMachineAxisExtents:
    """Test suite for axis_extents property."""

    def test_axis_extents_default(self, lite_context):
        """Test default axis_extents value."""
        machine = Machine(lite_context)
        assert machine.axis_extents == (200, 200)

    def test_axis_extents_setter(self, lite_context):
        """Test setting axis_extents."""
        machine = Machine(lite_context)
        machine.set_axis_extents(300, 400)
        assert machine.axis_extents == (300, 400)


@pytest.mark.usefixtures("lite_context")
class TestMachineWorkMargins:
    """Test suite for work_margins property."""

    def test_work_margins_default(self, lite_context):
        """Test default work_margins are all zero."""
        machine = Machine(lite_context)
        assert machine.work_margins == (0, 0, 0, 0)

    def test_work_area_computed_from_margins(self, lite_context):
        """Test work_area is computed from axis_extents and margins."""
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 600)
        machine.set_work_margins(50, 100, 75, 125)
        assert machine.work_area == (50, 100, 375, 375)

    def test_work_area_default_no_margins(self, lite_context):
        """Test default work_area equals full axis_extents."""
        machine = Machine(lite_context)
        machine.set_axis_extents(300, 400)
        assert machine.work_area == (0, 0, 300, 400)

    def test_set_axis_extents_clamps_margins(self, lite_context):
        """Test that set_axis_extents clamps margins if they don't fit."""
        machine = Machine(lite_context)
        machine.set_work_margins(50, 50, 50, 50)
        machine.set_axis_extents(80, 90)
        ml, mt, mr, mb = machine.work_margins
        assert ml + mr < 80
        assert mt + mb < 90

    def test_set_axis_extents_preserves_zero_margins(self, lite_context):
        """Test that set_axis_extents keeps zero margins unchanged."""
        machine = Machine(lite_context)
        machine.set_work_margins(0, 0, 0, 0)
        machine.set_axis_extents(300, 400)
        assert machine.work_margins == (0, 0, 0, 0)

    def test_work_area_size_clamped_to_positive(self, lite_context):
        """Test that work_area size is always positive."""
        machine = Machine(lite_context)
        machine.set_axis_extents(100, 100)
        machine.set_work_margins(99, 99, 99, 99)
        _, _, w, h = machine.work_area
        assert w >= 1
        assert h >= 1


@pytest.mark.usefixtures("lite_context")
class TestMachineSoftLimits:
    """Test suite for soft_limits property."""

    def test_soft_limits_default_none(self, lite_context):
        """Test that soft_limits defaults to None."""
        machine = Machine(lite_context)
        assert machine.soft_limits is None

    def test_soft_limits_setter(self, lite_context):
        """Test setting soft_limits."""
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(10, 20, 300, 400)
        assert machine.soft_limits == (10, 20, 300, 400)

    def test_soft_limits_clamped_to_axis_extents(self, lite_context):
        """Test that soft limits are clamped to axis extents."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        machine.set_soft_limits(10, 20, 500, 600)
        assert machine.soft_limits == (10, 20, 200, 300)

    def test_soft_limits_clamped_on_axis_extents_change(self, lite_context):
        """Test that soft limits are clamped when axis extents shrink."""
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(10, 20, 400, 450)
        machine.set_axis_extents(200, 300)
        assert machine.soft_limits == (10, 20, 200, 300)

    def test_soft_limits_negative_clamped(self, lite_context):
        """Test that negative soft limits are clamped to 0."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        machine.set_soft_limits(-50, -30, 150, 200)
        assert machine.soft_limits == (0, 0, 150, 200)

    def test_get_soft_limits_uses_axis_extents_when_none(self, lite_context):
        """Test get_soft_limits uses axis_extents when soft_limits is None."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        limits = machine.get_soft_limits()
        assert limits == (0.0, 0.0, 200.0, 300.0)

    def test_get_soft_limits_uses_custom_limits(self, lite_context):
        """Test get_soft_limits uses custom limits when set."""
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(50, 50, 400, 400)
        limits = machine.get_soft_limits()
        assert limits == (50.0, 50.0, 400.0, 400.0)

    def test_get_soft_limits_with_reversed_axes(self, lite_context):
        """Test get_soft_limits with reversed axes."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        machine.set_reverse_x_axis(True)
        machine.set_reverse_y_axis(True)
        limits = machine.get_soft_limits()
        assert limits == (-200.0, -300.0, 0.0, 0.0)

    def test_get_soft_limits_with_custom_limits_and_reversed(
        self, lite_context
    ):
        """Test get_soft_limits with custom limits and reversed axes."""
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(50, 50, 400, 400)
        machine.set_reverse_x_axis(True)
        machine.set_reverse_y_axis(True)
        limits = machine.get_soft_limits()
        assert limits == (-400.0, -400.0, -50.0, -50.0)


@pytest.mark.usefixtures("lite_context")
class TestMachineVisualExtentFrame:
    """Test suite for get_visual_extent_frame method."""

    def test_extent_frame_default_no_margins(self, lite_context):
        """Test extent frame position with no margins."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        x, y, w, h = machine.get_visual_extent_frame()
        assert (x, y, w, h) == (0.0, 0.0, 200.0, 300.0)

    def test_extent_frame_with_margins(self, lite_context):
        """Test extent frame position with margins."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        machine.set_work_margins(10, 20, 30, 40)
        x, y, w, h = machine.get_visual_extent_frame()
        assert (x, y, w, h) == (-10.0, -40.0, 200.0, 300.0)

    def test_extent_frame_returns_floats(self, lite_context):
        """Test that get_visual_extent_frame returns floats."""
        machine = Machine(lite_context)
        machine.set_axis_extents(200, 300)
        machine.set_work_margins(10, 20, 30, 40)
        frame = machine.get_visual_extent_frame()
        assert all(isinstance(v, float) for v in frame)


@pytest.mark.usefixtures("lite_context")
class TestMachineHasCustomWorkArea:
    """Test suite for has_custom_work_area method."""

    def test_no_custom_work_area_default(self, lite_context):
        """Test returns False when all margins are zero."""
        machine = Machine(lite_context)
        assert machine.has_custom_work_area() is False

    def test_has_custom_work_area_with_left_margin(self, lite_context):
        """Test returns True with non-zero left margin."""
        machine = Machine(lite_context)
        machine.set_work_margins(10, 0, 0, 0)
        assert machine.has_custom_work_area() is True

    def test_has_custom_work_area_with_top_margin(self, lite_context):
        """Test returns True with non-zero top margin."""
        machine = Machine(lite_context)
        machine.set_work_margins(0, 10, 0, 0)
        assert machine.has_custom_work_area() is True

    def test_has_custom_work_area_with_right_margin(self, lite_context):
        """Test returns True with non-zero right margin."""
        machine = Machine(lite_context)
        machine.set_work_margins(0, 0, 10, 0)
        assert machine.has_custom_work_area() is True

    def test_has_custom_work_area_with_bottom_margin(self, lite_context):
        """Test returns True with non-zero bottom margin."""
        machine = Machine(lite_context)
        machine.set_work_margins(0, 0, 0, 10)
        assert machine.has_custom_work_area() is True

    def test_has_custom_work_area_with_all_margins(self, lite_context):
        """Test returns True with all non-zero margins."""
        machine = Machine(lite_context)
        machine.set_work_margins(10, 20, 30, 40)
        assert machine.has_custom_work_area() is True


@pytest.mark.usefixtures("lite_context")
class TestMachineChangeSignals:
    """Test suite for change signals on new properties."""

    def test_signal_on_set_axis_extents(self, lite_context):
        """Test that changed signal fires on set_axis_extents."""
        machine = Machine(lite_context)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_axis_extents(300, 400)
        assert len(signal_calls) == 1
        assert signal_calls[0] is machine

    def test_signal_on_set_work_margins(self, lite_context):
        """Test that changed signal fires on set_work_margins."""
        machine = Machine(lite_context)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_work_margins(10, 20, 30, 40)
        assert len(signal_calls) == 1
        assert signal_calls[0] is machine

    def test_signal_on_set_soft_limits(self, lite_context):
        """Test that changed signal fires on set_soft_limits."""
        machine = Machine(lite_context)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_soft_limits(10, 20, 300, 400)
        assert len(signal_calls) == 1
        assert signal_calls[0] is machine

    def test_no_signal_on_same_axis_extents(self, lite_context):
        """Test that changed signal doesn't fire for same axis_extents."""
        machine = Machine(lite_context)
        machine.set_axis_extents(300, 400)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_axis_extents(300, 400)
        assert len(signal_calls) == 0

    def test_no_signal_on_same_work_margins(self, lite_context):
        """Test that changed signal doesn't fire for same work_margins."""
        machine = Machine(lite_context)
        machine.set_work_margins(10, 20, 30, 40)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_work_margins(10, 20, 30, 40)
        assert len(signal_calls) == 0

    def test_no_signal_on_same_soft_limits(self, lite_context):
        """Test that changed signal doesn't fire for same soft_limits."""
        machine = Machine(lite_context)
        machine.set_soft_limits(10, 20, 300, 400)
        signal_calls = []

        def signal_handler(sender):
            signal_calls.append(sender)

        machine.changed.connect(signal_handler)
        machine.set_soft_limits(10, 20, 300, 400)
        assert len(signal_calls) == 0

    def test_no_signal_on_same_rotary_axis(self, isolated_machine):
        calls = []

        def on_changed(sender):
            calls.append(True)

        isolated_machine.changed.connect(on_changed)
        isolated_machine.set_rotary_axis(Axis.A)
        assert calls == []

    def test_signal_on_set_rotary_axis(self, isolated_machine):
        calls = []

        def on_changed(sender):
            calls.append(True)

        isolated_machine.changed.connect(on_changed)
        isolated_machine.set_rotary_axis(Axis.U)
        assert len(calls) == 1

    def test_set_rotary_axis_rejects_combined_flags(self, isolated_machine):
        with pytest.raises(ValueError):
            isolated_machine.set_rotary_axis(Axis.X | Axis.A)


@pytest.mark.usefixtures("lite_context")
class TestMachineSerialization:
    """Test suite for serialization with new properties."""

    def test_to_dict_includes_new_properties(self, lite_context):
        """
        Test that to_dict includes axis_extents, work_margins, soft_limits.
        """
        machine = Machine(lite_context)
        machine.set_axis_extents(300, 400)
        machine.set_work_margins(50, 50, 50, 50)
        machine.set_soft_limits(10, 10, 280, 380)

        data = machine.to_dict()["machine"]
        assert data["axis_extents"] == [300, 400]
        assert data["work_margins"] == [50, 50, 50, 50]
        assert data["soft_limits"] == [10, 10, 280, 380]
        assert "work_surface" not in data

    def test_from_dict_reads_work_margins(self, lite_context):
        """Test that from_dict reads work_margins."""
        data = {
            "machine": {
                "axis_extents": [300, 400],
                "work_margins": [50, 50, 50, 50],
                "soft_limits": [10, 10, 280, 380],
            }
        }
        machine = Machine.from_dict(data)
        assert machine.axis_extents == (300, 400)
        assert machine.work_margins == (50, 50, 50, 50)
        assert machine.soft_limits == (10, 10, 280, 380)

    def test_from_dict_migrates_offsets_to_margins(self, lite_context):
        """Test that from_dict migrates old offsets to margins."""
        data = {
            "machine": {
                "dimensions": [400, 500],
                "offsets": [100, 150],
            }
        }
        machine = Machine.from_dict(data)
        assert machine.axis_extents == (400, 500)
        assert machine.work_margins == (100, 0, 0, 150)
        assert machine.work_area == (100, 0, 300, 350)

    def test_from_dict_dimensions_falls_back_to_axis_extents(
        self, lite_context
    ):
        """Test that from_dict reads dimensions into axis_extents."""
        data = {
            "machine": {
                "dimensions": [350, 450],
            }
        }
        machine = Machine.from_dict(data)
        assert machine.axis_extents == (350, 450)

    def test_work_margins_persist_through_serialization(self, lite_context):
        """Test that work_margins persist through to_dict/from_dict cycle."""
        machine1 = Machine(lite_context)
        machine1.set_axis_extents(500, 600)
        machine1.set_work_margins(100, 150, 100, 100)

        data = machine1.to_dict()

        machine2 = Machine.from_dict(data)
        assert machine2.axis_extents == (500, 600)
        assert machine2.work_margins == (100, 150, 100, 100)

    def test_to_dict_includes_rotary_axis(self, isolated_machine):
        isolated_machine.set_rotary_axis(Axis.U)
        data = isolated_machine.to_dict()
        assert data["machine"]["rotary_axis"] == "U"

    def test_from_dict_restores_rotary_axis(self, isolated_machine):
        isolated_machine.set_rotary_axis(Axis.U)
        data = isolated_machine.to_dict()
        restored = Machine.from_dict(data)
        assert restored.rotary_axis == Axis.U

    def test_from_dict_rotary_axis_defaults_to_a(self):
        data = {"machine": {"name": "test"}}
        machine = Machine.from_dict(data)
        assert machine.rotary_axis == Axis.A


@pytest.fixture
def rotary_doc():
    doc = Doc()
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_diameter(25.0)
    return doc


def _encode_rotary_line(machine, doc):
    ops = Ops()
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    gcode, _ = machine.encode_ops(ops, doc)
    return gcode


class TestRotaryAxisGcodeOutput:
    def test_default_axis_uses_a(self, isolated_machine, rotary_doc):
        gcode = _encode_rotary_line(isolated_machine, rotary_doc)
        assert "A" in gcode

    def test_custom_axis_uses_configured_letter(
        self, isolated_machine, rotary_doc
    ):
        isolated_machine.set_rotary_axis(Axis.U)
        gcode = _encode_rotary_line(isolated_machine, rotary_doc)
        assert "U" in gcode

    def test_non_rotary_uses_y(self, isolated_machine):
        doc = Doc()
        doc.active_layer.set_rotary_enabled(False)
        gcode = _encode_rotary_line(isolated_machine, doc)
        assert " Y" in gcode

    def test_rotary_axis_degree_conversion(self, isolated_machine, rotary_doc):
        gcode = _encode_rotary_line(isolated_machine, rotary_doc)
        diameter = 25.0
        circumference = diameter * math.pi
        expected_deg = (10.0 / circumference) * 360.0
        formatted_deg = f"{expected_deg:.3f}".rstrip("0").rstrip(".")
        assert formatted_deg in gcode
