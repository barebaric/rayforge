"""
Tests for the Machine data model.

This module tests the Machine class as a data model, focusing on:
- Machine properties and attributes
- Coordinate transformations
- Machine state management

The Machine class delegates driver lifecycle and command logic to
MachineController, so this module focuses on the data model aspects.
"""

import gc
import math

import pytest

from rayforge import config
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    BezierToCommand,
)
from rayforge.core.ops.axis import Axis
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.rotary_module import RotaryModule, RotaryMode
from rayforge.machine.transport import TransportStatus
from rayforge.machine.kinematic_mapping import KinematicMapping


@pytest.fixture(autouse=True)
def clean_context_singleton():
    """Override conftest's per-test cleanup. Cleanup is handled by
    the class-scoped lite_context fixture."""
    yield


@pytest.fixture(scope="class")
def lite_context(tmp_path_factory):
    """Class-scoped context shared across all TestMachineModel tests."""
    from rayforge.addon_mgr.lazy_loader import reset_addon_finder

    tmp_path = tmp_path_factory.mktemp("machine_model")
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_dialect_dir.mkdir(parents=True, exist_ok=True)
    temp_machine_dir.mkdir(parents=True, exist_ok=True)

    old_config = (config.CONFIG_DIR, config.DIALECT_DIR, config.MACHINE_DIR)
    config.CONFIG_DIR = temp_config_dir
    config.DIALECT_DIR = temp_dialect_dir
    config.MACHINE_DIR = temp_machine_dir

    ctx = get_context()
    ctx.initialize_lite_context(temp_machine_dir)
    ctx._dialect_mgr = DialectManager(temp_dialect_dir)
    yield ctx

    context_module._context_instance = None
    config.CONFIG_DIR, config.DIALECT_DIR, config.MACHINE_DIR = old_config
    reset_addon_finder()
    gc.collect()


@pytest.mark.usefixtures("lite_context")
class TestMachineModel:
    """Test suite for the Machine data model."""

    # -- Initialization and basic properties --

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

    # -- Axis extents --

    def test_axis_extents_default(self, lite_context):
        """Test default axis_extents value."""
        machine = Machine(lite_context)
        assert machine.axis_extents == (200, 200)

    def test_axis_extents_setter(self, lite_context):
        """Test setting axis_extents."""
        machine = Machine(lite_context)
        machine.set_axis_extents(300, 400)
        assert machine.axis_extents == (300, 400)

    # -- Work margins --

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

    # -- Soft limits --

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

    # -- Visual extent frame --

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

    # -- Custom work area --

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

    # -- Change signals --

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

    # -- Serialization --

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
        machine = Machine.from_dict(data, context=lite_context)
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
        machine = Machine.from_dict(data, context=lite_context)
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
        machine = Machine.from_dict(data, context=lite_context)
        assert machine.axis_extents == (350, 450)

    def test_work_margins_persist_through_serialization(self, lite_context):
        """Test that work_margins persist through to_dict/from_dict cycle."""
        machine1 = Machine(lite_context)
        machine1.set_axis_extents(500, 600)
        machine1.set_work_margins(100, 150, 100, 100)

        data = machine1.to_dict()

        machine2 = Machine.from_dict(data, context=lite_context)
        assert machine2.axis_extents == (500, 600)
        assert machine2.work_margins == (100, 150, 100, 100)


@pytest.fixture
def rotary_doc(isolated_machine):
    doc = Doc()
    doc.active_layer.set_rotary_enabled(True)
    doc.active_layer.set_rotary_diameter(25.0)
    rm = RotaryModule()
    rm.set_axis(Axis.A)
    isolated_machine.add_rotary_module(rm)
    doc.active_layer.set_rotary_module_uid(rm.uid)
    return doc


def _encode_rotary_line(machine, doc):
    ops = Ops()
    ops.add(MoveToCommand((0.0, 0.0, 0.0)))
    ops.add(LineToCommand((10.0, 10.0, 0.0)))
    for layer in doc.layers:
        rotary_axis = machine.get_rotary_axis_for_layer(layer)
        if rotary_axis is not None:
            mapping = KinematicMapping(
                rotary_axis=rotary_axis,
                diameter=layer.rotary_diameter,
            )
            mapping.apply(ops)
    gcode, _ = machine.encode_ops(ops, doc)
    return gcode


class TestRotaryAxisGcodeOutput:
    def test_default_axis_uses_a(self, isolated_machine, rotary_doc):
        gcode = _encode_rotary_line(isolated_machine, rotary_doc)
        assert "A" in gcode

    def test_custom_axis_uses_configured_letter(self, isolated_machine):
        rm = RotaryModule()
        rm.set_axis(Axis.U)
        isolated_machine.add_rotary_module(rm)
        doc = Doc()
        doc.active_layer.set_rotary_enabled(True)
        doc.active_layer.set_rotary_diameter(25.0)
        doc.active_layer.set_rotary_module_uid(rm.uid)
        gcode = _encode_rotary_line(isolated_machine, doc)
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

    # -- Supports curves --

    def test_default_is_false(self, lite_context):
        machine = Machine(lite_context)
        assert machine.supports_curves is False

    def test_set_supports_curves(self, lite_context):
        machine = Machine(lite_context)
        machine.set_supports_curves(True)
        assert machine.supports_curves is True

    def test_no_signal_on_same_value(self, lite_context):
        machine = Machine(lite_context)
        signals = []
        machine.changed.connect(lambda m: signals.append(m))
        machine.set_supports_curves(False)
        assert len(signals) == 0

    def test_serialization_round_trip(self, lite_context):
        machine = Machine(lite_context)
        machine.set_supports_curves(True)
        data = machine.to_dict()
        restored = Machine.from_dict(data, context=lite_context)
        assert restored.supports_curves is True

    def test_encode_ops_with_layer_markers_uses_b(self, isolated_machine):
        """encode_ops maps Y to B when layer has rotary enabled."""
        rm = RotaryModule()
        rm.set_axis(Axis.B)
        isolated_machine.add_rotary_module(rm)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        diameter = 25.0
        circumference = diameter * math.pi
        expected_deg = (10.0 / circumference) * 360.0
        formatted_deg = f"{expected_deg:.3f}".rstrip("0").rstrip(".")
        assert " B" in gcode
        assert formatted_deg in gcode
        assert "G0 X0 Y0" not in gcode.split("M5")[0]
        assert "G1" in gcode
        cut_line = [ln for ln in gcode.split("\n") if ln.startswith("G1")][0]
        assert " Y" not in cut_line

    def test_replacement_raw_y_in_gcode(self, isolated_machine):
        """AXIS_REPLACEMENT with mm_per_rotation=0 emits raw Y values."""
        rm = RotaryModule()
        rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
        isolated_machine.add_rotary_module(rm)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        assert " A" not in gcode
        assert " B" not in gcode
        cut_lines = [ln for ln in gcode.split("\n") if ln.startswith("G1")]
        assert len(cut_lines) >= 1
        assert " Y10" in cut_lines[0]

    def test_replacement_scaled_y_in_gcode(self, isolated_machine):
        """AXIS_REPLACEMENT with mm_per_rotation>0 scales Y values."""
        rm = RotaryModule()
        rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
        rm.set_mm_per_rotation(100.0)
        isolated_machine.add_rotary_module(rm)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        assert " A" not in gcode
        assert " B" not in gcode
        expected = 10.0 * 100.0 / (math.pi * 25.0)
        formatted = f"{expected:.3f}".rstrip("0").rstrip(".")
        assert formatted in gcode

    def test_prepare_ops_linearizes_by_default(self, lite_context):
        machine = Machine(lite_context)
        ops = Ops()
        ops.move_to(0, 0)
        ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))
        prepared = machine._prepare_ops_for_encoding(ops)
        assert not any(
            isinstance(c, BezierToCommand) for c in prepared.commands
        )
        assert any(isinstance(c, LineToCommand) for c in prepared.commands)

    def test_prepare_ops_preserves_curves_when_enabled(self, lite_context):
        machine = Machine(lite_context)
        machine.set_supports_curves(True)
        ops = Ops()
        ops.move_to(0, 0)
        ops.bezier_to(c1=(10, 0, 0), c2=(10, 10, 0), end=(0, 10, 0))
        prepared = machine._prepare_ops_for_encoding(ops)
        assert any(isinstance(c, BezierToCommand) for c in prepared.commands)

    def test_true_4th_axis_top_left_origin(self, isolated_machine):
        """TRUE_4TH_AXIS degrees come from world-space Y regardless of
        origin.  With TOP_LEFT the world→machine matrix flips Y but the
        A-axis rotation must still correspond to the world-space distance
        on the cylinder surface."""
        rm = RotaryModule()
        rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
        rm.set_axis(Axis.A)
        isolated_machine.add_rotary_module(rm)
        isolated_machine.set_origin(Origin.TOP_LEFT)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        diameter = 25.0
        circumference = diameter * math.pi
        expected_deg = (10.0 / circumference) * 360.0
        formatted_deg = f"{expected_deg:.3f}".rstrip("0").rstrip(".")
        assert formatted_deg in gcode
        assert " A" in gcode

    def test_replacement_scaled_y_top_left_origin(self, isolated_machine):
        """AXIS_REPLACEMENT computes degrees from world-space Y.

        After Phase 3 the kinematic mapping runs on world-space ops
        before world→machine, so the scaled-Y value in the G-code
        reflects the world-space surface distance (not machine-space).
        """
        rm = RotaryModule()
        rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
        rm.set_mm_per_rotation(100.0)
        isolated_machine.add_rotary_module(rm)
        isolated_machine.set_origin(Origin.TOP_LEFT)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        assert " A" not in gcode
        assert " B" not in gcode
        expected = 10.0 * 100.0 / (math.pi * 25.0)
        formatted = f"{expected:.3f}".rstrip("0").rstrip(".")
        assert formatted in gcode

    def test_true_4th_axis_reversed_y(self, isolated_machine):
        """TRUE_4TH_AXIS with reversed Y axis: degrees still match
        world-space surface distance."""
        rm = RotaryModule()
        rm.set_mode(RotaryMode.TRUE_4TH_AXIS)
        rm.set_axis(Axis.A)
        isolated_machine.add_rotary_module(rm)

        isolated_machine.set_reverse_y_axis(True)

        doc = Doc()
        layer = doc.active_layer
        layer.set_rotary_enabled(True)
        layer.set_rotary_diameter(25.0)
        layer.set_rotary_module_uid(rm.uid)

        ops = Ops()
        ops.job_start()
        ops.layer_start(layer_uid=layer.uid)
        ops.move_to(0, 0, 0)
        ops.line_to(10, 10, 0)
        ops.layer_end(layer_uid=layer.uid)
        ops.job_end()

        gcode, _ = isolated_machine.encode_ops(ops, doc)

        diameter = 25.0
        circumference = diameter * math.pi
        expected_deg = (10.0 / circumference) * 360.0
        formatted_deg = f"{expected_deg:.3f}".rstrip("0").rstrip(".")
        assert formatted_deg in gcode
