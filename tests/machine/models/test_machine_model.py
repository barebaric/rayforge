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
import logging

import pytest
from raygeo.ops.axis import Axis

from rayforge import config
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.core.doc import Doc
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.rotary_module import RotaryModule
from rayforge.machine.transport import TransportStatus


@pytest.fixture(autouse=True)
def clean_context_singleton():
    """Override conftest's per-test cleanup. Cleanup is handled by
    the class-scoped lite_context fixture."""
    yield


@pytest.fixture(scope="class")
def lite_context(tmp_path_factory):
    """Class-scoped context shared across all TestMachineModel tests."""
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

    def test_work_area_clamps_overlarge_margins(self, lite_context):
        """Over-large margins are clamped so the work area stays positive."""
        machine = Machine(lite_context)
        machine.set_axis_extents(100, 100)
        machine.set_work_margins(99, 99, 99, 99)
        _, _, w, h = machine.work_area
        assert w > 0
        assert h > 0

    def test_set_work_margins_warns_on_clamp(self, lite_context, caplog):
        """A warning is logged when margins are clamped."""
        machine = Machine(lite_context)
        machine.set_axis_extents(100, 100)
        with caplog.at_level(logging.WARNING):
            machine.set_work_margins(80, 80, 80, 80)
        assert any("clamped" in r.message.lower() for r in caplog.records)

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

    def test_placeholder_round_trip(self, lite_context):
        """The placeholder flag survives serialization and defaults off."""
        machine = Machine(lite_context)
        assert machine.placeholder is False

        data = machine.to_dict()
        assert data["machine"]["placeholder"] is False

        machine.placeholder = True
        restored = Machine.from_dict(machine.to_dict(), context=lite_context)
        assert restored.placeholder is True

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


@pytest.mark.usefixtures("lite_context")
class TestSupportsCurves:
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


class TestSetSupportsArcs:
    """Tests for the set_supports_arcs setter."""

    def test_default_is_true(self, lite_context):
        machine = Machine(lite_context)
        assert machine.supports_arcs is True

    def test_set_supports_arcs(self, lite_context):
        machine = Machine(lite_context)
        machine.set_supports_arcs(False)
        assert machine.supports_arcs is False
        machine.set_supports_arcs(True)
        assert machine.supports_arcs is True

    def test_signal_fired_on_change(self, lite_context):
        machine = Machine(lite_context)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_supports_arcs(False)
        assert len(signals) == 1

    def test_no_signal_on_same_value(self, lite_context):
        machine = Machine(lite_context)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_supports_arcs(True)
        assert len(signals) == 0

    def test_serialization_round_trip(self, lite_context):
        machine = Machine(lite_context)
        machine.set_supports_arcs(False)
        data = machine.to_dict()
        restored = Machine.from_dict(data, context=lite_context)
        assert restored.supports_arcs is False


class TestClearSoftLimits:
    """Tests for the clear_soft_limits method."""

    def test_clears_soft_limits(self, lite_context):
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(10, 20, 300, 400)
        assert machine.soft_limits is not None
        machine.clear_soft_limits()
        assert machine.soft_limits is None

    def test_signal_fired_when_limits_exist(self, lite_context):
        machine = Machine(lite_context)
        machine.set_axis_extents(500, 500)
        machine.set_soft_limits(10, 20, 300, 400)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.clear_soft_limits()
        assert len(signals) == 1

    def test_no_signal_when_already_none(self, lite_context):
        machine = Machine(lite_context)
        assert machine.soft_limits is None
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.clear_soft_limits()
        assert len(signals) == 0


class TestSetterEqualityGuards:
    """Tests for equality guards on setters that previously lacked them."""

    def test_set_home_on_start_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.home_on_start = True
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_home_on_start(True)
        assert len(signals) == 0

    def test_set_home_on_start_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.home_on_start = True
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_home_on_start(False)
        assert len(signals) == 1

    def test_set_max_travel_speed_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.max_travel_speed = 5000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_max_travel_speed(5000)
        assert len(signals) == 0

    def test_set_max_travel_speed_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.max_travel_speed = 5000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_max_travel_speed(6000)
        assert len(signals) == 1

    def test_set_max_cut_speed_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.max_cut_speed = 3000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_max_cut_speed(3000)
        assert len(signals) == 0

    def test_set_max_cut_speed_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.max_cut_speed = 3000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_max_cut_speed(4000)
        assert len(signals) == 1

    def test_set_acceleration_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.acceleration = 1000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_acceleration(1000)
        assert len(signals) == 0

    def test_set_acceleration_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.acceleration = 1000
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_acceleration(2000)
        assert len(signals) == 1

    def test_set_origin_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.set_origin(Origin.BOTTOM_LEFT)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_origin(Origin.BOTTOM_LEFT)
        assert len(signals) == 0

    def test_set_origin_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.set_origin(Origin.BOTTOM_LEFT)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_origin(Origin.TOP_LEFT)
        assert len(signals) == 1

    def test_set_soft_limits_enabled_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.soft_limits_enabled = False
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_soft_limits_enabled(False)
        assert len(signals) == 0

    def test_set_soft_limits_enabled_signal_on_change(self, lite_context):
        machine = Machine(lite_context)
        machine.soft_limits_enabled = False
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_soft_limits_enabled(True)
        assert len(signals) == 1


class TestHydratedDialectCache:
    """Tests for _hydrated_dialect cache invalidation."""

    def test_set_dialect_uid_clears_cache(self, lite_context):
        machine = Machine(lite_context)
        machine.hydrate()
        assert machine._hydrated_dialect is not None
        machine.set_dialect_uid("smoothieware")
        assert machine._hydrated_dialect is None

    def test_set_dialect_uid_no_clear_on_same(self, lite_context):
        machine = Machine(lite_context)
        machine.set_dialect_uid("grbl")
        machine.hydrate()
        assert machine._hydrated_dialect is not None
        machine.set_dialect_uid("grbl")
        assert machine._hydrated_dialect is not None

    def test_dialect_returns_fresh_after_uid_change(self, lite_context):
        from rayforge.machine.models.dialect import (
            GRBL_DIALECT,
            SMOOTHIEWARE_DIALECT,
        )

        machine = Machine(lite_context)
        assert machine.dialect == GRBL_DIALECT
        machine.hydrate()
        assert machine._hydrated_dialect == GRBL_DIALECT
        machine.set_dialect_uid("smoothieware")
        assert machine.dialect == SMOOTHIEWARE_DIALECT


class TestSetDriverEmitsChanged:
    """Tests for set_driver/set_driver_args emitting changed."""

    def test_set_driver_emits_changed_signal(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        from rayforge.machine.driver.dummy import NoDeviceDriver

        class TestDriver(NoDeviceDriver):
            pass

        machine.set_driver(TestDriver, {"port": "/dev/null"})
        assert len(signals) >= 1

    def test_set_driver_args_emits_changed_signal(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        from rayforge.machine.driver.dummy import NoDeviceDriver

        class TestDriver2(NoDeviceDriver):
            pass

        machine.set_driver(TestDriver2, {"port": "/dev/old"})
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_driver_args({"port": "/dev/new"})
        assert len(signals) >= 1

    def test_set_driver_no_signal_on_same(self, lite_context):
        machine = Machine(lite_context)
        lite_context.machine_mgr.add_machine(machine)
        from rayforge.machine.driver.dummy import NoDeviceDriver

        class TestDriver3(NoDeviceDriver):
            pass

        machine.set_driver(TestDriver3, {"port": "/dev/null"})
        signals = []

        def handler(sender):
            signals.append(sender)

        machine.changed.connect(handler)
        machine.set_driver(TestDriver3, {"port": "/dev/null"})
        assert len(signals) == 0
