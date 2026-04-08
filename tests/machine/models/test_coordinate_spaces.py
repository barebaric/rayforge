"""
Tests for coordinate space handling in the Machine class.

This module tests the coordinate space transformations and offset
calculations that are fundamental to correct positioning display
and G-code generation.

Coordinate Spaces:
- WORLD: Canvas/visual coordinates (normalized: Y-up, origin bottom-left)
- MACHINE: Raw machine coordinates (origin/axis per machine config)
- REFERENCE: The coordinate system shown to user (MACHINE minus wcs
  or workarea offset)
"""

import gc

import pytest

from rayforge import config
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.core.ops import Ops
from rayforge.machine.models.dialect_manager import DialectManager
from rayforge.machine.models.machine import Machine, Origin


@pytest.fixture(autouse=True)
def clean_context_singleton():
    """Override conftest's per-test cleanup. Cleanup is handled by
    the class-scoped lite_context fixture."""
    yield


@pytest.fixture(scope="class")
def lite_context(tmp_path_factory):
    """Class-scoped context shared across all TestCoordinateSpaces tests."""
    from rayforge.addon_mgr.lazy_loader import reset_addon_finder

    tmp_path = tmp_path_factory.mktemp("coordinate_spaces")
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = tmp_path / "machines"
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


@pytest.fixture
def machine(lite_context):
    """Provides a fresh Machine instance for each test."""
    return Machine(lite_context)


@pytest.mark.usefixtures("lite_context")
class TestCoordinateSpaces:
    """
    Combined tests for coordinate space transformations.

    Tests all coordinate space operations including:
    - World-to-machine point conversion
    - Workarea origin offsets
    - Reference offset calculations
    - Position display in workarea and WCS modes
    - Full coordinate pipeline
    - UI/G-code alignment
    """

    # -- World-to-machine --

    WORLD_TO_MACHINE_SCENARIOS = [
        # (origin, reverse_x, reverse_y, expected_mx, expected_my)
        # BOTTOM_LEFT: x_axis_right=False, y_axis_down=False
        (Origin.BOTTOM_LEFT, False, False, 10.0, 20.0),
        (Origin.BOTTOM_LEFT, True, False, -10.0, 20.0),
        (Origin.BOTTOM_LEFT, False, True, 10.0, -20.0),
        (Origin.BOTTOM_LEFT, True, True, -10.0, -20.0),
        # TOP_LEFT: x_axis_right=False, y_axis_down=True
        # Y transform: my = height - y = 100 - 20 = 80
        (Origin.TOP_LEFT, False, False, 10.0, 80.0),
        (Origin.TOP_LEFT, True, False, -10.0, 80.0),
        (Origin.TOP_LEFT, False, True, 10.0, -80.0),
        (Origin.TOP_LEFT, True, True, -10.0, -80.0),
        # BOTTOM_RIGHT: x_axis_right=True, y_axis_down=False
        # X transform: mx = width - x = 100 - 10 = 90
        (Origin.BOTTOM_RIGHT, False, False, 90.0, 20.0),
        (Origin.BOTTOM_RIGHT, True, False, -90.0, 20.0),
        (Origin.BOTTOM_RIGHT, False, True, 90.0, -20.0),
        (Origin.BOTTOM_RIGHT, True, True, -90.0, -20.0),
        # TOP_RIGHT: x_axis_right=True, y_axis_down=True
        # X transform: mx = width - x = 100 - 10 = 90
        # Y transform: my = height - y = 100 - 20 = 80
        (Origin.TOP_RIGHT, False, False, 90.0, 80.0),
        (Origin.TOP_RIGHT, True, False, -90.0, 80.0),
        (Origin.TOP_RIGHT, False, True, 90.0, -80.0),
        (Origin.TOP_RIGHT, True, True, -90.0, -80.0),
    ]

    @pytest.mark.parametrize(
        "origin,reverse_x,reverse_y,expected_mx,expected_my",
        WORLD_TO_MACHINE_SCENARIOS,
    )
    def test_world_point_to_machine_all_combinations(
        self, machine, origin, reverse_x, reverse_y, expected_mx, expected_my
    ):
        """
        Test world_point_to_machine for all combinations of origin
        and axis reversal.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(origin)
        machine.set_reverse_x_axis(reverse_x)
        machine.set_reverse_y_axis(reverse_y)

        space = machine.get_coordinate_space()
        result = space.world_point_to_machine(10.0, 20.0)
        assert result[0] == pytest.approx(expected_mx)
        assert result[1] == pytest.approx(expected_my)

    @pytest.mark.parametrize(
        "origin,reverse_x,reverse_y,expected_mx,expected_my",
        WORLD_TO_MACHINE_SCENARIOS,
    )
    def test_world_point_to_machine_matches_encoder(
        self, machine, origin, reverse_x, reverse_y, expected_mx, expected_my
    ):
        """
        Verify world_point_to_machine matches _prepare_ops_for_encoding.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(origin)
        machine.set_reverse_x_axis(reverse_x)
        machine.set_reverse_y_axis(reverse_y)
        machine.wcs_origin_is_workarea_origin = False
        machine.wcs_offsets["G54"] = (0.0, 0.0, 0.0)

        # Create an ops with a single point
        ops = Ops()
        ops.move_to(10.0, 20.0, 0.0)

        # Get encoder output
        prepared = machine._prepare_ops_for_encoding(ops)
        cmd = list(prepared.commands)[0]
        encoder_result = (cmd.end[0], cmd.end[1])

        # Get world_point_to_machine output
        space = machine.get_coordinate_space()
        w2m_result = space.world_point_to_machine(10.0, 20.0)

        # They should match
        assert w2m_result[0] == pytest.approx(encoder_result[0])
        assert w2m_result[1] == pytest.approx(encoder_result[1])

    # -- Workarea origin offset --
    # Margins are: (left, top, right, bottom) = (10, 20, 30, 40)
    # Machine extents: (100, 100)

    @pytest.mark.parametrize(
        "origin,expected_offset",
        [
            # Workarea origin in WORLD space
            (Origin.BOTTOM_LEFT, (10.0, 40.0)),
            (Origin.TOP_LEFT, (10.0, 80.0)),
            (Origin.BOTTOM_RIGHT, (70.0, 40.0)),
            (Origin.TOP_RIGHT, (70.0, 80.0)),
        ],
    )
    def test_workarea_origin_offset_all_corners(
        self, machine, origin, expected_offset
    ):
        """
        Verify get_workarea_origin_offset returns correct MACHINE space
        position for all origin corners.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(origin)

        result = machine.get_workarea_origin_offset()
        assert result == pytest.approx(expected_offset)

    # -- Workarea origin offset with reversed axes --
    # Axis reversal is a WORLD↔MACHINE display transformation concern.
    # The workarea origin offset is always in MACHINE space.

    def test_workarea_offset_unaffected_by_reverse_x(self, machine):
        """Axis reversal should not affect workarea origin offset."""
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)

        result_normal = machine.get_workarea_origin_offset()

        machine.set_reverse_x_axis(True)
        result_reversed = machine.get_workarea_origin_offset()

        assert result_normal == result_reversed
        assert result_normal == pytest.approx((10.0, 40.0))

    def test_workarea_offset_unaffected_by_reverse_y(self, machine):
        """Axis reversal should not affect workarea origin offset."""
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)

        result_normal = machine.get_workarea_origin_offset()

        machine.set_reverse_y_axis(True)
        result_reversed = machine.get_workarea_origin_offset()

        assert result_normal == result_reversed
        assert result_normal == pytest.approx((10.0, 40.0))

    # -- Reference offset --
    # Tests for Machine.get_reference_offset().
    # This method returns the appropriate offset for converting from
    # MACHINE coordinates to the user-visible REFERENCE coordinates.

    def test_returns_wcs_offset_when_wcs_mode(self, machine):
        """
        When wcs_origin_is_workarea_origin is False, should return WCS offset.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.wcs_origin_is_workarea_origin = False
        machine.set_active_wcs("G54")
        machine.wcs_offsets["G54"] = (25.0, 35.0, 0.0)

        result = machine.get_reference_offset()
        assert result == pytest.approx((25.0, 35.0, 0.0))

    def test_returns_workarea_offset_when_workarea_mode(self, machine):
        """
        When wcs_origin_is_workarea_origin is True, should return
        workarea origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = True

        result = machine.get_reference_offset()
        assert result == pytest.approx((10.0, 40.0, 0.0))

    def test_ignores_wcs_in_workarea_mode(self, machine):
        """
        In workarea mode, WCS offset should be ignored.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = True
        machine.wcs_offsets["G54"] = (999.0, 999.0, 0.0)

        result = machine.get_reference_offset()
        assert result == pytest.approx((10.0, 40.0, 0.0))

    # -- Position display in workarea mode --
    # Integration tests for position display in workarea mode.
    #
    # These tests simulate what the UI does:
    # 1. Get workpiece position in WORLD coords
    # 2. Convert to MACHINE coords via world_to_machine()
    # 3. Get workarea offset in WORLD space via get_reference_offset()
    # 4. Transform offset to MACHINE space via world_point_to_machine()
    # 5. Subtract to get REFERENCE coords
    #
    # Margins: left=10, top=20, right=30, bottom=40
    # Machine extents: (100, 100)

    def test_workarea_mode_bottom_left_origin(self, machine):
        """
        Workarea mode with BOTTOM_LEFT origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = True

        # Workarea origin in WORLD space: (10, 40)
        # Workpiece at WORLD (20, 50)
        pos_world = (20.0, 50.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # BOTTOM_LEFT: no transform, so (20, 50) - (10, 40) = (10, 10)
        assert pos_ref == pytest.approx((10.0, 10.0))

    def test_workarea_mode_top_left_origin(self, machine):
        """
        Workarea mode with TOP_LEFT origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.TOP_LEFT)
        machine.wcs_origin_is_workarea_origin = True

        # Workarea origin in WORLD space: (10, 80)
        # Workpiece at WORLD (20, 70) - 10 units from workarea origin
        pos_world = (20.0, 70.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # TOP_LEFT: Y flipped
        # pos_machine: (20, 100-70) = (20, 30)
        # offset_machine: (10, 100-80) = (10, 20)
        # pos_ref: (20-10, 30-20) = (10, 10)
        assert pos_ref == pytest.approx((10.0, 10.0))

    def test_workarea_mode_bottom_right_origin(self, machine):
        """
        Workarea mode with BOTTOM_RIGHT origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_RIGHT)
        machine.wcs_origin_is_workarea_origin = True

        # Workarea origin in WORLD space: (70, 40)
        # Workpiece at WORLD (80, 50) - 10 units from workarea origin
        pos_world = (80.0, 50.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # BOTTOM_RIGHT: X flipped
        # pos_machine: (100-80, 50) = (20, 50)
        # offset_machine: (100-70, 40) = (30, 40)
        # pos_ref: (20-30, 50-40) = (-10, 10)
        assert pos_ref == pytest.approx((-10.0, 10.0))

    def test_workarea_mode_top_right_origin(self, machine):
        """
        Workarea mode with TOP_RIGHT origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.TOP_RIGHT)
        machine.wcs_origin_is_workarea_origin = True

        # Workarea origin in WORLD space: (70, 80)
        # Workpiece at WORLD (80, 70) - 10 units from workarea origin
        pos_world = (80.0, 70.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # TOP_RIGHT: X and Y flipped
        # pos_machine: (100-80, 100-70) = (20, 30)
        # offset_machine: (100-70, 100-80) = (30, 20)
        # pos_ref: (20-30, 30-20) = (-10, 10)
        assert pos_ref == pytest.approx((-10.0, 10.0))

    def test_workarea_mode_with_reversed_x_axis(self, machine):
        """
        Workarea mode with BOTTOM_LEFT origin and reversed X axis.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_x_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        # Workpiece at WORLD (20, 50)
        pos_world = (20.0, 50.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # BOTTOM_LEFT with reverse_x: X is negated
        # pos_machine: (-20, 50)
        # offset_machine: (-10, 40)
        # pos_ref: (-20 - (-10), 50 - 40) = (-10, 10)
        assert pos_ref == pytest.approx((-10.0, 10.0))

    def test_workarea_mode_with_reversed_y_axis(self, machine):
        """
        Workarea mode with BOTTOM_LEFT origin and reversed Y axis.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_y_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        # Workpiece at WORLD (20, 50)
        pos_world = (20.0, 50.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # BOTTOM_LEFT with reverse_y: Y is negated
        # pos_machine: (20, -50)
        # offset_machine: (10, -40)
        # pos_ref: (20 - 10, -50 - (-40)) = (10, -10)
        assert pos_ref == pytest.approx((10.0, -10.0))

    def test_workarea_mode_with_both_axes_reversed(self, machine):
        """
        Workarea mode with BOTTOM_LEFT origin and both axes reversed.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_x_axis(True)
        machine.set_reverse_y_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        # Workpiece at WORLD (20, 50)
        pos_world = (20.0, 50.0)
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, (0, 0))
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )
        # BOTTOM_LEFT with both reversed: X and Y negated
        # pos_machine: (-20, -50)
        # offset_machine: (-10, -40)
        # pos_ref: (-20 - (-10), -50 - (-40)) = (-10, -10)
        assert pos_ref == pytest.approx((-10.0, -10.0))

    # -- Full coordinate pipeline --
    # End-to-end tests for the full coordinate transformation pipeline.
    # These tests verify: WORLD → MACHINE → REFERENCE
    # The UI does this to display workpiece positions.

    def test_world_to_reference_bottom_left(self, machine):
        """
        Full pipeline with BOTTOM_LEFT origin.
        WORLD coords are normalized (Y-up, origin bottom-left).
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = True

        # Workpiece at WORLD (15, 45) with size (10, 10)
        pos_world = (15.0, 45.0)
        size = (10.0, 10.0)

        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, size)
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )

        # BOTTOM_LEFT: no flip, so MACHINE == WORLD for items
        # pos_machine = (15, 45)
        # offset_world = (10, 40), offset_machine = (10, 40)
        # pos_ref = (15 - 10, 45 - 40) = (5, 5)
        assert pos_machine == pytest.approx((15.0, 45.0))
        assert offset_machine == pytest.approx((10.0, 40.0))
        assert pos_ref == pytest.approx((5.0, 5.0))

    def test_world_to_reference_top_left(self, machine):
        """
        Full pipeline with TOP_LEFT origin.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.TOP_LEFT)
        machine.wcs_origin_is_workarea_origin = True

        pos_world = (15.0, 45.0)
        size = (10.0, 10.0)

        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, size)
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )

        # TOP_LEFT: y_axis_down=True
        # pos_machine: mx=15, my=100-45-10=45
        # offset_world = (10, 80) - workarea origin in WORLD space
        # offset_machine: mx=10, my=100-80=20
        # pos_ref = (15-10, 45-20) = (5, 25)
        assert pos_machine == pytest.approx((15.0, 45.0))
        assert offset_machine == pytest.approx((10.0, 20.0))
        assert pos_ref == pytest.approx((5.0, 25.0))

    def test_world_to_reference_with_reverse_x(self, machine):
        """
        Full pipeline with BOTTOM_LEFT origin and reversed X axis.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_x_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        pos_world = (15.0, 45.0)
        size = (10.0, 10.0)

        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, size)
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )

        # BOTTOM_LEFT with reverse_x
        # pos_machine: (15, 45) -> (-15, 45) after flip
        # offset_world: (10, 40)
        # offset_machine: (10, 40) -> (-10, 40) after flip
        # pos_ref: (-15, 45) - (-10, 40) = (-5, 5)
        assert pos_machine == pytest.approx((-15.0, 45.0))
        assert offset_machine == pytest.approx((-10.0, 40.0))
        assert pos_ref == pytest.approx((-5.0, 5.0))

    def test_world_to_reference_with_reverse_y(self, machine):
        """
        Full pipeline with BOTTOM_LEFT origin and reversed Y axis.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_y_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        pos_world = (15.0, 45.0)
        size = (10.0, 10.0)

        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, size)
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )

        # BOTTOM_LEFT with reverse_y
        # pos_machine: (15, 45) -> (15, -45) after flip
        # offset_world: (10, 40)
        # offset_machine: (10, 40) -> (10, -40) after flip
        # pos_ref: (15, -45) - (10, -40) = (5, -5)
        assert pos_machine == pytest.approx((15.0, -45.0))
        assert offset_machine == pytest.approx((10.0, -40.0))
        assert pos_ref == pytest.approx((5.0, -5.0))

    # -- UI and G-code alignment --
    # These tests verify that the UI and G-code encoder produce the
    # same reference coordinates when axis reversal is enabled.
    #
    # The fix: world_point_to_machine() applies axis reversal, and the
    # offset is transformed through it before subtraction.

    def test_offset_transform_matches_encoder(self, machine):
        """
        Verify that transforming the offset through world_point_to_machine()
        produces the same result as the encoder's offset calculation.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_x_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        # get_reference_offset returns WORLD space offset
        ref_offset = machine.get_reference_offset()
        assert ref_offset[0] == pytest.approx(10.0)

        # Transform through world_point_to_machine to get MACHINE space offset
        space = machine.get_coordinate_space()
        offset_machine = space.world_point_to_machine(
            ref_offset[0], ref_offset[1]
        )

        # Encoder uses x_offset = -ml = -10 for BOTTOM_LEFT with reverse_x
        # world_point_to_machine flips: 10 -> -10
        assert offset_machine[0] == pytest.approx(-10.0)

    def test_ui_matches_gcode_with_reverse_x(self, machine):
        """
        Verify that UI calculation now matches G-code encoder output.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_work_margins(10.0, 20.0, 30.0, 40.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.set_reverse_x_axis(True)
        machine.wcs_origin_is_workarea_origin = True

        pos_world = (15.0, 45.0)
        size = (10.0, 10.0)

        # What the UI now calculates:
        space = machine.get_coordinate_space()
        pos_machine = space.world_item_to_machine(pos_world, size)
        offset_world = machine.get_reference_offset()
        offset_machine = space.world_point_to_machine(
            offset_world[0], offset_world[1]
        )
        ui_pos_ref = (
            pos_machine[0] - offset_machine[0],
            pos_machine[1] - offset_machine[1],
        )

        # What G-code encoder produces:
        # Transform: x' = -x, so (15, 45) -> (-15, 45)
        # Offset: -10 (negated for reverse_x)
        # Result: -15 - (-10) = -5
        assert ui_pos_ref[0] == pytest.approx(-5.0)
        assert ui_pos_ref[1] == pytest.approx(5.0)

    # -- Position display in WCS mode --
    # Regression tests for issue #190: Position display in WCS mode
    # (wcs_origin_is_workarea_origin=False) must correctly apply the WCS
    # offset to convert from MACHINE coordinates to REFERENCE coordinates.

    def test_wcs_offset_subtracted_from_machine_pos(self, machine):
        """
        With G55 active and WCO(30,30,0), a machine at MPos(30,30,0)
        should display reference position (0,0,0).
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G55"
        machine.wcs_offsets["G55"] = (30.0, 30.0, 0.0)

        ref_offset = machine.get_reference_offset()
        assert ref_offset == pytest.approx((30.0, 30.0, 0.0))

        machine_pos = (30.0, 30.0)
        space = machine.get_coordinate_space()
        offset_machine = space.world_point_to_machine(
            ref_offset[0], ref_offset[1]
        )
        ref_pos = (
            machine_pos[0] - offset_machine[0],
            machine_pos[1] - offset_machine[1],
        )
        assert ref_pos == pytest.approx((0.0, 0.0))

    def test_wcs_offset_with_nonzero_work_pos(self, machine):
        """
        G55 with WCO(50,30,0), machine at MPos(60,40,0).
        WPos = MPos - WCO = (10,10,0).
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G55"
        machine.wcs_offsets["G55"] = (50.0, 30.0, 0.0)

        ref_offset = machine.get_reference_offset()
        assert ref_offset == pytest.approx((50.0, 30.0, 0.0))

        machine_pos = (60.0, 40.0)
        space = machine.get_coordinate_space()
        offset_machine = space.world_point_to_machine(
            ref_offset[0], ref_offset[1]
        )
        ref_pos = (
            machine_pos[0] - offset_machine[0],
            machine_pos[1] - offset_machine[1],
        )
        assert ref_pos == pytest.approx((10.0, 10.0))

    def test_wcs_offset_zero_gives_machine_pos(self, machine):
        """
        With zero WCS offset, reference position equals machine position.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G54"
        machine.wcs_offsets["G54"] = (0.0, 0.0, 0.0)

        ref_offset = machine.get_reference_offset()
        assert ref_offset == pytest.approx((0.0, 0.0, 0.0))

        machine_pos = (25.0, 45.0)
        space = machine.get_coordinate_space()
        offset_machine = space.world_point_to_machine(
            ref_offset[0], ref_offset[1]
        )
        ref_pos = (
            machine_pos[0] - offset_machine[0],
            machine_pos[1] - offset_machine[1],
        )
        assert ref_pos == pytest.approx((25.0, 45.0))

    def test_wcs_offset_with_top_left_origin(self, machine):
        """
        WCS mode with TOP_LEFT origin and G55 WCO(20,10,0).
        Machine at MPos(30,50,0).
        TOP_LEFT flips Y: offset_machine Y = 100 - 10 = 90.
        ref_pos = (30-20, 50-90) = (10, -40).
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.TOP_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G55"
        machine.wcs_offsets["G55"] = (20.0, 10.0, 0.0)

        ref_offset = machine.get_reference_offset()
        assert ref_offset == pytest.approx((20.0, 10.0, 0.0))

        machine_pos = (30.0, 50.0)
        space = machine.get_coordinate_space()
        offset_machine = space.world_point_to_machine(
            ref_offset[0], ref_offset[1]
        )
        ref_pos = (
            machine_pos[0] - offset_machine[0],
            machine_pos[1] - offset_machine[1],
        )
        assert ref_pos == pytest.approx((10.0, -40.0))

    # -- Command offset Z --
    # Tests that get_command_offset always returns Z=0 regardless of WCS
    # Z offset. The WCS Z offset is handled by the controller (via the
    # G54-G59 preamble command), not by subtracting from G-code coords.

    def test_wcs_z_offset_not_subtracted(self, machine):
        """
        WCS Z offset should not appear in command offset.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.wcs_offsets["G54"] = (10.0, 20.0, 30.0)

        space = machine.get_coordinate_space()
        offset = space.get_command_offset(
            wcs_offset=machine.get_active_wcs_offset(),
            wcs_is_workarea_origin=False,
        )

        assert offset[0] == pytest.approx(10.0)
        assert offset[1] == pytest.approx(20.0)
        assert offset[2] == pytest.approx(0.0)

    def test_wcs_zero_z_offset(self, machine):
        """
        With Z=0 WCS offset, command Z should also be 0.
        """
        machine.set_axis_extents(100.0, 100.0)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.wcs_offsets["G54"] = (10.0, 20.0, 0.0)

        space = machine.get_coordinate_space()
        offset = space.get_command_offset(
            wcs_offset=machine.get_active_wcs_offset(),
            wcs_is_workarea_origin=False,
        )

        assert offset[2] == pytest.approx(0.0)
