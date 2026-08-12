"""
Unit tests for coordinate space classes.
"""

import numpy as np
import pytest

from rayforge.machine.models.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)


class TestMachineSpace:
    """Tests for MachineSpace coordinate system."""

    def test_default_properties(self):
        """Default MachineSpace should match WorldSpace orientation."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
        )

        assert space.origin == OriginCorner.BOTTOM_LEFT
        assert not space.x_reversed
        assert not space.y_reversed

    def test_top_left_origin(self):
        """MachineSpace with top-left origin should have Y-down."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
        )

        assert space.origin == OriginCorner.TOP_LEFT
        assert space.y_positive_direction == AxisDirection.POSITIVE_DOWN

    def test_bottom_right_origin(self):
        """MachineSpace with bottom-right origin should have X-left."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
        )

        assert space.origin == OriginCorner.BOTTOM_RIGHT
        assert space.x_positive_direction == AxisDirection.POSITIVE_LEFT

    def test_workarea_size(self):
        """Workarea size should account for margins."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(200.0, 200.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        width, height = space.workarea_size

        assert width == 160.0
        assert height == 140.0

    def test_get_workarea_origin_in_machine_bottom_left(self):
        """Workarea origin for BL origin machine should be at margins."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(200.0, 200.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        x, y = space.get_workarea_origin_in_machine()

        assert x == 10.0
        assert y == 40.0

    def test_get_workarea_origin_in_machine_top_left(self):
        """Workarea origin for TL origin machine should be at margins."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(200.0, 200.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        x, y = space.get_workarea_origin_in_machine()

        assert x == 10.0
        assert y == 20.0

    def test_get_workarea_origin_in_machine_top_right(self):
        """Workarea origin for TR origin machine should be at margins."""
        space = MachineSpace(
            origin=OriginCorner.TOP_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(200.0, 200.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        x, y = space.get_workarea_origin_in_machine()

        assert x == 30.0
        assert y == 20.0

    def test_transform_top_left_to_world(self):
        """Top-left origin machine coords should transform to world."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(100.0, 100.0),
        )

        x, y = space.transform_point_to_world(0.0, 0.0, (100.0, 100.0))

        assert x == 0.0
        assert y == 100.0

    def test_transform_top_left_bottom_right_to_world(self):
        """Bottom-right in TL origin should be (100, 0) in world."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(100.0, 100.0),
        )

        x, y = space.transform_point_to_world(100.0, 100.0, (100.0, 100.0))

        assert x == 100.0
        assert y == 0.0

    def test_world_to_machine_matrix_identity(self):
        """Test getting pipeline world-to-machine transform matrix."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
        )
        matrix = space.get_world_to_machine_matrix()
        np.testing.assert_array_almost_equal(matrix, np.identity(4))

    def test_world_to_machine_matrix_reversed_y(self):
        """Test pipeline matrix generation handling sign flips (reverse_y)."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
            reverse_y=True,
        )
        matrix = space.get_world_to_machine_matrix()
        expected = np.identity(4, dtype=np.float64)
        expected[1, 1] = -1.0
        np.testing.assert_array_almost_equal(matrix, expected)

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_axis_label_origin_native_matches_raw_wcs_offset(
        self, origin, reverse_x, reverse_y
    ):
        """Regression: in the native orientation, the WCS branch of
        get_axis_label_origin must return the raw offset unchanged,
        exactly as it did before workspace rotation existed."""
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin
            in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin
            in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )
            else AxisDirection.POSITIVE_UP
        )
        space = MachineSpace(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )

        assert space.get_axis_label_origin(
            wcs_offset=(50.0, 60.0, 0.0)
        ) == pytest.approx((50.0, 60.0, 0.0))

    def test_axis_label_origin_native_workarea_matches_margins(self):
        """Regression: native workarea-origin labels come from margins."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        assert space.get_axis_label_origin(
            wcs_is_workarea_origin=True
        ) == pytest.approx((10.0, 20.0, 0.0))


class TestCoordinateSpaceTransforms:
    """Tests for coordinate transformation matrices."""

    def test_bottom_left_origin_no_reverse(self):
        """BL origin with no reversal should be identity."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
        )

        matrix = space.get_transform_to_world((100.0, 100.0))

        expected = np.identity(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_top_left_origin_y_down(self):
        """TL origin with Y-down should flip and translate Y."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
        )

        matrix = space.get_transform_to_world((100.0, 100.0))

        expected = np.identity(4, dtype=np.float64)
        expected[1, 1] = -1.0
        expected[1, 3] = 100.0
        np.testing.assert_array_almost_equal(matrix, expected)


class TestMachineSpaceItemTransforms:
    """Tests for item position transforms with bounding box adjustment."""

    def test_world_item_to_machine_bottom_left(self):
        """Bottom-Left origin: identity transform for items."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (10, 10)

    def test_world_item_to_machine_top_left(self):
        """Top-Left origin: Y is flipped for items."""
        space = MachineSpace(
            origin=OriginCorner.TOP_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(100.0, 100.0),
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (10, 80)

    def test_world_item_to_machine_top_right(self):
        """Top-Right origin: both X and Y are flipped for items."""
        space = MachineSpace(
            origin=OriginCorner.TOP_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(100.0, 100.0),
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (80, 80)

    def test_world_item_to_machine_bottom_right(self):
        """Bottom-Right origin: X is flipped for items."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (80, 10)

    def test_machine_item_to_world_top_right(self):
        """Top-Right origin: inverse transform for items."""
        space = MachineSpace(
            origin=OriginCorner.TOP_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
            extents=(100.0, 100.0),
        )
        item_size = (10, 10)

        res = space.machine_item_to_world((80, 80), item_size)
        assert res == (10, 10)

        res = space.machine_item_to_world((0, 0), item_size)
        assert res == (90, 90)

    def test_world_item_to_machine_with_reverse_x(self):
        """Bottom-Left origin with reverse_x: X is negated."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
            reverse_x=True,
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (-10, 10)

    def test_world_item_to_machine_with_reverse_y(self):
        """Bottom-Left origin with reverse_y: Y is negated."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 100.0),
            reverse_y=True,
        )
        item_size = (10, 10)

        res = space.world_item_to_machine((10, 10), item_size)
        assert res == (10, -10)

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_item_round_trip_all_native_configurations(
        self, origin, reverse_x, reverse_y
    ):
        """world_item_to_machine and machine_item_to_world must be
        exact inverses across every native origin/reversal combo, for
        nonzero (asymmetric) item sizes."""
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin
            in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin
            in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )
            else AxisDirection.POSITIVE_UP
        )
        space = MachineSpace(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
        pos = (20.0, 30.0)
        size = (50.0, 70.0)
        machine_pos = space.world_item_to_machine(pos, size)
        assert space.machine_item_to_world(machine_pos, size) == (
            pytest.approx(pos[0]),
            pytest.approx(pos[1]),
        )

    @pytest.mark.parametrize("orientation", list(WorkspaceOrientation))
    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_workspace_transform_round_trip_all_configurations(
        self, orientation, origin, reverse_x, reverse_y
    ):
        """Point and item transforms round-trip for every orientation,
        origin, and reversal combination. The probe and item are placed
        within the smaller presented dimension so they stay on the bed
        in both native and rotated orientations."""
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin
            in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin
            in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )
            else AxisDirection.POSITIVE_UP
        )
        space = MachineSpace(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
            workspace_orientation=orientation,
        )

        world_point = (123.25, 77.5)
        machine_point = space.world_point_to_machine(*world_point)
        assert space.machine_point_to_world(*machine_point) == (
            pytest.approx(world_point[0]),
            pytest.approx(world_point[1]),
        )

        world_pos = (20.0, 30.0)
        item_size = (50.0, 70.0)
        machine_pos = space.world_item_to_machine(world_pos, item_size)
        assert space.machine_item_to_world(
            machine_pos, item_size
        ) == pytest.approx(world_pos)

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            # BL origin, no reversal, native extents (400, 800).
            # Hand-verified: ROTATED_LEFT maps (x, y) to (y, 800 - x);
            # ROTATED_RIGHT maps it to (400 - y, x).
            (WorkspaceOrientation.ROTATED_LEFT, (50.0, 700.0)),
            (WorkspaceOrientation.ROTATED_RIGHT, (350.0, 100.0)),
        ],
    )
    def test_workspace_rotation_maps_point_direction(
        self, orientation, expected
    ):
        """Pins the rotation direction so a consistent mirror (left
        swapped with right) cannot pass the round-trip test alone."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            workspace_orientation=orientation,
        )
        assert space.world_point_to_machine(100.0, 50.0) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        "origin, reverse_x, reverse_y, pos, expected",
        [
            # extents=(100, 100), size=(10, 10); hand-verified against
            # the scalar origin/size-corner formula.
            (OriginCorner.TOP_RIGHT, True, False, (10, 10), (-80, 80)),
            (
                OriginCorner.BOTTOM_RIGHT,
                True,
                False,
                (10, 10),
                (-80, 10),
            ),
            (OriginCorner.TOP_LEFT, False, True, (10, 10), (10, -80)),
        ],
    )
    def test_world_item_to_machine_origin_with_reversal(
        self, origin, reverse_x, reverse_y, pos, expected
    ):
        """Forward pin: non-bottom-left origins combined with axis
        reversal (configs the per-origin and per-reversal tests above
        do not cover together)."""
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin
            in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin
            in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )
            else AxisDirection.POSITIVE_UP
        )
        space = MachineSpace(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(100.0, 100.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
        assert space.world_item_to_machine(pos, (10, 10)) == expected

    @pytest.mark.parametrize(
        "origin, reverse_x, reverse_y, pos, expected",
        [
            # extents=(100, 100), size=(10, 10); hand-verified.
            (
                OriginCorner.BOTTOM_RIGHT,
                False,
                False,
                (80, 10),
                (10, 10),
            ),
            (
                OriginCorner.BOTTOM_LEFT,
                True,
                False,
                (-10, 10),
                (10, 10),
            ),
        ],
    )
    def test_machine_item_to_world_beyond_top_right(
        self, origin, reverse_x, reverse_y, pos, expected
    ):
        """Forward pin for machine_item_to_world across origins/reversals
        beyond the single TOP_RIGHT case covered above."""
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin
            in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin
            in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )
            else AxisDirection.POSITIVE_UP
        )
        space = MachineSpace(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(100.0, 100.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
        assert space.machine_item_to_world(pos, (10, 10)) == expected

    def test_bottom_right_origin_x_left(self):
        """BR origin with X-left should flip and translate X."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
        )

        matrix = space.get_transform_to_world((100.0, 100.0))

        expected = np.identity(4, dtype=np.float64)
        expected[0, 0] = -1.0
        expected[0, 3] = 100.0
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_top_right_origin_both_reversed(self):
        """TR origin should flip and translate both X and Y."""
        space = MachineSpace(
            origin=OriginCorner.TOP_RIGHT,
            x_positive_direction=AxisDirection.POSITIVE_LEFT,
            y_positive_direction=AxisDirection.POSITIVE_DOWN,
        )

        matrix = space.get_transform_to_world((100.0, 100.0))

        expected = np.identity(4, dtype=np.float64)
        expected[0, 0] = -1.0
        expected[0, 3] = 100.0
        expected[1, 1] = -1.0
        expected[1, 3] = 100.0
        np.testing.assert_array_almost_equal(matrix, expected)


class TestGCodePipeline:
    """End-to-end tests of the G-code coordinate pipeline.

    Simulates: WCS zeroed at a known world position → workpiece at a
    known world position → verify G-code.  Expected values are computed
    independently, NOT from the same matrices being tested.
    """

    @staticmethod
    def _make_space(origin, reverse_x, reverse_y):
        return MachineSpace(
            origin=origin,
            x_positive_direction=(
                AxisDirection.POSITIVE_LEFT
                if origin
                in (
                    OriginCorner.TOP_RIGHT,
                    OriginCorner.BOTTOM_RIGHT,
                )
                else AxisDirection.POSITIVE_RIGHT
            ),
            y_positive_direction=(
                AxisDirection.POSITIVE_DOWN
                if origin
                in (
                    OriginCorner.TOP_LEFT,
                    OriginCorner.TOP_RIGHT,
                )
                else AxisDirection.POSITIVE_UP
            ),
            extents=(100.0, 100.0),
            margins=(0.0, 0.0, 0.0, 0.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    @pytest.mark.parametrize(
        "wcs_world",
        [(10.0, 20.0), (50.0, 50.0)],
    )
    def test_wcs_zeroed_then_workpiece_at_origin(
        self, origin, reverse_x, reverse_y, wcs_world
    ):
        """WCS is zeroed at world position ``wcs_world`` (simulating
        click-to-zero).  A workpiece sits at world (0, 0).

        The G-code for the workpiece must equal the workpiece's position
        relative to the WCS origin, expressed in the machine's
        coordinate system (i.e. after origin-corner transform and axis
        sign-flip, but NOT after the WCS translation).

        Concretely:
          wcs_offset = world_point_to_machine(wcs_world)
          gcode      = world_point_to_machine(workpiece) - wcs_offset

        Both terms are in the same (sign-flipped) machine space, so the
        subtraction yields the correct G-code coordinate.
        """
        space = self._make_space(origin, reverse_x, reverse_y)

        # Simulate click-to-zero: WCS offset = machine coords of click
        wcs_offset = space.world_point_to_machine(*wcs_world)
        cmd = space.get_command_offset(
            wcs_offset=(wcs_offset[0], wcs_offset[1], 0.0),
            wcs_is_workarea_origin=False,
        )

        # Workpiece at world (0, 0)
        workpiece_machine = space.world_point_to_machine(0.0, 0.0)
        gcode_x = workpiece_machine[0] - cmd[0]
        gcode_y = workpiece_machine[1] - cmd[1]

        # Independently compute expected: the workpiece's position
        # relative to WCS in machine space.
        expected_x = workpiece_machine[0] - wcs_offset[0]
        expected_y = workpiece_machine[1] - wcs_offset[1]

        assert gcode_x == pytest.approx(expected_x, abs=1e-9)
        assert gcode_y == pytest.approx(expected_y, abs=1e-9)

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_wcs_at_origin_gcode_is_zero(self, origin, reverse_x, reverse_y):
        """When WCS is zeroed at world (0,0), a workpiece at (0,0) must
        produce G-code (0, 0)."""
        space = self._make_space(origin, reverse_x, reverse_y)
        wcs_offset = space.world_point_to_machine(0.0, 0.0)
        cmd = space.get_command_offset(
            wcs_offset=(wcs_offset[0], wcs_offset[1], 0.0),
            wcs_is_workarea_origin=False,
        )
        machine_pt = space.world_point_to_machine(0.0, 0.0)
        gcode_x = machine_pt[0] - cmd[0]
        gcode_y = machine_pt[1] - cmd[1]
        assert gcode_x == pytest.approx(0.0, abs=1e-9)
        assert gcode_y == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    @pytest.mark.parametrize(
        "manual_wcs",
        [(10.0, 20.0), (-15.0, -25.0)],
    )
    def test_manual_wcs_offset_consistency(
        self, origin, reverse_x, reverse_y, manual_wcs
    ):
        """When a WCS offset is entered manually (in machine
        coordinates), the WCS origin in world space must produce
        G-code (0, 0).

        The WCS offset is what the controller stores: the machine
        coordinate of the WCS origin.  We find the world position of
        that machine coordinate using the *inverse* of the full w2m
        matrix, then verify the pipeline produces zero there.
        """
        space = self._make_space(origin, reverse_x, reverse_y)
        w2m = space.get_world_to_machine_matrix()
        m2w = np.linalg.inv(w2m)

        cmd = space.get_command_offset(
            wcs_offset=(manual_wcs[0], manual_wcs[1], 0.0),
            wcs_is_workarea_origin=False,
        )

        # WCS origin in world space: inverse-transform the machine coord
        wcs_world = m2w @ np.array([manual_wcs[0], manual_wcs[1], 0.0, 1.0])

        # G-code at that world point
        machine_pt = w2m @ wcs_world
        gcode_x = machine_pt[0] - cmd[0]
        gcode_y = machine_pt[1] - cmd[1]

        assert gcode_x == pytest.approx(0.0, abs=1e-9)
        assert gcode_y == pytest.approx(0.0, abs=1e-9)
