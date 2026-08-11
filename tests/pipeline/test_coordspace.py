"""
Unit tests for coordinate space classes.
"""

import numpy as np
import pytest

from rayforge.pipeline.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
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

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_scalar_and_matrix_paths_agree(self, origin, reverse_x, reverse_y):
        """The two world->machine implementations must not drift apart.

        `world_point_to_machine` applies the origin/reversal rules as
        scalar branches (used by the UI), while the encoding pipeline
        consumes `get_world_to_machine_matrix`. They are separate code
        paths describing the same transform, so a change to one that is
        not mirrored in the other would silently desynchronise what the
        canvas shows from what the machine is told to do.
        """
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
        matrix = space.get_world_to_machine_matrix()
        width, height = space.extents

        # Includes both bed corners and asymmetric interior points, so a
        # missing translation term cannot pass by symmetry.
        for world_x, world_y in (
            (0.0, 0.0),
            (width, height),
            (width * 0.31, height * 0.17),
            (width * 0.73, height * 0.61),
        ):
            scalar = space.world_point_to_machine(world_x, world_y)
            transformed = matrix @ np.array([world_x, world_y, 0.0, 1.0])
            assert scalar == pytest.approx((transformed[0], transformed[1])), (
                f"paths disagree at ({world_x}, {world_y})"
            )


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
