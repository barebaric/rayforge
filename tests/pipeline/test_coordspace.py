"""
Unit tests for coordinate space classes.
"""

import numpy as np

from rayforge.pipeline.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
    PixelSpace,
    WorkareaSpace,
    WorldSpace,
)


class TestWorldSpace:
    """Tests for WorldSpace coordinate system."""

    def test_default_properties(self):
        """WorldSpace should have bottom-left origin, Y-up, X-right."""
        space = WorldSpace()

        assert space.origin == OriginCorner.BOTTOM_LEFT
        assert space.x_positive_direction == AxisDirection.POSITIVE_RIGHT
        assert space.y_positive_direction == AxisDirection.POSITIVE_UP
        assert not space.x_reversed
        assert not space.y_reversed

    def test_transform_identity(self):
        """WorldSpace to world should be identity transform."""
        space = WorldSpace()
        extents = (100.0, 200.0)

        matrix = space.get_transform_to_world(extents)

        expected = np.identity(4, dtype=np.float64)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_transform_point_identity(self):
        """Points in world space should remain unchanged."""
        space = WorldSpace()
        extents = (100.0, 200.0)

        x, y = space.transform_point_to_world(50.0, 100.0, extents)

        assert x == 50.0
        assert y == 100.0


class TestPixelSpace:
    """Tests for PixelSpace coordinate system."""

    def test_default_properties(self):
        """PixelSpace should have top-left origin, Y-down, X-right."""
        space = PixelSpace()

        assert space.origin == OriginCorner.TOP_LEFT
        assert space.x_positive_direction == AxisDirection.POSITIVE_RIGHT
        assert space.y_positive_direction == AxisDirection.POSITIVE_DOWN
        assert not space.x_reversed
        assert space.y_reversed

    def test_to_millimeters_top_left(self):
        """Top-left pixel should map to (0, height) in mm."""
        space = PixelSpace(dimensions=(100, 100))
        mm_dims = (50.0, 50.0)

        x, y = space.to_millimeters(0, 0, mm_dims)

        assert x == 0.0
        assert y == 50.0

    def test_to_millimeters_bottom_left(self):
        """Bottom-left pixel should map to (0, 0) in mm."""
        space = PixelSpace(dimensions=(100, 100))
        mm_dims = (50.0, 50.0)

        x, y = space.to_millimeters(0, 100, mm_dims)

        assert x == 0.0
        assert y == 0.0

    def test_to_millimeters_center(self):
        """Center pixel should map to center in mm."""
        space = PixelSpace(dimensions=(100, 100))
        mm_dims = (50.0, 50.0)

        x, y = space.to_millimeters(50, 50, mm_dims)

        assert x == 25.0
        assert y == 25.0

    def test_to_millimeters_top_right(self):
        """Top-right pixel should map to (width, height) in mm."""
        space = PixelSpace(dimensions=(100, 100))
        mm_dims = (50.0, 50.0)

        x, y = space.to_millimeters(100, 0, mm_dims)

        assert x == 50.0
        assert y == 50.0


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
        assert y == 180.0

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

        assert x == 170.0
        assert y == 180.0

    def test_to_command_coords_with_wcs(self):
        """Command coords should subtract WCS offset."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(200.0, 200.0),
        )

        x, y = space.to_command_coords(
            100.0, 50.0, wcs_offset=(20.0, 10.0, 0.0)
        )

        assert x == 80.0
        assert y == 40.0

    def test_to_command_coords_with_workarea_origin(self):
        """Command coords should subtract workarea origin when mode set."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(200.0, 200.0),
            margins=(10.0, 20.0, 30.0, 40.0),
        )

        x, y = space.to_command_coords(
            100.0, 50.0,
            wcs_offset=(0.0, 0.0, 0.0),
            wcs_is_workarea_origin=True,
        )

        assert x == 90.0
        assert y == 10.0

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


class TestWorkareaSpace:
    """Tests for WorkareaSpace coordinate system."""

    def test_extents_from_margins(self):
        """WorkareaSpace extents should be machine extents minus margins."""
        space = WorkareaSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(160.0, 140.0),
        )

        assert space.extents == (160.0, 140.0)


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
