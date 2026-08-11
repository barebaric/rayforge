"""
Unit tests for the MachineView display-facing facade.
"""

import pytest

from rayforge.machine.models.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)
from rayforge.machine.models.machine_view import MachineView


class TestMachineView:
    """Tests for the display-facing MachineView facade."""

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_native_matches_legacy_derivation(
        self, origin, reverse_x, reverse_y
    ):
        """In NATIVE orientation the view must reproduce the old
        origin/reversal derivation that the UI previously inlined."""
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
        view = MachineView(space)
        assert view.origin is origin
        assert view.x_axis_right == (
            origin in (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT)
        )
        assert view.y_axis_down == (
            origin in (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT)
        )
        assert view.x_axis_negative == reverse_x
        assert view.y_axis_negative == reverse_y

    @pytest.mark.parametrize(
        "native, expected",
        [
            (OriginCorner.BOTTOM_LEFT, OriginCorner.BOTTOM_RIGHT),
            (OriginCorner.TOP_LEFT, OriginCorner.BOTTOM_LEFT),
            (OriginCorner.TOP_RIGHT, OriginCorner.TOP_LEFT),
            (OriginCorner.BOTTOM_RIGHT, OriginCorner.TOP_RIGHT),
        ],
    )
    def test_rotated_left_origin_mapping(self, native, expected):
        """ROTATED_LEFT rotates the visible origin corner one step
        counter-clockwise around the bed."""
        space = MachineSpace(
            origin=native,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            workspace_orientation=WorkspaceOrientation.ROTATED_LEFT,
        )
        assert MachineView(space).origin is expected

    @pytest.mark.parametrize(
        "native, expected",
        [
            (OriginCorner.BOTTOM_LEFT, OriginCorner.TOP_LEFT),
            (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT),
            (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT),
            (OriginCorner.BOTTOM_RIGHT, OriginCorner.BOTTOM_LEFT),
        ],
    )
    def test_rotated_right_origin_mapping(self, native, expected):
        """ROTATED_RIGHT rotates the visible origin corner one step
        clockwise around the bed."""
        space = MachineSpace(
            origin=native,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            workspace_orientation=WorkspaceOrientation.ROTATED_RIGHT,
        )
        assert MachineView(space).origin is expected

    @pytest.mark.parametrize(
        "orientation, expected_origin",
        [
            (
                WorkspaceOrientation.ROTATED_LEFT,
                OriginCorner.BOTTOM_RIGHT,
            ),
            (
                WorkspaceOrientation.ROTATED_RIGHT,
                OriginCorner.TOP_LEFT,
            ),
        ],
    )
    def test_bottom_left_axis_direction_under_rotation(
        self, orientation, expected_origin
    ):
        """A BOTTOM_LEFT bed, rotated, moves its visible origin so the
        displayed axis flags flip consistently with the mapped corner."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            workspace_orientation=orientation,
        )
        view = MachineView(space)
        assert view.origin is expected_origin
        assert view.x_axis_right == (
            expected_origin
            in (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT)
        )
        assert view.y_axis_down == (
            expected_origin in (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT)
        )

    @pytest.mark.parametrize("orientation", list(WorkspaceOrientation))
    def test_axis_negative_swaps_under_rotation(self, orientation):
        """Rotation swaps which native reversal flag drives each
        displayed axis; NATIVE keeps them identity."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            reverse_x=True,
            reverse_y=False,
            workspace_orientation=orientation,
        )
        view = MachineView(space)
        if orientation == WorkspaceOrientation.NATIVE:
            assert view.x_axis_negative is True
            assert view.y_axis_negative is False
        else:
            assert view.x_axis_negative is False
            assert view.y_axis_negative is True
