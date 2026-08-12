"""
Unit tests for the MachinePanel display-facing facade.
"""

from typing import cast

import numpy as np
import pytest

from rayforge.machine.models.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.machine_panel import MachinePanel


class _StubMachine:
    """Minimal machine stand-in: MachinePanel only needs
    get_coordinate_space()."""

    def __init__(self, space: MachineSpace):
        self._space = space

    def get_coordinate_space(self) -> MachineSpace:
        return self._space


def _panel(**space_kwargs) -> MachinePanel:
    space = MachineSpace(**space_kwargs)
    return MachinePanel(cast(Machine, _StubMachine(space)))


class TestMachinePanel:
    """Tests for the display-facing MachinePanel facade."""

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_native_matches_legacy_derivation(
        self, origin, reverse_x, reverse_y
    ):
        """In NATIVE orientation the panel must reproduce the old
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
        view = _panel(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
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
        view = _panel(
            origin=native,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            workspace_orientation=WorkspaceOrientation.ROTATED_LEFT,
        )
        assert view.origin is expected

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
        view = _panel(
            origin=native,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            workspace_orientation=WorkspaceOrientation.ROTATED_RIGHT,
        )
        assert view.origin is expected

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
        view = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            workspace_orientation=orientation,
        )
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
        view = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            reverse_x=True,
            reverse_y=False,
            workspace_orientation=orientation,
        )
        if orientation == WorkspaceOrientation.NATIVE:
            assert view.x_axis_negative is True
            assert view.y_axis_negative is False
        else:
            assert view.x_axis_negative is False
            assert view.y_axis_negative is True

    @pytest.mark.parametrize("orientation", list(WorkspaceOrientation))
    def test_proxy_methods_match_space(self, orientation):
        """Every composed proxy on MachinePanel must return the same
        result as the underlying MachineSpace for all orientations."""
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            reverse_x=True,
            reverse_y=False,
            workspace_orientation=orientation,
        )
        panel = MachinePanel(cast(Machine, _StubMachine(space)))

        # Matrix proxies
        assert np.allclose(
            panel.get_world_to_machine_matrix(),
            space.get_world_to_machine_matrix(),
        )
        assert np.allclose(
            panel.get_machine_to_world_matrix(),
            space.get_machine_to_world_matrix(),
        )

        # Point proxies
        assert panel.world_point_to_machine(50.0, 60.0) == (
            space.world_point_to_machine(50.0, 60.0)
        )
        assert panel.machine_point_to_world(10.0, 20.0) == (
            space.machine_point_to_world(10.0, 20.0)
        )

        # Item proxies
        assert panel.world_item_to_machine((10.0, 20.0), (30.0, 40.0)) == (
            space.world_item_to_machine((10.0, 20.0), (30.0, 40.0))
        )
        assert panel.machine_item_to_world((5.0, 15.0), (25.0, 35.0)) == (
            space.machine_item_to_world((5.0, 15.0), (25.0, 35.0))
        )

        # Rect / size / position proxies
        assert panel.get_workarea_world_rect() == (
            space.get_workarea_world_rect()
        )
        assert panel.workarea_size == space.workarea_size
        assert panel.world_position_from_origin(
            10.0, 20.0, (30.0, 40.0)
        ) == space.world_position_from_origin(10.0, 20.0, (30.0, 40.0))

        # Axis label origin
        assert panel.get_axis_label_origin(
            (1.0, 2.0, 0.0), False
        ) == space.get_axis_label_origin((1.0, 2.0, 0.0), False)
        assert panel.get_axis_label_origin(
            (0.0, 0.0, 0.0), True
        ) == space.get_axis_label_origin((0.0, 0.0, 0.0), True)

        # Workarea origin / command offset
        assert panel.get_workarea_origin_in_machine() == (
            space.get_workarea_origin_in_machine()
        )
        assert panel.get_command_offset(
            (1.0, 2.0, 0.0), False
        ) == space.get_command_offset((1.0, 2.0, 0.0), False)
        assert panel.get_command_offset(
            (0.0, 0.0, 0.0), True
        ) == space.get_command_offset((0.0, 0.0, 0.0), True)
