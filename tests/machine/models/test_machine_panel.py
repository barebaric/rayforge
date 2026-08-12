"""
Unit tests for the MachinePanel display-facing facade.
"""

from typing import cast

import numpy as np
import pytest
from blinker import Signal

from rayforge.machine.models.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.machine_panel import (
    MachinePanel,
    PanelOrientation,
)


class _StubMachine:
    """Minimal machine stand-in for panel unit tests."""

    def __init__(self, space: MachineSpace):
        self._space = space
        self.changed = Signal()

    def get_coordinate_space(self) -> MachineSpace:
        return self._space

    @property
    def axis_extents(self) -> tuple[float, float]:
        return self._space.extents


def _panel(**space_kwargs) -> MachinePanel:
    orientation = space_kwargs.pop("orientation", PanelOrientation.NATIVE)
    space = MachineSpace(**space_kwargs)
    panel = MachinePanel(cast(Machine, _StubMachine(space)))
    panel._orientation = orientation
    return panel


class TestMachinePanelDisplayProperties:
    """Tests for display-facing properties under rotation."""

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
            orientation=PanelOrientation.ROTATED_LEFT,
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
            orientation=PanelOrientation.ROTATED_RIGHT,
        )
        assert view.origin is expected

    @pytest.mark.parametrize(
        "orientation, expected_origin",
        [
            (
                PanelOrientation.ROTATED_LEFT,
                OriginCorner.BOTTOM_RIGHT,
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
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
            orientation=orientation,
        )
        assert view.origin is expected_origin
        assert view.x_axis_right == (
            expected_origin
            in (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT)
        )
        assert view.y_axis_down == (
            expected_origin in (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT)
        )

    @pytest.mark.parametrize("orientation", list(PanelOrientation))
    def test_axis_negative_swaps_under_rotation(self, orientation):
        """Rotation swaps which native reversal flag drives each
        displayed axis; NATIVE keeps them identity."""
        view = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            reverse_x=True,
            reverse_y=False,
            orientation=orientation,
        )
        if orientation == PanelOrientation.NATIVE:
            assert view.x_axis_negative is True
            assert view.y_axis_negative is False
        else:
            assert view.x_axis_negative is False
            assert view.y_axis_negative is True


class TestMachinePanelComposedTransforms:
    """Tests for the composed world<->machine transforms that include
    the panel rotation on top of the native MachineSpace."""

    @pytest.mark.parametrize("orientation", list(PanelOrientation))
    def test_native_panel_matches_space(self, orientation):
        """In NATIVE orientation the panel's composed transforms must
        be identical to the underlying space's native transforms."""
        if orientation != PanelOrientation.NATIVE:
            return
        space = MachineSpace(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            reverse_x=True,
            reverse_y=False,
        )
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            reverse_x=True,
            reverse_y=False,
            orientation=orientation,
        )

        assert np.allclose(
            panel.get_world_to_machine_matrix(),
            space.get_world_to_machine_matrix(),
        )
        assert panel.world_point_to_machine(50.0, 60.0) == (
            space.world_point_to_machine(50.0, 60.0)
        )
        assert panel.machine_point_to_world(10.0, 20.0) == (
            space.machine_point_to_world(10.0, 20.0)
        )

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            (
                PanelOrientation.ROTATED_LEFT,
                (50.0, 700.0),
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                (350.0, 100.0),
            ),
        ],
    )
    def test_rotation_point_direction(self, orientation, expected):
        """Pin the rotation direction so a consistent mirror (left
        swapped with right) cannot pass the round-trip test alone.

        BL origin, no reversal, extents (400, 800):
        ROTATED_LEFT maps (x, y) -> (y, 800 - x);
        ROTATED_RIGHT maps it -> (400 - y, x).
        """
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            orientation=orientation,
        )
        assert panel.world_point_to_machine(100.0, 50.0) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize("orientation", list(PanelOrientation))
    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_point_round_trip(self, orientation, origin, reverse_x, reverse_y):
        """world_point_to_machine and machine_point_to_world must be
        exact inverses for every orientation/origin/reversal combo."""
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
        panel = _panel(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
            orientation=orientation,
        )

        world_point = (123.25, 77.5)
        machine_point = panel.world_point_to_machine(*world_point)
        result = panel.machine_point_to_world(*machine_point)
        assert result == (
            pytest.approx(world_point[0]),
            pytest.approx(world_point[1]),
        )

    @pytest.mark.parametrize("orientation", list(PanelOrientation))
    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_item_round_trip(self, orientation, origin, reverse_x, reverse_y):
        """world_item_to_machine and machine_item_to_world must be
        exact inverses for every orientation/origin/reversal combo."""
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
        panel = _panel(
            origin=origin,
            x_positive_direction=x_direction,
            y_positive_direction=y_direction,
            extents=(400.0, 800.0),
            reverse_x=reverse_x,
            reverse_y=reverse_y,
            orientation=orientation,
        )

        world_pos = (20.0, 30.0)
        item_size = (50.0, 70.0)
        machine_pos = panel.world_item_to_machine(world_pos, item_size)
        result = panel.machine_item_to_world(machine_pos, item_size)
        assert result == pytest.approx(world_pos)

    @pytest.mark.parametrize(
        "orientation, expected_extents",
        [
            (PanelOrientation.NATIVE, (400.0, 800.0)),
            (PanelOrientation.ROTATED_LEFT, (800.0, 400.0)),
            (PanelOrientation.ROTATED_RIGHT, (800.0, 400.0)),
        ],
    )
    def test_extents_swap(self, orientation, expected_extents):
        """Presented extents swap width/height under rotation."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            orientation=orientation,
        )
        assert panel.extents == expected_extents

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            (PanelOrientation.NATIVE, (10.0, 20.0, 30.0, 40.0)),
            (
                PanelOrientation.ROTATED_LEFT,
                (20.0, 30.0, 40.0, 10.0),
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                (40.0, 10.0, 20.0, 30.0),
            ),
        ],
    )
    def test_margins_rotate(self, orientation, expected):
        """Presented margins rotate edge order under rotation."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            orientation=orientation,
        )
        assert panel.margins == expected

    @pytest.mark.parametrize(
        "orientation, expected_native, expected_presented",
        [
            (
                PanelOrientation.NATIVE,
                (50.0, 60.0, 0.0),
                (50.0, 60.0, 0.0),
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                (50.0, 60.0, 0.0),
                (60.0, 50.0, 0.0),
            ),
        ],
    )
    def test_axis_label_origin_swap(
        self, orientation, expected_native, expected_presented
    ):
        """Axis label origin swaps x/y under rotation."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            orientation=orientation,
        )
        native = panel.space.get_axis_label_origin(
            wcs_offset=(50.0, 60.0, 0.0)
        )
        assert native == pytest.approx(expected_native)
        presented = panel.get_axis_label_origin(wcs_offset=(50.0, 60.0, 0.0))
        assert presented == pytest.approx(expected_presented)


class TestMachinePanelOrientationState:
    """Tests for orientation state management via a real Machine."""

    def test_set_orientation_sends_changed(self, test_machine_and_config):
        machine, _ = test_machine_and_config
        received = []

        def on_changed(*a, **kw):
            received.append(True)

        machine.changed.connect(on_changed)
        machine.set_panel_orientation(PanelOrientation.ROTATED_LEFT)
        assert machine.panel_orientation is PanelOrientation.ROTATED_LEFT
        assert len(received) == 1

    def test_set_orientation_noop_same_value(self, test_machine_and_config):
        machine, _ = test_machine_and_config
        machine.set_panel_orientation(PanelOrientation.NATIVE)
        received = []

        def on_changed(*a, **kw):
            received.append(True)

        machine.changed.connect(on_changed)
        machine.set_panel_orientation(PanelOrientation.NATIVE)
        assert len(received) == 0

    def test_supports_rotary(self, test_machine_and_config):
        machine, _ = test_machine_and_config
        assert machine.panel.supports_rotary is True
        machine.set_panel_orientation(PanelOrientation.ROTATED_LEFT)
        assert machine.panel.supports_rotary is False

    def test_serialization_round_trip(self, test_machine_and_config):
        machine, _ = test_machine_and_config
        machine.set_panel_orientation(PanelOrientation.ROTATED_RIGHT)
        data = machine.to_dict()
        assert (
            data["machine"]["panel_orientation"]
            == PanelOrientation.ROTATED_RIGHT.value
        )
        from rayforge.context import get_context

        restored = Machine.from_dict(data, get_context())
        assert restored.panel_orientation is PanelOrientation.ROTATED_RIGHT

    def test_serialization_default_native(self, test_machine_and_config):
        machine, _ = test_machine_and_config
        data = machine.to_dict()
        assert data["machine"]["panel_orientation"] == "native"
