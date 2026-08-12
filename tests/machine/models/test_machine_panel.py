"""
Unit tests for the MachinePanel display-facing facade.
"""

from typing import ClassVar, cast

import numpy as np
import pytest
from blinker import Signal
from raygeo.geo.types import Point3D
from raygeo.ops.axis import Axis

from rayforge.machine.models.coordspace import (
    AxisDirection,
    MachineSpace,
    OriginCorner,
)
from rayforge.machine.models.machine import JogDirection, Machine
from rayforge.machine.models.machine_panel import (
    MachinePanel,
    PanelOrientation,
)
from rayforge.machine.models.zone import Zone, ZoneShape


class _StubMachine:
    """Minimal machine stand-in for panel unit tests."""

    def __init__(self, space: MachineSpace, reverse_z: bool = False):
        self._space = space
        self.reverse_z = reverse_z
        self.changed = Signal()
        self.nogo_zones: dict[str, Zone] = {}

    def get_coordinate_space(self) -> MachineSpace:
        return self._space

    @property
    def axis_extents(self) -> tuple[float, float]:
        return self._space.extents

    def has_custom_work_area(self) -> bool:
        ml, mt, mr, mb = self._space.margins
        return ml != 0 or mt != 0 or mr != 0 or mb != 0

    def get_reference_offset(self) -> Point3D:
        x, y = self._space.get_workarea_origin_in_machine()
        return (x, y, 0.0)

    def calculate_jog(self, direction: JogDirection, distance: float) -> float:
        """Mirror Machine.calculate_jog for panel equivalence tests."""
        origin = self._space.origin
        x_axis_right = origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )
        y_axis_down = origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )
        if direction == JogDirection.EAST:
            delta = -distance if x_axis_right else distance
            return -delta if self._space.reverse_x else delta
        if direction == JogDirection.WEST:
            delta = distance if x_axis_right else -distance
            return -delta if self._space.reverse_x else delta
        if direction == JogDirection.NORTH:
            delta = -distance if y_axis_down else distance
            return -delta if self._space.reverse_y else delta
        if direction == JogDirection.SOUTH:
            delta = distance if y_axis_down else -distance
            return -delta if self._space.reverse_y else delta
        if direction == JogDirection.UP:
            return -distance if self.reverse_z else distance
        if direction == JogDirection.DOWN:
            return distance if self.reverse_z else -distance
        return 0.0


def _panel(**space_kwargs) -> MachinePanel:
    orientation = space_kwargs.pop("orientation", PanelOrientation.NATIVE)
    reverse_z = space_kwargs.pop("reverse_z", False)
    space = MachineSpace(**space_kwargs)
    panel = MachinePanel(cast(Machine, _StubMachine(space, reverse_z)))
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

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            (PanelOrientation.NATIVE, (-10.0, -40.0, 400.0, 800.0)),
            (
                PanelOrientation.ROTATED_LEFT,
                (-20.0, -10.0, 800.0, 400.0),
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                (-40.0, -30.0, 800.0, 400.0),
            ),
        ],
    )
    def test_extent_frame_rotates(self, orientation, expected):
        """The extent frame follows presented margins and extents."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            orientation=orientation,
        )
        assert panel.extent_frame == pytest.approx(expected)

    def test_has_custom_work_area_false_without_margins(self):
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
        )
        assert panel.has_custom_work_area is False

    def test_has_custom_work_area_true_with_margins(self):
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            margins=(10.0, 0.0, 0.0, 0.0),
        )
        assert panel.has_custom_work_area is True

    @pytest.mark.parametrize(
        "origin, extents, margins, expected",
        [
            (
                OriginCorner.BOTTOM_LEFT,
                (400.0, 800.0),
                (10.0, 20.0, 30.0, 40.0),
                (190.0, 410.0),
            ),
            (
                OriginCorner.TOP_LEFT,
                (400.0, 800.0),
                (10.0, 20.0, 30.0, 40.0),
                (190.0, 410.0),
            ),
            (
                OriginCorner.TOP_RIGHT,
                (400.0, 800.0),
                (10.0, 20.0, 30.0, 40.0),
                (190.0, 410.0),
            ),
        ],
    )
    def test_work_area_center(self, origin, extents, margins, expected):
        """The work-area center lands at the workarea origin plus half
        the work-area size, regardless of origin corner."""
        panel = _panel(
            origin=origin,
            x_positive_direction=(
                AxisDirection.POSITIVE_LEFT
                if origin
                in (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT)
                else AxisDirection.POSITIVE_RIGHT
            ),
            y_positive_direction=(
                AxisDirection.POSITIVE_DOWN
                if origin in (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT)
                else AxisDirection.POSITIVE_UP
            ),
            extents=extents,
            margins=margins,
        )
        assert panel.work_area_center() == pytest.approx(expected)

    @pytest.mark.parametrize("orientation", list(PanelOrientation))
    def test_work_area_center_matches_workarea_world_rect(self, orientation):
        """The center equals the center of get_workarea_world_rect."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(400.0, 800.0),
            margins=(10.0, 20.0, 30.0, 40.0),
            orientation=orientation,
        )
        wx, wy, w, h = panel.get_workarea_world_rect()
        assert panel.work_area_center() == pytest.approx(
            (wx + w / 2, wy + h / 2)
        )


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


class TestMachinePanelCalculateJog:
    """Tests for rotation-aware visual jog direction mapping."""

    AXIS_FOR_DIRECTION: ClassVar[dict[JogDirection, Axis]] = {
        JogDirection.EAST: Axis.X,
        JogDirection.WEST: Axis.X,
        JogDirection.NORTH: Axis.Y,
        JogDirection.SOUTH: Axis.Y,
        JogDirection.UP: Axis.Z,
        JogDirection.DOWN: Axis.Z,
    }

    @staticmethod
    def _directions_for(origin):
        x_direction = (
            AxisDirection.POSITIVE_LEFT
            if origin in (OriginCorner.TOP_RIGHT, OriginCorner.BOTTOM_RIGHT)
            else AxisDirection.POSITIVE_RIGHT
        )
        y_direction = (
            AxisDirection.POSITIVE_DOWN
            if origin in (OriginCorner.TOP_LEFT, OriginCorner.TOP_RIGHT)
            else AxisDirection.POSITIVE_UP
        )
        return x_direction, y_direction

    @pytest.mark.parametrize("origin", list(OriginCorner))
    @pytest.mark.parametrize("reverse_x", [False, True])
    @pytest.mark.parametrize("reverse_y", [False, True])
    def test_native_matches_machine_calculate_jog(
        self, origin, reverse_x, reverse_y
    ):
        """NATIVE orientation reproduces the machine's per-axis jog
        calculation exactly."""
        x_dir, y_dir = self._directions_for(origin)
        panel = _panel(
            origin=origin,
            x_positive_direction=x_dir,
            y_positive_direction=y_dir,
            reverse_x=reverse_x,
            reverse_y=reverse_y,
        )
        for direction, axis in self.AXIS_FOR_DIRECTION.items():
            expected = panel.machine.calculate_jog(direction, 10.0)
            assert panel.calculate_jog(direction, 10.0) == {axis: expected}

    @pytest.mark.parametrize(
        "orientation, direction, expected",
        [
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.EAST,
                {Axis.Y: 10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.WEST,
                {Axis.Y: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.NORTH,
                {Axis.X: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.SOUTH,
                {Axis.X: 10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.EAST,
                {Axis.Y: -10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.WEST,
                {Axis.Y: 10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.NORTH,
                {Axis.X: 10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.SOUTH,
                {Axis.X: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.UP,
                {Axis.Z: 10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                JogDirection.DOWN,
                {Axis.Z: -10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.UP,
                {Axis.Z: 10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                JogDirection.DOWN,
                {Axis.Z: -10.0},
            ),
        ],
    )
    def test_rotated_direction_maps_to_orthogonal_axis(
        self, orientation, direction, expected
    ):
        """Under rotation, visual cardinals drive the orthogonal native
        axis; Z is unaffected."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            orientation=orientation,
        )
        assert panel.calculate_jog(direction, 10.0) == expected

    @pytest.mark.parametrize(
        "orientation, origin, direction, expected",
        [
            (
                PanelOrientation.ROTATED_RIGHT,
                OriginCorner.TOP_LEFT,
                JogDirection.EAST,
                {Axis.Y: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                OriginCorner.TOP_LEFT,
                JogDirection.NORTH,
                {Axis.X: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                OriginCorner.TOP_RIGHT,
                JogDirection.EAST,
                {Axis.Y: -10.0},
            ),
            (
                PanelOrientation.ROTATED_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
                JogDirection.NORTH,
                {Axis.X: 10.0},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                OriginCorner.TOP_RIGHT,
                JogDirection.NORTH,
                {Axis.X: -10.0},
            ),
        ],
    )
    def test_rotated_origin_flips_compose_with_rotation(
        self, orientation, origin, direction, expected
    ):
        """Origin corner flips compose with the rotation matrix."""
        x_dir, y_dir = self._directions_for(origin)
        panel = _panel(
            origin=origin,
            x_positive_direction=x_dir,
            y_positive_direction=y_dir,
            orientation=orientation,
        )
        assert panel.calculate_jog(direction, 10.0) == expected

    def test_z_direction_honors_machine_reversal(self):
        """UP/DOWN delegate to the machine's rotation-unaware jog."""
        panel = _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            reverse_z=True,
        )
        assert panel.calculate_jog(JogDirection.UP, 10.0) == {Axis.Z: -10.0}
        assert panel.calculate_jog(JogDirection.DOWN, 10.0) == {Axis.Z: 10.0}


def _make_rect_zone(x, y, w, h, name="Zone"):
    zone = Zone()
    zone.set_name(name)
    zone.params["x"] = x
    zone.params["y"] = y
    zone.params["w"] = w
    zone.params["h"] = h
    return zone


def _make_cylinder_zone(x, y, name="Cylinder"):
    zone = Zone()
    zone.set_name(name)
    zone.set_shape(ZoneShape.CYLINDER)
    zone.params["x"] = x
    zone.params["y"] = y
    return zone


class TestMachinePanelMachineToPanel:
    """Tests for the 90-degree bed rotation from MACHINE to panel."""

    @staticmethod
    def _panel(orientation):
        return _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 200.0),
            orientation=orientation,
        )

    def test_native_is_identity(self):
        panel = self._panel(PanelOrientation.NATIVE)
        assert panel.machine_to_panel == pytest.approx(np.identity(4))

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            (
                PanelOrientation.ROTATED_RIGHT,
                np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 100.0]]),
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                np.array([[0.0, -1.0, 0.0, 200.0], [1.0, 0.0, 0.0, 0.0]]),
            ),
        ],
    )
    def test_rotation_maps_machine_to_panel(self, orientation, expected):
        """The matrix rotates MACHINE-bed geometry into the panel
        presentation."""
        panel = self._panel(orientation)
        matrix = panel.machine_to_panel
        assert matrix[:2, :4] == pytest.approx(expected)

    def test_machine_to_panel_round_trips_with_panel_to_native(self):
        panel = self._panel(PanelOrientation.ROTATED_LEFT)
        product = panel.machine_to_panel @ panel._panel_to_native_matrix
        assert product == pytest.approx(np.identity(4))


class TestMachinePanelNogoZones:
    """Tests for rotation-projected no-go-zone copies."""

    @staticmethod
    def _panel(orientation=PanelOrientation.NATIVE):
        return _panel(
            origin=OriginCorner.BOTTOM_LEFT,
            x_positive_direction=AxisDirection.POSITIVE_RIGHT,
            y_positive_direction=AxisDirection.POSITIVE_UP,
            extents=(100.0, 200.0),
            orientation=orientation,
        )

    def test_native_returns_detached_copy(self):
        panel = self._panel()
        zone = _make_rect_zone(10, 20, 30, 40)
        panel.machine.nogo_zones[zone.uid] = zone

        projected = panel.nogo_zones
        assert projected[zone.uid] is not zone
        assert projected[zone.uid].params == zone.params

    @pytest.mark.parametrize(
        "orientation, expected",
        [
            (
                PanelOrientation.ROTATED_RIGHT,
                {"x": 20, "y": 60, "w": 40, "h": 30},
            ),
            (
                PanelOrientation.ROTATED_LEFT,
                {"x": 140, "y": 10, "w": 40, "h": 30},
            ),
        ],
    )
    def test_rect_projection_rotates(self, orientation, expected):
        panel = self._panel(orientation)
        zone = _make_rect_zone(10, 20, 30, 40)
        panel.machine.nogo_zones[zone.uid] = zone

        params = panel.nogo_zones[zone.uid].params
        assert params["x"] == pytest.approx(expected["x"])
        assert params["y"] == pytest.approx(expected["y"])
        assert params["w"] == pytest.approx(expected["w"])
        assert params["h"] == pytest.approx(expected["h"])

    def test_cylinder_projection_rotates_center(self):
        panel = self._panel(PanelOrientation.ROTATED_RIGHT)
        zone = _make_cylinder_zone(10, 20)
        panel.machine.nogo_zones[zone.uid] = zone

        params = panel.nogo_zones[zone.uid].params
        assert params["x"] == pytest.approx(20)
        assert params["y"] == pytest.approx(90)

    def test_projection_reflects_machine_state_changes(self):
        panel = self._panel()
        zone = _make_rect_zone(10, 20, 30, 40)
        panel.machine.nogo_zones[zone.uid] = zone

        first = panel.nogo_zones
        assert first[zone.uid].params["x"] == 10

        zone.set_param("x", 50)
        second = panel.nogo_zones
        assert second is not first
        assert second[zone.uid].params["x"] == 50

        panel._orientation = PanelOrientation.ROTATED_RIGHT
        third = panel.nogo_zones
        assert third is not second
        assert third[zone.uid].params["x"] == pytest.approx(20)
        assert third[zone.uid].params["y"] == pytest.approx(20)

    def test_zones_are_not_modified_by_projection(self):
        panel = self._panel(PanelOrientation.ROTATED_RIGHT)
        zone = _make_rect_zone(10, 20, 30, 40)
        panel.machine.nogo_zones[zone.uid] = zone

        projected = panel.nogo_zones
        assert projected[zone.uid] is not zone
        assert zone.params == {"x": 10, "y": 20, "w": 30, "h": 40}
