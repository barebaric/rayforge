from unittest.mock import MagicMock

import pytest
from blinker import Signal

from rayforge.machine.models.coordspace import MachineSpace
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.machine_panel import PanelOrientation


@pytest.fixture
def mock_work_origin():
    element = MagicMock()
    return element


@pytest.fixture
def surface(mock_work_origin):
    """
    Creates a WorkSurface instance bypassing the GTK initialization
    (which would require a display connection) and mocking internal elements.
    """
    from rayforge.ui_gtk.canvas2d.surface import WorkSurface

    # Bypass GTK __init__ chain
    s = WorkSurface.__new__(WorkSurface)

    # Inject dependencies that would usually be created in __init__
    s._work_origin_element = mock_work_origin
    s.queue_draw = MagicMock()
    s.width_mm = 100.0
    s.height_mm = 100.0
    s._update_extent_frame = MagicMock()
    s._connected_doc = None
    s._connected_layer = None
    s._active_layer_wcs_conn = None

    active_layer = MagicMock()
    active_layer.rotary_enabled = False
    active_layer.wcs = None
    s.editor = MagicMock()
    s.editor.doc.active_layer = active_layer

    return s


@pytest.mark.parametrize(
    "orientation, delta, expected",
    [
        (PanelOrientation.NATIVE, (1.0, 0.0), (1.0, 0.0)),
        (PanelOrientation.NATIVE, (0.0, 1.0), (0.0, 1.0)),
        (PanelOrientation.ROTATED_RIGHT, (1.0, 0.0), (0.0, 1.0)),
        (PanelOrientation.ROTATED_RIGHT, (0.0, 1.0), (-1.0, 0.0)),
        (PanelOrientation.ROTATED_LEFT, (1.0, 0.0), (0.0, -1.0)),
        (PanelOrientation.ROTATED_LEFT, (0.0, 1.0), (1.0, 0.0)),
    ],
)
@pytest.mark.ui
def test_arrow_key_nudge_unrotates_panel_delta(
    surface, orientation, delta, expected
):
    """
    Arrow-key nudging must rotate the presented (PANEL) movement vector
    into WORLD space before applying it, so items move in the on-screen
    direction under a rotated panel.
    """
    from gi.repository import Gdk

    from rayforge.core.item import DocItem
    from rayforge.machine.models.coordspace import (
        AxisDirection,
        MachineSpace,
        OriginCorner,
    )
    from rayforge.machine.models.machine import Machine
    from rayforge.machine.models.machine_panel import MachinePanel

    surface._space_pressed = False
    surface.transform_initiated = Signal()

    item = MagicMock(spec=DocItem)
    elem = MagicMock()
    elem.data = item
    surface.get_selected_elements = MagicMock(return_value=[elem])

    space = MachineSpace(
        origin=OriginCorner.BOTTOM_LEFT,
        x_positive_direction=AxisDirection.POSITIVE_RIGHT,
        y_positive_direction=AxisDirection.POSITIVE_UP,
        extents=(400.0, 300.0),
        reverse_x=False,
        reverse_y=False,
    )
    machine = MagicMock(spec=Machine)
    machine.changed = Signal()
    machine.axis_extents = (400.0, 300.0)
    machine.get_coordinate_space.return_value = space
    panel = MachinePanel(machine)
    panel._orientation = orientation
    machine.panel = panel
    surface.machine = machine

    keyval = {1.0: Gdk.KEY_Right, -1.0: Gdk.KEY_Left}.get(delta[0])
    if keyval is None:
        keyval = {1.0: Gdk.KEY_Up, -1.0: Gdk.KEY_Down}[delta[1]]

    handled = surface.on_key_pressed(None, keyval, 0, 0)

    assert handled is True
    surface.editor.transform.nudge_items.assert_called_once_with(
        [item], *expected
    )


@pytest.mark.parametrize(
    "scenario",
    [
        # Case 1: Bottom-Left Origin, Positive Axis (Standard 3D Printer /
        # Cartesian)
        # Origin is at visual (0,0).
        # WCS (20, 20) -> Effective distance (20, 20)
        # Canvas (Bottom-Left 0,0) -> (20, 20)
        {
            "origin": Origin.BOTTOM_LEFT,
            "reverse_x": False,
            "reverse_y": False,
            "wcs": (20, 20, 0),
            "expected": (20, 20),
        },
        # Case 2: Bottom-Left Origin, Negative Axis (Standard CNC logic in
        # positive quadrant view)
        # Origin is at visual (0,0).
        # WCS (-20, -20) -> Negative axis implies distance is negated: 20, 20
        # Canvas (Bottom-Left 0,0) -> (20, 20)
        {
            "origin": Origin.BOTTOM_LEFT,
            "reverse_x": True,
            "reverse_y": True,
            "wcs": (-20, -20, 0),
            "expected": (20, 20),
        },
        # Case 3: Top-Right Origin, Negative Axis (Standard CNC with Homing
        # Top-Right)
        # Origin is at visual (Width, Height).
        # WCS (-20, -20) -> Effective distance 20, 20 from origin.
        # Canvas X = Width - 20 = 80
        # Canvas Y = Height - 20 = 80
        {
            "origin": Origin.TOP_RIGHT,
            "reverse_x": True,
            "reverse_y": True,
            "wcs": (-20, -20, 0),
            "expected": (80, 80),
        },
        # Case 4: Top-Right Origin, Positive Axis
        # Origin is at visual (Width, Height).
        # WCS (20, 20) -> Effective distance 20, 20 from origin.
        # Canvas X = Width - 20 = 80
        # Canvas Y = Height - 20 = 80
        {
            "origin": Origin.TOP_RIGHT,
            "reverse_x": False,
            "reverse_y": False,
            "wcs": (20, 20, 0),
            "expected": (80, 80),
        },
        # Case 5: Top-Left Origin, Mixed Axis (e.g. Laser with Y-Down)
        # Origin is at visual (0, Height).
        # X is positive right, Y is negative down.
        # WCS (20, -20).
        # Eff X = 20. Canvas X = 0 + 20 = 20.
        # Eff Y = -(-20) = 20. Canvas Y = Height - 20 = 80.
        {
            "origin": Origin.TOP_LEFT,
            "reverse_x": False,
            "reverse_y": True,
            "wcs": (20, -20, 0),
            "expected": (20, 80),
        },
    ],
)
@pytest.mark.ui
def test_wcs_visual_marker_location(surface, scenario):
    """
    Verifies that the Work Origin marker is placed at the correct visual
    coordinates on the canvas for various machine configurations.

    The Canvas coordinate system is assumed to be standard Cartesian with
    (0,0) at the Bottom-Left.
    """
    machine = MagicMock(spec=Machine)
    machine.changed = Signal()
    machine.axis_extents = (100.0, 100.0)
    machine.origin = scenario["origin"]
    machine.reverse_x_axis = scenario["reverse_x"]
    machine.reverse_y_axis = scenario["reverse_y"]
    machine.wcs_origin_is_workarea_origin = False
    machine.work_margins = (0.0, 0.0, 0.0, 0.0)

    # Configure derivative properties based on logic in Machine model
    machine.y_axis_down = scenario["origin"] in (
        Origin.TOP_LEFT,
        Origin.TOP_RIGHT,
    )
    machine.x_axis_right = scenario["origin"] in (
        Origin.TOP_RIGHT,
        Origin.BOTTOM_RIGHT,
    )

    machine.get_active_wcs_offset.return_value = scenario["wcs"]

    # Create a real MachineSpace from the mock machine
    space = MachineSpace.from_machine(machine)
    machine.get_coordinate_space.return_value = space

    # Wire the panel to delegate to the real space
    from rayforge.machine.models.machine_panel import MachinePanel

    machine.panel = MachinePanel(machine)

    # Attach machine to surface
    surface.machine = machine

    # Trigger the update method which calculates position
    surface._on_wcs_updated(machine)

    # Assert the element was moved to the expected pixel/mm location
    surface._work_origin_element.set_pos.assert_called_with(
        *scenario["expected"]
    )


@pytest.mark.ui
def test_set_stock_visible(surface):
    """The view toggle propagates to every stock element."""
    surface._stock_visible = True
    stock_elem = MagicMock()
    surface.find_by_type = MagicMock(return_value=[stock_elem])

    surface.set_stock_visible(False)
    stock_elem.set_view_visible.assert_called_once_with(False)
    assert surface._stock_visible is False

    surface.set_stock_visible(True)
    stock_elem.set_view_visible.assert_called_with(True)
    assert surface._stock_visible is True


@pytest.mark.ui
def test_set_stock_visible_noop_when_unchanged(surface):
    """Calling the toggle with the current state is a no-op."""
    surface._stock_visible = False
    surface.find_by_type = MagicMock()

    surface.set_stock_visible(False)
    surface.find_by_type.assert_not_called()


def _wire_move_head_mode(surface):
    """Attach the move-head mode state to a bare WorkSurface."""
    surface._click_to_zero_mode = False
    surface._move_head_mode = False
    surface.right_click_context = None
    surface.machine = None
    surface.edit_context = None
    surface.move_head_requested = Signal()
    surface.move_head_cancelled = Signal()
    surface._get_world_coords = MagicMock(return_value=(10.0, 20.0))
    surface.set_cursor = MagicMock()
    return surface


def _wire_machine(surface):
    machine = MagicMock()
    machine.panel.panel_point_to_machine.return_value = (5.0, 6.0)
    surface.machine = machine
    return machine


def _collect_coords(signal):
    """Connects a receiver and returns the collected (x, y) list."""
    emissions = []

    def on_requested(sender, **kw):
        emissions.append((kw["x"], kw["y"]))

    signal.connect(on_requested, weak=False)
    return emissions


@pytest.mark.ui
def test_click_in_move_head_mode_emits_machine_coords(surface):
    from gi.repository import Gdk

    _wire_move_head_mode(surface)
    machine = _wire_machine(surface)
    surface._move_head_mode = True

    emissions = _collect_coords(surface.move_head_requested)

    gesture = MagicMock()
    gesture.get_button.return_value = Gdk.BUTTON_PRIMARY
    surface.on_button_press(gesture, 1, 100.0, 100.0)

    assert emissions == [(5.0, 6.0)]
    machine.panel.panel_point_to_machine.assert_called_once_with(10.0, 20.0)


@pytest.mark.ui
def test_click_in_move_head_mode_without_machine_emits_world_coords(
    surface,
):
    from gi.repository import Gdk

    _wire_move_head_mode(surface)
    surface._move_head_mode = True

    emissions = _collect_coords(surface.move_head_requested)

    gesture = MagicMock()
    gesture.get_button.return_value = Gdk.BUTTON_PRIMARY
    surface.on_button_press(gesture, 1, 100.0, 100.0)

    assert emissions == [(10.0, 20.0)]


@pytest.mark.ui
def test_click_not_emitted_when_move_head_mode_inactive(surface, mocker):
    from gi.repository import Gdk

    from rayforge.ui_gtk.canvas.worldsurface import WorldSurface

    _wire_move_head_mode(surface)
    mocker.patch.object(WorldSurface, "on_button_press", return_value=None)

    emissions = _collect_coords(surface.move_head_requested)

    gesture = MagicMock()
    gesture.get_button.return_value = Gdk.BUTTON_PRIMARY
    surface.on_button_press(gesture, 1, 100.0, 100.0)

    assert emissions == []


@pytest.mark.ui
def test_right_click_cancels_move_head_mode(surface):
    _wire_move_head_mode(surface)
    surface._move_head_mode = True

    emissions = []

    def on_cancelled(sender):
        emissions.append(sender)

    surface.move_head_cancelled.connect(on_cancelled)

    surface.on_right_click_pressed(MagicMock(), 1, 5.0, 5.0)

    assert len(emissions) == 1


@pytest.mark.ui
def test_set_move_head_mode_resets_cursor(surface):
    _wire_move_head_mode(surface)

    surface.set_move_head_mode(True)
    assert surface._move_head_mode is True

    surface.set_move_head_mode(False)
    assert surface._move_head_mode is False
    surface.set_cursor.assert_called_with(None)
