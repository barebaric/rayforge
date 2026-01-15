import pytest
from unittest.mock import MagicMock
from rayforge.machine.models.machine import Machine, Origin


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

    return s


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
    machine.dimensions = (100.0, 100.0)
    machine.origin = scenario["origin"]
    machine.reverse_x_axis = scenario["reverse_x"]
    machine.reverse_y_axis = scenario["reverse_y"]

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

    # Attach machine to surface
    surface.machine = machine

    # Trigger the update method which calculates position
    surface._on_wcs_updated(machine)

    # Assert the element was moved to the expected pixel/mm location
    surface._work_origin_element.set_pos.assert_called_with(
        *scenario["expected"]
    )
