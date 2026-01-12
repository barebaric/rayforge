import pytest
from unittest.mock import MagicMock, patch
from rayforge.machine.models.machine import Machine, Origin
from rayforge.ui_gtk.canvas2d.surface import WorkSurface


# Parameters for the test:
# (Origin, Reverse X, Reverse Y, Expected Delta X Sign, Expected Delta Y Sign)
# Signs:
#   1 = Increasing (Right/Up),
#  -1 = Decreasing (Left/Down)
# in Canvas Coordinates.
SCENARIOS = [
    # 1. Machine Origin: Bottom Left, Axis Positive
    #    Expectation: Laser dot moves to top/right with increasing values.
    #    Canvas X increases (Right), Canvas Y increases (Up).
    (Origin.BOTTOM_LEFT, False, False, 1, 1),
    # 2. Machine Origin: Bottom Left, Axis Negative
    #    Expectation: Laser dot moves to bottom/left with increasing values.
    #    Canvas X decreases (Left), Canvas Y decreases (Down).
    (Origin.BOTTOM_LEFT, True, True, -1, -1),
    # 3. Machine Origin: Top Right, Axis Positive
    #    Expectation: Laser dot moves to bottom/left with increasing values.
    #    TR Origin means Machine (0,0) is at Canvas (W, H).
    #    Machine X+ (Left) -> Canvas X decreases.
    #    Machine Y+ (Down) -> Canvas Y decreases.
    (Origin.TOP_RIGHT, False, False, -1, -1),
    # 4. Machine Origin: Top Right, Axis Negative
    #    Expectation: Laser dot moves to top/right with increasing values.
    #    Inversion of Scenario 3.
    (Origin.TOP_RIGHT, True, True, 1, 1),
]

# Generate test parameters including G54 variations
TEST_PARAMS = []
for s in SCENARIOS:
    # Variant without G54
    TEST_PARAMS.append(s + (None,))
    # Variant with G54 set to (10, 20, 0)
    TEST_PARAMS.append(s + ((10.0, 20.0, 0.0),))


@pytest.mark.ui
@pytest.mark.parametrize(
    "origin, rev_x, rev_y, exp_dx, exp_dy, wcs_offset", TEST_PARAMS
)
def test_laser_dot_direction(
    ui_context_initializer,
    origin,
    rev_x,
    rev_y,
    exp_dx,
    exp_dy,
    wcs_offset,
):
    """
    Verifies that the laser dot on the WorkSurface moves in the visually
    correct direction based on the machine's origin and axis inversion
    settings.
    """
    # 1. Configure the Machine
    machine = Machine(ui_context_initializer)
    machine.set_dimensions(200, 200)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(rev_x)
    machine.set_reverse_y_axis(rev_y)

    if wcs_offset:
        machine.wcs_offsets["G54"] = wcs_offset
        machine.set_active_wcs("G54")

    # 2. Mock UI dependencies for WorkSurface
    mock_editor = MagicMock()
    mock_editor.doc = MagicMock()
    mock_window = MagicMock()

    # 3. Instantiate WorkSurface with SketchEditor patched out
    # We patch it where it is imported in surface.py to avoid GTK widget
    # parent issues
    with patch("rayforge.ui_gtk.canvas2d.surface.SketchEditor"):
        surface = WorkSurface(mock_editor, mock_window, machine)

        # 4. Set Initial Position (P1)
        # We use arbitrary machine coordinates.
        m_x1, m_y1 = 10.0, 10.0
        surface.set_laser_dot_position(m_x1, m_y1)

        # Retrieve P1 canvas coordinates from the dot element's transform
        # matrix
        # Note: .x and .y attributes on CanvasElement are initial values
        # and do not update with set_pos().
        tx1, ty1 = surface._laser_dot.transform.get_translation()

        # 5. Set Next Position (P2) - Increasing values
        m_x2, m_y2 = 20.0, 20.0
        surface.set_laser_dot_position(m_x2, m_y2)

        # Retrieve P2 canvas coordinates
        tx2, ty2 = surface._laser_dot.transform.get_translation()

        # 6. Calculate Deltas in Canvas Space
        delta_x = tx2 - tx1
        delta_y = ty2 - ty1

        # 7. Assert Directions
        # Check X Axis
        if exp_dx > 0:
            assert delta_x > 0, (
                f"Expected Laser Dot X to increase (move Right) for Origin "
                f"{origin} with ReverseX={rev_x}, but it decreased/stayed "
                f"same (dx={delta_x})."
            )
        else:
            assert delta_x < 0, (
                f"Expected Laser Dot X to decrease (move Left) for Origin "
                f"{origin} with ReverseX={rev_x}, but it increased/stayed "
                f"same (dx={delta_x})."
            )

        # Check Y Axis
        if exp_dy > 0:
            assert delta_y > 0, (
                f"Expected Laser Dot Y to increase (move Up) for Origin "
                f"{origin} with ReverseY={rev_y}, but it decreased/stayed "
                f"same (dy={delta_y})."
            )
        else:
            assert delta_y < 0, (
                f"Expected Laser Dot Y to decrease (move Down) for Origin "
                f"{origin} with ReverseY={rev_y}, but it increased/stayed "
                f"same (dy={delta_y})."
            )
