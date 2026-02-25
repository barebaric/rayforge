import pytest
from unittest.mock import MagicMock, patch
from rayforge.machine.models.machine import Machine, Origin
from rayforge.ui_gtk.canvas2d.surface import WorkSurface

# Machine dimensions for testing
WIDTH, HEIGHT = 200, 200

# A test point's distance from the origin walls
OFFSET = 10.0

# (origin, rev_x, rev_y, machine_coords, expected_canvas_coords)
POSITIONING_SCENARIOS = [
    # --- Origin: BOTTOM_LEFT ---
    # Positive Axes: Machine(10,10) -> Canvas(10,10)
    (Origin.BOTTOM_LEFT, False, False, (OFFSET, OFFSET), (OFFSET, OFFSET)),
    # Negative Axes: Machine(-10,-10) -> Canvas(10,10)
    (Origin.BOTTOM_LEFT, True, True, (-OFFSET, -OFFSET), (OFFSET, OFFSET)),
    # --- Origin: TOP_LEFT ---
    # Positive Axes: Machine(10,10) -> Canvas(10, 190)
    (
        Origin.TOP_LEFT,
        False,
        False,
        (OFFSET, OFFSET),
        (OFFSET, HEIGHT - OFFSET),
    ),
    # Negative Axes: Machine(-10,-10) -> Canvas(10, 190)
    (
        Origin.TOP_LEFT,
        True,
        True,
        (-OFFSET, -OFFSET),
        (OFFSET, HEIGHT - OFFSET),
    ),
    # --- Origin: TOP_RIGHT ---
    # Positive Axes: Machine(10,10) -> Canvas(190, 190)
    (
        Origin.TOP_RIGHT,
        False,
        False,
        (OFFSET, OFFSET),
        (WIDTH - OFFSET, HEIGHT - OFFSET),
    ),
    # Negative Axes: Machine(-10,-10) -> Canvas(190, 190)
    (
        Origin.TOP_RIGHT,
        True,
        True,
        (-OFFSET, -OFFSET),
        (WIDTH - OFFSET, HEIGHT - OFFSET),
    ),
    # --- Origin: BOTTOM_RIGHT ---
    # Positive Axes: Machine(10,10) -> Canvas(190, 10)
    (
        Origin.BOTTOM_RIGHT,
        False,
        False,
        (OFFSET, OFFSET),
        (WIDTH - OFFSET, OFFSET),
    ),
    # Negative Axes: Machine(-10,-10) -> Canvas(190, 10)
    (
        Origin.BOTTOM_RIGHT,
        True,
        True,
        (-OFFSET, -OFFSET),
        (WIDTH - OFFSET, OFFSET),
    ),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "origin, rev_x, rev_y, machine_coords, expected_canvas_coords",
    POSITIONING_SCENARIOS,
)
def test_laser_dot_absolute_positioning(
    ui_context_initializer,
    origin,
    rev_x,
    rev_y,
    machine_coords,
    expected_canvas_coords,
):
    """
    Verifies that a given machine coordinate maps to the correct absolute
    canvas coordinate, accounting for all combinations of machine origin
    and axis direction (positive/negative coordinates).
    """
    # 1. Configure the Machine
    machine = Machine(ui_context_initializer)
    machine.set_axis_extents(WIDTH, HEIGHT)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(rev_x)
    machine.set_reverse_y_axis(rev_y)

    # 2. Mock UI dependencies
    mock_editor = MagicMock()
    mock_editor.doc = MagicMock()
    mock_window = MagicMock()

    # 3. Instantiate WorkSurface
    with patch("rayforge.ui_gtk.canvas2d.surface.SketchEditor"):
        surface = WorkSurface(mock_editor, mock_window, machine)

        # 4. Set Laser Dot to the machine test coordinate
        m_x, m_y = machine_coords
        surface.set_laser_dot_position(m_x, m_y)

        # 5. Retrieve the dot's top-left canvas coordinate from its transform
        tx, ty = surface._laser_dot.transform.get_translation()

        # 6. Calculate the dot's CENTER position on the canvas.
        # This is the crucial step to fix the previous test's assertion errors.
        # The element's transform gives its top-left corner, but it's
        # positioned relative to its center.
        center_x = tx + surface._laser_dot.width / 2
        center_y = ty + surface._laser_dot.height / 2

        # 7. Assert that the calculated center matches the expected position
        exp_x, exp_y = expected_canvas_coords
        assert abs(center_x - exp_x) < 1e-5, (
            f"FAIL: O={origin.value},RX={rev_x} | "
            f"MachineX({m_x}) -> CanvasX({center_x}), Expected({exp_x})"
        )
        assert abs(center_y - exp_y) < 1e-5, (
            f"FAIL: O={origin.value},RY={rev_y} | "
            f"MachineY({m_y}) -> CanvasY({center_y}), Expected({exp_y})"
        )
