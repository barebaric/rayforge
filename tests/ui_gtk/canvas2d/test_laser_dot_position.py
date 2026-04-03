import pytest
from unittest.mock import MagicMock
from rayforge.machine.driver.driver import DeviceState
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


def _get_laser_dot_center(surface):
    """Helper to get the laser dot's center position on the canvas."""
    tx, ty = surface._laser_dot.transform.get_translation()
    center_x = tx + surface._laser_dot.width / 2
    center_y = ty + surface._laser_dot.height / 2
    return center_x, center_y


class TestLaserDotWithWCSOffset:
    """
    Tests that the laser dot is positioned at the machine coordinates
    (MPos), which is where the laser physically is. The WCS offset is
    handled separately by the grid/axis labels shifting, not by moving
    the laser dot.

    The fix for issue #190 ensures wcs_offsets is synced from device
    WCO on status reports, so the grid labels shift correctly and
    the laser dot (at MPos) visually aligns with the work coordinate
    grid.
    """

    @pytest.mark.ui
    def test_laser_dot_uses_machine_pos(self, ui_context_initializer):
        """
        The laser dot represents the laser's physical position and
        should always be placed at machine coordinates, regardless
        of WCS offset.

        G55 with WCO(30,30,0), machine at MPos(30,30,0).
        The laser dot should appear at canvas (30, 30).
        """
        machine = Machine(ui_context_initializer)
        machine.set_axis_extents(WIDTH, HEIGHT)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G55"
        machine.wcs_offsets["G55"] = (30.0, 30.0, 0.0)

        mock_editor = MagicMock()
        mock_editor.doc = MagicMock()
        mock_window = MagicMock()
        surface = WorkSurface(mock_editor, mock_window, machine)

        state = DeviceState(
            machine_pos=(30.0, 30.0, 0.0),
            work_pos=(0.0, 0.0, 0.0),
            wco=(30.0, 30.0, 0.0),
        )

        surface._on_machine_state_changed(machine, state)

        center_x, center_y = _get_laser_dot_center(surface)

        assert abs(center_x - 30.0) < 1e-5
        assert abs(center_y - 30.0) < 1e-5

    @pytest.mark.ui
    def test_laser_dot_position_matches_without_wco(
        self, ui_context_initializer
    ):
        """
        When WCO is zero, MPos == WPos, so the laser dot should be
        at the same position.
        """
        machine = Machine(ui_context_initializer)
        machine.set_axis_extents(WIDTH, HEIGHT)
        machine.set_origin(Origin.BOTTOM_LEFT)

        mock_editor = MagicMock()
        mock_editor.doc = MagicMock()
        mock_window = MagicMock()
        surface = WorkSurface(mock_editor, mock_window, machine)

        state = DeviceState(
            machine_pos=(OFFSET, OFFSET, 0.0),
            work_pos=(OFFSET, OFFSET, 0.0),
            wco=(0.0, 0.0, 0.0),
        )

        surface._on_machine_state_changed(machine, state)

        center_x, center_y = _get_laser_dot_center(surface)

        assert abs(center_x - OFFSET) < 1e-5
        assert abs(center_y - OFFSET) < 1e-5

    @pytest.mark.ui
    def test_laser_dot_at_nonzero_mpos_with_wco(self, ui_context_initializer):
        """
        G55 with WCO(50,30,0), machine at MPos(60,40,0).
        The laser dot should appear at canvas (60, 40) — the
        machine position — while the grid shifts by the WCS offset.
        """
        machine = Machine(ui_context_initializer)
        machine.set_axis_extents(WIDTH, HEIGHT)
        machine.set_origin(Origin.BOTTOM_LEFT)
        machine.wcs_origin_is_workarea_origin = False
        machine.active_wcs = "G55"
        machine.wcs_offsets["G55"] = (50.0, 30.0, 0.0)

        mock_editor = MagicMock()
        mock_editor.doc = MagicMock()
        mock_window = MagicMock()
        surface = WorkSurface(mock_editor, mock_window, machine)

        state = DeviceState(
            machine_pos=(60.0, 40.0, 0.0),
            work_pos=(10.0, 10.0, 0.0),
            wco=(50.0, 30.0, 0.0),
        )

        surface._on_machine_state_changed(machine, state)

        center_x, center_y = _get_laser_dot_center(surface)

        assert abs(center_x - 60.0) < 1e-5
        assert abs(center_y - 40.0) < 1e-5
