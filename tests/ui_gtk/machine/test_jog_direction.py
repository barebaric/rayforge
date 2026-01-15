import pytest
from unittest.mock import MagicMock, AsyncMock
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.driver.driver import Axis, DeviceState
from rayforge.machine.transport import TransportStatus  # <-- Import added

# Jog distance and speed for testing
JOG_DISTANCE = 10.0
JOG_SPEED = 1000

# (button, axis, origin, reversed, expectation)
JOG_BUTTON_SCENARIOS = [
    # Right (East) button
    ("east", Axis.X, Origin.BOTTOM_LEFT, False, JOG_DISTANCE),
    ("east", Axis.X, Origin.BOTTOM_LEFT, True, -JOG_DISTANCE),
    ("east", Axis.X, Origin.TOP_LEFT, False, JOG_DISTANCE),
    ("east", Axis.X, Origin.TOP_LEFT, True, -JOG_DISTANCE),
    ("east", Axis.X, Origin.TOP_RIGHT, False, -JOG_DISTANCE),
    ("east", Axis.X, Origin.TOP_RIGHT, True, JOG_DISTANCE),
    ("east", Axis.X, Origin.BOTTOM_RIGHT, False, -JOG_DISTANCE),
    ("east", Axis.X, Origin.BOTTOM_RIGHT, True, JOG_DISTANCE),
    # Left (West) button
    ("west", Axis.X, Origin.BOTTOM_LEFT, False, -JOG_DISTANCE),
    ("west", Axis.X, Origin.BOTTOM_LEFT, True, JOG_DISTANCE),
    ("west", Axis.X, Origin.TOP_LEFT, False, -JOG_DISTANCE),
    ("west", Axis.X, Origin.TOP_LEFT, True, JOG_DISTANCE),
    ("west", Axis.X, Origin.TOP_RIGHT, False, JOG_DISTANCE),
    ("west", Axis.X, Origin.TOP_RIGHT, True, -JOG_DISTANCE),
    ("west", Axis.X, Origin.BOTTOM_RIGHT, False, JOG_DISTANCE),
    ("west", Axis.X, Origin.BOTTOM_RIGHT, True, -JOG_DISTANCE),
    # Away (North) button
    ("north", Axis.Y, Origin.BOTTOM_LEFT, False, JOG_DISTANCE),
    ("north", Axis.Y, Origin.BOTTOM_LEFT, True, -JOG_DISTANCE),
    ("north", Axis.Y, Origin.TOP_LEFT, False, -JOG_DISTANCE),
    ("north", Axis.Y, Origin.TOP_LEFT, True, JOG_DISTANCE),
    ("north", Axis.Y, Origin.TOP_RIGHT, False, -JOG_DISTANCE),
    ("north", Axis.Y, Origin.TOP_RIGHT, True, JOG_DISTANCE),
    ("north", Axis.Y, Origin.BOTTOM_RIGHT, False, JOG_DISTANCE),
    ("north", Axis.Y, Origin.BOTTOM_RIGHT, True, -JOG_DISTANCE),
    # Toward (South) button
    ("south", Axis.Y, Origin.BOTTOM_LEFT, False, -JOG_DISTANCE),
    ("south", Axis.Y, Origin.BOTTOM_LEFT, True, JOG_DISTANCE),
    ("south", Axis.Y, Origin.TOP_LEFT, False, JOG_DISTANCE),
    ("south", Axis.Y, Origin.TOP_LEFT, True, -JOG_DISTANCE),
    ("south", Axis.Y, Origin.TOP_RIGHT, False, JOG_DISTANCE),
    ("south", Axis.Y, Origin.TOP_RIGHT, True, -JOG_DISTANCE),
    ("south", Axis.Y, Origin.BOTTOM_RIGHT, False, -JOG_DISTANCE),
    ("south", Axis.Y, Origin.BOTTOM_RIGHT, True, JOG_DISTANCE),
    # --- Z Axis Buttons ---
    # Up button - depends on reverse_z_axis
    ("z_plus", Axis.Z, Origin.BOTTOM_LEFT, False, JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.BOTTOM_LEFT, True, -JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.TOP_LEFT, False, JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.TOP_LEFT, True, -JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.TOP_RIGHT, False, JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.TOP_RIGHT, True, -JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.BOTTOM_RIGHT, False, JOG_DISTANCE),
    ("z_plus", Axis.Z, Origin.BOTTOM_RIGHT, True, -JOG_DISTANCE),
    # Down button - depends on reverse_z_axis
    ("z_minus", Axis.Z, Origin.BOTTOM_LEFT, False, -JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.BOTTOM_LEFT, True, JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.TOP_LEFT, False, -JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.TOP_LEFT, True, JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.TOP_RIGHT, False, -JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.TOP_RIGHT, True, JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.BOTTOM_RIGHT, False, -JOG_DISTANCE),
    ("z_minus", Axis.Z, Origin.BOTTOM_RIGHT, True, JOG_DISTANCE),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "button_name, expected_axis, origin, reversed, expectation",
    JOG_BUTTON_SCENARIOS,
)
def test_jog_button_direction(
    ui_context_initializer,
    button_name,
    expected_axis,
    origin,
    reversed,
    expectation,
):
    """
    Verifies that each jog button sends the correct signed distance to the
    machine for all combinations of origin and axis direction settings.

    The correct behavior is:
    - X axis: Right button sends positive X when origin is on the left,
      negative X when origin is on the right.
    - Y axis: North button sends positive Y when origin is on the bottom,
      negative Y when origin is on the top.
    - Z axis: Depends on reverse_z_axis setting.
    """
    # Import JogWidget here to avoid GTK import during test collection
    from rayforge.ui_gtk.machine.jog_widget import JogWidget

    # 1. Configure the Machine
    machine = Machine(ui_context_initializer)
    machine.set_dimensions(200, 200)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(expected_axis == Axis.X and reversed)
    machine.set_reverse_y_axis(expected_axis == Axis.Y and reversed)
    machine.set_reverse_z_axis(expected_axis == Axis.Z and reversed)

    # 2. Mock MachineCmd
    mock_machine_cmd = MagicMock()
    mock_jog = AsyncMock()
    mock_machine_cmd.jog = mock_jog

    # 3. Create JogWidget
    jog_widget = JogWidget()
    jog_widget.set_machine(machine, mock_machine_cmd)
    jog_widget.jog_distance = JOG_DISTANCE
    jog_widget.jog_speed = JOG_SPEED

    # 4. Get the button by name and click it
    button = getattr(jog_widget, f"{button_name}_btn")
    button.emit("clicked")

    # 5. Verify the jog command was called with expected distance
    mock_jog.assert_called_once_with(
        machine, {expected_axis: expectation}, JOG_SPEED
    )


LIMIT_SCENARIOS = [
    # Standard (Bottom-Left, no reverse), near top-right corner
    (
        Origin.BOTTOM_LEFT,
        False,
        False,
        (95.0, 95.0),
        {
            "east_btn",
            "north_btn",
            "north_east_btn",
            "north_west_btn",
            "south_east_btn",
        },
    ),
    # Standard (Bottom-Left, no reverse), near bottom-left corner
    (
        Origin.BOTTOM_LEFT,
        False,
        False,
        (5.0, 5.0),
        {
            "west_btn",
            "south_btn",
            "north_west_btn",
            "south_west_btn",
            "south_east_btn",
        },
    ),
    # Top-Left (Y-down), near bottom-left corner (visually)
    (
        Origin.TOP_LEFT,
        False,
        False,
        (5.0, 5.0),
        {
            "west_btn",
            "north_btn",
            "north_west_btn",
            "south_west_btn",
            "north_east_btn",
        },
    ),
    # Bottom-Left, reverse X. Limits are X:[-100, 0], Y:[0, 100].
    # Pos (-5, 5) is near visual bottom-left (X=0, Y=0).
    (
        Origin.BOTTOM_LEFT,
        True,
        False,
        (-5.0, 5.0),
        {
            "west_btn",
            "south_btn",
            "south_east_btn",
            "north_west_btn",
            "south_west_btn",
        },
    ),
    # Top-Right, reverse both. Limits are X:[-100, 0], Y:[-100, 0].
    # Pos (-5, -5) is near visual top-right corner (X=0, Y=0).
    (
        Origin.TOP_RIGHT,
        True,
        True,
        (-5.0, -5.0),
        {
            "east_btn",
            "north_btn",
            "north_east_btn",
            "north_west_btn",
            "south_east_btn",
        },
    ),
    # Center position, no warnings expected
    (Origin.BOTTOM_LEFT, False, False, (50.0, 50.0), set()),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "origin, reverse_x, reverse_y, current_pos, expected_warnings",
    LIMIT_SCENARIOS,
)
def test_jog_button_limit_warning(
    ui_context_initializer,
    origin,
    reverse_x,
    reverse_y,
    current_pos,
    expected_warnings,
):
    """
    Verifies that jog buttons show a 'warning' CSS class if the jog would
    exceed soft limits, considering machine origin and axis reversal.
    """
    from rayforge.ui_gtk.machine.jog_widget import JogWidget

    # 1. Configure the Machine
    machine = Machine(ui_context_initializer)
    machine.set_dimensions(100, 100)  # Limits are (0,0) to (100,100)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(reverse_x)
    machine.set_reverse_y_axis(reverse_y)
    machine.set_soft_limits_enabled(True)
    # THIS IS THE FIX: The widget only checks limits if machine is connected
    machine.connection_status = TransportStatus.CONNECTED

    # 2. Mock current position by setting device state
    mock_state = DeviceState()
    mock_state.machine_pos = (current_pos[0], current_pos[1], 0.0)
    machine.device_state = mock_state

    # 3. Create JogWidget and set its machine
    # This automatically calls _update_limit_status
    jog_widget = JogWidget()
    jog_widget.jog_distance = JOG_DISTANCE
    jog_widget.set_machine(machine, MagicMock())

    # 4. Collect all buttons that have the warning class
    buttons = {
        "east_btn": jog_widget.east_btn,
        "west_btn": jog_widget.west_btn,
        "north_btn": jog_widget.north_btn,
        "south_btn": jog_widget.south_btn,
        "north_east_btn": jog_widget.north_east_btn,
        "north_west_btn": jog_widget.north_west_btn,
        "south_east_btn": jog_widget.south_east_btn,
        "south_west_btn": jog_widget.south_west_btn,
    }
    warned_buttons = {
        name for name, btn in buttons.items() if btn.has_css_class("warning")
    }

    # 5. Assert that the set of warned buttons matches the expectation
    assert warned_buttons == expected_warnings
