import pytest
from unittest.mock import MagicMock, AsyncMock
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.driver.driver import Axis
from rayforge.ui_gtk.machine.jog_widget import JogWidget

# Jog distance and speed for testing
JOG_DISTANCE = 10.0
JOG_SPEED = 1000

# (button_name, expected_axis, rev_z, origin)
JOG_BUTTON_SCENARIOS = [
    # --- X Axis Buttons ---
    # Right (East) button - depends on origin
    ("x_plus", Axis.X, False, Origin.BOTTOM_LEFT),
    ("x_plus", Axis.X, True, Origin.BOTTOM_LEFT),
    ("x_plus", Axis.X, False, Origin.TOP_LEFT),
    ("x_plus", Axis.X, True, Origin.TOP_LEFT),
    ("x_plus", Axis.X, False, Origin.TOP_RIGHT),
    ("x_plus", Axis.X, True, Origin.TOP_RIGHT),
    ("x_plus", Axis.X, False, Origin.BOTTOM_RIGHT),
    ("x_plus", Axis.X, True, Origin.BOTTOM_RIGHT),
    # Left (West) button - depends on origin
    ("x_minus", Axis.X, False, Origin.BOTTOM_LEFT),
    ("x_minus", Axis.X, True, Origin.BOTTOM_LEFT),
    ("x_minus", Axis.X, False, Origin.TOP_LEFT),
    ("x_minus", Axis.X, True, Origin.TOP_LEFT),
    ("x_minus", Axis.X, False, Origin.TOP_RIGHT),
    ("x_minus", Axis.X, True, Origin.TOP_RIGHT),
    ("x_minus", Axis.X, False, Origin.BOTTOM_RIGHT),
    ("x_minus", Axis.X, True, Origin.BOTTOM_RIGHT),
    # --- Y Axis Buttons ---
    # Away (North) button - depends on origin
    ("y_plus", Axis.Y, False, Origin.BOTTOM_LEFT),
    ("y_plus", Axis.Y, True, Origin.BOTTOM_LEFT),
    ("y_plus", Axis.Y, False, Origin.TOP_LEFT),
    ("y_plus", Axis.Y, True, Origin.TOP_LEFT),
    ("y_plus", Axis.Y, False, Origin.TOP_RIGHT),
    ("y_plus", Axis.Y, True, Origin.TOP_RIGHT),
    ("y_plus", Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("y_plus", Axis.Y, True, Origin.BOTTOM_RIGHT),
    # Toward (South) button - depends on origin
    ("y_minus", Axis.Y, False, Origin.BOTTOM_LEFT),
    ("y_minus", Axis.Y, True, Origin.BOTTOM_LEFT),
    ("y_minus", Axis.Y, False, Origin.TOP_LEFT),
    ("y_minus", Axis.Y, True, Origin.TOP_LEFT),
    ("y_minus", Axis.Y, False, Origin.TOP_RIGHT),
    ("y_minus", Axis.Y, True, Origin.TOP_RIGHT),
    ("y_minus", Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("y_minus", Axis.Y, True, Origin.BOTTOM_RIGHT),
    # --- Z Axis Buttons ---
    # Up button - depends on reverse_z_axis
    ("z_plus", Axis.Z, False, Origin.BOTTOM_LEFT),
    ("z_plus", Axis.Z, True, Origin.BOTTOM_LEFT),
    ("z_plus", Axis.Z, False, Origin.TOP_LEFT),
    ("z_plus", Axis.Z, True, Origin.TOP_LEFT),
    ("z_plus", Axis.Z, False, Origin.TOP_RIGHT),
    ("z_plus", Axis.Z, True, Origin.TOP_RIGHT),
    ("z_plus", Axis.Z, False, Origin.BOTTOM_RIGHT),
    ("z_plus", Axis.Z, True, Origin.BOTTOM_RIGHT),
    # Down button - depends on reverse_z_axis
    ("z_minus", Axis.Z, False, Origin.BOTTOM_LEFT),
    ("z_minus", Axis.Z, True, Origin.BOTTOM_LEFT),
    ("z_minus", Axis.Z, False, Origin.TOP_LEFT),
    ("z_minus", Axis.Z, True, Origin.TOP_LEFT),
    ("z_minus", Axis.Z, False, Origin.TOP_RIGHT),
    ("z_minus", Axis.Z, True, Origin.TOP_RIGHT),
    ("z_minus", Axis.Z, False, Origin.BOTTOM_RIGHT),
    ("z_minus", Axis.Z, True, Origin.BOTTOM_RIGHT),
    # --- Diagonal Buttons ---
    # North-East - depends on origin
    ("x_plus_y_plus", Axis.X | Axis.Y, False, Origin.BOTTOM_LEFT),
    ("x_plus_y_plus", Axis.X | Axis.Y, True, Origin.BOTTOM_LEFT),
    ("x_plus_y_plus", Axis.X | Axis.Y, False, Origin.TOP_LEFT),
    ("x_plus_y_plus", Axis.X | Axis.Y, True, Origin.TOP_LEFT),
    ("x_plus_y_plus", Axis.X | Axis.Y, False, Origin.TOP_RIGHT),
    ("x_plus_y_plus", Axis.X | Axis.Y, True, Origin.TOP_RIGHT),
    ("x_plus_y_plus", Axis.X | Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("x_plus_y_plus", Axis.X | Axis.Y, True, Origin.BOTTOM_RIGHT),
    # North-West - depends on origin
    ("x_minus_y_plus", Axis.X | Axis.Y, False, Origin.BOTTOM_LEFT),
    ("x_minus_y_plus", Axis.X | Axis.Y, True, Origin.BOTTOM_LEFT),
    ("x_minus_y_plus", Axis.X | Axis.Y, False, Origin.TOP_LEFT),
    ("x_minus_y_plus", Axis.X | Axis.Y, True, Origin.TOP_LEFT),
    ("x_minus_y_plus", Axis.X | Axis.Y, False, Origin.TOP_RIGHT),
    ("x_minus_y_plus", Axis.X | Axis.Y, True, Origin.TOP_RIGHT),
    ("x_minus_y_plus", Axis.X | Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("x_minus_y_plus", Axis.X | Axis.Y, True, Origin.BOTTOM_RIGHT),
    # South-East - depends on origin
    ("x_plus_y_minus", Axis.X | Axis.Y, False, Origin.BOTTOM_LEFT),
    ("x_plus_y_minus", Axis.X | Axis.Y, True, Origin.BOTTOM_LEFT),
    ("x_plus_y_minus", Axis.X | Axis.Y, False, Origin.TOP_LEFT),
    ("x_plus_y_minus", Axis.X | Axis.Y, True, Origin.TOP_LEFT),
    ("x_plus_y_minus", Axis.X | Axis.Y, False, Origin.TOP_RIGHT),
    ("x_plus_y_minus", Axis.X | Axis.Y, True, Origin.TOP_RIGHT),
    ("x_plus_y_minus", Axis.X | Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("x_plus_y_minus", Axis.X | Axis.Y, True, Origin.BOTTOM_RIGHT),
    # South-West - depends on origin
    ("x_minus_y_minus", Axis.X | Axis.Y, False, Origin.BOTTOM_LEFT),
    ("x_minus_y_minus", Axis.X | Axis.Y, True, Origin.BOTTOM_LEFT),
    ("x_minus_y_minus", Axis.X | Axis.Y, False, Origin.TOP_LEFT),
    ("x_minus_y_minus", Axis.X | Axis.Y, True, Origin.TOP_LEFT),
    ("x_minus_y_minus", Axis.X | Axis.Y, False, Origin.TOP_RIGHT),
    ("x_minus_y_minus", Axis.X | Axis.Y, True, Origin.TOP_RIGHT),
    ("x_minus_y_minus", Axis.X | Axis.Y, False, Origin.BOTTOM_RIGHT),
    ("x_minus_y_minus", Axis.X | Axis.Y, True, Origin.BOTTOM_RIGHT),
]


@pytest.mark.ui
@pytest.mark.parametrize(
    "button_name, expected_axis, rev_z, origin",
    JOG_BUTTON_SCENARIOS,
)
def test_jog_button_direction(
    ui_context_initializer,
    button_name,
    expected_axis,
    rev_z,
    origin,
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
    # 1. Configure the Machine
    machine = Machine(ui_context_initializer)
    machine.set_dimensions(200, 200)
    machine.set_origin(origin)
    machine.set_reverse_x_axis(False)
    machine.set_reverse_y_axis(False)
    machine.set_reverse_z_axis(rev_z)

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

    # 5. Calculate expected distances based on origin and button direction
    # X axis: positive when origin is on left, negative when on right
    if origin in (Origin.TOP_RIGHT, Origin.BOTTOM_RIGHT):
        x_sign = -1.0
    else:
        x_sign = 1.0

    # Y axis: positive when origin is on bottom, negative when on top
    if origin in (Origin.TOP_LEFT, Origin.TOP_RIGHT):
        y_sign = -1.0
    else:
        y_sign = 1.0

    # Z axis: depends on reverse_z_axis
    z_sign = -1.0 if rev_z else 1.0

    if expected_axis & (Axis.X | Axis.Y) and not (
        expected_axis & Axis.X and expected_axis & Axis.Y
    ):
        # Single axis jog (X or Y only)
        if button_name == "x_plus":
            expected_dist = JOG_DISTANCE * x_sign
        elif button_name == "x_minus":
            expected_dist = -JOG_DISTANCE * x_sign
        elif button_name == "y_plus":
            expected_dist = JOG_DISTANCE * y_sign
        elif button_name == "y_minus":
            expected_dist = -JOG_DISTANCE * y_sign
        else:
            pytest.fail(f"Unknown button name: {button_name}")

        mock_jog.assert_called_once_with(
            machine, expected_axis, expected_dist, JOG_SPEED
        )
    elif expected_axis == Axis.Z:
        # Z axis jog
        if button_name == "z_plus":
            expected_dist = JOG_DISTANCE * z_sign
        elif button_name == "z_minus":
            expected_dist = -JOG_DISTANCE * z_sign
        else:
            pytest.fail(f"Unknown button name: {button_name}")

        mock_jog.assert_called_once_with(
            machine, expected_axis, expected_dist, JOG_SPEED
        )
    else:
        # Diagonal jog - called twice, once for X and once for Y
        assert mock_jog.call_count == 2
        calls = mock_jog.call_args_list

        # Determine expected X and Y distances based on button name
        if button_name == "x_plus_y_plus":
            expected_x = JOG_DISTANCE * x_sign
            expected_y = JOG_DISTANCE * y_sign
        elif button_name == "x_minus_y_plus":
            expected_x = -JOG_DISTANCE * x_sign
            expected_y = JOG_DISTANCE * y_sign
        elif button_name == "x_plus_y_minus":
            expected_x = JOG_DISTANCE * x_sign
            expected_y = -JOG_DISTANCE * y_sign
        elif button_name == "x_minus_y_minus":
            expected_x = -JOG_DISTANCE * x_sign
            expected_y = -JOG_DISTANCE * y_sign
        else:
            pytest.fail(f"Unknown button name: {button_name}")

        # Verify both calls were made (order may vary)
        call_params = [(call[0][1], call[0][2]) for call in calls]
        assert (Axis.X, expected_x) in call_params
        assert (Axis.Y, expected_y) in call_params
