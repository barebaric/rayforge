"""
Tests for the Move-tab jog shortcuts honouring the PANEL presentation.

The selection bounds arrive in WORLD coordinates, but the "Move to
Lower-Left / Center / Upper-Right" buttons refer to the presented
(PANEL) corners, so the target must be projected through the panel
rotation before being converted to machine coordinates.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rayforge.machine.models.machine_panel import PanelOrientation


@pytest.fixture
def bottom_panel(lite_context, sync_machine):
    """A BottomPanel instance with Gtk initialization bypassed, wired to
    a real 400x300 machine."""
    from rayforge.ui_gtk.doceditor.bottom_panel import BottomPanel

    sync_machine.set_axis_extents(400, 300)
    lite_context.config.set_machine(sync_machine)

    bottom: Any = BottomPanel.__new__(BottomPanel)
    bottom.machine = sync_machine
    bottom.machine_cmd = MagicMock()
    # Selection bounds in WORLD coordinates.
    bottom._get_bounds_callback = MagicMock(
        return_value=(100.0, 50.0, 120.0, 60.0)
    )
    return bottom


@pytest.mark.parametrize(
    "orientation, position, expected",
    [
        (PanelOrientation.NATIVE, "ll", (100.0, 50.0)),
        (PanelOrientation.NATIVE, "center", (110.0, 55.0)),
        (PanelOrientation.NATIVE, "ur", (120.0, 60.0)),
        (PanelOrientation.ROTATED_RIGHT, "ll", (120.0, 50.0)),
        (PanelOrientation.ROTATED_RIGHT, "center", (110.0, 55.0)),
        (PanelOrientation.ROTATED_RIGHT, "ur", (100.0, 60.0)),
        (PanelOrientation.ROTATED_LEFT, "ll", (100.0, 60.0)),
        (PanelOrientation.ROTATED_LEFT, "center", (110.0, 55.0)),
        (PanelOrientation.ROTATED_LEFT, "ur", (120.0, 50.0)),
    ],
)
@pytest.mark.ui
def test_move_to_position_honors_panel(
    bottom_panel, orientation, position, expected
):
    """The jog shortcuts target the presented corners of the selection."""
    bottom_panel.machine.panel.set_orientation(orientation)

    bottom_panel._on_move_to_position(None, position)

    bottom_panel.machine_cmd.move_to.assert_called_once_with(
        bottom_panel.machine, *expected
    )


@pytest.mark.ui
def test_move_to_position_no_bounds_returns(bottom_panel):
    bottom_panel._get_bounds_callback = MagicMock(return_value=None)

    bottom_panel._on_move_to_position(None, "ll")

    bottom_panel.machine_cmd.move_to.assert_not_called()
