from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from gi.repository import GLib

from ..config import config
from .elements.simulation_overlay import SimulationOverlay
from .simulation_controls import PreviewControls

if TYPE_CHECKING:
    from ..mainwindow import MainWindow
    from gi.repository import Gio

logger = logging.getLogger(__name__)


class SimulatorCmd:
    """Handles commands for controlling the execution simulation view."""

    def __init__(self, win: "MainWindow"):
        self._win = win
        self.simulation_overlay: Optional[SimulationOverlay] = None
        self.preview_controls: Optional[PreviewControls] = None
        self._preview_controls_handler_id: Optional[int] = None

    def toggle_mode(self, action: "Gio.SimpleAction", value: "GLib.Variant"):
        """Toggles the execution preview simulation overlay."""
        enabled = value.get_boolean()
        if enabled:
            self._enter_mode()
        else:
            self._exit_mode()
        action.set_state(value)

    def _enter_mode(self):
        """Enters preview mode by creating overlay and enabling it on the
        canvas."""
        win = self._win
        # If the 3D view is active, switch back to the 2D view because
        # simulation is only supported on the 2D canvas.
        if win.view_stack.get_visible_child_name() == "3d":
            win.view_stack.set_visible_child_name("2d")
            action = win.action_manager.get_action("show_3d_view")
            state = action.get_state()
            if state and state.get_boolean():
                action.set_state(GLib.Variant.new_boolean(False))

        # Get work area size
        if config.machine:
            work_area_size = config.machine.dimensions
        else:
            work_area_size = (100.0, 100.0)

        # Create simulation overlay
        self.simulation_overlay = SimulationOverlay(work_area_size)

        # Aggregate operations from all layers
        full_ops = win._aggregate_ops_for_3d_view()
        self.simulation_overlay.set_ops(full_ops)
        win._update_gcode_preview(full_ops)

        # Enable simulation mode on the canvas
        win.surface.set_simulation_mode(True, self.simulation_overlay)

        # Create and show preview controls
        self.preview_controls = PreviewControls(self.simulation_overlay)
        win.surface_overlay.add_overlay(self.preview_controls)
        self._preview_controls_handler_id = self.preview_controls.connect(
            "step-changed", self._on_simulation_step_changed
        )
        win.left_content_pane.set_position(win._last_gcode_previewer_width)

        # Auto-start playback
        self.preview_controls._start_playback()

    def _exit_mode(self):
        """Exits simulation mode by disabling it on the canvas."""
        win = self._win
        win.surface.set_simulation_mode(False)

        # Remove preview controls
        if self.preview_controls:
            if self._preview_controls_handler_id:
                self.preview_controls.disconnect(
                    self._preview_controls_handler_id
                )
                self._preview_controls_handler_id = None
            win.surface_overlay.remove_overlay(self.preview_controls)
            self.preview_controls = None

        self.simulation_overlay = None
        win.left_content_pane.set_position(0)
        win.gcode_previewer.clear_highlight()

    def _on_simulation_step_changed(self, sender, step):
        self._win.gcode_previewer.highlight_op(step)
