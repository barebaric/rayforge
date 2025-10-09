from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from gi.repository import GLib

from ..config import config
from ..core.ops import Ops
from .elements.simulation_overlay import SimulationOverlay
from .simulation_controls import PreviewControls
from ..pipeline.artifact.base import Artifact

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

    def reload_simulation(self, new_artifact: Optional[Artifact]):
        """
        Reloads the simulation with the Ops from a new, final job artifact.
        """
        if not self.simulation_overlay or not self.preview_controls:
            return

        logger.debug("Reloading simulation with new operations.")

        was_playing = self.preview_controls.playing

        ops_to_load = new_artifact.ops if new_artifact else Ops()

        self.simulation_overlay.set_ops(ops_to_load)

        self.preview_controls.reset()
        if was_playing:
            self.preview_controls._start_playback()

    def _enter_mode(self):
        """
        Enters preview mode synchronously. It then requests an immediate
        data refresh from the main window.
        """
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

        # Create an empty simulation overlay. It will be populated shortly.
        self.simulation_overlay = SimulationOverlay(work_area_size)
        win.surface.set_simulation_mode(True, self.simulation_overlay)

        # Create and show preview controls
        self.preview_controls = PreviewControls(self.simulation_overlay)
        win.surface_overlay.add_overlay(self.preview_controls)
        self._preview_controls_handler_id = self.preview_controls.connect(
            "step-changed", self._on_simulation_step_changed
        )
        # Ensure G-code preview is also visible
        gcode_action = win.action_manager.get_action("toggle_gcode_preview")
        state = gcode_action.get_state()
        if not (state and state.get_boolean()):
            gcode_action.change_state(GLib.Variant.new_boolean(True))

        # Call the public method on MainWindow to trigger the async data load.
        win.refresh_previews()

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
        win.gcode_previewer.clear_highlight()

    def _on_simulation_step_changed(self, sender, step):
        self._win.gcode_previewer.highlight_op(step)
