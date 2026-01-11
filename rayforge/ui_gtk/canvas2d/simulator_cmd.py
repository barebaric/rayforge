from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional

from gi.repository import GLib

from ...context import get_context
from ...pipeline.artifact import JobArtifact
from .elements.simulation_overlay import SimulationOverlay
from .simulation_controls import PreviewControls

if TYPE_CHECKING:
    from ...ui_gtk.mainwindow import MainWindow
    from gi.repository import Gio

logger = logging.getLogger(__name__)


class SimulatorCmd:
    """Handles commands for controlling the execution simulation view."""

    def __init__(self, win: "MainWindow"):
        self._win = win
        self.simulation_overlay: Optional[SimulationOverlay] = None
        self.preview_controls: Optional[PreviewControls] = None
        self._preview_controls_handler = None
        self._is_syncing = False  # Flag to prevent signal feedback loops

    def toggle_mode(self, action: "Gio.SimpleAction", value: "GLib.Variant"):
        """Toggles the execution preview simulation overlay."""
        enabled = value.get_boolean()
        if enabled:
            self._enter_mode()
        else:
            self._exit_mode()
        action.set_state(value)

    def sync_from_gcode(self, line_number: int):
        """
        Updates the simulation slider from an external event (e.g., a G-code
        editor click) without causing a feedback loop.
        """
        if not self.preview_controls:
            return
        self._is_syncing = True
        self.preview_controls.set_playback_position(line_number)
        self._is_syncing = False

    def reload_simulation(self, new_artifact: Optional[JobArtifact]):
        """
        Reloads the simulation with a new, final job artifact.
        """
        if not self.simulation_overlay or not self.preview_controls:
            return

        logger.debug("Reloading simulation with new artifact.")

        was_playing = self.preview_controls.playing

        if new_artifact:
            assert isinstance(new_artifact, JobArtifact)

            # The overlay gets the Ops for rendering geometry
            self.simulation_overlay.set_ops(new_artifact.ops)
            # The controls get the G-code and map to drive the timeline
            self.preview_controls.set_playback_source(
                new_artifact.machine_code_bytes, new_artifact.op_map_bytes
            )
        else:
            self.simulation_overlay.set_ops(None)
            self.preview_controls.set_playback_source(None, None)

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
        machine = get_context().machine
        if machine:
            work_area_size = machine.dimensions
        else:
            work_area_size = (100.0, 100.0)

        # Create an empty simulation overlay. It will be populated shortly.
        self.simulation_overlay = SimulationOverlay(work_area_size)
        win.surface.set_simulation_mode(True, self.simulation_overlay)

        # Create and show preview controls
        self.preview_controls = PreviewControls(self.simulation_overlay)
        win.surface_overlay.add_overlay(self.preview_controls)
        self.preview_controls.step_changed.connect(
            self._on_simulation_step_changed
        )
        self.preview_controls.close_requested.connect(self._on_close_requested)
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

        # Update the simulate_mode action state to False
        sim_action = win.action_manager.get_action("simulate_mode")
        if sim_action:
            sim_action.set_state(GLib.Variant.new_boolean(False))

        # Remove preview controls
        if self.preview_controls:
            self.preview_controls.step_changed.disconnect(
                self._on_simulation_step_changed
            )
            self.preview_controls.close_requested.disconnect(
                self._on_close_requested
            )
            win.surface_overlay.remove_overlay(self.preview_controls)
            self.preview_controls = None

        self.simulation_overlay = None
        win.gcode_previewer.clear_highlight()

    def _on_simulation_step_changed(self, sender, line_number):
        # If the update was triggered by a G-code click, do nothing to
        # prevent a feedback loop.
        if self._is_syncing:
            return
        self._win.gcode_previewer.highlight_line(line_number, use_align=True)

    def _on_close_requested(self, sender):
        """Handles close button signal from preview controls."""
        self._exit_mode()
