import logging
from typing import TYPE_CHECKING, Optional
from gi.repository import GLib, Adw
from ..context import get_context

if TYPE_CHECKING:
    from gi.repository import Gio
    from .canvas3d import Canvas3D
    from ..ui_gtk.mainwindow import MainWindow
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class ViewModeCmd:
    """Handles commands for switching and controlling views (2D/3D)."""

    def __init__(self, editor: "DocEditor", win: "MainWindow"):
        self._editor = editor
        self._win = win

    def toggle_3d_view(
        self,
        action: "Gio.SimpleAction",
        value: Optional["GLib.Variant"],
    ):
        """
        Handles the logic for switching between the 2D and 3D views.
        """
        from .canvas3d import initialized as canvas3d_initialized

        win = self._win
        current_state = action.get_state()
        is_3d = current_state.get_boolean() if current_state else False
        request_3d = value.get_boolean() if value else not is_3d

        if is_3d == request_3d:
            return

        gcode_action = win.action_manager.get_action("toggle_gcode_preview")

        if request_3d:
            if not canvas3d_initialized:
                logger.warning(
                    "Attempted to open 3D view, but it is not available."
                )
                toast = Adw.Toast.new(
                    _("3D view is not available due to missing dependencies.")
                )
                win.toast_overlay.add_toast(toast)
                return

            if not get_context().machine:
                logger.warning(
                    "Cannot show 3D view without an active machine."
                )
                toast = Adw.Toast.new(
                    _("Select a machine to open the 3D view.")
                )
                win.toast_overlay.add_toast(toast)
                return

            # If simulation mode is active, turn it off before switching
            # to the 3D view.
            sim_action = win.action_manager.get_action("simulate_mode")
            sim_state = sim_action.get_state() if sim_action else None
            if sim_state and sim_state.get_boolean():
                sim_action.change_state(GLib.Variant.new_boolean(False))

            action.set_state(GLib.Variant.new_boolean(True))
            if gcode_action:
                gcode_action.change_state(GLib.Variant.new_boolean(True))
            win.view_stack.set_visible_child_name("3d")

        else:
            action.set_state(GLib.Variant.new_boolean(False))
            if gcode_action:
                gcode_action.change_state(GLib.Variant.new_boolean(False))
            win.view_stack.set_visible_child_name("2d")
            win.surface.grab_focus()

    def set_view_top(self, canvas3d: Optional["Canvas3D"]):
        """Sets the 3D view to a top-down orientation."""
        if canvas3d:
            canvas3d.reset_view_top()

    def set_view_front(self, canvas3d: Optional["Canvas3D"]):
        """Sets the 3D view to a front-facing orientation."""
        if canvas3d:
            canvas3d.reset_view_front()

    def set_view_iso(self, canvas3d: Optional["Canvas3D"]):
        """Sets the 3D view to an isometric orientation."""
        if canvas3d:
            canvas3d.reset_view_iso()

    def toggle_perspective(
        self,
        canvas3d: Optional["Canvas3D"],
        action: "Gio.SimpleAction",
        value: "GLib.Variant",
    ):
        """Toggles the 3D camera between perspective and orthographic."""
        if canvas3d and canvas3d.camera:
            is_perspective = value.get_boolean()
            canvas3d.camera.is_perspective = is_perspective
            canvas3d.queue_render()
            action.set_state(value)
