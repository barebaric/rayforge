from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional
from gi.repository import GLib, Adw
from .. import config
from ..pipeline.job import generate_job_ops
from ..shared.tasker import task_mgr
from ..shared.tasker.context import ExecutionContext

if TYPE_CHECKING:
    from gi.repository import Gio
    from .canvas3d import Canvas3D
    from ..mainwindow import MainWindow
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class ViewModeCmd:
    """Handles commands for switching and controlling views (2D/3D)."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def toggle_3d_view(
        self,
        win: "MainWindow",
        action: "Gio.SimpleAction",
        value: Optional["GLib.Variant"],
    ):
        """
        Handles the logic for switching between the 2D and 3D views, including
        pre-flight checks and asynchronous loading of the 3D preview.
        """
        from .canvas3d import initialized as canvas3d_initialized

        current_state = action.get_state()
        is_3d = current_state.get_boolean() if current_state else False
        request_3d = value.get_boolean() if value else not is_3d

        if is_3d == request_3d:
            return

        if request_3d:
            if not canvas3d_initialized:
                logger.warning(
                    "Attempted to open 3D view, but it is not available."
                )
                toast = Adw.Toast.new(
                    _(
                        "3D view is not available due to missing dependencies."
                    )
                )
                win.toast_overlay.add_toast(toast)
                return

            machine = config.config.machine
            if not machine:
                logger.warning(
                    "Cannot show 3D view without an active machine."
                )
                toast = Adw.Toast.new(
                    _("Select a machine to open the 3D view.")
                )
                win.toast_overlay.add_toast(toast)
                return

            action.set_state(GLib.Variant.new_boolean(True))
            win.view_stack.set_visible_child_name("3d")

            if win.canvas3d and win.canvas3d.ops_renderer:
                win.canvas3d.ops_renderer.clear()
                win.canvas3d.queue_render()

            async def load_ops_coro(context: ExecutionContext):
                current_machine = config.config.machine
                if not current_machine:
                    return

                try:
                    logger.debug("Creating 3D preview")
                    context.set_message("Generating path preview...")
                    ops = await generate_job_ops(
                        self._editor.doc,
                        current_machine,
                        self._editor.ops_generator,
                        context,
                    )
                    if win.canvas3d:
                        win.canvas3d.set_ops(ops)
                    logger.debug("Preview ready")
                    context.set_message("Path preview loaded.")
                    context.set_progress(1.0)
                except Exception:
                    logger.error(
                        "Failed to generate ops for 3D view", exc_info=True
                    )
                    toast = Adw.Toast.new(
                        _("Failed to generate path preview.")
                    )
                    win.toast_overlay.add_toast(toast)
                    win.view_stack.set_visible_child_name("2d")
                    action.set_state(GLib.Variant.new_boolean(False))
                    raise

            task_mgr.add_coroutine(load_ops_coro, key="load-3d-preview")
        else:
            action.set_state(GLib.Variant.new_boolean(False))
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
