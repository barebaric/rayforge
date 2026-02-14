import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from gi.repository import GLib

from ..core.sketcher import Sketch
from ..core.undo import ListItemCommand
from ..core.workpiece import WorkPiece
from .doceditor import file_dialogs
from .sketcher.cmd import UpdateSketchCommand

if TYPE_CHECKING:
    from .mainwindow import MainWindow
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class SketchModeCmd:
    """Handles commands for entering, exiting, and managing sketch mode."""

    def __init__(self, win: "MainWindow", editor: "DocEditor"):
        self._win = win
        self._editor = editor
        self.active_sketch_workpiece: Optional[WorkPiece] = None
        self._is_editing_new_sketch = False

    def enter_sketch_mode(
        self, workpiece: WorkPiece, is_new_sketch: bool = False
    ):
        """Switches the view to the SketchStudio to edit a workpiece."""
        sketch = None
        if workpiece.sketch_uid:
            sketch = cast(
                Optional[Sketch],
                self._editor.doc.get_asset_by_uid(workpiece.sketch_uid),
            )

        if not sketch:
            logger.warning("Attempted to edit a non-sketch workpiece.")
            return

        try:
            self.active_sketch_workpiece = workpiece
            self._is_editing_new_sketch = is_new_sketch
            self._win.sketch_studio.set_sketch(sketch)
            self._win.main_stack.set_visible_child_name("sketch")

            self._win.menubar.set_menu_model(
                self._win.sketch_studio.menu_model
            )
            self._win.insert_action_group(
                "sketch", self._win.sketch_studio.action_group
            )
            self._win.add_controller(
                self._win.sketch_studio.shortcut_controller
            )
        except Exception as e:
            logger.error(
                f"Failed to load sketch for editing: {e}", exc_info=True
            )

    def exit_sketch_mode(self):
        """Returns to the main 2D/3D view from the SketchStudio."""
        self._win.menubar.set_menu_model(self._win.menu_model)
        self._win.insert_action_group("sketch", None)
        self._win.remove_controller(
            self._win.sketch_studio.shortcut_controller
        )

        self._win.main_stack.set_visible_child_name("main")
        self.active_sketch_workpiece = None
        self._is_editing_new_sketch = False

    def enter_sketch_definition_mode(self, sketch: Sketch):
        """Switches to SketchStudio to edit a sketch definition directly."""
        try:
            self.active_sketch_workpiece = None
            self._is_editing_new_sketch = False
            self._win.sketch_studio.set_sketch(sketch)
            self._win.main_stack.set_visible_child_name("sketch")

            self._win.menubar.set_menu_model(
                self._win.sketch_studio.menu_model
            )
            self._win.insert_action_group(
                "sketch", self._win.sketch_studio.action_group
            )
            self._win.add_controller(
                self._win.sketch_studio.shortcut_controller
            )
        except Exception as e:
            logger.error(
                f"Failed to load sketch definition for editing: {e}",
                exc_info=True,
            )

    def on_sketch_definition_activated(self, sender, *, sketch: Sketch):
        """Handles activation of a sketch definition from the sketch list."""
        self.enter_sketch_definition_mode(sketch)

    def on_sketch_finished(self, sender, *, sketch: Sketch):
        """Handles the 'finished' signal from the SketchStudio."""
        cmd = UpdateSketchCommand(
            doc=self._editor.doc,
            sketch_uid=sketch.uid,
            new_sketch_dict=sketch.to_dict(),
        )
        self._editor.history_manager.execute(cmd)

        if self._is_editing_new_sketch:
            center_x = self._win.sketch_studio.width_mm / 2
            center_y = self._win.sketch_studio.height_mm / 2
            self._editor.edit.add_sketch_instance(
                sketch.uid, (center_x, center_y)
            )

        self.exit_sketch_mode()

    def on_sketch_cancelled(self, sender):
        """Handles the 'cancelled' signal from the SketchStudio."""
        was_new = self._is_editing_new_sketch
        self.exit_sketch_mode()

        if was_new:
            self._editor.history_manager.undo()

    def on_new_sketch(self, action=None, param=None):
        """Action handler for creating a new sketch definition."""
        new_sketch = Sketch(name=_("New Sketch"))

        command = ListItemCommand(
            owner_obj=self._editor.doc,
            item=new_sketch,
            undo_command="remove_asset",
            redo_command="add_asset",
            name=_("Create Sketch Definition"),
        )
        self._editor.history_manager.execute(command)

        self.enter_sketch_definition_mode(new_sketch)
        self._is_editing_new_sketch = True

    def on_edit_sketch(self, action, param):
        """Action handler for editing the selected sketch."""
        selected_items = self._win.surface.get_selected_workpieces()
        if len(selected_items) == 1 and isinstance(
            selected_items[0], WorkPiece
        ):
            wp = selected_items[0]
            if wp.sketch_uid:
                self.enter_sketch_mode(wp)
            else:
                self._win._on_editor_notification(
                    self._win, _("Selected item is not an editable sketch.")
                )
        else:
            self._win._on_editor_notification(
                self._win, _("Please select a single sketch to edit.")
            )

    def on_edit_sketch_requested(self, sender, *, workpiece: WorkPiece):
        """Signal handler for edit sketch requests from the surface."""
        logger.debug(f"Sketch edit requested for workpiece {workpiece.name}")
        self.enter_sketch_mode(workpiece)

    def on_export_sketch(self, action, param):
        """Action handler for exporting the selected sketch."""
        selected_items = self._win.surface.get_selected_workpieces()
        if len(selected_items) == 1:
            file_dialogs.show_export_sketch_dialog(
                self._win,
                self._on_export_sketch_save_response,
                selected_items[0],
            )
        else:
            self._win._on_editor_notification(
                self._win, _("Please select a single sketch to export.")
            )

    def _on_export_sketch_save_response(self, dialog, result, user_data):
        """Callback for the export sketch dialog."""
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())

            selected = self._win.surface.get_selected_workpieces()
            if len(selected) != 1:
                return

            self._editor.file.export_sketch_to_path(file_path, selected[0])

        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
