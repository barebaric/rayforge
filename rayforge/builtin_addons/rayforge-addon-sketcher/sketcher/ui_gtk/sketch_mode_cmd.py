import logging
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

from gi.repository import GLib

from rayforge.core.undo import ListItemCommand
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.asset_cmd import UpdateAssetCommand
from rayforge.ui_gtk.doceditor import file_dialogs
from rayforge.usage import get_usage_tracker

from ..core.sketch import Sketch

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor
    from rayforge.ui_gtk.mainwindow import MainWindow

    from .studio import SketchStudio

logger = logging.getLogger(__name__)


def _get_sketch_studio() -> Optional["SketchStudio"]:
    """Get the SketchStudio instance, avoiding circular imports."""
    from . import get_sketch_studio

    return get_sketch_studio()


class SketchModeCmd:
    """Handles commands for entering, exiting, and managing sketch mode."""

    def __init__(self, win: "MainWindow", editor: "DocEditor"):
        self._win = win
        self._editor = editor
        self.active_sketch_workpiece: WorkPiece | None = None
        self._is_editing_new_sketch = False
        self._sketch_history_connected = False
        self._doc_was_saved_on_entry: bool = True
        # Asset state captured when the sketcher was entered; used as
        # the undo target for the UpdateAssetCommand created on finish.
        self._sketch_data_on_entry: dict | None = None

    def _on_sketch_history_changed(self, sender, **kwargs):
        """Updates the document's saved state based on the sketch history.

        If the sketch has been modified away from its entry state, the
        document is marked as unsaved. If the sketch is undone back to its
        entry state, the document's saved state is restored to what it was
        before entering the sketcher.
        """
        sketch_editor = self._get_active_sketch_editor()
        if not sketch_editor:
            return
        if sketch_editor.history_manager.is_at_checkpoint():
            if self._doc_was_saved_on_entry:
                self._editor.mark_as_saved()
            else:
                self._editor.mark_as_unsaved()
        else:
            self._editor.mark_as_unsaved()

    def _get_active_sketch_editor(self):
        sketch_studio = _get_sketch_studio()
        if not sketch_studio:
            return None
        return sketch_studio.canvas.sketch_editor

    def _connect_sketch_history(self, sketch_studio: "SketchStudio"):
        """Connects the sketch editor's history to the document's
        saved-state tracking so edits mark the project as changed."""
        if self._sketch_history_connected:
            return
        sketch_editor = sketch_studio.canvas.sketch_editor
        if sketch_editor:
            self._doc_was_saved_on_entry = self._editor.is_saved
            sketch_editor.history_manager.set_checkpoint()
            sketch_editor.history_manager.changed.connect(
                self._on_sketch_history_changed
            )
            self._sketch_history_connected = True

    def _disconnect_sketch_history(self, sketch_studio: "SketchStudio"):
        """Disconnects the sketch history handler."""
        if not self._sketch_history_connected:
            return
        sketch_editor = sketch_studio.canvas.sketch_editor
        if sketch_editor:
            sketch_editor.history_manager.changed.disconnect(
                self._on_sketch_history_changed
            )
        self._sketch_history_connected = False

    def enter_sketch_mode(
        self, workpiece: WorkPiece, is_new_sketch: bool = False
    ):
        """Switches the view to the SketchStudio to edit a workpiece."""
        sketch = None
        if workpiece.geometry_provider_uid:
            sketch = cast(
                Sketch | None,
                self._editor.doc.get_asset_by_uid(
                    workpiece.geometry_provider_uid
                ),
            )

        if not sketch:
            logger.warning("Attempted to edit a non-sketch workpiece.")
            return

        try:
            sketch_studio = _get_sketch_studio()
            if not sketch_studio:
                logger.error("SketchStudio not initialized")
                return

            self.active_sketch_workpiece = workpiece
            self._is_editing_new_sketch = is_new_sketch
            # Snapshot BEFORE the editor mutates the asset in place, so
            # finishing can build an UpdateAssetCommand whose undo
            # restores the pre-edit state.
            self._sketch_data_on_entry = sketch.to_dict()
            sketch_studio.set_sketch(sketch)
            self._win.open_modal_page("sketch")
            get_usage_tracker().track_page_view("/sketcher", "Sketch Editor")

            self._win.menubar.set_menu_model(sketch_studio.menu_model)
            self._win.insert_action_group("sketch", sketch_studio.action_group)
            self._win.add_controller(sketch_studio.shortcut_controller)
            self._connect_sketch_history(sketch_studio)
        except Exception:
            logger.exception("Failed to load sketch for editing")

    def exit_sketch_mode(self):
        """Returns to the main 2D/3D view from the SketchStudio."""
        sketch_studio = _get_sketch_studio()
        self._win.menubar.set_menu_model(self._win.menu_model)
        self._win.insert_action_group("sketch", None)
        if sketch_studio:
            self._win.remove_controller(sketch_studio.shortcut_controller)
            self._disconnect_sketch_history(sketch_studio)

        self._win.close_modal_page()
        self.active_sketch_workpiece = None
        self._is_editing_new_sketch = False

    def enter_sketch_definition_mode(self, sketch: Sketch):
        """Switches to SketchStudio to edit a sketch definition directly."""
        try:
            sketch_studio = _get_sketch_studio()
            if not sketch_studio:
                logger.error("SketchStudio not initialized")
                return

            self.active_sketch_workpiece = None
            self._is_editing_new_sketch = False
            # Snapshot BEFORE the editor mutates the asset in place.
            self._sketch_data_on_entry = sketch.to_dict()
            sketch_studio.set_sketch(sketch)
            self._win.open_modal_page("sketch")
            get_usage_tracker().track_page_view("/sketcher", "Sketch Editor")

            self._win.menubar.set_menu_model(sketch_studio.menu_model)
            self._win.insert_action_group("sketch", sketch_studio.action_group)
            self._win.add_controller(sketch_studio.shortcut_controller)
            self._connect_sketch_history(sketch_studio)
        except Exception:
            logger.exception("Failed to load sketch definition for editing")

    def on_sketch_definition_activated(self, sender, *, sketch: Sketch):
        """Handles activation of a sketch definition from the sketch list."""
        self.enter_sketch_definition_mode(sketch)

    def on_sketch_finished(self, sender, *, sketch: Sketch):
        """Handles the 'finished' signal from the SketchStudio."""
        cmd = UpdateAssetCommand(
            doc=self._editor.doc,
            asset_uid=sketch.uid,
            new_data=sketch.to_dict(),
            # The editor mutates the live asset in place, so the state
            # captured at command construction time would already be the
            # edited one. Use the snapshot taken on entry instead.
            old_data=self._sketch_data_on_entry,
        )
        self._editor.history_manager.execute(cmd)

        if self._is_editing_new_sketch:
            sketch_studio = _get_sketch_studio()
            if sketch_studio:
                center_x = sketch_studio.width_mm / 2
                center_y = sketch_studio.height_mm / 2
                self._editor.edit.add_geometry_provider_instance(
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
            if wp.geometry_provider_uid:
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

    def on_activate_sketch(self, action, param):
        """Action handler for activating a sketch definition."""
        asset_uid = param.get_string()
        sketch = self._editor.doc.get_asset_by_uid(asset_uid)
        if isinstance(sketch, Sketch):
            self.enter_sketch_definition_mode(sketch)

    def on_edit_sketch_item(self, action, param):
        """Action handler for editing a sketch-based workpiece."""
        item_uid = param.get_string()
        item = self._editor.doc.find_descendant_by_uid(item_uid)
        if isinstance(item, WorkPiece) and item.geometry_provider_uid:
            self.enter_sketch_mode(item)

    def on_export_object(self, action, param):
        """Action handler for exporting the selected object."""
        selected_items = self._win.surface.get_selected_workpieces()
        if len(selected_items) == 1:
            file_dialogs.show_export_object_dialog(
                self._win,
                self._on_export_object_save_response,
                selected_items[0],
            )
        else:
            self._win._on_editor_notification(
                self._win, _("Please select a single object to export.")
            )

    def _on_export_object_save_response(self, dialog, result, user_data):
        """Callback for the export object dialog."""
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())

            selected = self._win.surface.get_selected_workpieces()
            if len(selected) != 1:
                return

            self._editor.file.export_object_to_path(file_path, selected[0])

        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
