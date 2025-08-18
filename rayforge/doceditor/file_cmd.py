import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from gi.repository import Gtk, Gio, GLib
from ..core.workpiece import WorkPiece
from ..importer import importers, importer_by_mime_type, importer_by_extension
from ..undo import ListItemCommand
from ..shared.tasker import task_mgr
from ..shared.tasker.context import ExecutionContext
from ..config import config
from ..pipeline.job import generate_job_ops
from ..pipeline.encoder.gcode import GcodeEncoder

if TYPE_CHECKING:
    from ..mainwindow import MainWindow

logger = logging.getLogger(__name__)


def import_file(win: "MainWindow"):
    """
    Shows the file chooser dialog and triggers the import process upon
    user selection.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Open File"))

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    all_supported = Gtk.FileFilter()
    all_supported.set_name(_("All supported"))
    for importer_class in importers:
        file_filter = Gtk.FileFilter()
        file_filter.set_name(_(importer_class.label))
        for mime_type in importer_class.mime_types:
            file_filter.add_mime_type(mime_type)
            all_supported.add_mime_type(mime_type)
        filter_list.append(file_filter)
    filter_list.append(all_supported)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(all_supported)

    dialog.open(win, None, _on_import_dialog_response, win)


def _on_import_dialog_response(dialog, result, win: "MainWindow"):
    """Callback for when the user selects a file from the dialog."""
    try:
        file = dialog.open_finish(result)
        if not file:
            return

        file_path = Path(file.get_path())
        file_info = file.query_info(
            Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
            Gio.FileQueryInfoFlags.NONE,
            None,
        )
        mime_type = file_info.get_content_type()
        load_file_from_path(win, file_path, mime_type)
    except Exception:
        logger.exception("Error opening file")


def load_file_from_path(
    win: "MainWindow", filename: Path, mime_type: Optional[str]
):
    """
    Orchestrates the loading of a specific file path using the new
    hierarchical importer architecture.
    """
    importer_class = None
    if mime_type:
        importer_class = importer_by_mime_type.get(mime_type)
    if not importer_class:
        file_extension = filename.suffix.lower()
        if file_extension:
            importer_class = importer_by_extension.get(file_extension)

    if not importer_class:
        logger.error(f"No importer found for '{filename.name}'")
        return

    try:
        file_data = filename.read_bytes()
        importer = importer_class(file_data)
    except Exception as e:
        logger.error(
            f"Failed to instantiate importer for {filename.name}: {e}"
        )
        return

    cmd_name = _("Import {name}").format(name=filename.name)

    # 1. Attempt hierarchical import.
    imported_items = importer.get_doc_items()
    if imported_items:
        with win.doc.history_manager.transaction(cmd_name) as t:
            for item in imported_items:
                command = ListItemCommand(
                    owner_obj=win.doc.active_layer,
                    item=item,
                    undo_command="remove_child",
                    redo_command="add_child",
                )
                t.execute(command)
        # Hide properties widget in case something was selected before import
        win.item_revealer.set_reveal_child(False)
        return

    # 2. Fallback to flat vector import.
    ops = importer.get_vector_ops()
    if ops and not ops.is_empty():
        wp = WorkPiece.from_ops(ops, name=filename.name)
        _center_new_workpiece(win, wp)
        command = ListItemCommand(
            owner_obj=win.doc.active_layer,
            item=wp,
            undo_command="remove_workpiece",
            redo_command="add_workpiece",
            name=cmd_name,
        )
        win.doc.history_manager.execute(command)
        win.item_revealer.set_reveal_child(False)
        return

    # 3. Final fallback for raster/other types
    wp = WorkPiece.from_file(filename, importer_class)
    _center_new_workpiece(win, wp)
    command = ListItemCommand(
        owner_obj=win.doc.active_layer,
        item=wp,
        undo_command="remove_workpiece",
        redo_command="add_workpiece",
        name=cmd_name,
    )
    win.doc.history_manager.execute(command)
    win.item_revealer.set_reveal_child(False)


def _center_new_workpiece(win: "MainWindow", wp: WorkPiece):
    """Helper method to contain the centering logic for single items."""
    wswidth_mm, wsheight_mm = win.surface.get_size_mm()
    wp_width_mm, wp_height_mm = wp.size
    x_mm = (wswidth_mm - wp_width_mm) / 2
    y_mm = (wsheight_mm - wp_height_mm) / 2
    wp.pos = (x_mm, y_mm)


def _on_save_dialog_response(dialog, result, win: "MainWindow"):
    try:
        file = dialog.save_finish(result)
        if not file:
            return
        file_path = Path(file.get_path())
    except GLib.Error as e:
        logger.error(f"Error saving file: {e.message}")
        return

    def write_gcode_sync(path, gcode):
        """Blocking I/O function to be run in a thread."""
        with open(path, "w") as f:
            f.write(gcode)

    async def export_coro(context: ExecutionContext):
        machine = config.machine
        if not machine:
            return

        try:
            # 1. Generate Ops (async, reports progress)
            ops = await generate_job_ops(
                win.doc, machine, win.surface.ops_generator, context
            )

            # 2. Encode G-code (sync, but usually fast)
            context.set_message(_("Encoding G-code..."))
            encoder = GcodeEncoder.for_machine(machine)
            gcode = encoder.encode(ops, machine)

            # 3. Write to file (sync, potentially slow, run in thread)
            context.set_message(_(f"Saving to {file_path}..."))
            await asyncio.to_thread(write_gcode_sync, file_path, gcode)

            context.set_message(_("Export complete!"))
            context.set_progress(1.0)
            context.flush()

        except Exception:
            logger.error("Failed to export G-code", exc_info=True)
            raise  # Re-raise to be caught by the task manager

    # Add the coroutine to the task manager
    task_mgr.add_coroutine(export_coro, key="export-gcode")


def export_gcode(win: "MainWindow"):
    """Shows the save file dialog and handles the G-code export process."""
    # Create a file chooser dialog for saving the file
    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Save G-code File"))

    # Set the default file name
    dialog.set_initial_name("output.gcode")

    # Create a Gio.ListModel for the filters
    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    gcode_filter = Gtk.FileFilter()
    gcode_filter.set_name(_("G-code files"))
    gcode_filter.add_mime_type("text/x.gcode")
    filter_list.append(gcode_filter)

    # Set the filters for the dialog
    dialog.set_filters(filter_list)
    dialog.set_default_filter(gcode_filter)

    # Show the dialog and handle the response
    dialog.save(win, None, _on_save_dialog_response, win)
