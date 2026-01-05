import logging
from typing import Callable, TYPE_CHECKING, Any, Optional
from gi.repository import Gtk, Gio
from ...image import importers

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ..mainwindow import MainWindow

logger = logging.getLogger(__name__)


def show_import_dialog(
    win: "MainWindow", callback: Callable, user_data: Any = None
):
    """
    Shows the file chooser dialog for importing files.

    Args:
        win: The parent Gtk.Window.
        callback: The function to call with (dialog, result, user_data) upon
                  response.
        user_data: Custom data to pass to the callback.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Open File"))

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    all_supported = Gtk.FileFilter()
    all_supported.set_name(_("All supported"))
    for importer_class in importers:
        file_filter = Gtk.FileFilter()
        if importer_class.label:
            file_filter.set_name(_(importer_class.label))
        if importer_class.extensions:
            for ext in importer_class.extensions:
                pattern = f"*{ext}"
                file_filter.add_pattern(pattern)
                all_supported.add_pattern(pattern)
        if importer_class.mime_types:
            for mime_type in importer_class.mime_types:
                file_filter.add_mime_type(mime_type)
                all_supported.add_mime_type(mime_type)
        filter_list.append(file_filter)
    filter_list.append(all_supported)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(all_supported)

    dialog.open(win, None, callback, user_data)


def show_export_gcode_dialog(win: "MainWindow", callback: Callable):
    """
    Shows the save file dialog for exporting G-code.

    Args:
        win: The parent Gtk.Window.
        callback: The function to call with (dialog, result, user_data) upon
                  response. The window instance is passed as user_data.
    """
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
    dialog.save(win, None, callback, win)


def show_export_sketch_dialog(
    win: "MainWindow",
    callback: Callable,
    workpiece: Optional["WorkPiece"] = None,
):
    """
    Shows the save file dialog for exporting a Rayforge Sketch (.rfs).

    Args:
        win: The parent Gtk.Window.
        callback: The function to call with (dialog, result, user_data) upon
                  response.
        workpiece: Optional workpiece to use for default export location.
                   If provided, the dialog will default to the source file
                   location and name of the workpiece.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Export Sketch"))

    if workpiece and workpiece.source_file:
        dialog.set_initial_name(workpiece.source_file.name)
        try:
            folder = Gio.File.new_for_path(str(workpiece.source_file.parent))
            dialog.set_initial_folder(folder)
        except Exception:
            logger.debug(
                "Could not set initial folder for sketch export dialog"
            )
    else:
        dialog.set_initial_name("sketch.rfs")

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    sketch_filter = Gtk.FileFilter()
    sketch_filter.set_name(_("Rayforge Sketch"))
    sketch_filter.add_pattern("*.rfs")
    sketch_filter.add_mime_type("application/vnd.rayforge-sketch")
    filter_list.append(sketch_filter)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(sketch_filter)

    dialog.save(win, None, callback, win)
