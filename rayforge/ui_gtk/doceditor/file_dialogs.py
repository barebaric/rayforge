import logging
from typing import Callable, TYPE_CHECKING, Any, Optional
from gi.repository import Gtk, Gio
from ... import const

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...doceditor.editor import DocEditor
    from ..mainwindow import MainWindow

logger = logging.getLogger(__name__)


def show_import_dialog(
    win: "MainWindow",
    editor: "DocEditor",
    callback: Callable,
    user_data: Any = None,
):
    """
    Shows the file chooser dialog for importing files.

    Args:
        win: The parent Gtk.Window.
        editor: The DocEditor instance to retrieve supported file types.
        callback: The function to call with (dialog, result, user_data) upon
                  response.
        user_data: Custom data to pass to the callback.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Open File"))

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    all_supported = Gtk.FileFilter()
    all_supported.set_name(_("All supported"))

    # Get supported filters from the backend
    supported_types = editor.file.get_supported_import_filters()

    for file_type in supported_types:
        file_filter = Gtk.FileFilter()
        if file_type["label"]:
            file_filter.set_name(_(file_type["label"]))

        if file_type["extensions"]:
            for ext in file_type["extensions"]:
                pattern = f"*{ext}"
                file_filter.add_pattern(pattern)
                all_supported.add_pattern(pattern)

        if file_type["mime_types"]:
            for mime_type in file_type["mime_types"]:
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
    sketch_filter.set_name(
        _("{app_name} Sketch").format(app_name=const.APP_NAME))
    sketch_filter.add_pattern("*.rfs")
    sketch_filter.add_mime_type(const.MIME_TYPE_SKETCH)
    filter_list.append(sketch_filter)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(sketch_filter)

    dialog.save(win, None, callback, win)


def show_open_project_dialog(win: "MainWindow", callback: Callable):
    """
    Shows file chooser dialog for opening Rayforge project files.

    Args:
        win: The parent Gtk.Window.
        callback: The function to call with (dialog, result, user_data) upon
                  response.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(
        _("Open {app_name} Project").format(app_name=const.APP_NAME))

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    project_filter = Gtk.FileFilter()
    project_filter.set_name(
        _("{app_name} Project").format(app_name=const.APP_NAME))
    project_filter.add_pattern("*.ryp")
    filter_list.append(project_filter)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(project_filter)

    dialog.open(win, None, callback, win)


def show_save_project_dialog(
    win: "MainWindow",
    callback: Callable,
    initial_name: Optional[str] = None,
):
    """
    Shows save file dialog for saving Rayforge project files.

    Args:
        win: The parent Gtk.Window.
        callback: The function to call with (dialog, result, user_data) upon
                  response.
        initial_name: Optional initial file name for the dialog.
    """
    dialog = Gtk.FileDialog.new()
    dialog.set_title(
        _("Save {app_name} Project").format(app_name=const.APP_NAME))

    if initial_name:
        dialog.set_initial_name(initial_name)
    else:
        dialog.set_initial_name("untitled.ryp")

    filter_list = Gio.ListStore.new(Gtk.FileFilter)
    project_filter = Gtk.FileFilter()
    project_filter.set_name(
        _("{app_name} Project").format(app_name=const.APP_NAME)
    )
    project_filter.add_pattern("*.ryp")
    filter_list.append(project_filter)

    dialog.set_filters(filter_list)
    dialog.set_default_filter(project_filter)

    dialog.save(win, None, callback, win)
