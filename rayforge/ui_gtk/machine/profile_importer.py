import logging
from gettext import gettext as _
from pathlib import Path
from typing import Callable, Optional

from gi.repository import Gio, GLib, Gtk

from ...context import get_context
from ...machine.device.lightburn_importer import convert_to_profile
from ...machine.device.profile import DeviceProfile
from .lbdev_import_dialog import LBDevImportDialog

logger = logging.getLogger(__name__)


def open_profile_file(
    parent: Gtk.Window,
    callback: Callable[[Optional[DeviceProfile], Optional[str]], None],
):
    """Open a file dialog to import a device profile.

    Supports Rayforge ``.rfdevice.zip`` and LightBurn ``.lbdev``
    files.

    *callback* is called with ``(profile, None)`` on success or
    ``(None, error_message)`` on failure.  If the user cancels the
    dialog, *callback* is not invoked.
    """
    filter_list = Gio.ListStore.new(Gtk.FileFilter)

    rf_filter = Gtk.FileFilter()
    rf_filter.set_name(_("Device Profile archives"))
    rf_filter.add_pattern("*.rfdevice.zip")
    filter_list.append(rf_filter)

    lb_filter = Gtk.FileFilter()
    lb_filter.set_name(_("LightBurn device profiles"))
    lb_filter.add_pattern("*.lbdev")
    filter_list.append(lb_filter)

    all_filter = Gtk.FileFilter()
    all_filter.set_name(_("All files"))
    all_filter.add_pattern("*")
    filter_list.append(all_filter)

    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Import Device Profile"))
    dialog.set_filters(filter_list)
    dialog.set_default_filter(rf_filter)
    dialog.open(
        parent,
        None,
        _on_file_selected,
        (parent, callback),
    )


def open_profile_zip(
    parent: Gtk.Window,
    callback: Callable[[Optional[DeviceProfile], Optional[str]], None],
):
    """Legacy entry point -- delegates to :func:`open_profile_file`."""
    open_profile_file(parent, callback)


def _on_file_selected(dialog, result, user_data):
    parent, callback = user_data
    try:
        file = dialog.open_finish(result)
    except GLib.Error:
        return
    if not file:
        return

    file_path = Path(file.get_path())

    if file_path.suffix.lower() == ".lbdev":
        _handle_lbdev(parent, file_path, callback)
    else:
        _handle_zip(file_path, callback)


def _handle_zip(file_path: Path, callback):
    mgr = get_context().device_profile_mgr
    try:
        profile = mgr.install_from_zip(file_path)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        callback(None, str(e))
        return
    callback(profile, None)


def _handle_lbdev(
    parent: Gtk.Window,
    file_path: Path,
    callback: Callable[[Optional[DeviceProfile], Optional[str]], None],
):
    try:
        profile, summary = convert_to_profile(file_path)
    except Exception as e:
        logger.error(f"LightBurn import failed: {e}")
        callback(None, str(e))
        return

    def on_import():
        _install_lbdev_and_callback(file_path, callback)

    dialog = LBDevImportDialog(
        parent=parent,
        summary=summary,
        on_import=on_import,
    )
    dialog.present()


def _install_lbdev_and_callback(
    file_path: Path,
    callback: Callable[[Optional[DeviceProfile], Optional[str]], None],
):
    mgr = get_context().device_profile_mgr
    try:
        installed_profile, _ = mgr.install_from_lbdev(file_path)
    except Exception as e:
        logger.error(f"LightBurn install failed: {e}")
        callback(None, str(e))
        return
    callback(installed_profile, None)
