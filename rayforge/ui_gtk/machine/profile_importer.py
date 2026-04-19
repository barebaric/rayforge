import logging
from gettext import gettext as _
from pathlib import Path
from typing import Callable, Optional

from gi.repository import Gio, GLib, Gtk

from ...context import get_context
from ...machine.device.profile import DeviceProfile

logger = logging.getLogger(__name__)


def open_profile_zip(
    parent: Gtk.Window,
    callback: Callable[[Optional[DeviceProfile], Optional[str]], None],
):
    """Open a file dialog to import a device profile from a zip.

    *callback* is called with ``(profile, None)`` on success or
    ``(None, error_message)`` on failure.  If the user cancels the
    dialog, *callback* is not invoked.
    """
    filter_list = Gio.ListStore.new(Gtk.FileFilter)

    zip_filter = Gtk.FileFilter()
    zip_filter.set_name(_("Device Profile archives"))
    zip_filter.add_pattern("*.rfdevice.zip")
    filter_list.append(zip_filter)

    all_filter = Gtk.FileFilter()
    all_filter.set_name(_("All files"))
    all_filter.add_pattern("*")
    filter_list.append(all_filter)

    dialog = Gtk.FileDialog.new()
    dialog.set_title(_("Import Device Profile"))
    dialog.set_filters(filter_list)
    dialog.set_default_filter(zip_filter)
    dialog.open(
        parent,
        None,
        _on_file_selected,
        callback,
    )


def _on_file_selected(dialog, result, callback):
    try:
        file = dialog.open_finish(result)
    except GLib.Error:
        return
    if not file:
        return

    mgr = get_context().device_profile_mgr
    try:
        profile = mgr.install_from_zip(Path(file.get_path()))
    except Exception as e:
        logger.error(f"Import failed: {e}")
        callback(None, str(e))
        return

    callback(profile, None)
