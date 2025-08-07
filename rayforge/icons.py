import importlib.resources
import logging
from gi.repository import Gtk, Gdk, Gio  # type: ignore
from .resources import icons  # type: ignore


logger = logging.getLogger(__name__)


def get_icon_path(icon_name):
    """Retrieve the path of an icon inside the resource directory."""
    with importlib.resources.path(icons, f"{icon_name}.svg") as path:
        return str(path)


def get_icon(icon_name):
    """Retrieve the Gtk.Image from an icon inside the resource directory."""
    theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())

    if not theme.has_icon(icon_name):
        logger.debug(f"Icon '{icon_name}' not found in theme.")
        path = get_icon_path(icon_name)
        icon_file = Gio.File.new_for_path(str(path))
        icon = Gio.FileIcon.new(icon_file)
        return Gtk.Image.new_from_gicon(icon)

    # Create image with symbolic icon
    image = Gtk.Image.new_from_icon_name(icon_name)
    return image
