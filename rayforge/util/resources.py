import importlib.resources
from gi.repository import Gtk
from ..resources import icons


def get_icon_path(icon_name):
    """Retrieve the path of an icon inside the package."""
    with importlib.resources.path(icons, f"{icon_name}.svg") as path:
        return str(path)


def get_icon(icon_name):
    """Retrieve the path of an icon inside the package."""
    return Gtk.Image.new_from_file(get_icon_path(icon_name))
